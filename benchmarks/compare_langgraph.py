"""Head-to-head runtime overhead: IkaCore vs LangGraph on identical workloads.

    <venv-with-langgraph>/python -m benchmarks.compare_langgraph [--budget 1.0] [--rounds 5]

Both frameworks talk to the same in-process fake provider (installed under httpx,
which both the openai SDK used by langchain-openai and IkaCore go through), over
OpenAI Chat Completions, with the same tool function and the same scripted model
behavior. The only difference is how each framework is told to stop: IkaCore by
the model calling agent_end, LangGraph by a plain-text reply (its native signal).
Every scenario asserts both sides made the same number of LLM calls and tool runs
and returned the final answer.

Timings are net of the fake provider, with agent/graph construction untimed
(measured separately in the construct scenario). Variants are interleaved in
rounds so machine drift hits both equally.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import IkaCore  # noqa: F401
except ModuleNotFoundError:  # not pip-installed: benchmark the working tree
    sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarks import fake_provider, harness  # noqa: E402
from benchmarks.fake_provider import Call, FakeLLM, Policy, RequestView, Turn, end  # noqa: E402
from benchmarks.scenarios import FINAL, _payload, make_agent, make_tool  # noqa: E402

MODEL = "gpt-4o"


def _loop(tool: str, rounds: int, final: Turn, width: int = 1) -> Policy:
    def policy(view: RequestView) -> Turn:
        done = view.count(tool)
        if done >= rounds * width:
            return final
        step = done // width
        return Turn(calls=tuple(
            Call(tool, {"query": f"{tool} topic {step}.{slot}", "step": step, "slot": slot}) for slot in range(width)
        ))

    return policy


# --------------------------------------------------------------------------
# Framework adapters
# --------------------------------------------------------------------------

class ToolCounter:
    def __init__(self) -> None:
        self.runs = 0


def _tool_result(name: str, output_size: int, args: dict[str, Any]) -> dict[str, Any]:
    seed = f"{name}-{args.get('step', 0)}-{args.get('slot', 0)}"
    return {"tool": name, "query": args.get("query"), "result": _payload(output_size, seed)}


def ikacore_setup(n_tools: int, output_size: int, counter: ToolCounter) -> Callable[[], str]:
    tools = []
    for i in range(n_tools):
        tool = make_tool("lookup" if i == 0 else f"aux_{i}", output_size=output_size)
        inner = tool.execute_function

        def counted(args: dict[str, Any], inner: Any = inner) -> Any:
            counter.runs += 1
            return inner(args)

        tool.execute_function = counted
        tools.append(tool)
    agent = make_agent("ika", "openai_chat", tools=tools, maxsteps=200)
    return lambda: str(agent.execution()["final_message"])


def langgraph_setup(n_tools: int, output_size: int, counter: ToolCounter) -> Callable[[], str]:
    from langchain_core.tools import StructuredTool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    def make(name: str) -> StructuredTool:
        def fn(query: str, step: int, slot: int = 0) -> dict[str, Any]:
            counter.runs += 1
            return _tool_result(name, output_size, {"query": query, "step": step, "slot": slot})

        return StructuredTool.from_function(fn, name=name, description=f"Look up information with {name}.")

    tools = [make("lookup" if i == 0 else f"aux_{i}") for i in range(n_tools)]
    model = ChatOpenAI(model=MODEL, api_key="bench-lg", use_responses_api=False, temperature=0, max_retries=0)
    graph = create_react_agent(model, tools, prompt="You are a precise, concise research assistant.")

    def run() -> str:
        state = graph.invoke(
            {"messages": [("user", "Research the topic, use your tools, then answer.")]},
            {"recursion_limit": 1000},
        )
        return str(state["messages"][-1].content)

    return run


@dataclass(frozen=True)
class Case:
    name: str
    rounds: int
    width: int = 1
    n_tools: int = 2
    output_size: int = 96


CASES = [
    Case("single_turn", rounds=0),
    Case("tool_loop_10", rounds=10),
    Case("parallel_tools_16", rounds=2, width=16),
    Case("long_conversation_40", rounds=40, output_size=1024),
    Case("many_tools_30", rounds=5, n_tools=30),
]

FRAMEWORKS: dict[str, tuple[Callable[..., Callable[[], str]], Callable[[Case], Turn]]] = {
    "ikacore": (ikacore_setup, lambda case: end(FINAL)),
    "langgraph": (langgraph_setup, lambda case: Turn(text=FINAL)),
}
AGENT_KEY = {"ikacore": "ika", "langgraph": "lg"}


@dataclass
class Sample:
    net_ms: list[float]
    calls: int = 0
    tool_runs: int = 0
    kb_sent: float = 0.0


def _once(framework: str, case: Case, fake: FakeLLM) -> tuple[str, float, int]:
    setup, _ = FRAMEWORKS[framework]
    counter = ToolCounter()
    run = setup(case.n_tools, case.output_size, counter)
    fake.reset_counters()
    start = time.perf_counter()
    out = run()
    wall = time.perf_counter() - start
    return out, (wall - fake.fake_seconds) * 1000.0, counter.runs


def compare_case(case: Case, rounds: int, budget_s: float) -> dict[str, Sample]:
    policies = {AGENT_KEY[fw]: _loop("lookup", case.rounds, final(case), case.width) for fw, (_, final) in FRAMEWORKS.items()}
    fake = FakeLLM(policies=policies)
    fake_provider.activate(fake)
    samples: dict[str, Sample] = {}
    try:
        for fw in FRAMEWORKS:  # warm-up + correctness check
            out, _, tool_runs = _once(fw, case, fake)
            assert FINAL in out, f"{fw} {case.name}: unexpected final {out[:120]!r}"
            samples[fw] = Sample([], fake.calls, tool_runs, fake.bytes_sent / 1024)
        expected = case.rounds + 1
        for fw, sample in samples.items():
            assert sample.calls == expected, f"{fw} {case.name}: {sample.calls} LLM calls, expected {expected}"
            assert sample.tool_runs == case.rounds * case.width, f"{fw} {case.name}: {sample.tool_runs} tool runs"
        _, probe_ms, _ = _once("ikacore", case, fake)
        per_round = max(1, min(100, int((budget_s / rounds) / max(probe_ms / 1000.0, 1e-4))))
        for _ in range(rounds):
            for fw in FRAMEWORKS:
                for _ in range(per_round):
                    _, net_ms, _ = _once(fw, case, fake)
                    samples[fw].net_ms.append(net_ms)
    finally:
        fake_provider.activate(None)
    return samples


def construct_times(repeats: int) -> dict[str, float]:
    times: dict[str, list[float]] = {fw: [] for fw in FRAMEWORKS}
    for _ in range(repeats):
        for fw, (setup, _) in FRAMEWORKS.items():
            start = time.perf_counter()
            setup(8, 96, ToolCounter())
            times[fw].append((time.perf_counter() - start) * 1000.0)
    return {fw: statistics.median(v[1:] or v) for fw, v in times.items()}


def import_times(repeats: int) -> dict[str, float]:
    code = {
        "ikacore": "import IkaCore; IkaCore.IkaBaseAgent",
        "langgraph": "import langgraph.prebuilt, langchain_openai; langgraph.prebuilt.create_react_agent",
    }
    env_path = str(REPO_ROOT / "src")
    out: dict[str, float] = {}
    for fw, snippet in code.items():
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            subprocess.run([sys.executable, "-c", f"import sys; sys.path.insert(0, {env_path!r}); {snippet}"], check=True)
            samples.append((time.perf_counter() - start) * 1000.0)
        out[fw] = min(samples)
    return out


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.compare_langgraph", description=__doc__.split("\n\n")[0])
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--budget", type=float, default=1.0, help="seconds per framework per case")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    from importlib.metadata import version

    import langgraph  # noqa: F401  (fail fast with a clear error if not installed)

    fake_provider.install()
    print(f"IkaCore (working tree) vs langgraph {version('langgraph')} / langchain-openai {version('langchain-openai')}, "
          f"Python {platform.python_version()}\n")
    header = f"{'case':<22}{'calls':>6}{'IkaCore ms':>12}{'LangGraph ms':>14}{'ratio':>8}{'Ika KB':>9}{'LG KB':>8}"
    print(header)
    print("-" * len(header))
    report: dict[str, Any] = {}
    with harness.sandbox_cwd(), harness.quiet():
        results = {case.name: compare_case(case, args.rounds, args.budget) for case in CASES}
        construct = construct_times(7)
    for name, samples in results.items():
        ika, lg = samples["ikacore"], samples["langgraph"]
        mi, ml = statistics.median(ika.net_ms), statistics.median(lg.net_ms)
        print(f"{name:<22}{ika.calls:>6}{mi:>12.2f}{ml:>14.2f}{ml / mi:>7.1f}x{ika.kb_sent:>9.1f}{lg.kb_sent:>8.1f}")
        report[name] = {"calls": ika.calls, "ikacore_ms": mi, "langgraph_ms": ml, "ikacore_kb": ika.kb_sent, "langgraph_kb": lg.kb_sent}
    print(f"{'construct (8 tools)':<22}{'':>6}{construct['ikacore']:>12.2f}{construct['langgraph']:>14.2f}"
          f"{construct['langgraph'] / construct['ikacore']:>7.1f}x")
    imports = import_times(3)
    print(f"{'cold import (wall)':<22}{'':>6}{imports['ikacore']:>12.0f}{imports['langgraph']:>14.0f}"
          f"{imports['langgraph'] / imports['ikacore']:>7.1f}x")
    print("\nratio = LangGraph time / IkaCore time (>1 means IkaCore is faster). Net of fake-provider time.")
    report.update({"construct": construct, "import": imports})
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
