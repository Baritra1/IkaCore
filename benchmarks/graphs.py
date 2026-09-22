"""Workflow-graph benchmark: wall time against the critical-path optimum.

    python -m benchmarks.graphs [--reps 3] [--json out.json] [--compare base.json]

Each case is an async ``IkaWorkflow`` whose nodes have simulated model latencies
(the fake provider sleeps per request, outside its lock, so parallel requests
really overlap). For every case we report:

  optimal   the critical-path makespan under IkaCore's own semantics: a node's
            LLM calls run back to back, a node whose merged upstream context is
            over the summarisation threshold first pays one summary call (short
            context passes through verbatim), instances of a node run in
            parallel, and a node starts once all of its dependencies finish
  actual    measured wall time (median of --reps runs)
  eff       optimal / actual; 100% means the scheduler adds no avoidable wait
  summaries summarisation calls made vs. the number the semantics need

It also checks correctness: every node ran, and every node's context covered
all of its parents (verbatim, or via summaries that list their sources), so
a scheduling change can only make runs faster, never different.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import IkaCore  # noqa: F401
except ModuleNotFoundError:  # not pip-installed: benchmark the working tree
    sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarks import fake_provider, harness  # noqa: E402
from benchmarks.fake_provider import SUMMARY_TEXT, FakeLLM, RequestView, Turn  # noqa: E402
from benchmarks.scenarios import end, loop_policy, make_agent, make_tool  # noqa: E402
from IkaCore import IkaWorkflow, WorkflowEdge, WorkflowNode  # noqa: E402

_DONE = re.compile(r"\b([a-z][a-z0-9_]*) done:")
IKACORE_THRESHOLD = 4000  # IkaWorkflow's default summarize_context_above_tokens


@dataclass(frozen=True)
class Node:
    name: str
    latency: float  # seconds per LLM call
    rounds: int = 1  # tool rounds before agent_end, so rounds + 1 calls
    instances: int = 1
    output_chars: int = 0  # extra length of the node's final answer (to exercise summarisation)

    def final_text(self) -> str:
        return f"{self.name} done: findings of {self.name}." + (" detail" * (self.output_chars // 7))


@dataclass(frozen=True)
class GraphCase:
    name: str
    nodes: tuple[Node, ...]
    edges: tuple[tuple[str, str], ...]
    summary_latency: float = 0.05
    description: str = ""

    def parents(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {n.name: [] for n in self.nodes}
        for src, dst in self.edges:
            out[dst].append(src)
        return out

    def needs_summary(self, name: str, threshold: Optional[int]) -> bool:
        """Whether a node's merged upstream context exceeds the summarisation threshold."""
        parents = self.parents()[name]
        if not parents:
            return False
        nodes = {n.name: n for n in self.nodes}
        merged = sum((len(nodes[p].final_text()) + 2) * nodes[p].instances for p in parents)
        return threshold is None or merged // 4 > threshold

    def optimal_makespan(
        self, with_summaries: bool = True, summarise_initial: bool = False, threshold: Optional[int] = None
    ) -> float:
        """Longest path where a node costs one context summary (if it gets context) plus its own calls."""
        parents = self.parents()
        start = self.nodes[0].name
        finish: dict[str, float] = {}

        def done_at(name: str) -> float:
            if name not in finish:
                node = next(n for n in self.nodes if n.name == name)
                ready = max((done_at(p) for p in parents[name]), default=0.0)
                # IkaCore passes initial_context to the start node verbatim; summarise_initial models the old behavior.
                summarised = (summarise_initial and name == start) or self.needs_summary(name, threshold)
                summary = self.summary_latency if (summarised and with_summaries) else 0.0
                finish[name] = ready + summary + (node.rounds + 1) * node.latency
            return finish[name]

        return max(done_at(n.name) for n in self.nodes)


CASES = [
    GraphCase("chain_4", tuple(Node(f"c{i}", 0.08) for i in range(4)),
              (("c0", "c1"), ("c1", "c2"), ("c2", "c3")), description="linear chain"),
    GraphCase("diamond_3x4", (Node("root", 0.08), Node("b1", 0.08, instances=4), Node("b2", 0.08, instances=4),
                              Node("b3", 0.08, instances=4), Node("merge", 0.08)),
              (("root", "b1"), ("root", "b2"), ("root", "b3"), ("b1", "merge"), ("b2", "merge"), ("b3", "merge")),
              description="fan-out to 3 nodes x 4 instances, fan-in"),
    GraphCase("uneven_branches", (Node("root", 0.05), Node("a1", 0.05), Node("a2", 0.05), Node("a3", 0.05),
                                  Node("slow", 0.4), Node("join", 0.05)),
              (("root", "a1"), ("a1", "a2"), ("a2", "a3"), ("root", "slow"), ("a3", "join"), ("slow", "join")),
              description="fast 3-chain beside one slow node"),
    GraphCase("wide_8", (Node("root", 0.05),) + tuple(Node(f"w{i}", 0.05 + 0.04 * i) for i in range(8)),
              tuple(("root", f"w{i}") for i in range(8)), description="fan-out to 8 leaves of varied speed"),
    GraphCase("chain_4_long", tuple(Node(f"k{i}", 0.08, output_chars=6000) for i in range(4)),
              (("k0", "k1"), ("k1", "k2"), ("k2", "k3")), description="linear chain with ~1500-token results"),
    GraphCase("layered_10", (Node("src", 0.05), Node("l1a", 0.1), Node("l1b", 0.3), Node("l1c", 0.05),
                             Node("l2a", 0.05), Node("l2b", 0.2), Node("l2c", 0.05), Node("l3a", 0.05),
                             Node("l3b", 0.1), Node("sink", 0.05)),
              (("src", "l1a"), ("src", "l1b"), ("src", "l1c"), ("l1a", "l2a"), ("l1c", "l2a"), ("l1b", "l2b"),
               ("l1c", "l2c"), ("l2a", "l3a"), ("l2c", "l3a"), ("l2c", "l3b"), ("l2b", "sink"), ("l3a", "sink"),
               ("l3b", "sink")), description="3 layers with cross edges and mixed latencies"),
]


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _summary_with_sources(view: RequestView) -> Turn:
    # Scan decoded strings: in raw JSON a newline before a name reads as "\nname".
    sources = sorted({m for text in _strings(json.loads(view.raw)) for m in _DONE.findall(text)})
    return Turn(text=f"{SUMMARY_TEXT} Sources: {', '.join(sources) or 'none'}.")


def _fake_for(case: GraphCase, plain_text_finish: bool = False) -> FakeLLM:
    """``plain_text_finish``: nodes stop with a text reply (LangGraph's signal) instead of agent_end."""
    nodes = {n.name: n for n in case.nodes}
    policies = {
        n.name: loop_policy(
            "lookup", n.rounds,
            Turn(text=n.final_text()) if plain_text_finish else end(n.final_text()),
        )
        for n in case.nodes
    }
    fake = FakeLLM(policies=policies, record=True)
    original_turn = fake._turn_for

    def turn_for(view: RequestView) -> Turn:
        return _summary_with_sources(view) if not view.tool_names else original_turn(view)

    fake._turn_for = turn_for  # type: ignore[method-assign]
    fake.latency = lambda view: nodes[view.agent].latency if view.tool_names else case.summary_latency
    return fake


def _context_problems(case: GraphCase, fake: FakeLLM, results: dict[str, Any]) -> list[str]:
    problems = [f"node {n.name} missing from results" for n in case.nodes if n.name not in results]
    for node, parents in case.parents().items():
        if not parents:
            continue
        first_requests = [r for r in fake.recorded if r.agent == node and '"tools"' in r.body]
        texts = " ".join(_strings(json.loads(first_requests[0].body))) if first_requests else ""
        found = re.findall(r"Sources: ([^.]*)\.", texts)
        # Context arrives either summarised (the summary lists its sources) or verbatim (raw results).
        seen = (set(found[0].split(", ")) if found else set()) | set(_DONE.findall(texts))
        missing = sorted(set(parents) - seen)
        if missing:
            problems.append(f"{node} started without context from {missing}")
    return problems


@dataclass
class GraphResult:
    name: str
    optimal_s: float
    actual_s: float
    llm_calls: int
    summaries: int
    summaries_needed: int
    digest: str
    problems: list[str]

    @property
    def efficiency(self) -> float:
        return self.optimal_s / self.actual_s if self.actual_s else 0.0


def run_case(case: GraphCase, reps: int) -> GraphResult:
    walls: list[float] = []
    problems: list[str] = []
    fake = _fake_for(case)
    digest = ""
    for rep in range(-1, reps):  # rep -1 is an untimed warm-up (thread pools, first connection, lazy imports)
        fake.reset_counters()
        fake.record = rep == 0
        fake_provider.activate(fake)
        nodes = [WorkflowNode(name=n.name, agent=make_agent(n.name, tools=[make_tool("lookup")]), instances=n.instances)
                 for n in case.nodes]
        workflow = IkaWorkflow(name=case.name, description=case.description, nodes=nodes,
                               edges=[WorkflowEdge(src, dst) for src, dst in case.edges])
        try:
            with harness.quiet():
                start = time.perf_counter()
                results = workflow.run(initial_context="Survey the topic.", use_async=True)
                if rep >= 0:
                    walls.append(time.perf_counter() - start)
        finally:
            fake_provider.activate(None)
        if rep == 0:
            problems = _context_problems(case, fake, results)
            fp, _ = harness.fingerprint(fake, {k: v.final for k, v in results.items()}, order_insensitive=True)
            digest = fp.digest
            calls = fake.calls
            summaries = fake.toolless_calls
    needed = sum(1 for n in case.nodes if case.needs_summary(n.name, IKACORE_THRESHOLD))
    return GraphResult(case.name, case.optimal_makespan(threshold=IKACORE_THRESHOLD), statistics.median(walls), calls,
                       summaries, needed,
                       digest, problems)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.graphs", description=__doc__.split("\n\n")[0])
    parser.add_argument("-k", "--filter", action="append", default=[])
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args(argv)
    base = json.loads(args.compare.read_text(encoding="utf-8")) if args.compare else {}

    fake_provider.install()
    cases = [c for c in CASES if not args.filter or any(f in c.name for f in args.filter)]
    header = f"{'graph':<18}{'optimal s':>10}{'actual s':>10}{'eff':>7}{'calls':>7}{'summaries':>11}"
    if base:
        header += f"{'before s':>10}{'before eff':>12}"
    print(header)
    print("-" * len(header))
    results = []
    with harness.sandbox_cwd():
        for case in cases:
            r = run_case(case, args.reps)
            results.append(r)
            line = (f"{r.name:<18}{r.optimal_s:>10.2f}{r.actual_s:>10.2f}{100 * r.efficiency:>6.0f}%{r.llm_calls:>7}"
                    f"{f'{r.summaries}/{r.summaries_needed}':>11}")
            if base and r.name in base:
                b = base[r.name]
                line += f"{b['actual_s']:>10.2f}{100 * b['optimal_s'] / b['actual_s']:>11.0f}%"
            print(line)
            for problem in r.problems:
                print(f"    ! {problem}")
    effs = [r.efficiency for r in results]
    print(f"\nmean scheduling efficiency: {100 * statistics.mean(effs):.0f}% of the critical-path optimum")
    if args.json:
        args.json.write_text(json.dumps({r.name: {**r.__dict__, "efficiency": r.efficiency} for r in results}, indent=2)
                             + "\n", encoding="utf-8")
    return 1 if any(r.problems for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
