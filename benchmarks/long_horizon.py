"""Long-horizon workflow, IkaCore vs LangGraph: a multi-minute audit on identical prompts, tools and models.

    <venv-with-langgraph>/python -m benchmarks.long_horizon --fake [--latency 2.0]    # free, deterministic
    <venv-with-langgraph>/python -m benchmarks.long_horizon --keys PATH --providers deepseek [--yes]

Fourteen agents, ~140 model calls, a critical path of ~35 sequential calls:

              ┌─► analyst_<r> ─► reviewer_<r> ─┐   x6 regions     ┌─► converter ─┐
    planner ──┤        ...           ...       ├─► auditor ───────┤              ├─► writer
              └─► analyst_<r> ─► reviewer_<r> ─┘                  └──────────────┘
                                   (writer also reads every reviewer)

Analysts list their region's accounts and look up every balance, one per turn,
then total them; reviewers look up every flagged transaction, one per turn, and
apply the adjustments to their analyst's total. Regions differ in both counts,
as real workloads do, so a region with many accounts can have few flags. The
writer is scored on nine exact facts (six adjusted totals, the top region, its
risk score, its EUR amount): a framework that loses or garbles context scores lower.

Frameworks (same prompt text, same tools, same model, context passed as
"Context from upstream workflow steps:" + upstream reports):
  ikacore        IkaWorkflow, async dataflow
  langgraph      idiomatic StateGraph: one node per agent (prebuilt ReAct agents), join edges
  lg_pipelined   hand-tuned StateGraph: each region's analyst+reviewer fused into one node,
                 which removes the superstep barrier between them (fake mode only by default)

``--fake`` serves every call from the offline fake provider with a deterministic
jittered latency per (agent, turn) around ``--latency`` seconds, so the critical-path
optimum is exact. Live mode times each node from its own requests and reports the
critical path of those measured spans, i.e. the best any scheduler could have done
with the latencies the provider actually delivered.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import operator
import random
import re
import sys
import threading
import time
import warnings
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import IkaCore  # noqa: F401
except ModuleNotFoundError:  # not pip-installed: benchmark the working tree
    sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarks.live_ab import PROVIDERS, load_keys  # noqa: E402

# --------------------------------------------------------------------------
# Task data (deterministic)
# --------------------------------------------------------------------------

ACCOUNT_COUNTS = {"north": 18, "south": 9, "east": 14, "west": 6, "central": 11, "coastal": 4}
FLAG_COUNTS = {"north": 3, "south": 10, "east": 5, "west": 12, "central": 7, "coastal": 9}
EUR_RATE = 0.92


def _data() -> tuple[dict[str, dict[str, int]], dict[str, dict[str, int]], dict[str, int]]:
    rng = random.Random(20260922)
    balances, flagged = {}, {}
    for r, n in ACCOUNT_COUNTS.items():
        balances[r] = {f"{r[:2].upper()}-{100 + 7 * i + rng.randint(0, 6)}": rng.randint(300, 4000) for i in range(n)}
        flagged[r] = {f"T{r[:2].upper()}{50 + 3 * i + rng.randint(0, 2)}": rng.randint(-700, 450)
                      for i in range(FLAG_COUNTS[r])}
    risk = {r: rng.randint(12, 88) for r in ACCOUNT_COUNTS}
    return balances, flagged, risk


BALANCES, FLAGGED, RISK = _data()
REGIONS = list(ACCOUNT_COUNTS)
TOTALS = {r: sum(BALANCES[r].values()) for r in REGIONS}
ADJUSTED = {r: TOTALS[r] + sum(FLAGGED[r].values()) for r in REGIONS}
TOP = max(ADJUSTED, key=ADJUSTED.__getitem__)
EUR = round(ADJUSTED[TOP] * EUR_RATE, 2)
FACTS: dict[str, tuple[str, ...]] = {
    **{f"{r} adjusted": (str(ADJUSTED[r]),) for r in REGIONS},
    "top region": (TOP,),
    "risk score": (str(RISK[TOP]),),
    "EUR amount": tuple({f"{EUR:.2f}", f"{EUR:g}"}),
}
INITIAL = "Quarter-end audit: produce exact adjusted totals for every region."
CONTEXT_LABEL = "Context from upstream workflow steps:"


# --------------------------------------------------------------------------
# Tools: plain functions shared by both frameworks
# --------------------------------------------------------------------------


def list_accounts(region: str) -> dict[str, Any]:
    return {"region": region, "accounts": list(BALANCES.get(str(region).lower(), {}))}


def get_balance(account: str) -> dict[str, Any]:
    found = next((b[account] for b in BALANCES.values() if account in b), None)
    return {"account": account, "balance": found if found is not None else "unknown account"}


def list_flagged(region: str) -> dict[str, Any]:
    return {"region": region, "flagged_transactions": list(FLAGGED.get(str(region).lower(), {}))}


def get_transaction(transaction_id: str) -> dict[str, Any]:
    found = next((f[transaction_id] for f in FLAGGED.values() if transaction_id in f), None)
    return {"transaction_id": transaction_id, "adjustment": found if found is not None else "unknown transaction"}


def sum_values(values: str) -> dict[str, Any]:
    numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(values).replace(",", " "))]
    total = sum(numbers)
    return {"count": len(numbers), "sum": int(total) if total.is_integer() else round(total, 2)}


def get_risk_score(region: str) -> dict[str, Any]:
    return {"region": region, "risk_score": RISK.get(str(region).lower(), "unknown region")}


def convert_currency(amount: float, currency: str) -> dict[str, Any]:
    return {"amount": amount, "currency": currency, "converted": round(float(amount) * EUR_RATE, 2)}


TOOLS: dict[str, tuple[Callable[..., Any], str, dict[str, str]]] = {
    "list_accounts": (list_accounts, "List the account ids in a region.", {"region": "string"}),
    "get_balance": (get_balance, "Return the balance of one account.", {"account": "string"}),
    "list_flagged": (list_flagged, "List the flagged transaction ids in a region.", {"region": "string"}),
    "get_transaction": (get_transaction, "Return the adjustment of one flagged transaction.",
                        {"transaction_id": "string"}),
    "sum_values": (sum_values, "Add up numbers given as a comma-separated string, e.g. '12, -3, 40'.",
                   {"values": "string"}),
    "get_risk_score": (get_risk_score, "Return the risk score of a region.", {"region": "string"}),
    "convert_currency": (convert_currency, "Convert an amount to a currency.",
                         {"amount": "number", "currency": "string"}),
}


# --------------------------------------------------------------------------
# Agents and graph shape (framework-neutral)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentSpec:
    name: str
    task: str
    tools: tuple[str, ...]
    parents: tuple[str, ...]
    calls: int  # model calls when the agent follows its instructions exactly


def specs() -> list[AgentSpec]:
    out = [AgentSpec("planner", f"You are planner. You coordinate a quarter-end audit of these regions: "
                     f"{', '.join(REGIONS)}. Report a one-line instruction for each regional analyst.", (), (), 1)]
    for r in REGIONS:
        out.append(AgentSpec(
            f"analyst_{r}",
            f"You are analyst_{r}. Call list_accounts with region='{r}', then call get_balance for each account, "
            f"one account per turn. Then call sum_values once with all the balances to get the exact total. "
            f"Report every account's balance and the total, written as 'TOTAL {r} = <number>'.",
            ("list_accounts", "get_balance", "sum_values"), ("planner",), ACCOUNT_COUNTS[r] + 3))
    for r in REGIONS:
        out.append(AgentSpec(
            f"reviewer_{r}",
            f"You are reviewer_{r}. Call list_flagged with region='{r}', then call get_transaction for each flagged "
            f"transaction, one per turn. Then call sum_values once with the analyst's {r} total and every adjustment "
            f"to get the adjusted total. Report it written as 'ADJUSTED {r} = <number>'.",
            ("list_flagged", "get_transaction", "sum_values"), (f"analyst_{r}",), FLAG_COUNTS[r] + 3))
    reviewers = tuple(f"reviewer_{r}" for r in REGIONS)
    out += [
        AgentSpec("auditor", "You are auditor. From the reviewers' reports, identify the region with the largest "
                  "adjusted total and call get_risk_score for it. Report the top region, its exact adjusted total "
                  "and its risk score.", ("get_risk_score",), reviewers, 2),
        AgentSpec("converter", "You are converter. From the auditor's report, take the top region's exact adjusted "
                  "total and call convert_currency with that amount and currency 'EUR'. Report the top region and "
                  "the exact converted EUR amount.", ("convert_currency",), ("auditor",), 2),
        AgentSpec("writer", "You are writer. Write the final audit summary from all upstream reports. It must state "
                  "every region's exact adjusted total, the top region, its risk score, and the exact EUR amount.",
                  (), (*reviewers, "auditor", "converter"), 1),
    ]
    return out


IKACORE_FINISH = " Finish by calling agent_end with your report."
LANGGRAPH_FINISH = " Finish with your report as your final reply."
NODE_RE = re.compile(r"You are ((?:analyst|reviewer)_[a-z]+|planner|auditor|converter|writer)\.")


def critical_path(durations: dict[str, float]) -> float:
    done: dict[str, float] = {}
    for spec in specs():  # specs() is topologically ordered
        done[spec.name] = max((done[p] for p in spec.parents), default=0.0) + durations.get(spec.name, 0.0)
    return max(done.values())


def superstep_bound(durations: dict[str, float]) -> float:
    """Best case for a barrier-synchronised (BSP) scheduler: layer by layer, each waits for its slowest node."""
    depth: dict[str, int] = {}
    for spec in specs():
        depth[spec.name] = max((depth[p] + 1 for p in spec.parents), default=0)
    layers: dict[int, float] = {}
    for name, d in depth.items():
        layers[d] = max(layers.get(d, 0.0), durations.get(name, 0.0))
    return sum(layers.values())


def score(final: str) -> tuple[int, list[str]]:
    text = final.replace(",", "").lower()
    missing = [label for label, values in FACTS.items() if not any(v.lower() in text for v in values)]
    return len(FACTS) - len(missing), missing


# --------------------------------------------------------------------------
# Traffic and per-node timing, measured under the frameworks without changing requests
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CallRecord:
    node: str  # "summary" for a context-summary request
    input_tokens: int  # everything the provider had to read, cached or not
    cached_tokens: int
    output_tokens: int
    seconds: float


def _usage_tokens(usage: dict[str, Any]) -> tuple[int, int, int]:
    """(total input, cached input, output) across Anthropic, DeepSeek and OpenAI-style usage blocks."""
    if "input_tokens" in usage:  # Anthropic reports cache reads/writes separately from input_tokens
        cached = int(usage.get("cache_read_input_tokens") or 0)
        total = int(usage["input_tokens"] or 0) + cached + int(usage.get("cache_creation_input_tokens") or 0)
        return total, cached, int(usage.get("output_tokens") or 0)
    details = usage.get("prompt_tokens_details") or {}
    cached = int(usage.get("prompt_cache_hit_tokens") or details.get("cached_tokens") or 0)
    return int(usage.get("prompt_tokens") or 0), cached, int(usage.get("completion_tokens") or 0)


@dataclass
class Traffic:
    calls: int = 0
    summary_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    spans: dict[str, list[float]] = field(default_factory=dict)  # node -> [first request start, last response end]
    log: list[CallRecord] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def durations(self) -> dict[str, float]:
        return {name: end - start for name, (start, end) in self.spans.items()}


def _http_modules() -> list[Any]:
    import httpx

    modules: list[Any] = [httpx]
    with contextlib.suppress(ImportError):
        import httpx2  # type: ignore[import-not-found]

        modules.append(httpx2)
    return modules


def _record(traffic: Traffic, request: Any, response: Any, started: float) -> None:
    ended = time.perf_counter()
    raw = (request.content or b"").decode("utf-8", "replace")
    try:
        body = json.loads(raw or "{}")
        usage = json.loads(response.content or b"{}").get("usage") or {}
    except ValueError:
        body, usage = {}, {}
    match = NODE_RE.search(raw)
    total_in, cached, out = _usage_tokens(usage)
    summary = not (body.get("tools") or match)  # tool-less agents still name themselves
    with traffic.lock:
        traffic.calls += 1
        traffic.summary_calls += int(summary)
        traffic.input_tokens += total_in
        traffic.output_tokens += out
        traffic.log.append(CallRecord("summary" if summary else match.group(1) if match else "?",
                                      total_in, cached, out, ended - started))
        if match:
            span = traffic.spans.setdefault(match.group(1), [started, ended])
            span[0], span[1] = min(span[0], started), max(span[1], ended)


@contextlib.contextmanager
def measure(traffic: Traffic) -> Iterator[None]:
    originals = []
    for module in _http_modules():
        original = module.HTTPTransport.handle_request

        def handle(self: Any, request: Any, _original: Any = original) -> Any:
            started = time.perf_counter()
            response = _original(self, request)
            response.read()
            _record(traffic, request, response, started)
            return response

        originals.append((module, original))
        module.HTTPTransport.handle_request = handle
    try:
        yield
    finally:
        for module, original in originals:
            module.HTTPTransport.handle_request = original


# --------------------------------------------------------------------------
# IkaCore
# --------------------------------------------------------------------------


def run_ikacore(provider: str, api_key: str, fake: bool) -> str:
    from IkaCore import IkaBaseAgent, IkaTools, IkaWorkflow, WorkflowEdge, WorkflowNode

    def tool(name: str) -> IkaTools:
        fn, description, params = TOOLS[name]
        return IkaTools(
            name=name, description=description,
            parameters={p: {"type": t, "description": p, "required": True} for p, t in params.items()},
            execute_function=lambda args, _fn=fn, _p=tuple(params): _fn(**{k: args.get(k) for k in _p}),
        )

    nodes, edges = [], []
    for spec in specs():
        agent = IkaBaseAgent(
            name=spec.name, description=f"{spec.name} of the quarter-end audit", prompt=spec.task + IKACORE_FINISH,
            tools=[tool(t) for t in spec.tools], model_id=PROVIDERS[provider]["model"],
            api_key=f"bench-{spec.name}" if fake else api_key, max_tokens=1024, temperature=0.0, maxsteps=60,
        )
        nodes.append(WorkflowNode(name=spec.name, agent=agent))
        edges += [WorkflowEdge(p, spec.name) for p in spec.parents]
    workflow = IkaWorkflow(name="long_audit", description="long-horizon audit", nodes=nodes, edges=edges,
                           start_node="planner")
    results = workflow.run(initial_context=INITIAL, use_async=True)
    return results["writer"].final if "writer" in results else ""


# --------------------------------------------------------------------------
# LangGraph
# --------------------------------------------------------------------------


def _lc_model(provider: str, api_key: str, name: str, fake: bool) -> Any:
    key = f"bench-{name}" if fake else api_key
    model = PROVIDERS[provider]["model"]
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, api_key=key, max_tokens=1024, temperature=0.0, max_retries=2)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, api_key=key, base_url="https://api.deepseek.com", max_tokens=1024,
                      temperature=0.0, max_retries=2, use_responses_api=False)


def _lc_tools(names: tuple[str, ...]) -> list[Any]:
    from langchain_core.tools import StructuredTool

    return [StructuredTool.from_function(TOOLS[n][0], name=n, description=TOOLS[n][1]) for n in names]


def _lc_agents(provider: str, api_key: str, fake: bool) -> dict[str, Callable[[str], str]]:
    warnings.filterwarnings("ignore", message=".*create_react_agent.*")  # still the documented prebuilt
    from langgraph.prebuilt import create_react_agent

    runners: dict[str, Callable[[str], str]] = {}
    for spec in specs():
        agent = create_react_agent(_lc_model(provider, api_key, spec.name, fake), _lc_tools(spec.tools))

        def run(context: str, _agent: Any = agent, _task: str = spec.task + LANGGRAPH_FINISH) -> str:
            prompt = f"{CONTEXT_LABEL}\n{context}\n\n{_task}" if context else _task
            out = _agent.invoke({"messages": [("user", prompt)]}, {"recursion_limit": 200})
            last = out["messages"][-1]
            return str(getattr(last, "text", None) or last.content)

        runners[spec.name] = run
    return runners


class State(TypedDict, total=False):
    results: Annotated[dict[str, str], operator.or_]


def run_langgraph(provider: str, api_key: str, fake: bool, pipelined: bool = False) -> str:
    from langgraph.graph import END, START, StateGraph

    agents = _lc_agents(provider, api_key, fake)
    by_name = {s.name: s for s in specs()}

    def context(state: State, parents: tuple[str, ...]) -> str:
        if not parents:
            return INITIAL
        results = state.get("results", {})
        return "\n\n".join(results[p] for p in parents if results.get(p))

    def node(names: tuple[str, ...]) -> Callable[[State], State]:
        def fn(state: State) -> State:
            done: dict[str, str] = {}
            for name in names:  # a fused node runs its agents in sequence, feeding each the previous output
                merged = {**state.get("results", {}), **done}
                done[name] = agents[name](context({"results": merged}, by_name[name].parents))
            return {"results": done}
        return fn

    groups = ({f"region_{r}": (f"analyst_{r}", f"reviewer_{r}") for r in REGIONS} if pipelined else {})
    fused = {member: group for group, members in groups.items() for member in members}
    graph = StateGraph(State)
    node_of = {s.name: fused.get(s.name, s.name) for s in specs()}
    for name in dict.fromkeys(node_of.values()):
        graph.add_node(name, node(groups.get(name, (name,))))
    graph.add_edge(START, "planner")
    for spec in specs():
        sources = tuple(dict.fromkeys(node_of[p] for p in spec.parents if node_of[p] != node_of[spec.name]))
        if sources:
            graph.add_edge(sources[0] if len(sources) == 1 else list(sources), node_of[spec.name])
    graph.add_edge("writer", END)
    state = graph.compile().invoke({"results": {}}, {"recursion_limit": 200})
    return state["results"].get("writer", "")


FRAMEWORKS: dict[str, Callable[[str, str, bool], str]] = {
    "ikacore": run_ikacore,
    "langgraph": run_langgraph,
    "lg_pipelined": lambda provider, key, fake: run_langgraph(provider, key, fake, pipelined=True),
}


# --------------------------------------------------------------------------
# Offline mode: the fake provider plays every agent, with realistic latency
# --------------------------------------------------------------------------


def jitter_latency(base: float) -> Callable[[str, int], float]:
    """Deterministic per-(agent, turn) latency in [0.5, 1.5) x base, like real providers' spread."""
    return lambda agent, turn: base * (0.5 + (zlib.crc32(f"{agent}:{turn}".encode()) % 1000) / 1000)


def _finish(view: Any, text: str) -> Any:
    from benchmarks.fake_provider import Turn, end

    return end(text) if "agent_end" in view.tool_names else Turn(text=text)


def _lookup_loop(tools: tuple[str, str, str], region: str, items: dict[str, int], needs: str,
                 report: Callable[[], str]) -> Any:
    from benchmarks.fake_provider import Call, Turn

    list_tool, item_tool, arg = tools

    def policy(view: Any) -> Any:
        if needs not in view.raw:  # an agent can only use what actually reached it
            return _finish(view, f"Cannot finish: missing input ({needs!r}).")
        if not view.count(list_tool):
            return Turn(calls=(Call(list_tool, {"region": region}),))
        done = view.count(item_tool)
        if done < len(items):
            return Turn(calls=(Call(item_tool, {arg: list(items)[done]}),))
        if not view.count("sum_values"):
            return Turn(calls=(Call("sum_values", {"values": ", ".join(map(str, items.values()))}),))
        return _finish(view, report())
    return policy


def _fake_policies() -> dict[str, Any]:
    from benchmarks.fake_provider import Call, Turn

    policies: dict[str, Any] = {"planner": lambda v: _finish(v, "Each analyst: audit your region's accounts.")}
    for r in REGIONS:
        policies[f"analyst_{r}"] = _lookup_loop(
            ("list_accounts", "get_balance", "account"), r, BALANCES[r], f"You are analyst_{r}.",
            lambda r=r: ", ".join(f"{a}={b}" for a, b in BALANCES[r].items()) + f". TOTAL {r} = {TOTALS[r]}")
        policies[f"reviewer_{r}"] = _lookup_loop(
            ("list_flagged", "get_transaction", "transaction_id"), r, FLAGGED[r], f"TOTAL {r} = {TOTALS[r]}",
            lambda r=r: f"ADJUSTED {r} = {ADJUSTED[r]}")

    def auditor(view: Any) -> Any:
        if not all(f"ADJUSTED {r} = {ADJUSTED[r]}" in view.raw for r in REGIONS):
            return _finish(view, "Cannot finish: some reviewer reports are missing.")
        if not view.count("get_risk_score"):
            return Turn(calls=(Call("get_risk_score", {"region": TOP}),))
        return _finish(view, f"Top region {TOP}, adjusted total {ADJUSTED[TOP]}, risk score {RISK[TOP]}.")

    def converter(view: Any) -> Any:
        if str(ADJUSTED[TOP]) not in view.raw:
            return _finish(view, "Cannot finish: the auditor's total is missing.")
        if not view.count("convert_currency"):
            return Turn(calls=(Call("convert_currency", {"amount": ADJUSTED[TOP], "currency": "EUR"}),))
        return _finish(view, f"{TOP}: EUR {EUR:.2f}")

    def writer(view: Any) -> Any:  # reports exactly the facts that reached it
        present = [values[0] for values in FACTS.values() if any(v in view.raw for v in values)]
        return _finish(view, "Audit summary: " + "; ".join(present))

    policies.update(auditor=auditor, converter=converter, writer=writer)
    return policies


def make_fake(base_latency: float) -> Any:
    from benchmarks.fake_provider import FakeLLM

    fake = FakeLLM(policies=_fake_policies())
    latency = jitter_latency(base_latency)
    summarise = fake._turn_for

    def turn_for(view: Any) -> Any:
        match = NODE_RE.search(view.raw)
        if not view.tool_names and match and match.group(1) == view.agent:
            return fake.policies[view.agent](view)  # a tool-less agent (LangGraph planner/writer), not a summary
        return summarise(view)

    fake._turn_for = turn_for  # type: ignore[method-assign]
    fake.latency = lambda view: latency(view.agent, len(view.prior_calls))
    return fake


def fake_optimum(base_latency: float) -> tuple[float, float]:
    latency = jitter_latency(base_latency)
    durations = {s.name: sum(latency(s.name, k) for k in range(s.calls)) for s in specs()}
    return critical_path(durations), superstep_bound(durations)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------


@dataclass
class Run:
    framework: str
    provider: str
    score: int
    missing: list[str]
    traffic: Traffic
    wall: float
    error: str = ""

    @property
    def measured_optimum(self) -> float:
        return critical_path(self.traffic.durations())


def run_one(framework: str, provider: str, api_key: str, fake: bool) -> Run:
    traffic = Traffic()
    start = time.perf_counter()
    try:
        with measure(traffic), contextlib.redirect_stdout(io.StringIO()):
            final = FRAMEWORKS[framework](provider, api_key, fake)
    except Exception as exc:  # noqa: BLE001 - an API failure is a result to report
        return Run(framework, provider, 0, list(FACTS), traffic, time.perf_counter() - start,
                   f"{type(exc).__name__}: {str(exc)[:300]}")
    wall = time.perf_counter() - start
    points, missing = score(final)
    return Run(framework, provider, points, missing, traffic, wall)


def print_table(runs: list[Run]) -> None:
    head = (f"{'framework':<14}{'provider':<10}{'score':>7}{'calls':>7}{'summ':>6}{'in tok':>10}{'out tok':>9}"
            f"{'wall':>9}{'best possible':>15}{'efficiency':>12}")
    print(head)
    print("-" * len(head))
    for r in runs:
        opt = r.measured_optimum
        print(f"{r.framework:<14}{r.provider:<10}{r.score:>5}/{len(FACTS)}{r.traffic.calls:>7}"
              f"{r.traffic.summary_calls:>6}{r.traffic.input_tokens:>10,}{r.traffic.output_tokens:>9,}"
              f"{r.wall:>8.1f}s{opt:>14.1f}s{100 * opt / r.wall:>11.0f}%")
        if r.missing:
            print(f"    missing: {r.missing}")
        if r.error:
            print(f"    error: {r.error}")
    print("\nbest possible = critical path of the per-node spans measured in that run (what a perfect scheduler")
    print("would have achieved with the same model latencies); efficiency = best possible / wall.")


def run_fake(frameworks: list[str], latency: float) -> list[Run]:
    from benchmarks import fake_provider

    fake_provider.install()
    opt, bsp = fake_optimum(latency)
    print(f"fake provider, {latency:.1f}s mean per call (jittered 0.5-1.5x); "
          f"{sum(s.calls for s in specs())} calls planned")
    print(f"critical-path optimum {opt:.1f}s; best case for a layer-by-layer (superstep) scheduler {bsp:.1f}s\n")
    runs = []
    for framework in frameworks:
        fake_provider.activate(make_fake(latency))
        try:
            runs.append(run_one(framework, "deepseek", "unused", fake=True))
        finally:
            fake_provider.activate(None)
        print(f"  {framework}: {runs[-1].wall:.1f}s, {runs[-1].score}/{len(FACTS)}", flush=True)
    return runs


def _solve(rows: list[list[float]], ys: list[float]) -> list[float]:
    """Least squares via the normal equations (tiny systems only)."""
    n = len(rows[0])
    a = [[sum(r[i] * r[j] for r in rows) for j in range(n)] + [sum(r[i] * y for r, y in zip(rows, ys))]
         for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda k: abs(a[k][col]))
        a[col], a[pivot] = a[pivot], a[col]
        if abs(a[col][col]) < 1e-12:
            return [0.0] * n
        for k in range(n):
            if k != col:
                f = a[k][col] / a[col][col]
                a[k] = [x - f * y for x, y in zip(a[k], a[col])]
    return [a[i][n] / a[i][i] for i in range(n)]


def latency_breakdown(runs: list[Run]) -> None:
    """Fit seconds = fixed + per uncached input token + per cached input token + per output token."""
    for provider in dict.fromkeys(r.provider for r in runs):
        log = [c for r in runs if r.provider == provider for c in r.traffic.log if c.input_tokens]
        if len(log) < 8:
            continue
        rows = [[1.0, c.input_tokens - c.cached_tokens, c.cached_tokens, c.output_tokens] for c in log]
        fixed, per_in, per_cached, per_out = _solve(rows, [c.seconds for c in log])
        secs = sorted(c.seconds for c in log)
        print(f"\n{provider}: {len(log)} calls, latency median {secs[len(secs) // 2]:.2f}s, "
              f"p90 {secs[int(len(secs) * 0.9)]:.2f}s, max {secs[-1]:.2f}s")
        print(f"  fit: {fixed:.2f}s fixed + {1000 * per_in * 1000:.1f}ms per 1k uncached input tokens "
              f"+ {1000 * per_cached * 1000:.1f}ms per 1k cached + {1000 * per_out:.1f}ms per output token")
        for r in (r for r in runs if r.provider == provider and r.traffic.log):
            calls = [c for c in r.traffic.log if c.input_tokens]
            mean = lambda f: sum(map(f, calls)) / len(calls)  # noqa: E731
            parts = (fixed, per_in * mean(lambda c: c.input_tokens - c.cached_tokens),
                     per_cached * mean(lambda c: c.cached_tokens), per_out * mean(lambda c: c.output_tokens))
            print(f"  {r.framework:<13} avg call: {mean(lambda c: c.input_tokens):,.0f} in "
                  f"({mean(lambda c: c.cached_tokens):,.0f} cached), {mean(lambda c: c.output_tokens):.0f} out, "
                  f"{mean(lambda c: c.seconds):.2f}s = fixed {parts[0]:.2f} + input {parts[1] + parts[2]:.2f} "
                  f"+ output {parts[3]:.2f}")


def dump_calls(runs: list[Run], path: Path) -> None:
    with path.open("w", encoding="utf-8") as out:
        out.write("provider,framework,node,input_tokens,cached_tokens,output_tokens,seconds\n")
        for r in runs:
            for c in r.traffic.log:
                out.write(f"{r.provider},{r.framework},{c.node},{c.input_tokens},{c.cached_tokens},"
                          f"{c.output_tokens},{c.seconds:.3f}\n")


def run_live(frameworks: list[str], providers: list[str], keys: dict[str, str]) -> list[Run]:
    runs = []
    for provider in providers:
        for framework in frameworks:
            run = run_one(framework, provider, keys[PROVIDERS[provider]["env"]], fake=False)
            for secret in keys.values():
                run.error = run.error.replace(secret, "<redacted>")
            runs.append(run)
            print(f"  {provider} {framework}: {run.wall:.1f}s, {run.score}/{len(FACTS)}, "
                  f"in {run.traffic.input_tokens:,} out {run.traffic.output_tokens:,} {run.error}", flush=True)
    return runs


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.long_horizon", description=__doc__.split("\n\n")[0])
    parser.add_argument("--fake", action="store_true", help="offline fake provider with simulated latency")
    parser.add_argument("--latency", type=float, default=2.0, help="fake mode: mean seconds per model call")
    parser.add_argument("--frameworks", default=None, help="comma list of: " + ", ".join(FRAMEWORKS))
    parser.add_argument("--providers", default="deepseek")
    parser.add_argument("--keys", type=Path)
    parser.add_argument("--yes", action="store_true", help="live mode: actually call the providers")
    parser.add_argument("--calls-csv", type=Path, help="write every call's tokens and latency to this CSV")
    args = parser.parse_args(argv)
    default = "ikacore,langgraph,lg_pipelined" if args.fake else "ikacore,langgraph"
    frameworks = (args.frameworks or default).split(",")
    if args.fake:
        runs = run_fake(frameworks, args.latency)
    else:
        keys = load_keys(args.keys)
        providers = [p for p in args.providers.split(",") if keys.get(PROVIDERS[p]["env"])]
        print(f"live plan: {frameworks} x {providers} = {len(frameworks) * len(providers)} workflow runs, "
              f"~{sum(s.calls for s in specs())} model calls each")
        if not args.yes or not providers:
            print("not calling providers: pass --yes")
            return 0
        runs = run_live(frameworks, providers, keys)
    print()
    print_table(runs)
    if not args.fake:
        latency_breakdown(runs)
    if args.calls_csv:
        dump_calls(runs, args.calls_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
