"""Workflow graphs head to head: IkaCore vs LangGraph on the same DAGs and latencies.

    <venv-with-langgraph>/python -m benchmarks.compare_graphs [--reps 3]

Every case from ``benchmarks.graphs`` runs on both frameworks against the same
fake provider with the same simulated per-node model latency, so both are
scored against the same critical path. LangGraph runs two ways:

  mirrored  IkaCore's previous semantics: each node first summarises its upstream
            context with one LLM call (one shared summary node fanning out with
            ``Send`` for multi-instance nodes), then runs its agent. This
            compares the schedulers on identical work.
  native    idiomatic LangGraph: upstream results go straight into the prompt,
            no summary call. This is what a LangGraph user gets out of the box.

Nodes are prebuilt ReAct agents with a ``lookup`` tool making the same number
of calls as the IkaCore agents; multi-instance nodes use ``Send``, multi-parent
nodes use join edges (``add_edge([parents], node)``). Models are built before
timing starts, and each case runs once untimed as a warm-up.
"""

from __future__ import annotations

import argparse
import operator
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import IkaCore  # noqa: F401
except ModuleNotFoundError:  # not pip-installed: benchmark the working tree
    sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarks import fake_provider, graphs, harness  # noqa: E402
from benchmarks.graphs import GraphCase  # noqa: E402

INITIAL = "Survey the topic."


def _merge(left: dict[str, str], right: dict[str, str]) -> dict[str, str]:
    return {**left, **right}


class State(TypedDict, total=False):
    results: Annotated[list[str], operator.add]
    ctx: Annotated[dict[str, str], _merge]


def _lookup_tool() -> Any:
    from langchain_core.tools import StructuredTool

    def lookup(query: str, step: int, slot: int = 0) -> dict[str, Any]:
        return {"tool": "lookup", "query": query, "result": f"lookup-{step}-{slot}"}

    return StructuredTool.from_function(lookup, name="lookup", description="Look up information with lookup.")


def build_langgraph(case: GraphCase, mirrored: bool) -> Any:
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import create_react_agent
    from langgraph.types import Send

    nodes = {n.name: n for n in case.nodes}
    parents = case.parents()
    children: dict[str, list[str]] = defaultdict(list)
    for src, dst in case.edges:
        children[src].append(dst)
    start = case.nodes[0].name
    models = {name: ChatOpenAI(model="gpt-4o", api_key=f"bench-{name}", use_responses_api=False, max_retries=0)
              for name in nodes}
    agents = {name: create_react_agent(models[name], [_lookup_tool()]) for name in nodes}

    def upstream(state: State, name: str) -> str:
        if name == start:
            return INITIAL
        wanted = set(parents[name])
        return "\n\n".join(r for r in state.get("results", []) if r.split(" done:")[0] in wanted)

    def summarise(name: str, text: str) -> str:
        return str(models[name].invoke([("user", f"Summarise for the next step:\n\n{text}")]).content)

    def node_fn(name: str) -> Any:
        def run(state: State) -> dict[str, Any]:
            ctx = state.get("ctx", {}).get(name)
            if ctx is None:
                text = upstream(state, name)
                ctx = summarise(name, text) if mirrored else text
            out = agents[name].invoke({"messages": [("user", f"You are {name}. Context:\n{ctx}\nUse lookup, then answer.")]})
            return {"results": [str(out["messages"][-1].content)]}
        return run

    def ctx_fn(name: str) -> Any:
        return lambda state: {"ctx": {name: summarise(name, upstream(state, name))}}

    def fan_out(targets: list[str]) -> Any:
        def route(state: State) -> list[Any]:
            payload = {"results": state.get("results", []), "ctx": state.get("ctx", {})}
            return [Send(t, payload) for t in targets for _ in range(nodes[t].instances)]
        return route

    graph = StateGraph(State)
    for name, node in nodes.items():
        graph.add_node(name, node_fn(name))
        if node.instances > 1:
            assert len(parents[name]) == 1, "multi-instance nodes need a single parent in this harness"
            if mirrored:
                graph.add_node(f"ctx_{name}", ctx_fn(name))
                graph.add_edge(parents[name][0], f"ctx_{name}")
                graph.add_conditional_edges(f"ctx_{name}", fan_out([name]))
    graph.add_edge(START, start)
    for src, kids in children.items():
        sent = [k for k in kids if nodes[k].instances > 1 and not mirrored]
        if sent:
            graph.add_conditional_edges(src, fan_out(sent))
    for name in nodes:
        if nodes[name].instances > 1 or name == start:
            continue
        ps = parents[name]
        graph.add_edge(ps[0] if len(ps) == 1 else ps, name)
    for name in nodes:
        if not children.get(name):
            graph.add_edge(name, END)
    return graph.compile()


def run_langgraph(case: GraphCase, mirrored: bool, reps: int) -> tuple[float, int, list[str]]:
    fake = graphs._fake_for(case, plain_text_finish=True)
    walls: list[float] = []
    calls = 0
    problems: list[str] = []
    graph = build_langgraph(case, mirrored)
    for rep in range(-1, reps):
        fake.reset_counters()
        fake_provider.activate(fake)
        try:
            start = time.perf_counter()
            state = graph.invoke({"results": [], "ctx": {}}, {"recursion_limit": 1000})
            wall = time.perf_counter() - start
        finally:
            fake_provider.activate(None)
        if rep >= 0:
            walls.append(wall)
        if rep == 0:
            calls = fake.calls
            finished = [r.split(" done:")[0] for r in state["results"]]
            problems = [f"{n.name} ran {finished.count(n.name)}x, expected {n.instances}x"
                        for n in case.nodes if finished.count(n.name) != n.instances]
    return statistics.median(walls), calls, problems


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.compare_graphs", description=__doc__.split("\n\n")[0])
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args(argv)
    from importlib.metadata import version

    fake_provider.install()
    print(f"IkaCore (working tree) vs langgraph {version('langgraph')}; same DAGs, same simulated model latency\n")
    head = (f"{'graph':<17}{'optimum':>8}{'IkaCore':>14}{'LG mirrored':>16}{'calls I/LG':>12}"
            f"{'native opt':>11}{'LG native':>14}")
    print(head)
    print("-" * len(head))
    problems: list[str] = []
    with harness.sandbox_cwd(), harness.quiet():
        rows = []
        for case in graphs.CASES:
            ika = graphs.run_case(case, args.reps)
            lg_m, lg_calls, p1 = run_langgraph(case, mirrored=True, reps=args.reps)
            lg_n, _, p2 = run_langgraph(case, mirrored=False, reps=args.reps)
            problems += ika.problems + [f"LG mirrored {case.name}: {p}" for p in p1] + [f"LG native {case.name}: {p}" for p in p2]
            rows.append((case, ika, lg_m, lg_calls, lg_n))
    for case, ika, lg_m, lg_calls, lg_n in rows:
        opt = case.optimal_makespan(threshold=graphs.IKACORE_THRESHOLD)
        opt_native = case.optimal_makespan(with_summaries=False)
        opt_mirrored = case.optimal_makespan(summarise_initial=True)  # mirrored LG = IkaCore's old always-summarise work
        print(f"{case.name:<17}{opt:>7.2f}s{ika.actual_s:>7.2f}s ({100 * opt / ika.actual_s:>3.0f}%)"
              f"{lg_m:>8.2f}s ({100 * opt_mirrored / lg_m:>3.0f}%){f'{ika.llm_calls}/{lg_calls}':>12}"
              f"{opt_native:>10.2f}s{lg_n:>7.2f}s ({100 * opt_native / lg_n:>3.0f}%)")
    print("\n(%) = critical-path optimum / actual wall time for that setup; 100% = no avoidable waiting.")
    for problem in problems:
        print(f"  ! {problem}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
