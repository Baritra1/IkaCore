"""Benchmark scenarios covering IkaCore's execution surface.

Each scenario has an untimed ``setup`` that builds fresh agents (agents are
stateful, so every iteration needs its own) and returns the zero-argument
callable that is timed. ``check`` asserts the output is semantically right,
and the behavior fingerprint (see ``harness.py``) pins the exact requests
IkaCore sends and the exact results it returns.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from benchmarks.fake_provider import Call, Policy, RequestView, Turn, call, end
from IkaCore import IkaBaseAgent, IkaStage, IkaTools, IkaWorkflow, WorkflowEdge, WorkflowNode

Run = Callable[[], Any]


@dataclass
class Scenario:
    name: str
    group: str
    description: str
    setup: Callable[[], Run]
    policies: dict[str, Policy]
    check: Callable[[Any], None] = lambda out: None
    default_policy: Optional[Policy] = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    # IkaCore joins parallel results in completion order, so for concurrent
    # scenarios the fingerprint compares strings and per-agent requests as
    # multisets: any content change is caught, pure reordering is tolerated.
    order_insensitive: bool = False


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

PROVIDER_MODELS: dict[str, dict[str, Any]] = {
    "openai_responses": {"model_id": "gpt-4o"},
    "openai_chat": {"model_id": "gpt-4o", "use_responses_api": False},
    "anthropic": {"model_id": "claude-sonnet-4-5"},
    "gemini": {"model_id": "gemini-2.5-flash"},
    "deepseek": {"model_id": "deepseek-chat"},
    "openrouter": {"model_id": "openai/gpt-4o-mini"},
    "zai": {"model_id": "glm-4.6"},
    "codex": {"model_id": "gpt-5-codex"},
}

FINAL = "Final answer: the requested facts were gathered and verified end to end."


def _payload(size: int, seed: str) -> str:
    unit = f"{seed}:lorem ipsum dolor sit amet consectetur adipiscing elit; "
    return (unit * (size // len(unit) + 1))[:size]


def make_tool(name: str, output_size: int = 96, n_params: int = 2) -> IkaTools:
    params: dict[str, Any] = {
        "query": {"type": "string", "description": "What to look up", "required": True},
        "step": {"type": "integer", "description": "Step counter", "required": True},
    }
    for idx in range(max(0, n_params - 2)):
        params[f"opt_{idx}"] = f"Optional parameter number {idx}"

    def execute(args: dict[str, Any]) -> dict[str, Any]:
        seed = f"{name}-{args.get('step', 0)}-{args.get('slot', 0)}"
        return {"tool": name, "query": args.get("query"), "result": _payload(output_size, seed)}

    return IkaTools(
        name=name,
        description=f"Look up information with {name}. Returns a JSON document.",
        parameters=params,
        execute_function=execute,
        id=f"tool-{name}",
    )


# Provider for scenarios that don't pin one. Override (e.g. for provider A/B runs) with
# IKABENCH_PROVIDER; golden.json fingerprints assume the default.
DEFAULT_PROVIDER = os.environ.get("IKABENCH_PROVIDER", "openai_responses")


def make_agent(name: str, provider: Optional[str] = None, **kwargs: Any) -> IkaBaseAgent:
    provider = provider or DEFAULT_PROVIDER
    config: dict[str, Any] = {
        "name": name,
        "description": f"Benchmark agent {name}",
        "prompt": f"You are {name}. Research the topic, use your tools, then finish with agent_end.",
        "system_prompt": "You are a precise, concise research assistant.",
        "api_key": f"bench-{name}",
        **PROVIDER_MODELS[provider],
    }
    config.update(kwargs)
    return IkaBaseAgent(**config)


def loop_policy(tool: str, rounds: int, final: Turn, width: int = 1) -> Policy:
    """Call ``tool`` ``rounds`` times (``width`` parallel calls each), then ``final``."""

    def policy(view: RequestView) -> Turn:
        done = view.count(tool)
        if done >= rounds * width:
            return final
        step = done // width
        return Turn(calls=tuple(
            Call(tool, {"query": f"{tool} topic {step}.{slot}", "step": step, "slot": slot})
            for slot in range(width)
        ))

    return policy


def _final_message(out: Any) -> str:
    assert isinstance(out, dict), f"expected dict output, got {type(out).__name__}"
    return str(out.get("final_message", ""))


def expect_final(text: str) -> Callable[[Any], None]:
    def check(out: Any) -> None:
        got = _final_message(out)
        assert text in got, f"final_message mismatch: {got[:200]!r}"

    return check


# --------------------------------------------------------------------------
# Scenario families
# --------------------------------------------------------------------------


def _simple_end(provider: str) -> Scenario:
    name = f"single_turn[{provider}]"
    return Scenario(
        name=name,
        group="agent",
        description=f"One LLM call that immediately calls agent_end ({provider}).",
        setup=lambda: make_agent("solo", provider).execution,
        policies={"solo": lambda view: end(FINAL)},
        check=expect_final(FINAL),
        tags=("provider",),
    )


def _tool_loop(provider: str, rounds: int = 10) -> Scenario:
    return Scenario(
        name=f"tool_loop_{rounds}[{provider}]",
        group="agent",
        description=f"{rounds} sequential tool rounds then agent_end ({provider}).",
        setup=lambda: make_agent("looper", provider, tools=[make_tool("lookup"), make_tool("aux")]).execution,
        policies={"looper": loop_policy("lookup", rounds, end(FINAL))},
        check=expect_final(FINAL),
        tags=("provider",),
    )


def _parallel_tools(width: int = 16) -> Scenario:
    return Scenario(
        name=f"parallel_tools_{width}",
        group="agent",
        description=f"Two rounds of {width} parallel tool calls in a single response, then agent_end.",
        setup=lambda: make_agent("fanner", tools=[make_tool("lookup")]).execution,
        policies={"fanner": loop_policy("lookup", 2, end(FINAL), width=width)},
        check=expect_final(FINAL),
    )


def _long_conversation(rounds: int = 40) -> Scenario:
    return Scenario(
        name=f"long_conversation_{rounds}",
        group="scaling",
        description=f"{rounds} tool rounds with 1 KB outputs; exposes per-round cost growth with history size.",
        setup=lambda: make_agent("marathon", tools=[make_tool("lookup", output_size=1024)]).execution,
        policies={"marathon": loop_policy("lookup", rounds, end(FINAL))},
        check=expect_final(FINAL),
    )


def _large_payload() -> Scenario:
    def setup() -> Run:
        tools = [make_tool(f"tool_{i:02d}", output_size=4096, n_params=6) for i in range(40)]
        agent = make_agent(
            "heavy",
            tools=tools,
            system_prompt=_payload(20_000, "system"),
            prompt=_payload(8_000, "prompt"),
        )
        return agent.execution

    return Scenario(
        name="large_payload",
        group="scaling",
        description="20 KB system prompt, 8 KB prompt, 40 tool schemas, 4 KB tool outputs, 6 rounds.",
        setup=setup,
        policies={"heavy": loop_policy("tool_07", 6, end(FINAL))},
        check=expect_final(FINAL),
    )


def _stage_policy(view: RequestView) -> Turn:
    for tool in ("stage_a_tool", "stage_b_tool", "stage_c_tool"):
        if tool in view.tool_names:
            done = view.count(tool)
            if done < 2:
                return call(tool, query=f"{tool} item {done}", step=done)
            if tool == "stage_c_tool":
                return end(FINAL)
            return Turn(calls=(Call("stage_end", {"reason": f"{tool} complete"}),))
    return end("UNEXPECTED STAGE")


def _staged() -> Scenario:
    def setup() -> Run:
        stages = [
            IkaStage("Gather", "Gather the raw facts.", [make_tool("stage_a_tool")]),
            IkaStage("Analyse", "Analyse the gathered facts.", [make_tool("stage_b_tool")]),
            IkaStage("Report", "Write the final report.", [make_tool("stage_c_tool")]),
        ]
        return make_agent("staged", Stages=stages).execution

    return Scenario(
        name="staged_3x3",
        group="agent",
        description="Three stages, each with two tool rounds and a stage transition (stage_end / agent_end).",
        setup=setup,
        policies={"staged": _stage_policy},
        check=expect_final(FINAL),
    )


def _subagents(count: int = 3) -> Scenario:
    names = [f"worker_{i}" for i in range(count)]

    def parent_policy(view: RequestView) -> Turn:
        if view.count(*names) == 0:
            return Turn(calls=tuple(Call(n, {"input": f"Research sub-topic {i}"}) for i, n in enumerate(names)))
        return end(FINAL)

    def setup() -> Run:
        workers = [make_agent(n, tools=[make_tool("lookup")]) for n in names]
        return make_agent("lead", subagents=workers).execution

    policies: dict[str, Policy] = {"lead": parent_policy}
    policies.update({n: loop_policy("lookup", 2, end(f"{n} findings are complete and consistent.")) for n in names})
    return Scenario(
        name=f"subagents_{count}",
        group="multi-agent",
        description=f"Parent delegates to {count} subagents in one parallel turn; each runs 2 tool rounds.",
        setup=setup,
        policies=policies,
        check=expect_final(FINAL),
    )


def _workflow_chain(length: int = 4, use_async: bool = False) -> Scenario:
    names = [f"chain_{i}" for i in range(length)]

    def setup() -> Run:
        nodes = [WorkflowNode(name=n, agent=make_agent(n, tools=[make_tool("lookup")])) for n in names]
        edges = [WorkflowEdge(source=a, target=b) for a, b in zip(names, names[1:])]
        workflow = IkaWorkflow(name="chain", description="linear chain", nodes=nodes, edges=edges)
        return lambda: workflow.run(initial_context="Investigate the history of the number 42.", use_async=use_async)

    def check(out: Any) -> None:
        assert set(out) == set(names), f"workflow nodes mismatch: {sorted(out)}"
        assert "chain_3 done" in out[names[-1]].final, out[names[-1]].final

    mode = "async" if use_async else "sync"
    return Scenario(
        name=f"workflow_chain_{length}[{mode}]",
        group="multi-agent",
        description=f"Linear {length}-node workflow ({mode}); each node does 1 tool round and propagates a summary.",
        setup=setup,
        policies={n: loop_policy("lookup", 1, end(f"{n} done: summary for the next node.")) for n in names},
        check=check,
    )


def _workflow_fanout(instances: int = 4) -> Scenario:
    branches = ["branch_a", "branch_b", "branch_c"]

    def setup() -> Run:
        tool = [make_tool("lookup")]
        nodes = [WorkflowNode(name="root", agent=make_agent("root", tools=tool))]
        nodes += [
            WorkflowNode(
                name=b,
                agent=make_agent(b, tools=tool),
                instances=instances,
                instance_inputs=[f"Investigate facet {i} of {b}." for i in range(instances)],
            )
            for b in branches
        ]
        nodes.append(WorkflowNode(name="merge", agent=make_agent("merge", tools=tool)))
        edges = [WorkflowEdge(source="root", target=b) for b in branches]
        edges += [WorkflowEdge(source=b, target="merge") for b in branches]
        workflow = IkaWorkflow(name="fanout", description="diamond", nodes=nodes, edges=edges)
        return lambda: workflow.run(initial_context="Survey the topic broadly.", use_async=True)

    def check(out: Any) -> None:
        assert "merge" in out and "merge done" in out["merge"].final, sorted(out)

    policies: dict[str, Policy] = {"root": loop_policy("lookup", 1, end("root done: plan fanned out."))}
    policies.update({b: loop_policy("lookup", 2, end(f"{b} done: facet findings.")) for b in branches})
    policies["merge"] = loop_policy("lookup", 1, end("merge done: all branches reconciled."))
    return Scenario(
        name=f"workflow_fanout_3x{instances}[async]",
        group="multi-agent",
        description=f"Diamond DAG root -> 3 branches x {instances} instances -> merge, run async in parallel.",
        setup=setup,
        policies=policies,
        check=check,
        order_insensitive=True,
    )


def _async_agent(rounds: int = 10) -> Scenario:
    return Scenario(
        name=f"async_tool_loop_{rounds}",
        group="agent",
        description=f"use_async=True agent running {rounds} tool rounds.",
        setup=lambda: make_agent("asyncer", tools=[make_tool("lookup")], use_async=True).execution,
        policies={"asyncer": loop_policy("lookup", rounds, end(FINAL))},
        check=expect_final(FINAL),
    )


def _checkpointed(rounds: int = 10) -> Scenario:
    def setup() -> Run:
        path = os.path.join(tempfile.mkdtemp(prefix="ikabench-"), "checkpoints.db")
        return make_agent("saver", tools=[make_tool("lookup")], checkpoint=True, checkpoint_db_path=path).execution

    return Scenario(
        name=f"checkpointed_{rounds}",
        group="durability",
        description=f"checkpoint=True (SQLite) agent with {rounds} tool rounds, checkpointing every step.",
        setup=setup,
        policies={"saver": loop_policy("lookup", rounds, end(FINAL))},
        check=expect_final(FINAL),
    )


REPLY_MARKER = "BENCH-USER-REPLY: use the 2024 dataset."


def _hitl_policy(view: RequestView) -> Turn:
    if REPLY_MARKER not in view.raw:
        return Turn(calls=(Call("ask_user", {"question": "Which dataset should I use?"}),))
    done = view.count("lookup")
    if done < 2:
        return call("lookup", query=f"dataset item {done}", step=done)
    return end(FINAL)


def _hitl_resume() -> Scenario:
    def setup() -> Run:
        path = os.path.join(tempfile.mkdtemp(prefix="ikabench-"), "checkpoints.db")
        stage = IkaStage("Clarify", "Ask the user which dataset to use.", [make_tool("lookup")], hitl=True, checkpoint=True)
        agent = make_agent("hitl", Stages=[stage], checkpoint=True, checkpoint_db_path=path)

        def run() -> Any:
            first = agent.execution()
            assert first.get("status") == "awaiting_user_input", first.get("status")
            return agent.execution(checkpoint_uid=first["checkpoint_uid"], resume_input=REPLY_MARKER)

        return run

    return Scenario(
        name="hitl_interrupt_resume",
        group="durability",
        description="ask_user interrupt, checkpoint, then resume with user input and finish.",
        setup=setup,
        policies={"hitl": _hitl_policy},
        check=expect_final(FINAL),
    )


def mentions_revision(output: dict[str, Any]) -> bool:
    """The final answer must be the revised version."""
    return "revised" in str(output.get("final_message", ""))


def _validated() -> Scenario:
    def policy(view: RequestView) -> Turn:
        return end("revised answer after feedback.") if "[FEEDBACK]" in view.raw else end("first draft answer.")

    return Scenario(
        name="final_answer_retry",
        group="agent",
        description="final_answer_check fails once, forcing a feedback retry.",
        setup=lambda: make_agent("checked", final_answer_check=[mentions_revision]).execution,
        policies={"checked": policy},
        check=expect_final("revised answer"),
    )


def _next_agent() -> Scenario:
    def setup() -> Run:
        second = make_agent("second", tools=[make_tool("lookup")])
        first = make_agent("first", tools=[make_tool("lookup")], next_agent=second, summarize_final=True)
        return first.execution

    return Scenario(
        name="next_agent_summarized",
        group="multi-agent",
        description="next_agent hand-off with summarize_final=True (includes a summarization LLM call).",
        setup=setup,
        policies={
            "first": loop_policy("lookup", 2, end("first agent findings.")),
            "second": loop_policy("lookup", 2, end(FINAL)),
        },
        check=expect_final(FINAL),
    )


def _step_exhaustion() -> Scenario:
    def policy(view: RequestView) -> Turn:
        done = view.count("lookup")
        return call("lookup", query=f"never done {done}", step=done)

    return Scenario(
        name="maxsteps_exhaustion",
        group="agent",
        description="Agent never finishes: step extensions, what-remains summaries and forced final answer.",
        setup=lambda: make_agent(
            "stuck", tools=[make_tool("lookup")], maxsteps=3, max_tool_rounds=2, max_step_extensions=2, extend_steps_by=2
        ).execution,
        policies={"stuck": policy},
        check=expect_final("Summary:"),
    )


def _construct() -> Scenario:
    def setup() -> Run:
        def build() -> Any:
            subs = [make_agent(f"sub_{i}", tools=[make_tool("lookup")]) for i in range(2)]
            return make_agent("built", tools=[make_tool(f"t{i}") for i in range(8)], subagents=subs)

        return build

    return Scenario(
        name="construct_agent",
        group="overhead",
        description="Construct an agent with 8 tools and 2 subagents (no LLM calls).",
        setup=setup,
        policies={},
    )


def _transport_floor() -> Scenario:
    def setup() -> Run:
        import httpx

        client = httpx.Client()
        body = {"model": "gpt-4o", "input": [{"type": "message", "role": "user", "content": "hi"}], "tools": [
            {"type": "function", "name": "agent_end", "parameters": {}}
        ]}

        def run() -> Any:
            return client.post(
                "https://api.openai.com/v1/responses", json=body, headers={"Authorization": "Bearer bench-floor"}
            ).json()

        return run

    return Scenario(
        name="transport_floor",
        group="overhead",
        description="Bare httpx POST through the fake transport: the floor for one LLM round trip.",
        setup=setup,
        policies={"floor": lambda view: end(FINAL)},
    )


def all_scenarios() -> list[Scenario]:
    scenarios = [_transport_floor(), _construct()]
    scenarios += [_simple_end(p) for p in PROVIDER_MODELS]
    scenarios += [_tool_loop(p) for p in ("openai_responses", "openai_chat", "anthropic", "gemini", "codex")]
    scenarios += [
        _parallel_tools(),
        _async_agent(),
        _validated(),
        _step_exhaustion(),
        _staged(),
        _long_conversation(),
        _large_payload(),
        _subagents(),
        _next_agent(),
        _workflow_chain(use_async=False),
        _workflow_chain(use_async=True),
        _workflow_fanout(),
        _checkpointed(),
        _hitl_resume(),
    ]
    return scenarios
