"""Live A/B for workflow context passing on a longer-horizon task.

    python -m benchmarks.live_workflow_ab --keys PATH [--providers anthropic,deepseek] [--reps 2] [--yes]
    python -m benchmarks.live_workflow_ab --dry-run      # full harness against the offline fake provider

Arm A summarises every edge's upstream context with an LLM call (IkaCore's old
behavior, ``summarize_context_above_tokens=None``); arm B passes context under
1000 estimated tokens through verbatim (the default then; now 4000). Both run the same
6-node "regional audit" workflow, async:

              ┌─► analyst_north ┐  each: list the region's accounts, look up every balance,
    planner ──┼─► analyst_south ┼─► auditor ─► converter ─► writer   report the exact total
              └─► analyst_west  ┘                             ▲
                        └─────────────────────────────────────┘  (writer also reads the analysts)

The writer's final answer is scored on six exact facts (three regional totals,
the top region, its risk score, the EUR conversion), so information lost or
mangled on the way through summaries shows up as a lower score. Reported per
arm: score, model calls (and how many were summaries), input/output tokens and
wall time. Keys are read from a dotenv file and never printed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import statistics
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import IkaCore  # noqa: F401
except ModuleNotFoundError:  # not pip-installed: test the working tree
    sys.path.insert(0, str(REPO_ROOT / "src"))

import httpx  # noqa: E402

from benchmarks.live_ab import PROVIDERS, load_keys  # noqa: E402
from IkaCore import IkaBaseAgent, IkaTools, IkaWorkflow, WorkflowEdge, WorkflowNode  # noqa: E402

REGIONS = {
    "north": {"N-11": 1204, "N-27": 889, "N-35": 3051, "N-48": 476},
    "south": {"S-02": 2210, "S-19": 1733, "S-40": 918, "S-63": 2502},
    "west": {"W-07": 640, "W-21": 1985, "W-33": 377, "W-58": 1129},
}
RISK = {"north": 37, "south": 58, "west": 21}
EUR_RATE = 0.92
TOTALS = {region: sum(accounts.values()) for region, accounts in REGIONS.items()}
TOP = max(TOTALS, key=TOTALS.__getitem__)
EUR = round(TOTALS[TOP] * EUR_RATE, 2)
FACTS = {
    **{f"{region} total": str(total) for region, total in TOTALS.items()},
    "top region": TOP,
    "risk score": str(RISK[TOP]),
    "EUR amount": f"{EUR:.2f}",
}
ARMS = {"A_always_summarise": None, "B_threshold_1000": 1000}


# --------------------------------------------------------------------------
# Tools and agents
# --------------------------------------------------------------------------


def _tool(name: str, description: str, params: dict[str, str], fn: Any) -> IkaTools:
    return IkaTools(
        name=name,
        description=description,
        parameters={p: {"type": t, "description": p, "required": True} for p, t in params.items()},
        execute_function=fn,
    )


def _tools() -> dict[str, IkaTools]:
    def accounts(args: dict[str, Any]) -> Any:
        return {"region": args.get("region"), "accounts": sorted(REGIONS.get(str(args.get("region", "")).lower(), {}))}

    def balance(args: dict[str, Any]) -> Any:
        acc = str(args.get("account", ""))
        found = next((b[acc] for b in REGIONS.values() if acc in b), None)
        return {"account": acc, "balance": found if found is not None else "unknown account"}

    def risk(args: dict[str, Any]) -> Any:
        region = str(args.get("region", "")).lower()
        return {"region": region, "risk_score": RISK.get(region, "unknown region")}

    def convert(args: dict[str, Any]) -> Any:
        amount = float(args.get("amount", 0) or 0)
        return {"amount": amount, "currency": "EUR", "converted": round(amount * EUR_RATE, 2)}

    return {
        "accounts": _tool("get_region_accounts", "List the account ids in a region.", {"region": "string"}, accounts),
        "balance": _tool("get_balance", "Return the balance of one account.", {"account": "string"}, balance),
        "risk": _tool("get_risk_score", "Return the risk score of a region.", {"region": "string"}, risk),
        "convert": _tool("convert_currency", "Convert an amount to a currency.",
                         {"amount": "number", "currency": "string"}, convert),
    }


def _prompts() -> dict[str, tuple[str, list[str]]]:
    analyst = ("You are the {r} analyst. Call get_region_accounts with region='{r}', then call get_balance "
               "for each account, one account per turn. Finish with agent_end reporting every account's balance "
               "and the exact region total, written as 'TOTAL {r} = <number>'.")
    prompts = {"planner": ("You coordinate a quarterly audit of the north, south and west regions. Finish with "
                           "agent_end giving each regional analyst a one-line instruction.", [])}
    prompts.update({f"analyst_{r}": (analyst.format(r=r), ["accounts", "balance"]) for r in REGIONS})
    prompts["auditor"] = ("You receive regional reports. Identify the region with the largest total, call "
                          "get_risk_score for that region, then finish with agent_end stating: the top region, "
                          "its exact total, and its risk score.", ["risk"])
    prompts["converter"] = ("From the auditor's report, take the top region's exact total and call "
                            "convert_currency with that amount and currency 'EUR'. Finish with agent_end stating "
                            "the top region and the exact converted EUR amount.", ["convert"])
    prompts["writer"] = ("Write the final audit summary from all upstream reports. It must state each region's "
                         "exact total, the top region, its risk score, and the exact EUR amount. Finish with "
                         "agent_end containing the summary.", [])
    return prompts


def build_workflow(
    provider: str, api_key: str, threshold: Optional[int], model: Optional[str] = None, fake_keys: bool = False
) -> IkaWorkflow:
    spec = PROVIDERS[provider]
    tools = _tools()
    nodes = []
    for name, (prompt, tool_keys) in _prompts().items():
        agent = IkaBaseAgent(
            name=name, description=f"{name} of the regional audit", prompt=prompt,
            tools=[tools[k] for k in tool_keys], model_id=model or spec["model"],
            api_key=f"bench-{name}" if fake_keys else api_key,  # the offline fake routes by key
            max_tokens=1024, temperature=0.0, maxsteps=20,
        )
        nodes.append(WorkflowNode(name=name, agent=agent))
    analysts = [f"analyst_{r}" for r in REGIONS]
    edges = [WorkflowEdge("planner", a) for a in analysts]  # a workflow runs what is reachable from start_node
    edges += [WorkflowEdge(a, "auditor") for a in analysts] + [WorkflowEdge(a, "writer") for a in analysts]
    edges += [WorkflowEdge("auditor", "converter"), WorkflowEdge("auditor", "writer"), WorkflowEdge("converter", "writer")]
    return IkaWorkflow(name="regional_audit", description="longer-horizon audit", nodes=nodes, edges=edges,
                       start_node="planner", summarize_context_above_tokens=threshold)


# --------------------------------------------------------------------------
# Measuring real traffic without changing it
# --------------------------------------------------------------------------


@dataclass
class Traffic:
    calls: int = 0
    summary_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


@contextlib.contextmanager
def count_traffic(traffic: Traffic) -> Iterator[None]:
    """Wrap httpx's transport to count requests and read usage from responses; requests are unchanged."""
    original = httpx.HTTPTransport.handle_request

    def handle(self: Any, request: httpx.Request) -> httpx.Response:
        response = original(self, request)
        response.read()
        try:
            body = json.loads(request.content or b"{}")
            usage = json.loads(response.content or b"{}").get("usage") or {}
        except ValueError:
            body, usage = {}, {}
        with traffic.lock:
            traffic.calls += 1
            traffic.summary_calls += 0 if body.get("tools") else 1
            traffic.input_tokens += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            traffic.output_tokens += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
        return response

    httpx.HTTPTransport.handle_request = handle  # type: ignore[method-assign]
    try:
        yield
    finally:
        httpx.HTTPTransport.handle_request = original  # type: ignore[method-assign]


@dataclass
class RunResult:
    score: int
    missing: list[str]
    traffic: Traffic
    seconds: float
    error: str = ""


def run_once(
    provider: str, api_key: str, threshold: Optional[int], model: Optional[str] = None, fake_keys: bool = False
) -> RunResult:
    traffic = Traffic()
    start = time.perf_counter()
    try:
        workflow = build_workflow(provider, api_key, threshold, model, fake_keys)
        with count_traffic(traffic), contextlib.redirect_stdout(io.StringIO()):
            results = workflow.run(initial_context="Audit Q3 balances across all regions.", use_async=True)
        final = results["writer"].final if "writer" in results else ""
    except Exception as exc:  # noqa: BLE001 - an API failure is a result to report
        return RunResult(0, list(FACTS), traffic, time.perf_counter() - start, f"{type(exc).__name__}: {str(exc)[:200]}")
    normalized = final.replace(",", "")
    missing = [label for label, value in FACTS.items() if value.lower() not in normalized.lower()]
    return RunResult(len(FACTS) - len(missing), missing, traffic, time.perf_counter() - start)


# --------------------------------------------------------------------------
# Offline dry run: the same harness against the fake provider
# --------------------------------------------------------------------------


def dry_run() -> int:
    from benchmarks import fake_provider
    from benchmarks.fake_provider import Call, FakeLLM, RequestView, Turn, end

    def analyst(region: str) -> Any:
        accounts = sorted(REGIONS[region])

        def policy(view: RequestView) -> Turn:
            if f"You are the {region} analyst" not in view.raw:  # the agent must see its own task
                return end("I cannot complete this without knowing which region to audit.")
            if view.count("get_region_accounts") == 0:
                return Turn(calls=(Call("get_region_accounts", {"region": region}),))
            done = view.count("get_balance")
            if done < len(accounts):
                return Turn(calls=(Call("get_balance", {"account": accounts[done]}),))
            lines = ", ".join(f"{a}={REGIONS[region][a]}" for a in accounts)
            return end(f"{lines}. TOTAL {region} = {TOTALS[region]}")
        return policy

    policies = {f"analyst_{r}": analyst(r) for r in REGIONS}
    policies["auditor"] = lambda v: (
        end("I cannot complete this: no task.") if "Identify the region with the largest total" not in v.raw
        else Turn(calls=(Call("get_risk_score", {"region": TOP}),)) if not v.count("get_risk_score")
        else end(f"Top region {TOP}, total {TOTALS[TOP]}, risk {RISK[TOP]}."))
    policies["converter"] = lambda v: (Turn(calls=(Call("convert_currency", {"amount": TOTALS[TOP], "currency": "EUR"}),))
                                       if not v.count("convert_currency") else end(f"{TOP}: EUR {EUR:.2f}"))

    def writer(view: RequestView) -> Turn:  # "reads" its context: only facts that reached it can be reported
        if "Write the final audit summary" not in view.raw:
            return end("I cannot complete this: no task.")
        present = [value for value in FACTS.values() if value in view.raw]
        return end("Audit summary: " + "; ".join(present))

    policies["writer"] = writer
    policies["planner"] = lambda v: end("north, south and west analysts: audit your region.")
    fake_provider.install()
    for arm, threshold in ARMS.items():
        fake = FakeLLM(policies=policies)
        fake_provider.activate(fake)
        result = run_once("anthropic", "unused", threshold, fake_keys=True)
        print(f"{arm:<20} score {result.score}/6  model calls {fake.calls} (summaries {fake.toolless_calls})  "
              f"missing {result.missing or '-'} {result.error}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.live_workflow_ab", description=__doc__.split("\n\n")[0])
    parser.add_argument("--keys", type=Path)
    parser.add_argument("--providers", default="anthropic,deepseek")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--max-input-tokens", type=int, default=400_000)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="run the harness against the offline fake provider")
    args = parser.parse_args(argv)
    if args.dry_run:
        return dry_run()

    keys = load_keys(args.keys)
    active = [p for p in args.providers.split(",") if keys.get(PROVIDERS[p]["env"])]
    runs = len(active) * len(ARMS) * args.reps
    print(f"providers: {active}; plan: {len(ARMS)} arms x {args.reps} reps x {len(active)} providers = {runs} workflow runs")
    if not args.yes or not active:
        print("dry run: pass --yes to call the providers")
        return 0
    spent = 0
    rows: dict[tuple[str, str], list[RunResult]] = {}
    for provider in active:
        for rep in range(args.reps):
            for arm, threshold in ARMS.items():
                if spent >= args.max_input_tokens:
                    print(f"STOPPING: input-token cap reached ({spent:,})")
                    break
                r = run_once(provider, keys[PROVIDERS[provider]["env"]], threshold)
                for secret in keys.values():
                    r.error = r.error.replace(secret, "<redacted>")
                spent += r.traffic.input_tokens
                rows.setdefault((provider, arm), []).append(r)
                print(f"  {provider:<9} rep {rep} {arm:<20} score {r.score}/6 calls {r.traffic.calls} "
                      f"(summ {r.traffic.summary_calls}) in {r.traffic.input_tokens:,} out {r.traffic.output_tokens:,} "
                      f"{r.seconds:5.1f}s missing {r.missing or '-'} {r.error}")
    print(f"\n{'provider':<10}{'arm':<22}{'score':>8}{'calls':>7}{'summ':>6}{'in tok':>9}{'out tok':>9}{'wall s':>8}")
    for (provider, arm), rs in rows.items():
        print(f"{provider:<10}{arm:<22}{statistics.mean(r.score for r in rs):>6.1f}/6"
              f"{statistics.mean(r.traffic.calls for r in rs):>7.1f}{statistics.mean(r.traffic.summary_calls for r in rs):>6.1f}"
              f"{statistics.mean(r.traffic.input_tokens for r in rs):>9,.0f}{statistics.mean(r.traffic.output_tokens for r in rs):>9,.0f}"
              f"{statistics.mean(r.seconds for r in rs):>8.1f}")
    print(f"\ntotal input tokens: {spent:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
