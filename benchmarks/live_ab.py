"""Live quality A/B for the history-replay dedupe (Anthropic, Gemini, DeepSeek).

    python -m benchmarks.live_ab --keys PATH [--providers anthropic,gemini] [--reps 3] [--yes]

Arm A replays every persisted history entry (the pre-fix behavior, reproduced by
patching the replay filter); arm B is the current code. Both arms run the same
tasks with real models, interleaved, at temperature 0. Every task has a
deterministic, checkable answer and needs several dependent tool rounds, so
duplicated history would show up if it helped or hurt:

  ledger   four sequential balance lookups, then report the exact total
  chain    follow five dependent "next hop" lookups, then report the final city
  staged   stage 1 fetches a code with a tool; stage 2 has no tool and must
           recall that code from earlier-stage context

Keys are read from a dotenv-style file (ANTHROPIC_API_KEY=..., GEMINI_API_KEY=...,
DEEPSEEK_API_KEY=...) or the environment. They are never printed. Without --yes
the script only shows the plan and an upper-bound token estimate.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import IkaCore  # noqa: F401
except ModuleNotFoundError:  # not pip-installed: test the working tree
    sys.path.insert(0, str(REPO_ROOT / "src"))

import IkaModel.history_replay as _history_replay  # noqa: E402
from IkaCore import IkaBaseAgent, IkaStage, IkaTools  # noqa: E402

PROVIDERS = {
    "anthropic": {"env": "ANTHROPIC_API_KEY", "model": "claude-haiku-4-5"},
    "gemini": {"env": "GEMINI_API_KEY", "model": "gemini-3.6-flash"},
    "deepseek": {"env": "DEEPSEEK_API_KEY", "model": "deepseek-chat"},
}

# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------


def _replay_all(message_history: dict[str, Any], messages: list[Any]) -> list[dict[str, Any]]:
    return [e for e in (message_history.get("messages") or {}).values() if isinstance(e, dict)]


@contextlib.contextmanager
def arm_a_old() -> Iterator[None]:
    with mock.patch.object(_history_replay, "history_entries_to_replay", _replay_all):
        yield


ARMS: dict[str, Callable[[], contextlib.AbstractContextManager[None]]] = {
    "A_old": arm_a_old,
    "B_new": contextlib.nullcontext,
}

# --------------------------------------------------------------------------
# Tasks
# --------------------------------------------------------------------------

BALANCES = {"ACC-17": 1204, "ACC-42": 889, "ACC-63": 3051, "ACC-88": 476}
HOPS = {"Lisbon": "Oslo", "Oslo": "Quito", "Quito": "Hanoi", "Hanoi": "Perth", "Perth": "Tunis"}
SECRET = "KESTREL-7Q4"


def _tool(name: str, description: str, param: str, fn: Callable[[str], Any]) -> IkaTools:
    return IkaTools(
        name=name,
        description=description,
        parameters={param: {"type": "string", "description": f"The {param}", "required": True}},
        execute_function=lambda args: fn(str(args.get(param, ""))),
    )


def _ledger(common: dict[str, Any]) -> IkaBaseAgent:
    tool = _tool("get_balance", "Return the balance of one account.", "account",
                 lambda acc: {"account": acc, "balance": BALANCES.get(acc, "unknown account")})
    return IkaBaseAgent(
        name="ledger", description="Balance auditor", tools=[tool], **common,
        prompt=("Look up the balances of accounts ACC-17, ACC-42, ACC-63 and ACC-88. Call get_balance for ONE "
                "account per turn, sequentially. Then call agent_end with the exact integer total of all four."),
    )


def _chain(common: dict[str, Any]) -> IkaBaseAgent:
    tool = _tool("next_hop", "Return the next city after the given city.", "city",
                 lambda city: {"city": city, "next": HOPS.get(city, "none")})
    return IkaBaseAgent(
        name="chain", description="Route follower", tools=[tool], **common,
        prompt=("Start at Lisbon. Call next_hop on the current city to learn the next one, one call per turn. "
                "Do this exactly five times, then call agent_end with only the name of the final city."),
    )


def _staged(common: dict[str, Any]) -> IkaBaseAgent:
    tool = _tool("read_vault", "Read the access code stored in the vault.", "vault", lambda _v: {"code": SECRET})
    stages = [
        IkaStage("Retrieve", "Call read_vault once with vault='main' to obtain the access code, then call stage_end.",
                 [tool]),
        IkaStage("Report", "Report the access code you retrieved in the previous stage by calling agent_end with "
                           "only the code. You have no tools in this stage besides agent_end.", []),
    ]
    return IkaBaseAgent(name="staged", description="Vault reader", prompt="Retrieve and report the vault code.",
                        Stages=stages, **common)


def _encode(code: str) -> str:
    return "-".join(reversed(code.split("-")))


def _pipeline(common: dict[str, Any]) -> IkaBaseAgent:
    """A middle stage that must use its own tool: skipping it gives a wrong answer."""
    read = _tool("read_vault", "Read the access code stored in the vault.", "vault", lambda _v: {"code": SECRET})
    encode = _tool("encode_code", "Encode an access code for transmission.", "code", lambda c: {"encoded": _encode(c)})
    stages = [
        IkaStage("Retrieve", "Call read_vault once with vault='main' to obtain the access code, then call stage_end.",
                 [read]),
        IkaStage("Encode", "Call encode_code with the access code from the previous stage, then call stage_end.",
                 [encode]),
        IkaStage("Report", "Report the encoded code from the previous stage by calling agent_end with only that "
                           "encoded value.", []),
    ]
    return IkaBaseAgent(name="pipeline", description="Vault encoder", prompt="Retrieve, encode and report the code.",
                        Stages=stages, **common)


@dataclass(frozen=True)
class Task:
    name: str
    build: Callable[[dict[str, Any]], IkaBaseAgent]
    expected: str


TASKS = [
    Task("ledger", _ledger, str(sum(BALANCES.values()))),
    Task("chain", _chain, "Tunis"),
    Task("staged", _staged, SECRET),
    Task("pipeline", _pipeline, _encode(SECRET)),
]

# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


@dataclass
class Outcome:
    success: bool
    error: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    final: str = ""


@dataclass
class Cell:
    outcomes: list[Outcome] = field(default_factory=list)

    def summary(self) -> str:
        n = len(self.outcomes)
        ok = sum(o.success for o in self.outcomes)
        errs = sum(bool(o.error) for o in self.outcomes)
        tokens = statistics.median(o.input_tokens for o in self.outcomes) if n else 0
        secs = statistics.median(o.seconds for o in self.outcomes) if n else 0
        return f"{ok}/{n} ok, {errs} err, {tokens:>7,.0f} in-tok, {secs:5.1f}s"


def load_keys(path: Optional[Path]) -> dict[str, str]:
    keys: dict[str, str] = {}
    if path is not None:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                keys[name.strip()] = value.strip().strip('"').strip("'")
    for spec in PROVIDERS.values():
        if spec["env"] not in keys and os.environ.get(spec["env"]):
            keys[spec["env"]] = os.environ[spec["env"]]
    return keys


def run_once(task: Task, provider: str, api_key: str, model: str) -> Outcome:
    common = {"model_id": model, "api_key": api_key, "max_tokens": 1024, "temperature": 0.0, "maxsteps": 20}
    start = time.perf_counter()
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            agent = task.build(common)
            out = agent.execution()
        final = str(out.get("final_message", ""))
        usage = out.get("usage", {}) or {}
        return Outcome(
            success=task.expected.lower() in final.lower(),
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            seconds=time.perf_counter() - start,
            final=final[:200],
        )
    except Exception as exc:  # noqa: BLE001 - an API rejection is a result, not a crash
        return Outcome(success=False, error=f"{type(exc).__name__}: {str(exc)[:300]}", seconds=time.perf_counter() - start)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.live_ab", description=__doc__.split("\n\n")[0])
    parser.add_argument("--keys", type=Path, help="dotenv-style file with provider API keys")
    parser.add_argument("--providers", default="anthropic,gemini,deepseek")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--tasks", default=",".join(t.name for t in TASKS), help="comma-separated task names")
    parser.add_argument("--arms", default=",".join(ARMS), help="comma-separated arms to run (A_old,B_new)")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--yes", action="store_true", help="actually call the providers")
    parser.add_argument("--max-input-tokens", type=int, default=600_000,
                        help="hard stop once reported input tokens across all runs reach this")
    args = parser.parse_args(argv)

    keys = load_keys(args.keys)
    wanted = [p.strip() for p in args.providers.split(",") if p.strip()]
    active = [p for p in wanted if keys.get(PROVIDERS[p]["env"])]
    missing = [p for p in wanted if p not in active]
    tasks = [t for t in TASKS if t.name in {n.strip() for n in args.tasks.split(",")}]
    arms = {name: ARMS[name] for name in args.arms.split(",") if name in ARMS}
    runs = len(active) * len(tasks) * len(arms) * args.reps
    print(f"providers with keys: {active or 'none'}; missing: {missing or 'none'}")
    print(f"plan: {len(tasks)} tasks x {len(arms)} arms x {args.reps} reps x {len(active)} providers = {runs} agent runs "
          f"(upper bound ~{runs * 12_000:,} input tokens on small models)")
    if not args.yes or not active:
        print("dry run: pass --yes to call the providers")
        return 0

    results: dict[tuple[str, str, str], Cell] = {}
    spent_in = spent_out = 0
    for provider in active:
        spec = PROVIDERS[provider]
        for task in tasks:
            for rep in range(args.reps):
                for arm_name, arm in arms.items():  # interleaved so provider drift hits both arms
                    if spent_in >= args.max_input_tokens:
                        print(f"STOPPING: input-token cap reached ({spent_in:,} >= {args.max_input_tokens:,})")
                        return _report(results, active, args.json, spent_in, spent_out)
                    with arm():
                        outcome = run_once(task, provider, keys[spec["env"]], spec["model"])
                    for secret in keys.values():  # provider errors can echo URLs/headers carrying the key
                        outcome.error = outcome.error.replace(secret, "<redacted>")
                    results.setdefault((provider, task.name, arm_name), Cell()).outcomes.append(outcome)
                    spent_in += outcome.input_tokens
                    spent_out += outcome.output_tokens
                    flag = "ok " if outcome.success else ("ERR" if outcome.error else "bad")
                    print(f"  {provider:<9} {task.name:<7} rep {rep} {arm_name}: {flag} {outcome.error or outcome.final[:60]!r}")
    return _report(results, active, args.json, spent_in, spent_out)


def _report(
    results: dict[tuple[str, str, str], Cell], active: list[str], out: Optional[Path], spent_in: int, spent_out: int
) -> int:
    print(f"\n{'provider':<10}{'task':<9}{'A_old (replay everything)':<44}{'B_new (current code)':<44}")
    for provider in active:
        for task in TASKS:
            a, b = results.get((provider, task.name, "A_old")), results.get((provider, task.name, "B_new"))
            if a or b:
                cells = [c.summary() if c else "-" for c in (a, b)]
                print(f"{provider:<10}{task.name:<9}{cells[0]:<44}{cells[1]:<44}")
    print(f"\ntotal reported usage: {spent_in:,} input tokens, {spent_out:,} output tokens")
    if out:
        payload = {f"{p}/{t}/{a}": [o.__dict__ for o in cell.outcomes] for (p, t, a), cell in results.items()}
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
