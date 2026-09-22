"""Error-path benchmark: how long IkaCore waits, and how it ends, under provider faults.

    python -m benchmarks.faults [--json out.json] [--compare base.json]

Each scenario runs a small agent (one tool round, then agent_end) against the
fake provider while it injects a scripted failure (429s carrying each kind of
retry hint, 5xx, Anthropic 529, a context-length 400, permanent 4xx, timeouts,
connection errors). Sleeps are virtualised: ``time.sleep``/``asyncio.sleep``
record the requested delay and return immediately, so a run that would stall
for 20 s finishes in milliseconds while reporting the 20 s it would have cost.

Reported per scenario: outcome (recovered / failed with which error), provider
attempts, and the total virtual wait. ``expect`` states the correct outcome;
the wait is what we optimise.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import random
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
except ModuleNotFoundError:  # not pip-installed: benchmark the working tree
    sys.path.insert(0, str(REPO_ROOT / "src"))

from benchmarks import fake_provider, harness  # noqa: E402
from benchmarks.fake_provider import SUMMARY_TEXT, FakeLLM, Fault, RequestView  # noqa: E402
from benchmarks.scenarios import FINAL, end, loop_policy, make_agent, make_tool  # noqa: E402

# --------------------------------------------------------------------------
# Virtual clock
# --------------------------------------------------------------------------


@dataclass
class VirtualClock:
    sleeps: list[float] = field(default_factory=list)

    @property
    def total(self) -> float:
        return sum(self.sleeps)


@contextlib.contextmanager
def virtual_sleep(clock: VirtualClock) -> Iterator[None]:
    real_async_sleep = asyncio.sleep

    def fake_sleep(seconds: float) -> None:
        clock.sleeps.append(float(seconds))

    async def fake_async_sleep(seconds: float, result: Any = None) -> Any:
        clock.sleeps.append(float(seconds))
        await real_async_sleep(0)
        return result

    with mock.patch.object(time, "sleep", fake_sleep), mock.patch.object(asyncio, "sleep", fake_async_sleep):
        yield


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------


def once(fault: Fault, at: int = 0) -> Callable[[RequestView, int], Optional[Fault]]:
    return lambda view, index: fault if index == at else None


def always(fault: Fault) -> Callable[[RequestView, int], Optional[Fault]]:
    return lambda view, index: fault


def until_summarised(fault: Fault) -> Callable[[RequestView, int], Optional[Fault]]:
    """Context overflow persists for tool requests until the history carries a summary."""
    return lambda view, index: fault if view.tool_names and SUMMARY_TEXT not in view.raw else None


RATE_LIMIT = {"error": {"message": "Rate limit reached for requests", "type": "rate_limit_error"}}
CONTEXT = {"error": {"message": "This model's maximum context length is 128000 tokens. However, your messages "
                                "resulted in 130512 tokens.", "code": "context_length_exceeded"}}
GEMINI_RETRY = {"error": {"code": 429, "message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED", "details": [
    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "3s"}]}}


@dataclass(frozen=True)
class FaultCase:
    name: str
    provider: str
    plan: Callable[[RequestView, int], Optional[Fault]]
    expect: str  # "recover" or "fail"
    note: str = ""
    use_async: bool = False


CASES = [
    FaultCase("429 retry-after: 2", "openai_chat", once(Fault(429, RATE_LIMIT, (("retry-after", "2"),))), "recover",
              "server asks for 2 s"),
    FaultCase("429 retry-after: 0.8", "openai_chat", once(Fault(429, RATE_LIMIT, (("retry-after", "0.8"),))), "recover",
              "fractional hint"),
    FaultCase("429 retry-after-ms: 500", "openai_chat", once(Fault(429, RATE_LIMIT, (("retry-after-ms", "500"),))),
              "recover", "OpenAI millisecond hint"),
    FaultCase("429 x-ratelimit-reset: 1.5s", "openai_chat",
              once(Fault(429, RATE_LIMIT, (("x-ratelimit-reset-requests", "1.5s"),))), "recover", "OpenAI reset header"),
    FaultCase("429 gemini retryDelay 3s", "gemini", once(Fault(429, GEMINI_RETRY)), "recover", "hint in JSON body"),
    FaultCase("429 no hint", "openai_chat", once(Fault(429, RATE_LIMIT)), "recover"),
    FaultCase("500 once", "openai_chat", once(Fault(500, {"error": {"message": "server error"}})), "recover"),
    FaultCase("503 twice", "openai_chat",
              lambda v, i: Fault(503, {"error": {"message": "unavailable"}}) if i < 2 else None, "recover"),
    FaultCase("529 overloaded (anthropic)", "anthropic",
              once(Fault(529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})),
              "recover"),
    FaultCase("read timeout once", "openai_chat", once(Fault(exc="ReadTimeout")), "recover"),
    FaultCase("connect error once", "openai_chat", once(Fault(exc="ConnectError")), "recover"),
    FaultCase("400 context length", "openai_chat", until_summarised(Fault(400, CONTEXT)), "recover",
              "fails until history is summarised"),
    FaultCase("404 model not found", "openai_chat",
              always(Fault(404, {"error": {"message": "The model `gpt-9` does not exist", "code": "model_not_found"}})),
              "fail", "permanent"),
    FaultCase("401 invalid key", "openai_chat",
              always(Fault(401, {"error": {"message": "Incorrect API key provided", "code": "invalid_api_key"}})),
              "fail", "permanent"),
    FaultCase("400 bad request", "anthropic",
              always(Fault(400, {"type": "error", "error": {"type": "invalid_request_error",
                                                            "message": "tools.0.name: invalid"}})), "fail", "permanent"),
    FaultCase("500 forever", "openai_chat", always(Fault(500, {"error": {"message": "server error"}})), "fail",
              "gives up eventually"),
    FaultCase("429 no hint (async agent)", "openai_chat", once(Fault(429, RATE_LIMIT)), "recover", use_async=True),
    FaultCase("codex 429 retry-after: 2", "codex", once(Fault(429, RATE_LIMIT, (("retry-after", "2"),))), "recover"),
    FaultCase("codex 500 once", "codex", once(Fault(500, {"error": {"message": "server error"}})), "recover"),
    FaultCase("codex 404", "codex", always(Fault(404, {"error": {"message": "not found"}})), "fail", "permanent"),
]


@dataclass
class FaultResult:
    name: str
    expect: str
    outcome: str
    ok: bool
    attempts: int
    waited_s: float
    sleeps: list[float]


def run_case(case: FaultCase) -> FaultResult:
    fake = FakeLLM(policies={"faulty": loop_policy("lookup", 1, end(FINAL))}, faults=case.plan)
    fake_provider.activate(fake)
    clock = VirtualClock()
    random.seed(1234)  # jittered backoff becomes reproducible
    try:
        agent = make_agent("faulty", case.provider, tools=[make_tool("lookup")], use_async=case.use_async)
        with virtual_sleep(clock), harness.quiet():
            try:
                out = agent.execution()
                final = str(out.get("final_message", ""))
                outcome = "recovered" if FINAL in final else f"wrong answer: {final[:60]!r}"
            except Exception as exc:  # noqa: BLE001 - the error class is the result
                outcome = f"failed: {type(exc).__name__}"
    finally:
        fake_provider.activate(None)
    ok = outcome == "recovered" if case.expect == "recover" else outcome.startswith("failed")
    return FaultResult(case.name, case.expect, outcome, ok, fake.attempts.get("faulty", 0), clock.total, clock.sleeps)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.faults", description=__doc__.split("\n\n")[0])
    parser.add_argument("-k", "--filter", action="append", default=[])
    parser.add_argument("--json", type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args(argv)
    base = json.loads(args.compare.read_text(encoding="utf-8")) if args.compare else {}

    fake_provider.install()
    cases = [c for c in CASES if not args.filter or any(f in c.name for f in args.filter)]
    header = f"{'scenario':<30}{'expect':>8}  {'outcome':<30}{'attempts':>9}{'waited s':>10}"
    if base:
        header += f"{'before s':>10}"
    print(header)
    print("-" * len(header))
    results = []
    with harness.sandbox_cwd():
        for case in cases:
            r = run_case(case)
            results.append(r)
            mark = "" if r.ok else "  <-- WRONG OUTCOME"
            line = f"{r.name:<30}{r.expect:>8}  {r.outcome:<30}{r.attempts:>9}{r.waited_s:>10.1f}"
            if base and r.name in base:
                line += f"{base[r.name]['waited_s']:>10.1f}"
            print(line + mark)
    total = sum(r.waited_s for r in results)
    print(f"\ntotal virtual wait: {total:.1f} s across {len(results)} scenarios; "
          f"{sum(not r.ok for r in results)} wrong outcome(s)")
    if base:
        before = sum(base[r.name]["waited_s"] for r in results if r.name in base)
        print(f"before: {before:.1f} s")
    if args.json:
        args.json.write_text(json.dumps({r.name: r.__dict__ for r in results}, indent=2) + "\n", encoding="utf-8")
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
