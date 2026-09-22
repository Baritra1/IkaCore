"""Ablation study: knock out one IkaCore component at a time and measure the saving.

    python -m benchmarks.ablate                      # default scenario set, all ablations
    python -m benchmarks.ablate -k tool_loop -a console_output -a tool_batch_threads
    python -m benchmarks.ablate --list

Each ablation is a temporary patch applied only while its variant runs. IkaCore's
source is never modified. Variants run interleaved with the baseline, in rounds,
so machine drift affects all of them equally. Every variant is also run once in
recording mode and compared with golden.json:

  neutral  requests and output identical to golden: the component can be made
           cheaper without any observable change
  changed  same LLM calls but a different fingerprint: timing is valid, but the
           knockout alters observable behavior
  broken   error or a different number of LLM calls: timing is not comparable

Savings are per scenario and are not additive. The ``all`` row applies every
neutral-by-construction ablation at once and shows what would be left.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import statistics
import sys
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
try:
    import IkaCore  # noqa: F401
except ModuleNotFoundError:  # not pip-installed: benchmark the working tree
    sys.path.insert(0, str(REPO_ROOT / "src"))

import httpx  # noqa: E402

from benchmarks import fake_provider, harness  # noqa: E402
from benchmarks.scenarios import Scenario, all_scenarios  # noqa: E402

DEFAULT_SCENARIOS = [
    "single_turn[openai_responses]",
    "tool_loop_10[openai_responses]",
    "tool_loop_10[anthropic]",
    "parallel_tools_16",
    "async_tool_loop_10",
    "staged_3x3",
    "long_conversation_40",
    "large_payload",
    "subagents_3",
    "workflow_fanout_3x4[async]",
    "checkpointed_10",
    "hitl_interrupt_resume",
]


# --------------------------------------------------------------------------
# Knockouts
# --------------------------------------------------------------------------


def _noop(*_args: Any, **_kwargs: Any) -> None:
    return None


class InlineExecutor:
    """Executor stand-in that runs work immediately in the calling thread."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self._shutdown = False

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 - mirrors what a worker thread would capture
            future.set_exception(exc)
        return future

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        self._shutdown = True


@contextlib.contextmanager
def _console_output() -> Iterator[None]:
    from IkaCore.cli_output import CLIOutput

    names = ("emit", "agent_init", "agent_response", "summarization", "workflow_status", "tool_call", "tool_result")
    with contextlib.ExitStack() as stack:
        for name in names:
            stack.enter_context(mock.patch.object(CLIOutput, name, _noop))
        yield


@contextlib.contextmanager
def _tool_batch_threads() -> Iterator[None]:
    from IkaModel.chat_interface import tool_execution_sync

    with mock.patch.object(tool_execution_sync, "ThreadPoolExecutor", InlineExecutor):
        yield


@contextlib.contextmanager
def _tool_timeout_pool() -> Iterator[None]:
    from IkaModel.chat_interface import tool_execution_sync

    inline = InlineExecutor()
    with mock.patch.object(tool_execution_sync, "_ensure_tool_executor_pool", lambda: inline):
        yield


@contextlib.contextmanager
def _httpx_pipeline() -> Iterator[None]:
    """Skip httpx's client machinery (URL merge, auth, redirects, cookies, hooks) but keep request encoding."""

    def respond(url: Any, headers: Any, body: Any) -> httpx.Response:
        active = fake_provider._ACTIVE
        assert active is not None
        return active.respond(httpx.Request("POST", url, headers=headers, json=body))

    def post(self: Any, url: Any, *, headers: Any = None, json: Any = None, **_: Any) -> httpx.Response:
        return respond(url, headers, json)

    async def apost(self: Any, url: Any, *, headers: Any = None, json: Any = None, **_: Any) -> httpx.Response:
        return respond(url, headers, json)

    with mock.patch.object(httpx.Client, "post", post), mock.patch.object(httpx.AsyncClient, "post", apost):
        yield


@contextlib.contextmanager
def _event_loop_per_round() -> Iterator[None]:
    """Reuse one event loop per thread instead of asyncio.run() per chat round."""
    from IkaCore import agent_chat_support

    local = threading.local()
    loops: list[asyncio.AbstractEventLoop] = []

    def run(coro: Any) -> Any:
        loop = getattr(local, "loop", None)
        if loop is None:
            loop = local.loop = asyncio.new_event_loop()
            loops.append(loop)
        return loop.run_until_complete(coro)

    try:
        with mock.patch.object(agent_chat_support, "asyncio", SimpleNamespace(run=run)):
            yield
    finally:
        for loop in loops:
            loop.close()


@contextlib.contextmanager
def _checkpoint_sqlite_io() -> Iterator[None]:
    """Keep checkpoint JSON serialization, replace SQLite connect/commit with a dict."""
    from IkaCore.checkpoint import CheckpointPersistenceMixin, CheckpointSchemaMixin

    store: dict[tuple[str, str], str] = {}

    def save(self: Any, scope: str, payload: dict[str, Any], uid: Optional[str] = None) -> str:
        import uuid

        checkpoint_uid = uid or str(uuid.uuid4())
        store[(self.db_path, checkpoint_uid)] = json.dumps(payload)
        return checkpoint_uid

    def load(self: Any, uid: str) -> Optional[dict[str, Any]]:
        raw = store.get((self.db_path, uid))
        return None if raw is None else json.loads(raw)

    def delete(self: Any, uid: str) -> None:
        store.pop((self.db_path, uid), None)

    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(CheckpointSchemaMixin, "_init_db", _noop))
        stack.enter_context(mock.patch.object(CheckpointPersistenceMixin, "save_checkpoint", save))
        stack.enter_context(mock.patch.object(CheckpointPersistenceMixin, "load_checkpoint", load))
        stack.enter_context(mock.patch.object(CheckpointPersistenceMixin, "delete_checkpoint", delete))
        yield


@contextlib.contextmanager
def _logging_calls() -> Iterator[None]:
    with contextlib.ExitStack() as stack:
        for name in ("debug", "info", "warning", "error", "exception", "critical"):
            stack.enter_context(mock.patch.object(logging.Logger, name, _noop))
        yield


@contextlib.contextmanager
def _token_budget_scan() -> Iterator[None]:
    from IkaModel.chat_interface import chat_request

    with mock.patch.object(chat_request, "get_total_tokens", lambda _history: 0):
        yield


@contextlib.contextmanager
def _control() -> Iterator[None]:
    """Patches nothing: its 'saving' is the measurement noise floor."""
    yield


@dataclass(frozen=True)
class Ablation:
    name: str
    component: str
    description: str
    apply: Callable[[], contextlib.AbstractContextManager[None]]


ABLATIONS = [
    Ablation("console_output", "IkaCore/cli_output.py",
             "Box rendering, wrapping and colouring for every console event.", _console_output),
    Ablation("tool_batch_threads", "tool_execution_sync._execute_parallel_tool_plan",
             "New ThreadPoolExecutor (fresh OS threads) for every tool batch.", _tool_batch_threads),
    Ablation("tool_timeout_pool", "tool_execution_sync.execute_tool",
             "Second hop through the shared 8-worker pool, used only to enforce the tool timeout.", _tool_timeout_pool),
    Ablation("httpx_pipeline", "httpx.Client.post / AsyncClient.post",
             "httpx client machinery (URL merge, auth, redirects, cookies, hooks); keeps JSON encoding.", _httpx_pipeline),
    Ablation("event_loop_per_round", "agent_chat_support: asyncio.run per chat round",
             "New event loop per async chat round (async agents only).", _event_loop_per_round),
    Ablation("checkpoint_sqlite_io", "IkaCore/checkpoint.py",
             "SQLite connect + commit per checkpoint save (JSON serialization kept).", _checkpoint_sqlite_io),
    Ablation("logging_calls", "logging.Logger.*",
             "Logger calls that pass the level check, plus the stderr fallback handler.", _logging_calls),
    Ablation("token_budget_scan", "chat_request.get_total_tokens",
             "Full-history token recount before every chat call.", _token_budget_scan),
]


@contextlib.contextmanager
def _all_ablations() -> Iterator[None]:
    with contextlib.ExitStack() as stack:
        for ablation in ABLATIONS:
            stack.enter_context(ablation.apply())
        yield


ALL = Ablation("all", "(combined)", "Every ablation above at once: the residual core cost.", _all_ablations)
CONTROL = Ablation("control", "(nothing)", "No-op variant; any 'saving' it shows is noise.", _control)


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------


@dataclass
class VariantResult:
    status: str  # neutral | changed | broken
    net_ms: list[float]
    detail: str = ""

    @property
    def median(self) -> float:
        return statistics.median(self.net_ms) if self.net_ms else float("nan")


def _verify(scenario: Scenario, fake: fake_provider.FakeLLM, golden: Optional[harness.Fingerprint]) -> tuple[str, str, int]:
    fake.record = True
    try:
        out, _, _ = harness._run_once(scenario, fake)
        scenario.check(out)
        fp, _ = harness.fingerprint(fake, out, scenario.order_insensitive)
    except Exception as exc:  # noqa: BLE001 - report, don't abort the study
        return "broken", f"{type(exc).__name__}: {exc}", -1
    finally:
        fake.record = False
    if golden is None:
        return "changed", "no golden fingerprint", fp.calls
    if fp.digest == golden.digest:
        return "neutral", "", fp.calls
    if fp.calls != golden.calls:
        return "broken", f"LLM calls {golden.calls} -> {fp.calls}", fp.calls
    return "changed", "; ".join(harness.describe_mismatch(golden, fp)), fp.calls


def _variant_context(ablation: Optional[Ablation]) -> contextlib.AbstractContextManager[None]:
    return ablation.apply() if ablation is not None else contextlib.nullcontext()


def study_scenario(
    scenario: Scenario, ablations: list[Ablation], rounds: int, budget_s: float, golden: Optional[harness.Fingerprint]
) -> dict[str, VariantResult]:
    fake = harness._fake_for(scenario)
    fake_provider.activate(fake)
    variants: list[Optional[Ablation]] = [None, *ablations]
    results: dict[str, VariantResult] = {}
    try:
        for variant in variants:
            key = variant.name if variant else "baseline"
            with _variant_context(variant):
                status, detail, _ = _verify(scenario, fake, golden)
            results[key] = VariantResult(status, [], detail)

        # Calibrate iterations per round from the baseline so each round costs ~budget/rounds per variant.
        _, wall, _ = harness._run_once(scenario, fake)
        per_round = max(1, min(200, int((budget_s / rounds) / max(wall, 1e-4))))
        for _ in range(rounds):
            for variant in variants:
                key = variant.name if variant else "baseline"
                if results[key].status == "broken":
                    continue
                with _variant_context(variant):
                    for _ in range(per_round):
                        _, wall, fake_s = harness._run_once(scenario, fake)
                        results[key].net_ms.append((wall - fake_s) * 1000.0)
    finally:
        fake_provider.activate(None)
    return results


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


_LABELS = {
    "console_output": "console", "tool_batch_threads": "batch-thr", "tool_timeout_pool": "tmout-pool",
    "httpx_pipeline": "httpx", "event_loop_per_round": "evloop", "checkpoint_sqlite_io": "ckpt-io",
    "logging_calls": "logging", "token_budget_scan": "tok-scan", "all": "ALL", "control": "control",
}


def _cell(base: VariantResult, variant: VariantResult) -> tuple[str, Optional[float]]:
    if variant.status == "broken":
        return "broken", None
    saved = 100.0 * (base.median - variant.median) / base.median
    return f"{saved:.1f}%{'' if variant.status == 'neutral' else '*'}", saved


def report(study: dict[str, dict[str, VariantResult]], ablations: list[Ablation]) -> None:
    labels = [_LABELS.get(a.name, a.name[:10]) for a in ablations]
    print("\nshare of net runtime removed by each knockout (* = observable behavior changed)\n")
    print(f"{'scenario':<31}{'base ms':>9}" + "".join(f"{label:>11}" for label in labels))
    print("-" * (40 + 11 * len(labels)))
    columns: list[list[float]] = [[] for _ in ablations]
    for name, variants in study.items():
        base = variants["baseline"]
        row = f"{name[:30]:<31}{base.median:>9.2f}"
        for idx, ablation in enumerate(ablations):
            text, saved = _cell(base, variants[ablation.name])
            if saved is not None:
                columns[idx].append(saved)
            row += f"{text:>11}"
        print(row)
    print("-" * (40 + 11 * len(labels)))
    print(f"{'mean':<40}" + "".join(f"{statistics.mean(c):>10.1f}%" if c else f"{'n/a':>11}" for c in columns))
    print()
    for ablation in ablations:
        for name, variants in study.items():
            v = variants[ablation.name]
            if v.status != "neutral":
                print(f"  {ablation.name} on {name}: {v.status} - {v.detail[:160]}")


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.ablate", description=__doc__.split("\n\n")[0])
    parser.add_argument("-k", "--filter", action="append", default=[], help="scenario name substring")
    parser.add_argument("-a", "--ablation", action="append", default=[], help="ablation name (repeatable)")
    parser.add_argument("--rounds", type=int, default=5, help="interleaved rounds per scenario")
    parser.add_argument("--budget", type=float, default=0.6, help="seconds per variant per scenario")
    parser.add_argument("--json", type=Path, help="write raw results")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    catalogue = {a.name: a for a in [CONTROL, *ABLATIONS, ALL]}
    if args.list:
        for a in catalogue.values():
            print(f"{a.name:<22} {a.component:<48} {a.description}")
        return 0
    unknown = [a for a in args.ablation if a not in catalogue]
    if unknown:
        print(f"unknown ablation(s): {unknown}; use --list")
        return 2
    chosen = [catalogue[a] for a in args.ablation if a != "control"] if args.ablation else [*ABLATIONS, ALL]
    ablations = [CONTROL, *chosen]
    scenarios = {s.name: s for s in all_scenarios()}
    selected = [scenarios[n] for n in DEFAULT_SCENARIOS] if not args.filter else [
        s for s in scenarios.values() if any(f in s.name for f in args.filter)
    ]

    fake_provider.install()
    golden = harness.load_golden()
    started = time.perf_counter()
    study: dict[str, dict[str, VariantResult]] = {}
    with harness.sandbox_cwd():
        for scenario in selected:
            print(f"  ablating {scenario.name} ...", flush=True)
            with harness.quiet():
                study[scenario.name] = study_scenario(
                    scenario, ablations, args.rounds, args.budget, golden.get(scenario.name)
                )
    report(study, ablations)
    print(f"elapsed: {time.perf_counter() - started:.1f} s")
    if args.json:
        payload = {
            name: {k: {"status": v.status, "median_ms": v.median, "detail": v.detail} for k, v in variants.items()}
            for name, variants in study.items()
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
