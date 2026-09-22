"""Timing, behavior fingerprinting, profiling and reporting."""

from __future__ import annotations

import contextlib
import cProfile
import dataclasses
import gc
import hashlib
import io
import json
import os
import pstats
import re
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from benchmarks import fake_provider
from benchmarks.fake_provider import FakeLLM, normalize_text, recorded_by_agent
from benchmarks.scenarios import Scenario

GOLDEN_PATH = Path(__file__).with_name("golden.json")


class _NullWriter(io.TextIOBase):
    """Swallows IkaCore's console output while keeping its rendering cost in the measurement."""

    def write(self, s: str) -> int:
        return len(s)

    def flush(self) -> None:
        pass


@contextlib.contextmanager
def quiet() -> Iterator[None]:
    with contextlib.redirect_stdout(_NullWriter()), contextlib.redirect_stderr(_NullWriter()):
        yield


@contextlib.contextmanager
def sandbox_cwd() -> Iterator[Path]:
    """Run scenarios in a throwaway directory so stray files (logs, dbs) never touch the repo."""
    previous = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="ikabench-cwd-") as tmp:
        os.chdir(tmp)
        try:
            yield Path(tmp)
        finally:
            os.chdir(previous)


# --------------------------------------------------------------------------
# Behavior fingerprints
# --------------------------------------------------------------------------

_ADDR_RE = re.compile(r"0x[0-9a-fA-F]{6,}")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    return repr(value)


def canonical_output(out: Any) -> str:
    text = json.dumps(out, sort_keys=True, default=_json_default)
    return _ADDR_RE.sub("<addr>", normalize_text(text))


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Fingerprint:
    calls: int
    output: str
    requests: dict[str, list[str]]  # agent -> per-request body hashes, in that agent's order
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.digest:
            payload = json.dumps({"calls": self.calls, "output": self.output, "requests": self.requests}, sort_keys=True)
            self.digest = _sha(payload)

    def to_json(self) -> dict[str, Any]:
        return {"digest": self.digest, "calls": self.calls, "output": self.output, "requests": self.requests}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Fingerprint:
        return cls(calls=data["calls"], output=data["output"], requests=data["requests"], digest=data["digest"])


def _order_free(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(sorted(_TOKEN_RE.findall(value)))
    if isinstance(value, list):
        return [_order_free(v) for v in value]
    if isinstance(value, dict):
        return {k: _order_free(v) for k, v in value.items()}
    return value


def _order_free_text(text: str) -> str:
    return json.dumps(_order_free(json.loads(text)), sort_keys=True)


def _identity(text: str) -> str:
    return text


def fingerprint(fake: FakeLLM, out: Any, order_insensitive: bool = False) -> tuple[Fingerprint, dict[str, Any]]:
    grouped = recorded_by_agent(fake.recorded)
    canon = _order_free_text if order_insensitive else _identity
    requests = {
        agent: [_sha(r.path + "\n" + canon(r.body)) for r in items] for agent, items in sorted(grouped.items())
    }
    if order_insensitive:
        requests = {agent: sorted(hashes) for agent, hashes in requests.items()}
    output_text = canonical_output(out)
    dump = {
        "output": json.loads(output_text) if output_text.startswith(("{", "[")) else output_text,
        "requests": {
            agent: [{"path": r.path, "body": json.loads(r.body)} for r in items] for agent, items in sorted(grouped.items())
        },
    }
    return Fingerprint(calls=fake.calls, output=_sha(canon(output_text)), requests=requests), dump


def describe_mismatch(expected: Fingerprint, actual: Fingerprint) -> list[str]:
    problems: list[str] = []
    if expected.calls != actual.calls:
        problems.append(f"LLM calls: expected {expected.calls}, got {actual.calls}")
    if expected.output != actual.output:
        problems.append("final output differs")
    for agent in sorted(set(expected.requests) | set(actual.requests)):
        exp, got = expected.requests.get(agent, []), actual.requests.get(agent, [])
        if exp == got:
            continue
        if len(exp) != len(got):
            problems.append(f"agent {agent!r}: {len(exp)} requests expected, {len(got)} sent")
        first = next((i for i, (a, b) in enumerate(zip(exp, got)) if a != b), None)
        if first is not None:
            problems.append(f"agent {agent!r}: request #{first} payload differs")
    return problems


def load_golden() -> dict[str, Fingerprint]:
    if not GOLDEN_PATH.exists():
        return {}
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return {name: Fingerprint.from_json(entry) for name, entry in data.get("scenarios", {}).items()}


def save_golden(prints: dict[str, Fingerprint]) -> None:
    data = {
        "_comment": "Behavior fingerprints for `python -m benchmarks`. Regenerate only for intentional behavior changes.",
        "scenarios": {name: fp.to_json() for name, fp in sorted(prints.items())},
    }
    GOLDEN_PATH.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Running scenarios
# --------------------------------------------------------------------------


@dataclass
class Result:
    name: str
    group: str
    iterations: int
    calls: int
    summary_calls: int = 0
    bytes_sent: int = 0
    connections: int = 0
    wall_ms: list[float] = field(default_factory=list)
    net_ms: list[float] = field(default_factory=list)
    fingerprint: Optional[Fingerprint] = None
    behavior: str = "unchecked"  # ok | mismatch | new | unchecked | error
    problems: list[str] = field(default_factory=list)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.wall_ms) if self.wall_ms else 0.0

    @property
    def net_median_ms(self) -> float:
        return statistics.median(self.net_ms) if self.net_ms else 0.0

    @property
    def spread_pct(self) -> float:
        """Median absolute deviation as a percentage of the median: a noise indicator."""
        if len(self.net_ms) < 3 or not self.net_median_ms:
            return 0.0
        mad = statistics.median(abs(x - self.net_median_ms) for x in self.net_ms)
        return 100.0 * mad / self.net_median_ms

    @property
    def us_per_call(self) -> Optional[float]:
        return 1000.0 * self.net_median_ms / self.calls if self.calls else None

    def to_json(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "iterations": self.iterations,
            "llm_calls": self.calls,
            "summary_calls": self.summary_calls,
            "kb_sent": round(self.bytes_sent / 1024, 2),
            "est_input_tokens": self.bytes_sent // 4,
            "connections": self.connections,
            "median_ms": round(self.median_ms, 4),
            "net_median_ms": round(self.net_median_ms, 4),
            "min_net_ms": round(min(self.net_ms), 4) if self.net_ms else None,
            "spread_pct": round(self.spread_pct, 2),
            "net_us_per_llm_call": round(self.us_per_call, 2) if self.us_per_call is not None else None,
            "behavior": self.behavior,
            "digest": self.fingerprint.digest if self.fingerprint else None,
        }


def _fake_for(scenario: Scenario) -> FakeLLM:
    return FakeLLM(policies=dict(scenario.policies), default_policy=scenario.default_policy)


def _run_once(scenario: Scenario, fake: FakeLLM) -> tuple[Any, float, float]:
    fake.reset_counters()
    run = scenario.setup()
    gc.collect()
    start = time.perf_counter()
    out = run()
    wall = time.perf_counter() - start
    return out, wall, fake.fake_seconds


def run_scenario(
    scenario: Scenario,
    budget_s: float,
    min_iters: int,
    max_iters: int,
    cap_s: float,
    golden: Optional[Fingerprint],
    dump_dir: Optional[Path] = None,
) -> Result:
    fake = _fake_for(scenario)
    fake_provider.activate(fake)
    result = Result(scenario.name, scenario.group, iterations=0, calls=0)
    try:
        # Recording iteration (untimed): validates semantics and pins behavior.
        fake.record = True
        out, _, _ = _run_once(scenario, fake)
        fake.record = False
        if fake.errors:
            raise AssertionError("; ".join(fake.errors))
        scenario.check(out)
        result.calls = fake.calls
        result.summary_calls = fake.toolless_calls
        result.bytes_sent = fake.bytes_sent
        result.connections = fake.connections
        result.fingerprint, dump = fingerprint(fake, out, scenario.order_insensitive)
        if dump_dir is not None:
            dump_dir.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", scenario.name)
            (dump_dir / f"{safe}.json").write_text(json.dumps(dump, indent=1, sort_keys=True), encoding="utf-8")
        _judge_behavior(result, golden)

        # Keep iterating until the budget is spent and min_iters are done, but never
        # past cap_s: slow scenarios then get fewer (at least one) timed iterations.
        begin = time.perf_counter()
        while result.iterations < max_iters:
            elapsed = time.perf_counter() - begin
            if result.iterations and (elapsed >= cap_s or (elapsed >= budget_s and result.iterations >= min_iters)):
                break
            _, wall, fake_s = _run_once(scenario, fake)
            if fake.calls != result.calls:
                raise AssertionError(f"LLM call count changed between iterations: {result.calls} -> {fake.calls}")
            result.wall_ms.append(wall * 1000.0)
            result.net_ms.append((wall - fake_s) * 1000.0)
            result.iterations += 1
    except Exception as exc:  # noqa: BLE001 - a broken scenario must not abort the whole suite
        result.behavior = "error"
        result.problems.append(f"{type(exc).__name__}: {exc}")
    finally:
        fake_provider.activate(None)
    return result


def _judge_behavior(result: Result, golden: Optional[Fingerprint]) -> None:
    assert result.fingerprint is not None
    if golden is None:
        result.behavior = "new"
    elif golden.digest == result.fingerprint.digest:
        result.behavior = "ok"
    else:
        result.behavior = "mismatch"
        result.problems.extend(describe_mismatch(golden, result.fingerprint))


def profile_scenario(scenario: Scenario, iterations: int, sort: str, limit: int) -> str:
    fake = _fake_for(scenario)
    fake_provider.activate(fake)
    profiler = cProfile.Profile()
    try:
        _run_once(scenario, fake)  # warm caches and lazy imports
        for _ in range(iterations):
            fake.reset_counters()
            run = scenario.setup()
            profiler.enable()
            run()
            profiler.disable()
    finally:
        fake_provider.activate(None)
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs().sort_stats(sort).print_stats(limit)
    return stream.getvalue()


def measure_import(repeats: int, src_dir: Path) -> dict[str, float]:
    """Cold-start cost of importing the public API, net of bare interpreter startup."""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(src_dir), os.environ.get("PYTHONPATH")])))
    env.pop("PYTHONDONTWRITEBYTECODE", None)

    def best(code: str) -> float:
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            subprocess.run([sys.executable, "-c", code], check=True, env=env)
            samples.append(time.perf_counter() - start)
        return min(samples) * 1000.0

    bare = best("pass")
    full = best("import IkaCore; IkaCore.IkaBaseAgent; IkaCore.IkaWorkflow; IkaCore.IkaStage")
    return {"interpreter_ms": round(bare, 2), "import_ms": round(full, 2), "net_import_ms": round(full - bare, 2)}
