"""Upper bounds for lossless graph rewrites, replayed on a measured long-horizon trace.

    python -m benchmarks.lossless_ablation CALLS.csv [--framework ikacore] [--rtt-ms 336] [--tool-ms 1]

Input is the ``--calls-csv`` written by ``benchmarks.long_horizon`` (every model call's node,
tokens and latency, in completion order per node). Each node's duration is the sum of its
sequential calls; the workflow's time is the critical path through the audit DAG. Each
technique is applied to the recorded calls and the critical path is recomputed, so every row is
the best that technique could do on this exact run, without building it:

  cache warming      prefill the next request's prefix early (Teola "prefill splitting"): removes
                     the fitted per-input-token latency from every call
  no round trips     dependent calls chained provider-side (Parrot): removes one network round
                     trip per call; hosted APIs don't offer this, shown for scale
  early tool start   run a tool as soon as its call streams in (ALTO/Teola pipelining): removes
                     tool time between calls
  speculation p=..   predict each tool-call turn exactly, run the tool, send the next request
                     while the current one is in flight (Speculative Actions), at most one
                     speculative request ahead; lossless only on an exact match, a miss wastes a call
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Callable, Optional

from benchmarks.long_horizon import _solve, critical_path

Calls = dict[str, list[dict[str, float]]]


def load(path: Path, framework: str) -> Calls:
    calls: Calls = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["framework"] != framework or row["node"] in ("summary", "?"):
                continue
            calls.setdefault(row["node"], []).append(
                {k: float(row[k]) for k in ("input_tokens", "cached_tokens", "output_tokens", "seconds")})
    return calls


def fit(calls: Calls) -> tuple[float, float, float, float]:
    log = [c for cs in calls.values() for c in cs]
    rows = [[1.0, c["input_tokens"] - c["cached_tokens"], c["cached_tokens"], c["output_tokens"]] for c in log]
    fixed, per_in, per_cached, per_out = _solve(rows, [c["seconds"] for c in log])
    return fixed, per_in, per_cached, per_out


def makespan(calls: Calls, latency: Callable[[dict[str, float]], float]) -> float:
    return critical_path({node: sum(max(0.0, latency(c)) for c in cs) for node, cs in calls.items()})


def speculative_node(durations: list[float], p: float, rng: random.Random) -> tuple[float, int]:
    """One node's time with depth-1 speculation; returns (duration, wasted calls)."""
    start, prev_end, end, wasted = 0.0, 0.0, 0.0, 0
    for k, d in enumerate(durations):
        if k:
            hit = rng.random() < p  # was call k-1's output predicted exactly?
            wasted += 0 if hit else 1
            start = max(start, prev_end) if hit else end  # hit: overlaps call k-1, one request ahead
        prev_end, end = end, start + d
    return end, wasted


def speculation(calls: Calls, p: float, trials: int = 400) -> tuple[float, float]:
    rng = random.Random(7)
    total, wasted = 0.0, 0
    for _ in range(trials):
        durations = {}
        for node, cs in calls.items():
            durations[node], w = speculative_node([c["seconds"] for c in cs], p, rng)
            wasted += w
        total += critical_path(durations)
    return total / trials, wasted / trials


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.lossless_ablation", description=__doc__.split("\n\n")[0])
    parser.add_argument("csv", type=Path)
    parser.add_argument("--framework", default="ikacore")
    parser.add_argument("--rtt-ms", type=float, default=336.0, help="measured warm round trip to the provider")
    parser.add_argument("--tool-ms", type=float, default=1.0, help="tool + framework time between calls")
    args = parser.parse_args(argv)
    calls = load(args.csv, args.framework)
    n_calls = sum(len(cs) for cs in calls.values())
    fixed, per_in, per_cached, per_out = fit(calls)
    base = makespan(calls, lambda c: c["seconds"])
    print(f"{args.framework}: {n_calls} calls; fit {fixed:.2f}s fixed + {1e6 * per_in:.1f}ms/1k uncached "
          f"+ {1e6 * per_cached:.1f}ms/1k cached + {1e3 * per_out:.1f}ms/output token")
    print(f"critical path of the recorded calls: {base:.1f}s\n")
    rows: list[tuple[str, float, str]] = [
        ("cache warming", makespan(calls, lambda c: c["seconds"] - max(0.0, per_in) * (c["input_tokens"] - c["cached_tokens"])
                                   - max(0.0, per_cached) * c["cached_tokens"]), "no extra calls"),
        ("no round trips", makespan(calls, lambda c: c["seconds"] - args.rtt_ms / 1000), "not offered by hosted APIs"),
        ("early tool start", base - args.tool_ms / 1000 * max(len(cs) for cs in calls.values()), "no extra calls"),
    ]
    for p in (0.25, 0.55, 0.8, 1.0):
        span, wasted = speculation(calls, p)
        rows.append((f"speculation p={p:.2f}", span, f"+{wasted:.0f} wasted calls (+{100 * wasted / n_calls:.0f}%), "
                                                        f"plus a predictor per tool turn"))
    print(f"{'technique':<22}{'critical path':>14}{'saved':>9}   cost")
    for name, span, cost in rows:
        print(f"{name:<22}{span:>13.1f}s{100 * (base - span) / base:>8.0f}%   {cost}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
