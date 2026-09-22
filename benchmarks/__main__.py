"""IkaCore runtime benchmark.

    python -m benchmarks                     # full suite, checks behavior against golden.json
    python -m benchmarks --quick             # shorter per-scenario budget
    python -m benchmarks -k workflow         # only scenarios whose name contains "workflow"
    python -m benchmarks --json out.json     # save results
    python -m benchmarks --compare out.json  # show speedup vs a saved run
    python -m benchmarks --profile staged_3x3
    python -m benchmarks --update-golden     # re-pin behavior (only for intentional changes)

Timings are *net*: time spent inside the fake provider is subtracted, so the
numbers are IkaCore + httpx client overhead per scenario.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
try:
    import IkaCore  # noqa: F401
except ModuleNotFoundError:  # not pip-installed: benchmark the working tree
    sys.path.insert(0, str(SRC_DIR))

from benchmarks import fake_provider, harness  # noqa: E402
from benchmarks.scenarios import Scenario, all_scenarios  # noqa: E402


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m benchmarks", description=__doc__.split("\n\n")[0])
    parser.add_argument("-k", "--filter", action="append", default=[], help="substring filter on scenario names")
    parser.add_argument("--quick", action="store_true", help="shorter time budget per scenario")
    parser.add_argument("--budget", type=float, help="seconds per scenario (default 0.5, quick 0.15)")
    parser.add_argument("--min-iters", type=int, help="minimum timed iterations (default 5, quick 3)")
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--cap", type=float, help="hard cap in seconds of timed runs per scenario (default 2, quick 0.75)")
    parser.add_argument("--json", type=Path, help="write results as JSON")
    parser.add_argument("--compare", type=Path, help="compare against a previous --json run")
    parser.add_argument("--update-golden", action="store_true", help="re-pin behavior fingerprints")
    parser.add_argument("--dump", type=Path, help="write normalized requests/outputs per scenario for diffing")
    parser.add_argument("--no-import", action="store_true", help="skip the cold import measurement")
    parser.add_argument("--profile", metavar="SCENARIO", help="cProfile one scenario instead of benchmarking")
    parser.add_argument("--profile-iters", type=int, default=20)
    parser.add_argument("--profile-sort", default="cumulative", help="pstats sort key (cumulative, tottime, ...)")
    parser.add_argument("--profile-limit", type=int, default=40)
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    return parser.parse_args(argv)


def _select(scenarios: list[Scenario], filters: list[str]) -> list[Scenario]:
    if not filters:
        return scenarios
    return [s for s in scenarios if any(f in s.name for f in filters)]


def _fmt_ms(value: float) -> str:
    return f"{value:9.3f}" if value < 100 else f"{value:9.1f}"


def _print_table(results: list[harness.Result], baseline: dict[str, Any]) -> None:
    compare = bool(baseline)
    header = (
        f"{'scenario':<34} {'calls':>5} {'summ':>4} {'KB out':>7} {'conns':>5} "
        f"{'net ms':>9} {'us/call':>9} {'+-%':>5} {'iters':>5}  behavior"
    )
    if compare:
        header += f"  {'base ms':>9} {'speedup':>8}"
    print(header)
    print("-" * len(header))
    group = None
    for r in results:
        if r.group != group:
            group = r.group
            print(f"[{group}]")
        per_call = f"{r.us_per_call:9.1f}" if r.us_per_call is not None else f"{'-':>9}"
        line = (
            f"  {r.name:<32} {r.calls:>5} {r.summary_calls:>4} {r.bytes_sent / 1024:>7.1f} {r.connections:>5} "
            f"{_fmt_ms(r.net_median_ms)} {per_call} "
            f"{r.spread_pct:5.1f} {r.iterations:>5}  {r.behavior}"
        )
        base = baseline.get(r.name)
        if compare and base and base.get("net_median_ms") and r.net_median_ms:
            line += f"  {_fmt_ms(base['net_median_ms'])} {base['net_median_ms'] / r.net_median_ms:7.2f}x"
        print(line)
        for problem in r.problems:
            print(f"      ! {problem}")


def _geomean(values: list[float]) -> float:
    values = [v for v in values if v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else 0.0


def _summary(results: list[harness.Result], baseline: dict[str, Any], imports: Optional[dict[str, float]]) -> None:
    timed = [r for r in results if r.net_ms]
    total_calls = sum(r.calls for r in timed)
    total_ms = sum(r.net_median_ms for r in timed if r.group != "overhead")
    print()
    if imports:
        print(f"cold import: {imports['net_import_ms']:.1f} ms net ({imports['import_ms']:.1f} ms incl. interpreter)")
    print(f"suite: {len(timed)} scenarios, {total_calls} LLM calls/iteration, sum of net medians {total_ms:.1f} ms")
    agentic = [r for r in timed if r.group != "overhead"]
    calls = sum(r.calls for r in agentic)
    summaries = sum(r.summary_calls for r in agentic)
    conns = sum(r.connections for r in agentic)
    sent = sum(r.bytes_sent for r in agentic)
    if calls:
        print(
            f"traffic: {summaries}/{calls} LLM calls are summarization ({100 * summaries / calls:.0f}%), "
            f"{sent / 1024:.0f} KB sent (~{sent // 4:,} tokens), {conns} new connections "
            f"({conns / calls:.2f} per LLM call)"
        )
    if baseline:
        ratios = [
            baseline[r.name]["net_median_ms"] / r.net_median_ms
            for r in timed
            if r.name in baseline and baseline[r.name].get("net_median_ms") and r.net_median_ms
        ]
        if ratios:
            print(f"geometric-mean speedup vs baseline: {_geomean(ratios):.3f}x over {len(ratios)} scenarios")


def _behavior_verdict(results: list[harness.Result], update: bool) -> int:
    broken = [r for r in results if r.behavior in ("mismatch", "error")]
    new = [r for r in results if r.behavior == "new"]
    if update:
        errors = [r for r in results if r.behavior == "error"]
        if errors:
            print(f"\nnot updating golden: {len(errors)} scenario(s) errored")
            return 1
        prints = {name: fp for name, fp in harness.load_golden().items()}
        prints.update({r.name: r.fingerprint for r in results if r.fingerprint})
        harness.save_golden(prints)
        print(f"\nupdated {harness.GOLDEN_PATH.name} with {len(results)} fingerprint(s)")
        return 0
    if broken:
        print(f"\nBEHAVIOR CHANGED in {len(broken)} scenario(s): {', '.join(r.name for r in broken)}")
        print("Run with --dump DIR on both revisions and diff the files to see exactly what changed.")
        return 1
    if new:
        print(f"\n{len(new)} scenario(s) have no golden fingerprint yet; run --update-golden to pin them.")
    else:
        print("\nbehavior: all fingerprints match golden.json")
    return 0


def _run_profile(scenarios: list[Scenario], args: argparse.Namespace) -> int:
    matches = [s for s in scenarios if s.name == args.profile] or [s for s in scenarios if args.profile in s.name]
    if not matches:
        print(f"no scenario matches {args.profile!r}; use --list")
        return 2
    scenario = matches[0]
    with harness.sandbox_cwd(), harness.quiet():
        report = harness.profile_scenario(scenario, args.profile_iters, args.profile_sort, args.profile_limit)
    print(f"profile of {scenario.name} over {args.profile_iters} iterations (includes fake-provider time)\n")
    print(report)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    scenarios = _select(all_scenarios(), args.filter)
    if args.list:
        for s in scenarios:
            print(f"{s.name:<34} [{s.group}] {s.description}")
        return 0

    fake_provider.install()
    if args.profile:
        return _run_profile(scenarios, args)

    budget = args.budget if args.budget is not None else (0.15 if args.quick else 0.5)
    min_iters = args.min_iters if args.min_iters is not None else (3 if args.quick else 5)
    cap = args.cap if args.cap is not None else (0.75 if args.quick else 2.0)
    golden = harness.load_golden()
    baseline = json.loads(args.compare.read_text(encoding="utf-8"))["scenarios"] if args.compare else {}

    started = time.perf_counter()
    results: list[harness.Result] = []
    with harness.sandbox_cwd():
        for scenario in scenarios:
            with harness.quiet():
                results.append(harness.run_scenario(
                    scenario, budget, min_iters, args.max_iters, cap, golden.get(scenario.name), args.dump
                ))
    imports = None if (args.no_import or args.filter) else harness.measure_import(3 if args.quick else 5, SRC_DIR)

    _print_table(results, baseline)
    _summary(results, baseline, imports)
    print(f"elapsed: {time.perf_counter() - started:.1f} s on Python {platform.python_version()} ({platform.system()})")

    if args.json:
        payload = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "budget_s": budget,
            "import": imports,
            "scenarios": {r.name: r.to_json() for r in results},
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return _behavior_verdict(results, args.update_golden)


if __name__ == "__main__":
    sys.exit(main())
