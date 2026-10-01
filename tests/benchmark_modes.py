"""Repeatable end-to-end timing comparison of the three bridge models.

This is deliberately not a pytest module: a full run takes minutes and wall-clock performance is
machine dependent.  It has no pass/fail threshold.  Run it from the repository root with
``python -B tests/benchmark_modes.py``; the unmodified ``gfl-example.pes`` is used except for the
bridge-model selection.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from peslite import Simulation, load  # noqa: E402


MODES = ("switching", "pwm_averaging", "averaging")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=1,
                        help="complete runs per mode; the order rotates between repeats")
    parser.add_argument("--t-end", type=float, default=None,
                        help="optional shorter stop time for profiling; default: use the example")
    parser.add_argument("--mode", action="append", choices=MODES,
                        help="benchmark only this mode; repeat to select more than one")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    config = ROOT / "examples" / "gfl-example.pes"
    base = load(config)
    modes = tuple(dict.fromkeys(args.mode)) if args.mode else MODES
    samples = {mode: [] for mode in modes}
    with tempfile.TemporaryDirectory(prefix=".benchmark-modes-", dir=ROOT) as temporary:
        output_root = Path(temporary)
        for repeat in range(args.repeats):
            order = modes[repeat % len(modes):] + modes[:repeat % len(modes)]
            for mode in order:
                params = base.replace(
                    **{f"units.{name}.bridge.model": mode for name in base.units}
                )
                simulation = Simulation(params)
                gc.collect()
                cpu0 = time.process_time()
                result = simulation.run(t_end=args.t_end,
                                        out_dir=output_root / f"{repeat}-{mode}")
                sample = {
                    "repeat": repeat + 1,
                    "wall_time": result.wall_time,
                    "cpu_time": time.process_time() - cpu0,
                    "n_rhs": result.n_rhs,
                    "t_stop": result.summary.get("t_stop"),
                }
                samples[mode].append(sample)
                print(
                    f"{mode:15s} repeat {repeat + 1}: wall {sample['wall_time']:.3f} s, "
                    f"CPU {sample['cpu_time']:.3f} s, RHS {sample['n_rhs']}",
                    flush=True,
                )

    summary = {}
    for mode in modes:
        rows = samples[mode]
        summary[mode] = {
            "wall_time": statistics.median(row["wall_time"] for row in rows),
            "cpu_time": statistics.median(row["cpu_time"] for row in rows),
            "n_rhs": int(statistics.median(row["n_rhs"] for row in rows)),
        }
    baseline_mode = "switching" if "switching" in summary else modes[0]
    baseline = summary[baseline_mode]["wall_time"]
    print(f"\n| mode | median wall | median CPU | RHS | speedup vs {baseline_mode} |")
    print("|---|---:|---:|---:|---:|")
    for mode in modes:
        row = summary[mode]
        print(
            f"| {mode} | {row['wall_time']:.3f} s | {row['cpu_time']:.3f} s | "
            f"{row['n_rhs']:,} | {baseline / row['wall_time']:.3f}x |"
        )
    print("BENCHMARK_JSON " + json.dumps({"samples": samples, "summary": summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
