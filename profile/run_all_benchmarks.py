"""
Run the full benchmark suite for the PIV speed report.

  1. benchmark_batch_and_resolution.py  (batch sweep, resolution, save I/O)
  2. benchmark_scaling.py --sweep threads  (thread scaling via Dask)
  3. benchmark_scaling.py --sweep workers  (worker scaling via Dask)

Usage:
    python profile/run_all_benchmarks.py           # Run all three
    python profile/run_all_benchmarks.py --step 2  # Run only step 2 (thread scaling)
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime

PROFILE_DIR = __file__.replace("\\", "/").rsplit("/", 1)[0]
PYTHON = sys.executable

STEPS = [
    {
        "name": "Batch size + Resolution + Save I/O",
        "cmd": [PYTHON, f"{PROFILE_DIR}/benchmark_batch_and_resolution.py"],
    },
    {
        "name": "Thread scaling (Dask)",
        "cmd": [PYTHON, f"{PROFILE_DIR}/benchmark_scaling.py", "--sweep", "threads"],
    },
    {
        "name": "Worker scaling (Dask)",
        "cmd": [PYTHON, f"{PROFILE_DIR}/benchmark_scaling.py", "--sweep", "workers"],
    },
]


def run_step(step, index):
    print(f"\n{'='*70}")
    print(f"STEP {index+1}/{len(STEPS)}: {step['name']}")
    print(f"  cmd: {' '.join(step['cmd'])}")
    print(f"  started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}\n")

    t0 = time.perf_counter()
    result = subprocess.run(step["cmd"])
    elapsed = time.perf_counter() - t0

    status = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"\n  [{status}] {step['name']} — {elapsed/60:.1f} minutes")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run full PIV benchmark suite")
    parser.add_argument("--step", type=int, choices=[1, 2, 3],
                        help="Run only a specific step")
    args = parser.parse_args()

    started = datetime.now()
    print(f"PIV Benchmark Suite — {started.strftime('%Y-%m-%d %H:%M:%S')}")

    if args.step:
        steps_to_run = [(args.step - 1, STEPS[args.step - 1])]
    else:
        steps_to_run = list(enumerate(STEPS))

    results = []
    for idx, step in steps_to_run:
        ok = run_step(step, idx)
        results.append((step["name"], ok))

    elapsed = (datetime.now() - started).total_seconds()
    print(f"\n\n{'='*70}")
    print(f"BENCHMARK SUITE COMPLETE — {elapsed/60:.1f} minutes total")
    print(f"{'='*70}")
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print()


if __name__ == "__main__":
    main()
