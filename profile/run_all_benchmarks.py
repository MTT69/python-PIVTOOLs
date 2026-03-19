"""
Run the full benchmark suite for the PIV speed report.

  1. I/O decomposition          (image read breakdown, save combos, Dask overhead, waterfall)
  2. Batch size + Resolution     (batch sweep, resolution scaling, save I/O matrix)
  3. Thread scaling (Dask)       (1 worker, vary threads — steady-state with warmup)
  4. Worker scaling (Dask)       (4 threads/worker, vary workers — steady-state with warmup)
  5. Create 1MP crops            (centre-crop 4MP -> 1MP for scaling benchmarks)
  6. Worker scaling — 4MP 2-pass (64->32, vary workers)
  7. Worker scaling — 1MP 2-pass (64->32, vary workers)
  8. Detailed correlator profile (per-section timing, 4MP, 20 pairs)

Usage:
    python profile/run_all_benchmarks.py           # Run all 8
    python profile/run_all_benchmarks.py --step 3  # Run only step 3
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime

PROFILE_DIR = __file__.replace("\\", "/").rsplit("/", 1)[0]
PYTHON = sys.executable

SOURCE_1MP = (
    r"C:\Users\mtt1e23\OneDrive - University of Southampton\Documents"
    r"\#current_processing\4000_images_channel\planar_images\1mp"
)

STEPS = [
    {
        "name": "I/O decomposition (read, write, Dask overhead, waterfall)",
        "cmd": [PYTHON, f"{PROFILE_DIR}/benchmark_io_decomposition.py"],
    },
    {
        "name": "Batch size + Resolution + Save I/O",
        "cmd": [PYTHON, f"{PROFILE_DIR}/benchmark_batch_and_resolution.py"],
    },
    {
        "name": "Thread scaling (Dask, steady-state)",
        "cmd": [PYTHON, f"{PROFILE_DIR}/benchmark_scaling.py", "--sweep", "threads"],
    },
    {
        "name": "Worker scaling (Dask, steady-state)",
        "cmd": [PYTHON, f"{PROFILE_DIR}/benchmark_scaling.py", "--sweep", "workers"],
    },
    {
        "name": "Create 1MP crops (for scaling benchmarks)",
        "cmd": [PYTHON, f"{PROFILE_DIR}/create_1mp_crops.py", "--pairs", "1000"],
    },
    {
        "name": "Worker scaling — 4MP 2-pass (64->32)",
        "cmd": [PYTHON, f"{PROFILE_DIR}/benchmark_scaling.py", "--sweep", "workers",
                "--windows", "64,32"],
    },
    {
        "name": "Worker scaling — 1MP 2-pass (64->32)",
        "cmd": [PYTHON, f"{PROFILE_DIR}/benchmark_scaling.py", "--sweep", "workers",
                "--source", SOURCE_1MP, "--windows", "64,32"],
    },
    {
        "name": "Detailed correlator profile (4MP, 20 pairs, 10 threads)",
        "cmd": [PYTHON, f"{PROFILE_DIR}/profile_piv.py", "4mp", "--pairs", "20", "--threads", "10"],
    },
]


def run_step(step, index, total):
    print(f"\n{'='*70}")
    print(f"STEP {index+1}/{total}: {step['name']}")
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
    parser.add_argument("--step", type=int, choices=list(range(1, len(STEPS) + 1)),
                        help="Run only a specific step")
    args = parser.parse_args()

    started = datetime.now()
    print(f"PIV Benchmark Suite — {started.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {len(STEPS)} steps")

    if args.step:
        steps_to_run = [(args.step - 1, STEPS[args.step - 1])]
    else:
        steps_to_run = list(enumerate(STEPS))

    total = len(steps_to_run)
    results = []
    for idx, step in steps_to_run:
        ok = run_step(step, idx, total)
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
