"""Standalone microbenchmark for the fused image-warp kernel (`libfusedwarp`).

Times the two warp impls in one loaded ``.so`` via the runtime ``fused_warp_set_impl``
flag — impl 0 = scalar reference (always bounds-checked), impl 1 = the optimised
path (interior/border split, later SIMD). Reports per-pair ms, speedup, and the
achieved tap throughput, plus the max scalar-vs-optimised difference so a numerical
regression shows up immediately. No Dask, no production pipeline — just the kernel on
synthetic data, so it iterates in seconds while tuning the C.

Usage:
    python profile/bench_warp.py [--size 2048] [--pairs 1] [--iters 7] [--mode both]

Run it from inside the worktree so it loads *that* tree's freshly-built libfusedwarp.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_PROFILE_DIR = Path(__file__).resolve().parent
_WORKTREE = _PROFILE_DIR.parent
# Reuse the synthetic-data generators from the test (single source of truth).
sys.path.insert(0, str(_WORKTREE / "unit-tests"))
sys.path.insert(0, str(_PROFILE_DIR))
from test_fused_warp import (  # noqa: E402
    make_grid_image,
    make_poiseuille_predictor,
    make_window_centres,
)

# Taps per output pixel: Phase A (2× bicubic 4×4 = 32) + Phase C (2× sample).
_PHASE_A_TAPS = 32
_TAPS_PHASE_C = {0: 2 * 16, 1: 2 * 36}  # bicubic 2×16, lanczos 2×36
_MODE_NAME = {0: "bicubic", 1: "lanczos"}
_NDP = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")


def _load_lib() -> ctypes.CDLL:
    ext = ".dll" if sys.platform.startswith("win") else ".so"
    path = _WORKTREE / "pivtools_cli" / "lib" / f"libfusedwarp{ext}"
    if not path.is_file():
        raise FileNotFoundError(
            f"libfusedwarp not found at {path} — run `python setup.py build`"
        )
    lib = ctypes.CDLL(str(path))
    lib.fused_symmetric_warp_batch.restype = ctypes.c_int
    lib.fused_symmetric_warp_batch.argtypes = [
        _NDP,
        _NDP,
        _NDP,
        _NDP,  # imgs_a, imgs_b, outs_a, outs_b
        _NDP,
        _NDP,  # pred_dy, pred_dx
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,  # N, H, W
        ctypes.c_int,
        ctypes.c_int,  # nPY, nPX
        _NDP,
        _NDP,  # ctrs_y, ctrs_x
        ctypes.c_int,
        ctypes.c_int,  # interp_mode, shared_predictor
    ]
    lib.fused_warp_set_impl.argtypes = [ctypes.c_int]
    lib.fused_warp_set_impl.restype = None
    lib.fused_warp_get_impl.argtypes = []
    lib.fused_warp_get_impl.restype = ctypes.c_int
    return lib, path


def _make_batch(N: int, H: int, W: int):
    """Synthetic batch: grid images + a per-image Poiseuille predictor + centres."""
    nPY = nPX = 32
    ctrs_y, ctrs_x = make_window_centres(H, W, nPY, nPX)
    pdy, pdx = make_poiseuille_predictor(nPY, nPX, ctrs_y, ctrs_x, u_max=8.0, H=H)
    imgs_a = np.ascontiguousarray(
        np.stack([make_grid_image(H, W, spacing=16) for _ in range(N)]),
        dtype=np.float32,
    )
    imgs_b = np.ascontiguousarray(
        np.stack([make_grid_image(H, W, spacing=20) for _ in range(N)]),
        dtype=np.float32,
    )
    pred_dy = np.ascontiguousarray(np.stack([pdy] * N), dtype=np.float32)
    pred_dx = np.ascontiguousarray(np.stack([pdx] * N), dtype=np.float32)
    return (
        imgs_a,
        imgs_b,
        pred_dy,
        pred_dx,
        np.ascontiguousarray(ctrs_y, np.float32),
        np.ascontiguousarray(ctrs_x, np.float32),
        nPY,
        nPX,
    )


def _run(lib, data, N, H, W, interp_mode, impl):
    imgs_a, imgs_b, pred_dy, pred_dx, ctrs_y, ctrs_x, nPY, nPX = data
    out_a = np.zeros((N, H, W), dtype=np.float32)
    out_b = np.zeros((N, H, W), dtype=np.float32)
    lib.fused_warp_set_impl(impl)
    ret = lib.fused_symmetric_warp_batch(
        imgs_a,
        imgs_b,
        out_a,
        out_b,
        pred_dy,
        pred_dx,
        N,
        H,
        W,
        nPY,
        nPX,
        ctrs_y,
        ctrs_x,
        interp_mode,
        0,
    )
    if ret != 0:
        raise RuntimeError(f"kernel returned {ret}")
    return out_a, out_b


def _time(lib, data, N, H, W, interp_mode, impl, iters):
    _run(lib, data, N, H, W, interp_mode, impl)  # warm-up (plans, page-touch)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _run(lib, data, N, H, W, interp_mode, impl)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--size", type=int, default=2048, help="image side (square)")
    p.add_argument("--pairs", type=int, default=1, help="N image pairs in the batch")
    p.add_argument("--iters", type=int, default=7, help="timed repeats (median)")
    p.add_argument("--mode", choices=["bicubic", "lanczos", "both"], default="both")
    p.add_argument(
        "--threads",
        type=int,
        default=1,
        help="OMP_NUM_THREADS (default 1: isolates per-core SIMD from threading)",
    )
    args = p.parse_args(argv)

    os.environ["OMP_NUM_THREADS"] = str(
        args.threads
    )  # must precede the first kernel call

    H = W = args.size
    N = args.pairs
    lib, lib_path = _load_lib()
    data = _make_batch(N, H, W)
    modes = [0, 1] if args.mode == "both" else [0 if args.mode == "bicubic" else 1]

    try:
        import bench_common as bc

        cpu = bc._cpu_model()
    except Exception:
        import platform

        cpu = platform.processor() or "unknown"
    mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(lib_path.stat().st_mtime))
    print(f"\nlibfusedwarp: {lib_path}  (built {mtime})")
    print(
        f"CPU: {cpu}   batch N={N}  {H}×{W}  threads={args.threads}  iters={args.iters}\n"
    )
    print(
        f"{'mode':<9}{'impl0 ref ms':>13}{'impl1 opt ms':>13}{'speedup':>9}"
        f"{'opt Gtaps/s':>12}{'max|Δ|':>10}"
    )
    print("-" * 66)

    for mode in modes:
        t_ref = _time(lib, data, N, H, W, mode, 0, args.iters)
        t_opt = _time(lib, data, N, H, W, mode, 1, args.iters)
        ref_a, ref_b = _run(lib, data, N, H, W, mode, 0)
        opt_a, opt_b = _run(lib, data, N, H, W, mode, 1)
        max_d = float(max(np.max(np.abs(opt_a - ref_a)), np.max(np.abs(opt_b - ref_b))))
        taps = N * H * W * (_PHASE_A_TAPS + _TAPS_PHASE_C[mode])
        ms_ref, ms_opt = 1e3 * t_ref / N, 1e3 * t_opt / N
        gtaps = (taps / t_opt) / 1e9
        print(
            f"{_MODE_NAME[mode]:<9}{ms_ref:>13.2f}{ms_opt:>13.2f}"
            f"{t_ref/t_opt:>8.2f}x{gtaps:>12.2f}{max_d:>10.2e}"
        )
    print()


if __name__ == "__main__":
    main()
