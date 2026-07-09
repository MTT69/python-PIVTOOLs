#!/usr/bin/env python3
"""Unified PIV benchmark/profile CLI.

Three subcommands, one harness:

    bench.py scaling  --config base.yaml --dataset DIR [...]   # worker×thread sweep (end-to-end)
    bench.py profile  --config base.yaml --dataset DIR [...]   # complete per-pair budget (serial)
    bench.py compare  --a RESULT_A --b RESULT_B                # FFTW-vs-codelet A/B figures

The harness is binary-agnostic: it measures whatever ``libbulkxcorr2d`` is built in
the worktree it runs from and stamps provenance (git SHA, FFT backend, host, CPU,
cache policy). Run ``scaling``/``profile`` once per worktree (FFTW in ``PyPIVTools/``,
codelet in ``PyPIVTools-fftw/``), then ``compare`` joins the two result sets and
guards that they are actually comparable.

Base ``config.yaml`` supplies all PIV settings (windows, passes, fit, save mode);
the CLI overrides only the dataset location, image count, sweep axes, and thread/
worker counts — so the science stays pinned and reproducible across runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Make sibling modules importable however bench.py is invoked, and ensure the
# worktree root is on the path so `import pivtools_cli` resolves to *this* worktree.
_PROFILE_DIR = Path(__file__).resolve().parent
_WORKTREE_ROOT = _PROFILE_DIR.parent
for _p in (str(_PROFILE_DIR), str(_WORKTREE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_common as bc  # noqa: E402

_FIGURES_DIR = _WORKTREE_ROOT / "figures" / "debug"


def _split_csv_arg(value: str | None) -> list[str] | None:
    """``"a.tif,b.tif"`` -> ``["a.tif", "b.tif"]``; ``None`` stays ``None``."""
    if value is None:
        return None
    return [s.strip() for s in value.split(",") if s.strip()]


# --- scaling ---------------------------------------------------------------


def cmd_scaling(args: argparse.Namespace) -> int:
    import bench_plots
    import bench_scaling as bs

    if args.plots_only:
        bench_plots.plot_scaling(Path(args.plots_only),
                                 Path(args.plots_only).resolve().parent)
        return 0

    total_cores = args.total_cores or os.cpu_count() or 1
    max_workers = args.max_workers or total_cores
    configs = bs.build_config_list(
        args.sweep, total_cores, max_workers, worker_sweep_threads=args.worker_threads,
        extra_workers=args.extra_workers,
    )
    if not configs:
        print("No configs generated for this sweep/core budget.")
        return 1

    if args.dry_run:
        bs.print_dry_run(configs, args.iterations, args.n_images, batch_hint=args.batch_hint)
        return 0

    if not Path(args.dataset).is_dir():
        print(f"ERROR: dataset directory not found: {args.dataset}")
        return 1

    csv_path = Path(args.resume) if args.resume else bc.timestamped_path("scaling", "csv")
    bs.run_sweep(
        args.config, args.dataset, configs, csv_path,
        n_images=args.n_images, n_iterations=args.iterations,
        total_cores=total_cores, max_workers=max_workers,
        image_format=_split_csv_arg(args.image_format), start_index=args.start_index,
        worker_memory=args.worker_memory, fftw_wisdom=args.fftw_wisdom,
        warmup_batches=args.warmup_batches,
    )
    bench_plots.plot_scaling(csv_path, csv_path.parent)
    return 0


# --- profile ---------------------------------------------------------------


def cmd_profile(args: argparse.Namespace) -> int:
    import bench_plots
    import bench_profile as bp

    if not Path(args.dataset).is_dir():
        print(f"ERROR: dataset directory not found: {args.dataset}")
        return 1

    config = bc.resolve_config(
        args.config, dataset=args.dataset, n_images=args.n_images,
        image_format=_split_csv_arg(args.image_format), start_index=args.start_index,
        threads=args.threads,
    )
    payload = bp.profile(
        args.mode, config=config, dataset=args.dataset, n_images=args.n_images,
        iterations=args.iterations, cache_policy=args.cache,
        do_warmup=not args.no_warmup, camera=args.camera,
    )

    out_json = Path(args.out) if args.out else bc.timestamped_path("profile", "json")
    bc.write_json(out_json, payload)
    print(f"\nProfile written: {out_json}")
    b = payload["budget_per_pair_ms"]
    print(f"  per-pair: read {b['read']:.2f} + filter {b['filter']:.2f} "
          f"+ compute {b['compute']:.2f} + save {b['save']:.2f} = {b['total']:.2f} ms  "
          f"[{payload['cache_policy']}]")
    ks = payload.get("kernel_split")
    if ks and ks.get("xcorr_fft_ms") is not None:
        print(f"  kernel split: FFT {ks['xcorr_fft_ms']:.3f} ms, "
              f"peak-fit {ks['peak_fit_ms']:.3f} ms (FFT frac {ks['fft_fraction']:.3f})")

    if not args.no_plot:
        fig = bp_plot_path(payload)
        bench_plots.plot_budget(payload, fig)
        print(f"  figure: {fig}")
    return 0


def bp_plot_path(payload: dict) -> Path:
    backend = payload.get("provenance", {}).get("fft_backend", "x")
    day = datetime.now().strftime("%Y-%m-%d")
    return _FIGURES_DIR / f"{day}-bench-profile-budget-{backend}.png"


# --- compare ---------------------------------------------------------------


def _resolve_result(path_str: str) -> dict[str, Path | None]:
    """Resolve an A/B input to its scaling CSV and/or profile JSON.

    A file is used directly (by extension). A directory is searched for the most
    recent ``scaling_*.csv`` and ``profile_*.json``."""
    p = Path(path_str)
    if p.is_file():
        if p.suffix == ".csv":
            return {"scaling_csv": p, "profile_json": None}
        if p.suffix == ".json":
            return {"scaling_csv": None, "profile_json": p}
        raise ValueError(f"unrecognised result file (need .csv or .json): {p}")
    if p.is_dir():
        csvs = sorted(p.glob("scaling_*.csv"))
        jsons = sorted(p.glob("profile_*.json"))
        return {
            "scaling_csv": csvs[-1] if csvs else None,
            "profile_json": jsons[-1] if jsons else None,
        }
    raise FileNotFoundError(f"A/B input not found: {p}")


def _provenance_from_csv(csv_path: Path) -> dict:
    """Synthesise a provenance-like dict from a scaling CSV's first valid row."""
    import csv as _csv

    with open(csv_path, newline="") as f:
        for r in _csv.DictReader(f):
            return {
                "fft_backend": r.get("fft_backend"),
                "git_sha": r.get("git_sha"),
                "git_dirty": r.get("git_dirty") in ("True", "true", True),
                "hostname": r.get("hostname"),
                "cpu_count": r.get("cpu_count"),
                "cpu_model": r.get("cpu_model"),
                # not stored in CSV — None==None matches, so they don't trip the guard
                "filesystem": None, "cache_policy": None, "platform": None,
                "pivtools_version": None,
            }
    return {}


def cmd_compare(args: argparse.Namespace) -> int:
    import bench_plots

    a, b = _resolve_result(args.a), _resolve_result(args.b)
    out_dir = Path(args.out_dir) if args.out_dir else _FIGURES_DIR
    day = datetime.now().strftime("%Y-%m-%d")

    # Provenance guard — prefer the richer profile-JSON stamp; fall back to CSV.
    def _prov(side: dict) -> dict | None:
        if side["profile_json"]:
            return json.loads(side["profile_json"].read_text()).get("provenance", {})
        if side["scaling_csv"]:
            return _provenance_from_csv(side["scaling_csv"])
        return None

    pa, pb = _prov(a), _prov(b)
    if pa is not None and pb is not None:
        guard = bc.compare_provenance(pa, pb)
        print("\nProvenance guard:")
        for note in guard["notes"]:
            print(f"  note: {note}")
        for warn in guard["warnings"]:
            print(f"  !! WARNING: {warn}")
        if guard["ok"]:
            print("  OK — comparable (differs only on the FFT backend).")
        else:
            print("  -> comparison proceeds but the figures may not be apples-to-apples.")
    else:
        print("\nProvenance guard skipped (one side has no result file).")

    wrote = []
    if a["scaling_csv"] and b["scaling_csv"]:
        p = bench_plots.plot_throughput_ab(
            a["scaling_csv"], b["scaling_csv"],
            out_dir / f"{day}-ab-fftw-vs-codelet-throughput.png",
        )
        wrote.append(p)
    if a["profile_json"] and b["profile_json"]:
        payload_a = json.loads(a["profile_json"].read_text())
        payload_b = json.loads(b["profile_json"].read_text())
        p = bench_plots.plot_budget_ab(
            payload_a, payload_b, out_dir / f"{day}-ab-fftw-vs-codelet-budget.png",
        )
        wrote.append(p)

    if not wrote:
        print("\nNothing to compare: need matching result types on both sides "
              "(two scaling CSVs and/or two profile JSONs).")
        return 1
    print("\nA/B figures:")
    for p in wrote:
        print(f"  {p}")
    return 0


# --- argument parsing ------------------------------------------------------


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, help="Base config.yaml (PIV settings pinned)")
    p.add_argument("--dataset", required=True, help="Image directory")
    p.add_argument("--n-images", type=int, default=None, help="Number of pairs")
    p.add_argument("--image-format", default=None,
                   help="Comma-separated printf pattern(s), e.g. 'B%%05d_A.tif,B%%05d_B.tif'")
    p.add_argument("--start-index", type=int, default=None, help="First frame index")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    # scaling
    s = sub.add_parser("scaling", help="worker×thread sweep (end-to-end throughput)")
    _add_common(s)
    s.add_argument("--sweep", choices=["threads", "workers", "matrix", "oversub", "all"],
                   default="all")
    s.add_argument("--total-cores", type=int, default=None,
                   help="Physical core budget (default: os.cpu_count())")
    s.add_argument("--max-workers", type=int, default=None,
                   help="Worker cap (RAM ceiling; default: total-cores)")
    s.add_argument("--worker-threads", type=int, default=2,
                   help="Threads/worker for the worker sweep axis (default: 2)")
    s.add_argument("--extra-workers", type=int, nargs="*", default=None,
                   help="Extra worker counts to inject into the worker sweep that the "
                        "geometric progression skips (e.g. 48 to keep the single-socket "
                        "edge on a 192-core node). Capped to fit total-cores/max-workers.")
    s.add_argument("--worker-memory", default=None, help="Per-worker memory limit, e.g. '3.2GB'")
    s.add_argument("--iterations", type=int, default=1)
    s.add_argument("--fftw-wisdom", choices=["shared", "per-worker"], default="shared",
                   help="FFTW wisdom policy (precondition 3). 'shared' is current behaviour.")
    s.add_argument("--warmup-batches", type=int, default=1,
                   help="Untimed warmup batches per worker before the timed run")
    s.add_argument("--batch-hint", type=int, default=10,
                   help="Batch size assumed for --dry-run pair counts (display only)")
    s.add_argument("--resume", metavar="CSV", default=None, help="Append to / resume a CSV")
    s.add_argument("--plots-only", metavar="CSV", default=None,
                   help="Re-render figures from an existing CSV and exit")
    s.add_argument("--dry-run", action="store_true", help="Print the matrix and exit")
    s.set_defaults(func=cmd_scaling)

    # profile
    pr = sub.add_parser("profile", help="complete per-pair time budget (serial)")
    _add_common(pr)
    pr.add_argument("--mode", choices=["instantaneous", "ensemble"], default="instantaneous")
    pr.add_argument("--threads", type=int, default=1,
                    help="OMP threads (default 1: kernel split reconciles with the wall)")
    pr.add_argument("--iterations", type=int, default=3)
    pr.add_argument("--cache", choices=["warm", "cold"], default="warm")
    pr.add_argument("--no-warmup", action="store_true")
    pr.add_argument("--camera", type=int, default=None)
    pr.add_argument("--out", default=None, help="JSON output path (default: results/profile_<ts>.json)")
    pr.add_argument("--no-plot", action="store_true")
    pr.set_defaults(func=cmd_profile)

    # compare
    c = sub.add_parser("compare", help="A/B figures from two result sets")
    c.add_argument("--a", required=True, help="Result A (CSV/JSON file or results dir)")
    c.add_argument("--b", required=True, help="Result B (CSV/JSON file or results dir)")
    c.add_argument("--out-dir", default=None, help="Figure output dir (default: figures/debug)")
    c.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
