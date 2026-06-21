"""Figures for ``bench.py``.

So far: :func:`plot_budget` — the per-pair time budget (the augmented version of the
classic compute-breakdown figure, now including read and save). Scaling and A/B
side-by-side figures are added alongside their data in later steps.

Section → display-category mapping lives here (not in ``bench_profile``) so the raw
per-section timings stay in the JSON and only the *presentation* buckets them.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Compute sections → the five figure categories. PC sub-sections are intentionally
# omitted (they are nested inside predictor_corrector; counting them double-bills warp).
_CATEGORIES: list[tuple[str, list[str], str]] = [
    ("Prep/Warp", ["predictor_corrector", "set_lib_args", "padding_stacking"], "#c0504d"),
    ("Cross-corr", ["bulkxcorr2d"], "#4f81bd"),
    ("Outlier", ["outlier_detection", "secondary_peaks"], "#9bbb59"),
    ("Infilling", ["infilling"], "#8064a2"),
    ("Other", ["post_processing", "result_construction"], "#a6a6a6"),
]
_PC_SUB_SECTIONS = {"pc_gaussian_smooth", "pc_predictor_remap", "pc_fused_warp"}
# bulkxcorr2d FFT/peak-fit split sections — excluded from the category buckets (they
# are a finer breakdown of Cross-corr, drawn as sub-segments when single-thread).
_KERNEL_SUB_SECTIONS = {"xcorr_fft", "peak_fit"}
_READ_COLOR = "#404040"
_FILTER_COLOR = "#17becf"  # teal — distinct from the navy Cross-corr bar
_SAVE_COLOR = "#e8a33d"
# Cross-corr 3-way split (correlation family — shades of the navy Cross-corr blue)
_XCORR_FFT_COLOR = "#4f81bd"
_PEAK_FIT_COLOR = "#94b3d6"
_KERNEL_OTHER_COLOR = "#cdddf0"


def _bucket_pass(sections_ms: dict[str, dict[str, float]]) -> dict[str, float]:
    """Bucket one pass's per-section mean times into the five display categories.
    Any section not explicitly mapped (and not a PC sub-section) lands in 'Other', so
    nothing is silently dropped from the total."""
    mapped = {name: 0.0 for name, _, _ in _CATEGORIES}
    claimed = set(_PC_SUB_SECTIONS) | _KERNEL_SUB_SECTIONS
    for name, keys, _ in _CATEGORIES:
        for k in keys:
            if k in sections_ms:
                mapped[name] += sections_ms[k]["mean_ms"]
                claimed.add(k)
    for sec, stat in sections_ms.items():
        if sec not in claimed:
            mapped["Other"] += stat["mean_ms"]
    return mapped


def plot_budget(payload: dict[str, Any], out_path: Path) -> Path:
    """Render the per-pair time budget: one stacked bar per pass (compute categories)
    plus separate Read and Save bars, so I/O and disk are visible alongside compute.

    :param payload: a ``bench_profile`` result dict.
    :param out_path: PNG destination (parent created if needed).
    """
    passes = payload["passes"]
    prov = payload.get("provenance", {})
    backend = prov.get("fft_backend", "?")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # x positions: Read | Filter | pass bars... | Save
    labels = ["Read", "Filter"]
    labels += [f"Pass {p['pass_idx'] + 1}\n({p['window'][0]} px)" for p in passes]
    labels += ["Save"]
    x = list(range(len(labels)))

    # Read bar
    ax.bar(x[0], payload["budget_per_pair_ms"]["read"], color=_READ_COLOR, label="Read (I/O)")

    # Filter bar (production preprocessing stage between read and correlate)
    ftypes = payload.get("filter_types") or []
    flabel = f"Filter ({', '.join(ftypes)})" if ftypes else "Filter (none)"
    ax.bar(x[1], payload["budget_per_pair_ms"]["filter"], color=_FILTER_COLOR, label=flabel)

    # Cross-corr splits into FFT / peak-fit / kernel-other only when the run was
    # single-thread (the C timers are thread-summed, so at >1 thread they overshoot
    # the bulkxcorr2d wall and would not fit inside the bar).
    split = payload.get("omp_threads") == 1 and all(
        "xcorr_fft" in p["sections_ms"] and "peak_fit" in p["sections_ms"] for p in passes
    )

    def _draw_segment(xi, h, bottom, colour, label, legend_done):
        if h <= 0:
            return bottom
        ax.bar(xi, h, bottom=bottom, color=colour,
               label=label if label not in legend_done else None)
        legend_done.add(label)
        return bottom + h

    # Per-pass stacked compute bars
    legend_done = set()
    for i, p in enumerate(passes):
        bucket = _bucket_pass(p["sections_ms"])
        bottom = 0.0
        for name, _, colour in _CATEGORIES:
            h = bucket[name]
            if h <= 0:
                continue
            if name == "Cross-corr" and split:
                # h is the bulkxcorr2d section; split into FFT, peak-fit, remainder.
                s = p["sections_ms"]
                fft = s["xcorr_fft"]["mean_ms"]
                fit = s["peak_fit"]["mean_ms"]
                other = max(0.0, h - fft - fit)
                bottom = _draw_segment(x[2 + i], fft, bottom, _XCORR_FFT_COLOR,
                                       "Cross-corr: FFT", legend_done)
                bottom = _draw_segment(x[2 + i], fit, bottom, _PEAK_FIT_COLOR,
                                       "Cross-corr: peak-fit", legend_done)
                bottom = _draw_segment(x[2 + i], other, bottom, _KERNEL_OTHER_COLOR,
                                       "Cross-corr: kernel-other", legend_done)
            else:
                bottom = _draw_segment(x[2 + i], h, bottom, colour, name, legend_done)

    # Save bar
    ax.bar(x[-1], payload["save_ms"]["mean_ms"], color=_SAVE_COLOR, label="Save (I/O)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Time per pair (ms)")
    total = payload["budget_per_pair_ms"]["total"]
    omp = payload.get("omp_threads", "?")
    ax.set_title(
        f"Per-pair time budget — {payload['mode']} [{backend}]  "
        f"(total {total:.2f} ms/pair, n={payload['n_images']}, cache={payload['cache_policy']}, "
        f"{omp} thread{'s' if omp != 1 else ''})"
    )
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# --- scaling CSV helpers ---------------------------------------------------


def load_scaling_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Load a scaling CSV, coercing numeric columns and keeping only valid rows."""
    rows: list[dict[str, Any]] = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("valid") != "true":
                continue
            for k in ("workers", "threads", "total_cores", "n_pairs"):
                r[k] = int(float(r[k]))
            for k in ("pairs_per_s", "per_pair_ms", "load_s", "correlate_s",
                      "oversub_ratio"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


def _grouped_throughput(rows: list[dict[str, Any]]) -> dict[tuple[int, int], dict[str, float]]:
    """Mean throughput/per-pair per (workers, threads) across iterations."""
    buckets: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        buckets[(r["workers"], r["threads"])].append(r)
    out = {}
    for key, rs in buckets.items():
        out[key] = {
            "pairs_per_s": float(np.mean([r["pairs_per_s"] for r in rs])),
            "per_pair_ms": float(np.mean([r["per_pair_ms"] for r in rs])),
            "total_cores": key[0] * key[1],
        }
    return out


def _phys_cores(rows: list[dict[str, Any]]) -> int:
    """Largest non-oversubscribed core count — the reference line for the figures."""
    within = [r["total_cores"] for r in rows if r["oversub_ratio"] <= 1.0]
    return max(within) if within else max((r["total_cores"] for r in rows), default=1)


def plot_scaling(csv_path: Path, out_dir: Path) -> list[Path]:
    """Render the scaling figures from one CSV: strong-scaling (speedup +
    efficiency) from the worker sweep with a 1-worker baseline, and a
    throughput-vs-cores overlay. Returns the written paths."""
    rows = load_scaling_csv(Path(csv_path))
    if not rows:
        print("plot_scaling: no valid rows.")
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(csv_path).stem
    grouped = _grouped_throughput(rows)
    phys = _phys_cores(rows)
    written: list[Path] = []

    # --- throughput vs total cores ---
    fig, ax = plt.subplots(figsize=(9, 6))
    for (w, t), v in sorted(grouped.items(), key=lambda kv: kv[1]["total_cores"]):
        cores = v["total_cores"]
        ax.scatter(cores, v["pairs_per_s"],
                   color="tab:red" if cores > phys else "steelblue", s=60, zorder=3)
        ax.annotate(f"{w}w×{t}t", (cores, v["pairs_per_s"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7)
    base = grouped.get((1, 1))
    if base and base["pairs_per_s"] > 0:
        xs = np.arange(1, max(v["total_cores"] for v in grouped.values()) + 1)
        ax.plot(xs, base["pairs_per_s"] * xs, "--", color="gray", alpha=0.5,
                label="Linear scaling")
    ax.axvline(phys, color="red", ls=":", alpha=0.5, label=f"{phys} physical cores")
    ax.set_xlabel("Total logical cores (workers × threads)")
    ax.set_ylabel("Pairs / second")
    ax.set_title(f"Throughput vs cores — {rows[0].get('fft_backend', '?')}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / f"{stem}_throughput.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    written.append(p)

    # --- strong scaling: pick the fixed-thread worker sweep with a w=1 baseline ---
    by_thread: dict[int, list[tuple[int, dict[str, float]]]] = defaultdict(list)
    for (w, t), v in grouped.items():
        by_thread[t].append((w, v))
    best_t = max(
        (t for t, d in by_thread.items() if any(w == 1 for w, _ in d)),
        key=lambda t: len(by_thread[t]), default=None,
    )
    if best_t is not None and len(by_thread[best_t]) >= 2:
        data = sorted(by_thread[best_t], key=lambda x: x[0])
        workers = [w for w, _ in data]
        t1 = next(v["per_pair_ms"] for w, v in data if w == 1)
        speedup = [t1 / v["per_pair_ms"] for _, v in data]
        eff = [s / w * 100 for s, w in zip(speedup, workers)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Strong scaling — {best_t} threads/worker "
                     f"[{rows[0].get('fft_backend', '?')}]", fontsize=13)
        ax1.plot(workers, speedup, "s-", color="tab:orange", lw=2, ms=8, label="Measured")
        ax1.plot(workers, workers, "--", color="gray", alpha=0.5, label="Ideal")
        ax1.set_xlabel("Workers (N)")
        ax1.set_ylabel("Speedup S(N) = T(1)/T(N)")
        ax1.set_title("Speedup")
        ax1.set_xticks(workers)
        ax1.legend()
        ax1.grid(alpha=0.3)

        colors = ["tab:green" if e >= 70 else "tab:orange" if e >= 50 else "tab:red"
                  for e in eff]
        ax2.bar(range(len(workers)), eff, tick_label=[str(w) for w in workers],
                color=colors, edgecolor="black", lw=0.5)
        for i, e in enumerate(eff):
            ax2.text(i, e + 2, f"{e:.0f}%", ha="center", fontsize=9, fontweight="bold")
        ax2.axhline(100, color="gray", ls="--", alpha=0.5)
        ax2.set_xlabel("Workers (N)")
        ax2.set_ylabel("Parallel efficiency E(N) = S(N)/N ×100%")
        ax2.set_title("Efficiency")
        ax2.set_ylim(0, 120)
        ax2.grid(alpha=0.3, axis="y")
        fig.tight_layout()
        p = out_dir / f"{stem}_strong_scaling.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        written.append(p)

    for p in written:
        print(f"  saved {p}")
    return written


# --- A/B compare figures ---------------------------------------------------


def plot_throughput_ab(
    csv_a: Path, csv_b: Path, out_path: Path,
    label_a: str = "A", label_b: str = "B",
) -> Path:
    """Overlay throughput-vs-cores for two backends (FFTW vs codelet). On
    I/O-bound data the two curves sit on top of each other — the honest parity."""
    ra, rb = load_scaling_csv(Path(csv_a)), load_scaling_csv(Path(csv_b))
    ga, gb = _grouped_throughput(ra), _grouped_throughput(rb)
    la = ra[0].get("fft_backend", label_a) if ra else label_a
    lb = rb[0].get("fft_backend", label_b) if rb else label_b

    fig, ax = plt.subplots(figsize=(9, 6))
    for g, lab, mk, col in ((ga, la, "o", "steelblue"), (gb, lb, "s", "tab:red")):
        pts = sorted(((v["total_cores"], v["pairs_per_s"]) for v in g.values()))
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, mk + "-", color=col, label=lab, ms=7)
    ax.set_xlabel("Total logical cores (workers × threads)")
    ax.set_ylabel("Pairs / second")
    ax.set_title("Throughput A/B — FFTW vs codelet")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def _kernel_split_ms(payload: dict[str, Any]) -> Optional[dict[str, float]]:
    """Pull the FFT vs peak-fit per-pair split out of a profile payload, if present."""
    ks = payload.get("kernel_split")
    if not ks or ks.get("xcorr_fft_ms") is None:
        return None
    return {"xcorr_fft": ks["xcorr_fft_ms"], "peak_fit": ks["peak_fit_ms"]}


def plot_budget_ab(
    payload_a: dict[str, Any], payload_b: dict[str, Any], out_path: Path,
) -> Path:
    """The money figure: per-pair budgets side by side (FFTW vs codelet). Same
    bars as :func:`plot_budget` (Read | per-pass stacked compute | Save) but two
    adjacent bars per slot, so the Cross-corr segment visibly shrinks for the
    codelet while Read/Save/Other stay fixed — and you can read off whether the
    per-pair *total* actually moves."""
    pa, pb = payload_a["passes"], payload_b["passes"]
    if len(pa) != len(pb):
        raise ValueError(f"pass count differs: A={len(pa)} B={len(pb)} — not comparable")
    ba = payload_a.get("provenance", {}).get("fft_backend", "A")
    bb = payload_b.get("provenance", {}).get("fft_backend", "B")

    slots = ["Read"] + [f"Pass {p['pass_idx']+1}\n({p['window'][0]} px)" for p in pa] + ["Save"]
    x = np.arange(len(slots))
    w = 0.38

    fig, ax = plt.subplots(figsize=(11, 6))

    def _draw(payload, passes, offset, hatch, legend):
        done: set[str] = set()  # emit each legend label at most once

        def _lab(name):
            if not legend or name in done:
                return None
            done.add(name)
            return name

        ax.bar(x[0] + offset, payload["budget_per_pair_ms"]["read"], w,
               color=_READ_COLOR, hatch=hatch, label=_lab("Read (I/O)"))
        for i, p in enumerate(passes):
            bucket = _bucket_pass(p["sections_ms"])
            bottom = 0.0
            for name, _, colour in _CATEGORIES:
                h = bucket[name]
                if h <= 0:
                    continue
                ax.bar(x[1 + i] + offset, h, w, bottom=bottom, color=colour, hatch=hatch,
                       label=_lab(name))
                bottom += h
        ax.bar(x[-1] + offset, payload["save_ms"]["mean_ms"], w,
               color=_SAVE_COLOR, hatch=hatch, label=_lab("Save (I/O)"))

    _draw(payload_a, pa, -w / 2, "", True)    # A: solid, contributes the legend
    _draw(payload_b, pb, +w / 2, "//", False)  # B: hatched

    ax.set_xticks(x)
    ax.set_xticklabels(slots)
    ax.set_ylabel("Time per pair (ms)")
    ta = payload_a["budget_per_pair_ms"]["total"]
    tb = payload_b["budget_per_pair_ms"]["total"]
    ax.set_title(
        f"Per-pair budget A/B — left/solid={ba} ({ta:.2f} ms) vs right/hatched={bb} ({tb:.2f} ms)"
    )
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Annotate the kernel FFT-vs-fit split underneath, when both arms have it —
    # that's the isolated FFT speedup, the sharpest A/B number.
    ka, kb = _kernel_split_ms(payload_a), _kernel_split_ms(payload_b)
    if ka and kb:
        txt = (f"Kernel split (thread-summed, ms/pair): "
               f"{ba} FFT={ka['xcorr_fft']:.2f} fit={ka['peak_fit']:.2f} | "
               f"{bb} FFT={kb['xcorr_fft']:.2f} fit={kb['peak_fit']:.2f}")
        ax.text(0.5, -0.12, txt, transform=ax.transAxes, ha="center", fontsize=8,
                color="#333")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path
