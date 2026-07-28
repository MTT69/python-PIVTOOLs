"""Correlation-quality diagnostics over instantaneous PIV vector files.

Compute layer + figure rendering shared by:
  - ``manual_tools/correlation_quality_report.py`` (ad-hoc CLI)
  - the ``correlation_quality`` statistic in ``instantaneous_statistics.py``
  - the ``/backend/plot/plot_corr_quality`` viewer endpoint

Aggregates the per-window quality channels stored in per-frame ``.mat``
files (``peak_mag``, ``peak_ratio``, ``nan_mask``, ``nan_reason``,
``b_mask``) into per-frame time series and spatial maps. Pure aggregation —
no images are read and no correlation is recomputed. The quality channels
exist only in UNCALIBRATED vector files (calibrated files carry just
ux/uy/b_mask), so sweeps must point at the uncalibrated data dir.

Design spec: docs/superpowers/specs/2026-07-28-correlation-quality-report-design.md

Conventions (verified against save_results.py / cpu_instantaneous.py):
    b_mask True  = statically masked window (excluded from every denominator)
    nan_mask     = the honest failure record (ux may be infilled)
    invariant    (nan_reason != 0) == nan_mask

This module does not force a matplotlib backend — callers (GUI, CLI) set
Agg themselves before importing pyplot-using entry points.
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib.colors import LogNorm

from pivtools_cli.piv.piv_backend.nan_reason_codes import (
    NAN_MASKED,
    NAN_REASON_LABELS,
    NAN_VALID,
)

logger = logging.getLogger(__name__)

# Frame files look like 00001.mat or B00001.mat; coordinates.mat is excluded.
FRAME_FILE_RE = re.compile(r"^B?(\d+)\.mat$")


# ===================== compute layer =======================================


@dataclass
class FrameRecord:
    """Per-frame quality planes and scalars extracted from one .mat file."""

    frame: int
    path: str
    peak_mag: np.ndarray  # (ny, nx) float32, NaN where invalid
    nan_mask: np.ndarray  # (ny, nx) bool, True = vector invalidated
    b_mask: np.ndarray  # (ny, nx) bool, True = statically masked
    reason_counts: Dict[int, int]  # nan_reason code -> count (this frame)
    peak_ratio: Optional[np.ndarray]  # (ny, nx) float32, or None if absent
    error: Optional[str] = None  # set instead of data on read failure


@dataclass
class Aggregates:
    """Everything the figures and downstream consumers need, reduced over frames."""

    frames: np.ndarray  # (T,) int
    mean_peak_mag: np.ndarray  # (T,) float64
    nan_pct: np.ndarray  # (T,) float64
    median_peak_ratio: Optional[np.ndarray]  # (T,) float64 or None
    peak_mag_map: np.ndarray  # (ny, nx) time-mean, NaN on masked
    nan_pct_map: np.ndarray  # (ny, nx) % of frames NaN, NaN on masked
    ratio_map: Optional[np.ndarray]  # (ny, nx) time-median or None
    reason_codes: np.ndarray  # (C,) int, sorted, excludes -1 and 0
    reason_counts: np.ndarray  # (C, T) int
    b_mask: np.ndarray  # (ny, nx) bool
    n_unmasked: int
    skipped: List[str]  # paths that failed to read


def extract_frame_record(mat_path: Path, pass_idx: int) -> FrameRecord:
    """Load one vector file and pull out the quality planes for one pass.

    Never raises on a bad file — returns a record with ``error`` set so a
    single corrupt frame cannot kill an executor map. All other failure
    modes (missing field, empty pass) are real errors and do raise.
    """
    frame = int(FRAME_FILE_RE.match(Path(mat_path).name).group(1))
    try:
        mat = scipy.io.loadmat(
            str(mat_path), struct_as_record=False, squeeze_me=True
        )
    except Exception as exc:  # corrupt/unreadable file — skip, visibly
        return FrameRecord(
            frame=frame,
            path=str(mat_path),
            peak_mag=np.empty(0),
            nan_mask=np.empty(0, dtype=bool),
            b_mask=np.empty(0, dtype=bool),
            reason_counts={},
            peak_ratio=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    passes = np.atleast_1d(mat["piv_result"])
    p = passes[pass_idx]

    if not hasattr(p, "peak_mag"):
        raise ValueError(
            f"{Path(mat_path).name}: pass {pass_idx + 1} has no peak_mag "
            "field — correlation quality needs UNCALIBRATED vector files "
            "(calibrated files carry only ux/uy/b_mask)"
        )
    peak_mag = np.asarray(p.peak_mag, dtype=np.float32)
    if peak_mag.size == 0:
        raise ValueError(
            f"{Path(mat_path).name}: pass {pass_idx + 1} has empty peak_mag"
        )
    nan_mask = np.asarray(p.nan_mask).astype(bool)
    b_mask = np.asarray(p.b_mask).astype(bool)

    nan_reason = np.asarray(p.nan_reason)
    codes, counts = np.unique(nan_reason, return_counts=True)
    reason_counts = {int(c): int(n) for c, n in zip(codes, counts)}

    peak_ratio: Optional[np.ndarray] = None
    ratio_raw = getattr(p, "peak_ratio", None)
    if ratio_raw is not None:
        ratio_arr = np.asarray(ratio_raw, dtype=np.float32)
        if ratio_arr.size > 0:
            peak_ratio = ratio_arr

    return FrameRecord(
        frame=frame,
        path=str(mat_path),
        peak_mag=peak_mag,
        nan_mask=nan_mask,
        b_mask=b_mask,
        reason_counts=reason_counts,
        peak_ratio=peak_ratio,
    )


def aggregate(records: List[FrameRecord]) -> Aggregates:
    """Reduce per-frame records to time series and spatial maps.

    Masked windows (b_mask True) are excluded from every denominator; the
    maps carry NaN there so plots can render them distinctly.
    """
    skipped = [r.path for r in records if r.error is not None]
    for r in records:
        if r.error is not None:
            logger.warning("skipped %s (%s)", r.path, r.error)
    good = sorted(
        (r for r in records if r.error is None), key=lambda r: r.frame
    )
    if not good:
        raise RuntimeError("no readable frame files — nothing to aggregate")

    shape = good[0].peak_mag.shape
    for r in good:
        if r.peak_mag.shape != shape:
            raise ValueError(
                f"grid shape mismatch: {r.path} has {r.peak_mag.shape}, "
                f"expected {shape} (from {good[0].path})"
            )

    b_mask = good[0].b_mask
    unmasked = ~b_mask
    n_unmasked = int(unmasked.sum())
    if n_unmasked == 0:
        raise RuntimeError("every window is masked — nothing to report")

    T = len(good)
    frames = np.array([r.frame for r in good], dtype=np.int64)

    have_ratio = all(r.peak_ratio is not None for r in good)
    if not have_ratio and any(r.peak_ratio is not None for r in good):
        # Mixed presence would mean frames from different runs — refuse.
        raise ValueError(
            "peak_ratio present in some frames but not others — "
            "directory mixes runs?"
        )

    # Stacks: ~(T, ny, nx) float32; 2100 x 683 x 9 ~ 50 MB each. Acceptable
    # per the spec; a streaming quantile is the swap-in point if a future
    # dataset outgrows this.
    peak_stack = np.stack([r.peak_mag for r in good])
    peak_stack[:, b_mask] = np.nan
    nan_stack = np.stack([r.nan_mask & unmasked for r in good])

    with warnings.catch_warnings():
        # all-NaN window columns (e.g. permanently dead windows) are
        # expected — the NaN result is the correct answer for the map.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean_peak_mag = np.nanmean(
            peak_stack.reshape(T, -1), axis=1
        ).astype(np.float64)
        peak_mag_map = np.nanmean(peak_stack, axis=0)

    nan_pct = 100.0 * nan_stack.reshape(T, -1).sum(axis=1) / n_unmasked
    nan_pct_map = 100.0 * nan_stack.sum(axis=0) / T
    nan_pct_map = np.where(unmasked, nan_pct_map, np.nan)

    median_peak_ratio: Optional[np.ndarray] = None
    ratio_map: Optional[np.ndarray] = None
    if have_ratio:
        ratio_stack = np.stack([r.peak_ratio for r in good])
        # Ratios are only meaningful where positive and finite (dead
        # windows carry 0/NaN); log-scale plots require > 0 anyway.
        ratio_stack = np.where(
            np.isfinite(ratio_stack) & (ratio_stack > 0) & unmasked,
            ratio_stack,
            np.nan,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            median_peak_ratio = np.nanmedian(
                ratio_stack.reshape(T, -1), axis=1
            ).astype(np.float64)
            ratio_map = np.nanmedian(ratio_stack, axis=0)

    # nan_reason breakdown: all codes seen, minus masked (-1, static) and
    # valid (0).
    all_codes = sorted(
        {
            c
            for r in good
            for c in r.reason_counts
            if c not in (NAN_MASKED, NAN_VALID)
        }
    )
    reason_codes = np.array(all_codes, dtype=np.int64)
    reason_counts = np.zeros((len(all_codes), T), dtype=np.int64)
    for t, r in enumerate(good):
        for ci, c in enumerate(all_codes):
            reason_counts[ci, t] = r.reason_counts.get(c, 0)

    return Aggregates(
        frames=frames,
        mean_peak_mag=mean_peak_mag,
        nan_pct=nan_pct,
        median_peak_ratio=median_peak_ratio,
        peak_mag_map=peak_mag_map,
        nan_pct_map=nan_pct_map,
        ratio_map=ratio_map,
        reason_codes=reason_codes,
        reason_counts=reason_counts,
        b_mask=b_mask,
        n_unmasked=n_unmasked,
        skipped=skipped,
    )


# ===================== discovery / probing =================================


def discover_frame_files(vector_dir: Path) -> List[Path]:
    """Find per-frame .mat files (B00001.mat or 00001.mat style)."""
    files = [
        p for p in Path(vector_dir).iterdir() if FRAME_FILE_RE.match(p.name)
    ]
    if not files:
        raise FileNotFoundError(
            f"no frame files matching B?NNNNN.mat in {vector_dir}"
        )
    return sorted(files, key=lambda p: int(FRAME_FILE_RE.match(p.name).group(1)))


def probe_passes(first_file: Path) -> List[int]:
    """Return 0-based indices of non-empty passes in the first frame file."""
    mat = scipy.io.loadmat(
        str(first_file), struct_as_record=False, squeeze_me=True
    )
    passes = np.atleast_1d(mat["piv_result"])
    return [
        i for i, p in enumerate(passes) if np.asarray(p.ux).size > 0
    ]


def load_timeseries_mat(path: Path, run: int) -> Aggregates:
    """Load one run's time series from corr_quality_timeseries.mat.

    Returns an ``Aggregates`` with the spatial-map fields empty (the maps
    live in mean_stats.mat, not in the time-series file) — sufficient for
    ``plot_timeseries`` and ``plot_nan_reasons``.

    Raises ``ValueError`` when the requested 1-based run has no data,
    naming the runs that do.
    """
    mat = scipy.io.loadmat(str(path), struct_as_record=False, squeeze_me=True)
    struct = np.atleast_1d(mat["corr_quality"])
    available = [
        i + 1
        for i, e in enumerate(struct)
        if np.asarray(e.frames).size > 0
    ]
    if run < 1 or run > struct.size or run not in available:
        raise ValueError(
            f"correlation quality has no data for run {run}; "
            f"available runs: {available}"
        )
    entry = struct[run - 1]

    median_raw = np.asarray(entry.median_peak_ratio, dtype=np.float64).ravel()
    median = median_raw if median_raw.size > 0 else None

    return Aggregates(
        frames=np.asarray(entry.frames, dtype=np.int64).ravel(),
        mean_peak_mag=np.asarray(entry.mean_peak_mag, dtype=np.float64).ravel(),
        nan_pct=np.asarray(entry.nan_pct, dtype=np.float64).ravel(),
        median_peak_ratio=median,
        peak_mag_map=np.empty((0, 0)),
        nan_pct_map=np.empty((0, 0)),
        ratio_map=None,
        reason_codes=np.atleast_1d(
            np.asarray(entry.reason_codes, dtype=np.int64).squeeze()
        ),
        reason_counts=np.atleast_2d(np.asarray(entry.reason_counts, dtype=np.int64)),
        b_mask=np.empty((0, 0), dtype=bool),
        n_unmasked=int(entry.n_unmasked),
        skipped=[],
    )


# ===================== plotting ============================================


def _masked_cmap(name: str):
    """Colormap rendering NaN (masked windows) as grey."""
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(color="0.6")
    return cmap


def _grid_extent(win_x: np.ndarray, win_y: np.ndarray):
    """imshow extent (left, right, bottom, top) for y-down window centres."""
    return [win_x[0], win_x[-1], win_y[-1], win_y[0]]


def plot_timeseries(agg: Aggregates, title: str, out: Path) -> None:
    n_panels = 2 if agg.median_peak_ratio is None else 3
    fig, axes = plt.subplots(
        n_panels, 1, figsize=(12, 2.8 * n_panels), sharex=True
    )
    axes[0].plot(agg.frames, agg.mean_peak_mag, lw=0.6, color="tab:blue")
    axes[0].set_ylabel("mean peak_mag\n[corr counts, a.u.]")
    axes[0].set_title(title)

    axes[1].plot(agg.frames, agg.nan_pct, lw=0.6, color="tab:red")
    axes[1].set_ylabel(f"NaN % of {agg.n_unmasked}\nunmasked windows")

    if agg.median_peak_ratio is not None:
        axes[2].plot(
            agg.frames, agg.median_peak_ratio, lw=0.6, color="tab:green"
        )
        axes[2].set_yscale("log")
        axes[2].set_ylabel("median peak_ratio\n[peak1/peak2]")

    axes[-1].set_xlabel("frame number")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, format="png")
    plt.close(fig)


def plot_spatial_corr(
    agg: Aggregates,
    win_x: np.ndarray,
    win_y: np.ndarray,
    title: str,
    out: Path,
) -> None:
    n_panels = 1 if agg.ratio_map is None else 2
    fig, axes = plt.subplots(
        1, n_panels, figsize=(4.5 * n_panels, 10), squeeze=False
    )
    axes = axes[0]
    ext = _grid_extent(win_x, win_y)

    im0 = axes[0].imshow(
        agg.peak_mag_map,
        extent=ext,
        aspect="auto",
        cmap=_masked_cmap("viridis"),
        origin="upper",
    )
    axes[0].set_title("time-mean peak_mag")
    fig.colorbar(im0, ax=axes[0], label="corr counts [a.u.]")

    if agg.ratio_map is not None:
        finite = agg.ratio_map[np.isfinite(agg.ratio_map)]
        im1 = axes[1].imshow(
            agg.ratio_map,
            extent=ext,
            aspect="auto",
            cmap=_masked_cmap("viridis"),
            origin="upper",
            norm=LogNorm(vmin=finite.min(), vmax=finite.max()),
        )
        axes[1].set_title("per-window median peak_ratio")
        fig.colorbar(im1, ax=axes[1], label="peak1/peak2")

    for ax in axes:
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px, image y-down]")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_spatial_nan(
    agg: Aggregates,
    win_x: np.ndarray,
    win_y: np.ndarray,
    title: str,
    out: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 10))
    im = ax.imshow(
        agg.nan_pct_map,
        extent=_grid_extent(win_x, win_y),
        aspect="auto",
        cmap=_masked_cmap("magma"),
        origin="upper",
    )
    ax.set_title(f"{title}\nNaN % per window (grey = statically masked)")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px, image y-down]")
    fig.colorbar(im, ax=ax, label="% of frames NaN")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_nan_reasons(agg: Aggregates, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    if agg.reason_codes.size > 0:
        labels = [
            f"{c}: {NAN_REASON_LABELS.get(int(c), 'unknown code')}"
            for c in agg.reason_codes
        ]
        ax.stackplot(
            agg.frames, agg.reason_counts, labels=labels, alpha=0.85
        )
        ax.legend(loc="upper right", fontsize=8)
    else:
        ax.text(
            0.5,
            0.5,
            "no invalidated vectors in any frame",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_xlabel("frame number")
    ax.set_ylabel("invalidated vectors [count]")
    ax.set_title(f"{title} — nan_reason breakdown")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, format="png")
    plt.close(fig)
