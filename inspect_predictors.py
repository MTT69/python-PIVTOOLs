"""
Quick diagnostic: visualize predictor fields vs ux output for each pass
of a multi-pass ensemble result.

Focuses on edge behaviour to validate Method C predictor interpolation.
"""

import numpy as np
import scipy.io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

MAT_PATH = Path(
    r"C:\Users\mtt1e23\OneDrive - University of Southampton\Documents"
    r"\#current_processing\4000_images_channel\planar_images"
    r"\validation_100_kspace_predictorupdate\uncalibrated_piv\100\Cam1\ensemble"
    r"\ensemble_result.mat"
)
OUT_DIR = MAT_PATH.parent


def load_passes(path):
    mat = scipy.io.loadmat(str(path), struct_as_record=False, squeeze_me=True)
    data = mat["ensemble_result"]
    passes = list(data.flat) if isinstance(data, np.ndarray) and data.dtype == object else [data]
    results = []
    for i, p in enumerate(passes):
        d = {"pass": i + 1}
        for field in ["ux", "uy", "pred_x", "pred_y", "win_ctrs_x", "win_ctrs_y",
                       "window_size", "peakheight", "b_mask", "nan_reason"]:
            val = getattr(p, field, None)
            if val is not None:
                arr = np.asarray(val)
                if arr.size > 0:
                    d[field] = arr
                    continue
            d[field] = None
        results.append(d)
    return results


def align_predictor(pred_x, ux):
    """Find correct padding offset by brute-forcing all splits.

    pred_x is the padded version of this pass's ux, so the core
    must match ux exactly. Returns (aligned_pred, pre_y, pre_x).
    """
    ny, nx = ux.shape
    pred_ny, pred_nx = pred_x.shape
    pad_total_y = pred_ny - ny
    pad_total_x = pred_nx - nx
    best_err = np.inf
    best_pre_y, best_pre_x = 0, 0
    for try_pre_y in range(pad_total_y + 1):
        for try_pre_x in range(pad_total_x + 1):
            crop = pred_x[try_pre_y:try_pre_y + ny, try_pre_x:try_pre_x + nx]
            err = np.nanmean(np.abs(crop - ux))
            if err < best_err:
                best_err = err
                best_pre_y, best_pre_x = try_pre_y, try_pre_x
    aligned = pred_x[best_pre_y:best_pre_y + ny, best_pre_x:best_pre_x + nx]
    return aligned, best_pre_y, best_pre_x, best_err


def main():
    print(f"Loading: {MAT_PATH}")
    passes = load_passes(MAT_PATH)
    n_passes = len(passes)
    print(f"Found {n_passes} passes\n")

    for i, p in enumerate(passes):
        print(f"--- Pass {i+1} ---")
        for key in ["ux", "uy", "pred_x", "pred_y", "window_size"]:
            v = p.get(key)
            if v is not None:
                if v.ndim >= 2:
                    print(f"  {key}: shape={v.shape}, range=[{np.nanmin(v):.4f}, {np.nanmax(v):.4f}]")
                else:
                    print(f"  {key}: {v}")
            else:
                print(f"  {key}: None")

    # ── Figure 1: Vertical profiles at mid-column ──
    # predictor vs actual ux for each pass
    fig, axes = plt.subplots(2, n_passes, figsize=(7 * n_passes, 12))
    if n_passes == 1:
        axes = axes.reshape(2, 1)

    for i, p in enumerate(passes):
        ux = p["ux"]
        pred_x = p.get("pred_x")  # predictor for THIS pass (from previous pass)
        ny, nx = ux.shape
        mid_col = nx // 2

        # Compute padding offset so predictor aligns with ux grid
        # pred_x is padded ux from THIS pass — the core must match ux exactly.
        # Brute-force the correct (pre_y, pre_x) offset.
        if pred_x is not None and pred_x.ndim == 2:
            pred_ny, pred_nx = pred_x.shape
            pad_total_y = pred_ny - ny
            pad_total_x = pred_nx - nx
            best_err = np.inf
            best_pre_y, best_pre_x = 0, 0
            for try_pre_y in range(pad_total_y + 1):
                for try_pre_x in range(pad_total_x + 1):
                    crop = pred_x[try_pre_y:try_pre_y + ny, try_pre_x:try_pre_x + nx]
                    err = np.nanmean(np.abs(crop - ux))
                    if err < best_err:
                        best_err = err
                        best_pre_y, best_pre_x = try_pre_y, try_pre_x
            n_pre_y = best_pre_y
            n_post_y = pad_total_y - best_pre_y
            n_pre_x = best_pre_x
            n_post_x = pad_total_x - best_pre_x
            pred_x_aligned = pred_x[n_pre_y:n_pre_y + ny, n_pre_x:n_pre_x + nx]
            pred_mid = nx // 2
            print(f"  Pass {i+1}: pred padding: pre_y={n_pre_y}, post_y={n_post_y}, "
                  f"pre_x={n_pre_x}, post_x={n_post_x}, aligned shape={pred_x_aligned.shape}, "
                  f"alignment_err={best_err:.6f}")
        else:
            pred_x_aligned = None
            pred_mid = None

        # Top row: full vertical profile
        ax = axes[0, i]
        ax.plot(ux[:, mid_col], np.arange(ny), "b-", lw=1.5, label="ux output")
        if pred_x_aligned is not None:
            ax.plot(pred_x_aligned[:, pred_mid], np.arange(ny), "r--", lw=1.2,
                    label="pred_x (aligned)")
        ax.set_xlabel("velocity [px]")
        ax.set_ylabel("window row index")
        ax.set_title(f"Pass {i+1}: vertical profile (mid-col)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

        # Bottom row: zoom on bottom edge (wall region, last 15%)
        ax = axes[1, i]
        edge_rows = max(3, int(ny * 0.15))
        y_start = ny - edge_rows
        ax.plot(ux[y_start:, mid_col], np.arange(y_start, ny), "b-", lw=2, label="ux output")
        if pred_x_aligned is not None:
            ax.plot(pred_x_aligned[y_start:, pred_mid],
                    np.arange(y_start, ny), "r--", lw=1.5,
                    label="pred_x (aligned)")
        ax.set_xlabel("velocity [px]")
        ax.set_ylabel("window row index")
        ax.set_title(f"Pass {i+1}: WALL zoom (bottom {edge_rows} rows)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

    fig.suptitle("Predictor vs ux output per pass (mid-column vertical profiles)", fontsize=14)
    fig.tight_layout()
    out1 = OUT_DIR / "inspect_pred_vs_ux_profiles.png"
    fig.savefig(str(out1), dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out1}")
    plt.close(fig)

    # ── Figure 2: 2D maps — pred_x vs ux side by side per pass ──
    fig2, axes2 = plt.subplots(n_passes, 3, figsize=(21, 6 * n_passes))
    if n_passes == 1:
        axes2 = axes2.reshape(1, 3)

    for i, p in enumerate(passes):
        ux = p["ux"]
        pred_x = p.get("pred_x")
        ny, nx = ux.shape

        # Strip padding from predictor to align with ux
        if pred_x is not None and pred_x.ndim == 2:
            pred_aligned, _, _, _ = align_predictor(pred_x, ux)
        else:
            pred_aligned = None

        vmin = np.nanpercentile(ux, 0.5)
        vmax = np.nanpercentile(ux, 99.5)

        # Col 0: ux output
        ax = axes2[i, 0]
        im = ax.imshow(ux, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"Pass {i+1}: ux output ({ux.shape})")
        plt.colorbar(im, ax=ax, fraction=0.046)

        # Col 1: pred_x (aligned / stripped)
        ax = axes2[i, 1]
        if pred_aligned is not None:
            im = ax.imshow(pred_aligned, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_title(f"Pass {i+1}: pred_x aligned ({pred_aligned.shape})")
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.text(0.5, 0.5, "No predictor\n(pass 1)", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14)
            ax.set_title(f"Pass {i+1}: pred_x (N/A)")

        # Col 2: difference (ux - pred_x aligned)
        ax = axes2[i, 2]
        if pred_aligned is not None:
            diff = ux - pred_aligned
            dmax = max(abs(np.nanpercentile(diff, 1)), abs(np.nanpercentile(diff, 99)), 0.01)
            im = ax.imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax, aspect="auto")
            ax.set_title(f"Pass {i+1}: ux - pred_x\n(mean={np.nanmean(diff):.4f}, std={np.nanstd(diff):.4f})")
            plt.colorbar(im, ax=ax, fraction=0.046)
        else:
            ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                    ha="center", va="center", fontsize=14)
            ax.set_title(f"Pass {i+1}: difference (N/A)")

    fig2.suptitle("2D fields: ux output vs predictor per pass", fontsize=14)
    fig2.tight_layout()
    out2 = OUT_DIR / "inspect_pred_vs_ux_2d.png"
    fig2.savefig(str(out2), dpi=150, bbox_inches="tight")
    print(f"Saved: {out2}")
    plt.close(fig2)

    # ── Figure 3: Edge-focused — horizontal profiles at top/bottom edges ──
    fig3, axes3 = plt.subplots(2, n_passes, figsize=(7 * n_passes, 10))
    if n_passes == 1:
        axes3 = axes3.reshape(2, 1)

    for i, p in enumerate(passes):
        ux = p["ux"]
        pred_x = p.get("pred_x")
        ny, nx = ux.shape

        # Strip padding from predictor
        if pred_x is not None and pred_x.ndim == 2:
            pred_aligned, _, _, _ = align_predictor(pred_x, ux)
        else:
            pred_aligned = None

        # Top edge row
        ax = axes3[0, i]
        ax.plot(ux[0, :], "b-", lw=1.5, label="ux[0,:] (top row)")
        ax.plot(ux[1, :], "b--", lw=1, alpha=0.5, label="ux[1,:]")
        if pred_aligned is not None:
            ax.plot(pred_aligned[0, :], "r-", lw=1.5, label="pred_x[0,:] (top)")
            ax.plot(pred_aligned[1, :], "r--", lw=1, alpha=0.5, label="pred_x[1,:]")
        ax.set_xlabel("column index")
        ax.set_ylabel("velocity [px]")
        ax.set_title(f"Pass {i+1}: TOP edge horizontal")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # Bottom edge row
        ax = axes3[1, i]
        ax.plot(ux[-1, :], "b-", lw=1.5, label=f"ux[{ny-1},:] (bottom row)")
        ax.plot(ux[-2, :], "b--", lw=1, alpha=0.5, label=f"ux[{ny-2},:]")
        if pred_aligned is not None:
            ax.plot(pred_aligned[-1, :], "r-", lw=1.5, label=f"pred_x[{ny-1},:] (bottom)")
            ax.plot(pred_aligned[-2, :], "r--", lw=1, alpha=0.5, label=f"pred_x[{ny-2},:]")
        ax.set_xlabel("column index")
        ax.set_ylabel("velocity [px]")
        ax.set_title(f"Pass {i+1}: BOTTOM edge horizontal")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig3.suptitle("Edge horizontal profiles: predictor vs ux output", fontsize=14)
    fig3.tight_layout()
    out3 = OUT_DIR / "inspect_pred_vs_ux_edges.png"
    fig3.savefig(str(out3), dpi=150, bbox_inches="tight")
    print(f"Saved: {out3}")
    plt.close(fig3)

    print("\nDone!")


if __name__ == "__main__":
    main()
