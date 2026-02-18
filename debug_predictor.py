"""
Diagnostic script: compare ux vs pred_x across ensemble passes.

pred_x[pass i] stores the PADDED output ux of pass i — i.e., what will
become the predictor fed into pass i+1 after interpolation to the finer grid.

So the meaningful comparison for convergence is:
  - pred_x[pass i-1] (on coarser grid) → interpolated to pass i's grid → vs ux[pass i]

This script shows:
  1. ux and pred_x side-by-side for each pass (with shapes)
  2. The center crop of pred_x overlaid on ux (since pred_x includes padding)
  3. Cross-pass convergence: pred_x[pass i-1] upscaled vs ux[pass i]
"""

import numpy as np
import scipy.io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path


MAT_PATH = Path(
    r"C:\Users\mtt1e23\OneDrive - University of Southampton\Documents"
    r"\#current_processing\4000_images_channel\planar_images"
    r"\validation_first_kspace_4000\uncalibrated_piv\4000\Cam1\ensemble"
    r"\ensemble_result.mat"
)


def load_ensemble(path):
    """Load ensemble_result.mat keeping raw saved values (no sign flipping)."""
    mat = scipy.io.loadmat(str(path), struct_as_record=False, squeeze_me=True)
    data = mat["ensemble_result"]

    passes = []
    if isinstance(data, np.ndarray) and data.dtype == object:
        for s in data.flat:
            passes.append(s)
    else:
        passes.append(data)
    return passes


def safe_get(struct, name):
    """Get field from struct, return None if empty."""
    val = getattr(struct, name, None)
    if val is None:
        return None
    arr = np.asarray(val, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr


def main():
    print(f"Loading: {MAT_PATH}")
    passes = load_ensemble(MAT_PATH)
    n_passes = len(passes)
    print(f"Found {n_passes} passes\n")

    # ── 1. Summary table ─────────────────────────────────────────────
    print("=" * 80)
    print("FIELD SUMMARY PER PASS")
    print("=" * 80)
    for i, p in enumerate(passes):
        ux = safe_get(p, "ux")
        uy = safe_get(p, "uy")
        pred_x = safe_get(p, "pred_x")
        pred_y = safe_get(p, "pred_y")
        ws = getattr(p, "window_size", None)
        print(f"\nPass {i + 1}:")
        print(f"  window_size = {np.asarray(ws).flatten() if ws is not None else 'N/A'}")
        if ux is not None:
            print(f"  ux      shape={str(ux.shape):>20s}  range=[{ux.min():.4f}, {ux.max():.4f}]  mean={np.nanmean(ux):.4f}")
        else:
            print("  ux      None")
        if uy is not None:
            print(f"  uy      shape={str(uy.shape):>20s}  range=[{uy.min():.4f}, {uy.max():.4f}]  mean={np.nanmean(uy):.4f}")
        else:
            print("  uy      None")
        if pred_x is not None:
            print(f"  pred_x  shape={str(pred_x.shape):>20s}  range=[{pred_x.min():.4f}, {pred_x.max():.4f}]  mean={np.nanmean(pred_x):.4f}")
        else:
            print("  pred_x  None")
        if pred_y is not None:
            print(f"  pred_y  shape={str(pred_y.shape):>20s}  range=[{pred_y.min():.4f}, {pred_y.max():.4f}]  mean={np.nanmean(pred_y):.4f}")
        else:
            print("  pred_y  None")

        # Show the padding amounts (difference between pred_x and ux shapes)
        if ux is not None and pred_x is not None:
            pad_y = pred_x.shape[0] - ux.shape[0]
            pad_x = pred_x.shape[1] - ux.shape[1]
            print(f"  padding (pred - ux): dy={pad_y}, dx={pad_x}")

    # ── 2. Per-pass: ux vs pred_x (center crop) ──────────────────────
    print("\n" + "=" * 80)
    print("FIGURE 1: ux vs pred_x (center-cropped) per pass")
    print("=" * 80)

    fig1, axes1 = plt.subplots(n_passes, 4, figsize=(20, 5 * n_passes))
    if n_passes == 1:
        axes1 = axes1[np.newaxis, :]

    for i, p in enumerate(passes):
        ux = safe_get(p, "ux")
        pred_x = safe_get(p, "pred_x")

        if ux is None:
            for ax in axes1[i]:
                ax.set_title(f"Pass {i+1}: no data")
                ax.axis("off")
            continue

        # Center-crop pred_x to match ux shape
        if pred_x is not None and pred_x.shape != ux.shape:
            dy = pred_x.shape[0] - ux.shape[0]
            dx = pred_x.shape[1] - ux.shape[1]
            pre_y = dy // 2
            pre_x = dx // 2
            pred_x_crop = pred_x[pre_y:pre_y + ux.shape[0], pre_x:pre_x + ux.shape[1]]
        elif pred_x is not None:
            pred_x_crop = pred_x
        else:
            pred_x_crop = None

        vmin = np.nanpercentile(ux, 1)
        vmax = np.nanpercentile(ux, 99)

        im0 = axes1[i, 0].imshow(ux, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        axes1[i, 0].set_title(f"Pass {i+1}: ux  {ux.shape}")
        plt.colorbar(im0, ax=axes1[i, 0], fraction=0.046)

        if pred_x_crop is not None:
            im1 = axes1[i, 1].imshow(pred_x_crop, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
            axes1[i, 1].set_title(f"Pass {i+1}: pred_x (cropped) {pred_x.shape}→{pred_x_crop.shape}")
            plt.colorbar(im1, ax=axes1[i, 1], fraction=0.046)

            diff = ux - pred_x_crop
            dmax = max(abs(np.nanpercentile(diff, 1)), abs(np.nanpercentile(diff, 99)))
            if dmax < 1e-10:
                dmax = 1.0
            im2 = axes1[i, 2].imshow(diff, cmap="RdBu_r", vmin=-dmax, vmax=dmax, aspect="auto")
            axes1[i, 2].set_title(f"Pass {i+1}: ux - pred_x  (max|diff|={np.nanmax(np.abs(diff)):.4f})")
            plt.colorbar(im2, ax=axes1[i, 2], fraction=0.046)

            # Scatter: pred_x vs ux
            mask = np.isfinite(ux) & np.isfinite(pred_x_crop)
            axes1[i, 3].scatter(pred_x_crop[mask].ravel()[::3], ux[mask].ravel()[::3],
                                s=1, alpha=0.3, c="steelblue")
            lims = [min(vmin, np.nanmin(pred_x_crop)), max(vmax, np.nanmax(pred_x_crop))]
            axes1[i, 3].plot(lims, lims, "k--", lw=0.8, label="1:1")
            axes1[i, 3].set_xlabel("pred_x (center crop)")
            axes1[i, 3].set_ylabel("ux")
            axes1[i, 3].set_title(f"Pass {i+1}: ux vs pred_x scatter")
            axes1[i, 3].legend()
            axes1[i, 3].set_aspect("equal")
        else:
            for j in [1, 2, 3]:
                axes1[i, j].set_title(f"Pass {i+1}: pred_x = None")
                axes1[i, j].axis("off")

    fig1.suptitle("Per-pass: ux vs pred_x (center-cropped to same shape)", fontsize=14, y=1.01)
    fig1.tight_layout()
    out_dir = MAT_PATH.parent
    fig1.savefig(str(out_dir / "debug_fig1_ux_vs_predx.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_dir / 'debug_fig1_ux_vs_predx.png'}")

    # ── 3. Cross-pass convergence ─────────────────────────────────────
    # pred_x[pass i] is the padded output of pass i.
    # When it becomes the predictor for pass i+1, it gets interpolated
    # from the coarser grid to the finer grid.
    # Here we do a simple upscale of ux[pass i-1] to pass i's grid
    # to show the "prediction error" at each pass.
    print("\n" + "=" * 80)
    print("FIGURE 2: Cross-pass convergence (ux[pass i-1] upscaled vs ux[pass i])")
    print("=" * 80)

    if n_passes > 1:
        fig2, axes2 = plt.subplots(n_passes - 1, 4, figsize=(20, 5 * (n_passes - 1)))
        if n_passes == 2:
            axes2 = axes2[np.newaxis, :]

        for i in range(1, n_passes):
            ux_prev = safe_get(passes[i - 1], "ux")
            ux_curr = safe_get(passes[i], "ux")

            if ux_prev is None or ux_curr is None:
                for ax in axes2[i - 1]:
                    ax.set_title(f"Pass {i}→{i+1}: no data")
                    ax.axis("off")
                continue

            # Interpolate ux_prev to ux_curr's grid
            ny_prev, nx_prev = ux_prev.shape
            ny_curr, nx_curr = ux_curr.shape

            y_prev = np.arange(ny_prev, dtype=np.float64)
            x_prev = np.arange(nx_prev, dtype=np.float64)
            interp = RegularGridInterpolator(
                (y_prev, x_prev), ux_prev.astype(np.float64),
                method="linear", bounds_error=False, fill_value=None
            )

            y_curr = np.linspace(0, ny_prev - 1, ny_curr)
            x_curr = np.linspace(0, nx_prev - 1, nx_curr)
            yy, xx = np.meshgrid(y_curr, x_curr, indexing="ij")
            ux_prev_upscaled = interp((yy, xx)).astype(np.float32)

            vmin = np.nanpercentile(ux_curr, 1)
            vmax = np.nanpercentile(ux_curr, 99)

            im0 = axes2[i - 1, 0].imshow(ux_prev_upscaled, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
            axes2[i - 1, 0].set_title(f"ux[pass {i}] upscaled to pass {i+1} grid  ({ux_prev.shape}→{ux_curr.shape})")
            plt.colorbar(im0, ax=axes2[i - 1, 0], fraction=0.046)

            im1 = axes2[i - 1, 1].imshow(ux_curr, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
            axes2[i - 1, 1].set_title(f"ux[pass {i+1}]  {ux_curr.shape}")
            plt.colorbar(im1, ax=axes2[i - 1, 1], fraction=0.046)

            residual = ux_curr - ux_prev_upscaled
            rmax = max(abs(np.nanpercentile(residual, 1)), abs(np.nanpercentile(residual, 99)))
            if rmax < 1e-10:
                rmax = 1.0
            im2 = axes2[i - 1, 2].imshow(residual, cmap="RdBu_r", vmin=-rmax, vmax=rmax, aspect="auto")
            axes2[i - 1, 2].set_title(
                f"Residual: ux[{i+1}] - upscaled(ux[{i}])\n"
                f"mean={np.nanmean(residual):.4f}  std={np.nanstd(residual):.4f}  "
                f"max|r|={np.nanmax(np.abs(residual)):.4f}"
            )
            plt.colorbar(im2, ax=axes2[i - 1, 2], fraction=0.046)

            mask = np.isfinite(ux_curr) & np.isfinite(ux_prev_upscaled)
            axes2[i - 1, 3].scatter(
                ux_prev_upscaled[mask].ravel()[::3], ux_curr[mask].ravel()[::3],
                s=1, alpha=0.3, c="steelblue"
            )
            lims = [min(vmin, np.nanmin(ux_prev_upscaled)), max(vmax, np.nanmax(ux_prev_upscaled))]
            axes2[i - 1, 3].plot(lims, lims, "k--", lw=0.8, label="1:1")
            axes2[i - 1, 3].set_xlabel(f"ux[pass {i}] upscaled")
            axes2[i - 1, 3].set_ylabel(f"ux[pass {i+1}]")
            axes2[i - 1, 3].set_title(f"Convergence scatter: pass {i}→{i+1}")
            axes2[i - 1, 3].legend()
            axes2[i - 1, 3].set_aspect("equal")

        fig2.suptitle("Cross-pass convergence: predictor (previous ux upscaled) vs output ux", fontsize=14, y=1.01)
        fig2.tight_layout()
        fig2.savefig(str(out_dir / "debug_fig2_convergence.png"), dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_dir / 'debug_fig2_convergence.png'}")

    # ── 4. Line profiles through center ───────────────────────────────
    print("\n" + "=" * 80)
    print("FIGURE 3: Horizontal & vertical line profiles through field center")
    print("=" * 80)

    fig3, axes3 = plt.subplots(2, n_passes, figsize=(7 * n_passes, 10))
    if n_passes == 1:
        axes3 = axes3[:, np.newaxis]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for i, p in enumerate(passes):
        ux = safe_get(p, "ux")
        pred_x = safe_get(p, "pred_x")
        if ux is None:
            continue

        mid_row = ux.shape[0] // 2
        mid_col = ux.shape[1] // 2

        # Center crop pred_x
        if pred_x is not None and pred_x.shape != ux.shape:
            dy = pred_x.shape[0] - ux.shape[0]
            dx = pred_x.shape[1] - ux.shape[1]
            pre_y = dy // 2
            pre_x = dx // 2
            pred_x_crop = pred_x[pre_y:pre_y + ux.shape[0], pre_x:pre_x + ux.shape[1]]
        elif pred_x is not None:
            pred_x_crop = pred_x
        else:
            pred_x_crop = None

        # Horizontal profile at mid-row
        axes3[0, i].plot(ux[mid_row, :], label="ux", lw=1.5, color=colors[0])
        if pred_x_crop is not None:
            axes3[0, i].plot(pred_x_crop[mid_row, :], label="pred_x (crop)", lw=1.5, ls="--", color=colors[1])
        axes3[0, i].set_title(f"Pass {i+1}: horizontal profile (row {mid_row})")
        axes3[0, i].set_xlabel("column index")
        axes3[0, i].set_ylabel("velocity [px]")
        axes3[0, i].legend()
        axes3[0, i].grid(True, alpha=0.3)

        # Vertical profile at mid-col
        axes3[1, i].plot(ux[:, mid_col], label="ux", lw=1.5, color=colors[0])
        if pred_x_crop is not None:
            axes3[1, i].plot(pred_x_crop[:, mid_col], label="pred_x (crop)", lw=1.5, ls="--", color=colors[1])
        axes3[1, i].set_title(f"Pass {i+1}: vertical profile (col {mid_col})")
        axes3[1, i].set_xlabel("row index")
        axes3[1, i].set_ylabel("velocity [px]")
        axes3[1, i].legend()
        axes3[1, i].grid(True, alpha=0.3)

    fig3.suptitle("Line profiles: ux vs pred_x (center-cropped)", fontsize=14)
    fig3.tight_layout()
    fig3.savefig(str(out_dir / "debug_fig3_profiles.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_dir / 'debug_fig3_profiles.png'}")

    # ── 5. Edge artefact check ────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FIGURE 4: Edge region analysis (top/bottom/left/right 5 rows/cols)")
    print("=" * 80)

    fig4, axes4 = plt.subplots(n_passes, 2, figsize=(14, 5 * n_passes))
    if n_passes == 1:
        axes4 = axes4[np.newaxis, :]

    for i, p in enumerate(passes):
        ux = safe_get(p, "ux")
        pred_x = safe_get(p, "pred_x")
        if ux is None:
            continue

        # pred_x full (with padding) — show padding ring
        if pred_x is not None:
            im0 = axes4[i, 0].imshow(pred_x, cmap="RdBu_r", aspect="auto")
            axes4[i, 0].set_title(f"Pass {i+1}: pred_x FULL (incl padding) {pred_x.shape}")
            plt.colorbar(im0, ax=axes4[i, 0], fraction=0.046)
            # Draw rectangle showing where ux sits inside pred_x
            dy = pred_x.shape[0] - ux.shape[0]
            dx = pred_x.shape[1] - ux.shape[1]
            pre_y = dy // 2
            pre_x = dx // 2
            from matplotlib.patches import Rectangle
            rect = Rectangle((pre_x - 0.5, pre_y - 0.5), ux.shape[1], ux.shape[0],
                              linewidth=2, edgecolor="lime", facecolor="none", linestyle="--")
            axes4[i, 0].add_patch(rect)
            axes4[i, 0].legend(["ux region"], loc="upper right")
        else:
            axes4[i, 0].set_title(f"Pass {i+1}: pred_x = None")
            axes4[i, 0].axis("off")

        # ux with edge stats annotation
        im1 = axes4[i, 1].imshow(ux, cmap="RdBu_r", aspect="auto")
        axes4[i, 1].set_title(f"Pass {i+1}: ux {ux.shape}")
        plt.colorbar(im1, ax=axes4[i, 1], fraction=0.046)

        # Print edge stats
        edge = 5
        if ux.shape[0] > 2 * edge and ux.shape[1] > 2 * edge:
            top = ux[:edge, :]
            bot = ux[-edge:, :]
            left = ux[:, :edge]
            right = ux[:, -edge:]
            interior = ux[edge:-edge, edge:-edge]
            print(f"  Pass {i+1} edge stats (5-row/col border):")
            print(f"    top:      mean={np.nanmean(top):.4f}  std={np.nanstd(top):.4f}")
            print(f"    bottom:   mean={np.nanmean(bot):.4f}  std={np.nanstd(bot):.4f}")
            print(f"    left:     mean={np.nanmean(left):.4f}  std={np.nanstd(left):.4f}")
            print(f"    right:    mean={np.nanmean(right):.4f}  std={np.nanstd(right):.4f}")
            print(f"    interior: mean={np.nanmean(interior):.4f}  std={np.nanstd(interior):.4f}")

    fig4.suptitle("Edge artefact analysis: pred_x padding vs ux", fontsize=14)
    fig4.tight_layout()
    fig4.savefig(str(out_dir / "debug_fig4_edges.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_dir / 'debug_fig4_edges.png'}")

    plt.close("all")
    print("\nDone. All figures saved to:", out_dir)


if __name__ == "__main__":
    main()
