"""
Diagnostic: investigate predictor field going negative in passes 2-3.

Plots predictor fields (pred_x, pred_y) stored in ensemble_result.mat
to diagnose where negative values originate.
"""

import numpy as np
import scipy.io
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from pathlib import Path

MAT_PATH = Path(
    r"C:\Users\mtt1e23\OneDrive - University of Southampton\Documents"
    r"\#current_processing\4000_images_channel\planar_images"
    r"\validation_4000_kspace_predictorupdate\uncalibrated_piv\4000\Cam1\ensemble"
    r"\ensemble_result.mat"
)
OUT_DIR = MAT_PATH.parent


def load_passes(path):
    mat = scipy.io.loadmat(str(path), struct_as_record=False, squeeze_me=True)
    data = mat["ensemble_result"]
    passes = list(data.flat) if isinstance(data, np.ndarray) and data.dtype == object else [data]
    results = []
    for p in passes:
        d = {}
        for field in ["ux", "uy", "pred_x", "pred_y", "win_ctrs_x", "win_ctrs_y", "window_size"]:
            val = getattr(p, field, None)
            if val is not None:
                arr = np.asarray(val)
                if arr.size > 0:
                    d[field] = arr
                    continue
            d[field] = None
        results.append(d)
    return results


def compute_padded_centers(centers, spacing, extent):
    """Same logic as base.py _build_interpolation_grids."""
    pre = np.arange(1, centers[0] - spacing / 2, spacing)
    if pre.size == 0:
        pre = np.array([1.0])
    pre = pre - 1
    while len(pre) < 2:
        extra = pre[0] - spacing
        pre = np.concatenate([[max(0, extra)], pre])

    post = np.arange(extent, centers[-1] + spacing / 2, -spacing)
    if post.size == 0:
        post = np.array([float(extent)])
    post = post - 1
    while len(post) < 2:
        extra = post[-1] + spacing
        post = np.concatenate([post, [min(extent - 1, extra)]])

    centers_all = np.concatenate([pre, centers, post[::-1]]).astype(np.float32)
    return centers_all, len(pre), len(post)


def main():
    print("=" * 80)
    print("PREDICTOR FIELD DIAGNOSTIC")
    print("=" * 80)

    passes = load_passes(MAT_PATH)
    n_passes = len(passes)

    # ── Fig 1: Stored predictor fields (pred_x, pred_y) per pass ──
    fig1, axes1 = plt.subplots(n_passes, 3, figsize=(21, 5 * n_passes))
    if n_passes == 1:
        axes1 = axes1[np.newaxis, :]

    for pi in range(n_passes):
        p = passes[pi]
        ws = p["window_size"]
        print(f"\nPass {pi+1}: window_size={ws}, ux shape={p['ux'].shape}")

        # pred_x/pred_y 2D heatmaps
        for di, (field, label) in enumerate([("pred_x", "pred_x (ux)"), ("pred_y", "pred_y (uy)")]):
            ax = axes1[pi, di]
            if p[field] is not None:
                data = p[field]
                vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
                if vmax == 0:
                    vmax = 1
                im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
                plt.colorbar(im, ax=ax, shrink=0.7)
                neg_frac = np.sum(data < 0) / data.size * 100
                ax.set_title(f"Pass {pi+1}: {label}\nrange=[{np.nanmin(data):.3f}, {np.nanmax(data):.3f}]\n{neg_frac:.1f}% negative")
                print(f"  {field}: shape={data.shape}, "
                      f"range=[{np.nanmin(data):.3f}, {np.nanmax(data):.3f}], "
                      f"{neg_frac:.1f}% negative")
            else:
                ax.set_title(f"Pass {pi+1}: {label}\n(not stored)")
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)

        # ux field (the velocity being fed as predictor to next pass)
        ax = axes1[pi, 2]
        ux = p["ux"]
        vmax = max(abs(np.nanmin(ux)), abs(np.nanmax(ux)))
        if vmax == 0:
            vmax = 1
        im = ax.imshow(ux, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.7)
        neg_frac = np.sum(ux < 0) / ux.size * 100
        ax.set_title(f"Pass {pi+1}: ux (velocity)\nrange=[{np.nanmin(ux):.3f}, {np.nanmax(ux):.3f}]\n{neg_frac:.1f}% negative")

    fig1.suptitle("Stored predictor fields and velocity per pass", fontsize=14, y=1.01)
    fig1.tight_layout()
    fname1 = "diagnose_predictor_fields.png"
    fig1.savefig(str(OUT_DIR / fname1), dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT_DIR / fname1}")
    plt.close(fig1)

    # ── Fig 2: Edge pad vs linear extrapolation — wall profiles per pass ──
    for target_pass in range(1, n_passes):
        print(f"\n{'='*80}")
        print(f"PASS {target_pass + 1}: Comparing padding methods")
        print(f"{'='*80}")

        prev = passes[target_pass - 1]
        prev_ux = prev["ux"].astype(np.float32)
        prev_uy = prev["uy"].astype(np.float32)
        prev_ws = prev["window_size"]
        ny_prev, nx_prev = prev_ux.shape

        # Get actual window centers
        if prev.get("win_ctrs_y") is not None:
            ctrs_y = prev["win_ctrs_y"].astype(np.float32).ravel()
            ctrs_x = prev["win_ctrs_x"].astype(np.float32).ravel()
            sp_y = ctrs_y[1] - ctrs_y[0] if len(ctrs_y) > 1 else float(prev_ws[0])
            sp_x = ctrs_x[1] - ctrs_x[0] if len(ctrs_x) > 1 else float(prev_ws[1])
        else:
            print("  WARNING: No window centers stored, skipping")
            continue

        H = int(ctrs_y[-1] + prev_ws[0] // 2)
        W = int(ctrs_x[-1] + prev_ws[1] // 2)

        ctrs_y_all, n_pre_y, n_post_y = compute_padded_centers(ctrs_y, sp_y, H)
        ctrs_x_all, n_pre_x, n_post_x = compute_padded_centers(ctrs_x, sp_x, W)

        print(f"  Image: {H}x{W}, prev grid: {ny_prev}x{nx_prev}")
        print(f"  Padding: pre=({n_pre_y},{n_pre_x}), post=({n_post_y},{n_post_x})")

        # Predictor: [uy, ux] — note uy is negated in file, un-negate for image coords
        pred_raw = np.stack([-prev_uy, prev_ux], axis=-1)

        # Edge pad (production method)
        pred_edge = np.pad(pred_raw, ((n_pre_y, n_post_y), (n_pre_x, n_post_x), (0, 0)), mode="edge")

        # Build remap maps
        map_x_1d = np.interp(np.arange(W, dtype=np.float32), ctrs_x_all,
                              np.arange(len(ctrs_x_all), dtype=np.float32))
        map_y_1d = np.interp(np.arange(H, dtype=np.float32), ctrs_y_all,
                              np.arange(len(ctrs_y_all), dtype=np.float32))
        map_y_2d, map_x_2d = np.meshgrid(map_y_1d, map_x_1d, indexing="ij")
        map_x_2d = map_x_2d.astype(np.float32)
        map_y_2d = map_y_2d.astype(np.float32)

        # Remap to dense
        dense_edge = np.zeros((H, W, 2), dtype=np.float32)
        for d in range(2):
            dense_edge[..., d] = cv2.remap(
                pred_edge[..., d].astype(np.float32),
                map_x_2d, map_y_2d, cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )

        mid_col = W // 2
        pixel_y = np.arange(H)

        # Source window center values (un-padded)
        src_ux = pred_raw[:, nx_prev // 2, 1]

        # ── Plot: profiles + 2D heatmap ──
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))

        # Full vertical profile (ux component)
        ax = axes[0]
        ax.plot(dense_edge[:, mid_col, 1], pixel_y, "b-", lw=1.5, label="Edge pad")
        ax.plot(src_ux, ctrs_y, "ko", ms=3, label="Source (prev pass)")
        ax.axvline(0, color="gray", ls=":", lw=0.5)
        ax.set_xlabel("ux predictor [px]")
        ax.set_ylabel("pixel y")
        ax.set_title("Full vertical profile")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

        # Wall zoom
        ax = axes[1]
        wall_start = int(H * 0.80)
        ax.plot(dense_edge[wall_start:, mid_col, 1], pixel_y[wall_start:],
                "b-", lw=2, label="Edge pad")
        wall_mask = ctrs_y >= wall_start
        if wall_mask.any():
            ax.plot(src_ux[wall_mask], ctrs_y[wall_mask], "ko", ms=5, label="Source")
        ax.axvline(0, color="gray", ls=":", lw=0.5)
        ax.set_xlabel("ux predictor [px]")
        ax.set_ylabel("pixel y")
        ax.set_title("WALL zoom (bottom 20%)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()

        # 2D heatmap of dense predictor (ux component)
        ax = axes[2]
        data = dense_edge[..., 1]
        vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
        if vmax == 0:
            vmax = 1
        im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.7)
        neg_frac = np.sum(data < 0) / data.size * 100
        ax.set_title(f"Dense ux: Edge pad\n{neg_frac:.1f}% negative, range=[{np.nanmin(data):.3f}, {np.nanmax(data):.3f}]")

        fig.suptitle(f"Pass {target_pass+1}: Edge pad predictor field", fontsize=14)
        fig.tight_layout()
        fname = f"diagnose_pass{target_pass+1}_edge_pad.png"
        fig.savefig(str(OUT_DIR / fname), dpi=150, bbox_inches="tight")
        print(f"\n  Saved: {OUT_DIR / fname}")
        plt.close(fig)

        # Print key stats
        ux_col = dense_edge[:, mid_col, 1]
        neg_region = ux_col < 0
        print(f"\n  Edge pad:")
        print(f"    ux range: [{ux_col.min():.4f}, {ux_col.max():.4f}]")
        print(f"    Negative pixels: {neg_region.sum()}/{len(ux_col)} ({neg_region.sum()/len(ux_col)*100:.1f}%)")
        if neg_region.any():
            first_neg = np.argmax(neg_region)
            print(f"    First negative at y={first_neg} (y/H={first_neg/H:.3f})")

    print("\nDone!")


if __name__ == "__main__":
    main()
