"""
Warp interpolation comparison: nearest-neighbour vs bicubic.

Creates synthetic greyscale grid images at 1 MP, 4 MP, and 25 MP, warps them
using a channel-flow (Poiseuille) velocity profile, and shows original vs
warped for each method with a near-wall zoom.

Usage:
    python scripts/warp_interpolation_comparison.py
"""

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ── Synthetic image ─────────────────────────────────────────────────────────

def make_grid_image(height: int, width: int) -> np.ndarray:
    y = np.arange(height, dtype=np.float64)
    x = np.arange(width, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)

    # 3-pixel fine stripes (heavily affected by interpolation)
    stripe_x = 0.20 * np.sin(2 * np.pi * xx / 3.0)
    stripe_y = 0.20 * np.sin(2 * np.pi * yy / 3.0)
    # 12-pixel medium bars
    med_x = 0.20 * np.sin(2 * np.pi * xx / 12.0)
    med_y = 0.20 * np.sin(2 * np.pi * yy / 12.0)
    # 48-pixel coarse bars
    coarse_x = 0.15 * np.sin(2 * np.pi * xx / 48.0)
    coarse_y = 0.15 * np.sin(2 * np.pi * yy / 48.0)

    img = 0.5 + stripe_x + stripe_y + med_x + med_y + coarse_x + coarse_y
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


# ── Displacement field ──────────────────────────────────────────────────────

def channel_flow_displacement(height, width, u_max_px=12.0):
    y_norm = np.linspace(-1, 1, height, dtype=np.float32)
    u_profile = u_max_px * (1.0 - y_norm ** 2)
    return np.broadcast_to(u_profile[:, None], (height, width)).copy()


# ── Warping ─────────────────────────────────────────────────────────────────

def build_maps(h, w):
    y_map = np.arange(h, dtype=np.float32)[:, None] * np.ones(w, dtype=np.float32)
    x_map = np.ones(h, dtype=np.float32)[:, None] * np.arange(w, dtype=np.float32)
    return y_map, x_map


def warp_nearest(img, dx, y_map, x_map):
    map_x = (x_map - dx).astype(np.float32)
    return cv2.remap(img, map_x, y_map,
                     interpolation=cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def warp_bicubic(img, dx, y_map, x_map):
    map_x = (x_map - dx).astype(np.float32)
    return cv2.remap(img, map_x, y_map,
                     interpolation=cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def time_warp(func, img, dx, y_map, x_map, n_repeats=5):
    func(img, dx, y_map, x_map)  # warm-up
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        result = func(img, dx, y_map, x_map)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return result, min(times)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    resolutions = [
        ("1 MP",   1000, 1000),
        ("4 MP",   2000, 2000),
        ("25 MP",  5000, 5000),
    ]

    methods = [
        ("Nearest Neighbour", warp_nearest,  "#e74c3c"),
        ("Bicubic",           warp_bicubic,  "#2ecc71"),
    ]

    n_res = len(resolutions)
    n_meth = len(methods)

    # Layout:  per resolution we get 2 rows (full + zoom) x 3 cols (orig, NN, bicubic)
    n_rows = n_res * 2
    n_cols = 1 + n_meth  # original + methods

    fig = plt.figure(figsize=(5.5 * n_cols, 4.0 * n_res * 2 + 1.2))

    # Outer grid: one block per resolution
    outer = GridSpec(n_res, 1, figure=fig, hspace=0.30,
                     left=0.02, right=0.98, top=0.95, bottom=0.02)

    fig.suptitle("Warp Interpolation: Nearest Neighbour vs Bicubic\n"
                 "Channel flow Poiseuille displacement (u_max = 12 px)",
                 fontsize=15, fontweight="bold", y=0.99)

    for i_res, (res_label, H, W) in enumerate(resolutions):
        print(f"\n{'='*60}")
        print(f"  {res_label}  ({H} x {W})")
        print(f"{'='*60}")

        img = make_grid_image(H, W)
        dx = channel_flow_displacement(H, W, u_max_px=12.0)
        y_map, x_map = build_maps(H, W)

        # Zoom region: bottom 10% of image (near-wall), middle 15% in x
        y_lo = int(H * 0.90)
        y_hi = H
        x_lo = int(W * 0.42)
        x_hi = int(W * 0.58)
        crop = np.s_[y_lo:y_hi, x_lo:x_hi]

        # Inner grid: 2 rows (full, zoom) x 3 cols (orig, NN, bicubic)
        inner = outer[i_res].subgridspec(2, n_cols, hspace=0.15, wspace=0.06)

        # ── Original ────────────────────────────────────────────────────
        ax_full = fig.add_subplot(inner[0, 0])
        ax_full.imshow(img, cmap="gray", vmin=0, vmax=255, aspect="equal")
        ax_full.set_title(f"Original  --  {res_label} ({H}x{W})", fontsize=10,
                          fontweight="bold")
        ax_full.set_xticks([]); ax_full.set_yticks([])

        # Draw zoom box on full image
        rect = plt.Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                              linewidth=2, edgecolor="yellow", facecolor="none",
                              linestyle="--")
        ax_full.add_patch(rect)

        ax_zoom = fig.add_subplot(inner[1, 0])
        ax_zoom.imshow(img[crop], cmap="gray", vmin=0, vmax=255,
                       interpolation="nearest", aspect="equal")
        ax_zoom.set_title("Wall zoom (original)", fontsize=9)
        ax_zoom.set_xticks([]); ax_zoom.set_yticks([])
        for spine in ax_zoom.spines.values():
            spine.set_edgecolor("yellow")
            spine.set_linewidth(2)

        # ── Each method ─────────────────────────────────────────────────
        for i_m, (m_label, m_func, m_col) in enumerate(methods):
            warped, t_ms = time_warp(m_func, img, dx, y_map, x_map)
            print(f"  {m_label:22s}  {t_ms:8.1f} ms")

            col = 1 + i_m

            # Full warped image
            ax_f = fig.add_subplot(inner[0, col])
            ax_f.imshow(warped, cmap="gray", vmin=0, vmax=255, aspect="equal")
            ax_f.set_title(f"{m_label}   [{t_ms:.1f} ms]", fontsize=10,
                           fontweight="bold", color=m_col)
            ax_f.set_xticks([]); ax_f.set_yticks([])
            for spine in ax_f.spines.values():
                spine.set_edgecolor(m_col)
                spine.set_linewidth(2.5)

            # Draw zoom box
            rect = plt.Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                                  linewidth=2, edgecolor="yellow", facecolor="none",
                                  linestyle="--")
            ax_f.add_patch(rect)

            # Wall zoom
            ax_z = fig.add_subplot(inner[1, col])
            ax_z.imshow(warped[crop], cmap="gray", vmin=0, vmax=255,
                        interpolation="nearest", aspect="equal")
            ax_z.set_title(f"Wall zoom ({m_label.lower()})", fontsize=9,
                           color=m_col)
            ax_z.set_xticks([]); ax_z.set_yticks([])
            for spine in ax_z.spines.values():
                spine.set_edgecolor(m_col)
                spine.set_linewidth(2.5)

    out = "warp_interpolation_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved -> {out}")
    plt.show()


if __name__ == "__main__":
    main()
