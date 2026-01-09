"""
Warped Image Visualization for Ensemble PIV Diagnostics

This script creates comparison figures showing how images change across PIV passes.
Useful for diagnosing Reynolds stress reduction issues related to image warping.

Usage:
    python visualize_warped_images.py <output_dir> [--particle_d <size>]

Example:
    python visualize_warped_images.py tests/rs_particle_d3/output_3pass --particle_d 3
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import tifffile


def load_warped_images(filters_dir: Path) -> dict:
    """
    Load all warped diagnostic images and stats from a filters directory.

    Parameters
    ----------
    filters_dir : Path
        Directory containing warped images and stats JSON files

    Returns
    -------
    dict
        Dictionary with keys:
        - 'original_A', 'original_B': Original (pre-warp) images
        - 'pass1_A', 'pass1_B', 'pass2_A', ...: Warped images per pass
        - 'stats': Dictionary of intensity statistics per pass
    """
    data = {
        'images': {},
        'stats': {},
    }

    # Load original/filtered images
    for frame in ['A', 'B']:
        orig_path = filters_dir / f"filtered_{frame}.tif"
        if orig_path.exists():
            data['images'][f'original_{frame}'] = tifffile.imread(str(orig_path))

    # Load warped images and stats for each pass
    for pass_idx in range(1, 10):  # Up to 10 passes
        for frame in ['A', 'B']:
            img_path = filters_dir / f"pass{pass_idx}_{frame}_warped.tif"
            if img_path.exists():
                data['images'][f'pass{pass_idx}_{frame}'] = tifffile.imread(str(img_path))

        # Load stats JSON
        stats_path = filters_dir / f"pass{pass_idx}_intensity_stats.json"
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                data['stats'][f'pass{pass_idx}'] = json.load(f)

    return data


def create_warped_comparison_figure(
    output_dir: Path,
    particle_d: float = None,
    save_path: Path = None,
) -> None:
    """
    Create figure showing how a single image changes across passes with histograms.

    Layout (matplotlib subplots):
    ┌──────────────────────────────────────────────────────────┐
    │ Row 1: Frame A images                                    │
    │ [filtered_A] [pass1_A_warped] [pass2_A_warped] [pass3_A] │
    ├──────────────────────────────────────────────────────────┤
    │ Row 2: Frame A intensity histograms                      │
    │ [hist_orig]  [hist_pass1]     [hist_pass2]    [hist_p3]  │
    ├──────────────────────────────────────────────────────────┤
    │ Row 3: Frame B images                                    │
    │ [filtered_B] [pass1_B_warped] [pass2_B_warped] [pass3_B] │
    ├──────────────────────────────────────────────────────────┤
    │ Row 4: Frame B intensity histograms                      │
    │ [hist_orig]  [hist_pass1]     [hist_pass2]    [hist_p3]  │
    └──────────────────────────────────────────────────────────┘

    Parameters
    ----------
    output_dir : Path
        Output directory containing 'filters' subdirectory with diagnostic images
    particle_d : float, optional
        Particle diameter (for title annotation)
    save_path : Path, optional
        Where to save the figure. Defaults to output_dir/warped_comparison.png
    """
    filters_dir = output_dir / "filters"
    if not filters_dir.exists():
        # Try looking in nested path structure
        for subdir in output_dir.rglob("filters"):
            filters_dir = subdir
            break

    if not filters_dir.exists():
        print(f"Error: No 'filters' directory found in {output_dir}")
        return

    print(f"Loading images from: {filters_dir}")
    data = load_warped_images(filters_dir)

    # Determine number of passes
    passes = sorted([k for k in data['images'].keys() if k.startswith('pass') and k.endswith('_A')])
    n_passes = len(passes)

    if n_passes == 0:
        print("Error: No warped images found")
        return

    # Add original if available
    has_original = 'original_A' in data['images']
    n_cols = n_passes + (1 if has_original else 0)

    print(f"Found {n_passes} passes, has_original={has_original}")

    # Create figure: 4 rows (A images, A hists, B images, B hists)
    fig, axes = plt.subplots(4, n_cols, figsize=(4 * n_cols, 12))

    # Title
    title = "Warped Image Comparison Across Passes"
    if particle_d:
        title += f" (Particle: {particle_d}px)"
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Helper to get image statistics text
    def stats_text(img):
        return f"mean={img.mean():.1f}\nstd={img.std():.1f}\nmin={img.min():.0f}\nmax={img.max():.0f}"

    # Determine global colorbar range from all images
    all_images = list(data['images'].values())
    vmin = min(img.min() for img in all_images)
    vmax = max(img.max() for img in all_images)

    col_idx = 0

    # Plot original images (if available)
    if has_original:
        for row_offset, frame in enumerate(['A', 'B']):
            row_img = row_offset * 2
            row_hist = row_img + 1

            img = data['images'][f'original_{frame}']

            # Image
            ax_img = axes[row_img, col_idx]
            im = ax_img.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
            ax_img.set_title(f"Original {frame}\n{stats_text(img)}", fontsize=10)
            ax_img.axis('off')
            plt.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

            # Histogram
            ax_hist = axes[row_hist, col_idx]
            ax_hist.hist(img.ravel(), bins=100, color='blue', alpha=0.7, density=True)
            ax_hist.axvline(img.mean(), color='red', linestyle='--', linewidth=2, label=f'mean={img.mean():.1f}')
            ax_hist.set_xlabel('Intensity')
            ax_hist.set_ylabel('Density')
            ax_hist.set_title(f"Histogram {frame}")
            ax_hist.legend(fontsize=8)

        col_idx += 1

    # Plot each pass
    for pass_key in passes:
        pass_num = int(pass_key.replace('pass', '').replace('_A', ''))

        for row_offset, frame in enumerate(['A', 'B']):
            row_img = row_offset * 2
            row_hist = row_img + 1

            img_key = f'pass{pass_num}_{frame}'
            if img_key not in data['images']:
                continue

            img = data['images'][img_key]

            # Image
            ax_img = axes[row_img, col_idx]
            im = ax_img.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
            ax_img.set_title(f"Pass {pass_num} {frame} (warped)\n{stats_text(img)}", fontsize=10)
            ax_img.axis('off')
            plt.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)

            # Histogram
            ax_hist = axes[row_hist, col_idx]
            ax_hist.hist(img.ravel(), bins=100, color='blue', alpha=0.7, density=True)
            ax_hist.axvline(img.mean(), color='red', linestyle='--', linewidth=2, label=f'mean={img.mean():.1f}')
            ax_hist.set_xlabel('Intensity')
            ax_hist.set_ylabel('Density')
            ax_hist.set_title(f"Histogram {frame}")
            ax_hist.legend(fontsize=8)

        col_idx += 1

    # Row labels
    axes[0, 0].set_ylabel("Frame A", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("Histogram A", fontsize=12)
    axes[2, 0].set_ylabel("Frame B", fontsize=12, fontweight='bold')
    axes[3, 0].set_ylabel("Histogram B", fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save figure
    if save_path is None:
        save_path = output_dir / "warped_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to: {save_path}")

    plt.close()


def create_intensity_trend_figure(
    output_dirs: list,
    particle_diameters: list,
    save_path: Path = None,
) -> None:
    """
    Create figure showing intensity trends across passes for multiple particle sizes.

    Parameters
    ----------
    output_dirs : list of Path
        List of output directories (one per particle size)
    particle_diameters : list
        List of particle diameters corresponding to output_dirs
    save_path : Path, optional
        Where to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(particle_diameters)))

    for idx, (output_dir, particle_d) in enumerate(zip(output_dirs, particle_diameters)):
        filters_dir = output_dir / "filters"
        if not filters_dir.exists():
            for subdir in output_dir.rglob("filters"):
                filters_dir = subdir
                break

        if not filters_dir.exists():
            continue

        data = load_warped_images(filters_dir)

        # Extract mean intensity per pass
        passes = []
        means_A = []
        means_B = []

        for pass_num in range(1, 10):
            stats_key = f'pass{pass_num}'
            if stats_key in data['stats']:
                stats = data['stats'][stats_key]
                passes.append(pass_num)
                means_A.append(stats['frame_A_warped']['mean'])
                means_B.append(stats['frame_B_warped']['mean'])

        if passes:
            axes[0].plot(passes, means_A, 'o-', color=colors[idx], label=f'{particle_d}px', linewidth=2, markersize=8)
            axes[1].plot(passes, means_B, 'o-', color=colors[idx], label=f'{particle_d}px', linewidth=2, markersize=8)

    axes[0].set_xlabel('Pass', fontsize=12)
    axes[0].set_ylabel('Mean Intensity', fontsize=12)
    axes[0].set_title('Frame A Mean Intensity vs Pass', fontsize=12)
    axes[0].legend(title='Particle Diameter')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Pass', fontsize=12)
    axes[1].set_ylabel('Mean Intensity', fontsize=12)
    axes[1].set_title('Frame B Mean Intensity vs Pass', fontsize=12)
    axes[1].legend(title='Particle Diameter')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Intensity Trends Across Passes by Particle Diameter', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved intensity trend figure to: {save_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize warped images across PIV passes')
    parser.add_argument('output_dir', type=str, help='Output directory containing filters subdirectory')
    parser.add_argument('--particle_d', type=float, help='Particle diameter for annotation')
    parser.add_argument('--save', type=str, help='Output path for figure')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)

    save_path = Path(args.save) if args.save else None

    create_warped_comparison_figure(
        output_dir=output_dir,
        particle_d=args.particle_d,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
