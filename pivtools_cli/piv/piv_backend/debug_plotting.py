"""
Debug plotting utilities for PIV correlation planes.

This module contains visualization functions for debugging
PIV correlation results.
"""

import logging
import numpy as np
from pathlib import Path


def plot_corr_planes(
    corr_planes: np.ndarray,
    n_win_y: int,
    n_win_x: int,
    win_h: int,
    win_w: int,
    pass_idx: int,
    output_path: str,
    title: str = "Correlation Planes"
) -> None:
    """
    Visualize ensemble-averaged correlation planes for PIV in a spatial grid.

    Parameters
    ----------
    corr_planes : np.ndarray
        Correlation planes with shape (n_win_y, n_win_x, win_h, win_w)
    n_win_y : int
        Number of interrogation windows in y direction
    n_win_x : int
        Number of interrogation windows in x direction
    win_h : int
        Height of each correlation plane (pixels)
    win_w : int
        Width of each correlation plane (pixels)
    pass_idx : int
        Pass index for naming the output file
    output_path : str
        Full path to save the figure (PNG)
    title : str
        Figure title
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("Matplotlib not available, skipping correlation "
                        "plane visualization")
        return

    # Verify shape
    if corr_planes.shape != (n_win_y, n_win_x, win_h, win_w):
        logging.error("Shape mismatch: expected ({}, {}, {}, {}), "
                      "got {}".format(n_win_y, n_win_x, win_h, win_w,
                                      corr_planes.shape))
        return

    fig, axes = plt.subplots(
        n_win_y, n_win_x, figsize=(3 * n_win_x, 3 * n_win_y)
    )

    # Handle single window case
    if n_win_y == 1 and n_win_x == 1:
        axes = np.array([[axes]])
    elif n_win_y == 1:
        axes = axes.reshape(1, -1)
    elif n_win_x == 1:
        axes = axes.reshape(-1, 1)

    for i in range(n_win_y):
        for j in range(n_win_x):
            ax = axes[i, j]
            plane = corr_planes[i, j]
            im = ax.imshow(plane, origin="lower", cmap="viridis")
            ax.set_title(f"W{i},{j}", fontsize=10)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{title} - Pass {pass_idx}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logging.info(f"Saved: {output_path}")
    plt.close()


def plot_correlation_debug_visualizations(
    R_AA_raw_reshaped: np.ndarray,
    R_BB_raw_reshaped: np.ndarray,
    R_AB_raw_reshaped: np.ndarray,
    R_AA_bg_reshaped: np.ndarray,
    R_BB_bg_reshaped: np.ndarray,
    R_AB_bg_reshaped: np.ndarray,
    R_AA_reshaped: np.ndarray,
    R_BB_reshaped: np.ndarray,
    R_AB_reshaped: np.ndarray,
    n_win_y: int,
    n_win_x: int,
    corr_size: tuple,
    pass_idx: int,
    outdir: Path,
) -> None:
    """
    Generate grid visualizations for all correlation planes in a loop.

    Parameters
    ----------
    R_AA_raw_reshaped : np.ndarray
        Raw AA correlation planes
    R_BB_raw_reshaped : np.ndarray
        Raw BB correlation planes
    R_AB_raw_reshaped : np.ndarray
        Raw AB correlation planes
    R_AA_bg_reshaped : np.ndarray
        Background AA correlation planes
    R_BB_bg_reshaped : np.ndarray
        Background BB correlation planes
    R_AB_bg_reshaped : np.ndarray
        Background AB correlation planes
    R_AA_reshaped : np.ndarray
        Ensemble AA correlation planes
    R_BB_reshaped : np.ndarray
        Ensemble BB correlation planes
    R_AB_reshaped : np.ndarray
        Ensemble AB correlation planes
    n_win_y : int
        Number of windows in y
    n_win_x : int
        Number of windows in x
    corr_size : tuple
        Correlation plane size (h, w)
    pass_idx : int
        Pass index
    outdir : Path
        Output directory
    """
    # Define the correlation planes and their metadata
    corr_data = [
        (R_AA_raw_reshaped, "<A⋆A> Raw (With Background)", "corr_AA_raw"),
        (R_BB_raw_reshaped, "<B⋆B> Raw (With Background)", "corr_BB_raw"),
        (R_AB_raw_reshaped, "<A⋆B> Raw (With Background)", "corr_AB_raw"),
        (R_AA_bg_reshaped, "<A>⋆<A> Background", "corr_AA_bg"),
        (R_BB_bg_reshaped, "<B>⋆<B> Background", "corr_BB_bg"),
        (R_AB_bg_reshaped, "<A>⋆<B> Background", "corr_AB_bg"),
        (R_AA_reshaped, "R_AA Ensemble (Mean-Removed)", "corr_AA_ensemble"),
        (R_BB_reshaped, "R_BB Ensemble (Mean-Removed)", "corr_BB_ensemble"),
        (R_AB_reshaped, "R_AB Ensemble (Mean-Removed) **USED FOR FITTING**",
         "corr_AB_ensemble"),
    ]

    # Plot each correlation plane
    # for corr_planes, title, filename_prefix in corr_data:
    #     output_path = str(outdir / f"{filename_prefix}_pass_{pass_idx}.png")
    #     plot_corr_planes(
    #         corr_planes, n_win_y, n_win_x, corr_size[0], corr_size[1],
    #         pass_idx, output_path, title=title
    #     )


def plot_window_surface(
    window: np.ndarray,
    win_h: int,
    win_w: int,
    output_path: str,
    title: str = "Window Surface"
) -> None:
    """
    Plot a 3D surface of a correlation window.

    Parameters
    ----------
    window : np.ndarray
        1D window array of shape (win_h * win_w,)
    win_h : int
        Height of the window
    win_w : int
        Width of the window
    output_path : str
        Full path to save the figure (PNG)
    title : str
        Figure title
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("Matplotlib not available, skipping window "
                        "surface plot")
        return

    # Reshape to 2D
    window_2d = window.reshape(win_h, win_w)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.arange(win_w)
    y = np.arange(win_h)
    X, Y = np.meshgrid(x, y)
    
    ax.plot_surface(X, Y, window_2d, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Value')
    ax.set_title(title)
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logging.info(f"Saved surface plot: {output_path}")
    plt.close()
