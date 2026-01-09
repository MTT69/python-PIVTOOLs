"""
Centralized window positioning and sizing utilities for PIV processing.

This module provides unified functions for computing interrogation window centers,
padding, and grid positioning across all PIV modes (instantaneous, ensemble standard,
and ensemble single mode).

Coordinate System Convention:
    - 0-based indexing throughout
    - Cartesian convention: (0, 0) represents bottom-left conceptually
    - Arrays stored row-major: array[y, x] where y=vertical, x=horizontal
    - Window centers use floating-point positions (e.g., 63.5 for center of 128-pixel window)

Mathematical Formulas:
    Standard Mode:
        first_ctr = (win_size - 1) / 2.0
        last_ctr = image_size - (win_size + 1) / 2.0
        n_windows = floor((last_ctr - first_ctr) / spacing) + 1

    Single Mode (Ensemble):
        Small window positioned on grid (e.g., 4x4)
        Large SumWindow used for correlation (e.g., 16x16)
        Padding applied to accommodate SumWindow around small window positions
"""

import logging
from typing import Tuple, Optional, Dict, Any

import numpy as np


class WindowCenterResult:
    """Result container for window center calculations.

    Attributes
    ----------
    win_ctrs_x : np.ndarray
        X-coordinates of window centers (horizontal axis), shape (n_win_x,)
    win_ctrs_y : np.ndarray
        Y-coordinates of window centers (vertical axis), shape (n_win_y,)
    n_win_x : int
        Number of windows in X direction
    n_win_y : int
        Number of windows in Y direction
    win_spacing_x : int
        Spacing between windows in X direction (pixels)
    win_spacing_y : int
        Spacing between windows in Y direction (pixels)
    padding : Optional[Tuple[int, int, int, int]]
        Padding applied (top, bottom, left, right), only for single mode
    """

    def __init__(
        self,
        win_ctrs_x: np.ndarray,
        win_ctrs_y: np.ndarray,
        n_win_x: int,
        n_win_y: int,
        win_spacing_x: int,
        win_spacing_y: int,
        padding: Optional[Tuple[int, int, int, int]] = None
    ):
        self.win_ctrs_x = win_ctrs_x
        self.win_ctrs_y = win_ctrs_y
        self.n_win_x = n_win_x
        self.n_win_y = n_win_y
        self.win_spacing_x = win_spacing_x
        self.win_spacing_y = win_spacing_y
        self.padding = padding

    def __repr__(self):
        return (
            f"WindowCenterResult(n_win_x={self.n_win_x}, n_win_y={self.n_win_y}, "
            f"spacing=({self.win_spacing_x}, {self.win_spacing_y}), "
            f"padding={self.padding})"
        )


def compute_padding_for_single_mode(
    window_size: Tuple[int, int],
    sum_window: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Compute padding required for ensemble single mode.

    In single mode, Frame A uses a small window (e.g., 4x4) but Frame B uses
    a larger SumWindow (e.g., 16x16) for correlation. The image must be padded
    to accommodate the larger window around the small window positions.

    Padding is distributed asymmetrically using ceil/floor to handle odd differences.
    This matches the MATLAB implementation in PIV_2D_wdef_ensemble.m lines 161-164.

    Parameters
    ----------
    window_size : Tuple[int, int]
        (height, width) of small window in pixels (Frame A)
    sum_window : Tuple[int, int]
        (height, width) of large sum window in pixels (Frame B)

    Returns
    -------
    padding : Tuple[int, int, int, int]
        (pad_top, pad_bottom, pad_left, pad_right) in pixels

    Examples
    --------
    >>> compute_padding_for_single_mode((4, 4), (16, 16))
    (6, 6, 6, 6)

    >>> compute_padding_for_single_mode((4, 4), (17, 17))
    (7, 6, 7, 6)  # Ceil on top/left, floor on bottom/right

    Notes
    -----
    MATLAB Reference:
        padtop = ceil((SumWindow(1) - wsize(1)) / 2);
        padbot = floor((SumWindow(1) - wsize(1)) / 2);
        padleft = ceil((SumWindow(2) - wsize(2)) / 2);
        padright = floor((SumWindow(2) - wsize(2)) / 2);
    """
    win_h, win_w = window_size
    sum_h, sum_w = sum_window

    # Validate inputs
    if sum_h < win_h or sum_w < win_w:
        raise ValueError(
            f"SumWindow {sum_window} must be >= window_size {window_size} for single mode"
        )

    # Compute padding (matches MATLAB: ceil for top/left, floor for bottom/right)
    pad_top = int(np.ceil((sum_h - win_h) / 2))
    pad_bottom = int(np.floor((sum_h - win_h) / 2))
    pad_left = int(np.ceil((sum_w - win_w) / 2))
    pad_right = int(np.floor((sum_w - win_w) / 2))

    # Verify padding is correct
    assert pad_top + pad_bottom == sum_h - win_h, "Vertical padding mismatch"
    assert pad_left + pad_right == sum_w - win_w, "Horizontal padding mismatch"

    return (pad_top, pad_bottom, pad_left, pad_right)


def compute_window_centers(
    image_shape: Tuple[int, int],
    window_size: Tuple[int, int],
    overlap: float,
    validate: bool = True
) -> WindowCenterResult:
    """
    Compute window center positions for standard PIV processing.

    This is the unified formula used across instantaneous PIV, ensemble standard mode,
    and mask computation. It ensures only full windows fit within the image boundaries.

    The formula guarantees:
    - First window is fully within image (center at (win_size-1)/2)
    - Last window is fully within image (center at image_size - (win_size+1)/2)
    - All intermediate windows are evenly spaced

    Parameters
    ----------
    image_shape : Tuple[int, int]
        (H, W) - height and width of image in pixels
    window_size : Tuple[int, int]
        (win_height, win_width) - interrogation window size in pixels
    overlap : float
        Overlap percentage (0-100). E.g., 50 means 50% overlap between adjacent windows
    validate : bool, optional
        If True, perform input validation. Default True.

    Returns
    -------
    result : WindowCenterResult
        Object containing window centers, counts, spacing, and metadata

    Examples
    --------
    >>> result = compute_window_centers((512, 512), (64, 64), overlap=50)
    >>> result.n_win_x, result.n_win_y
    (16, 16)
    >>> result.win_spacing_x, result.win_spacing_y
    (32, 32)
    >>> result.win_ctrs_x[0]  # First center
    31.5
    >>> result.win_ctrs_x[-1]  # Last center
    480.5

    Notes
    -----
    Mathematical Derivation:
        Window half-width: hw = win_width / 2
        First center must be at least hw from left edge
        Last center must be at least hw from right edge

        Using 0-based indexing where pixel 0 is leftmost:
        - First valid center: (win_width - 1) / 2.0
        - Last valid center: image_width - (win_width + 1) / 2.0

    This matches:
        - cpu_instantaneous.py lines 515-549
        - MATLAB PIV_2D_wdef.m lines 121-122 (converted to 0-based)
    """
    H, W = image_shape
    win_height, win_width = window_size

    # Input validation
    if validate:
        if win_height > H or win_width > W:
            raise ValueError(
                f"Window size {window_size} exceeds image dimensions {image_shape}"
            )
        if overlap < 0 or overlap >= 100:
            raise ValueError(f"Overlap must be in range [0, 100), got {overlap}")
        if win_height < 1 or win_width < 1:
            raise ValueError(f"Window size must be positive, got {window_size}")

    # Calculate window spacing (pixels between adjacent window centers)
    win_spacing_x = round((1 - overlap / 100) * win_width)
    win_spacing_y = round((1 - overlap / 100) * win_height)

    # Ensure at least 1 pixel spacing
    win_spacing_x = max(1, win_spacing_x)
    win_spacing_y = max(1, win_spacing_y)

    # UNIFIED FORMULA: Ensures only full windows fit
    # Equivalent to both:
    #   - (win_width - 1) / 2.0
    #   - -0.5 + win_width / 2.0
    first_ctr_x = (win_width - 1) / 2.0
    last_ctr_x = W - (win_width + 1) / 2.0
    first_ctr_y = (win_height - 1) / 2.0
    last_ctr_y = H - (win_height + 1) / 2.0

    # Compute number of windows that fit
    n_win_x = int(np.floor((last_ctr_x - first_ctr_x) / win_spacing_x)) + 1
    n_win_y = int(np.floor((last_ctr_y - first_ctr_y) / win_spacing_y)) + 1

    # Ensure at least one window
    n_win_x = max(1, n_win_x)
    n_win_y = max(1, n_win_y)

    # Generate evenly-spaced window centers
    win_ctrs_x = np.linspace(
        first_ctr_x,
        first_ctr_x + win_spacing_x * (n_win_x - 1),
        n_win_x,
        dtype=np.float32
    )
    # Anchor to BOTTOM of image, but return ASCENDING order for np.interp compatibility
    # Gap will be at TOP (pixels 0 to first_ctr_y - window/2 are unmeasured)
    # Same positions as descending, just ascending array order
    first_ctr_y_anchored = last_ctr_y - win_spacing_y * (n_win_y - 1)
    win_ctrs_y = np.linspace(
        first_ctr_y_anchored,  # Near top (low pixel y, high physical y)
        last_ctr_y,            # At bottom (high pixel y, low physical y)
        n_win_y,
        dtype=np.float32
    )

    logging.debug(
        f"Computed window centers: image_shape={image_shape}, window_size={window_size}, "
        f"overlap={overlap}%, n_windows=({n_win_x}, {n_win_y}), "
        f"spacing=({win_spacing_x}, {win_spacing_y}), "
        f"first_center=({first_ctr_x:.1f}, {first_ctr_y:.1f}), "
        f"last_center=({win_ctrs_x[-1]:.1f}, {win_ctrs_y[-1]:.1f})"
    )

    return WindowCenterResult(
        win_ctrs_x=win_ctrs_x,
        win_ctrs_y=win_ctrs_y,
        n_win_x=n_win_x,
        n_win_y=n_win_y,
        win_spacing_x=win_spacing_x,
        win_spacing_y=win_spacing_y,
        padding=None
    )


def compute_window_centers_single_mode(
    image_shape: Tuple[int, int],
    window_size: Tuple[int, int],
    sum_window: Tuple[int, int],
    overlap: float,
    validate: bool = True
) -> WindowCenterResult:
    """
    Compute window center positions for ensemble single mode.

    Single mode is a specialized ensemble PIV technique where:
    - Frame A uses a small window (e.g., 4x4) with center=1, rest=0 weighting
    - Frame B uses a large SumWindow (e.g., 16x16) with all=1 weighting
    - This reduces particle dropout bias in ensemble averaging

    The computation:
    1. Applies padding to image to accommodate SumWindow around small window positions
    2. Grid spacing is based on SMALL window size (determines resolution)
    3. Window centers are positioned relative to LARGE SumWindow (determines correlation size)

    This matches MATLAB PIV_2D_wdef_ensemble.m lines 172-186.

    Parameters
    ----------
    image_shape : Tuple[int, int]
        (H, W) - height and width of original image in pixels
    window_size : Tuple[int, int]
        (win_height, win_width) - SMALL window size (Frame A)
    sum_window : Tuple[int, int]
        (sum_height, sum_width) - LARGE window size (Frame B, for correlation)
    overlap : float
        Overlap percentage (0-100), applied to SMALL window size
    validate : bool, optional
        If True, perform input validation. Default True.

    Returns
    -------
    result : WindowCenterResult
        Object containing window centers, counts, spacing, and padding metadata
        result.padding = (pad_top, pad_bottom, pad_left, pad_right)

    Examples
    --------
    >>> result = compute_window_centers_single_mode(
    ...     image_shape=(512, 512),
    ...     window_size=(4, 4),
    ...     sum_window=(16, 16),
    ...     overlap=50
    ... )
    >>> result.n_win_x  # Many windows due to small 4x4 spacing
    255
    >>> result.win_spacing_x  # Spacing based on 4x4 window
    2
    >>> result.padding  # Padding to accommodate 16x16 SumWindow
    (6, 6, 6, 6)

    Notes
    -----
    MATLAB Reference (PIV_2D_wdef_ensemble.m):
        Lines 161-164: Padding computation
        Lines 172-186: Window center positioning with special handling for wsize==1

    The special case for wsize==1 (single pixel):
        - Center at integer position (1 + SumWindow/2) instead of half-pixel
        - Rarely used in practice but included for completeness
    """
    H, W = image_shape
    win_height, win_width = window_size
    sum_height, sum_width = sum_window

    # Input validation
    if validate:
        if sum_height < win_height or sum_width < win_width:
            raise ValueError(
                f"SumWindow {sum_window} must be >= window_size {window_size} for single mode"
            )
        if win_height > H or win_width > W:
            raise ValueError(
                f"Window size {window_size} exceeds image dimensions {image_shape}"
            )
        if overlap < 0 or overlap >= 100:
            raise ValueError(f"Overlap must be in range [0, 100), got {overlap}")

    # Compute padding required
    pad_top, pad_bottom, pad_left, pad_right = compute_padding_for_single_mode(
        window_size, sum_window
    )

    # Padded image dimensions
    Nx = W + pad_left + pad_right
    Ny = H + pad_top + pad_bottom

    # Window spacing based on SMALL window size (determines grid resolution)
    win_spacing_x = round((1 - overlap / 100) * win_width)
    win_spacing_y = round((1 - overlap / 100) * win_height)
    win_spacing_x = max(1, win_spacing_x)
    win_spacing_y = max(1, win_spacing_y)

    # Window center positioning in padded coordinates
    # Uses same formula as standard mode: (win_size - 1) / 2.0 for first center
    # Shifted by padding offset to place grid correctly in padded image
    #
    # This ensures consistency with standard mode and correct 0-based C indexing:
    # - C extraction: row_min = floor(center - (size-1)/2 + 0.5)
    # - For center=7.5, window=16: row_min = floor(7.5 - 7.5 + 0.5) = 0
    # - Extracts pixels [0, 15] which is correct
    first_ctr_x = pad_left + (win_width - 1) / 2.0
    first_ctr_y = pad_top + (win_height - 1) / 2.0

    # Last center: mirror of standard mode formula in padded coordinates
    # Standard mode uses: image_size - (win_size + 1) / 2.0
    # In padded coords: pad + original_size - (win_size + 1) / 2.0
    last_ctr_x = pad_left + W - (win_width + 1) / 2.0
    last_ctr_y = pad_top + H - (win_height + 1) / 2.0

    # Compute number of windows
    n_win_x = int(np.floor((last_ctr_x - first_ctr_x) / win_spacing_x)) + 1
    n_win_y = int(np.floor((last_ctr_y - first_ctr_y) / win_spacing_y)) + 1
    n_win_x = max(1, n_win_x)
    n_win_y = max(1, n_win_y)

    # Generate window centers (on padded image coordinate system)
    win_ctrs_x = np.linspace(
        first_ctr_x,
        first_ctr_x + win_spacing_x * (n_win_x - 1),
        n_win_x,
        dtype=np.float32
    )
    # Anchor to BOTTOM of image, but return ASCENDING order for np.interp compatibility
    # Gap will be at TOP (pixels 0 to first_ctr_y - window/2 are unmeasured)
    first_ctr_y_anchored = last_ctr_y - win_spacing_y * (n_win_y - 1)
    win_ctrs_y = np.linspace(
        first_ctr_y_anchored,  # Near top (low pixel y, high physical y)
        last_ctr_y,            # At bottom (high pixel y, low physical y)
        n_win_y,
        dtype=np.float32
    )

    # Verify last window fits within padded image (catches off-by-one errors early)
    max_ctr_x = win_ctrs_x[-1]
    max_ctr_y = win_ctrs_y[-1]  # Last element is largest (ascending order)
    half_win_x = (sum_width - 1) / 2.0
    half_win_y = (sum_height - 1) / 2.0

    if max_ctr_x + half_win_x >= Nx or max_ctr_y + half_win_y >= Ny:
        raise ValueError(
            f"Single mode window centers exceed padded image bounds. "
            f"max_center=({max_ctr_x:.1f}, {max_ctr_y:.1f}), "
            f"half_window=({half_win_x:.1f}, {half_win_y:.1f}), "
            f"padded_size=({Nx}, {Ny}), sum_window={sum_window}"
        )

    logging.debug(
        f"Computed single-mode window centers: image_shape={image_shape}, "
        f"window_size={window_size}, sum_window={sum_window}, overlap={overlap}%, "
        f"padded_shape=({Ny}, {Nx}), padding=(T{pad_top},B{pad_bottom},L{pad_left},R{pad_right}), "
        f"n_windows=({n_win_x}, {n_win_y}), spacing=({win_spacing_x}, {win_spacing_y}), "
        f"first_center=({first_ctr_x:.1f}, {first_ctr_y:.1f})"
    )

    return WindowCenterResult(
        win_ctrs_x=win_ctrs_x,
        win_ctrs_y=win_ctrs_y,
        n_win_x=n_win_x,
        n_win_y=n_win_y,
        win_spacing_x=win_spacing_x,
        win_spacing_y=win_spacing_y,
        padding=(pad_top, pad_bottom, pad_left, pad_right)
    )


def apply_single_mode_padding(
    image: np.ndarray,
    window_size: Tuple[int, int],
    sum_window: Tuple[int, int],
    pad_value: float = 0.0
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Apply zero-padding to image for ensemble single mode processing.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (N, H, W) for batch
    window_size : Tuple[int, int]
        (height, width) of small window
    sum_window : Tuple[int, int]
        (height, width) of large sum window
    pad_value : float, optional
        Value to use for padding. Default 0.0.

    Returns
    -------
    padded_image : np.ndarray
        Padded image with same number of dimensions as input
    padding : Tuple[int, int, int, int]
        (pad_top, pad_bottom, pad_left, pad_right)

    Examples
    --------
    >>> img = np.random.randn(512, 512)
    >>> padded, padding = apply_single_mode_padding(img, (4, 4), (16, 16))
    >>> padded.shape
    (524, 524)  # Added 12 pixels in each dimension
    >>> padding
    (6, 6, 6, 6)
    """

def apply_single_mode_padding(
    image: np.ndarray,
    window_size: Tuple[int, int],
    sum_window: Tuple[int, int],
    pad_value: float = 0.0
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Apply zero-padding to image for ensemble single mode processing.

    Supports:
    - (H, W)
    - (N, H, W)
    - (N, 2, H, W)

    Returns
    -------
    padded_image : np.ndarray
        Padded image with same number of dimensions as input
    padding : Tuple[int, int, int, int]
        (pad_top, pad_bottom, pad_left, pad_right)
    """
    pad_top, pad_bottom, pad_left, pad_right = compute_padding_for_single_mode(
        window_size, sum_window
    )

    if image.ndim == 2:
        # (H, W)
        pad_width = (
            (pad_top, pad_bottom),
            (pad_left, pad_right),
        )

    elif image.ndim == 3:
        # (N, H, W)
        pad_width = (
            (0, 0),
            (pad_top, pad_bottom),
            (pad_left, pad_right),
        )

    elif image.ndim == 4:
        # (N, 2, H, W)
        pad_width = (
            (0, 0),  # batch (N)
            (0, 0),  # channel (2)
            (pad_top, pad_bottom),
            (pad_left, pad_right),
        )

    else:
        raise ValueError(
            f"Image must be 2D, 3D, or 4D, got shape {image.shape}"
        )

    padded = np.pad(
        image,
        pad_width,
        mode="constant",
        constant_values=pad_value,
    )

    return padded, (pad_top, pad_bottom, pad_left, pad_right)



def validate_window_configuration(
    image_shape: Tuple[int, int],
    window_size: Tuple[int, int],
    overlap: float,
    sum_window: Optional[Tuple[int, int]] = None,
    mode: str = 'standard'
) -> Tuple[bool, str]:
    """
    Validate that window configuration is feasible for given image.

    Parameters
    ----------
    image_shape : Tuple[int, int]
        (H, W) image dimensions
    window_size : Tuple[int, int]
        (height, width) interrogation window size
    overlap : float
        Overlap percentage (0-100)
    sum_window : Optional[Tuple[int, int]]
        (height, width) sum window for single mode, if applicable
    mode : str
        'standard' or 'single'

    Returns
    -------
    is_valid : bool
        True if configuration is valid
    message : str
        Explanation if invalid, empty string if valid

    Examples
    --------
    >>> validate_window_configuration((512, 512), (64, 64), 50)
    (True, '')

    >>> validate_window_configuration((100, 100), (256, 256), 50)
    (False, 'Window size (256, 256) exceeds image dimensions (100, 100)')
    """
    H, W = image_shape
    win_h, win_w = window_size

    # Check basic window size
    if win_h > H or win_w > W:
        return False, f"Window size {window_size} exceeds image dimensions {image_shape}"

    # Check overlap range
    if overlap < 0 or overlap >= 100:
        return False, f"Overlap must be in range [0, 100), got {overlap}"

    # Check minimum window size
    if win_h < 1 or win_w < 1:
        return False, f"Window size must be positive, got {window_size}"

    # For single mode, check sum window
    if mode == 'single':
        if sum_window is None:
            return False, "sum_window must be specified for single mode"
        sum_h, sum_w = sum_window
        if sum_h < win_h or sum_w < win_w:
            return False, f"SumWindow {sum_window} must be >= window_size {window_size}"

    # Try to compute window centers to check if at least 1 window fits
    try:
        if mode == 'single' and sum_window is not None:
            result = compute_window_centers_single_mode(
                image_shape, window_size, sum_window, overlap, validate=False
            )
        else:
            result = compute_window_centers(
                image_shape, window_size, overlap, validate=False
            )

        if result.n_win_x < 1 or result.n_win_y < 1:
            return False, f"Configuration produces no valid windows"

    except Exception as e:
        return False, f"Window computation failed: {str(e)}"

    return True, ""


def get_window_grid_shape(
    image_shape: Tuple[int, int],
    window_size: Tuple[int, int],
    overlap: float,
    sum_window: Optional[Tuple[int, int]] = None,
    mode: str = 'standard'
) -> Tuple[int, int]:
    """
    Get the output grid shape (number of windows) without computing full centers.

    Useful for pre-allocating output arrays.

    Parameters
    ----------
    image_shape : Tuple[int, int]
        (H, W) image dimensions
    window_size : Tuple[int, int]
        (height, width) interrogation window size
    overlap : float
        Overlap percentage (0-100)
    sum_window : Optional[Tuple[int, int]]
        (height, width) sum window for single mode
    mode : str
        'standard' or 'single'

    Returns
    -------
    grid_shape : Tuple[int, int]
        (n_win_y, n_win_x) - number of windows in vertical and horizontal directions

    Examples
    --------
    >>> get_window_grid_shape((512, 512), (64, 64), 50)
    (16, 16)
    """
    if mode == 'single' and sum_window is not None:
        result = compute_window_centers_single_mode(
            image_shape, window_size, sum_window, overlap
        )
    else:
        result = compute_window_centers(image_shape, window_size, overlap)

    return (result.n_win_y, result.n_win_x)
