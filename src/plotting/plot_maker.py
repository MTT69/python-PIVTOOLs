from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import FixedFormatter, FixedLocator

from config import Config

mpl.use("Agg")
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["xtick.major.size"] = 5
mpl.rcParams["ytick.major.size"] = 5
mpl.rcParams["xtick.minor.size"] = 3
mpl.rcParams["ytick.minor.size"] = 3
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["xtick.major.pad"] = 6
mpl.rcParams["ytick.major.pad"] = 6


# Settings object to drive plot_scalar_field conveniently from Config
@dataclass
class Settings:
    variableName: str
    variableUnits: str = ""
    length_units: str = "mm"
    title: str = ""
    levels: int | list = 500
    cmap: str | None = None
    corners: tuple | None = None
    lower_limit: float | None = None
    upper_limit: float | None = None
    _xlabel: str = "x"
    _ylabel: str = "y"
    _fontsize: int = 12
    _title_fontsize: int = 14
    save_name: str | None = None
    save_extension: str = ".png"
    save_pickle: bool = False
    # New: optional coordinates and symmetric scaling control
    coords_x: np.ndarray | None = None
    coords_y: np.ndarray | None = None
    symmetric_around_zero: bool = True
    # Transformation parameters
    rotation: int = 0
    flip_horizontal: bool = False
    flip_vertical: bool = False
    transpose: bool = False


def make_scalar_settings(
    config: Config,
    *,
    variable: str,
    run_label: int,
    save_basepath: Path,
    title: str | None = None,
    variable_units: str = "",
    length_units: str = "mm",
    cmap: str | None = None,
    levels: int | list = 100,
    lower_limit: float | None = None,
    upper_limit: float | None = None,
    corners: tuple | None = None,
    coords_x: np.ndarray | None = None,
    coords_y: np.ndarray | None = None,
    symmetric_around_zero: bool = True,
    rotation: int = 0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
    transpose: bool = False,
) -> Settings:
    return Settings(
        variableName=variable,
        variableUnits=variable_units,
        length_units=length_units,
        title=title or f"{variable} pass {run_label}",
        levels=levels,
        cmap=cmap,
        corners=corners,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        _xlabel="x",
        _ylabel="y",
        _fontsize=config.plot_fontsize,
        _title_fontsize=config.plot_title_fontsize,
        save_name=str(save_basepath),
        save_extension=config.plot_save_extension,
        save_pickle=config.plot_save_pickle,
        coords_x=coords_x,
        coords_y=coords_y,
        symmetric_around_zero=symmetric_around_zero,
        rotation=rotation,
        flip_horizontal=flip_horizontal,
        flip_vertical=flip_vertical,
        transpose=transpose,
    )


# Function to apply transformations to data arrays
def apply_transformations( # efe
    data: np.ndarray,
    transpose: bool = False,
    rotation: int = 0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> np.ndarray:
    """
    Apply transformations to a 2D numpy array in the specified order:
    1. Transpose
    2. Rotation (90-degree increments)
    3. Horizontal flip
    4. Vertical flip
    
    Args:
        data: 2D numpy array to transform
        transpose: If True, swap X and Y axes
        rotation: Number of 90-degree clockwise rotations (0, 1, 2, or 3)
        flip_horizontal: If True, flip left-right
        flip_vertical: If True, flip up-down
    
    Returns:
        Transformed numpy array
    """
    result = data.copy()
    
    # 1. Transpose (swap axes)
    if transpose:
        result = np.transpose(result)
    
    # 2. Rotation (k=1 means 90° clockwise, k=2 means 180°, k=3 means 270° clockwise)
    if rotation > 0:
        result = np.rot90(result, k=-rotation)  # Negative for clockwise rotation
    
    # 3. Horizontal flip (left-right)
    if flip_horizontal:
        result = np.fliplr(result)
    
    # 4. Vertical flip (up-down)
    if flip_vertical:
        result = np.flipud(result)
    
    return result


# Function to plot a scalar field with masking and customizable settings
def plot_scalar_field(variable, mask, settings): # efe
    # Extract plot settings
    plt.rcParams.update({"font.size": settings._fontsize})
    plt.rcParams["axes.titlesize"] = settings._title_fontsize

    cm_label = settings.variableName + " (" + settings.variableUnits + ")"

    # Store original shape before transformations for coordinate handling
    original_var_shape = variable.shape
    
    # Debug logging
    from loguru import logger
    logger.info(f"plot_scalar_field: flip_horizontal={settings.flip_horizontal}, flip_vertical={settings.flip_vertical}, transpose={settings.transpose}, rotation={settings.rotation}")
    
    # Apply transformations to variable and mask
    variable = apply_transformations(
        variable,
        transpose=settings.transpose,
        rotation=settings.rotation,
        flip_horizontal=settings.flip_horizontal,
        flip_vertical=settings.flip_vertical,
    )
    
    mask = apply_transformations(
        mask,
        transpose=settings.transpose,
        rotation=settings.rotation,
        flip_horizontal=settings.flip_horizontal,
        flip_vertical=settings.flip_vertical,
    )

    # Mask the variable array where mask is True
    masked_var = np.ma.array(variable, mask=mask)

    # Generate coordinate arrays: prefer provided coords_x/coords_y, else corners, else indices
    X = Y = None
    
    if settings.coords_x is not None and settings.coords_y is not None:
        from loguru import logger
        
        cx, cy = np.asarray(settings.coords_x), np.asarray(settings.coords_y)
        
        # Check if any transformations are applied
        has_transforms = (settings.rotation != 0 or settings.flip_horizontal or 
                         settings.flip_vertical or settings.transpose)
        
        if not has_transforms:
            # No transformations - use original coordinates as-is
            if cx.ndim == 2 and cy.ndim == 2:
                X, Y = cx, cy
            elif cx.ndim == 1 and cy.ndim == 1:
                X, Y = np.meshgrid(cx, cy)
        else:
            # Apply the SAME transformations to coordinates as applied to data
            if cx.ndim == 1 and cy.ndim == 1:
                # Convert 1D coordinates to 2D grid first
                cx_2d, cy_2d = np.meshgrid(cx, cy)
            else:
                cx_2d, cy_2d = cx, cy
            
            # Apply transformations to coordinate arrays
            X = apply_transformations(
                cx_2d,
                transpose=settings.transpose,
                rotation=settings.rotation,
                flip_horizontal=settings.flip_horizontal,
                flip_vertical=settings.flip_vertical,
            )
            Y = apply_transformations(
                cy_2d,
                transpose=settings.transpose,
                rotation=settings.rotation,
                flip_horizontal=settings.flip_horizontal,
                flip_vertical=settings.flip_vertical,
            )
            
            logger.info(f"Transformed coordinates: X[{X.min():.2f}→{X.max():.2f}], Y[{Y.min():.2f}→{Y.max():.2f}]")
                
    if X is None or Y is None:
        # No coordinates provided - use simple pixel coordinates
        ny_new, nx_new = variable.shape  # After transformation
        X, Y = np.meshgrid(np.arange(nx_new), np.arange(ny_new))

    # Create the plot (object-oriented API)
    fig, ax = plt.subplots()
    ax.set_facecolor("gray")  # <-- gray shows through masked holes
    ax.set_aspect('equal', adjustable='box')  # Maintain physical aspect ratio

    # Determine limits
    if settings.lower_limit is not None and settings.upper_limit is not None:
        vmin, vmax = settings.lower_limit, settings.upper_limit
    else:
        vmin, vmax = float(masked_var.min()), float(masked_var.max())

    # Enforce symmetric scale around zero if data spans negative and positive
    use_two_slope = False
    actual_min = vmin
    actual_max = vmax
    if settings.symmetric_around_zero and vmin < 0 and vmax > 0:
        vabs = max(abs(vmin), abs(vmax))
        vmin, vmax = -vabs, vabs
        use_two_slope = True

    # Select colormap & norm
    if settings.cmap is not None:
        cmap = plt.get_cmap(settings.cmap)
        norm = (
            TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
            if use_two_slope
            else Normalize(vmin=vmin, vmax=vmax)
        )
    else:
        if use_two_slope:
            cmap = plt.get_cmap("bwr")
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            bwr = plt.get_cmap("bwr")
            if vmax <= 0:
                colors = bwr(np.linspace(0.0, 0.5, 256))
                cmap = mpl.colors.LinearSegmentedColormap.from_list("bwr_lower", colors)
                norm = Normalize(vmin=vmin, vmax=vmax)
            else:
                colors = bwr(np.linspace(0.5, 1.0, 256))
                cmap = mpl.colors.LinearSegmentedColormap.from_list("bwr_upper", colors)
                norm = Normalize(vmin=vmin, vmax=vmax)

    # Use ax.contourf (object-oriented)
    im = ax.contourf(X, Y, masked_var, levels=settings.levels, cmap=cmap, norm=norm)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required for some Matplotlib versions

    # Use object-oriented colorbar
    cbar = fig.colorbar(sm, ax=ax, label=cm_label)

    if isinstance(settings.levels, np.ndarray):
        if isinstance(norm, TwoSlopeNorm):
            ticks = [norm.vmin, 0.0, norm.vmax]
        else:
            ticks = np.linspace(norm.vmin, norm.vmax, 7)
    else:
        ticks = np.linspace(norm.vmin, norm.vmax, 7)

    # Optional: nice fixed tick count
    ticks = np.linspace(actual_min, actual_max, 7)
    labels = [f"{t:.2f}" for t in ticks]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    cbar.ax.set_ylim(actual_min, actual_max)
    cbar.ax.yaxis.set_major_locator(FixedLocator(ticks))
    cbar.ax.yaxis.set_major_formatter(FixedFormatter(labels))

    ax.set_title(f"{settings.title}")
    
    # Swap axis labels if transpose is applied
    xlabel = settings._xlabel
    ylabel = settings._ylabel
    if settings.transpose:
        xlabel, ylabel = ylabel, xlabel
    
    if settings.length_units:
        ax.set_xlabel(xlabel + f" ({settings.length_units})")
        ax.set_ylabel(ylabel + f" ({settings.length_units})")
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    return fig, ax, im
