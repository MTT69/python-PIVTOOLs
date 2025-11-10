from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import FixedFormatter, FixedLocator

from ..config import Config

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
    )

# Function to plot a scalar field with masking and customizable settings
def plot_scalar_field(variable, mask, settings): # efe
    # Extract plot settings
    plt.rcParams.update({"font.size": settings._fontsize})
    plt.rcParams["axes.titlesize"] = settings._title_fontsize

    cm_label = settings.variableName + " (" + settings.variableUnits + ")"

    # Mask the variable array where mask is True
    masked_var = np.ma.array(variable, mask=mask)

    # Generate coordinate arrays: prefer provided coords_x/coords_y, else corners, else indices
    X = Y = None
    if settings.coords_x is not None and settings.coords_y is not None:
        cx, cy = np.asarray(settings.coords_x), np.asarray(settings.coords_y)
        # 2D grid case matching variable shape
        if (
            cx.ndim == 2
            and cy.ndim == 2
            and cx.shape == variable.shape
            and cy.shape == variable.shape
        ):
            X, Y = cx, cy
        # 1D axes case
        elif cx.ndim == 1 and cy.ndim == 1:
            ny, nx = variable.shape
            if cx.size == nx and cy.size == ny:
                X, Y = np.meshgrid(cx, cy)
    if X is None or Y is None:
        if settings.corners is not None and all(
            c is not None for c in settings.corners
        ):
            x0, y0, x1, y1 = settings.corners
            ny, nx = variable.shape
            x = np.linspace(x0, x1, nx)
            y = np.linspace(y0, y1, ny)
        else:
            ny, nx = variable.shape
            x = np.arange(nx)
            y = np.arange(ny-1, -1, -1)
        X, Y = np.meshgrid(x, y)

    # Create the plot (object-oriented API)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor("gray")  # <-- gray shows through masked holes

    # Determine limits
    if settings.lower_limit is not None and settings.upper_limit is not None:
        vmin, vmax = settings.lower_limit, settings.upper_limit
    else:
        # Use nanmin/nanmax to handle NaN values properly
        valid_data = masked_var.compressed()  # Get unmasked data
        if len(valid_data) > 0:
            vmin = float(np.nanmin(valid_data))
            vmax = float(np.nanmax(valid_data))
            # Check if min/max are still NaN (all valid data is NaN)
            if np.isnan(vmin) or np.isnan(vmax) or np.isinf(vmin) or np.isinf(vmax):
                # Fallback to sensible defaults
                vmin, vmax = 0.0, 1.0
        else:
            # No valid data at all - use defaults
            vmin, vmax = 0.0, 1.0

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
    if settings.length_units:
        ax.set_xlabel(settings._xlabel + f" ({settings.length_units})")
        ax.set_ylabel(settings._ylabel + f" ({settings.length_units})")
    else:
        ax.set_xlabel(settings._xlabel)
        ax.set_ylabel(settings._ylabel)

    return fig, ax, im