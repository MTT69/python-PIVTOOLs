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


# Settings object to drive plot_scalar_field conveniently from Config
@dataclass
class Settings:
    variableName: str
    variableUnits: str = ""
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
def plot_scalar_field(variable, mask, settings):
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
            y = np.arange(ny)
        X, Y = np.meshgrid(x, y)

    # Create the plot
    fig, ax = plt.subplots(
        figsize=(12, 7)
    )  # The size to be adjusted to meet the UI requirements!
    ax.set_facecolor("gray")  # <-- gray shows through masked holes

    # Determine limits
    if settings.lower_limit is not None and settings.upper_limit is not None:
        vmin, vmax = settings.lower_limit, settings.upper_limit
    else:
        # Compute from data
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
        # Default colormap selection when no explicit cmap provided in settings.
        # For diverging data that spans negative and positive, use the full 'bwr'
        # with a TwoSlopeNorm. For one-sided data, use half of 'bwr':
        # - vmax <= 0 : use lower-half (blue->white) so values near zero are white
        # - vmin >= 0 : use upper-half (white->red) so values near zero are white
        if use_two_slope:
            cmap = plt.get_cmap("bwr")
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            bwr = plt.get_cmap("bwr")
            if vmax <= 0:
                # lower half: blue -> white (0.0 -> 0.5 of bwr)
                colors = bwr(np.linspace(0.0, 0.5, 256))
                cmap = mpl.colors.LinearSegmentedColormap.from_list("bwr_lower", colors)
                norm = Normalize(vmin=vmin, vmax=vmax)
            else:  # vmin >= 0
                # upper half: white -> red (0.5 -> 1.0 of bwr)
                colors = bwr(np.linspace(0.5, 1.0, 256))
                cmap = mpl.colors.LinearSegmentedColormap.from_list("bwr_upper", colors)
                norm = Normalize(vmin=vmin, vmax=vmax)

    # Contourf
    im = plt.contourf(X, Y, masked_var, levels=settings.levels, cmap=cmap, norm=norm)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # required for some Matplotlib versions
    cbar = fig.colorbar(sm, ax=ax, label=cm_label)

    if isinstance(settings.levels, np.ndarray):
        # pick a stable set, e.g. 7 ticks or [vmin, 0, vmax] for diverging
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
    ax.set_xlabel(settings._xlabel)
    ax.set_ylabel(settings._ylabel)

    # labels = [f"{t:.2f}" for t in ticks]
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels(labels)
    # cbar.ax.yaxis.set_major_locator(FixedLocator(ticks))
    # cbar.ax.yaxis.set_major_formatter(FixedFormatter(labels))

    # ax.set_title(f"{settings.title}")  # Set plot title
    # ax.set_xlabel(settings._xlabel)     # Set x-axis label
    # ax.set_ylabel(settings._ylabel)     # Set y-axis label

    # Do not save or close here; return figure to caller for handling
    return fig, ax, im
    # ax.set_ylabel(settings._ylabel)     # Set y-axis label
    # # Save figure if save_name is provided
    # if settings.save_name:
    #     fig.savefig(f"{settings.save_name}{settings.save_extension}", dpi=1200, bbox_inches='tight')
    # plt.close(fig)

    # # Optionally save variable as pickle file
    # if settings.save_pickle:
    #     import pickle
    #     with open(f"{settings.save_name}.pkl", 'wb') as f:
    #         pickle.dump(variable, f)
