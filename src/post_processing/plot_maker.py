import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from config import Config

##### EFE ADDED needs to 

# Function to plot a scalar field with masking and customizable settings
def plot_scalar_field(variable, mask, settings):
    # Extract plot settings
    plt.rcParams.update({'font.size': settings._fontsize})
    plt.rcParams['axes.titlesize'] = settings._title_fontsize


    cm_label = settings.variableName + " (" + settings.variableUnits + ")"

    # Mask the variable array where mask is True
    masked_var = np.ma.array(variable, mask=mask)
    var_min, var_max = np.min(masked_var), np.max(masked_var) # Determine the min and max values

    # Generate coordinate arrays based on corners and variable shape
    if settings.corners is not None and all(c is not None for c in settings.corners):
        x0, y0, x1, y1 = settings.corners
        ny, nx = variable.shape
        x = np.linspace(x0, x1, nx)
        y = np.linspace(y0, y1, ny)
    else:
        # Fallback: use array indices as coordinates
        ny, nx = variable.shape
        x = np.arange(nx)
        y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))  # The size to be adjusted to meet the UI requirements!
    ax.set_facecolor('gray')  # <-- gray shows through masked holes

    # Determine limits
    if settings.lower_limit is not None and settings.upper_limit is not None:
        vmin, vmax = settings.lower_limit, settings.upper_limit
    else:
        vmin, vmax = var_min, var_max

    # Select colormap & norm
    if settings.cmap is not None:
        cmap = plt.get_cmap(settings.cmap).copy()
        norm = Normalize(vmin=vmin, vmax=vmax)
    else:
        if vmin <= 0 <= vmax:
            cmap = "bwr"
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        elif vmax <= 0:
            cmap = "Blues_r"
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:  # vmin >= 0
            cmap = "Reds"
            norm = Normalize(vmin=vmin, vmax=vmax)


    im = ax.contourf(X, Y, masked_var, levels=settings.levels, cmap=cmap, norm=norm)

    fig.colorbar(im, ax=ax, label=cm_label)  # Add colorbar
    ax.set_title(f"{settings.title}")  # Set plot title
    ax.set_xlabel(settings._xlabel)     # Set x-axis label
    ax.set_ylabel(settings._ylabel)     # Set y-axis label

    plt.show()  # Display the plot

    # Save figure if save_name is provided
    if settings.save_name:
        fig.savefig(f"{settings.save_name}{settings.save_extension}", dpi=300)
    
    # Optionally save variable as pickle file
    if settings.save_pickle:
        import pickle
        with open(f"{settings.save_name}.pkl", 'wb') as f:
            pickle.dump(variable, f)


if __name__ == "__main__":
    ux, uy, b_mask = read_mat_contents("1000/Cam1/Instantaneous/00014.mat")
    coords_x, coords_y = read_coords_file("1000/Cam1/Instantaneous/Co_ords.mat")

    plot_settings = PlotSettings()

    plot_settings.corners = (coords_x[0, 0], coords_y[0, 0], coords_x[-1, -1], coords_y[-1, -1])
    plot_settings.variableName = r"$u_x$" # maybe a result class that will contain the variable name and units?
    plot_settings.variableUnits = r"mm/s"

    # plot_settings.upper_limit = 100
    # plot_settings.lower_limit =  30

    plot_scalar_field(uy, b_mask, plot_settings)