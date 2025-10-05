import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec

def load_results(filepath):
    """
    Load Reynolds stress results from file.
    
    Args:
        filepath: Path to results file
    
    Returns:
        dict: Dictionary containing arrays for each column
    """
    data = np.loadtxt(filepath)
    
    results = {
        'x': data[:, 0],
        'y': data[:, 1],
        'count': data[:, 2],
        'mean_dx': data[:, 3],
        'mean_dy': data[:, 4],
        'mean_dz': data[:, 5],
        'uu': data[:, 6],
        'vv': data[:, 7],
        'ww': data[:, 8],
        'uv': data[:, 9],
        'uw': data[:, 10],
        'vw': data[:, 11]
    }
    
    return results

def create_2d_field(x, y, values, method='linear'):
    """
    Create 2D gridded field from scattered data.
    
    Args:
        x, y: Coordinate arrays
        values: Value array
        method: Interpolation method
    
    Returns:
        X, Y, Z: Meshgrid coordinates and interpolated values
    """
    from scipy.interpolate import griddata
    
    # Create regular grid
    xi = np.linspace(x.min(), x.max(), 200)
    yi = np.linspace(y.min(), y.max(), 200)
    X, Y = np.meshgrid(xi, yi)
    
    # Interpolate
    Z = griddata((x, y), values, (X, Y), method=method)
    
    return X, Y, Z

def plot_reynolds_stress_and_means(results, output_path):
    """
    Create figure with Reynolds stress diagonal terms and mean components.
    
    Args:
        results: Dictionary of results from load_results()
        output_path: Path to save figure
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Define plots: (data_key, title, colormap)
    plots = [
        ('uu', r"$\langle u'u' \rangle$ (mm²)", 'viridis'),
        ('vv', r"$\langle v'v' \rangle$ (mm²)", 'viridis'),
        ('ww', r"$\langle w'w' \rangle$ (mm²)", 'viridis'),
        ('mean_dx', r'$\langle \Delta x \rangle$ (mm)', 'RdBu_r'),
        ('mean_dy', r'$\langle \Delta y \rangle$ (mm)', 'RdBu_r'),
        ('mean_dz', r'$\langle \Delta z \rangle$ (mm)', 'RdBu_r')
    ]
    
    for idx, (key, title, cmap) in enumerate(plots):
        row = idx // 3
        col = idx % 3
        ax = fig.add_subplot(gs[row, col])
        
        # Create 2D field
        X, Y, Z = create_2d_field(results['x'], results['y'], results[key])
        
        # Plot
        if 'mean' in key:
            # Symmetric colorbar for mean displacements
            vmax = np.nanmax(np.abs(Z))
            im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading='auto', 
                              vmin=-vmax, vmax=vmax)
        else:
            # Standard colorbar for Reynolds stresses
            im = ax.pcolormesh(X, Y, Z, cmap=cmap, shading='auto')
        
        # Scatter original points
        ax.scatter(results['x'], results['y'], c='black', s=1, alpha=0.3)
        
        ax.set_xlabel('x (mm)', fontsize=12)
        ax.set_ylabel('y (mm)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=10)
    
    plt.suptitle('Reynolds Stress Tensor (Diagonal) and Mean Displacements', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")
    plt.close()

def plot_statistics_summary(results, output_path):
    """
    Create additional figure with data statistics.
    
    Args:
        results: Dictionary of results
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Sample count distribution
    ax = axes[0, 0]
    X, Y, Z = create_2d_field(results['x'], results['y'], results['count'])
    im = ax.pcolormesh(X, Y, Z, cmap='plasma', shading='auto')
    ax.set_title('Sample Count per Bin', fontweight='bold')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='Count')
    
    # Turbulent kinetic energy (TKE = 0.5 * (uu + vv + ww))
    ax = axes[0, 1]
    tke = 0.5 * (results['uu'] + results['vv'] + results['ww'])
    X, Y, Z = create_2d_field(results['x'], results['y'], tke)
    im = ax.pcolormesh(X, Y, Z, cmap='hot', shading='auto')
    ax.set_title('Turbulent Kinetic Energy (mm²)', fontweight='bold')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='TKE')
    
    # Histogram of Reynolds stress values
    ax = axes[1, 0]
    ax.hist(results['uu'], bins=50, alpha=0.5, label="u'u'", density=True)
    ax.hist(results['vv'], bins=50, alpha=0.5, label="v'v'", density=True)
    ax.hist(results['ww'], bins=50, alpha=0.5, label="w'w'", density=True)
    ax.set_xlabel('Reynolds Stress (mm²)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Distribution of Reynolds Stresses', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Mean displacement magnitude
    ax = axes[1, 1]
    displacement_mag = np.sqrt(results['mean_dx']**2 + 
                              results['mean_dy']**2 + 
                              results['mean_dz']**2)
    X, Y, Z = create_2d_field(results['x'], results['y'], displacement_mag)
    im = ax.pcolormesh(X, Y, Z, cmap='magma', shading='auto')
    ax.set_title('Mean Displacement Magnitude (mm)', fontweight='bold')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label='|Δ|')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Summary figure saved to {output_path}")
    plt.close()

def main():
    """Main visualization function."""
    # Load results
    results_file = Path("/Users/morgan/Documents/CODE/PIVTOOLS_FULL_STACK/PyPIVTools/piv_output_shear_numba/reynolds_stress_results.txt")
    output_dir = Path("/Users/morgan/Documents/CODE/PIVTOOLS_FULL_STACK/PyPIVTools/piv_output_shear_numba")
    
    print("Loading results...")
    results = load_results(results_file)
    
    print(f"Loaded {len(results['x'])} bins")
    print(f"X range: {results['x'].min():.2f} to {results['x'].max():.2f} mm")
    print(f"Y range: {results['y'].min():.2f} to {results['y'].max():.2f} mm")
    
    # Create main figure
    print("\nCreating main figure...")
    main_fig_path = output_dir / "reynolds_stress_visualization.png"
    plot_reynolds_stress_and_means(results, main_fig_path)
    
    # Create summary figure
    print("Creating summary figure...")
    summary_fig_path = output_dir / "reynolds_stress_summary.png"
    plot_statistics_summary(results, summary_fig_path)
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    main()
