# /Users/morgan/Documents/CODE/PIVTOOLS_FULL_STACK/PyPIVTools/process_displacement_reynolds.py

import numpy as np
from pathlib import Path
from collections import defaultdict
import gc
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

class BinnedReynoldsStress:
    """
    Compute binned mean displacement and Reynolds stress tensor
    in a memory-efficient manner.
    """
    
    def __init__(self, bin_size=1.0):
        """
        Initialize with specified bin size in mm.
        
        Args:
            bin_size: Bin size in x and y directions (mm)
        """
        self.bin_size = bin_size
        # Use defaultdict to store statistics per bin
        self.bin_stats = defaultdict(lambda: {
            'count': 0,
            'dz_sum': 0.0,
            'dz_sq_sum': 0.0,
            'dx_sum': 0.0,
            'dy_sum': 0.0,
            'dx_sq_sum': 0.0,
            'dy_sq_sum': 0.0,
            'dx_dy_sum': 0.0,
            'dx_dz_sum': 0.0,
            'dy_dz_sum': 0.0
        })
    
    def _get_bin_key(self, x, y):
        """Convert x, y coordinates to bin indices."""
        bin_x = int(np.floor(x / self.bin_size))
        bin_y = int(np.floor(y / self.bin_size))
        return (bin_x, bin_y)
    
    def process_pair(self, file_a_path, file_b_path):
        """
        Process a single pair of files.
        
        Args:
            file_a_path: Path to _A.data file
            file_b_path: Path to _B.data file
        """
        # Load data
        data_a = np.loadtxt(file_a_path)
        data_b = np.loadtxt(file_b_path)
        
        # Compute displacements
        dx = data_b[:, 0] - data_a[:, 0]
        dy = data_b[:, 1] - data_a[:, 1]
        dz = data_b[:, 2] - data_a[:, 2]
        
        # Use average position for binning
        x_avg = (data_a[:, 0] + data_b[:, 0]) / 2
        y_avg = (data_a[:, 1] + data_b[:, 1]) / 2
        
        # Accumulate statistics for each point
        for i in range(len(x_avg)):
            bin_key = self._get_bin_key(x_avg[i], y_avg[i])
            stats = self.bin_stats[bin_key]
            
            stats['count'] += 1
            stats['dz_sum'] += dz[i]
            stats['dz_sq_sum'] += dz[i]**2
            stats['dx_sum'] += dx[i]
            stats['dy_sum'] += dy[i]
            stats['dx_sq_sum'] += dx[i]**2
            stats['dy_sq_sum'] += dy[i]**2
            stats['dx_dy_sum'] += dx[i] * dy[i]
            stats['dx_dz_sum'] += dx[i] * dz[i]
            stats['dy_dz_sum'] += dy[i] * dz[i]
    
    def merge_stats(self, other_stats):
        """Merge statistics from another dictionary (for parallel processing)."""
        for bin_key, stats in other_stats.items():
            my_stats = self.bin_stats[bin_key]
            my_stats['count'] += stats['count']
            my_stats['dz_sum'] += stats['dz_sum']
            my_stats['dz_sq_sum'] += stats['dz_sq_sum']
            my_stats['dx_sum'] += stats['dx_sum']
            my_stats['dy_sum'] += stats['dy_sum']
            my_stats['dx_sq_sum'] += stats['dx_sq_sum']
            my_stats['dy_sq_sum'] += stats['dy_sq_sum']
            my_stats['dx_dy_sum'] += stats['dx_dy_sum']
            my_stats['dx_dz_sum'] += stats['dx_dz_sum']
            my_stats['dy_dz_sum'] += stats['dy_dz_sum']
    
    def compute_results(self):
        """
        Compute final mean displacements and Reynolds stress tensor for each bin.
        
        Returns:
            dict: Results keyed by bin index with mean displacements and Reynolds stresses
        """
        results = {}
        
        for bin_key, stats in self.bin_stats.items():
            n = stats['count']
            if n == 0:
                continue
            
            # Mean displacements
            mean_dx = stats['dx_sum'] / n
            mean_dy = stats['dy_sum'] / n
            mean_dz = stats['dz_sum'] / n
            
            # Fluctuations (Reynolds decomposition: u' = u - <u>)
            # Reynolds stresses: <u'_i u'_j> = <u_i u_j> - <u_i><u_j>
            uu = stats['dx_sq_sum'] / n - mean_dx**2  # <u'u'>
            vv = stats['dy_sq_sum'] / n - mean_dy**2  # <v'v'>
            ww = stats['dz_sq_sum'] / n - mean_dz**2  # <w'w'>
            uv = stats['dx_dy_sum'] / n - mean_dx * mean_dy  # <u'v'>
            uw = stats['dx_dz_sum'] / n - mean_dx * mean_dz  # <u'w'>
            vw = stats['dy_dz_sum'] / n - mean_dy * mean_dz  # <v'w'>
            
            results[bin_key] = {
                'bin_center_x': (bin_key[0] + 0.5) * self.bin_size,
                'bin_center_y': (bin_key[1] + 0.5) * self.bin_size,
                'count': n,
                'mean_dx': mean_dx,
                'mean_dy': mean_dy,
                'mean_dz': mean_dz,
                'reynolds_stress': {
                    'uu': uu,
                    'vv': vv,
                    'ww': ww,
                    'uv': uv,
                    'uw': uw,
                    'vw': vw
                }
            }
        
        return results
    
    def save_results(self, output_path):
        """Save results to file."""
        results = self.compute_results()
        
        with open(output_path, 'w') as f:
            f.write("# bin_center_x bin_center_y count mean_dx mean_dy mean_dz ")
            f.write("uu vv ww uv uw vw\n")
            
            for bin_key in sorted(results.keys()):
                res = results[bin_key]
                rs = res['reynolds_stress']
                f.write(f"{res['bin_center_x']:.6f} {res['bin_center_y']:.6f} ")
                f.write(f"{res['count']} ")
                f.write(f"{res['mean_dx']:.6f} {res['mean_dy']:.6f} {res['mean_dz']:.6f} ")
                f.write(f"{rs['uu']:.6f} {rs['vv']:.6f} {rs['ww']:.6f} ")
                f.write(f"{rs['uv']:.6f} {rs['uw']:.6f} {rs['vw']:.6f}\n")


def process_pair_worker(file_paths, bin_size):
    """
    Worker function for parallel processing of a single file pair.
    
    Args:
        file_paths: Tuple of (file_a_path, file_b_path)
        bin_size: Bin size in mm
    
    Returns:
        Dictionary of bin statistics
    """
    file_a_path, file_b_path = file_paths
    
    # Load data
    data_a = np.loadtxt(file_a_path)
    data_b = np.loadtxt(file_b_path)
    
    # Compute displacements
    dx = data_b[:, 0] - data_a[:, 0]
    dy = data_b[:, 1] - data_a[:, 1]
    dz = data_b[:, 2] - data_a[:, 2]
    
    # Use average position for binning
    x_avg = (data_a[:, 0] + data_b[:, 0]) / 2
    y_avg = (data_a[:, 1] + data_b[:, 1]) / 2
    
    # Accumulate statistics
    bin_stats = defaultdict(lambda: {
        'count': 0,
        'dz_sum': 0.0,
        'dz_sq_sum': 0.0,
        'dx_sum': 0.0,
        'dy_sum': 0.0,
        'dx_sq_sum': 0.0,
        'dy_sq_sum': 0.0,
        'dx_dy_sum': 0.0,
        'dx_dz_sum': 0.0,
        'dy_dz_sum': 0.0
    })
    
    for i in range(len(x_avg)):
        bin_x = int(np.floor(x_avg[i] / bin_size))
        bin_y = int(np.floor(y_avg[i] / bin_size))
        bin_key = (bin_x, bin_y)
        
        stats = bin_stats[bin_key]
        stats['count'] += 1
        stats['dz_sum'] += dz[i]
        stats['dz_sq_sum'] += dz[i]**2
        stats['dx_sum'] += dx[i]
        stats['dy_sum'] += dy[i]
        stats['dx_sq_sum'] += dx[i]**2
        stats['dy_sq_sum'] += dy[i]**2
        stats['dx_dy_sum'] += dx[i] * dy[i]
        stats['dx_dz_sum'] += dx[i] * dz[i]
        stats['dy_dz_sum'] += dy[i] * dz[i]
    
    return dict(bin_stats)


def main():
    """Main processing function."""
    data_dir = Path("/Users/morgan/Documents/CODE/PIVTOOLS_FULL_STACK/PyPIVTools/piv_output_shear_numba")
    output_file = Path("/Users/morgan/Documents/CODE/PIVTOOLS_FULL_STACK/PyPIVTools/piv_output_shear_numba/reynolds_stress_results.txt")
    
    bin_size = 1.0
    n_cores = cpu_count() - 1  # Leave one core free
    
    # Collect all file pairs
    print("Scanning for file pairs...")
    file_pairs = []
    for i in range(1, 1001):
        file_a = data_dir / f"B{i:05d}_A.data"
        file_b = data_dir / f"B{i:05d}_B.data"
        
        if file_a.exists() and file_b.exists():
            file_pairs.append((file_a, file_b))
        else:
            print(f"Warning: Missing pair {i:05d}")
    
    print(f"Found {len(file_pairs)} file pairs")
    print(f"Using {n_cores} CPU cores for parallel processing")
    
    # Initialize processor
    processor = BinnedReynoldsStress(bin_size=bin_size)
    
    # Process pairs in parallel with progress bar
    worker_fn = partial(process_pair_worker, bin_size=bin_size)
    
    with Pool(processes=n_cores) as pool:
        results = list(tqdm(
            pool.imap(worker_fn, file_pairs),
            total=len(file_pairs),
            desc="Processing pairs",
            unit="pair"
        ))
    
    # Merge all results
    print("Merging results from parallel workers...")
    for result in tqdm(results, desc="Merging", unit="result"):
        processor.merge_stats(result)
    
    # Save results
    print("Computing and saving final results...")
    processor.save_results(output_file)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()