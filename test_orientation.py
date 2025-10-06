import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from config import Config
from image_handling.load_images import read_pair
from pypivtools.piv.piv_backend.factory import make_correlator_backend

config = Config()
camera_num = config.camera_numbers[0]
source_path = config.source_paths[0]
camera_path = source_path / f"Cam{camera_num}"
image_pair = read_pair(1, camera_path, config)
images = image_pair[np.newaxis, ...]
correlator = make_correlator_backend(config)
piv_result = correlator.correlate_batch(images, config=config, mask=None)

ux = piv_result.passes[0].ux_mat
uy = piv_result.passes[0].uy_mat

print("="*60)
print("VELOCITY ORIENTATION CHECK")
print("="*60)
print(f"Shape: {ux.shape}")
print()
print("Ux (should be horizontal, LARGE for channel flow):")
print(f"  Mean: {np.nanmean(ux):.2f}")
print(f"  Range: [{np.nanmin(ux):.2f}, {np.nanmax(ux):.2f}]")
print()
print("Uy (should be vertical, SMALL for channel flow):")
print(f"  Mean: {np.nanmean(uy):.2f}")
print(f"  Range: [{np.nanmin(uy):.2f}, {np.nanmax(uy):.2f}]")
print()

if np.abs(np.nanmean(ux)) > 10:
    print("✓ Ux is LARGE")
else:
    print("✗ Ux is SMALL - WRONG!")

if np.abs(np.nanmean(uy)) < 2:
    print("✓ Uy is SMALL")
else:
    print("✗ Uy is LARGE - WRONG!")
