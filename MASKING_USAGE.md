# Masking Feature Usage

## Overview

The masking feature allows you to invalidate PIV vectors in specific regions of your images. This is useful for:
- Excluding regions with obstructions (walls, objects)
- Removing boundary effects
- Focusing analysis on regions of interest

## How Masks Work

- Masks are **boolean arrays** of shape `(H, W)` matching your image dimensions
- `True` values indicate **masked regions** (vectors will be invalidated/set to NaN)
- `False` values indicate **valid regions** (vectors computed normally)
- Masks are loaded **once per camera** at the start of processing for efficiency

## Configuration

### 1. Enable Masking in `config.yaml`

```yaml
masking:
  enabled: true                    # Set to false to disable masking
  mask_file_pattern: mask_Cam%d.mat  # Filename pattern (%d = camera number)
```

### 2. Mask File Location

Mask files are expected in your source path (same directory as camera folders):

```
source_path/
├── Cam1/
│   ├── image_001.tif
│   └── ...
├── Cam2/
│   └── ...
├── mask_Cam1.mat  # Mask for Camera 1
└── mask_Cam2.mat  # Mask for Camera 2
```

## Creating Masks

### Using the Flask Masking App

1. Start the masking server (see `src/masking/app/`)
2. Draw polygons to define masked regions
3. Save the mask using the `/save_mask_array` endpoint
4. The mask is automatically saved as `mask_Cam{N}.mat` in the source path

### Mask File Format

Masks are saved as MATLAB `.mat` files containing:
- `mask`: Boolean numpy array of shape `(H, W)`
- `polygons`: Array of polygon definitions (for visualization/editing)

## How Masking is Applied

1. **Load Once Per Camera**: When processing starts, `load_mask_for_camera()` loads the mask file
2. **Pass Through Pipeline**: The mask is passed to all PIV processing functions
3. **Vector Invalidation**: During correlation, vectors whose window centers fall in masked regions are invalidated
4. **Set to NaN**: Invalidated vectors are set to NaN (consistent with other invalid vectors)

## Implementation Details

### Key Functions

- **`load_mask_for_camera()`** in `src/image_handling/load_images.py`
  - Loads mask file for a specific camera
  - Returns `None` if masking disabled or file not found
  - Logs mask statistics (percentage of masked pixels)

- **`_apply_mask_to_vectors()`** in `pypivtools/piv/piv_backend/cpu_instantaneous.py`
  - Samples mask at window center locations
  - Returns boolean array indicating which vectors to invalidate
  - Integrated into the correlation pipeline

### Modified Files

1. `config.yaml` - Added masking configuration section
2. `src/config.py` - Added masking properties and `get_mask_path()` method
3. `src/image_handling/load_images.py` - Added `load_mask_for_camera()` function
4. `pypivtools/example.py` - Loads mask and passes to PIV pipeline
5. `pypivtools/piv/piv.py` - Updated to accept and pass mask parameter
6. `pypivtools/piv/piv_backend/base.py` - Updated base class signature
7. `pypivtools/piv/piv_backend/cpu_instantaneous.py` - Applies mask to vectors

## Example Usage

```python
from config import Config
from image_handling.load_images import load_images, load_mask_for_camera
from pypivtools.piv.piv import perform_piv_and_save

# Load configuration
config = Config()

# Load images for camera 1
images = load_images(camera_num=1, config=config)

# Load mask for camera 1 (returns None if disabled)
mask = load_mask_for_camera(camera_num=1, config=config)

# Perform PIV with masking
results = perform_piv_and_save(
    images=images,
    config=config,
    client=dask_client,
    output_path=output_path,
    mask=mask  # Pass mask here
)
```

## Disabling Masking

To disable masking:

```yaml
masking:
  enabled: false  # Vectors will be computed in all regions
```

Or simply remove the mask files from your source path.

## Notes

- Masks are applied **after** edge detection and other built-in validity checks
- Masked vectors are still used for visualization of masked regions
- The same mask is used for all passes in multi-pass PIV
- Mask files are small (~few KB for typical image sizes) and load quickly
