# Transforms and Merging Guide

This guide explains how to apply geometric transformations and merge multi-camera vector fields in PIVTools. Both operations work with **instantaneous** and **ensemble** data types.

## Table of Contents

1. [Overview](#overview)
2. [Data Types](#data-types)
3. [Transforms](#transforms)
   - [GUI Usage](#gui-usage-transforms)
   - [CLI Usage](#cli-usage-transforms)
   - [Config Settings](#config-settings-transforms)
4. [Merging](#merging)
   - [GUI Usage](#gui-usage-merging)
   - [CLI Usage](#cli-usage-merging)
   - [Config Settings](#config-settings-merging)
5. [File Structure](#file-structure)
6. [Ensemble-Specific Considerations](#ensemble-specific-considerations)

---

## Overview

### Transforms

Geometric transformations modify vector fields in place. Supported operations:

| Operation | Description |
|-----------|-------------|
| `flip_ud` | Flip vertically (up-down) |
| `flip_lr` | Flip horizontally (left-right) |
| `rotate_90_cw` | Rotate 90 degrees clockwise |
| `rotate_90_ccw` | Rotate 90 degrees counter-clockwise |
| `rotate_180` | Rotate 180 degrees |

Transforms are applied to:
- Vector components (ux, uy, uz)
- Masks (b_mask)
- Stress fields (UU_stress, VV_stress, UV_stress) for ensemble data
- Coordinate grids (x, y)

### Merging

Merging combines overlapping vector fields from multiple cameras into a single unified field using Hanning window blending. This produces smooth transitions in overlap regions.

Merging preserves:
- Velocity fields (ux, uy, uz)
- Masks (b_mask)
- Stress fields for ensemble data (UU_stress, VV_stress, UV_stress)

---

## Data Types

PIVTools supports two primary data types for both transforms and merging:

| Type | File Structure | Result Key | Use Case |
|------|---------------|------------|----------|
| **instantaneous** | `00001.mat`, `00002.mat`, ... | `piv_result` | Per-frame analysis |
| **ensemble** | `ensemble_result.mat` | `ensemble_result` | Time-averaged analysis |

The `type_name` parameter controls which data type is processed.

---

## Transforms

### GUI Usage (Transforms)

#### Single Frame Transform

1. Open the Vector Viewer
2. Navigate to the desired frame
3. Select the data source (Calibrated, Ensemble, Merged, etc.)
4. Use the transform buttons in the toolbar:
   - **Rotate Left** (counter-clockwise 90)
   - **Rotate Right** (clockwise 90)
   - **Flip Horizontal**
   - **Flip Vertical**

Transforms are applied immediately to the current frame and stored as "pending transformations."

#### Apply to All Frames

After applying transforms to a single frame:

1. Click **"Apply to All Frames"** button
2. The pending transformations from the current frame are applied to all frames across all cameras
3. A progress dialog shows the operation status

#### Clear Transforms

To reset a frame to its original state:

1. Click **"Clear Transforms"** button
2. The frame is restored from its `_original` backup

### CLI Usage (Transforms)

#### Basic Command

```bash
pivtools-cli transform [options]
```

#### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--camera` | `-c` | Camera number (default: all from config) |
| `--type-name` | `-t` | Data type: `instantaneous` or `ensemble` |
| `--operations` | `-o` | Comma-separated transforms to apply |
| `--merged` | `-m` | Transform merged data instead of per-camera |

#### Examples

**Transform camera 1 with specific operations:**
```bash
pivtools-cli transform -c 1 -o "flip_ud,rotate_90_cw"
```

**Transform ensemble data:**
```bash
pivtools-cli transform -t ensemble
```

**Transform using config.yaml settings:**
```bash
pivtools-cli transform
```

#### Direct Python Execution

You can also run the transform module directly:

```bash
cd /path/to/experiment
python -m pivtools_gui.transforms.transform_production
```

This uses `config.yaml` in the current directory.

### Config Settings (Transforms)

Add these settings to your `config.yaml`:

```yaml
transforms:
  # Base path index (into paths.base_paths)
  base_path_idx: 0

  # Data type: "instantaneous" or "ensemble"
  type_name: instantaneous

  # Per-camera transform operations
  cameras:
    1:
      operations:
        - rotate_90_cw
    2:
      operations:
        - flip_ud
        - rotate_90_ccw
```

#### Transform Simplification

Transforms are automatically simplified before application:

- `flip_ud` + `flip_ud` = no operation
- `rotate_90_cw` + `rotate_90_cw` = `rotate_180`
- `rotate_90_cw` + `rotate_90_ccw` = no operation

---

## Merging

### GUI Usage (Merging)

#### Merge Workflow

1. Open the Merging Panel
2. Select base path and cameras to merge
3. Choose data type (instantaneous or ensemble)
4. Click **"Start Merge"**
5. Monitor progress via the status display

#### Viewing Merged Data

After merging:

1. In Vector Viewer, select **"Merged"** from the data source dropdown
2. Merged data appears in `calibrated_piv/{N}/merged/Cam1/{type}/`

### CLI Usage (Merging)

#### Basic Command

```bash
pivtools-cli merge [options]
```

#### Options

| Option | Short | Description |
|--------|-------|-------------|
| `--cameras` | `-c` | Comma-separated camera numbers to merge |
| `--type-name` | `-t` | Data type: `instantaneous` or `ensemble` |

#### Examples

**Merge cameras 1 and 2 for instantaneous data:**
```bash
pivtools-cli merge -c 1,2 -t instantaneous
```

**Merge ensemble data using config settings:**
```bash
pivtools-cli merge -t ensemble
```

**Merge with default config:**
```bash
pivtools-cli merge
```

### Config Settings (Merging)

Add these settings to your `config.yaml`:

```yaml
merging:
  # Data type: "instantaneous" or "ensemble"
  type_name: instantaneous

  # Base path index (into paths.base_paths)
  base_path_idx: 0

  # Cameras to merge (must have at least 2)
  cameras:
    - 1
    - 2
```

---

## File Structure

### Instantaneous Data

```
base_path/
└── calibrated_piv/
    └── {num_frame_pairs}/
        ├── Cam1/
        │   └── instantaneous/
        │       ├── 00001.mat          # piv_result with runs
        │       ├── 00002.mat
        │       └── coordinates.mat
        ├── Cam2/
        │   └── instantaneous/
        │       ├── 00001.mat
        │       ├── 00002.mat
        │       └── coordinates.mat
        └── merged/
            └── Cam1/
                └── instantaneous/
                    ├── 00001.mat      # Merged result
                    └── coordinates.mat
```

### Ensemble Data

```
base_path/
└── calibrated_piv/
    └── {num_frame_pairs}/
        ├── Cam1/
        │   └── ensemble/
        │       ├── ensemble_result.mat  # ensemble_result with runs
        │       └── coordinates.mat
        ├── Cam2/
        │   └── ensemble/
        │       ├── ensemble_result.mat
        │       └── coordinates.mat
        └── merged/
            └── Cam1/
                └── ensemble/
                    ├── ensemble_result.mat  # Merged ensemble
                    └── coordinates.mat
```

---

## Ensemble-Specific Considerations

### Stress Fields

Ensemble data contains additional stress tensor fields that are preserved through transforms and merging:

| Field | Description |
|-------|-------------|
| `UU_stress` | u'u' (streamwise normal stress) |
| `VV_stress` | v'v' (transverse normal stress) |
| `UV_stress` | u'v' (Reynolds shear stress) |

These fields are:
- **Transformed** alongside velocity components using the same operations
- **Merged** using the same Hanning blend weights as velocity fields

### Single File Processing

Unlike instantaneous data with multiple numbered files, ensemble data has a single `ensemble_result.mat` per camera. The processing automatically handles this difference:

- **Instantaneous**: Processes `00001.mat`, `00002.mat`, etc.
- **Ensemble**: Processes only `ensemble_result.mat`

### Config Example for Ensemble

```yaml
# Set calibration to work with ensemble data
calibration:
  piv_type: ensemble

# Transforms for ensemble
transforms:
  type_name: ensemble
  cameras:
    1:
      operations:
        - rotate_90_cw

# Merging for ensemble
merging:
  type_name: ensemble
  cameras:
    - 1
    - 2

# Statistics computed from ensemble
statistics:
  type_name: ensemble
```

---

## Complete Config Example

```yaml
paths:
  base_paths:
    - /path/to/experiment
  source_paths:
    - /path/to/images
  camera_numbers:
    - 1
    - 2
  camera_count: 2

images:
  num_images: 100
  num_frame_pairs: 100
  vector_format: '%05d.mat'

# For instantaneous processing
transforms:
  base_path_idx: 0
  type_name: instantaneous
  cameras:
    1:
      operations:
        - flip_ud
    2:
      operations:
        - rotate_90_cw

merging:
  type_name: instantaneous
  cameras:
    - 1
    - 2

# For ensemble processing, change type_name to "ensemble":
# transforms:
#   type_name: ensemble
#   cameras: ...
#
# merging:
#   type_name: ensemble
```

---

## Troubleshooting

### Transform Not Applied

1. **Check data source**: Ensure you're viewing the correct data source (Calibrated vs Merged, Instantaneous vs Ensemble)
2. **Verify type_name**: The `type_name` must match your data (e.g., `ensemble` for ensemble data)
3. **Check file exists**: Verify the `.mat` files exist in the expected location

### Merge Fails

1. **Minimum cameras**: Merging requires at least 2 cameras
2. **Overlapping regions**: Cameras must have overlapping coordinate regions
3. **Matching runs**: All cameras must have the same run structure

### Statistics After Transform

Transforms do **not** update statistics files. After applying transforms:

1. Clear existing statistics (optional)
2. Recompute statistics via GUI or CLI:
   ```bash
   pivtools-cli statistics -t instantaneous
   # or for ensemble:
   pivtools-cli statistics -t ensemble
   ```

---

## API Reference

### Python API (Transforms)

```python
from pivtools_gui.transforms.transform_production import TransformProcessor

processor = TransformProcessor(
    base_dir="/path/to/experiment",
    camera_transforms={1: ["flip_ud", "rotate_90_cw"]},
    type_name="instantaneous",  # or "ensemble"
    use_merged=False,
)

result = processor.process_all_cameras()
```

### Python API (Merging)

```python
from pivtools_gui.vector_merging.vector_merger import VectorMerger

merger = VectorMerger(
    base_dir="/path/to/experiment",
    cameras=[1, 2],
    type_name="instantaneous",  # or "ensemble"
)

result = merger.merge_all_frames()
```
