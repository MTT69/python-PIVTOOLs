# Calibration Application Tests — Design

## Goal

Three new test suites verifying that the full calibration pipeline — detection, model fitting, AND model application — produces correct physical outputs. Extends the existing recovery tests (which only verify intrinsics recovery) to cover velocity calibration and coordinate conventions.

## Context

The existing recovery tests prove: "given synthetic images of a known camera, the pipeline recovers fx/fy/cx/cy within tight tolerances." But they don't test whether the recovered model is **applied** correctly to PIV vectors. The coordinate convention chain (`_uncal_to_raw → _pixels_to_world_mm → y-negation → unit conversion`) had multiple bugs (documented in MEMORY.md) and is the highest-risk part of the pipeline.

## Test 1: Apply-Calibration (Pinhole)

**File:** `unit-tests/test_apply_calibration_pinhole.py`

**What it tests:** Given a known camera model and known physical velocities, does `VectorCalibrator.process_run()` produce correct calibrated output?

### Setup

1. Load ground-truth camera from `synthetic_calibration/ground_truth.npz` (camera_matrix, dist_coeffs=zeros, rvec[0], tvec[0])
2. Create a grid of world points on the Z=0 plane (e.g. 10x8, 10mm spacing, centred on the calibration board)
3. Project world points to raw pixels via `cv2.projectPoints()`
4. Convert raw pixels to uncalibrated convention: `x_uncal = x_raw + 1`, `y_uncal = image_height - y_raw`
5. Choose known physical velocity: `ux_phys = 1.0 m/s`, `uy_phys = 0.5 m/s`, `dt = 1e-3 s`
6. Compute pixel displacement for each grid point:
   - Displaced world position = `world_pt + [ux_phys * dt * 1000, uy_phys * dt * 1000, 0]` (m/s → mm displacement over dt)
   - But y in world is y-down (model convention), and uy_phys is y-up, so: displaced_world_y = world_y - uy_phys * dt * 1000
   - Project displaced position to raw pixels
   - Pixel displacement = displaced_raw - start_raw
   - Convert displacement to uncalibrated: `dux = dx_raw`, `duy = -dy_raw` (y-flip on displacement)

### Fake .mat files

Write to a temp directory mimicking the uncalibrated PIV output:

```
tmpdir/
  uncalibrated_piv/N/Cam1/instantaneous/
    coordinates.mat   (x_uncal, y_uncal grids)
    B00001.mat        (ux_px, uy_px, b_mask=zeros)
  calibration/Cam1/dotboard_planar/model/
    dotboard_model.mat  (ground truth camera_matrix, dist_coeffs, rvecs, tvecs, image_size, dot_spacing_mm)
```

The dotboard_model.mat must match the format that `VectorCalibrator._load_calibration_model()` expects.

### Run

Create `VectorCalibrator(base_dir=tmpdir, camera_num=1, model_type="dotboard", dt=dt)` and call `process_run()`.

### Assertions

- Calibrated `ux` ≈ 1.0 m/s at all grid points (within 1%)
- Calibrated `uy` ≈ 0.5 m/s at all grid points (within 1%)
- Calibrated x-coordinates increase left-to-right (mm)
- Calibrated y-coordinates increase bottom-to-top (mm, y-up)
- Output files exist at the calibrated_piv path

### Edge cases

- Test with non-zero uy to verify y-negation is correct (this is where the historical bugs were)
- Points at corners of the grid to test near-edge calibration

---

## Test 2: Polynomial Model Recovery

**File:** Extend existing `test_charuco_calibration_recovery.py` and `test_dotboard_calibration_recovery.py`

**What it tests:** Does `model_type="polynomial"` produce a working polynomial model with reasonable mm_per_pixel and low RMS error?

### Approach

Add a second calibration fixture parametrized by model_type:

```python
@pytest.fixture(scope="module", params=["pinhole", "polynomial"])
def calibration_result(request, ground_truth):
    model_type = request.param
    calibrator = ChArUcoCalibrator(..., model_type=model_type)
    result = calibrator.process_camera(1, save_visualizations=False)
    ...
```

### Complication

The polynomial path calls `save_polynomial_to_config()` which writes to `config.yaml`. In tests we don't have a real config. Options:
- **Option A:** Mock `get_config()` to return a temp config (complex)
- **Option B:** Pass `config=None` — the code handles this by skipping config save but still returning the fit result dict
- **Option C:** Create separate non-parametrized tests for polynomial that handle config setup

Recommend **Option C** — separate test functions for polynomial, since the assertions are completely different (no camera_matrix/dist_coeffs, instead check mm_per_pixel and rms_fit_error).

### Assertions (polynomial)

- `result["success"]` is True
- `result["mm_per_pixel"]` is reasonable (close to expected: board_physical_size / image_size)
- `result["rms_fit_error_px"]` < 1.0 px
- Model file exists on disk

---

## Test 3: Distorted Synthetic Data

**File:** `unit-tests/test_calibration_recovery_distorted.py` (new file, or extend generator)

**What it tests:** Does the pipeline correctly recover non-zero distortion coefficients?

### Approach

1. Modify `generate_synthetic_calibration.py` to accept optional `--dist-coeffs k1 k2` CLI args
2. Add a second set of synthetic data with known distortion, e.g. `k1=-0.1, k2=0.01` (barrel distortion)
3. Save to `synthetic_calibration_distorted/` with its own `ground_truth.npz`
4. New test file runs the same charuco/dotboard recovery but verifies:
   - Recovered `k1` within 20% of true value
   - Recovered `k2` within 50% of true value (k2 is harder to recover)
   - `fx`, `fy` still within 2%
   - `cx`, `cy` still within 3px
   - RMS < 1.0px (slightly relaxed since distortion adds complexity)

### Generator changes

- `generate_synthetic_calibration.py` already passes `dist_coeffs = np.zeros(5)` to `cv2.projectPoints()` — just need to make it configurable
- ChArUco warping: the frontal image → perspective warp approach still works because we project ALL corner 3D positions through the distorted camera model, then warp
- Dotboard rendering: `cv2.projectPoints()` already handles distortion, so dots land at distorted positions naturally

### Risk

Distorted images may be harder for the detector (especially dotboard blob detection, where heavily distorted dots near edges become elliptical). May need to use moderate distortion values.

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `unit-tests/test_apply_calibration_pinhole.py` | **Create** — apply-calibration with fake .mat files |
| `unit-tests/test_polynomial_calibration_recovery.py` | **Create** — polynomial model fitting from charuco + dotboard |
| `unit-tests/test_calibration_recovery_distorted.py` | **Create** — recovery with known distortion |
| `unit-tests/generate_synthetic_calibration.py` | **Modify** — add `--dist-coeffs` option, save separate distorted dataset |

No production code changes.

## Dependencies

- Test 1 (apply-calibration) requires the existing `synthetic_calibration/ground_truth.npz` (already generated)
- Test 2 (polynomial) requires the existing synthetic images (already generated)
- Test 3 (distorted) requires new distorted synthetic images (generator must be run with `--dist-coeffs`)

## Execution order

Tests 1, 2, 3 are independent and can be implemented in any order. Recommend starting with Test 1 (apply-calibration) since it tests the highest-risk code path.
