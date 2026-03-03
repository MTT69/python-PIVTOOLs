# Stereo PIV Self-Calibration: Implementation Specification

**Document Version:** 1.0  
**Author:** Morgan Taylor  
**Date:** February 2025  
**Purpose:** Technical specification for integrating stereo self-calibration into an existing PIV processing pipeline.

---

## 1. Overview

### 1.1 Problem Statement

Stereo PIV calibration is performed using a calibration target at a known Z-position (typically Z=0). However, the laser sheet illuminating particles during experiments may not coincide exactly with this calibration plane. This misalignment manifests as:

- **Z-offset:** The laser sheet is displaced from Z=0 by some distance ΔZ
- **Tilt:** The laser sheet is rotated relative to the calibration plane (tilt about X and Y axes)

If uncorrected, this misalignment introduces systematic errors in the reconstructed three-component velocity field.

### 1.2 Solution: Self-Calibration

Self-calibration uses the particle images themselves to detect and correct for laser sheet misalignment. The method:

1. Dewarps particle images from both cameras to a common reference plane using the existing pinhole calibration
2. Cross-correlates Camera 1 against Camera 2 (same time instant, different viewpoints)
3. Measures the residual disparity field (non-zero disparity indicates misalignment)
4. Fits the disparity to a plane to extract Z-offset and tilt angles
5. Corrects the dewarping maps to account for the true laser sheet position
6. Iterates until disparity converges to an acceptable level

### 1.3 Key Distinction

| Correlation Type | Images Correlated | What It Measures |
|------------------|-------------------|------------------|
| PIV (temporal) | Frame A vs Frame B (same camera, different times) | Particle displacement (velocity) |
| Self-cal (spatial) | Camera 1 vs Camera 2 (same time, different cameras) | Calibration error (disparity) |

---

## 2. Prerequisites

### 2.1 Required Inputs

1. **Existing camera calibration**:
   - Pinhole model: K (intrinsic matrix), distortion coefficients, R (rotation), t (translation)
   - images from camera 1 and 2

### 2.2 Expected Misalignment Ranges

| Parameter | Typical Range | Severe Misalignment |
|-----------|---------------|---------------------|
| Z-offset | 0.1–0.5 mm | > 1 mm |
| Tilt angles | 0.05–0.2° | > 0.5° |
| Initial disparity | 1–3 pixels | > 5 pixels |

If initial disparity exceeds ~10 pixels, check for gross calibration errors before proceeding.

---

## 3. Algorithm: Detailed Steps

### 3.1 Step 1: Dewarp Both Cameras to Common Plane

Using the existing calibration, dewarp images from both cameras to the same world coordinate system assuming Z=0.

**Input:**
- Raw image from Camera 1
- Raw image from Camera 2
- Dewarping maps for each camera (or parameters to compute them)

**Process:**
```
dewarped_cam1 = remap(raw_cam1, map_x_cam1, map_y_cam1)
dewarped_cam2 = remap(raw_cam2, map_x_cam2, map_y_cam2)
```

**Output:**
- Two dewarped images in the same coordinate system
- If calibration were perfect and particles were at Z=0, these would be identical

**Implementation note:** Use cubic interpolation for remapping. Border pixels should be set to zero or marked as invalid. (look at how this is achieved in the main piv pipeline)

---

### 3.2 Step 2: Define Correlation Grid

Disparity varies smoothly across the image (it's caused by a planar offset). We don't need to compute it at every pixel; a coarse grid suffices - use windowing like the normal piv alogorithm as implemented there offer window size as we do. we dont need single mode or anything here....

---

### 3.3 Step 3: Ensemble Cross-Correlation

This is the core of the method. For each grid point, accumulate correlation planes across all image pairs, then extract a single disparity from the ensemble. we have done this in detail in teh ensemble pipeline - however we dont need the reynolds stresses here - just look in the pipeline how we accumulate correlaiton maps and subtract the minimum - wee should make a new peak fitter that just operates on the correlaiton planes here we do not need the complexities of the marquadt script - we can make this in python not c for now. 


#### 3.3.3 Accumulate Correlation Planes

**Critical:** Do NOT extract the peak from each pair. Instead, sum the correlation planes:

```
ensemble_correlation[i, j] += correlation_plane
```

After processing all N image pairs, each grid point has an accumulated correlation plane that is the sum of N individual planes. again this is already demonstrated in the ensemble pipeline so don't deviate from teh core approach please. find the peak fit using a 6 point fit as we outline in the instantaneous pipeline....


### 3.4 Step 4: Extract Disparity from Ensemble Peak

After accumulating all pairs, extract a single disparity vector at each grid point.

---

### 3.5 Step 5: Fit Disparity to Plane

The disparity field should be approximately planar (linear in X and Y) because it arises from a planar laser sheet offset.

#### 3.5.1 Physical Model

For a symmetric stereo setup with cameras at ±θ from normal:

```
Z(X, Y) = Z_offset + X × tan(tilt_y) + Y × tan(tilt_x)

disparity_x ≈ -2 × Z(X, Y) × tan(θ) / mm_per_pixel
```

The sign depends on camera ordering convention so please look our existing stereo calibration methodology.

#### 3.5.2 Plane Fit

Fit the disparity field to a plane using least squares:

```
disparity_x = a + b×X + c×Y

where X, Y are in mm (grid coordinates × mm_per_pixel)
```

Solve the linear system:

```
| 1  X₁  Y₁ |   | a |     | dx₁ |
| 1  X₂  Y₂ | × | b |  =  | dx₂ |
| ...       |   | c |     | ... |
```

#### 3.5.3 Extract Physical Parameters

```
conversion = mm_per_pixel / (2 × tan(θ))

Z_offset = -a × conversion
tilt_y   = arctan(-b × conversion)   // dZ/dX
tilt_x   = arctan(-c × conversion)   // dZ/dY
```

#### 3.5.4 Compute Fit Residual

```
disparity_fitted = a + b×X + c×Y
residual_rms = sqrt(mean((disparity_x - disparity_fitted)²))
```

A high residual (> 0.5 px) indicates either:
- Non-planar laser sheet (curved or wrinkled)
- Poor correlation quality
- Outliers in the disparity field

---

### 3.6 Step 6: Correct the Dewarping Maps

The original maps project world coordinates to image coordinates assuming particles are at Z=0. The corrected maps should assume particles are at Z(X, Y).

#### 3.6.1 For Pinhole Model

Rebuild the dewarping maps using the corrected Z surface:

```
For each output pixel (i, j):
    X = x_min + j × mm_per_pixel
    Y = y_min + i × mm_per_pixel
    Z = Z_offset + X × tan(tilt_y) + Y × tan(tilt_x)   // NEW: not zero
    
    world_point = [X, Y, Z]
    image_point = project(world_point, K, dist, R, t)
    
    map_x[i, j] = image_point.x
    map_y[i, j] = image_point.y
```

**Key insight:** The pinhole parameters (K, dist, R, t) do NOT change. What changes is the Z value used when building the maps.

#### 3.6.2 For Polynomial Mapping

Apply the correction as a Z-dependent shift in the mapping coefficients, or regenerate the mapping at the corrected Z surface. This depends on your polynomial formulation.

---

### 3.7 Step 7: Iterate Until Convergence

One pass of self-calibration may not fully correct the misalignment due to:
- Non-linearity in the projection
- Correlation uncertainty
- Approximations in the plane fit

Iterate the process:

```
cumulative_Z = 0
cumulative_tilt_x = 0
cumulative_tilt_y = 0

for iteration in range(max_iterations):
    
    1. Build maps using cumulative correction
    2. Compute ensemble disparity
    3. Check convergence: if RMS disparity < threshold, stop
    4. Fit residual misalignment
    5. Add to cumulative correction:
       cumulative_Z += delta_Z
       cumulative_tilt_x += delta_tilt_x
       cumulative_tilt_y += delta_tilt_y
```

#### 3.7.1 Convergence Criteria

```
RMS disparity < 0.1 pixels  →  Converged (good)
RMS disparity < 0.5 pixels  →  Acceptable
RMS disparity > 1.0 pixels after 5 iterations  →  Problem, investigate
```

#### 3.7.2 Typical Convergence

- Realistic misalignment: 2–3 iterations
- Severe misalignment: 4–6 iterations
- If not converging after 10 iterations: flag error


## 4. Quality Metrics and Validation

### 4.1 Correlation Quality

At each grid point, record the peak correlation value:

```
peak_quality = max(ensemble_correlation[i, j]) / n_images
```

- peak_quality > 0.5: Good
- peak_quality 0.3–0.5: Acceptable
- peak_quality < 0.3: Suspect; consider excluding from fit

### 4.3 Outlier Detection

employ outlier detection as in the main pipelines - do not recode use existing modules where possible this goes for everything
```

### 4.4 Residual Analysis

After plane fit:
- Mean residual should be ~0
- RMS residual indicates fit quality
- Systematic patterns in residual indicate non-planar sheet or higher-order errors

---

## 5. Diagnostic Figures

Generate the following figures for validation and documentation.

### 5.1 Figure 1: Dewarped Image Overlay

**Purpose:** Visual check of particle alignment before/after correction.

**Content:**
- RGB composite: Red = Camera 1, Green = Camera 2
- Before correction: Red/green separation visible
- After correction: Yellow particles (aligned)

**Layout:**
```
┌─────────────────┬─────────────────┐
│  Raw Camera 1   │  Raw Camera 2   │
├─────────────────┼─────────────────┤
│ Overlay BEFORE  │ Overlay AFTER   │
│ (red/green)     │ (yellow)        │
└─────────────────┴─────────────────┘
```

### 5.2 Figure 2: Disparity Field Before Correction

**Purpose:** Show the initial misalignment pattern.

**Content:**
- Panel 1: Disparity X as colourmap (diverging, centred at zero)
- Panel 2: Disparity Y as colourmap
- Panel 3: Disparity magnitude with vector overlay

**Annotations:**
- Mean disparity (pixels)
- RMS disparity (pixels)
- Colourbar with units

### 5.3 Figure 3: Disparity Field After Correction

**Purpose:** Confirm correction effectiveness.

**Content:**
- Same layout as Figure 2
- Use same colour scale as Figure 2 for direct comparison

**Expected result:**
- Near-zero disparity everywhere
- RMS < 0.1 pixels

### 5.4 Figure 4: Summary Comparison

**Purpose:** Single figure showing before/after and parameters.

**Content:**
- Panel 1: Disparity magnitude BEFORE
- Panel 2: Disparity magnitude AFTER (same colour scale)
- Panel 3: Histogram of disparity X (before and after overlaid)
- Panel 4: Parameter table

**Parameter table contents:**
```
┌─────────────────┬──────────┬───────────┬─────────┐
│ Parameter       │ True*    │ Estimated │ Error   │
├─────────────────┼──────────┼───────────┼─────────┤
│ Z-offset (mm)   │ N/A      │ 0.298     │         │
│ Tilt X (deg)    │ N/A      │ 0.114     │         │
│ Tilt Y (deg)    │ N/A      │ -0.059    │         │
├─────────────────┼──────────┼───────────┼─────────┤
│ Metric          │ Before   │ After     │ Reduct. │
├─────────────────┼──────────┼───────────┼─────────┤
│ RMS disp. (px)  │ 2.22     │ 0.04      │ 55×     │
└─────────────────┴──────────┴───────────┴─────────┘

* "True" column only applicable for synthetic validation
```

### 5.5 Figure 5: Convergence History

**Purpose:** Show iterative refinement progress.

**Content:**
- Panel 1: RMS disparity vs iteration (log scale Y-axis)
- Panel 2: Z-offset vs iteration (with true value line if known)
- Panel 3: Tilt X vs iteration
- Panel 4: Tilt Y vs iteration

**Annotations:**
- Convergence threshold line on disparity plot
- True values as dashed lines (for synthetic tests)

---

## 10. References

1. Wieneke, B. (2005). "Stereo-PIV using self-calibration on particle images." *Experiments in Fluids*, 39(2), 267–280.

2. Scarano, F., et al. (2005). "Self-calibration of multi-camera systems for 3D-PIV." *Proceedings of the 6th International Symposium on PIV*.

3. Raffel, M., et al. (2018). *Particle Image Velocimetry: A Practical Guide* (3rd ed.). Springer. Chapter 8: Stereoscopic PIV.

---

# Synthetic Test Environment: Code Outline

**Purpose:** This document outlines the code used to generate synthetic stereo particle images for validating the self-calibration implementation. A developer can use this to create their own test environment.

---

## 1. Overview

The synthetic test creates:
1. Two pinhole cameras in a symmetric stereo configuration
2. Random particles on a tilted/offset plane (simulating misaligned laser sheet)
3. Rendered particle images as Gaussian spots
4. Dewarping maps assuming Z=0 (the "wrong" calibration)

This allows testing self-calibration against known ground truth.

---

## 2. Camera Model

### 2.1 Data Structure

```python
@dataclass
class PinholeCamera:
    K: np.ndarray          # 3×3 intrinsic matrix
    dist: np.ndarray       # Distortion coefficients (5 elements, can be zeros)
    R: np.ndarray          # 3×3 rotation matrix (world to camera frame)
    t: np.ndarray          # 3×1 translation vector (world to camera frame)
    image_size: Tuple[int, int]  # (width, height) in pixels
```

### 2.2 Projection Function

Projects 3D world points to 2D image coordinates:

```python
def project(self, points_world: np.ndarray) -> np.ndarray:
    """
    Project 3D world points to 2D image coordinates.
    
    Parameters
    ----------
    points_world : ndarray, shape (N, 3)
        3D points in world coordinates [X, Y, Z] in mm.
        
    Returns
    -------
    points_image : ndarray, shape (N, 2)
        2D points in image coordinates [x, y] in pixels.
    """
    # Convert rotation matrix to Rodrigues vector (OpenCV format)
    rvec, _ = cv2.Rodrigues(self.R)
    
    # Project using OpenCV
    points_image, _ = cv2.projectPoints(
        points_world.reshape(-1, 1, 3),  # OpenCV expects (N, 1, 3)
        rvec, 
        self.t, 
        self.K, 
        self.dist
    )
    
    return points_image.reshape(-1, 2)
```

### 2.3 Creating a Stereo Pair

Creates two cameras positioned symmetrically about the Y-Z plane, both looking at the origin:

```python
def create_stereo_cameras(
    stereo_angle_deg: float = 30.0,    # Half-angle between cameras
    focal_length_px: float = 1000.0,   # Focal length in pixels
    image_size: Tuple[int, int] = (1024, 1024),
    baseline_mm: float = 200.0         # Distance between camera centres
) -> Tuple[PinholeCamera, PinholeCamera]:
    
    w, h = image_size
    theta = np.radians(stereo_angle_deg)
    
    # Intrinsic matrix (same for both cameras)
    # Principal point at image centre, square pixels
    K = np.array([
        [focal_length_px, 0, w / 2],
        [0, focal_length_px, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # No distortion for simplicity
    dist = np.zeros(5)
    
    # Camera distance from origin
    Z_cam = baseline_mm / (2 * np.tan(theta))
    
    # ----- Camera 1: rotated +theta about Y-axis -----
    # Rotation matrix: rotates world coordinates into camera frame
    R1 = np.array([
        [np.cos(theta),  0, np.sin(theta)],
        [0,              1, 0             ],
        [-np.sin(theta), 0, np.cos(theta) ]
    ])
    
    # Camera 1 position in world frame
    cam1_pos_world = np.array([-baseline_mm / 2, 0, Z_cam])
    
    # Translation: t = -R @ camera_position
    t1 = -R1 @ cam1_pos_world.reshape(3, 1)
    
    # ----- Camera 2: rotated -theta about Y-axis -----
    R2 = np.array([
        [np.cos(-theta),  0, np.sin(-theta)],
        [0,               1, 0              ],
        [-np.sin(-theta), 0, np.cos(-theta) ]
    ])
    
    cam2_pos_world = np.array([baseline_mm / 2, 0, Z_cam])
    t2 = -R2 @ cam2_pos_world.reshape(3, 1)
    
    cam1 = PinholeCamera(K=K, dist=dist, R=R1, t=t1, image_size=image_size)
    cam2 = PinholeCamera(K=K, dist=dist, R=R2, t=t2, image_size=image_size)
    
    return cam1, cam2
```

**Geometry diagram:**

```
                    Z (optical axis of system)
                    ↑
                    │
                    │
        Camera 1    │    Camera 2
            ╲       │       ╱
             ╲  θ   │   θ  ╱
              ╲     │     ╱
               ╲    │    ╱
                ╲   │   ╱
                 ╲  │  ╱
                  ╲ │ ╱
                   ╲│╱
    ────────────────┼────────────────→ X
                    │
                 (origin)
                    
    Baseline = distance between cameras along X
    θ = stereo_angle_deg
```

---

## 3. Particle Generation

### 3.1 Generate Particles on Tilted Plane

Creates random particle positions on the laser sheet plane, which may be offset and tilted from Z=0:

```python
def generate_particles(
    n_particles: int,
    x_range: Tuple[float, float],      # (min, max) in mm
    y_range: Tuple[float, float],      # (min, max) in mm
    z_offset: float = 0.0,             # mm, laser sheet offset from Z=0
    tilt_x: float = 0.0,               # radians, tilt about X-axis
    tilt_y: float = 0.0                # radians, tilt about Y-axis
) -> np.ndarray:
    """
    Generate random particle positions on tilted laser sheet.
    
    Laser sheet equation:
        Z(X, Y) = z_offset + X × tan(tilt_y) + Y × tan(tilt_x)
    
    Returns
    -------
    particles : ndarray, shape (n_particles, 3)
        Particle positions [X, Y, Z] in mm.
    """
    # Random X and Y positions
    X = np.random.uniform(x_range[0], x_range[1], n_particles)
    Y = np.random.uniform(y_range[0], y_range[1], n_particles)
    
    # Z determined by plane equation
    Z = z_offset + X * np.tan(tilt_y) + Y * np.tan(tilt_x)
    
    return np.column_stack([X, Y, Z])
```

**Note on tilt convention:**
- `tilt_x`: Rotation about X-axis → affects how Z varies with Y (dZ/dY)
- `tilt_y`: Rotation about Y-axis → affects how Z varies with X (dZ/dX)

---

## 4. Image Rendering

### 4.1 Render Particles as Gaussian Spots

Projects particles to the image and renders each as a Gaussian intensity distribution:

```python
def render_particles(
    particles: np.ndarray,             # (N, 3) world coordinates
    camera: PinholeCamera,
    particle_sigma: float = 2.0,       # Gaussian sigma in pixels
    intensity: float = 200.0           # Peak intensity (0-255)
) -> np.ndarray:
    """
    Render particles as Gaussian spots in synthetic image.
    
    Returns
    -------
    image : ndarray, shape (height, width), dtype uint8
        Rendered particle image.
    """
    w, h = camera.image_size
    image = np.zeros((h, w), dtype=np.float64)
    
    # Project all particles to image coordinates
    points_2d = camera.project(particles)
    
    # Pre-compute Gaussian kernel
    size = int(4 * particle_sigma)  # Kernel half-size
    x_local = np.arange(-size, size + 1)
    y_local = np.arange(-size, size + 1)
    xx, yy = np.meshgrid(x_local, y_local)
    gaussian = np.exp(-(xx**2 + yy**2) / (2 * particle_sigma**2))
    
    # Render each particle
    for pt in points_2d:
        x, y = pt
        
        # Skip if outside image (with margin for kernel)
        if x < size or x >= w - size or y < size or y >= h - size:
            continue
        
        # Add Gaussian at particle location
        x_int, y_int = int(round(x)), int(round(y))
        y1, y2 = y_int - size, y_int + size + 1
        x1, x2 = x_int - size, x_int + size + 1
        
        image[y1:y2, x1:x2] += gaussian * intensity
    
    # Clip to valid range and convert to uint8
    return np.clip(image, 0, 255).astype(np.uint8)
```

**Typical parameters:**
- `particle_sigma = 2.0`: Creates ~4-pixel diameter particles (typical for PIV)
- `intensity = 200`: Bright but not saturated
- `n_particles = 2000`: Moderate seeding density

---

## 5. Dewarping Maps

### 5.1 Compute Dewarping Maps

Creates maps that transform image coordinates to world coordinates at a specified Z plane:

```python
def compute_dewarp_maps(
    camera: PinholeCamera,
    output_size: Tuple[int, int],      # (width, height) of dewarped image
    world_bounds: Tuple[float, float, float, float],  # (x_min, x_max, y_min, y_max) in mm
    z_offset: float = 0.0,             # Z-plane offset
    tilt_x: float = 0.0,               # Tilt about X
    tilt_y: float = 0.0                # Tilt about Y
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dewarping maps for cv2.remap().
    
    For each pixel in the OUTPUT (dewarped) image, compute the corresponding
    pixel location in the INPUT (raw) image.
    
    Parameters
    ----------
    z_offset, tilt_x, tilt_y : float
        Plane parameters. Set all to zero for "uncorrected" maps.
        Set to estimated values for "corrected" maps.
    
    Returns
    -------
    map_x, map_y : ndarray, shape (out_height, out_width), dtype float32
        Coordinate maps for cv2.remap().
    """
    out_w, out_h = output_size
    x_min, x_max, y_min, y_max = world_bounds
    
    # Create grid of world X, Y coordinates
    X = np.linspace(x_min, x_max, out_w)
    Y = np.linspace(y_min, y_max, out_h)
    XX, YY = np.meshgrid(X, Y)
    
    # Compute Z for each point (plane equation)
    ZZ = z_offset + XX * np.tan(tilt_y) + YY * np.tan(tilt_x)
    
    # Stack into (N, 3) array of world points
    world_points = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    
    # Project to image coordinates
    image_points = camera.project(world_points)
    
    # Reshape to maps
    map_x = image_points[:, 0].reshape(out_h, out_w).astype(np.float32)
    map_y = image_points[:, 1].reshape(out_h, out_w).astype(np.float32)
    
    return map_x, map_y
```

### 5.2 Apply Dewarping

```python
def dewarp_image(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Apply dewarping maps to transform image to world coordinates."""
    return cv2.remap(
        image, 
        map_x, 
        map_y, 
        cv2.INTER_LINEAR,           # Bilinear interpolation
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=0               # Black outside valid region
    )
```

---

## 6. Putting It Together

### 6.1 Complete Test Setup

```python
# === Configuration ===
TRUE_Z_OFFSET = 0.3      # mm (known ground truth)
TRUE_TILT_X = 0.002      # radians (~0.11°)
TRUE_TILT_Y = -0.001     # radians (~-0.06°)

STEREO_ANGLE = 30.0      # degrees
N_IMAGES = 20
N_PARTICLES = 2000

WORLD_BOUNDS = (-40.0, 40.0, -40.0, 40.0)  # mm
OUTPUT_SIZE = (512, 512)  # pixels

# === Create cameras ===
cam1, cam2 = create_stereo_cameras(
    stereo_angle_deg=STEREO_ANGLE,
    focal_length_px=1000.0,
    image_size=(1024, 1024),
    baseline_mm=200.0
)

# === Compute "wrong" dewarp maps (assuming Z=0) ===
maps_cam1 = compute_dewarp_maps(cam1, OUTPUT_SIZE, WORLD_BOUNDS, 
                                 z_offset=0, tilt_x=0, tilt_y=0)
maps_cam2 = compute_dewarp_maps(cam2, OUTPUT_SIZE, WORLD_BOUNDS,
                                 z_offset=0, tilt_x=0, tilt_y=0)

# === Generate synthetic image pairs ===
images_cam1 = []
images_cam2 = []

for i in range(N_IMAGES):
    # Particles at TRUE (misaligned) position
    particles = generate_particles(
        n_particles=N_PARTICLES,
        x_range=(WORLD_BOUNDS[0] + 5, WORLD_BOUNDS[1] - 5),
        y_range=(WORLD_BOUNDS[2] + 5, WORLD_BOUNDS[3] - 5),
        z_offset=TRUE_Z_OFFSET,
        tilt_x=TRUE_TILT_X,
        tilt_y=TRUE_TILT_Y
    )
    
    # Render in both cameras
    images_cam1.append(render_particles(particles, cam1))
    images_cam2.append(render_particles(particles, cam2))

# === Now run self-calibration ===
# The self-cal should recover TRUE_Z_OFFSET, TRUE_TILT_X, TRUE_TILT_Y
# from the disparity between dewarped images
```

### 6.2 Expected Results

With the parameters above:

| Metric | Expected Value |
|--------|----------------|
| Initial RMS disparity | ~2.2 pixels |
| Iterations to converge | 2–3 |
| Final RMS disparity | < 0.1 pixels |
| Z-offset error | < 0.01 mm |
| Tilt errors | < 0.01° |

---

## 7. Validation Checklist

The synthetic test validates that your implementation:

- [ ] Correctly computes disparity from dewarped images
- [ ] Fits disparity to a plane accurately
- [ ] Converts disparity to physical parameters (Z, tilts) correctly
- [ ] Iterates and converges
- [ ] Recovers known ground truth within tolerance

**Test variations to try:**

1. **Zero misalignment:** Set all TRUE values to zero. Disparity should be ~0 from the start.

2. **Large offset:** Set TRUE_Z_OFFSET = 1.0 mm. Should still converge but may need more iterations.

3. **Tilt only:** Set TRUE_Z_OFFSET = 0, non-zero tilts. Disparity should show gradient but zero mean.

4. **Different stereo angles:** Test with 15°, 30°, 45°. Sensitivity changes with angle.

5. **Fewer images:** Test with N_IMAGES = 5, 10, 20, 50. Observe SNR improvement.

---

## 8. File Reference

The complete working implementation is in:

```
self_calibration_demonstration.py
```

Key functions:
- `create_stereo_cameras()` — lines ~70–110
- `generate_particles()` — lines ~115–140
- `render_particles()` — lines ~145–190
- `compute_dewarp_maps()` — lines ~195–240
- `run_demonstration()` — lines ~750–950 (main test driver)

The developer can extract these functions or rewrite them in their preferred language/framework.