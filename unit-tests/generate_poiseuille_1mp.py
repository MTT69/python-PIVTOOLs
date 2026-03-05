"""
Generate a 1MP synthetic PIV image pair with Poiseuille (parabolic) velocity profile.

Velocity field:
  u_x(y) = u_max * (1 - (2y/H - 1)^2)   parabolic across rows
  u_y    = 0                               zero vertical displacement

Uses sub-pixel accurate Gaussian particle rendering via synthetic_piv.render_particles().
Saves images + displacement parameters as JSON sidecar for test reconstruction.

Output: unit-tests/poiseuille_1mp/B00001_A.tif, B00001_B.tif, params.json
"""
import json

import numpy as np
import tifffile
from pathlib import Path
from synthetic_piv import render_particles

# ── Parameters ──────────────────────────────────────────────────────────────
IMAGE_SHAPE = (1000, 1000)
NUM_PARTICLES = 40_000
PARTICLE_DIAMETER = 3.0
SIGMA = PARTICLE_DIAMETER / 2.355  # FWHM to Gaussian sigma
U_MAX = 5.0                        # max displacement in pixels
SEED = 42

OUTPUT_DIR = Path(__file__).parent / "poiseuille_1mp"
# ────────────────────────────────────────────────────────────────────────────


def _normalize_uint16(img):
    mx = img.max()
    if mx > 0:
        return (img / mx * 65535).astype(np.uint16)
    return img.astype(np.uint16)


def main():
    rng = np.random.default_rng(SEED)
    H, W = IMAGE_SHAPE
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating Poiseuille 1MP image pair -> {OUTPUT_DIR}")
    print(f"  Image size:         {H}x{W}")
    print(f"  Particle diameter:  {PARTICLE_DIAMETER} px  (sigma = {SIGMA:.3f})")
    print(f"  Particles:          {NUM_PARTICLES}")
    print(f"  u_max:              {U_MAX} px")

    # Random particle positions
    x_pos = rng.uniform(0, W, NUM_PARTICLES)
    y_pos = rng.uniform(0, H, NUM_PARTICLES)
    intensities = rng.uniform(200, 255, NUM_PARTICLES)

    # Poiseuille displacement: u_x(y) = u_max * (1 - (2y/H - 1)^2), u_y = 0
    y_norm = 2.0 * y_pos / H - 1.0
    dx_image = U_MAX * (1.0 - y_norm ** 2)
    dy_image = np.zeros_like(dx_image)

    # Render frames
    img_a = render_particles(IMAGE_SHAPE, x_pos, y_pos, intensities, SIGMA)
    img_b = render_particles(IMAGE_SHAPE, x_pos + dx_image, y_pos + dy_image,
                             intensities, SIGMA)

    tifffile.imwrite(OUTPUT_DIR / "B00001_A.tif", _normalize_uint16(img_a))
    tifffile.imwrite(OUTPUT_DIR / "B00001_B.tif", _normalize_uint16(img_b))

    # Save parameters for test reconstruction
    params = {
        "image_shape": list(IMAGE_SHAPE),
        "num_particles": NUM_PARTICLES,
        "particle_diameter": PARTICLE_DIAMETER,
        "sigma": SIGMA,
        "u_max": U_MAX,
        "seed": SEED,
    }
    with open(OUTPUT_DIR / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    print(f"\nDone. Files saved:")
    print(f"  {OUTPUT_DIR / 'B00001_A.tif'}")
    print(f"  {OUTPUT_DIR / 'B00001_B.tif'}")
    print(f"  {OUTPUT_DIR / 'params.json'}")


if __name__ == "__main__":
    main()
