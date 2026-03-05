"""
Generate 20 synthetic PIV image pairs with Poiseuille (parabolic) velocity profile.

Each pair uses a different random seed (different particle positions) but the same
analytical velocity field. This produces small per-pair sub-pixel estimation noise,
giving non-zero Reynolds stresses for ensemble fitting to work with.

Velocity field:
  u_x(y) = U_MAX * (1 - (2y/H - 1)^2)   parabolic across rows
  u_y    = 0                               zero vertical displacement

Output: unit-tests/poiseuille_ensemble/B{00001..00020}_A.tif, B{00001..00020}_B.tif, params.json
"""
import json

import numpy as np
import tifffile
from pathlib import Path
from synthetic_piv import render_particles

# ── Parameters ──────────────────────────────────────────────────────────────
IMAGE_SHAPE = (500, 500)
NUM_PARTICLES = 30_000
PARTICLE_DIAMETER = 3.0
SIGMA = PARTICLE_DIAMETER / 2.355  # FWHM to Gaussian sigma
U_MAX = 5.0                        # max displacement in pixels
NUM_PAIRS = 20
BASE_SEED = 100

OUTPUT_DIR = Path(__file__).parent / "poiseuille_ensemble"
# ────────────────────────────────────────────────────────────────────────────


def _normalize_uint16(img):
    mx = img.max()
    if mx > 0:
        return (img / mx * 65535).astype(np.uint16)
    return img.astype(np.uint16)


def main():
    H, W = IMAGE_SHAPE
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating Poiseuille ensemble ({NUM_PAIRS} pairs) -> {OUTPUT_DIR}")
    print(f"  Image size:         {H}x{W}")
    print(f"  Particle diameter:  {PARTICLE_DIAMETER} px  (sigma = {SIGMA:.3f})")
    print(f"  Particles/pair:     {NUM_PARTICLES}")
    print(f"  u_max:              {U_MAX} px")

    for pair_idx in range(1, NUM_PAIRS + 1):
        seed = BASE_SEED + pair_idx
        rng = np.random.default_rng(seed)

        # Random particle positions
        x_pos = rng.uniform(0, W, NUM_PARTICLES)
        y_pos = rng.uniform(0, H, NUM_PARTICLES)
        intensities = rng.uniform(200, 255, NUM_PARTICLES)

        # Poiseuille displacement: u_x(y) = U_MAX * (1 - (2y/H - 1)^2), u_y = 0
        y_norm = 2.0 * y_pos / H - 1.0
        dx_image = U_MAX * (1.0 - y_norm ** 2)
        dy_image = np.zeros_like(dx_image)

        # Render frames
        img_a = render_particles(IMAGE_SHAPE, x_pos, y_pos, intensities, SIGMA)
        img_b = render_particles(IMAGE_SHAPE, x_pos + dx_image, y_pos + dy_image,
                                 intensities, SIGMA)

        tifffile.imwrite(OUTPUT_DIR / f"B{pair_idx:05d}_A.tif", _normalize_uint16(img_a))
        tifffile.imwrite(OUTPUT_DIR / f"B{pair_idx:05d}_B.tif", _normalize_uint16(img_b))

        if pair_idx % 5 == 0:
            print(f"  Generated {pair_idx}/{NUM_PAIRS} pairs")

    # Save parameters for test reconstruction
    params = {
        "image_shape": list(IMAGE_SHAPE),
        "num_particles": NUM_PARTICLES,
        "particle_diameter": PARTICLE_DIAMETER,
        "sigma": SIGMA,
        "u_max": U_MAX,
        "num_pairs": NUM_PAIRS,
        "base_seed": BASE_SEED,
    }
    with open(OUTPUT_DIR / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    print(f"\nDone. {NUM_PAIRS} pairs saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
