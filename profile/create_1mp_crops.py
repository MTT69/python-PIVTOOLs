"""Create 1MP centre crops from 4MP images for benchmarking."""
import argparse
import os

import cv2

SOURCE_4MP = (
    r"C:\Users\mtt1e23\OneDrive - University of Southampton\Documents"
    r"\#current_processing\4000_images_channel\planar_images"
)
SOURCE_1MP = os.path.join(SOURCE_4MP, "1mp")


def create_crops(n_pairs=1000):
    os.makedirs(SOURCE_1MP, exist_ok=True)
    created = 0
    for idx in range(1, n_pairs + 1):
        for suffix in ("A", "B"):
            src = os.path.join(SOURCE_4MP, f"B{idx:05d}_{suffix}.tif")
            dst = os.path.join(SOURCE_1MP, f"B{idx:05d}_{suffix}.tif")
            if os.path.exists(dst):
                continue
            if not os.path.exists(src):
                print(f"  Source not found: {src} — stopping at {idx - 1} pairs")
                return
            img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            h, w = img.shape[:2]
            cy, cx = h // 2, w // 2
            crop = img[cy - 500 : cy + 500, cx - 500 : cx + 500]
            cv2.imwrite(dst, crop)
            created += 1
    print(f"  Created {created} new files, {n_pairs} pairs ready in {SOURCE_1MP}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create 1MP centre crops from 4MP images")
    parser.add_argument("--pairs", type=int, default=1000,
                        help="Number of pairs to create (default: 1000)")
    args = parser.parse_args()
    create_crops(args.pairs)
