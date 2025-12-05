import cv2
import numpy as np
from pathlib import Path

# ================= CONFIGURATION =================
# Input directory (Where your current .tif images are)
INPUT_DIR = Path(r"/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/download_from_jhtdb/bottom_channel/stereo/Cam2/calibration")

# Output directory (Where enhanced images will be saved)
# I'm creating a subfolder named 'enhanced' to avoid overwriting originals
OUTPUT_DIR = INPUT_DIR 

# File pattern to search for
FILE_PATTERN = "planar_calibration_plate_*.tif"

# Target settings
NEW_RADIUS = 15  # Target radius in pixels
COLOR = 255      # White
# =================================================

def process_images():
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all files matching the pattern
    image_files = sorted(list(INPUT_DIR.glob(FILE_PATTERN)))

    if not image_files:
        print(f"No images found matching '{FILE_PATTERN}' in {INPUT_DIR}")
        return

    print(f"Found {len(image_files)} images. Starting enhancement...")
    print("-" * 50)

    total_dots = 0

    for img_path in image_files:
        try:
            # 1. Read the image (Grayscale)
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                print(f"Skipping {img_path.name} - Could not read image.")
                continue

            # 2. Threshold to find the existing dots
            # We use OTSU binarization to automatically find the best threshold 
            # to separate white dots from black background
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # 3. Find Contours (the shapes of the current 9px dots)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 4. Create a canvas to draw on 
            # We copy the original to keep the background dimensions/type
            # If you want a purely clean black background, use: np.zeros_like(img)
            output_img = img.copy()

            dots_in_image = 0

            for cnt in contours:
                # Calculate the center (centroid) of the dot
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # 5. Draw the new larger circle
                    # -1 thickness means "fill the circle"
                    cv2.circle(output_img, (cX, cY), NEW_RADIUS, COLOR, -1)
                    dots_in_image += 1

            # 6. Save the new image
            save_path = OUTPUT_DIR / img_path.name
            cv2.imwrite(str(save_path), output_img)
            
            total_dots += dots_in_image
            print(f"Processed {img_path.name}: Enhanced {dots_in_image} dots")

        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")

    print("-" * 50)
    print(f"Processing complete.")
    print(f"Total images processed: {len(image_files)}")
    print(f"Total dots enhanced: {total_dots}")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_images()