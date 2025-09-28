import os

import numpy as np
from PIL import Image


def add_random_noise(img_array, amount=50):
    """
    Add random noise to the image array.
    Each pixel gets random noise in the range [-amount, amount].
    """
    noise = np.random.randint(-amount, amount + 1, img_array.shape, dtype=np.int16)
    noisy_img = img_array.astype(np.int16) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img


def process_images(base_dir, save_dir, amount=50):
    os.makedirs(save_dir, exist_ok=True)
    for fname in os.listdir(base_dir):
        if fname.lower().endswith(".tif"):
            img_path = os.path.join(base_dir, fname)
            img = Image.open(img_path)
            img_array = np.array(img)
            noisy_array = add_random_noise(img_array, amount=amount * 2)
            noisy_img = Image.fromarray(noisy_array)
            save_path = os.path.join(save_dir, fname)
            noisy_img.save(save_path)
            print(f"Saved noisy image: {save_path}")


if __name__ == "__main__":
    # Example usage:
    base_dir = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/Planar_Images_with_wall/Cam1"  # <-- change this
    save_dir = "/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/Planar_Images_with_wall/noisy/Cam1"  # <-- change this
    # You can adjust amount as needed
    process_images(base_dir, save_dir)
