import sys
sys.path.insert(0, '/Users/morgan/Documents/CODE/PIVTOOLS_FULL_STACK/PyPIVTools/piv/lib/python3.13/site-packages')

import cinereader as cr
import matplotlib.pyplot as plt

cine_path = '/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/Cavity/181124_polyamide/90degree_400light_100hz_3000dt_4/Camera1.cine'

# Read metadata
metadata = cr.read_metadata(cine_path)

# Read first two frames (frame A and frame B)
frame_a = cr.read_image(metadata, cine_path, metadata.FirstImageNo)
frame_b = cr.read_image(metadata, cine_path, metadata.FirstImageNo + 1)

# Create subplot figure
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(frame_a, cmap='gray')
axes[0].set_title('Frame A')
axes[0].axis('off')

axes[1].imshow(frame_b, cmap='gray')
axes[1].set_title('Frame B')
axes[1].axis('off')

plt.tight_layout()
plt.show()