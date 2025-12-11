import os
from PIL import Image

directory = '/Users/morgan/Library/CloudStorage/OneDrive-UniversityofSouthampton/Documents/#current_processing/query_JHTDB/download_from_jhtdb/bottom_channel/stereo/Cam2/calibration'

for filename in os.listdir(directory):
    if filename.endswith('.tif'):
        filepath = os.path.join(directory, filename)
        img = Image.open(filepath)
        flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
        flipped.save(filepath)
        print(f'Flipped {filename}')


