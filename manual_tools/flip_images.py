import os
from PIL import Image

directory = 'particle_positions/planar'

for filename in os.listdir(directory):
    if filename.endswith('.tif'):
        filepath = os.path.join(directory, filename)
        img = Image.open(filepath)
        flipped = img.transpose(Image.FLIP_TOP_BOTTOM)
        flipped.save(filepath)
        print(f'Flipped {filename}')


