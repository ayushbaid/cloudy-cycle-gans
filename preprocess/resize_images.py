import os
import glob
from PIL import Image
from math import ceil

input_dir = 'data'
output_dir = 'data_small'

files = glob.glob(os.path.join(input_dir, '*', '*', '*.jpg'))

for f in files:
  img = Image.open(f)
  width, height = img.size

  min_val = min(width, height)
  scale = 256.0/min_val
  resized_image = img.resize((int(ceil(width*scale)), int(ceil(height*scale))))

  resized_image.save(f.replace(input_dir, output_dir))
