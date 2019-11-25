import os
import glob
from PIL import Image

input_dir = 'data'
output_dir = 'data_small'

files = glob.glob(os.path.join(input_dir, '*', '*', '*.jpg'))

for f in files:
  img = Image.open(f)
  resized_image = img.resize((128, 128))

  resized_image.save(f.replace(input_dir, output_dir))
