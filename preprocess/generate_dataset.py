'''
Generate the dataset from downloaded data by image resizing and stuff
'''
import os
import glob
from random import shuffle
from shutil import copyfile

reference_folder = '/media/ayush/OS/Users/Ayush/cloudy_data/weather_database/'

class_names = ('sunny', 'cloudy')

final_folder = 'data_small/'

# split between train, val, test
split_vals = (0.7, 0.15, 0.15)

for c in class_names:
  input_folder = os.path.join(reference_folder, c, '*')

  files = glob.glob(input_folder)

  num_files = len(files)

  num_train = int(split_vals[0] * num_files)
  num_val = int(split_vals[1] * num_files)
  num_test = num_files - num_train - num_val

  # shuffling the files
  shuffle(files)

  for f in files[:num_train]:
    base_name = os.path.basename(f)
    out_name = os.path.join(final_folder, 'train', c, base_name)

    copyfile(f, out_name)

  for f in files[num_train:num_train+num_val]:
    base_name = os.path.basename(f)
    out_name = os.path.join(final_folder, 'val', c, base_name)

    copyfile(f, out_name)

  for f in files[-num_test:]:
    base_name = os.path.basename(f)
    out_name = os.path.join(final_folder, 'test', c, base_name)

    copyfile(f, out_name)
