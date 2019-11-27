'''
This script computes the mean and variance of sunny and cloudy datasets
'''

import glob
import os

import numpy as np

from sklearn.preprocessing import StandardScaler
from PIL import Image


def compute_stats(im_class) -> (np.array, np.array):
  '''
  Compute the mean and the variance of the dataset.

  Note: do convert the image in grayscale and then in [0,1] before computing mean and variance

  Hints: use StandardScalar (check import statement)

  Args:
  -   dir_name: the path of the root dir
  Returns:
  -   mean: mean value of the dataset (np.array containing a scalar value)
  -   std: standard deviation of th dataset (np.array containing a scalar value)
  '''

  file_names = glob.glob(os.path.join('data_small', '*', im_class, '*.jpg'))

  scaler = StandardScaler(with_mean=True, with_std=True)
  for file_name in file_names:
    with open(file_name, 'rb') as f:
      img = np.asarray(Image.open(f).convert('RGB'), dtype='float32')/255.0
      scaler.partial_fit(img.reshape(-1, 1))

  mean = scaler.mean_
  std = scaler.scale_

  return mean, std


print('stats for sunny', compute_stats('sunny'))
print('stats for cloudy', compute_stats('cloudy'))
