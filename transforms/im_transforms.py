'''
Defining image transforms for data
'''
import torchvision.transforms as torchtransforms

import numpy as np


def get_fundamental_transforms(im_size, mean_val=0.0, std_val=1.0):
  '''
  Performs the most basic transforms on input images
  '''

  return torchtransforms.Compose([
      torchtransforms.RandomCrop(size=im_size),
      torchtransforms.ToTensor(),
      torchtransforms.Normalize(mean=np.array(
          [mean_val]), std=np.array([std_val]))
  ])
