import glob
import os
import random

import torch.utils.data as data

from PIL import Image


class TwoClassLoader(data.Dataset):
  def __init__(self, root_dir, transforms, split='train', classes=('cloudy', 'sunny')):
    self.root_dir = root_dir
    self.transforms = transforms

    self.files_classA = glob.glob(os.path.join(
        self.root_dir, split, classes[0], '*.jpg'
    ))

    # load all the images in class #A
    self.data_classA = [Image.open(f) for f in self.files_classA]

    self.files_classB = glob.glob(os.path.join(
        self.root_dir, split, classes[1], '*.jpg'
    ))

    # load all the images in class #B
    self.data_classB = [Image.open(f) for f in self.files_classB]

    # as we might have unequal number of A and B, we will take the max length as the length of the dataset
    self.num_items = max(len(self.data_classA), len(self.data_classB))

  def __getitem__(self, idx):
    # as we have an imbalance, we will pick A as per the idx (fitting it into a valid value), and B randomly
    # random values of second class might help generalize better

    return (
        self.transforms(self.data_classA[len(self.data_classA) % idx]),
        self.transforms(random.choice(self.data_classB))
    )

  def __len__(self):
    return self.num_items
