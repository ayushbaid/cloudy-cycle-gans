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

    self.files_classB = glob.glob(os.path.join(
        self.root_dir, split, classes[1], '*.jpg'
    ))

    # as we might have unequal number of A and B, we will take the max length as the length of the dataset
    self.num_items = max(len(self.files_classA), len(self.files_classB))

  def __getitem__(self, idx):
    # as we have an imbalance, we will pick A as per the idx (fitting it into a valid value), and B randomly
    # random values of second class might help generalize better

    return (
        self.transforms(Image.open(
            self.files_classA[len(self.files_classA) % idx])),
        self.transforms(Image.open(random.choice(self.files_classB)))
    )

  def __len__(self):
    return self.num_items
