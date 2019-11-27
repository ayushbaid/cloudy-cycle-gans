import glob
import os
import random

from transforms.im_transforms import get_fundamental_transforms

import torch.utils.data as data

from PIL import Image


class TwoClassLoader(data.Dataset):
  def __init__(self,
               root_dir,
               transforms_A=get_fundamental_transforms(
                   im_size=(256, 256), mean_val=0.49000312, std_val=0.27253689
               ),
               transforms_B=get_fundamental_transforms(
                   im_size=(256, 256), mean_val=0.47342853, std_val=0.26870837
               ),
               split='train',
               classes=('cloudy', 'sunny')):
    self.root_dir = root_dir
    self.transforms_A = transforms_A
    self.transforms_B = transforms_B

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
        self.transforms_A(Image.open(
            self.files_classA[idx % len(self.files_classA)])),
        self.transforms_B(Image.open(random.choice(self.files_classB)))
    )

  def __len__(self):
    return self.num_items
