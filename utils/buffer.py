import random

import torch

from torch.autograd import Variable


class ImageBuffer(object):
  '''
  Implement the image buffer to be used in training

  With p=0.35, return the input images; otherwise return data from buffer and replace them with input
  '''
  # TODO: write tests

  def __init__(self, buffer_size, is_cuda):
    self.buffer_size = buffer_size
    self.buffer_list = []
    self.is_cuda = is_cuda

  def get(self, input_images):
    '''
    Args:
    * input_images: list of images

    '''
    result = []

    for image in input_images.data:
      image = torch.unsqueeze(image, 0)
      if len(self.buffer_list) < self.buffer_size:
        # buffer is not full yet
        self.buffer_list.append(image)
        result.append(image)
      else:
        p = random.uniform(0, 1) > 0.35  # simulating coin toss

        if p:
          # fetch value from the buffer
          idx = random.randint(0, self.buffer_size-1)
          result.append(self.buffer_list[idx])
          self.buffer_list[idx] = image
        else:
          result.append(image)

      # return the result as a single tensor
      if self.is_cuda:
        return Variable(torch.cat(result, 0).cuda())
      else:
        return Variable(torch.cat(result, 0))
