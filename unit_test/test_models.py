from models.generator import Generator
from models.descriminator import Descriminator

import torch


def test_generator():
  im_size = (2, 3, 300, 300)

  test_img = torch.rand(im_size)

  generator_obj = Generator()

  out_img = generator_obj(test_img)

  assert out_img.shape == test_img.shape


def test_discriminator():
  im_size = (2, 3, 300, 300)

  test_img = torch.rand(im_size)

  obj = Descriminator()

  output = obj(test_img)

  assert list(output.shape) == [2, 1]
