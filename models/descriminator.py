import torch.nn as nn


class Descriminator(nn.Module):
  def __init__(self):
    '''
    The paper describes the model as C64-C128-C256-C512
    (no instance norm in first layer; leaky relus; last conv to produce a scalar)

    We will implement as C64-C128-C256
    '''

    super(Descriminator, self).__init__()

    self.net = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=4, stride=2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 128, kernel_size=4, stride=2),
        nn.InstanceNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(128, 256, kernel_size=4, stride=2),
        nn.InstanceNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(256, 1, kernel_size=35)
    )

  def forward(self, x):
    return self.net(x).reshape(x.shape[0], 1)
