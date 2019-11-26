import torch.nn as nn


class ResidualBlock(nn.Module):
  def __init__(self, num_features):
    '''
    Define the residual block as suggested in the cycle-GAN paper

    They refer "He at al. Deep residual learning for image recognition. In CVPR, 2016" for residual block

    The only difference is the use of instance normalizatin
    '''

    super(ResidualBlock, self).__init__()

    self.net = nn.Sequential(
        # adding padding of 1 to remain of the same size
        nn.ReflectionPad2d(1),
        nn.Conv2d(num_features, num_features, kernel_size=3, padding=0),
        nn.InstanceNorm2d(num_features),
        nn.ReLU(True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(num_features, num_features, kernel_size=3, padding=0),
        nn.InstanceNorm2d(num_features)
    )

    return None

    # we will add the residual in the forward method

  def forward(self, x):
    '''
    Forward function with residual connection implementation
    '''

    return x + self.net(x)
