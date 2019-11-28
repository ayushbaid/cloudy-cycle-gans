import torch.nn as nn

from models.residual_block import ResidualBlock


class Generator(nn.Module):
  '''
  Define the generator model
  '''

  def __init__(self, num_residual_blocks=3):
    '''
    The model defined in the paper is:
    c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3

    we will use a smaller model:
    c7s1-64,d128,d128,R128,R128,R128,u128,u64,c7s1-3
    '''

    super(Generator, self).__init__()

    # c7s1-64
    self.pre_residual = nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(3, 64, kernel_size=7, stride=1),
        nn.InstanceNorm2d(64),
        nn.ReLU(inplace=True)
    )

    # d128
    self.downsampling = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm2d(128),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
        nn.InstanceNorm2d(128),
        nn.ReLU(inplace=True),
    )

    temp_list = []
    for _ in range(num_residual_blocks):
      temp_list.append(ResidualBlock(128))

    # R128,R128,R128
    self.residual_net = nn.Sequential(*temp_list)

    # u64
    self.upsampling = nn.Sequential(
        nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2,
                           padding=0, output_padding=0),
        nn.InstanceNorm2d(128),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2,
                           padding=0, output_padding=0),
        nn.InstanceNorm2d(64),
        nn.ReLU(inplace=True),
    )

    self.output_conv = nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(64, 3, kernel_size=7, stride=1),
        nn.Tanh()  # tanh because we want the output to be an image (and hence bounded values)
    )

  def forward(self, x):
    return self.output_conv(self.upsampling(
        self.residual_net(
            self.downsampling(self.pre_residual(
                x
            ))
        )
    ))
