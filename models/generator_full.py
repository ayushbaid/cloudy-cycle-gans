import torch.nn as nn

from models.residual_block import ResidualBlock


class GeneratorFull(nn.Module):
  '''
  Define the generator model
  '''

  def __init__(self, num_residual_blocks=6):
    '''
    The model defined in the paper is:
    c7s1-64,d128,d256,R256,R256,R256,R256,R256,R256,u128,u64,c7s1-3

    we will use a smaller model:
    c7s1-64,R64,R64,R64,R64,c7s1-3
    '''

    super(GeneratorFull, self).__init__()

    # c7s1-64
    self.pre_residual = nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(3, 64, kernel_size=7, stride=1),
        nn.InstanceNorm2d(64),
        nn.ReLU(inplace=True)
    )

    temp_list = []
    for _ in range(num_residual_blocks):
      temp_list.append(ResidualBlock(64))

    # R128,R128,R128
    self.residual_net = nn.Sequential(*temp_list)

    self.output_conv = nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(64, 3, kernel_size=7, stride=1),
        nn.Tanh()  # tanh because we want the output to be an image (and hence bounded values)
    )

  def forward(self, x):
    return self.output_conv(
        self.residual_net(
            self.pre_residual(x)
        )
    )
