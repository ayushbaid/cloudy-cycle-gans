import torch.nn as nn
import torchvision.models as models


class ContentSimilarityChecker(nn.Module):
  '''
  We will use a the first layer of any image classification model to enforce content similarity
  '''

  def __init__(self):

    super(ContentSimilarityChecker, self).__init__()
    vgg_pretrained = models.vgg11(pretrained=True)

    '''
    Just extract the first layer from it
    '''

    self.net = nn.Sequential(
        vgg_pretrained.features[0],
        nn.InstanceNorm2d(vgg_pretrained.features[0].out_channels),
        nn.ReLU(inplace=True)
    )

    self.net[0].weight.requires_grad = False
    self.net[0].bias.requires_grad = False

    self.loss_criterion = nn.MSELoss()

  def forward(self, inp_a, inp_b):
    return self.loss_criterion(self.net(inp_a), self.net(inp_b))
