import torch.nn as nn
import torchvision.models as models


class ContentSimilarityChecker(nn.Module):
  '''
  We will use a the first layer of any image classification model to enforce content similarity
  '''

  def __init__(self):

    super(ContentSimilarityChecker, self).__init__()
    alexnet_pretrained = models.alexnet(pretrained=True)

    '''
    Just extract the first layer from it
    '''

    self.net = nn.Sequential(
        alexnet_pretrained.features[0],
        nn.ReLU(inplace=True)
    )

    self.net[0].weight.requires_grad = False
    self.net[0].bias.requires_grad = False

    self.loss_criterion = nn.MSELoss()

  def forward(self, inp_a, inp_b):
    return self.loss_criterion(self.net(inp_a), self.net(inp_b))
