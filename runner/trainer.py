import os

import torch.utils

from torch.autograd import Variable
import matplotlib.pyplot as plt

import torchvision.datasets.ImageFolder as ImageFolder


class Trainer():
  '''
  This class makes training the model easier
  '''

  def __init__(self,
               data_dir,
               model,
               optimizer,
               model_dir,
               train_data_transforms,
               test_data_transforms,
               batch_size=100,
               load_from_disk=True,
               cuda=False
               ):
    self.model_dir = model_dir

    self.model = model

    self.cuda = cuda
    if cuda:
      self.model.cuda()

    dataloader_args = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    self.train_dataset = ImageFolder(root=os.path.join(data_dir, 'train'),
                                     target_transform=train_data_transforms
                                     )
    self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                                    **dataloader_args)

    self.test_dataset = ImageFolder(root=os.path.join(data_dir, 'test'),
                                    target_transform=test_data_transforms
                                    )
    self.test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True,
                                                   **dataloader_args
                                                   )

    self.optimizer = optimizer

    self.train_loss_history = []
    self.validation_loss_history = []

    # load the model from the disk if it exists
    if os.path.exists(model_dir) and load_from_disk:
      checkpoint = torch.load(os.path.join(self.model_dir, 'checkpoint.pt'))
      self.model.load_state_dict(checkpoint['model_state_dict'])
      self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

      # TODO: write code to load and save epoch and loss history too, so that we can truly pause and resume

    self.model.train()

  def save_model(self):
    '''
    Saves the model state and optimizer state on the dict
    '''
    torch.save({
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict()
    }, os.path.join(self.model_dir, 'checkpoint.pt'))

  def train(self, num_epochs):
    '''
    The main train loop
    '''
    self.model.train()
    for epoch_idx in range(num_epochs):
      for batch_idx, batch in enumerate(self.train_loader):
        if self.cuda:
          input_data, target_data = Variable(
              batch[0]).cuda(), Variable(batch[1]).cuda()
        else:
          input_data, target_data = Variable(batch[0]), Variable(batch[1])

        output_data = self.model(input_data)
        loss = self.model.loss_critereon(output_data, target_data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

      self.train_loss_history.append(float(loss))
      self.model.eval()
      self.eval_on_test()
      self.model.train()

      if epoch_idx % 1 == 0:
        print('Epoch:{}, Loss:{:.4f}'.format(epoch_idx+1, float(loss)))
        # self.save_model()

  def eval_on_test(self):
    '''
    Get loss on test set
    '''
    test_loss = 0.0

    num_examples = 0
    for batch_idx, batch in enumerate(self.test_loader):
      if self.cuda:
        input_data, target_data = Variable(
            batch[0]).cuda(), Variable(batch[1]).cuda()
      else:
        input_data, target_data = Variable(batch[0]), Variable(batch[1])

      num_examples += input_data.shape[0]
      output_data = self.model.forward(input_data)
      loss = self.model.loss_critereon(
          output_data, target_data, is_normalize=False)

      test_loss += float(loss)

    self.validation_loss_history.append(test_loss/num_examples)

    return self.validation_loss_history[-1]

  def plot_loss_history(self):
    '''
    Plots the loss history
    '''
    plt.figure()
    ep = range(len(self.train_loss_history))

    plt.plot(ep, self.train_loss_history, '-b', label='training')
    plt.plot(ep, self.validation_loss_history, '-r', label='validation')
    plt.title("Loss history")
    plt.legend()
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.show()
