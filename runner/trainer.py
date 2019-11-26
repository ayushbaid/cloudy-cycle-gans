import os
import torch
import torch.utils
import torchvision.datasets.ImageFolder as ImageFolder
from torch.autograd import Variable
import matplotlib.pyplot as plt
from runner.dataloader import TwoClassLoader
from transforms.im_transforms import get_fundamental_transforms
from models.cycle_gans import CycleGAN

class Trainer():
  '''
  This class makes training the model easier
  '''

  def __init__(self,
               data_dir,
               optimizer,
               model_dir,
               batch_size=100,
               load_from_disk=True,
               cuda=False
               ):
    self.model_dir = model_dir

    self.cycle_gan = CycleGAN(is_cuda=cuda)

    dataloader_args = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    self.train_dataset = TwoClassLoader(
        data_dir, get_fundamental_transforms(im_size=(128, 128)), split='train')

    self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                                    **dataloader_args)

    self.val_dataset = TwoClassLoader(
        data_dir, get_fundamental_transforms(im_size=(128, 128)), split='val')
    self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True,
                                                  **dataloader_args
                                                  )

    # Optimizers & LR schedulers
    self.optimizer_G = torch.optim.Adam(itertools.chain(self.cycle_gan.generator_A2B.parameters(),
                                                        self.cycle_gan.generator_B2A.parameters()),
                                                        lr=opt.lr, betas=(0.5, 0.999))
    self.optimizer_D_A = torch.optim.Adam(self.cycle_gan.discriminator_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    self.optimizer_D_B = torch.optim.Adam(self.cycle_gan.discriminator_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    # total number of epochs = 200
    # starting epoch = 0
    # epoch to start linearly decaying the learning rate to 0 = 100
    self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lr_lambda=LambdaLR(200, 0, 100).step)
    self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_A, lr_lambda=LambdaLR(200, 0, 100).step)
    self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D_B, lr_lambda=LambdaLR(200, 0, 100).step)

    self.train_loss_history = []
    self.validation_loss_history = []

    # load the model from the disk if it exists
    if os.path.exists(model_dir) and load_from_disk:
      checkpoint = torch.load(os.path.join(model_dir, 'checkpoint.pt'))
      self.cycle_gan.load_from_state(checkpoint['model_state_dict'])

      # TODO: write code to load and save epoch and loss history too, so that we can truly pause and resume

    self.cycle_gan.train_mode()

  def save_model(self):
    '''
    Saves the model state and optimizer state on the dict
    '''
    torch.save({
        'model_state_dict': self.cycle_gan.extract_state(),
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
