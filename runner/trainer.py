import os
import itertools
import time

from runner.dataloader import TwoClassLoader
from transforms.im_transforms import get_fundamental_transforms
from models.cycle_gans import CycleGAN
from runner.train_metadata import TrainMetadata
from utils.lrScheduler import LambdaLR

import torch
import torch.utils
from torch.autograd import Variable


class Trainer():
  '''
  This class makes training the model easier
  '''

  def __init__(self,
               data_dir,
               model_dir,
               batch_size=100,
               load_from_disk=True,
               cuda=False
               ):
    self.model_dir = model_dir
    self.cuda = cuda

    self.cycle_gan = CycleGAN(is_cuda=cuda)

    dataloader_args = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    self.train_dataset = TwoClassLoader(
        data_dir, get_fundamental_transforms(im_size=(256, 256)), split='train')

    self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True,
                                                    **dataloader_args)

    self.val_dataset = TwoClassLoader(
        data_dir, get_fundamental_transforms(im_size=(256, 256)), split='val')
    self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=batch_size, shuffle=True,
                                                  **dataloader_args
                                                  )

    # Optimizers & LR schedulers
    self.optimizer_G = torch.optim.Adam(itertools.chain(self.cycle_gan.generator_A2B.parameters(),
                                                        self.cycle_gan.generator_B2A.parameters()),
                                        lr=2e-4, betas=(0.5, 0.999))
    self.optimizer_D_A = torch.optim.Adam(
        self.cycle_gan.discriminator_A.parameters(), lr=2e-4, betas=(0.5, 0.999))
    self.optimizer_D_B = torch.optim.Adam(
        self.cycle_gan.discriminator_B.parameters(), lr=2e-4, betas=(0.5, 0.999))

    # total number of epochs = 200
    # starting epoch = 0
    # epoch to start linearly decaying the learning rate to 0 = 100
    self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        self.optimizer_G, lr_lambda=LambdaLR(200, 0, 100).step)
    self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        self.optimizer_D_A, lr_lambda=LambdaLR(200, 0, 100).step)
    self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        self.optimizer_D_B, lr_lambda=LambdaLR(200, 0, 100).step)

    self.train_history = TrainMetadata()

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
    time_start_global = time.time()
    time_start_prev_log = time.time()
    self.cycle_gan.train_mode()
    for epoch_idx in range(num_epochs):
      for batch_idx, batch in enumerate(self.train_loader):
        if batch_idx % 100 == 0:
          print('Epoch #{} Batch#{}'.format(epoch_idx, batch_idx))
          curr_time = time.time()
          print('Time elapsed: Global = {}s Last log = {}s'.format(curr_time-time_start_global,
                                                                   curr_time-time_start_prev_log
                                                                   ))
          time_start_prev_log = curr_time

        if self.cuda:
          inputA, inputB = Variable(batch[0]).cuda(), Variable(batch[1]).cuda()
        else:
          inputA, inputB = Variable(batch[0]), Variable(batch[1])

        # do the loss computation
        loss_generator, loss_cycle = self.cycle_gan.forward_train(
            inputA, inputB)

        # train the generator
        self.optimizer_G.zero_grad()
        (loss_generator + loss_cycle).backward()
        self.optimizer_G.step()

        # train the discriminator A
        loss_discriminator_A = self.cycle_gan.forward_train_DA(inputA, inputB)
        self.optimizer_D_A.zero_grad()
        loss_discriminator_A.backward()
        self.optimizer_D_A.step()

        # train the discriminator B
        loss_discriminator_B = self.cycle_gan.forward_train_DB(inputA, inputB)
        self.optimizer_D_B.zero_grad()
        loss_discriminator_B.backward()
        self.optimizer_D_B.step()

        # get scalar values
        scalar_loss_generator = float(loss_generator.detach().cpu().item())
        scalar_loss_discriminator = float(
            (loss_discriminator_A + loss_discriminator_B).detach().cpu().item())
        scalar_loss_cycle = float(loss_cycle.detach().cpu().item())

        self.train_history.aggregate_loss_vals(
            scalar_loss_generator+scalar_loss_discriminator+scalar_loss_cycle,
            scalar_loss_generator,
            scalar_loss_discriminator,
            scalar_loss_cycle,
            inputA.shape[0])

        if batch_idx % 100 == 0:
          # commit the loss aggregations
          self.train_history.log_losses(epoch_idx)
          self.train_history.save_train_loss(self.model_dir)
          self.save_model()

      # update learning rates
      self.lr_scheduler_G.step()
      self.lr_scheduler_D_A.step()
      self.lr_scheduler_D_B.step()

    self.train_history.log_losses(epoch_idx)
    self.train_history.plot_train_loss()
    self.train_history.plot_train_loss()
