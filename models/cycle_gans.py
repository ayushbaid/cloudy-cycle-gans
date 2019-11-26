from models.generator import Generator
from models.descriminator import Descriminator
from utils.buffer import ImageBuffer

import torch

import torch.nn as nn


class CycleGAN(object):
  def __init__(self, is_cuda, use_identity_loss=False, buffer_size=50):

    self.is_cuda = is_cuda
    self.use_identity_loss = use_identity_loss

    # setting up target labels
    self.target_real = self.get_target_tensor(1)

    self.target_fake = self.get_target_tensor(0)

    # init different models
    self.generator_A2B = Generator()
    self.generator_B2A = Generator()

    self.discriminator_A = Descriminator()
    self.discriminator_B = Descriminator()

    if is_cuda:
      self.generator_A2B.cuda()
      self.generator_B2A.cuda()
      self.discriminator_A.cuda()
      self.discriminator_B.cuda()

    # define the loss functions
    # mse loss is used instead of cross entropy to reduce model osscilation
    self.gan_loss_criterion = nn.MSELoss()
    self.cycle_loss_criterion = nn.L1Loss()

    if self.use_identity_loss:
      self.identity_loss_criterion = nn.L1Loss()

    self.fake_buffer_a = ImageBuffer(buffer_size)
    self.fake_buffer_b = ImageBuffer(buffer_size)

  def get_target_tensor(self, value):
    if self.is_cuda:
      return torch.cuda.FloatTensor([value])
    else:
      return torch.FloatTensor([value])

  def forward_train(self, inputA, inputB, lambda_=10):

    # apply the generatorB to convert into B space
    gen_A2B = self.generator_A2B(inputA)
    discriminatorB_output_fake = self.discriminator_B(
        self.fake_buffer_b.get(gen_A2B))
    discriminatorB_output_real = self.discriminator_B(inputB)

    # adding gan loss

    # generator wants to fool the descriminator
    loss_generator_A2B = self.gan_loss_criterion(
        discriminatorB_output_fake,
        self.target_real.expand_as(discriminatorB_output_fake)
    )

    # discriminator wants to prevent being fooled and classify correctly
    loss_discriminatorB = self.gan_loss_criterion(
        discriminatorB_output_real,
        self.target_real.expand_as(discriminatorB_output_real)
    ) + self.gan_loss_criterion(
        discriminatorB_output_fake,
        self.target_fake.expand_as(discriminatorB_output_fake)
    )
    loss_discriminatorB *= 0.5

    # applying cycle on gen_A2B
    gen_A2B2A = self.generator_B2A(gen_A2B)

    loss_cycle_A2B = lambda_*self.cycle_loss_criterion(inputA, gen_A2B2A)

    # apply the generatorA to convert into A space
    gen_B2A = self.generator_B2A(inputB)
    discriminatorA_output_fake = self.discriminator_A(
        self.fake_buffer_a.get(gen_B2A))
    discriminatorA_output_real = self.discriminator_A(inputA)

    # generator wants to fool the descriminator
    loss_generator_B2A = self.gan_loss_criterion(
        discriminatorA_output_fake,
        self.target_real.expand_as(discriminatorA_output_fake)
    )

    # discriminator wants to prevent being fooled and classify correctly
    loss_discriminatorA = self.gan_loss_criterion(
        discriminatorA_output_real,
        self.target_real.expand_as(discriminatorA_output_real)
    ) + self.gan_loss_criterion(
        discriminatorA_output_fake,
        self.target_fake.expand_as(discriminatorA_output_fake)
    )
    loss_discriminatorA *= 0.5

    # applying cycle on gen_B2A
    generator_B2A2B = self.generator_A2B(gen_B2A)

    loss_cycle_B2A = lambda_*self.cycle_loss_criterion(inputB, generator_B2A2B)

    return (loss_generator_A2B + loss_generator_B2A,  # gan loss for generator
            loss_cycle_A2B + loss_cycle_B2A,  # cycle loss
            loss_discriminatorA,  # discriminator_A loss
            loss_discriminatorB  # discriminator_B loss
            )

  def generate_images(self, inputA, inputB, switch_modes=True):
    # put everything in eval mode
    self.generator_A2B.eval()
    self.generator_B2A.eval()
    self.discriminator_A.eval()
    self.discriminator_B.eval()

    with torch.no_grad():
      gen_A2B = self.generator_A2B(inputA)
      gen_B2A = self.generator_B2A(inputB)

      gen_A2B2A = self.generator_B2A(gen_A2B)
      gen_B2A2B = self.generator_A2B(gen_B2A)

    self.generator_A2B.train()
    self.generator_B2A.train()
    self.discriminator_A.train()
    self.discriminator_B.train()

    return gen_A2B, gen_B2A, gen_A2B2A, gen_B2A2B

  def load_from_state(self, state_dict):
    # load the state from disk

    self.generator_A2B.load_state_dict(state_dict['generator_A2B'])
    self.generator_B2A.load_state_dict(state_dict['generator_A2B'])
    self.discriminator_A.load_state_dict(state_dict['discriminator_A'])
    self.discriminator_B.load_state_dict(state_dict['discriminator_B'])

  def extract_state(self):
    return {
        'generator_A2B': self.generator_A2B.state_dict(),
        'generator_B2A': self.generator_B2A.state_dict(),
        'discriminator_A': self.discriminator_A.state_dict(),
        'discriminator_B': self.discriminator_B.state_dict()
    }

  def train_mode(self):
    '''
    Sets all the models to train mode
    '''
    self.generator_A2B.train()
    self.generator_B2A.train()
    self.discriminator_A.train()
    self.discriminator_B.train()

  def eval_mode(self):
    '''
    Sets all the models to eval mode
    '''
    self.generator_A2B.eval()
    self.generator_B2A.eval()
    self.discriminator_A.eval()
    self.discriminator_B.eval()
