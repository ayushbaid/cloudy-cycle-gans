from models.generator import Generator
from models.descriminator import Descriminator

import torch.nn as nn


class CycleGAN(object):
  def __init__(self, is_cuda, use_identity_loss=False):

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
    self.gan_loss_criterion = nn.BCEWithLogitsLoss()
    self.cycle_loss_criterion = nn.L1Loss()

    if self.use_identity_loss:
      self.identity_loss_criterion = nn.L1Loss()

  def get_target_tensor(self, value):
    if self.is_cuda:
      return torch.cuda.LongTensor(value)
    else:
      return torch.LongTensor(value)

  def forward_train(self, inputA, inputB, lambda_=10):

    # apply the generatorB to convert into B space
    gen_A2B = self.generator_A2B(inputA)
    discriminatorB_output_fake = self.discriminator_B(gen_A2B)
    discriminatorB_output_real = self.discriminator_B(inputB)

    # adding gan loss
    gan_loss_A2B = self.gan_loss_criterion(
        discriminatorB_output_fake,
        self.target_fake.expand_as(discriminatorB_output_fake)
    ) + self.gan_loss_criterion(
        discriminatorB_output_real,
        self.target_real.expand_as(discriminatorB_output_real)
    )

    # applying cycle on gen_A2B
    gen_A2B2A = self.generator_B2A(gen_A2B)

    cycle_loss_A2B = lambda_*self.cycle_loss_criterion(inputA, gen_A2B2A)

    # apply the generatorA to convert into A space
    gen_B2A = self.generator_B2A(inputB)
    discriminatorA_output_fake = self.discriminator_A(gen_B2A)
    discriminatorA_output_real = self.discriminator_A(inputA)

    # adding gan loss
    gan_loss_A2B = self.gan_loss_criterion(
        discriminatorA_output_fake,
        self.target_fake.expand_as(discriminatorA_output_fake)
    ) + self.gan_loss_criterion(
        discriminatorA_output_real,
        self.target_real.expand_as(discriminatorA_output_real)
    )

    # applying cycle on gen_B2A
    generator_B2A2B = self.generator_A2B(gen_B2A)

    cycle_loss_B2A = lambda_*self.cycle_loss_criterion(inputB, generator_B2A2B)

    return gan_loss_A2B + gen_B2A + cycle_loss_B2A + cycle_loss_A2B

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
