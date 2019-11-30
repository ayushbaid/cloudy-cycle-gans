import os

import matplotlib.pyplot as plt


class TrainMetadata(object):
  '''
  class to save loss values, and plot them too
  '''

  def __init__(self):
    self.loss_train_total = []
    self.loss_train_generator_A2B = []
    self.loss_train_generator_B2A = []
    self.loss_train_generator_identity = []
    self.loss_train_discriminator = []
    self.loss_train_cycle = []
    self.epoch_vals = []

    # variables for aggegating over an epoch
    self.agg_num_samples = 0
    self.agg_loss_total = 0
    self.agg_loss_generator_A2B = 0
    self.agg_loss_generator_B2A = 0
    self.agg_loss_generator_identity = 0
    self.agg_loss_discriminator = 0
    self.agg_loss_cycle = 0

  def aggregate_loss_vals(self,
                          total_loss,
                          generator_loss_A2B,
                          generator_loss_B2A,
                          generator_loss_identity,
                          discriminator_loss,
                          cycle_loss,
                          batch_size):
    self.agg_num_samples += batch_size
    self.agg_loss_total += total_loss*batch_size
    self.agg_loss_generator_A2B += generator_loss_A2B*batch_size
    self.agg_loss_generator_B2A += generator_loss_B2A*batch_size
    self.agg_loss_generator_identity += generator_loss_identity*batch_size
    self.agg_loss_discriminator += discriminator_loss*batch_size
    self.agg_loss_cycle += cycle_loss*batch_size

  def log_losses(self, epoch_idx):
    '''
    Logs the losses after some granularity 
    '''
    if self.agg_num_samples == 0:
      return

    self.loss_train_total.append(
        self.agg_loss_total/self.agg_num_samples
    )
    self.loss_train_generator_A2B.append(
        self.agg_loss_generator_A2B/self.agg_num_samples)
    self.loss_train_generator_B2A.append(
        self.agg_loss_generator_B2A/self.agg_num_samples)
    self.loss_train_generator_identity.append(
        self.agg_loss_generator_identity/self.agg_num_samples)
    self.loss_train_discriminator.append(
        self.agg_loss_discriminator/self.agg_num_samples)
    self.loss_train_cycle.append(
        self.agg_loss_cycle/self.agg_num_samples)
    self.epoch_vals.append(epoch_idx)

    print('Losses: T={} G_a2b={}, G_b2a={}, G_identity={}, D={}, C={}'.format(self.loss_train_total[-1],
                                                 self.loss_train_generator_A2B[-1],
                                                 self.loss_train_generator_B2A[-1],
                                                 self.loss_train_generator_identity[-1],
                                                 self.loss_train_discriminator[-1],
                                                 self.loss_train_cycle[-1]
                                                 ))

    # reset aggegations
    self.agg_num_samples = 0
    self.agg_loss_total = 0
    self.agg_loss_generator_A2B = 0
    self.agg_loss_generator_B2A = 0
    self.agg_loss_generator_identity = 0
    self.agg_loss_discriminator = 0
    self.agg_loss_cycle = 0

  def save_train_loss(self, dir_name):
    fig = plt.figure()
    plt.plot(self.loss_train_total, label='total')
    plt.plot(self.loss_train_generator_A2B, label='generatorA2B')
    plt.plot(self.loss_train_generator_B2A, label='generatorB2A')
    plt.plot(self.loss_train_generator_identity, label='generator_identity')
    plt.plot(self.loss_train_discriminator, label='discriminator')
    plt.plot(self.loss_train_cycle, label='cycle')

    plt.title('Loss plots')
    plt.legend()
    fig.savefig(os.path.join(dir_name, 'loss_plot.jpg'))
    plt.close()

  def plot_train_loss(self):
    plt.figure()
    plt.plot(self.loss_train_total, label='total')
    plt.plot(self.loss_train_generator_A2B, label='generatorA2B')
    plt.plot(self.loss_train_generator_B2A, label='generatorB2A')
    plt.plot(self.loss_train_generator_identity, label='generator_identity')
    plt.plot(self.loss_train_discriminator, label='discriminator')
    plt.plot(self.loss_train_cycle, label='cycle')

    plt.title('Loss plots')
    plt.legend()
    plt.show()
