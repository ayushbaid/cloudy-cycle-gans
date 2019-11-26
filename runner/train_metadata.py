import matplotlib.pyplot as plt


class TrainMetadata(object):
  '''
  class to save loss values, and plot them too
  '''

  def __init__(self):
    self.loss_train_total = []
    self.loss_train_generator = []
    self.loss_train_discriminator = []
    self.loss_train_cycle = []

    # variables for aggegating over an epoch
    self.agg_num_samples = 0
    self.agg_loss_total = 0
    self.agg_loss_generator = 0
    self.agg_loss_discriminator = 0
    self.agg_loss_cycle = 0

  def aggregate_loss_vals(self,
                          total_loss,
                          generator_loss,
                          discriminator_loss,
                          cycle_loss,
                          batch_size):
    self.agg_num_samples += batch_size
    self.agg_loss_total += total_loss
    self.agg_loss_generator += generator_loss
    self.agg_loss_discriminator += discriminator_loss
    self.agg_loss_cycle += cycle_loss

  def log_losses(self):
    '''
    Logs the losses after some granularity 
    '''
    self.loss_train_total.append(
        self.agg_loss_total/self.agg_num_samples
    )
    self.loss_train_generator.append(
        self.agg_loss_generator/self.agg_num_samples)
    self.loss_train_discriminator.append(
        self.agg_loss_discriminator/self.agg_num_samples)
    self.loss_train_cycle.append(
        self.agg_loss_cycle/self.agg_num_samples)

    # reset aggegations
    self.agg_num_samples = 0
    self.agg_loss_total = 0
    self.agg_loss_generator = 0
    self.agg_loss_discriminator = 0
    self.agg_loss_cycle = 0

  def plot_train_loss(self):
    plt.figure()
    plt.plot(self.loss_train_total, label='total')
    plt.plot(self.loss_train_generator, label='generator')
    plt.plot(self.loss_train_discriminator, label='discriminator')
    plt.plot(self.loss_train_cycle, label='cycle')

    plt.title('Loss plots')
