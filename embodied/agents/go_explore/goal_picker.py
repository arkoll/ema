import tensorflow as tf
from tensorflow_probability import distributions as tfd

from . import tfutils

class Null(tfutils.Module):

    def __init__(self, wm, act_space, config):
        self.config = config
        self.act_space = act_space
        self.goal_shape = (1, )

    def initial(self, batch_size):
        return tf.zeros(batch_size)

    def policy(self, latent, state):
        batch_size = len(state)
        shape = (batch_size,)
        return tf.zeros(shape)

    def train(self, imagine, start, data):
        return None, {}

    def report(self, data):
        return {}