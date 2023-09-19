import tensorflow as tf
from tensorflow_probability import distributions as tfd

from . import tfutils

class Random(tfutils.Module):
    def __init__(self, wm, act_space, config):
        self.wm = wm
        self.config = config
        self.act_space = act_space
        self.obs_shape = (2, ) # TODO: generalize for all envs
        self.goal_shape = (self.wm.rssm._deter, )

    def initial(self, batch_size):
        return tf.zeros(batch_size)
    
    def policy(self, latent, state):
        batch_size = len(state['step'])
        shape = (batch_size, ) + self.obs_shape
        goal = tf.random.uniform(shape, minval=0, maxval=10) # TODO: generalize for all envs
        goal_obs = {'observation': goal}
        goal_embed = self.wm.encoder(goal_obs)
        return goal_embed

    def train(self, imagine, start, data):
        return None, {}

    def report(self, data):
        return {}

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