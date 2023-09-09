import functools

import embodied
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from . import agent
from . import expl
from . import nets
from . import tfutils

from . import goal_picker
# import expl
# import worker
from . import behaviors

class PEG(tfutils.Module):

    def __init__(self, wm, act_space, config):
        self.wm = wm
        self.act_space = act_space
        self.config = config
        self.goal_picker = getattr(goal_picker, config.goal_picker)(
            self.wm, self.act_space, self.config
        )
        self.config.expl_rewards[config.explorer]
        self.explorer = behaviors.Explore(self.wm, self.act_space, self.config)
        self.worker = getattr(behaviors, config.worker)(
            self.wm, self.act_space, self.config
        )
        self.goal_shape = self.goal_picker.goal_shape
    
    def policy(self, latent, carry):
        sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)

        update = carry['step'] == 0
        switch = lambda x, y: (
                tf.einsum('i,i...->i...', 1 - update.astype(x.dtype), x) +
                tf.einsum('i,i...->i...', update.astype(x.dtype), y))
        goal = sg(switch(carry['goal'], self.goal_picker.policy(latent, carry)))
        if (carry['step'] < self.config.gc_duration).all():
            actor = self.worker
            obs = sg({**latent, 'goal': goal})
        else:
            actor = self.explorer
            obs = sg(latent)
        outs, _ = actor.policy(obs, carry)
        carry = {'step': carry['step'] + 1, 'goal': goal}
        return outs, carry

    def initial(self, batch_size):
        return {
                'step': tf.zeros((batch_size,), tf.int64),
                'goal': tf.zeros((batch_size,) + self.goal_shape, tf.float32),
        }
    
    def train(self, imageine, start, data):
        return None, {}

    def report(self, data):
        return {}