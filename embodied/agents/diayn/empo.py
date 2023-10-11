import functools

import embodied
import numpy as np
import tensorflow as tf
from tensorflow_probability import distributions as tfd

from . import agent
from . import expl
from . import nets
from . import tfutils


class DIAYN(tfutils.Module):

    def __init__(self, wm, act_space, config):
        self.config = config
        self.wm = wm
        skills = self.config.skill_shape[0]
        probs = tf.fill([1, skills], 1 / skills)
        self.prior = tfutils.OneHotDist(probs=probs)
        self.skill_enc = nets.MLP((skills,), **self.config.skill_encoder)
        self.reward = lambda traj: (
            self.skill_enc(traj).log_prob(traj['skill']) - 
            self.prior.log_prob(traj['skill'])
        )[1:]
        self.opt = tfutils.Optimizer('skill', **config.skill_opt)

        wconfig = config.update({
            'actor.inputs': self.config.worker_inputs,
            'critic.inputs': self.config.worker_inputs,
        })
        self.worker = agent.ImagActorCritic({
            'diayn': agent.VFunction(self.reward, wconfig),
        }, {'diayn': 1.0}, act_space, wconfig)

    def initial(self, batch_size):
        return {
            'step': tf.zeros((batch_size,), tf.int64),
            'skill': tf.zeros(
                (batch_size,) + self.config.skill_shape, tf.float32
            ),
        }
    
    def policy(self, latent, carry, imag=False):
        duration = self.config.train_skill_duration if imag else (
            self.config.env_skill_duration
        )
        sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
        update = (carry['step'] % duration) == 0
        switch = lambda x, y: (
            tf.einsum('i,i...->i...', 1 - update.astype(x.dtype), x) +
            tf.einsum('i,i...->i...', update.astype(x.dtype), y)
        )
        skill = sg(switch(
            carry['skill'],
            tf.squeeze(self.prior.sample((carry['skill'].shape[0],)))
        ))
        dist = self.worker.actor(sg({**latent, 'skill': skill}))
        outs = {'action': dist}
        carry = {'step': carry['step'] + 1, 'skill': skill}
        return outs, carry

    def train(self, imagine, start, data):
        metrics = {}
        skill = tf.squeeze(self.prior.sample((start['action'].shape[0],)))
        start = start.copy()
        sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
        with tf.GradientTape(persistent=True) as tape:
            worker = lambda s: self.worker.actor(
                sg({**s, 'skill': skill,})
            ).sample()
            traj = imagine(worker, start, self.config.imag_horizon)
            traj['skill'] = tf.repeat(
                skill[None], 1 + self.config.imag_horizon, 0
            )
            loss = - self.reward(sg(traj)).mean()
        metrics.update(self.opt(tape, loss, [self.skill_enc]))
        mets = self.worker.update(traj, tape)
        metrics.update({f'worker_{k}': v for k, v in mets.items()})
        return None, metrics

    def report(self, data):
        states, _ = self.wm.rssm.observe(
            self.wm.encoder(data), data['action'], data['is_first']
        )
        decoder = self.wm.heads['decoder']
        skills = self.skill_enc(states).mode()
        reshape = lambda x: x.reshape([x.shape[0] * x.shape[1],] + x.shape[2:])
        skills = reshape(self.skill_enc(states).mode())
        r_coords = reshape(data['absolute_position'])
        coords = reshape(decoder(states)['absolute_position'].mode())
        skills = tf.where(skills)[:, 1]
        all_coord = tf.concat([r_coords, coords], 0)
        min_lim = tf.reduce_min(all_coord, 0)[:2]
        max_lim = tf.reduce_max(all_coord, 0)[:2]
        img_data = (coords, r_coords, skills, min_lim, max_lim)
        return {'skill_map': img_data}
