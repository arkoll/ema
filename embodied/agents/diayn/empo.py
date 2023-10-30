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

        # DIAYN skills
        skills = self.config.skill_shape[0]
        probs = tf.fill([1, skills], 1 / skills)
        self.prior = tfutils.OneHotDist(probs=probs)
        self.skill_enc = nets.MLP((skills,), **self.config.skill_encoder)
        self.reward = lambda traj: (
            self.skill_enc(traj).log_prob(traj['skill']) - 
            self.prior.log_prob(traj['skill'])
        )[1:]
        self.opt = tfutils.Optimizer('skill', **config.skill_opt)

        # DIAYN Worker
        wconfig = config.update({
            'actor.inputs': self.config.worker_inputs,
            'critic.inputs': self.config.worker_inputs,
        })
        self.worker = agent.ImagActorCritic({
            'diayn': agent.VFunction(self.reward, wconfig),
        }, {'diayn': 1.0}, act_space, wconfig)

        # Goal achiever
        aconfig = config.update({
            'actor.inputs': self.config.achiever_inputs,
            'critic.inputs': self.config.achiever_inputs,
        })
        self.achiever = agent.ImagActorCritic({
            'goal': agent.VFunction(lambda s: s['reward_goal'], aconfig),
        }, {'goal': 1.0}, act_space, aconfig)

        # VAE landmarks
        shape = config.landmark_shape
        self.landmark_prior = tfutils.OneHotDist(tf.zeros(shape))
        self.feat = nets.Input(['deter'])
        self.goal_shape = (self.config.rssm.deter,)
        self.goal_enc = nets.MLP(shape, dims='context', **config.goal_encoder)
        self.goal_dec = nets.MLP(
            self.goal_shape, dims='context', **self.config.goal_decoder
        )
        self.goal_kl = tfutils.AutoAdapt((), **self.config.encdec_kl)
        self.goal_opt = tfutils.Optimizer('goal', **config.encdec_opt)

        # Explorer
        self.expl_reward = expl.Disag(wm, act_space, config)
        econfig = config.update({
            'discount': config.expl_discount,
            'retnorm': config.expl_retnorm,
        })
        self.explorer = agent.ImagActorCritic({
            'expl': agent.VFunction(self.expl_reward, econfig),
        }, {'expl': 1.0}, act_space, config)

    def initial(self, batch_size):
        return {
            'step': tf.zeros((batch_size,), tf.int64),
            'skill': tf.zeros(
                (batch_size,) + self.config.skill_shape, tf.float32
            ),
        }
    
    def policy(self, latent, carry, imag=False):
        # duration = self.config.train_skill_duration if imag else (
        #     self.config.env_skill_duration
        # )
        sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
        # update = (carry['step'] % duration) == 0
        # switch = lambda x, y: (
        #     tf.einsum('i,i...->i...', 1 - update.astype(x.dtype), x) +
        #     tf.einsum('i,i...->i...', update.astype(x.dtype), y)
        # )
        # skill = sg(switch(
        #     carry['skill'],
        #     tf.squeeze(self.prior.sample((carry['skill'].shape[0],)))
        # ))
        # dist = self.worker.actor(sg({**latent, 'skill': skill}))
        dist = self.explorer.actor(sg({**latent}))
        outs = {'action': dist}
        carry = {'step': carry['step'] + 1, 'skill': carry['skill']}
        return outs, carry

    def train(self, imagine, start, data):
        metrics = {}
        # Disagreement
        metrics.update(self.expl_reward.train(data))
        # Landmarks
        metrics.update(self.train_vae_replay(data))
        # Explorer
        traj, mets = self.explorer.train(imagine, start, data)
        metrics.update({f'explorer_{k}': v for k, v in mets.items()})
        # Achiever
        mets = self.train_achiever(imagine, start)
        metrics.update({f'achiever_{k}': v for k, v in mets.items()})
        # DIAYN
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
    
    def train_achiever(self, imagine, start):
        start = start.copy()
        goal = start['deter']
        shift = tfd.Geometric(
            probs=tf.fill([goal.shape[0]], self.config.goal_shift_prob)
        ).sample().astype(tf.int32)
        indices = tf.range(goal.shape[0], dtype=tf.int32)
        goal_indices = tf.math.floormod(indices + shift + 1, goal.shape[0])
        goal = tf.gather(goal, goal_indices)
        
        sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
        with tf.GradientTape(persistent=True) as tape:
            achiever = lambda s: self.achiever.actor(
                sg({**s, 'goal': goal,})
            ).sample()
            traj = imagine(achiever, start, self.config.imag_horizon)
            traj['goal'] = tf.repeat(
                goal[None], 1 + self.config.imag_horizon, 0
            )
            traj['reward_goal'] = self.goal_reward(traj['goal'], traj['deter'])
        mets = self.achiever.update(traj, tape)
        success = tf.reduce_sum(
            (traj['reward_goal'] > -1).astype(tf.int16), axis=0
        )
        success = tf.reduce_mean((success > 0).astype(tf.int16))
        mets['success'] = success
        mets['mean_shift'] = tf.reduce_mean(shift + 1)
        return mets
    
    def train_vae_replay(self, data):
        metrics = {}
        feat = self.feat(data).astype(tf.float32)
        goal = context = feat
        with tf.GradientTape() as tape:
            enc = self.goal_enc({'goal': goal, 'context': context})
            dec = self.goal_dec({'landmark': enc.sample(), 'context': context})
            rec = -dec.log_prob(tf.stop_gradient(goal))
            if self.config.goal_kl:
                kl = tfd.kl_divergence(enc, self.landmark_prior)
                kl, mets = self.goal_kl(kl)
                metrics.update({f'goalkl_{k}': v for k, v in mets.items()})
                assert rec.shape == kl.shape, (rec.shape, kl.shape)
            else:
                kl = 0.0
            loss = (rec + kl).mean()
        metrics.update(
            self.goal_opt(tape, loss, [self.goal_enc, self.goal_dec])
        )
        metrics['goalrec_mean'] = rec.mean()
        metrics['goalrec_std'] = rec.std()
        return metrics
    
    def eval_goals(self, goals):
        start = {
            'deter': tf.cast(goals, tf.float16),
            'stoch': self.wm.rssm.get_stoch(goals),
            **self.wm.rssm.get_stats(goals)
        }
        n_goals = goals.shape[0]
        start['is_terminal'] = tf.zeros(n_goals)
        samples = self.config.expl_samples
        start = {
            k: tf.repeat(v, samples, 0) for k, v in start.items()
        }
        explorer = lambda s: self.explorer.actor({**s}).sample()
        traj = self.wm.imagine(explorer, start, self.config.expl_horizon)
        reward = self.expl_reward(traj)

        length = self.config.expl_horizon
        reward = tf.reshape(reward, (length, n_goals, samples, -1))
        reward = tf.squeeze(reward.mean([0, 2]))
        max_ind = tf.argmax(reward)
        decoder = self.wm.heads['decoder']
        traj = {
            k: tf.gather(tf.reshape(
                v, (length + 1, n_goals, samples,) + v.shape[2:]
            ), [max_ind], axis=1) 
            for k, v in traj.items()
        }
        traj = tf.squeeze(decoder(traj)['absolute_position'].mode())
        return reward, traj
    
    def goal_reward(self, goal, state):
        goal = tf.stop_gradient(goal.astype(tf.float32))
        delta_2 = tf.math.squared_difference(goal, state.astype(tf.float32))
        eps = tf.math.square(self.config.goal_epsilon * goal)
        distance = tf.exp(
            -tf.reduce_sum(tf.math.divide_no_nan(delta_2, eps), -1)
        ) - 1
        return distance[1:]

    def report(self, data):
        states, _ = self.wm.rssm.observe(
            self.wm.encoder(data)[:1, :5], data['action'][:1, :5],
            data['is_first'][:1, :5]
        )
        n_skills = self.config.skill_shape[0]
        n_samp = self.config.skill_samples
        skill = tf.repeat(tf.eye(n_skills), n_samp, 0)
        worker = lambda s: self.worker.actor({**s, 'skill': skill,}).sample()
        decoder = self.wm.heads['decoder']
        
        # Imagine skill trajs from random state
        start = {k: v[:1, 4] for k, v in states.items()}
        start['is_terminal'] = data['is_terminal'][:1, 4]
        start = {
            k: tf.repeat(v, n_skills * n_samp, 0) for k, v in start.items()
        }
        traj = self.wm.imagine(worker, start, self.config.worker_report_horizon)
        initial = decoder(start)['absolute_position'].mode()[0]
        rollout = decoder(traj)['absolute_position'].mode()
        length = 1 + self.config.worker_report_horizon
        rollout = tf.reshape(rollout, (length, n_skills, n_samp, -1))
        outputs = {'skill_trajs': (initial, rollout)}

        # Imagine skill trajs from start
        start = self.wm.rssm.initial(n_skills * n_samp)
        start['is_terminal'] = tf.zeros(n_skills * n_samp)
        traj = self.wm.imagine(worker, start, self.config.skill_horizon)
        indices = tf.range(0, self.config.skill_horizon + 1, 10)
        traj = {k: tf.gather(v, indices, axis=0) for k, v in traj.items()}
        skill_goals = decoder(traj)['absolute_position'].mode()
        skill_goals = tf.reshape(
            skill_goals, (len(indices), n_skills, n_samp, -1)
        )
        outputs['skill_goals'] = skill_goals

        # Eval VAE goals
        landmarks = tf.eye(self.config.landmark_shape[0])
        goals = self.goal_dec(
            {'landmark': landmarks, 'context': landmarks}
        ).mode()
        reward, max_traj = self.eval_goals(goals)
        goals = decoder(
            {'deter': goals, 'stoch': self.wm.rssm.get_stoch(goals)}
        )['absolute_position'].mode()
        outputs['landmarks'] = (goals, reward, max_traj)
        return outputs
