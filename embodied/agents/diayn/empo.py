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
            'actor': self.config.actor_ach,
            'critic.inputs': self.config.achiever_inputs,
            'actent': self.config.actent_ach
        })
        self.achiever = agent.ImagActorCritic({
            'goal': agent.VFunction(lambda s: self.goal_reward(s), aconfig),
        }, {'goal': 1.0}, act_space, aconfig)
        self.dyndist = nets.MLP((), **self.config.dyndist)
        self.dd_opt = tfutils.Optimizer('dyndist', **config.dyndist_opt)
        self.dd_cur_idxs, self.dd_fut_idxs = self.get_future_idxs()

        # VAE landmarks
        if self.config.explore_goals == 'VAE':
            shape = config.landmark_shape
            self.landmark_prior = tfutils.OneHotDist(tf.zeros(shape))
            self.feat = nets.Input(['deter'])
            self.goal_shape = (self.config.rssm.deter,)
            self.goal_enc = nets.MLP(
                shape, dims='context', **config.goal_encoder
            )
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
            'goal': tf.zeros(
                (batch_size,) + self.config.goal_shape, tf.float32
            ),
            'goal_pos': tf.zeros((batch_size, 2), tf.float32),
            'skill': tf.zeros(
                (batch_size,) + self.config.skill_shape, tf.float32
            ),
        }
    
    def policy(self, latent, carry, mode):
        duration = self.config.goal_duration
        sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
        update = (carry['step'] % duration) == 0
        update_exp = ((carry['step'] // duration) % 2) == 0
        switch = lambda x, y, upd: (
            tf.einsum('i,i...->i...', 1 - upd.astype(x.dtype), x) +
            tf.einsum('i,i...->i...', upd.astype(x.dtype), y)
        )
        if tf.math.reduce_any(update):
            self.update_goal_buffer()

        if mode == 'train' or mode == 'explore':
            goals = self.goal_buffer_embed
            bs = carry['goal'].shape[0]
            goal = tfd.Categorical(probs=self.goal_buffer_prob).sample(bs)
            goal_pos = tf.gather(self.goal_buffer_pos, goal)
            goal = tf.gather(goals, goal)
            goal = sg(switch(carry['goal'], goal, update))
            goal_pos = sg(switch(carry['goal_pos'], goal_pos, update))
            skill = sg(switch(
                carry['skill'],
                tf.squeeze(self.prior.sample((carry['skill'].shape[0],))),
                update
            ))
            if self.config.use_goals:
                a_act = self.achiever.actor(
                    sg({**latent, 'goal': goal})
                ).sample()
            else:
                a_act = self.worker.actor(
                    sg({**latent, 'skill': skill})
                ).sample()
            e_act = self.explorer.actor(sg({**latent})).sample()
            act = switch(e_act, a_act, update_exp)
        elif mode == 'eval':
            skill = carry['skill']
            goal = carry['goal']
            goal_pos = carry['goal_pos']
            act = self.achiever.actor(sg({**latent, 'goal': goal})).mode()
        outs = {'action': act}
        outs['log_position'] = self.wm.heads['decoder'](latent)[
            'absolute_position'
        ].mode()
        outs['log_cgoal'] = goal_pos 
        outs['log_update'] = update
        outs['log_update_exp'] = update_exp
        carry = {'step': carry['step'] + 1, 'goal': goal, 'goal_pos': goal_pos, 'skill': skill}
        return outs, carry

    def train(self, imagine, start, data):
        metrics = {}
        # Disagreement
        metrics.update(self.expl_reward.train(data))
        # Landmarks
        if self.config.explore_goals == 'VAE':
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
    
    def update_goal_buffer(self):
        goal_states = self.propose_explore_goals()
        weights = self.eval_goals(goal_states)
        embed = self.wm.heads['embed'](goal_states).mode()
        position = self.wm.heads['decoder'](goal_states)[
            'absolute_position'
        ].mode()
        if not hasattr(self, "goal_buffer_pos"):
            self.goal_buffer_pos = tf.Variable(
                tf.zeros_like(position), trainable=False, name="goal_buffer_pos"
            )
        if not hasattr(self, "goal_buffer_embed"):
            self.goal_buffer_embed = tf.Variable(
                tf.zeros_like(embed), trainable=False, name="goal_buffer_embed"
            )
        if not hasattr(self, "goal_buffer_prob"):
            self.goal_buffer_prob = tf.Variable(
                tf.zeros_like(weights), trainable=False, name="goal_buffer_prob"
            )
        self.goal_buffer_embed.assign(embed)
        self.goal_buffer_pos.assign(position)
        self.goal_buffer_prob.assign(weights)

    def propose_explore_goals(self):
        if self.config.explore_goals == 'DIAYN':
            n_skills = self.config.skill_shape[0]
            n_samp = self.config.skill_samples
            skill = tf.repeat(tf.eye(n_skills), n_samp, 0)
            pi = lambda s: self.worker.actor({**s, 'skill': skill,}).sample()
            traj = self.wm.rssm.initial(n_skills * n_samp)
            traj['is_terminal'] = tf.zeros(n_skills * n_samp)
            traj = self.wm.imagine(pi, traj, self.config.horizon_to_find_goals)
            goals = {k: v[-1] for k, v in traj.items()}
        elif self.config.explore_goals == 'VAE':
            landmarks = tf.eye(self.config.landmark_shape[0])
            goals = self.goal_dec(
                {'landmark': landmarks, 'context': landmarks}
            ).mode()
            goals = {
                'deter':  tf.cast(goals, tf.float16),
                'stoch': self.wm.rssm.get_stoch(goals),
                **self.wm.rssm.get_stats(goals)
            }
        return goals

    def propose_train_goals(self, start):
        if self.config.train_goals == 'batch':
            goal = start['embed']
            ids = tf.random.shuffle(tf.range(tf.shape(goal)[0]))
            goal = tf.gather(goal, ids)
        elif self.config.train_goals == 'buffer':
            self.update_goal_buffer()
            goals = self.goal_buffer_embed
            bs = start['embed'].shape[0]
            goal = tfd.Categorical(probs=self.goal_buffer_prob).sample(bs)
            goal = tf.gather(goals, goal)
        return tf.stop_gradient(goal)
    
    def get_future_idxs(self):
        cur_idx_list = []
        fut_idx_list = []
        for cur_idx in tf.range(self.config.imag_horizon + 1):
            for fut_idx in tf.range(cur_idx, self.config.imag_horizon + 1):
                cur_idx_list.append(cur_idx)
                fut_idx_list.append(fut_idx)
        return tf.concat(cur_idx_list, 0), tf.concat(fut_idx_list, 0)
    
    def dd_loss(self, inp):
        seq_len = inp.shape[0]
        sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
        n_negatives = int(self.config.dd_batch_size * self.config.dd_negatives)
        n_positives = self.config.dd_batch_size - n_negatives

        # positives
        indices = tfd.Categorical(
            logits=tf.zeros(len(self.dd_cur_idxs))
        ).sample(n_positives)
        b_ind = tfd.Categorical(
            logits=tf.zeros(self.config.replay_chunk * self.config.batch_size)
        ).sample(n_positives)
        c_ind = tf.stack([tf.gather(self.dd_cur_idxs, indices), b_ind], axis=1)
        f_ind = tf.stack([tf.gather(self.dd_fut_idxs, indices), b_ind], axis=1) 
        state1 = tf.gather_nd(inp, c_ind)
        state2 = tf.gather_nd(inp, f_ind)
        distance = (f_ind[:, 0] - c_ind[:, 0]).astype(tf.float32) / seq_len

        # negatives
        seq_ind = tfd.Categorical(logits=tf.zeros(seq_len)).sample(
            (2, n_negatives)
        )
        b_ind = tfd.Categorical(
            logits=tf.zeros(self.config.replay_chunk * self.config.batch_size)
        ).sample(n_negatives)
        neg_ch_ind = tfd.Categorical(
            logits=tf.zeros(self.config.batch_size - 1)
        ).sample(n_negatives)
        neg_goal_ind = tfd.Categorical(
            logits=tf.zeros(self.config.replay_chunk)
        ).sample(n_negatives)
        neg_b_ind = tf.math.floormod(
            tf.math.floordiv(b_ind, self.config.replay_chunk) + neg_ch_ind + 1,
            self.config.batch_size
        ) * self.config.replay_chunk + neg_goal_ind
        c_ind = tf.stack([seq_ind[0], b_ind], axis=1)
        f_ind = tf.stack([seq_ind[1], neg_b_ind], axis=1)
        state1 = tf.concat([state1, tf.gather_nd(inp, c_ind)], axis=0)
        state2 = tf.concat([state2, tf.gather_nd(inp, f_ind)], axis=0)
        distance = tf.stop_gradient(tf.concat(
            [distance, tf.ones(n_negatives, dtype=tf.float32)], axis=0
        ))
        dist = self.dyndist(sg({'embed': state1, 'goal': state2}))
        loss = - dist.log_prob(distance).mean()
        return loss
    
    def train_achiever(self, imagine, start):
        start = start.copy()
        goal = self.propose_train_goals(start)
        sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
        with tf.GradientTape(persistent=True) as tape:
            achiever = lambda s: self.achiever.actor(
                sg({**s, 'goal': goal,})
            ).sample()
            traj = imagine(achiever, start, self.config.imag_horizon)
            traj['goal'] = tf.repeat(
                goal[None], 1 + self.config.imag_horizon, 0
            )
        mets = self.achiever.update(traj, tape)
        with tf.GradientTape() as tape:
            dd_loss = self.dd_loss(self.wm.heads['embed'](traj).mode())
        mets.update(self.dd_opt(tape, dd_loss, [self.dyndist]))
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
        if self.config.eval_goals == 'DIAYN':
            n_skills = self.config.skill_shape[0]
            n_goals = goals['deter'].shape[0]
            skill = tf.tile(tf.eye(n_skills), [n_goals, 1])
            start = {
                k: tf.repeat(v, n_skills, 0) for k, v in goals.items()
            }
            start['skill'] = skill
            weights = self.worker.critics['diayn'].net(start).mode()
            weights = tf.reshape(weights, [n_goals, n_skills])
            weights = tf.reduce_mean(weights, 1)
        elif self.config.eval_goals == 'Explore':
            start = goals.copy()
            n_goals = goals['deter'].shape[0]
            start['is_terminal'] = tf.zeros(n_goals)
            samples = self.config.expl_samples
            start = {k: tf.repeat(v, samples, 0) for k, v in start.items()}
            explorer = lambda s: self.explorer.actor({**s}).sample()
            traj = self.wm.imagine(explorer, start, self.config.expl_horizon)
            reward = self.expl_reward(traj)
            length = self.config.expl_horizon
            reward = tf.reshape(reward, (length, n_goals, samples, -1))
            weights = tf.squeeze(reward.mean([0, 2]))
        weights = (weights - weights.min()) / (
            weights.max() - weights.min()
        )
        weights = tf.exp(weights * self.config.goal_temp)
        weights = weights / weights.sum()
        return weights

    def goal_reward(self, traj):
        embed = self.wm.heads['embed'](traj).mode()
        goal = tf.stop_gradient(traj['goal'].astype(tf.float32))
        reward = - self.dyndist({'embed': embed, 'goal': goal}).mode()
        return reward[1:]

    def report(self, data):
        states, _ = self.wm.rssm.observe(
            self.wm.encoder(data)[:1], data['action'][:1],
            data['is_first'][:1]
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

        # Explore buffer goals
        goals = self.goal_buffer_embed
        states, _ = self.wm.rssm.observe(
            tf.expand_dims(goals, 1).astype(tf.float16), 
            tf.zeros((goals.shape[0], 1, data['action'].shape[-1])),
            tf.ones((goals.shape[0], 1))
        )
        goals = tf.squeeze(decoder(states)['absolute_position'].mode())
        saved_goals = self.goal_buffer_pos
        goal_weights = self.goal_buffer_prob
        outputs['buffer_goals'] = (goals, saved_goals, goal_weights)
        return outputs
