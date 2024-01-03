import tensorflow as tf
from tensorflow_probability import distributions as tfd

from . import agent
from . import expl
from . import tfutils
from . import nets


class Greedy(tfutils.Module):

    def __init__(self, wm, act_space, config):
        rewfn = lambda s: wm.heads['reward'](s).mean()[1:]
        if config.critic_type == 'vfunction':
            critics = {'extr': agent.VFunction(rewfn, config)}
        elif config.critic_type == 'qfunction':
            critics = {'extr': agent.QFunction(rewfn, config)}
        self.ac = agent.ImagActorCritic(
            critics, {'extr': 1.0}, act_space, config
        )

    def initial(self, batch_size):
        return self.ac.initial(batch_size)

    def policy(self, latent, state):
        return self.ac.policy(latent, state)

    def train(self, imagine, start, data):
        return self.ac.train(imagine, start, data)

    def report(self, data):
        return {}


class Random(tfutils.Module):

    def __init__(self, wm, act_space, config):
        self.config = config
        self.act_space = act_space

    def initial(self, batch_size):
        return tf.zeros(batch_size)

    def policy(self, latent, state):
        batch_size = len(state)
        shape = (batch_size,) + self.act_space.shape
        if self.act_space.discrete:
            dist = tfutils.OneHotDist(tf.zeros(shape))
        else:
            dist = tfd.Uniform(-tf.ones(shape), tf.ones(shape))
            dist = tfd.Independent(dist, 1)
        return {'action': dist}, state

    def train(self, imagine, start, data):
        return None, {}

    def report(self, data):
        return {}


class Explore(tfutils.Module):

    REWARDS = {
        'disag': expl.Disag,
        'vae': expl.LatentVAE,
        'ctrl': expl.CtrlDisag,
        'pbe': expl.PBE,
    }

    def __init__(self, wm, act_space, config):
        self.config = config
        self.rewards = {}
        critics = {}
        for key, scale in config.expl_rewards.items():
            if not scale:
                continue
            else:
                reward = self.REWARDS[key](wm, act_space, config)
                critics[key] = agent.VFunction(reward, config.update(
                    discount=config.expl_discount, retnorm=config.expl_retnorm,
                ))
                self.rewards[key] = reward
        scales = {k: v for k, v in config.expl_rewards.items() if v}
        self.ac = agent.ImagActorCritic(critics, scales, act_space, config)

    def initial(self, batch_size):
        return self.ac.initial(batch_size)

    def policy(self, latent, state):
        outs, carry = self.ac.policy(latent, state)
        outs['action'] = outs['action'].sample()
        return outs, carry

    def train(self, imagine, start, data):
        metrics = {}
        for key, reward in self.rewards.items():
            metrics.update(reward.train(data))
        traj, mets = self.ac.train(imagine, start, data)
        metrics.update(mets)
        return traj, metrics

    def report(self, data):
        return {}


class GoalCond(tfutils.Module):

    def __init__(self, wm, act_space, config):
        self.config = config
        self.wm = wm

        aconfig = config.update({
            'actor': self.config.actor_ach,
            'critic.inputs': self.config.actor_ach.inputs,
            'actent': self.config.actent_ach
        })
        self.achiever = agent.ImagActorCritic({
            'goal': agent.VFunction(lambda s: self.goal_reward(s), aconfig),
        }, {'goal': 1.0}, act_space, aconfig)
        self.dyndist = nets.MLP((), **self.config.dyndist)
        self.dd_opt = tfutils.Optimizer('dyndist', **config.dyndist_opt)
        self.dd_cur_idxs, self.dd_fut_idxs = self.get_future_idxs()

    def initial(self, batch_size):
        return {
            'goal': tf.zeros(
                (batch_size,) + self.config.goal_shape, tf.float32
            ),
        }
    
    def policy(self, latent, carry, mode):
        sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
        goal = carry['goal']
        if mode == 'train':
            act = self.achiever.actor(sg({**latent, 'goal': goal})).sample()
        elif mode == 'eval':
            act = self.achiever.actor(sg({**latent, 'goal': goal})).mode()
        outs = {'action': act}
        carry = {'goal': goal}
        return outs, carry

    def train(self, imagine, start, data):
        start = start.copy()
        goal, raw_goal = self.propose_train_goals(start)
        sg = lambda x: tf.nest.map_structure(tf.stop_gradient, x)
        with tf.GradientTape(persistent=True) as tape:
            achiever = lambda s: self.achiever.actor(
                sg({**s, 'goal': goal,})
            ).sample()
            traj = imagine(achiever, start, self.config.imag_horizon)
            traj['goal'] = tf.repeat(
                goal[None], 1 + self.config.imag_horizon, 0
            )
            traj['raw_goal'] = tf.repeat(
                raw_goal[None], 1 + self.config.imag_horizon, 0
            )
        mets = self.achiever.update(traj, tape)
        with tf.GradientTape() as tape:
            dd_loss = self.dd_loss(sg(self.wm.heads['embed'](traj).mode()))
        mets.update(self.dd_opt(tape, dd_loss, [self.dyndist]))
        return traj, mets
    
    def get_goal(self, is_first, goal, carry):
        ind = tf.range(goal.shape[0] // 2)
        embed = self.wm.encoder({'observation': tf.gather(goal, ind)})
        embed = embed.astype(tf.float32)
        upd = tf.gather(is_first, ind)
        switch = lambda x, y: (
            tf.einsum('i,i...->i...', 1 - upd.astype(x.dtype), x) +
            tf.einsum('i,i...->i...', upd.astype(x.dtype), y)
        )
        carry = {'goal': switch(carry['goal'], embed)}
        return carry
    
    def get_eval_goal(self, is_first, goal, carry):
        embed = self.wm.encoder({'observation': goal}).astype(tf.float32)
        switch = lambda x, y: (
            tf.einsum('i,i...->i...', 1 - is_first.astype(x.dtype), x) +
            tf.einsum('i,i...->i...', is_first.astype(x.dtype), y)
        )
        carry = {'goal': switch(carry['goal'], embed)}
        return carry

    def propose_train_goals(self, start):
        # goals are from observation space
        if self.config.train_goals == 'batch':
            goal = start['embed']
            ids = tf.random.shuffle(tf.range(tf.shape(goal)[0]))
            goal = tf.gather(goal, ids)
            raw_goal = tf.gather(start['observation'], ids)
        elif self.config.train_goals == 'env':
            goal = self.wm.encoder({'observation': start['goal']})
            raw_goal = start['goal']
        return tf.stop_gradient(goal), tf.stop_gradient(raw_goal)
    
    def get_future_idxs(self):
        cur_idx_list = []
        fut_idx_list = []
        for cur_idx in tf.range(self.config.imag_horizon + 1):
            for fut_idx in tf.range(cur_idx, self.config.imag_horizon + 1):
                cur_idx_list.append(cur_idx)
                fut_idx_list.append(fut_idx)
        return tf.concat(cur_idx_list, 0), tf.concat(fut_idx_list, 0)
    
    def dd_loss(self, inp):
        # TODO: Take into account absorbing state — final state
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

    def goal_reward(self, traj):
        if self.config.goal_reward == 'dd':
            embed = self.wm.heads['embed'](traj).mode()
            goal = tf.stop_gradient(traj['goal'].astype(tf.float32))
            reward = - self.dyndist({'embed': embed, 'goal': goal}).mode()[1:]
        elif self.config.goal_reward == 'l1_dif':
            pos = self.wm.heads['decoder'](traj)['observation'].mode()[..., :2]
            goal = tf.stop_gradient(traj['raw_goal'][..., :2])
            manhattan = tf.reduce_sum(tf.math.abs(goal - pos), axis=-1)
            reward = manhattan[:-1] - manhattan[1:]
        return reward

    def report(self, data):
        # TODO: add eval on episodes to show dd
        return {}