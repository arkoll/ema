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
        self.wm = wm
        self.config = config
        self.act_space = act_space
        self.goal_shape = (self.wm.rssm._deter, )

    def initial(self, batch_size):
        return tf.zeros(batch_size)

    def policy(self, latent, state):
        batch_size = len(state['step'])
        shape = (batch_size, ) + self.goal_shape
        return tf.zeros(shape)

    def train(self, imagine, start, data):
        return None, {}

    def report(self, data):
        return {}
    
class SubgoalPlanner:
    def __init__(
            self,
            wm,
            actor,
            reward_fn,
            gc_input,
            goal_dim, # D dims
            goal_min, #    D dims for min / max
            goal_max, #    D dims for min / max
            act_space,
            state_key,
            planner="shooting_cem",
            horizon=15,
            mpc_steps=10,
            batch=5,
            cem_elite_ratio=0.2,
            optimization_steps=5,
            std_scale=1.0,
            mppi_gamma=10.0,
            init_candidates=None,
            dataset=None,
            evaluate_only=False, #    don't run CEM, just evaluate goals with model.
            repeat_samples=0,
            mega_prior=False, # an instance of MEGA
            sample_env_goals_fn=None,
            env_goals_percentage=None,
            vis_fn=None,
        ):
        self.wm = wm
        self.dtype = wm.dtype
        self.actor = actor
        self.reward_fn = reward_fn
        self.gc_input = gc_input
        self.obs2goal = None
        self.goal_dim = goal_dim
        self.act_space = act_space
        if isinstance(act_space, dict):
            self.act_space = act_space['action']
        self.state_key = state_key
        self.planner = planner
        self.horizon = horizon
        self.mpc_steps = mpc_steps
        self.batch = batch
        self.cem_elite_ratio = cem_elite_ratio
        self.optimization_steps = optimization_steps
        self.std_scale = std_scale
        self.mppi_gamma= mppi_gamma
        self.env_goals_percentage = env_goals_percentage
        self.sample_env_goals = env_goals_percentage > 0
        self.sample_env_goals_fn = sample_env_goals_fn

        self.min_action = goal_min
        self.max_action = goal_max

        self.mega = mega_prior
        self.init_distribution = None
        if init_candidates is not None:
            self.create_init_distribution(init_candidates)

        self.dataset = dataset
        self.evaluate_only = evaluate_only
        if self.evaluate_only:
            assert self.dataset is not None, "need to sample from replay buffer."

        self.repeat_samples = repeat_samples
        self.vis_fn = vis_fn
        self.will_update_next_call = True
        self.mega_sample = None


    def search_goal(self, obs, state=None, mode='train'):
        if self.will_update_next_call is False:
            return self.sample_goal()

        elite_size = int(self.batch * self.cem_elite_ratio)
        if state is None:
            latent = self.wm.rssm.initial(1)
            action = tf.zeros((1,1,) + self.act_space.shape)
            state = latent, action
            # print("make new state")
        else:
            latent, action = state
            action = tf.expand_dims(action, 0)
            # action should be (1,1, D)
            # print("using exisitng state")


        # create start state.
        embed = self.wm.encoder(obs)
        # posterior is q(s' | s,a,e)
        post, prior = self.wm.rssm.observe(
                embed, action, obs['is_first'], latent)
        init_start = {k: v[:, -1] for k, v in post.items()}
        # print(action.shape)
        # for k,v in latent.keys():
        #     print(k, v.shape)
        @tf.function
        def eval_fitness(goal):
            # should be (55,128).
            start = {k: v for k, v in init_start.items()}
            start['feat'] = self.wm.rssm.get_feat(start) # (1, 1800)
            start = tf.nest.map_structure(lambda x: tf.repeat(x, goal.shape[0],0), start)
            if self.gc_input == "embed":
                goal_obs = start.copy()
                goal_obs[self.state_key] = goal
                goal_input = self.wm.encoder(goal_obs)
            elif self.gc_input == "state":
                goal_input = tf.cast(goal, self.dtype)

            actor_inp = tf.concat([start['feat'], goal_input], -1)
            start['action'] = tf.zeros_like(self.actor(actor_inp).mode())
            seq = {k: [v] for k, v in start.items()}
            for _ in range(self.horizon):
                actor_inp = tf.concat([seq['feat'][-1], goal_input], -1)
                action = self.actor(actor_inp).sample()
                state = self.wm.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
                feat = self.wm.rssm.get_feat(state)
                for key, value in {**state, 'action': action, 'feat': feat}.items():
                    seq[key].append(value)
            seq = {k: tf.stack(v, 0) for k, v in seq.items()}
            # rewards should be (batch,1)
            rewards = self.reward_fn(seq)
            returns = tf.reduce_sum(rewards, 0)
            # rewards = tf.ones([goal.shape[0],])
            return returns, seq

        # CEM loop
        # rewards = []
        # act_losses = []
        if self.init_distribution is None:
            # print("getting init distribtion from obs")
            means, stds = self.get_distribution_from_obs(obs)
        else:
            # print("getting init distribtion from init candidates")
            means, stds = self.init_distribution
        # print(means, stds)
        opt_steps = 1 if self.evaluate_only else self.optimization_steps
        for i in range(opt_steps):
            # Sample action sequences and evaluate fitness
            if i == 0 and (self.dataset or self.mega or self.sample_env_goals):
                if self.dataset:
                    # print("getting init distribution from dataset")
                    random_batch = next(self.dataset)
                    random_batch = self.wm.preprocess(random_batch)
                    samples = tf.reshape(random_batch[self.state_key], (-1,) + tuple(random_batch[self.state_key].shape[2:]))
                    if self.obs2goal is not None:
                        samples = self.obs2goal(samples)
                elif self.sample_env_goals:
                    num_cem_samples = int(self.batch * self.env_goals_percentage)
                    num_env_samples = self.batch - num_cem_samples
                    cem_samples = tfd.MultivariateNormalDiag(means, stds).sample(sample_shape=[num_cem_samples])
                    env_samples = self.sample_env_goals_fn(num_env_samples)
                    samples = tf.concat([cem_samples, env_samples], 0)

                elif self.mega:
                    # print("getting init distribution from MEGA")
                    samples = self.mega.sample_goal(obs)[None]
                    self.mega_sample = samples
                    # since mega only returns 1 goal, repeat it.
                    samples = tf.repeat(samples, self.batch, 0)
                # initialize means states.
                means, vars = tf.nn.moments(samples, 0)
                # stds = tf.sqrt(vars + 1e-6)
                # stds = tf.concat([[0.5, 0.5], stds[2:]], axis=0)
                # assert np.prod(means.shape) == self.goal_dim, f"{np.prod(means.shape)}, {self.goal_dim}"
                samples = tfd.MultivariateNormalDiag(means, stds).sample(sample_shape=[self.batch])
                # print(i, samples)
                samples = tf.clip_by_value(samples, self.min_action, self.max_action)
            else:
                samples = tfd.MultivariateNormalDiag(means, stds).sample(sample_shape=[self.batch])
                samples = tf.clip_by_value(samples, self.min_action, self.max_action)

            if self.repeat_samples > 1:
                repeat_samples = tf.repeat(samples, self.repeat_samples, 0)
                repeat_fitness, seq = eval_fitness(repeat_samples)
                fitness = tf.reduce_mean(tf.stack(tf.split(repeat_fitness, self.repeat_samples)), 0)
            else:
                fitness, seq = eval_fitness(samples)
            # Refit distribution to elite samples
            if self.planner == 'shooting_mppi':
                # MPPI
                weights = tf.expand_dims(tf.nn.softmax(self.mppi_gamma * fitness), axis=1)
                means = tf.reduce_sum(weights * samples, axis=0)
                stds = tf.sqrt(tf.reduce_sum(weights * tf.square(samples - means), axis=0))
                # rewards.append(tf.reduce_sum(fitness * weights[:, 0]).numpy())
            elif self.planner == 'shooting_cem':
                # CEM
                elite_score, elite_inds = tf.nn.top_k(fitness, elite_size, sorted=False)
                elite_samples = tf.gather(samples, elite_inds)
                # print(elite_samples)
                means, vars = tf.nn.moments(elite_samples, 0)
                stds = tf.sqrt(vars + 1e-6)
                # rewards.append(tf.reduce_mean(tf.gather(fitness, elite_inds)).numpy())

        if self.planner == 'shooting_cem':
            self.vis_fn(elite_inds, elite_samples, seq, self.wm)
            self.elite_inds = elite_inds
            self.elite_samples = elite_samples
            self.final_seq = seq
        elif self.planner == 'shooting_mppi':
            # print("mppi mean", means)
            # print("mppi std", stds)
            # TODO: figure out what elite inds means for shooting mppi.
            # self.vis_fn(elite_inds, elite_samples, seq, self.wm)
            self.elite_inds = None
            self.elite_samples = None
            self.final_seq = seq
        # TODO: potentially store these as initialization for the next update.
        self.means = means
        self.stds = stds

        if self.evaluate_only:
            self.elite_samples = elite_samples
            self.elite_score = elite_score

        return self.sample_goal()

    def sample_goal(self, batch=1):
        if self.evaluate_only:
            # samples = tfd.MultivariateNormalDiag(self.means, self.stds).sample(sample_shape=[batch])
            # weights = tf.nn.softmax(self.elite_score)
            weights = self.elite_score / self.elite_score.sum()
            idxs = tf.squeeze(tf.random.categorical(tf.math.log([weights]), batch), 0)
            samples = tf.gather(self.elite_samples, idxs)
        else:
            samples = tfd.MultivariateNormalDiag(self.means, self.stds).sample(sample_shape=[batch])
        return samples

    def create_init_distribution(self, init_candidates):
        """Create the starting distribution for seeding the planner.
        """
        def _create_init_distribution(init_candidates):
            means = tf.reduce_mean(init_candidates, 0)
            stds = tf.math.reduce_std(init_candidates, 0)
            # if there's only 1 candidate, set std to default
            if init_candidates.shape[0] == 1:
                stds = tf.ones(self.goal_dim, dtype=tf.float32) * self.std_scale
            return means, stds
        self.init_distribution = _create_init_distribution(init_candidates)

    def get_distribution_from_obs(self, obs):
        ob = tf.squeeze(obs[self.state_key])
        if self.gc_input == "state":
            ob = self.obs2goal(ob)
        means = tf.cast(tf.identity(ob), tf.float32)
        assert np.prod(means.shape) == self.goal_dim, f"{np.prod(means.shape)}, {self.goal_dim}"
        stds = tf.ones(self.goal_dim, dtype=tf.float32) * self.std_scale
        init_distribution = tf.identity(means), tf.identity(stds)
        return init_distribution

    def get_init_distribution(self):
        if self.init_distribution is None:
            means = tf.zeros(self.goal_dim, dtype=tf.float32)
            stds = tf.ones(self.goal_dim, dtype=tf.float32) * self.std_scale
            self.init_distribution = tf.identity(means), tf.identity(stds)

        return self.init_distribution
