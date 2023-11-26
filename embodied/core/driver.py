import collections

import numpy as np

from .convert import convert


class Driver:

    _CONVERSION = {
            np.floating: np.float32,
            np.signedinteger: np.int32,
            np.uint8: np.uint8,
            bool: bool,
    }

    def __init__(self, env, **kwargs):
        assert len(env) > 0
        self._env = env
        self._kwargs = kwargs
        self._on_steps = []
        self._on_episodes = []
        self.reset()

    def reset(self):
        self._obs = {
                k: convert(np.zeros((len(self._env),) + v.shape, v.dtype))
                for k, v in self._env.obs_space.items()}
        self._obs['is_last'] = np.ones(len(self._env), bool)
        self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
        self._state = None

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def __call__(self, policy, steps=0, episodes=0):
        step, episode = 0, 0
        while step < steps or episode < episodes:
            step, episode = self._step(policy, step, episode)

    def _step(self, policy, step, episode):
        acts, self._state = policy(self._obs, self._state, **self._kwargs)
        acts['reset'] = np.zeros(len(self._env), bool)
        if self._obs['is_last'].any():
            assert self._obs['is_last'].all()
            acts = {
                    k: v * self._expand(1 - self._obs['is_last'], len(v.shape))
                    for k, v in acts.items()}
            acts['reset'] = self._obs['is_last']
        acts = {k: convert(v) for k, v in acts.items()}
        assert all(len(x) == len(self._env) for x in acts.values()), acts
        self._obs = self._env.step(acts)
        assert all(len(x) == len(self._env) for x in self._obs.values()), self._obs
        # if self._state is not None:
        #     self._obs['real_goal'] = self._state[2]['real_goal']
        # else:
        #     import tensorflow as tf
        #     self._obs['real_goal'] = tf.zeros((4, 2))

        self._obs = {k: convert(v) for k, v in self._obs.items()}
        trns = {**self._obs, **acts}
        if self._obs['is_first'].any():
            for i, first in enumerate(self._obs['is_first']):
                if not first:
                    continue
                self._eps[i].clear()
        for i in range(len(self._env)):
            trn = {k: v[i] for k, v in trns.items()}
            [self._eps[i][k].append(v) for k, v in trn.items()]
            [fn(trn, i, **self._kwargs) for fn in self._on_steps]
            step += 1
        if self._obs['is_last'].any():
            for i, done in enumerate(self._obs['is_last']):
                if not done:
                    continue
                ep = {k: convert(v) for k, v in self._eps[i].items()}
                [fn(ep.copy(), i, **self._kwargs) for fn in self._on_episodes]
                episode += 1
        return step, episode

    def _expand(self, value, dims):
        while len(value.shape) < dims:
            value = value[..., None]
        return value
    
def eval_episode_render_fn(env, ep, goal, clear: bool = False):
    env_time_limit = 0
    env.step({'action': [0], 'reset': [1]})

    if clear:
        env.clear_obs()
    
    image_trajectory = []
    goal_images = []
    executions = []

    env.set_state([np.array(goal)])
    goal_img = env.render()[0]
    goal_images.append(goal_img[None])
    for state in ep['observation']:
        env.set_state([np.array(state)])

        # Pay attention how PEG authors return images in run_goal_cond.
        img = env.render()[0]
        image_trajectory.append(img)
        env_time_limit += 1

    image_trajectory = np.stack(image_trajectory, 0)
    T = image_trajectory.shape[0]
    image_trajectory = np.pad(image_trajectory, ((0, (env_time_limit) - T), (0,0), (0,0), (0,0)), 'constant', constant_values=(0))
    executions.append(image_trajectory[None]) # 1 x T x H x W x C
    return goal_images, executions

class EvalDriver:

    _CONVERSION = {
            np.floating: np.float32,
            np.signedinteger: np.int32,
            np.uint8: np.uint8,
            bool: bool,
    }

    def __init__(self, env, logger, drawer, **kwargs):
        assert len(env) > 0
        assert len(env) == 1 # TODO: make the driver support multiple envs
        self._env = env
        self.drawer = drawer
        self.logger = logger
        # self.goals = kwargs['goals']
        self.goals = [[9, 9], [6, 9], [0, 9], [9, 6], [9, 0], [6, 6], [6, 3], [3, 6], [3, 3], [0, 3], [3, 0]]
        # self.threshold = kwargs['threshold']
        self.threshold = 0.7
        self._kwargs = kwargs
        self._on_steps = []
        self._on_episodes = []
        self.reset()

    def reset(self):
        self._obs = {
                k: convert(np.zeros((len(self._env),) + v.shape, v.dtype))
                for k, v in self._env.obs_space.items()}
        self._obs['is_last'] = np.ones(len(self._env), bool)
        self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
        self._state = None

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def __call__(self, policy, repeat=10):
        metrics = [{'dist': [], 'success': []} for _ in range(len(self.goals))]
        goal_images = []
        execution_image_trajectories = []
        for i, goal in enumerate(self.goals):
            for j in range(repeat):
                create_video = j == 0
                self._step(policy, goal)
                while not self._obs['is_last'][0]:
                    # self._env.render() TODO: collect renders during policy execition
                    self._step(policy, goal)
                ep = self._eps[0]
                dist = np.linalg.norm(np.array(ep['observation']) - np.array(goal), axis=1).min()
                success = dist < self.threshold
                metrics[i]['dist'].append(dist)
                metrics[i]['success'].append(success)
                if not create_video:
                    continue
                _goals, _executions = eval_episode_render_fn(self._env, ep, goal)

                goal_images.extend(_goals)
                execution_image_trajectories.extend(_executions)

        if True:
            execution_image_trajectories = np.concatenate(execution_image_trajectories, 0) # num_goals x T x H x W x C
            print(execution_image_trajectories.shape)
            goal_images = np.stack(goal_images, 0) # num_goals x 1 x H x W x C
            print(goal_images.shape)
            goal_images = np.repeat(goal_images, execution_image_trajectories.shape[1], 1)
            gc_video = np.concatenate([goal_images, execution_image_trajectories], -3)
            self.logger.video(f'eval_gc_policy', gc_video)
            
        for i, met in enumerate(metrics):
            self.logger.scalar(f'dist/goal_{i}', np.mean(met['dist']))
            self.logger.scalar(f'success/goal_{i}', np.mean(met['success']))
        self.logger.scalar(f'dist/goal_all', np.mean([np.mean(met['dist']) for met in metrics]))
        self.logger.scalar(f'success/goal_all', np.mean([np.mean(met['success']) for met in metrics]))
        for name, image in self.drawer.draw().items():
            self.logger.image(name, image)


    def _step(self, policy, goal, step=0, episode=0):
        self._obs['loag'] = np.ones((len(self._env), 2))
        self._obs['loag'][0] = np.array(goal)
        acts, self._state = policy(self._obs, self._state, **self._kwargs)
        acts['reset'] = np.zeros(len(self._env), bool)
        if self._obs['is_last'].any():
            acts = {
                    k: v * self._expand(1 - self._obs['is_last'], len(v.shape))
                    for k, v in acts.items()}
            acts['reset'] = self._obs['is_last']
        acts = {k: convert(v) for k, v in acts.items()}
        assert all(len(x) == len(self._env) for x in acts.values()), acts
        self._obs = self._env.step(acts)
        assert all(len(x) == len(self._env) for x in self._obs.values()), self._obs
        self._obs = {k: convert(v) for k, v in self._obs.items()}
        trns = {**self._obs, **acts}
        if self._obs['is_first'].any():
            for i, first in enumerate(self._obs['is_first']):
                if not first:
                    continue
                self._eps[i].clear()
        for i in range(len(self._env)):
            trn = {k: v[i] for k, v in trns.items()}
            [self._eps[i][k].append(v) for k, v in trn.items()]
            [fn(trn, i, **self._kwargs) for fn in self._on_steps]
            step += 1
        if self._obs['is_last'].any():
            for i, done in enumerate(self._obs['is_last']):
                if not done:
                    continue
                ep = {k: convert(v) for k, v in self._eps[i].items()}
                [fn(ep.copy(), i, **self._kwargs) for fn in self._on_episodes]
                episode += 1
        return step, episode

    def _expand(self, value, dims):
        while len(value.shape) < dims:
            value = value[..., None]
        return value
