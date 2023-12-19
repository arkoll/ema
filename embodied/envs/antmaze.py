import gymnasium as gym
import embodied

import numpy as np


class AntMaze(embodied.Env):

    def __init__(self, task, mode, repeat=1, length=500, resets=True):
        a = 'r'
        g = 'g'
        maze_map = [
            [0, 0, 1, 1, 1],
            [0, 0, 1, g, 1],
            [1, 1, 1, 0, 1],
            [1, a, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
        self._env = gym.make(
            'AntMaze_UMaze-v4', maze_map=maze_map, max_episode_steps=length,
            continuing_task=mode=='train', reset_target=mode=='train'
        )
        self._done = True
        self._other_dims = np.concatenate([
            [
                6.08193526e-01,  9.87496030e-01, 1.82685311e-03,
                -6.82827458e-03,  1.57485326e-01,  5.14617396e-02,
                1.22386603e+00, -6.58701813e-02, -1.06980319e+00,
                5.09069276e-01, -1.15506861e+00,  5.25953435e-01,
                7.11716520e-01
            ], np.zeros(14)
        ])

    @property
    def obs_space(self):
        return {
            'reward': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
            'absolute_position': self._convert(
                self._env.observation_space['achieved_goal']
            ),
            'desired_observation': self._convert(
                self._env.observation_space['observation']
            ),
            **{
                k: self._convert(v) 
                for k, v in self._env.observation_space.items()
            }
        }

    @property
    def act_space(self):
        return {
            'action': self._convert(self._env.action_space),
            'reset': embodied.Space(bool),
        }

    def step(self, action):  
        if action['reset'] or self._done:
            self._done = False
            obs, info = self._env.reset()
            return self._obs(obs, 0.0, info, is_first=True)
        obs, rew, term, trunc, info = self._env.step(action['action'])
        self._done = term or trunc
        return self._obs(obs, rew, info, is_last=self._done, is_terminal=term)
    
    def _obs(
        self, obs, reward, info, is_first=False, is_last=False,
        is_terminal=False
    ):
        return dict(
            reward=reward,
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
            absolute_position=obs['achieved_goal'],
            desired_observation=self._other_dims,
            **obs,
        )
    
    def _convert(self, space):
        return embodied.Space(
            space.dtype, space.shape, space.low, space.high
        )
    