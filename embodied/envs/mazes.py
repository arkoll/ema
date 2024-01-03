from os import path
from typing import Optional

import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from gymnasium.utils.ezpickle import EzPickle

from gymnasium_robotics.envs.maze.point import PointEnv
from gymnasium_robotics.utils.mujoco_utils import MujocoModelNames
from gymnasium_robotics.envs.maze.point_maze import PointMazeEnv
from gymnasium_robotics.envs.maze.ant_maze_v4 import AntMazeEnv

import embodied


class CustomPointMaze(PointMazeEnv):

    def __init__(self,
        render_mode: Optional[str] = None,
        reward_type: str = "sparse",
        continuing_task: bool = True,
        reset_target: bool = False,
        **kwargs,
    ):
        point_xml_file_path = path.join(
            path.dirname(path.realpath(__file__)), "assets/point.xml"
        )
        a = 'r'
        g = 'g'
        maze_map = [
            [0, 0, 1, 1, 1],
            [0, 0, 1, g, 1],
            [1, 1, 1, 0, 1],
            [1, a, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]

        super(PointMazeEnv, self).__init__(
            agent_xml_path=point_xml_file_path,
            maze_map=maze_map,
            maze_size_scaling=1,
            maze_height=1,
            reward_type=reward_type,
            continuing_task=continuing_task,
            reset_target=reset_target,
            **kwargs,
        )

        maze_length = len(maze_map)
        default_camera_config = {"distance": 12.5 if maze_length > 8 else 8.8}

        self.point_env = PointEnv(
            xml_file=self.tmp_xml_file_path,
            render_mode=render_mode,
            default_camera_config=default_camera_config,
            **kwargs,
        )
        self._model_names = MujocoModelNames(self.point_env.model)
        self.target_site_id = self._model_names.site_name2id["target"]

        self.action_space = self.point_env.action_space
        obs_shape: tuple = self.point_env.observation_space.shape
        self.observation_space = spaces.Dict(
            dict(
                observation=spaces.Box(-np.inf, np.inf, obs_shape, "float64"),
                achieved_goal=spaces.Box(-np.inf, np.inf, (2,), "float64"),
                desired_goal=spaces.Box(-np.inf, np.inf, (2,), "float64"),
            )
        )

        self.render_mode = render_mode

        EzPickle.__init__(
            self,
            render_mode,
            reward_type,
            continuing_task,
            reset_target,
            **kwargs,
        )

    def convert_observation(self, obs):
        return dict(
            observation = obs['observation'].copy(),
            goal = np.concatenate([obs['desired_goal'], np.zeros(2)]),
            obs_pos = obs['achieved_goal'].copy(),
            goal_pos = obs['desired_goal'].copy(),
        )
    
    def convert_obs_space(self):
        obs_shape = self.point_env.observation_space.shape
        return spaces.Dict(dict(
            observation = spaces.Box(-np.inf, np.inf, obs_shape, "float64"),
            goal = spaces.Box(-np.inf, np.inf, obs_shape, "float64"),
            obs_pos = spaces.Box(-np.inf, np.inf, (2,), "float64"),
            goal_pos = spaces.Box(-np.inf, np.inf, (2,), "float64"),
        ))
    

class CustomAntMaze(AntMazeEnv):
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        reward_type: str = "sparse",
        continuing_task: bool = True,
        reset_target: bool = False,
        **kwargs,
    ):
        ant_xml_file_path = path.join(
            path.dirname(path.realpath(__file__)), "assets/ant.xml"
        )
        a = 'r'
        g = 'g'
        maze_map = [
            [0, 0, 1, 1, 1],
            [0, 0, 1, g, 1],
            [1, 1, 1, 0, 1],
            [1, a, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]

        super(AntMazeEnv, self).__init__(
            agent_xml_path=ant_xml_file_path,
            maze_map=maze_map,
            maze_size_scaling=4,
            maze_height=4,
            reward_type=reward_type,
            continuing_task=continuing_task,
            reset_target=reset_target,
            **kwargs,
        )
        
        self.ant_env = AntEnv(
            xml_file=self.tmp_xml_file_path,
            exclude_current_positions_from_observation=False,
            render_mode=render_mode,
            reset_noise_scale=0.0,
            **kwargs,
        )
        self._model_names = MujocoModelNames(self.ant_env.model)
        self.target_site_id = self._model_names.site_name2id["target"]

        self.action_space = self.ant_env.action_space
        obs_shape: tuple = self.ant_env.observation_space.shape
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(
                -np.inf, np.inf, (obs_shape[0] - 2,), "float64"
            ),
            achieved_goal=spaces.Box(-np.inf, np.inf, (2,), "float64"),
            desired_goal=spaces.Box(-np.inf, np.inf, (2,), "float64"),
        ))

        self.render_mode = render_mode
        EzPickle.__init__(
            self,
            render_mode,
            reward_type,
            continuing_task,
            reset_target,
            **kwargs,
        )

    def convert_observation(self, obs):
        other_dims = np.concatenate([
            [
                6.08193526e-01,  9.87496030e-01, 1.82685311e-03,
                -6.82827458e-03,  1.57485326e-01,  5.14617396e-02,
                1.22386603e+00, -6.58701813e-02, -1.06980319e+00,
                5.09069276e-01, -1.15506861e+00,  5.25953435e-01,
                7.11716520e-01
            ], np.zeros(14)
        ])
        return dict(
            observation = np.concatenate(
                [obs['achieved_goal'], obs['observation']]
            ),
            goal = np.concatenate([obs['desired_goal'], other_dims]),
            obs_pos = obs['achieved_goal'].copy(),
            goal_pos = obs['desired_goal'].copy(),
        )
    
    def convert_obs_space(self):
        obs_shape = self.ant_env.observation_space.shape
        return spaces.Dict(dict(
            observation = spaces.Box(-np.inf, np.inf, obs_shape, "float64"),
            goal = spaces.Box(-np.inf, np.inf, obs_shape, "float64"),
            obs_pos = spaces.Box(-np.inf, np.inf, (2,), "float64"),
            goal_pos = spaces.Box(-np.inf, np.inf, (2,), "float64"),
        ))


class Maze(embodied.Env):
    
    def __init__(self, task, mode, length=500):

        if task == 'point':
            self._env = CustomPointMaze(
                continuing_task=mode=='train', reset_target=mode=='train'
            )
        elif task == 'ant':
            self._env = CustomAntMaze(
                continuing_task=mode=='train', reset_target=mode=='train'
            )
        else:
            raise NotImplementedError
        self._done = True

    @property
    def obs_space(self):
        return {
            'reward': embodied.Space(np.float32),
            'is_first': embodied.Space(bool),
            'is_last': embodied.Space(bool),
            'is_terminal': embodied.Space(bool),
            **{
                k: self._convert(v)
                for k, v in self._env.convert_obs_space().items()
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
        new_obs = self._env.convert_observation(obs)
        return dict(
            reward=reward,
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
            **new_obs
        )
    
    def _convert(self, space):
        return embodied.Space(
            space.dtype, space.shape, space.low, space.high
        )
