import gym
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from .mazes import mazes_dict
import embodied

class PointMaze2D(embodied.Env):
    def __init__(self, test=False, length=0):
        self.maze = mazes_dict['square_large']['maze']
        self.dist_threshold = 0.15

        self.s_xy = np.array(self.maze.sample_start())
        self.g_xy = np.array(self.maze.sample_goal(
            min_wall_dist=0.025 + self.dist_threshold
        ))
        self.test = test
        self.background = None
        self.cur_step = 0
        self.length = length

    @property
    def obs_space(self):
        ob_space = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
        goal_space = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
        return {
            'observation': ob_space,
            'goal': goal_space,
            'reward': gym.spaces.Box(-np.inf, np.inf, (), dtype=np.float32),
            'is_first': gym.spaces.Box(0, 1, (), dtype=np.bool_),
            'is_last': gym.spaces.Box(0, 1, (), dtype=np.bool_),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=np.bool_),
        }

    @property
    def act_space(self):
        # return {'action': gym.spaces.Box(-0.95, 0.95, (2,), dtype=np.float32)}
        return {
                'action': embodied.Space(np.float32, None, np.array([-0.95, -0.95]), np.array([0.95, 0.95])),
                'reset': embodied.Space(bool),
        }

    def seed(self, seed=None):
        return self.maze.seed(seed=seed)

    def step(self, action):
        if action['reset']:
            return self.reset()
        action = action['action']
        # print(action)
        try:
            s_xy = np.array(self.maze.move(tuple(self.s_xy), tuple(action)))
        except:
            print('failed to move', tuple(self.s_xy), tuple(action))
            raise

        self.s_xy = s_xy
        reward = self.compute_reward(s_xy, self.g_xy)

        if self.test:
            done = np.allclose(0., reward)
        else:
            done = False

        if self.length and self.cur_step > self.length:
            truncated = True
        self.cur_step += 1

        return {
            'observation': s_xy,
            # 'goal': self.g_xy,
            'reward': 0.0,
            'is_first': False,
            'is_last': done and truncated,
            'is_terminal': done,
        }

    def reset(self):
        s_xy = np.array(self.maze.sample_start())
        self.s_xy = s_xy
        g_xy = np.array(self.maze.sample_goal(
            min_wall_dist=0.025 + self.dist_threshold
        ))
        self.cur_step = 0
        self.g_xy = g_xy
        return {
            'observation': s_xy,
            # 'goal': self.g_xy,
            'reward': 0.0,
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
        }

    def render(self):
        if self.background is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(1, 1))
            self.maze.plot(self.ax)  # plot the walls
            self.ax.axis('off')
            self.fig.tight_layout(pad=0)
            self.fig.canvas.draw()
            self.background = self.fig.canvas.copy_from_bbox(self.ax.bbox)
            self.scatter = self.ax.scatter([], [], s=10, color='red')
            self.goal_scatter = self.ax.scatter(
                [], [], s=20, color='b', marker='*'
            )

        self.fig.canvas.restore_region(self.background)
        self.scatter.set_offsets(self.s_xy)
        # self.goal_scatter.set_offsets(self.g_xy)
        self.ax.draw_artist(self.scatter)
        # self.ax.draw_artist(self.goal_scatter)
        self.fig.canvas.blit(self.ax.bbox)
        image_from_plot = np.frombuffer(
            self.fig.canvas.tostring_rgb(), dtype=np.uint8
        )
        image_from_plot = image_from_plot.reshape(
            self.fig.canvas.get_width_height()[::-1] + (3,)
        )
        return image_from_plot
    
    def set_state(self, state):
        self.s_xy = state

    def clear_plots(self):
        plt.clf()
        plt.cla()
        plt.close(self.fig)
        self.background = None

    def compute_reward(self, achieved_goal, desired_goal):
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(d >= self.dist_threshold).astype(np.float32)
    
    def render_ep(self, episode):
        g_xy, s_xy = self.g_xy.copy(), self.s_xy.copy()
        all_img = []
        for goal_xy, xy in zip(episode['goal'], episode['observation']):
            self.g_xy = goal_xy
            self.s_xy = xy
            img = self.render()
            all_img.append(img)
        self.clear_plots()
        all_img = np.stack(all_img, 0)
        self.g_xy, self.s_xy = g_xy, s_xy
        return all_img.astype(np.float32) / 255.0
    
    def render_predictions(self, model, truth, goal):
        model_imgs = self.render_ep({
            'goal': goal.numpy().reshape(-1, 2),
            'observation': model.numpy().reshape(-1, 2)
        })
        truth_imgs = self.render_ep({
            'goal': goal.numpy().reshape(-1, 2),
            'observation': truth.numpy().reshape(-1, 2)
        })

        sh = [*goal.shape[:2], *model_imgs.shape[-3:]]
        model_imgs = model_imgs.reshape(sh)
        truth_imgs = truth_imgs.reshape(sh)

        error = (model_imgs - truth_imgs + 1) / 2
        video = np.concatenate([truth_imgs, model_imgs, error], 2)
        B, T, H, W, C = video.shape
        return {
            'openl_image': video.transpose((1, 2, 0, 3, 4)).reshape((
                T, H, B * W, C
            ))
        }
    
    def add_states_to_buffer(self, buffer, states):
        for s in states:
            if 'x' in buffer:
                buffer['x'].append(s[0])
            else:
                buffer['x'] = [s[0]]

            if 'y' in buffer:
                buffer['y'].append(s[1])
            else:
                buffer['y'] = [s[1]]

    def render_state_goals(self, states, goals, g_alpha=1):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=500)
        self.maze.plot(ax)
        ax.axis('off')
        fig.tight_layout(pad=0)
        if 'x' in states and 'y' in states:
            ax.scatter(states['x'], states['y'], s=1, color='red')
        if 'x' in goals and 'y' in goals:
            ax.scatter(goals['x'], goals['y'], s=1, color='blue', alpha=g_alpha)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img.astype(float) / 255.
    
    def render_trajectories(self, trajectories, alpha):
        # trajs x samples x steps x X x Y
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=500)
        ax.axis('off')
        fig.tight_layout(pad=0)
        cmap = mpl.colormaps['jet']

        trajs = trajectories.shape[0]
        for ind, trajectory in enumerate(trajectories):
            for sample in trajectory:
                ax.plot(
                    sample[:, 0], sample[:, 1], color=cmap(ind / trajs),
                    alpha=alpha[ind]
                )
        for x, y in self.maze._walls:
            ax.plot(x, y, 'k-')
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img.astype(float) / 255.


class MultiGoalPointMaze2D(PointMaze2D):
    def __init__(self, test=False):
        super().__init__(test)
        self.maze = mazes_dict['multigoal_square_large']['maze']
        self.goal_idx = 0

    def reset(self):
        s_xy = np.array(self.maze.sample_start())
        self.s_xy = s_xy
        if self.test:  # use set goal_idx.
            g_xy = np.array(self.maze.sample_goal(
                min_wall_dist=0.025 + self.dist_threshold,
                goal_idx=self.goal_idx
            ))
        else:  # sample any goal.
            g_xy = np.array(self.maze.sample_goal(
                min_wall_dist=0.025 + self.dist_threshold
            ))
        self.g_xy = g_xy
        return {
            'observation': s_xy,
            'goal': self.g_xy,
            'reward': 0.0,
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
            'metric_success': 0.0,
            'metric_dist': np.linalg.norm(s_xy - self.g_xy)
        }

    def step(self, action):
        action = action['action']
        try:
            s_xy = np.array(self.maze.move(tuple(self.s_xy), tuple(action)))
        except:
            print('failed to move', tuple(self.s_xy), tuple(action))
            raise

        self.s_xy = s_xy
        reward = self.compute_reward(s_xy, self.g_xy)

        if self.test:
            done = np.allclose(0., reward)
        else:
            done = False

        return {
            'observation': s_xy,
            'goal': self.g_xy,
            'reward': reward,
            'is_first': False,
            'is_last': done,
            'is_terminal': done,
            'metric_success': reward + 1,
            'metric_dist': np.linalg.norm(s_xy - self.g_xy)
        }

    def get_goal_idx(self):
        return self.goal_idx

    def set_goal_idx(self, idx):
        self.goal_idx = idx

    def get_goals(self):
        return self.maze.goal_squares

    def compute_reward(self, achieved_goal, desired_goal, dist_threshold=None):
        if dist_threshold is None:
            dist_threshold = self.dist_threshold
        d = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
        return -(d >= dist_threshold).astype(np.float32)