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
        
        self.explorer = behaviors.Explore(self.wm, self.act_space, self.config)
        wconfig = config.update({
                'actor.inputs': self.config.worker_inputs,
                'critic.inputs': self.config.worker_inputs,
        })
        self.worker = agent.ImagActorCritic({
            'goal': agent.VFunction(lambda s: self.goal_reward(s), wconfig),
        }, config.worker_rews, act_space, wconfig)

        goal_picker_cls = getattr(goal_picker, config.goal_strategy)
        p_cfg = config.planner
        if config.goal_strategy == "Greedy":
            goal_strategy = goal_picker_cls(replay, agnt.wm, agnt._expl_behavior._intr_reward, config.state_key, config.goal_key, 1000)
        elif config.goal_strategy == 'Random':
            goal_strategy = goal_picker_cls(self.wm, self.act_space, config)
        elif config.goal_strategy == "SampleReplay":
            goal_strategy = goal_picker_cls(agnt.wm, dataset, config.state_key, config.goal_key)
        elif config.goal_strategy == "SubgoalPlanner":
            init_cand = None

            def vis_fn(elite_inds, elite_samples, seq, wm):
                pass

            goal_dataset = None
            env_goals_percentage = p_cfg.init_env_goal_percent
            mega_prior = None
            sample_env_goals_fn = None

            goal_strategy = goal_picker_cls(
                self.wm,
                self.worker.actor,
                self.explorer.ac.critics['expl'].reward_fn,
                value_fn=self.explorer.ac.critics['expl'].target,
                gc_input=config.gc_input,
                goal_dim=2,
                goal_min=np.array(p_cfg.goal_min, dtype=np.float32),
                goal_max=np.array(p_cfg.goal_max, dtype=np.float32),
                act_space=self.act_space,
                state_key=config.state_key,
                planner=p_cfg.planner_type,
                horizon=p_cfg.horizon,
                batch=p_cfg.batch,
                cem_elite_ratio=p_cfg.cem_elite_ratio,
                optimization_steps=p_cfg.optimization_steps,
                std_scale=p_cfg.std_scale,
                mppi_gamma=p_cfg.mppi_gamma,
                init_candidates=init_cand,
                dataset=goal_dataset,
                evaluate_only=p_cfg.evaluate_only,
                repeat_samples=p_cfg.repeat_samples,
                mega_prior=mega_prior,
                sample_env_goals_fn=sample_env_goals_fn,
                env_goals_percentage=env_goals_percentage,
                vis_fn=vis_fn
            )
        elif config.goal_strategy in {"MEGA", "Skewfit"}:
            goal_strategy = goal_picker_cls(agnt, replay, env.act_space, config.state_key, config.time_limit, obs2goal_fn)
        else:
            raise NotImplementedError
        
        self.goal_picker = goal_strategy
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