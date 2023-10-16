import collections
import re
from typing import Any
import warnings

import embodied
import numpy as np
import matplotlib.pyplot as plt

class BufferDrawer():
    def __init__(self) -> None:
        self.buffer = {'states': {'x': [], 'y': []},
                       'estates': {'x': [], 'y': []},
                       'goals': {'x': [], 'y': []}}
        self.ax = plt.subplot(1, 1, 1) 

    def add_traj(self, trajs):
        for traj in trajs:
            self.buffer['goals']['x'].append(traj['real_goal'][0][0])
            self.buffer['goals']['y'].append(traj['real_goal'][0][1])
            for xy in traj['observation'][:25]:
                self.buffer['states']['x'].append(xy[0])
                self.buffer['states']['y'].append(xy[1])  
            for xy in traj['observation'][25:]:
                self.buffer['estates']['x'].append(xy[0])
                self.buffer['estates']['y'].append(xy[1])        

    def draw(self):
        self.fig, self.ax = plt.subplots(1, 1)
        self.ax.scatter(self.buffer['states']['x'], self.buffer['states']['y'], s=0.1, color='green')
        self.ax.scatter(self.buffer['estates']['x'], self.buffer['estates']['y'], s=0.1, color='red')
        self.ax.scatter(self.buffer['goals']['x'], self.buffer['goals']['y'], s=0.1, color='blue')
        fig = self.ax.figure
        fig.canvas.draw()
        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        for k in self.buffer:
            for k2 in self.buffer[k]:
                self.buffer[k][k2] = []
        return data[None, ...]
        

def train_gceval(agent, env, eval_env, replay, logger, args):

    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print('Logdir', logdir)
    should_train = embodied.when.Every(args.train_every)
    should_log = embodied.when.Every(args.log_every)
    should_expl = embodied.when.Until(args.expl_until)
    should_video = embodied.when.Every(args.eval_every)
    step = logger.step

    timer = embodied.Timer()
    timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
    timer.wrap('env', env, ['step'])
    if hasattr(replay, '_sample'):
        timer.wrap('replay', replay, ['_sample'])

    nonzeros = set()
    buffer_drawer = BufferDrawer()
    def per_episode(ep):
        metrics = {}
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        print(f'Episode has {length} steps and return {score:.1f}.')
        metrics['length'] = length
        metrics['score'] = score
        metrics['reward_rate'] = (ep['reward'] - ep['reward'].min() >= 0.1).mean()
        logs = {}
        for key, value in ep.items():
            if not args.log_zeros and key not in nonzeros and (value == 0).all():
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                logs[f'sum_{key}'] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                logs[f'mean_{key}'] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                logs[f'max_{key}'] = ep[key].max(0).mean()
        # if should_video(step):
        #     for key in args.log_keys_video:
        #         metrics[f'policy_{key}'] = ep[key]
        buffer_drawer.add_traj([ep])
        logger.add(metrics, prefix='episode')
        logger.add(logs, prefix='logs')
        logger.add(replay.stats, prefix='replay')
        logger.write()

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(replay.add)

    driver_eval = embodied.EvalDriver(eval_env, logger, buffer_drawer)

    train_fill = max(0, args.train_fill - len(replay))
    if train_fill:
        print(f'Fill train dataset ({train_fill} steps).')
        random_agent = embodied.RandomAgent(env.act_space)
        driver(random_agent.policy, steps=train_fill, episodes=1)

    dataset = iter(agent.dataset(replay.dataset))
    state = [None]    # To be writable from train step function below.
    assert args.pretrain > 0    # At least one step to initialize variables.
    for _ in range(args.pretrain):
        _, state[0], _ = agent.train(next(dataset), state[0])

    metrics = collections.defaultdict(list)
    batch = [None]
    def train_step(tran, worker):
        if should_train(step):
            for _ in range(args.train_steps):
                batch[0] = next(dataset)
                outs, state[0], mets = agent.train(batch[0], state[0])
                [metrics[key].append(value) for key, value in mets.items()]
                if 'priority' in outs:
                    replay.prioritize(outs['key'], outs['priority'])
        if should_log(step):
            with warnings.catch_warnings():    # Ignore empty slice warnings.
                warnings.simplefilter('ignore', category=RuntimeWarning)
                for name, values in metrics.items():
                    logger.scalar('train/' + name, np.nanmean(values, dtype=np.float64))
                    metrics[name].clear()
            logger.add(agent.report(batch[0]), prefix='report')
            logger.add(timer.stats(), prefix='timer')
            logger.write(fps=True)
    driver.on_step(train_step)

    checkpoint = embodied.Checkpoint(logdir / 'checkpoint.pkl')
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.replay = replay
    checkpoint.load_or_save()

    print('Start training loop.')
    policy = lambda *args: agent.policy(
            *args, mode='explore' if should_expl(step) else 'train')
    policy_eval = lambda *args: agent.policy(*args, mode='eval')

    while step < args.steps:
        driver(policy, steps=args.eval_every)
        driver_eval.reset()
        driver_eval(policy_eval)
        checkpoint.save()
