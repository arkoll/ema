import collections
import re
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import embodied


def plot_episode(ep):
    pos = ep['obs_pos']
    goal = ep['goal_pos']
    fig, ax = plt.subplots(figsize=(5, 5), dpi=500)
    ax.set_aspect('equal')
    ax.scatter(pos[:, 0], pos[:, 1], c=np.arange(len(pos)), marker='.')
    ax.scatter(goal[:, 0], goal[:, 1], c='g', marker='*', s=50)
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))  
    img = np.transpose(img, [1, 2, 0])
    img = img.astype(float) / 255.
    plt.close(fig)
    return {'trajectory': img}


def distance_metrics(ep):
    metrics = {}
    delta = np.abs(ep['goal_pos'] - ep['obs_pos'])
    euclid = np.sqrt(np.sum(delta * delta, 1))
    metrics['l2_dist_mean'] = euclid.mean()
    metrics['l2_dist_min'] = euclid.min()
    metrics['normed_l2_dist_mean'] = euclid.mean() / euclid[0]
    metrics['normed_l2_dist_min'] = euclid.min() / euclid[0]
    manhattan = np.sum(delta, 1)
    metrics['l1_dist_mean'] = manhattan.mean()
    metrics['l1_dist_min'] = manhattan.min()
    metrics['normed_l1_dist_mean'] = manhattan.mean() / manhattan[0]
    metrics['normed_l1_dist_min'] = manhattan.min() / manhattan[0]
    return metrics


def train_eval(
        agent, env, eval_env, train_replay, logger, args
    ):

    logdir = embodied.Path(args.logdir)
    logdir.mkdirs()
    print('Logdir', logdir)
    should_train = embodied.when.Every(args.train_every)
    should_log = embodied.when.Every(args.log_every)
    should_plot = embodied.when.EveryRepeat(args.log_every)
    step = logger.step

    timer = embodied.Timer()
    timer.wrap('agent', agent, ['policy', 'train', 'report', 'save'])
    timer.wrap('env', env, ['step'])
    if hasattr(train_replay, '_sample'):
        timer.wrap('replay', train_replay, ['_sample'])

    nonzeros = set()
    def per_episode(ep, worker, mode):
        metrics = {}
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        print(
            f'{mode.title()} episode has {length} steps and return {score:.1f}.'
        )
        metrics['length'] = length
        metrics['score'] = score
        metrics['reward_rate'] = (
            ep['reward'] - ep['reward'].min() >= 0.1
        ).mean()
        metrics.update(distance_metrics(ep))
       
        logs = {}
        for key, value in ep.items():
            if (
                not args.log_zeros and key not in nonzeros
                and (value == 0).all()
            ):
                continue
            nonzeros.add(key)
            if re.match(args.log_keys_sum, key):
                logs[f'sum_{key}'] = ep[key].sum()
            if re.match(args.log_keys_mean, key):
                logs[f'mean_{key}'] = ep[key].mean()
            if re.match(args.log_keys_max, key):
                logs[f'max_{key}'] = ep[key].max(0).mean()

        logger.add(metrics, prefix=f'{mode}_episode')
        logger.add(logs, prefix=f'{mode}_logs')
        if mode == 'train':
            logger.add(train_replay.stats, prefix=f'{mode}_replay')
        if should_plot(step):
            logger.add(plot_episode(ep), prefix=f'{mode}_{worker}_plots')
        logger.write()
        
    eval_metrics = collections.defaultdict(list)
    def per_eval_sample(ep):
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        eval_metrics['length'].append(length)
        eval_metrics['score'].append(score)
        eval_metrics['reward_rate'].append(
            (ep['reward'] - ep['reward'].min() >= 0.1).mean()
        )
        for k, v in distance_metrics(ep).items():
            eval_metrics[k].append(v)
        for k, v in plot_episode(ep).items():
            eval_metrics[f'plot_{k}'].append(v)

    random_agent = embodied.RandomAgent(env.act_space)
    eval_driver = embodied.Driver(eval_env)
    eval_driver.on_episode(lambda ep, work: per_episode(ep, work, mode='eval'))
    eval_driver.on_episode(lambda ep, work: per_eval_sample(ep))

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep, worker, mode='train'))
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(train_replay.add)
    fill = max(0, args.train_fill - len(train_replay))
    if fill:
        print(f'Fill train dataset ({fill} steps).')
        driver(random_agent.policy, steps=fill, episodes=1)

    dataset_train = iter(agent.dataset(train_replay.dataset))
    state = [None]    # To be writable from train step function below.
    assert args.pretrain > 0    # At least one step to initialize variables.
    for _ in range(args.pretrain):
        _, state[0], _ = agent.train(next(dataset_train), state[0])

    metrics = collections.defaultdict(list)
    batch = [None]
    def train_step(tran, worker):
        if should_train(step):
            for _ in range(args.train_steps):
                batch[0] = next(dataset_train)
                outs, state[0], mets = agent.train(batch[0], state[0])
                [metrics[key].append(value) for key, value in mets.items()]
        if should_log(step):
            with warnings.catch_warnings():    # Ignore empty slice warnings.
                warnings.simplefilter('ignore', category=RuntimeWarning)
                for name, values in metrics.items():
                    logger.scalar(
                        'train/' + name, np.nanmean(values, dtype=np.float64)
                    )
                    metrics[name].clear()
            logger.add(agent.report(batch[0]), prefix='report')
            logger.add(timer.stats(), prefix='timer')
            logger.write(fps=True)
    driver.on_step(train_step)
 
    # Use buffer observations as goals
    current_goals = [next(dataset_train)[
        'observation'
    ].numpy().astype(np.float32)[:len(env), 0], next(dataset_train)[
        'obs_pos'
    ].numpy().astype(np.float32)[:len(env), 0]]
    def change_goal(obs):
        if obs['is_first'].any():
            b = next(dataset_train)
            goal = b['observation'].numpy().astype(np.float32)
            goal_pos = b['obs_pos'].numpy().astype(np.float32)
            for k in range(len(env)):
                if obs['is_first'][k]:
                    i = np.random.randint(0, goal.shape[0])
                    j = np.random.randint(0, goal.shape[1])
                    current_goals[0][k] = goal[i, j].copy()
                    current_goals[1][k] = goal_pos[i, j].copy()
        changed = {
            k: v  for k, v in obs.items() if k not in ('goal', 'goal_pos')
        }
        changed['goal'] = current_goals[0].copy()
        changed['goal_pos'] = current_goals[1].copy()
        return changed
    driver.on_observation(change_goal)

    checkpoint = embodied.Checkpoint(logdir / 'checkpoint.pkl')
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.train_replay = train_replay
    checkpoint.load_or_save()

    print('Start training loop.')
    policy = lambda *args: agent.policy(*args, mode='train')
    eval_policy = lambda *args: agent.policy(*args, mode='eval')
    while step < args.steps:
        logger.write()
        driver(policy, steps=args.eval_every)
        checkpoint.save()

        eval_driver(eval_policy, episodes=args.eval_samples)
        for k, v in eval_metrics.items():
            if k.startswith('plot'):
                for i, im in enumerate(v):
                    logger.image(f'eval_metrics/{i}_' + k, im)
                continue
            logger.scalar('eval_metrics/mean_' + k, np.mean(v))
            logger.scalar('eval_metrics/std_' + k, np.std(v))
        eval_metrics.clear()
        #TODO: add agent metrics on eval episodes
