import collections
import re
import warnings

import embodied
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def plot_goals(skill_goals):
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=500)
    cmap = mpl.colormaps['jet']
    ax.set_aspect('equal')
    steps, n_skills, *_ = skill_goals.shape
    for step in range(steps):
        for sk in range(n_skills):
            ax.scatter(
                skill_goals[step, sk, :, 0], skill_goals[step, sk, :, 1], s=10,
                color=cmap(step / steps), marker=f'${step}$'
            )
    for sk in range(n_skills):
        ax.scatter(
            skill_goals[0, sk, :, 0], skill_goals[0, sk, :, 1], s=30,
            color='k', marker='o'
        )
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))  
    img = np.transpose(img, [1, 2, 0])
    img = img.astype(float) / 255.      
    plt.close(fig)
    return img


def plot_trajs(initial, rollout):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=500)
    cmap = mpl.colormaps['tab20']
    ax.set_aspect('equal')
    for sk in range(rollout.shape[1]):
        for s in range(rollout.shape[2]):
            points = rollout[:, sk, s, :2]
            ax.scatter(points[:, 0], points[:, 1], s=5, color=cmap(sk))
    ax.scatter(initial[0], initial[1], s=20, color='k')
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))  
    img = np.transpose(img, [1, 2, 0])
    img = img.astype(float) / 255.  
    plt.close(fig)
    return img


def plot_gtrajs(rec, goal_rollout):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=500)
    cmap = mpl.colormaps['tab20']
    ax.set_aspect('equal')
    ax.plot(rec[:, 0], rec[:, 1], color='k')
    for s in range(goal_rollout.shape[1]):
        ax.plot(goal_rollout[:, s, 0], goal_rollout[:, s, 1], color='r')
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))  
    img = np.transpose(img, [1, 2, 0])
    img = img.astype(float) / 255.  
    plt.close(fig)
    return img


def plot_landmarks(goals, reward, max_traj):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=500)
    cmap = mpl.colormaps['viridis']
    ax.set_aspect('equal')
    indices = np.argsort(reward)
    n_points = len(reward)
    ax.plot(
        max_traj[:, :, 0], max_traj[:, :, 1], color='r', alpha=0.3, zorder=1
    )
    for i, g in enumerate(indices):
        ax.scatter(goals[g, 0], goals[g, 1], s=20, color=cmap(i / n_points))
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))  
    img = np.transpose(img, [1, 2, 0])
    img = img.astype(float) / 255.  
    plt.close(fig)
    return img


def postprocess_report(data):
    for key, value in data.items():
        if 'skill_goals' in key:
            data[key] = plot_goals(value)
        if 'skill_trajs' in key:
            data[key] = plot_trajs(*value)
        if 'goal_trajs' in key:
            data[key] = plot_gtrajs(*value)
        if 'landmarks' in key:
            data[key] = plot_landmarks(*value)
    return data


def train_with_viz(agent, env, train_replay, eval_replay, logger, args):

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
    if hasattr(train_replay, '_sample'):
        timer.wrap('replay', train_replay, ['_sample'])

    nonzeros = set()
    def per_episode(ep):
        metrics = {}
        length = len(ep['reward']) - 1
        score = float(ep['reward'].astype(np.float64).sum())
        print(f'Episode has {length} steps and return {score:.1f}.')
        metrics['length'] = length
        metrics['score'] = score
        metrics['reward_rate'] = (
            ep['reward'] - ep['reward'].min() >= 0.1
        ).mean()
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
        if should_video(step):
            for key in args.log_keys_video:
                if key == 'none':
                    continue
                metrics[f'policy_{key}'] = ep[key]
                if 'log_goal' in ep:
                    if ep['image'].shape == ep['log_goal'].shape:
                        goal = (255 * ep['log_goal']).astype(np.uint8)
                        metrics[f'policy_{key}_with_goal'] = np.concatenate(
                            [ep['image'], goal], 2
                        )
        logger.add(metrics, prefix='episode')
        logger.add(logs, prefix='logs')
        logger.add(train_replay.stats, prefix='replay')
        logger.write()

    fill = max(0, args.eval_fill - len(eval_replay))
    if fill:
        print(f'Fill eval dataset ({fill} steps).')
        eval_driver = embodied.Driver(env)
        eval_driver.on_step(eval_replay.add)
        random_agent = embodied.RandomAgent(env.act_space)
        eval_driver(random_agent.policy, steps=fill, episodes=1)
        del eval_driver

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep))
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(train_replay.add)
    fill = max(0, args.train_fill - len(train_replay))
    if fill:
        print(f'Fill train dataset ({fill} steps).')
        random_agent = embodied.RandomAgent(env.act_space)
        driver(random_agent.policy, steps=fill, episodes=1)

    dataset_train = iter(agent.dataset(train_replay.dataset))
    dataset_eval = iter(agent.dataset(eval_replay.dataset))
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
                if 'priority' in outs:
                    train_replay.prioritize(outs['key'], outs['priority'])
        if should_log(step):
            with warnings.catch_warnings():    # Ignore empty slice warnings.
                warnings.simplefilter('ignore', category=RuntimeWarning)
                for name, values in metrics.items():
                    logger.scalar(
                        'train/' + name, np.nanmean(values, dtype=np.float64)
                    )
                    metrics[name].clear()
            logger.add(
                postprocess_report(agent.report(batch[0])), prefix='report'
            )
            logger.add(
                postprocess_report(agent.report(next(dataset_eval))),
                prefix='eval'
            )
            logger.add(timer.stats(), prefix='timer')
            logger.write(fps=True)
    driver.on_step(train_step)

    checkpoint = embodied.Checkpoint(logdir / 'checkpoint.pkl')
    checkpoint.step = step
    checkpoint.agent = agent
    checkpoint.train_replay = train_replay
    checkpoint.eval_replay = eval_replay
    checkpoint.load_or_save()

    print('Start training loop.')
    policy = lambda *args: agent.policy(
            *args, mode='explore' if should_expl(step) else 'train')
    while step < args.steps:
        logger.write()
        driver(policy, steps=args.eval_every)
        checkpoint.save()

