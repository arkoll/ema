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


def plot_episode_traj(rollout, episode, goals, update, update_exp):
    changes = np.argwhere(update)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=500)
    ax.set_aspect('equal')
    ax.plot(rollout[1:, 0], rollout[1:, 1], color='k', zorder=1)
    ax.plot(episode[:, 0], episode[:, 1], color='g', zorder=1)
    if update_exp[1]:
        g_col = 'b'
        st_col = 'r'
    else:
        g_col = 'lightsteelblue'
        st_col = 'salmon'
    ax.scatter(
        episode[0, 0], episode[0, 1], s=50, color=st_col, marker='*', zorder=2
    )
    ax.scatter(
        rollout[1, 0], rollout[1, 1], s=50, color=st_col, marker='*', zorder=2
    )
    ax.scatter(
        goals[1, 0], goals[1, 1], s=50, color=g_col, marker='$0$', zorder=2
    )
    for i, change in enumerate(changes):
        ch = change[0]
        goal = goals[ch]
        if update_exp[ch]:
            g_col = 'b'
            st_col = 'r'
        else:
            g_col = 'lightsteelblue'
            st_col = 'salmon'
        ax.scatter(
            goal[0], goal[1], s=50, color=g_col, marker=f'${i+1}$', zorder=2
        )
        ep_step = episode[ch-1]
        ax.scatter(
            ep_step[0], ep_step[1], s=50, color=st_col, marker=f'${i+1}$',
            zorder=2
        )
        ro_step = rollout[ch]
        ax.scatter(
            ro_step[0], ro_step[1], s=50, color=st_col, marker=f'${i+1}$',
            zorder=2
        )
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))  
    img = np.transpose(img, [1, 2, 0])
    img = img.astype(float) / 255. 
    plt.close(fig)
    return img


def plot_gtrajs(goal_true_coord, goal_traj):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=500)
    ax.set_aspect('equal')
    cmap = mpl.colormaps['tab20']
    for g in range(goal_traj.shape[1]):
        for s in range(goal_traj.shape[2]):
            ax.plot(
                goal_traj[:, g, s, 0], goal_traj[:, g, s, 1], color=cmap(g),
                alpha=0.3, zorder=1
            )
    ax.scatter(
        goal_true_coord[:, 0], goal_true_coord[:, 1],
        color=cmap(range(goal_traj.shape[1])), s=20, zorder=2
    )
    ax.scatter(
        goal_traj[0, :, 0, 0], goal_traj[0, :, 0, 1], color='k', s=20, zorder=2
    )
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


def plot_bgoals(cur_goals, saved_goals):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=500)
    ax.set_aspect('equal')
    cmap = mpl.colormaps['tab20']
    for i, sk in enumerate(cur_goals):
        ax.scatter(sk[:, 0], sk[:, 1], color=cmap(i))
    for i, sk in enumerate(saved_goals):
        ax.scatter(sk[:, 0], sk[:, 1], color=cmap(i), marker='+')
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
        if 'buffer_goals' in key:
            data[key] = plot_bgoals(*value)
    return data


def train_with_viz(
        agent, env, eval_env, train_replay, eval_replay, logger, args
    ):

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
    def per_episode(ep, mode):
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
        if mode == 'train':
            should = should_video(step)
        else:
            should = True
        if should:
            if 'log_position' in ep:
                metrics[f'policy_position'] = plot_episode_traj(
                    ep['log_position'], ep['absolute_position'],
                    ep['log_cgoal'], ep['log_update'], ep['log_update_exp']
                )
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
        replay = dict(train=train_replay, eval=eval_replay)[mode]
        logger.add(metrics, prefix=f'{mode}_episode')
        logger.add(logs, prefix=f'{mode}_logs')
        logger.add(replay.stats, prefix=f'{mode}_replay')
        logger.write()

    random_agent = embodied.RandomAgent(env.act_space)
    eval_driver = embodied.Driver(eval_env)
    eval_driver.on_step(eval_replay.add)
    eval_driver.on_episode(lambda ep, worker: per_episode(ep, mode='eval'))
    
    eval_metrics = collections.defaultdict(list)
    def eval_episode(ep):
        g = ep['GOAL_absolute_position']
        s = ep['absolute_position']
        d = np.linalg.norm(g - s, axis=1)
        eval_metrics['distance'].append(d.min())
        d = d / d[0]
        eval_metrics['normed_distance'].append(d.min())
    eval_driver.on_episode(lambda ep, worker: eval_episode(ep))

    fill = max(0, args.eval_fill - len(eval_replay))
    if fill:
        print(f'Fill eval dataset ({fill} steps).')
        eval_driver(random_agent.policy, steps=fill, episodes=1)

    driver = embodied.Driver(env)
    driver.on_episode(lambda ep, worker: per_episode(ep, mode='train'))
    driver.on_step(lambda tran, _: step.increment())
    driver.on_step(train_replay.add)
    fill = max(0, args.train_fill - len(train_replay))
    if fill:
        print(f'Fill train dataset ({fill} steps).')
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
        *args, mode='explore' if should_expl(step) else 'train'
    )
    eval_policy = lambda *args: agent.policy(*args, mode='eval')
    while step < args.steps:
        logger.write()
        eval_driver.reset()
        eval_metrics.clear()
        eval_driver(eval_policy, episodes=max(len(eval_env), args.eval_eps))
        mets = {k: np.mean(v) for k, v in eval_metrics.items()}
        logger.add(mets, prefix='eval')
        driver(policy, steps=args.eval_every)
        checkpoint.save()

