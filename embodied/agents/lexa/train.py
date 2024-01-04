import pathlib
import sys
import warnings

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')

directory = pathlib.Path(__file__)
try:
    import google3    # noqa
except ImportError:
    directory = directory.resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

import embodied


def main(argv=None):
    from . import agent as agnt
    from . import custom_train

    parsed, other = embodied.Flags(
        configs=['defaults'], actor_id=0, actors=0,
    ).parse_known(argv)
    config = embodied.Config(agnt.Agent.configs['defaults'])
    for name in parsed.configs:
        config = config.update(agnt.Agent.configs[name])
    config = embodied.Flags(config).parse(other)

    config = config.update(logdir=str(embodied.Path(config.logdir)))
    args = embodied.Config(
        logdir=config.logdir, length=config.env.length, **config.train
    )
    args = args.update(expl_until=args.expl_until // config.env.repeat)
    print(config)

    # hack to limit available gpus
    if config.tf.device != 'all':
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.tf.device)

    logdir = embodied.Path(config.logdir)
    step = embodied.Counter()
    cleanup = []

    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(config.filter),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.WandBOutput(config.filter, logdir, config),
    ], multiplier=config.env.repeat)

    chunk = config.replay_chunk
    def make_replay(name, capacity):
        directory = logdir / name
        store = embodied.replay.CkptRAMStore(
            directory, capacity, parallel=True
        )
        cleanup.append(store)
        return embodied.replay.FixedLength(
            store, chunk, **config.replay_fixed
        )

    try:
        config = config.update({
            'env.seed': hash((config.seed, parsed.actor_id))
        })
        env = embodied.envs.load_env(
            config.task, mode='train', logdir=logdir, **config.env
        )
        agent = agnt.Agent(env.obs_space, env.act_space, step, config)
        eval_env = embodied.envs.load_env(
            config.task, mode='eval', logdir=logdir, **config.env
        )
        replay = make_replay('episodes', config.replay_size)
        custom_train.train_eval(agent, env, eval_env, replay, logger, args)
        cleanup.append(eval_env)
    finally:
        for obj in cleanup:
            obj.close()


if __name__ == '__main__':
    main()
