import argparse
import os
import shutil
import datetime
from pathlib import Path

from agent.runners.DreamerRunner import DreamerRunner
from configs import Experiment
from configs.EnvConfigs import StarCraftConfig, EnvCurriculumConfig, SMAXConfig

# for smac
from configs.dreamer.DreamerControllerConfig import DreamerControllerConfig
from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig

# for SMAX
from configs.dreamer.smax.SMAXLearnerConfig import SMAXDreamerLearnerConfig
from configs.dreamer.smax.SMAXControllerConfig import SMAXDreamerControllerConfig

from environments import Env

import torch
import numpy as np
import random


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="flatland", help='Flatland or SMAC env')
    parser.add_argument('--env_name', type=str, default="5_agents", help='Specific setting')
    parser.add_argument('--n_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--steps', type=int, default=1e6, help='Number of steps')
    parser.add_argument('--mode', type=str, default='disabled')
    return parser.parse_args()


def train_dreamer(exp, n_workers): 
    runner = DreamerRunner(exp.env_config, exp.learner_config, exp.controller_config, n_workers)
    runner.run(exp.steps, exp.episodes, save_interval=200000, save_mode='interval')


def get_env_info(configs, env):
    if not env.discrete:
        assert hasattr(env, 'individual_action_space')
        individual_action_space = env.individual_action_space
    else:
        individual_action_space = None

    for config in configs:
        config.IN_DIM = env.n_obs
        config.ACTION_SIZE = env.n_actions
        config.NUM_AGENTS = env.n_agents
        config.CONTINUOUS_ACTION = not env.discrete
        config.ACTION_SPACE = individual_action_space
    
    print(f'Observation dims: {env.n_obs}')
    print(f'Action dims: {env.n_actions}')
    print(f'Num agents: {env.n_agents}')
    print(f'Continuous action for control? -> {not env.discrete}')
    
    if hasattr(env, 'individual_action_space'):
        print(f'Individual action space: {env.individual_action_space}')

    env.close()


def prepare_starcraft_configs(env_name):
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    env_config = StarCraftConfig(env_name, RANDOM_SEED)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 2000),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}


def prepare_smax_configs(env_name):
    agent_configs = [SMAXDreamerControllerConfig(), SMAXDreamerLearnerConfig()]
    env_config = SMAXConfig(env_name, RANDOM_SEED)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 5000),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}

if __name__ == "__main__":
    RANDOM_SEED = 23
    args = parse_args()
    RANDOM_SEED += args.seed * 100
    
    if args.env == Env.STARCRAFT:
        configs = prepare_starcraft_configs(args.env_name)
    elif args.env == Env.SMAX:
        configs = prepare_smax_configs(args.env_name)
    else:
        raise Exception("Unknown environment")
    
    # seed everywhere
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    torch.autograd.set_detect_anomaly(True)
    # --------------------
    
    configs["env_config"][0].ENV_TYPE = Env(args.env)
    configs["learner_config"].ENV_TYPE = Env(args.env)
    configs["controller_config"].ENV_TYPE = Env(args.env)
    
    configs["learner_config"].seed = RANDOM_SEED

    current_date = datetime.datetime.now()
    current_date_string = current_date.strftime("%m%d")

    # make run directory
    run_dir = Path(os.path.dirname(os.path.abspath(__file__)) + f"/{current_date_string}_results") / args.env / args.env_name
    if not run_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                            str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))
        os.makedirs(str(run_dir / "ckpt"))

    shutil.copytree(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "agent"), dst=run_dir / "agent")
    shutil.copytree(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "configs"), dst=run_dir / "configs")
    shutil.copytree(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "networks"), dst=run_dir / "networks")
    shutil.copyfile(src=(Path(os.path.dirname(os.path.abspath(__file__))) / "train.py"), dst=run_dir / "train.py")
    
    print(f"Run files are saved at {str(run_dir)}\n")
    # -------------------

    configs["learner_config"].RUN_DIR = str(run_dir)
    configs["learner_config"].map_name = args.env_name

    run_name = f'mamba_{args.env_name}_seed_{RANDOM_SEED}'

    global wandb
    import wandb
    wandb.init(
        project=args.env,
        mode=args.mode,
        group=f'mamba_{args.env_name}',
        name=run_name,
        config=configs["learner_config"].to_dict() if hasattr(configs["learner_config"], 'to_dict') else {},
        notes="",
    )

    print(f'mamba_{args.env_name}')
    print(run_name)

    exp = Experiment(steps=args.steps,
                     episodes=50000,
                     random_seed=RANDOM_SEED,
                     env_config=EnvCurriculumConfig(*zip(configs["env_config"]), Env(args.env),
                                                    obs_builder_config=configs["obs_builder_config"],
                                                    reward_config=configs["reward_config"]),
                     controller_config=configs["controller_config"],
                     learner_config=configs["learner_config"])

    train_dreamer(exp, n_workers=args.n_workers)
