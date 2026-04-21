"""Training script for baseline hybrid IL+RL algorithms.

Supports: sqil, bc, bc_ppo, rlpd, awr_gbrl_expert, awr_nn_expert

Uses the same environment setup as the main train.py but with baseline algos.
Supports wandb sweep integration via SWEEP_ID environment variable.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_PATH))

import warnings
warnings.filterwarnings("ignore")

import wandb
if not hasattr(wandb, 'START_TIME'):
    wandb.START_TIME = time.time()

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from env.register_minigrid import register_minigrid_tests
from env.wrappers import (CategoricalDummyVecEnv,
                          MiniGridCategoricalObservationWrapper)
from minigrid.wrappers import FlatObsWrapper
from utils.helpers import set_seed, make_minigrid_vec_env


EXPERT_DATASETS_DEFAULT = {
    "moveball": "datasets/minigrid/moveball_expert.npz",
    "keydoor": "datasets/minigrid/keydoor_expert.npz",
    "boxkey": "datasets/minigrid/boxkey_expert.npz",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train baseline IL+RL algorithms")

    # Environment
    parser.add_argument("--env_name", type=str, default="MiniGrid-MultiRoomCorridor-v0")
    parser.add_argument("--num_envs", type=int, default=1)

    # Algorithm
    parser.add_argument("--algo", type=str, required=True,
                        choices=["sqil", "bc", "bc_ppo", "rlpd",
                                 "awr_gbrl_expert", "awr_nn_expert"],
                        help="Baseline algorithm to run")

    # Training
    parser.add_argument("--total_n_steps", type=int, default=2000000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--run_name", type=str, default=None)

    # Expert data
    parser.add_argument("--expert_datasets", type=str, default=None,
                        help='JSON dict of {name: path} for expert .npz files')

    # DQN / shared RL params (for SQIL, RLPD)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_starts", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target_update_interval", type=int, default=1000)
    parser.add_argument("--exploration_fraction", type=float, default=0.1)
    parser.add_argument("--exploration_final_eps", type=float, default=0.05)
    parser.add_argument("--train_freq", type=int, default=4)
    parser.add_argument("--gradient_steps", type=int, default=1)

    # BC params
    parser.add_argument("--bc_epochs", type=int, default=50)
    parser.add_argument("--bc_lr", type=float, default=1e-3)
    parser.add_argument("--bc_ent_weight", type=float, default=1e-3)

    # PPO finetune params (for bc_ppo)
    parser.add_argument("--ppo_lr", type=float, default=3e-4)
    parser.add_argument("--ppo_n_steps", type=int, default=2048)
    parser.add_argument("--ppo_batch_size", type=int, default=64)
    parser.add_argument("--ppo_ent_coef", type=float, default=0.01)

    # AWR params (for awr_gbrl_expert, awr_nn_expert)
    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--ent_coef", type=float, default=0.5)
    parser.add_argument("--vf_coef", type=float, default=0.56)
    parser.add_argument("--weights_max", type=float, default=20.0)
    parser.add_argument("--policy_lr", type=float, default=0.007556)
    parser.add_argument("--value_lr", type=float, default=0.005158)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--n_bins", type=int, default=250)
    parser.add_argument("--grow_policy", type=str, default="oblivious")
    parser.add_argument("--split_score_func", type=str, default="Cosine")
    parser.add_argument("--generator_type", type=str, default="Quantile")
    parser.add_argument("--fixed_std", type=str, default="False")
    parser.add_argument("--log_std_lr", type=str, default="lin_0.0017")
    parser.add_argument("--min_log_std_lr", type=float, default=0.000475)

    # Logging
    parser.add_argument("--log_interval", type=int, default=4)

    return parser.parse_args()


def _inject_sweep_config():
    """Bridge wandb sweep config -> argparse by rebuilding sys.argv.

    Same pattern as train_runner.py's _inject_args_from_wandb_config.
    """
    wandb.init(sync_tensorboard=True, monitor_gym=True)
    config = dict(wandb.config)

    algo = config.get('algo', 'baseline')
    env = config.get('env_name', '')
    seed = config.get('seed', 0)
    wandb.run.name = f"{algo}_{env}_s{seed}"

    print(f"[SWEEP] Received {len(config)} params from sweep controller")
    print(f"[SWEEP] Config: {config}")

    sys.argv = [sys.argv[0]]
    for key, val in config.items():
        if key.startswith('_'):
            continue
        if val is None:
            continue
        if isinstance(val, list):
            sys.argv.append(f'--{key}')
            sys.argv.extend(str(v) for v in val)
        elif isinstance(val, dict):
            sys.argv.extend([f'--{key}', json.dumps(val)])
        else:
            sys.argv.extend([f'--{key}', str(val)])
    print(f"[SWEEP] Injected sys.argv: {' '.join(sys.argv)}")


def make_env(args, categorical=False):
    """Create MiniGrid environment.

    categorical=True for GBRL algos (CategoricalDummyVecEnv + MiniGridCategoricalObservationWrapper)
    categorical=False for NN algos (DummyVecEnv + FlatObsWrapper)
    """
    register_minigrid_tests()
    if categorical:
        env = make_minigrid_vec_env(
            args.env_name,
            n_envs=args.num_envs,
            seed=args.seed,
            wrapper_class=MiniGridCategoricalObservationWrapper,
            vec_env_cls=CategoricalDummyVecEnv,
        )
    else:
        env = make_minigrid_vec_env(
            args.env_name,
            n_envs=args.num_envs,
            seed=args.seed,
            wrapper_class=FlatObsWrapper,
            vec_env_cls=DummyVecEnv,
        )
    return env


def _run_baseline():
    """Single baseline training run (called directly or from sweep agent)."""
    args = parse_args()
    set_seed(args.seed)

    # Parse expert datasets
    if args.expert_datasets:
        expert_datasets = json.loads(args.expert_datasets)
    else:
        expert_datasets = EXPERT_DATASETS_DEFAULT

    run_name = args.run_name or f"baseline_{args.algo}_seed_{args.seed}"
    log_file = ROOT_PATH / f"logs/{run_name}.log"
    os.makedirs(log_file.parent, exist_ok=True)

    # wandb init for non-sweep runs (sweep runs already called wandb.init)
    use_wandb = os.environ.get('WANDB_PROJECT') or os.environ.get('SWEEP_ID')
    if use_wandb and wandb.run is None:
        wandb.init(
            project=os.environ.get('WANDB_PROJECT', 'baselines_minigrid'),
            entity=os.environ.get('WANDB_ENTITY', 'nvidia-mellanox'),
            name=run_name,
            config=vars(args),
            sync_tensorboard=True,
            monitor_gym=True,
        )

    print(f"=" * 60)
    print(f"Baseline: {args.algo}")
    print(f"Env: {args.env_name}")
    print(f"Seed: {args.seed}")
    print(f"Steps: {args.total_n_steps}")
    print(f"Device: {args.device}")
    print(f"Expert datasets: {list(expert_datasets.keys())}")
    print(f"=" * 60)

    categorical = args.algo in ("awr_gbrl_expert",)
    env = make_env(args, categorical=categorical)
    print(f"Obs space: {env.observation_space}")
    print(f"Act space: {env.action_space}")

    if args.algo == "sqil":
        from algos.baselines.sqil_baseline import SQILBaseline
        agent = SQILBaseline(
            env=env,
            expert_datasets=expert_datasets,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            gamma=args.gamma,
            target_update_interval=args.target_update_interval,
            exploration_fraction=args.exploration_fraction,
            exploration_final_eps=args.exploration_final_eps,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            seed=args.seed,
            device=args.device,
            verbose=args.verbose,
        )

    elif args.algo == "bc":
        from algos.baselines.bc_baseline import BCBaseline
        agent = BCBaseline(
            env=env,
            expert_datasets=expert_datasets,
            batch_size=args.batch_size,
            n_epochs=args.bc_epochs,
            lr=args.bc_lr,
            ent_weight=args.bc_ent_weight,
            seed=args.seed,
            device=args.device,
            verbose=args.verbose,
        )

    elif args.algo == "bc_ppo":
        from algos.baselines.bc_baseline import BCPPOFinetuneBaseline
        agent = BCPPOFinetuneBaseline(
            env=env,
            expert_datasets=expert_datasets,
            bc_epochs=args.bc_epochs,
            bc_batch_size=args.batch_size,
            bc_lr=args.bc_lr,
            bc_ent_weight=args.bc_ent_weight,
            ppo_lr=args.ppo_lr,
            ppo_n_steps=args.ppo_n_steps,
            ppo_batch_size=args.ppo_batch_size,
            ppo_ent_coef=args.ppo_ent_coef,
            seed=args.seed,
            device=args.device,
            verbose=args.verbose,
        )

    elif args.algo == "rlpd":
        from algos.baselines.rlpd import RLPDBaseline
        agent = RLPDBaseline(
            env=env,
            expert_datasets=expert_datasets,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=0,  # Start immediately with expert data
            gamma=args.gamma,
            target_update_interval=args.target_update_interval,
            exploration_fraction=args.exploration_fraction,
            exploration_final_eps=args.exploration_final_eps,
            train_freq=args.train_freq,
            gradient_steps=4,  # Higher UTD ratio for RLPD
            seed=args.seed,
            device=args.device,
            verbose=args.verbose,
        )

    elif args.algo == "awr_gbrl_expert":
        from algos.baselines.awr_gbrl_expert import AWR_GBRL_Expert

        # Build policy_kwargs matching the canonical GBRL AWR structure
        policy_kwargs = {
            'shared_tree_struct': True,
            'tree_struct': {
                'max_depth': args.max_depth,
                'n_bins': args.n_bins,
                'grow_policy': args.grow_policy,
            },
            'tree_optimizer': {
                'params': {
                    'split_score_func': args.split_score_func,
                    'generator_type': args.generator_type,
                },
                'device': args.device,
                'policy_optimizer': {
                    'policy_lr': args.policy_lr,
                },
                'value_optimizer': {
                    'value_lr': args.value_lr,
                },
            },
        }

        agent = AWR_GBRL_Expert(
            env=env,
            expert_datasets=expert_datasets,
            train_freq=args.train_freq,
            beta=args.beta,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            weights_max=args.weights_max,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            vf_coef=args.vf_coef,
            gradient_steps=args.gradient_steps,
            fixed_std=(args.fixed_std.lower() == 'true'),
            log_std_lr=args.log_std_lr,
            min_log_std_lr=args.min_log_std_lr,
            policy_kwargs=policy_kwargs,
            seed=args.seed,
            device=args.device,
            verbose=args.verbose,
        )

    elif args.algo == "awr_nn_expert":
        from algos.baselines.awr_nn_expert import AWR_NN_Expert

        agent = AWR_NN_Expert(
            policy="MlpPolicy",
            env=env,
            expert_datasets=expert_datasets,
            train_freq=args.train_freq,
            beta=args.beta,
            gamma=args.gamma,
            ent_coef=args.ent_coef,
            weights_max=args.weights_max,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            learning_rate=args.learning_rate,
            policy_kwargs={
                "optimizer_kwargs": {
                    "actor_lr": args.learning_rate,
                    "critic_lr": args.learning_rate,
                },
            },
            seed=args.seed,
            device=args.device,
            verbose=args.verbose,
        )

    else:
        raise ValueError(f"Unknown baseline: {args.algo}")

    print(f"\nStarting training...")
    agent.learn(
        total_timesteps=args.total_n_steps,
        log_interval=args.log_interval,
    )

    print(f"\nTraining complete: {run_name}")

    if wandb.run is not None:
        wandb.finish()


def main():
    """Entry point: direct run or wandb sweep agent."""
    sweep_id = os.environ.get('SWEEP_ID')
    if sweep_id:
        project = os.environ.get('WANDB_PROJECT', 'baselines_minigrid')
        entity = os.environ.get('WANDB_ENTITY', 'nvidia-mellanox')
        print(f"[SWEEP] Running as sweep agent for {entity}/{project}/{sweep_id}")

        def sweep_fn():
            _inject_sweep_config()
            _run_baseline()

        wandb.agent(sweep_id, function=sweep_fn, project=project, entity=entity)
    else:
        _run_baseline()


if __name__ == "__main__":
    main()
