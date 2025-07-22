##############################################################################
# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/gbrl_sb3/license.html
#
##############################################################################
import os
import sys
import warnings
from pathlib import Path

from typing import Optional
import json

ROOT_PATH = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_PATH))

from wandb.integration.sb3 import WandbCallback
from wandb.sdk.wandb_run import Run as WandbRun

from stable_baselines3.common.callbacks import (
    CallbackList, CheckpointCallback, EvalCallback,
    StopTrainingOnNoModelImprovement)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import (DummyVecEnv, VecFrameStack,
                                              VecNormalize, VecVideoRecorder)

from callback.callbacks import (OffPolicyDistillationCallback,
                                OnPolicyDistillationCallback,
                                StopTrainingOnNoImprovementInTraining)
from env.equation import register_equation_tests
from env.minigrid import register_minigrid_tests
from env.rickety_bridge import register_rickety_bridge_tests
from env.wrappers import (CategoricalDummyVecEnv,
                          MiniGridCategoricalObservationWrapper)
from utils.helpers import make_ram_atari_env, set_seed

warnings.filterwarnings("ignore")

import wandb

from stable_baselines3.a2c.a2c import A2C
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.ppo.ppo import PPO

from algos.a2c import A2C_GBRL
from algos.awr import AWR_GBRL
from algos.awr_nn import AWR
from algos.dqn import DQN_GBRL
from algos.ppo import PPO_GBRL
from algos.sac import SAC_GBRL
from config.args import parse_args, process_policy_kwargs

from functools import partial

sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
from userlib.auto_resume import AutoResume

NAME_TO_ALGO = {'ppo_gbrl': PPO_GBRL, 'a2c_gbrl': A2C_GBRL, 'sac_gbrl': SAC_GBRL, 'awr_gbrl': AWR_GBRL,
                'ppo_nn': PPO, 'a2c_nn': A2C, 'dqn_gbrl': DQN_GBRL, 'awr_nn': AWR, 'dqn_nn': DQN}
CATEGORICAL_ALGOS = [algo for algo in NAME_TO_ALGO if 'gbrl' in algo]
ON_POLICY_ALGOS = ['ppo_gbrl', 'a2c_gbrl']
OFF_POLICY_ALGOS = ['sac_gbrl', 'dqn_gbrl', 'awr_gbrl']


def process_slurm_logging(args, callback_list):
    if not args.wandb:
        tb_name = f'n_{args.env_type}_{args.run_name}_{args.env_name}_seed_{args.seed}'
        tensorboard_log = f"runs/{tb_name}"
        return tensorboard_log, None

    print(f'args.wand: {args.wandb}')
    save_dir = os.path.join(args.save_path, f'{args.env_type}/' f'{args.env_name}/{args.algo_type}')
    wandb_run_id_file = Path(save_dir, "wandb_run_id.json")
    wandb_run_id = json.loads(wandb_run_id_file.read_text())["wandb_run_id"] if \
        wandb_run_id_file.exists() else None

    if wandb_run_id:
        print(f"WANDB RESUMING RUN: {wandb_run_id}")
        os.environ["WANDB_RUN_ID "] = wandb_run_id

    # run_name = args.project
    run = wandb.init(project=args.project, group=None if args.group_name is None else
               args.group_name + '_' + args.algo_type + '_' + args.env_type + '_' + args.env_name,
               name=args.run_name + '_' + args.algo_type + '_' + args.env_type + '_' +
               args.env_name + '_seed_' + str(args.seed), mode="online",
               config=args.__dict__, entity=args.entity, monitor_gym=True, sync_tensorboard=True,
               save_code=False,
               resume="allow" if wandb_run_id else None,
               id=wandb_run_id if wandb_run_id else None,)

    wandb_run_id_file.write_text(json.dumps({"wandb_run_id": run.id}))
    print(f"WandB run initialized with run id: {run.id}, entity: {run.entity}, project: {run.project}, sweep_id: {run.sweep_id}")

    print("Requesting autoresume.")
    AutoResume.request_resume(user_dict={'job_id': os.environ.get("SLURM_JOB_ID"), 'reason': 'logger'})
    if run is not None:
        print("Mark preempting for wandb.")
        run.mark_preempting()

    callback_list.append(WandbCallback(
                         gradient_save_freq=0,
                         model_save_path=None,
                         verbose=1)
                         )
    tb_name = ''
    if args.group_name is not None:
        tb_name += f'g_{args.group_name}_'
    tb_name += f'n_{args.env_type}_{args.run_name}_{args.env_name}_seed_{args.seed}'
    tensorboard_log = f"runs/{tb_name}"
    # args.wandb = wandb
    print('finished setting up wandb')
    return tensorboard_log, run


class SLURMCheckpointCallback(CheckpointCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    By default, it only saves model checkpoints,
    you need to pass ``save_replay_buffer=True``,
    and ``save_vecnormalize=True`` to also save replay buffer checkpoints
    and normalization statistics checkpoints.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq: Save checkpoints every ``save_freq`` call of the callback.
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param save_replay_buffer: Save the model replay buffer
    :param save_vecnormalize: Save the ``VecNormalize`` statistics
    :param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        wandb_run: Optional[WandbRun] = None,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        super().__init__(save_freq, save_path, name_prefix=name_prefix,
                         save_replay_buffer=save_replay_buffer,
                         save_vecnormalize=save_vecnormalize,
                         verbose=verbose)
        self.run = wandb_run


    def _on_step(self) -> bool:
        response = super()._on_step()

        print("Received preemption indication, requesting resume...")
        AutoResume.request_resume(user_dict={'job_id': os.environ.get("SLURM_JOB_ID"), 'reason': 'timelimit'})

        SIGTERM_SIGNAL = 15
        # SIGTERM inspired exit code, indicates job is preempting, just for wandb!
        PREEMTING_EXIT_CODE = 128 + SIGTERM_SIGNAL

        if self.run is not None:
            print("Mark preempting for wandb.")
            self.run.mark_preempting()
            self.run.finish(exit_code=PREEMTING_EXIT_CODE)
            # Kill wandb agent processes before exiting
        exit(PREEMTING_EXIT_CODE)  # exit with 0 for auto-resume to launch next slurm job

        return response

if __name__ == '__main__':
    args = parse_args()

    job_id = os.environ.get("SLURM_JOB_ID")
    print(f"Running slurm job {job_id}")

    args.save_path = os.path.join(args.save_path, job_id)

    callback_list = []
    if args.distil and args.distil_kwargs:
        if args.algo_type in ON_POLICY_ALGOS:
            callback_list.append(OnPolicyDistillationCallback(args.distil_kwargs,
                                                              args.distil_kwargs.get('distil_verbose', 0)))
        elif args.algo_type in OFF_POLICY_ALGOS:
            callback_list.append(OffPolicyDistillationCallback(args.distil_kwargs,
                                                               args.distil_kwargs.get('distil_verbose', 0)))

    tensorboard_log, wandb_run = process_slurm_logging(args, callback_list)
    env, eval_env = None, None
    if args.env_kwargs is None:
        args.env_kwargs = {}
    learn_kwargs = {}
    if 'atari' in args.env_type:
        env_kwargs = {'full_action_space': False}
        vec_env_cls = None
        vec_env_kwargs = None
        env = make_ram_atari_env(args.env_name, n_envs=args.num_envs, seed=args.seed,
                                 wrapper_kwargs=args.atari_wrapper_kwargs, env_kwargs=env_kwargs,
                                 vec_env_cls=vec_env_cls, vec_env_kwargs=vec_env_kwargs)
        if args.evaluate:
            eval_env = make_ram_atari_env(args.env_name, n_envs=1, wrapper_kwargs=args.atari_wrapper_kwargs,
                                          env_kwargs=env_kwargs, vec_env_cls=vec_env_cls,
                                          vec_env_kwargs=vec_env_kwargs)
        if args.atari_wrapper_kwargs and 'frame_stack' in args.atari_wrapper_kwargs:
            env = VecFrameStack(env, n_stack=args.atari_wrapper_kwargs['frame_stack'])
            if eval_env:
                eval_env = VecFrameStack(eval_env, n_stack=args.atari_wrapper_kwargs['frame_stack'])
    elif args.env_type == 'minigrid':
        register_minigrid_tests()
        from minigrid.wrappers import FlatObsWrapper
        wrapper_class = MiniGridCategoricalObservationWrapper if args.algo_type in CATEGORICAL_ALGOS else FlatObsWrapper
        vec_env_cls = CategoricalDummyVecEnv if args.algo_type in CATEGORICAL_ALGOS else DummyVecEnv
        env = make_vec_env(args.env_name, n_envs=args.num_envs, seed=args.seed, env_kwargs=args.env_kwargs,
                           wrapper_class=wrapper_class, vec_env_cls=vec_env_cls)
        if args.evaluate:
            eval_env_kwargs = args.env_kwargs.copy()
            eval_env = make_vec_env(args.env_name, n_envs=1, env_kwargs=eval_env_kwargs, wrapper_class=wrapper_class,
                                    vec_env_cls=vec_env_cls)
    elif args.env_type == 'football':
        try:
            from env.football import FootballGymSB3
        except ModuleNotFoundError:
            print("Could not find gfootball! please run pip install gfootball")
        args.env_kwargs['env_name'] = args.env_name
        env = make_vec_env(FootballGymSB3, n_envs=args.num_envs, seed=args.seed, env_kwargs=args.env_kwargs)
        if args.evaluate:
            eval_env = make_vec_env(FootballGymSB3, n_envs=1, env_kwargs=args.env_kwargs)
    elif args.env_type in ['mujoco', 'gym']:
        env = make_vec_env(args.env_name, n_envs=args.num_envs, seed=args.seed, env_kwargs=args.env_kwargs)
        if args.evaluate:
            eval_env = make_vec_env(args.env_name, n_envs=1, env_kwargs=args.env_kwargs)
    elif args.env_type == 'equation':
        register_equation_tests()
        vec_env_cls = DummyVecEnv
        env = make_vec_env(args.env_name, n_envs=args.num_envs, seed=args.seed, env_kwargs=args.env_kwargs,
                           vec_env_cls=vec_env_cls)
        if args.evaluate:
            eval_env = make_vec_env(args.env_name, n_envs=1, env_kwargs=args.env_kwargs, vec_env_cls=vec_env_cls)
    elif args.env_type == 'rickety_bridge':
        register_rickety_bridge_tests()
        vec_env_cls = DummyVecEnv
        env = make_vec_env(args.env_name, n_envs=args.num_envs, seed=args.seed, env_kwargs=args.env_kwargs,
                           vec_env_cls=vec_env_cls)
        if args.evaluate:
            eval_env = make_vec_env(args.env_name, n_envs=1, env_kwargs=args.env_kwargs, vec_env_cls=vec_env_cls)
    else:
        print("Invalid env_type!")
    if args.wrapper == 'normalize':
        args.wrapper_kwargs['gamma'] = args.gamma
        env = VecNormalize(env, **args.wrapper_kwargs)
        if eval_env is not None:
            args.wrapper_kwargs['training'] = False
            args.wrapper_kwargs['norm_reward'] = False
            eval_env = VecNormalize(eval_env, **args.wrapper_kwargs)
    if args.save_every and args.save_every > 0 and args.specific_seed == args.seed:
        callback_list.append(SLURMCheckpointCallback(save_freq=int(args.save_every / args.num_envs),
                                                save_path=os.path.join(args.save_path,
                                                                       f'{args.env_type}/'
                                                                       f'{args.env_name}/{args.algo_type}'),
                                                name_prefix=f'{args.save_name}_seed_{args.seed}',
                                                verbose=1,
                                                save_vecnormalize=True if args.env_type != 'football' else False,
                                                wandb_run=wandb_run))
    if args.no_improvement_kwargs:
        callback_list.append(StopTrainingOnNoImprovementInTraining(**args.no_improvement_kwargs, verbose=args.verbose))
    if eval_env is not None:
        stop_train_callback = None
        if args.eval_kwargs and args.eval_kwargs.get('stop_train', False):
            stop_train_callback = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=args.eval_kwargs.get('max_no_improvement_evals', 3),
                min_evals=args.eval_kwargs.get('min_evals', 5),
                verbose=1)
        if args.eval_kwargs.get('record', False):
            video_path = ROOT_PATH / f'videos/{args.env_type}/{args.env_name}/{args.algo_type}'
            if not os.path.exists(video_path):
                os.makedirs(video_path, exist_ok=True)

            eval_env = VecVideoRecorder(eval_env,
                                        video_folder=str(video_path),
                                        # record_video_trigger=lambda x: x == int(args.eval_kwargs['eval_freq'] / args.num_envs),
                                        record_video_trigger=lambda x: x == args.eval_kwargs['eval_freq'],
                                        name_prefix=f'{args.save_name}_seed_{args.seed}_eval',
                                        video_length=args.eval_kwargs.get('video_length', 2000))

        callback_list.append(EvalCallback(
                                    eval_env,
                                    callback_on_new_best=None,
                                    callback_after_eval=stop_train_callback,
                                    best_model_save_path=None,
                                    log_path=None,
                                    deterministic=args.eval_kwargs.get('deterministic', True),
                                    eval_freq=int(args.eval_kwargs.get('eval_freq', 10000) / args.num_envs),
                                    n_eval_episodes=args.eval_kwargs.get('n_eval_episodes', 5),
                                    verbose=args.eval_kwargs.get('verbose', 1),
                                    ))
    callback = None
    if callback_list:
        callback = CallbackList(callback_list)
    set_seed(args.seed)

    algo_kwargs = process_policy_kwargs(args)
    print(f"Training with algo_kwargs: {algo_kwargs}")

    algo = NAME_TO_ALGO[args.algo_type](env=env, tensorboard_log=tensorboard_log, _init_setup_model=True, **algo_kwargs)

    algo.learn(total_timesteps=args.total_n_steps, callback=callback, log_interval=args.log_interval,
               progress_bar=False, **learn_kwargs)

    if args.save_every > 0:
        print("End of training save")
        name_prefix = f'{args.env_type}/{args.env_name}/{args.algo_type}'
        model_path = os.path.join(args.save_path,
                                  f"{name_prefix}/{args.save_name}_seed_{args.seed}_{args.total_n_steps}_steps.zip")
        algo.save(model_path)
        if args.verbose >= 2:
            print(f"Saving model checkpoint to {model_path}")
        save_vec_normalize = True if args.env_type != 'football' else False
        if save_vec_normalize and algo.get_vec_normalize_env() is not None:
            # Save the VecNormalize statistics
            vec_normalize_path = os.path.join(args.save_path,
                                              f"{name_prefix}_vecnormalize_{args.total_n_steps}_steps.pkl")
            algo.get_vec_normalize_env().save(vec_normalize_path)
            if args.verbose >= 2:
                print(f"Saving model VecNormalize to {vec_normalize_path}")

    AutoResume.stop_resuming()
    # wandb_run_id_file = Path(ckpt_dir, "wandb_run_id.json")
    # if wandb_run_id_file.exists():
    #     wandb_run_id_file.unlink()
    print("Training completed successfully.")
