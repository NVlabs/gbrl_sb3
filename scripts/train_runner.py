#!/usr/bin/env python3.10
"""Training runner with checkpointing, wandb integration, and optional SLURM auto-resume.

Wraps scripts/train.py to add:
  - **Checkpoint & resume**: saves model + args on interruption, reloads on restart.
  - **wandb sweep agent**: replaces ``wandb agent <sweep_id>`` in the terminal
    with an in-process agent that integrates with checkpointing.
  - **SLURM auto-resume** (optional): on ADLR clusters with ``exec_and_report``,
    preempted jobs are automatically resubmitted.

Adapted from d3rlpy_gbrl/scripts/train_runner.py for the SB3-based GBRL repo.

Operating modes
---------------
1. **Direct run** (no ``SWEEP_ID`` env var) — works on any machine::

       python scripts/train_runner.py --env_type minigrid --algo_type split_rl --env_name MiniGrid-Corner-v0 --seed 0

   Always checkpoint-enabled.  On interrupt (Ctrl-C, SIGTERM, SLURM preemption),
   the model is saved.  Re-running the exact same command auto-detects the
   checkpoint and resumes from where it left off, including reconnecting to the
   same wandb run.

2. **Continuous sweep** (``SWEEP_ID`` set, no ``AUTORESUME``) — works on any machine::

       export SWEEP_ID=<id> WANDB_PROJECT=gbrl-sb3 WANDB_ENTITY=<entity>
       python scripts/train_runner.py

   Equivalent to ``wandb agent <sweep_id>``.  Runs trials back-to-back.
   No checkpointing — if killed mid-trial, that trial is marked crashed.

3. **Auto-resume sweep** (``SWEEP_ID`` + ``AUTORESUME=1``)::

       export AUTORESUME=1 SWEEP_ID=<id> WANDB_PROJECT=gbrl-sb3 WANDB_ENTITY=<entity>
       python scripts/train_runner.py

   One trial per invocation (``count=1``).  On interrupt, saves checkpoint.
   With ADLR: auto-resubmits.  Without ADLR: re-run manually.

4. **Cross-machine resume** (``--resume_dir``)::

       python scripts/train_runner.py --resume_dir /path/to/runs/my_run_dir

Checkpoint directory layout
---------------------------
::

    <log_dir>/
    ├── checkpoint.zip               # SB3 model save (algo.save())
    ├── checkpoint_args.json          # Training args snapshot + _checkpoint_step
    ├── checkpoint_vecnormalize.pkl   # VecNormalize stats (if applicable)
    └── wandb_run_id.txt              # wandb run ID for reconnection

Key differences from d3rlpy_gbrl's train_runner.py:
  - SB3 uses ``algo.save()`` / ``AlgoClass.load()`` (not d3rlpy's format)
  - SB3 callbacks (CheckpointCallback) used for periodic saves
  - SB3 uses ``algo.learn(total_timesteps=N)`` with internal step counter
  - Checkpoint extension is ``.zip`` (not ``.d3``)
  - Single training function that dispatches on env_type/algo_type
"""

import json
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

import wandb

# ── wandb compatibility patch ────────────────────────────────────────────────
if not hasattr(wandb, 'START_TIME'):
    wandb.START_TIME = time.time()

import inspect as _inspect
_HAS_RESUME_FROM = 'resume_from' in _inspect.signature(wandb.init).parameters

# ── Optional ADLR AutoResume ────────────────────────────────────────────────
try:
    sys.path.append(os.environ.get("SUBMIT_SCRIPTS", "."))
    from userlib.auto_resume import AutoResume
    HAS_ADLR_AUTORESUME = True
except (ImportError, ModuleNotFoundError):
    HAS_ADLR_AUTORESUME = False

ROOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_PATH))


# ── Local resume state (non-ADLR fallback) ──────────────────────────────────

def _resume_state_path(sweep_id=None):
    if sweep_id:
        safe = sweep_id.replace("/", "_")
        return ROOT_PATH / f".resume_state_{safe}.json"
    return ROOT_PATH / ".resume_state.json"


def _save_local_resume(details, sweep_id=None):
    path = _resume_state_path(sweep_id)
    path.write_text(json.dumps(details))


def _load_local_resume(sweep_id=None):
    path = _resume_state_path(sweep_id)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _clear_local_resume(sweep_id=None):
    path = _resume_state_path(sweep_id)
    if path.exists():
        path.unlink(missing_ok=True)


# ── Signal handling ──────────────────────────────────────────────────────────
_signal_received = False


def _preemption_signal_handler(signum, frame):
    global _signal_received
    _signal_received = True
    sig_name = signal.Signals(signum).name
    print(f"\nReceived {sig_name} (signal {signum}), "
          f"will save checkpoint at next callback invocation...")


def _install_signal_handlers():
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _preemption_signal_handler)
        signal.signal(signal.SIGUSR1, _preemption_signal_handler)
        signal.signal(signal.SIGINT, _preemption_signal_handler)


def handle_preemption(resume_details, run, sweep_id=None):
    """Save resume state and exit."""
    print("Interrupt detected, saving state for resume...")
    if HAS_ADLR_AUTORESUME:
        try:
            AutoResume.request_resume(user_dict=resume_details)
            print("ADLR AutoResume: requested job resubmission.")
        except Exception as e:
            print(f"Warning: AutoResume.request_resume failed: {e}")
    _save_local_resume(resume_details, sweep_id=sweep_id)
    if run is not None:
        print(f"Leaving wandb run {run.id} open (will show as crashed until resumed).")
    os._exit(0)


# ── SB3 Preemption Callback ─────────────────────────────────────────────────

from stable_baselines3.common.callbacks import BaseCallback


class PreemptionCheckpointCallback(BaseCallback):
    """SB3 callback that checks for preemption signals and saves checkpoints.

    Replaces the manual step_callback from d3rlpy's train_runner.
    Integrates with ADLR AutoResume and Unix signal handlers.
    """

    def __init__(
        self,
        resume_details: dict,
        checkpoint_dir: str,
        args,
        wandb_run,
        use_adlr: bool = False,
        checkpoint_freq: int = 0,
        sweep_id: str = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.resume_details = resume_details
        self.checkpoint_dir = Path(checkpoint_dir)
        self.args = args
        self.wandb_run = wandb_run
        self.use_adlr = use_adlr
        self.checkpoint_freq = checkpoint_freq
        self.sweep_id = sweep_id
        self._last_ckpt_step = 0

    def _save_checkpoint(self):
        ckpt_path = self.checkpoint_dir / "checkpoint.zip"
        self.model.save(str(ckpt_path))
        step = self.num_timesteps
        print(f"Saved checkpoint to {ckpt_path} (step {step})")

        # Save VecNormalize if present
        vec_norm = self.model.get_vec_normalize_env()
        if vec_norm is not None:
            vec_norm_path = self.checkpoint_dir / "checkpoint_vecnormalize.pkl"
            vec_norm.save(str(vec_norm_path))

        # Save args + step
        args_dict = {k: v for k, v in vars(self.args).items()
                     if not k.startswith('_') and _is_json_serializable(v)}
        args_dict['_checkpoint_step'] = step
        args_path = self.checkpoint_dir / "checkpoint_args.json"
        args_path.write_text(json.dumps(args_dict, default=str))

    def _on_step(self) -> bool:
        should_stop = _signal_received
        if self.use_adlr and HAS_ADLR_AUTORESUME:
            should_stop = should_stop or AutoResume.termination_requested()

        if should_stop:
            self._save_checkpoint()
            handle_preemption(self.resume_details, self.wandb_run,
                              sweep_id=self.sweep_id)
            return False  # handle_preemption calls os._exit, but just in case

        # Periodic checkpoint for crash resilience
        if (self.checkpoint_freq > 0 and
                self.num_timesteps - self._last_ckpt_step >= self.checkpoint_freq):
            self._save_checkpoint()
            self._last_ckpt_step = self.num_timesteps

        return True


def _is_json_serializable(v):
    """Check if a value can be JSON serialized."""
    try:
        json.dumps(v, default=str)
        return True
    except (TypeError, ValueError):
        return False


# ── wandb config → argparse bridge ──────────────────────────────────────────

def _inject_args_from_wandb_config():
    """Bridge wandb sweep config → argparse by rebuilding sys.argv."""
    wandb.init(sync_tensorboard=True, monitor_gym=True)
    config = dict(wandb.config)

    algo = config.get('algo_type', '')
    env = config.get('env_name', '')
    seed = config.get('seed', 0)
    wandb.run.name = f"{algo}_{env}_s{seed}"

    sys.argv = [sys.argv[0]]
    for key, val in config.items():
        if key.startswith('_'):
            continue
        if val is None:
            continue
        if isinstance(val, list):
            sys.argv.append(f'--{key}')
            sys.argv.extend(str(v) for v in val)
        else:
            sys.argv.extend([f'--{key}', str(val)])
    print(f"Injected {len(config)} sweep params into sys.argv")


def _inject_args_from_dict(saved):
    """Rebuild sys.argv from a saved args dict (checkpoint_args.json)."""
    sys.argv = [sys.argv[0]]
    for key, val in saved.items():
        if key.startswith('_'):
            continue
        if val is None:
            continue
        if isinstance(val, list):
            sys.argv.append(f"--{key}")
            sys.argv.extend(str(v) for v in val)
        else:
            sys.argv.extend([f"--{key}", str(val)])


# ── Build env (extracted from train.py) ──────────────────────────────────────

def _build_env(args):
    """Create training and eval environments from args.

    This mirrors the env-creation logic from scripts/train.py.
    """
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import (DummyVecEnv, VecFrameStack,
                                                  VecNormalize, VecVideoRecorder)
    from env.wrappers import (CategoricalDummyVecEnv,
                              MiniGridCategoricalObservationWrapper)
    from env.mujoco_wrappers import MujocoTreeObsWrapper
    from env.flatland import make_flatland_vec_env
    from env.highway import make_highway_vec_env
    from env.sumo import make_sumo_vec_env
    from utils.helpers import make_ram_atari_env, make_cost_vec_env, make_minigrid_vec_env
    from config.args import SAFETY_ENVS

    CATEGORICAL_ALGOS = [a for a in ['ppo_gbrl', 'a2c_gbrl', 'sac_gbrl', 'awr_gbrl',
                                      'dqn_gbrl', 'split_rl'] if 'gbrl' in a or a == 'split_rl']

    if args.env_kwargs is None:
        args.env_kwargs = {}

    env, eval_env = None, None

    if 'atari' in args.env_type:
        env_kwargs = {'full_action_space': False}
        env = make_ram_atari_env(args.env_name, n_envs=args.num_envs, seed=args.seed,
                                 wrapper_kwargs=args.atari_wrapper_kwargs, env_kwargs=env_kwargs)
        if args.evaluate:
            eval_env = make_ram_atari_env(args.env_name, n_envs=1,
                                          wrapper_kwargs=args.atari_wrapper_kwargs, env_kwargs=env_kwargs)
        if args.atari_wrapper_kwargs and 'frame_stack' in args.atari_wrapper_kwargs:
            env = VecFrameStack(env, n_stack=args.atari_wrapper_kwargs['frame_stack'])
            if eval_env:
                eval_env = VecFrameStack(eval_env, n_stack=args.atari_wrapper_kwargs['frame_stack'])

    elif args.env_type == 'minigrid':
        from env.register_minigrid import register_minigrid_tests
        register_minigrid_tests()
        from minigrid.wrappers import FlatObsWrapper
        wrapper_class = MiniGridCategoricalObservationWrapper if args.algo_type in CATEGORICAL_ALGOS else FlatObsWrapper
        vec_env_cls = CategoricalDummyVecEnv if args.algo_type in CATEGORICAL_ALGOS else DummyVecEnv
        vec_env_fnc = make_cost_vec_env if args.env_name in SAFETY_ENVS else make_minigrid_vec_env
        env = vec_env_fnc(args.env_name, n_envs=args.num_envs, seed=args.seed,
                          env_kwargs=args.env_kwargs, wrapper_class=wrapper_class, vec_env_cls=vec_env_cls)
        if args.evaluate:
            eval_env = vec_env_fnc(args.env_name, n_envs=1, env_kwargs=args.env_kwargs.copy(),
                                   wrapper_class=wrapper_class, vec_env_cls=vec_env_cls)

    elif args.env_type == 'football':
        try:
            from env.football import FootballGymSB3
        except ModuleNotFoundError:
            raise RuntimeError("Could not find gfootball! please run pip install gfootball")
        args.env_kwargs['env_name'] = args.env_name
        env = make_vec_env(FootballGymSB3, n_envs=args.num_envs, seed=args.seed,
                           env_kwargs=args.env_kwargs)
        if args.evaluate:
            eval_env = make_vec_env(FootballGymSB3, n_envs=1, env_kwargs=args.env_kwargs)

    elif args.env_type in ['mujoco', 'gym']:
        mujoco_wrapper = None
        if args.mujoco_tree_obs and args.env_type == 'mujoco':
            mujoco_wrapper = lambda env: MujocoTreeObsWrapper(env, env_name=args.env_name)
        env = make_vec_env(args.env_name, n_envs=args.num_envs, seed=args.seed,
                           env_kwargs=args.env_kwargs, wrapper_class=mujoco_wrapper)
        if args.evaluate:
            eval_env = make_vec_env(args.env_name, n_envs=1, env_kwargs=args.env_kwargs,
                                   wrapper_class=mujoco_wrapper)

    elif args.env_type == 'highway':
        highway_kwargs = args.env_kwargs or {}
        env = make_highway_vec_env(env_name=args.env_name, n_envs=args.num_envs,
                                   seed=args.seed, **highway_kwargs)
        if args.evaluate:
            eval_env = make_highway_vec_env(env_name=args.env_name, n_envs=1,
                                            seed=args.seed, **highway_kwargs)

    elif args.env_type == 'equation':
        from env.equation import register_equation_tests
        register_equation_tests()
        env = make_vec_env(args.env_name, n_envs=args.num_envs, seed=args.seed,
                           env_kwargs=args.env_kwargs, vec_env_cls=DummyVecEnv)
        if args.evaluate:
            eval_env = make_vec_env(args.env_name, n_envs=1, env_kwargs=args.env_kwargs,
                                   vec_env_cls=DummyVecEnv)

    elif args.env_type == 'flatland':
        flatland_kwargs = args.env_kwargs or {}
        env = make_flatland_vec_env(env_name=args.env_name, n_envs=args.num_envs,
                                    seed=args.seed, **flatland_kwargs)
        if args.evaluate:
            eval_env = make_flatland_vec_env(env_name=args.env_name, n_envs=1,
                                            seed=args.seed, **flatland_kwargs)

    elif args.env_type == 'sumo':
        sumo_kwargs = args.env_kwargs or {}
        env = make_sumo_vec_env(env_name=args.env_name, n_envs=args.num_envs,
                                seed=args.seed, **sumo_kwargs)
        if args.evaluate:
            eval_env = make_sumo_vec_env(env_name=args.env_name, n_envs=1,
                                        seed=args.seed, **sumo_kwargs)
    else:
        raise ValueError(f"Invalid env_type: {args.env_type}")

    if args.wrapper == 'normalize':
        args.wrapper_kwargs['gamma'] = args.gamma
        env = VecNormalize(env, **args.wrapper_kwargs)

    return env, eval_env


# ── Build callbacks (extracted from train.py) ────────────────────────────────

def _build_callbacks(args, eval_env):
    """Create the SB3 callback list from args, mirroring scripts/train.py."""
    from stable_baselines3.common.callbacks import (
        CallbackList, CheckpointCallback, EvalCallback,
        StopTrainingOnNoModelImprovement)
    from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder
    from callback.callbacks import (OffPolicyDistillationCallback,
                                    OnPolicyDistillationCallback,
                                    StopTrainingOnNoImprovementInTraining,
                                    SafetyEvalCallback)
    from config.args import SAFETY_ENVS

    ON_POLICY_ALGOS = ['ppo_gbrl', 'a2c_gbrl', 'split_rl']
    OFF_POLICY_ALGOS = ['sac_gbrl', 'dqn_gbrl', 'awr_gbrl']

    callback_list = []

    if args.distil and args.distil_kwargs:
        if args.algo_type in ON_POLICY_ALGOS:
            callback_list.append(OnPolicyDistillationCallback(
                args.distil_kwargs, args.distil_kwargs.get('distil_verbose', 0)))
        elif args.algo_type in OFF_POLICY_ALGOS:
            callback_list.append(OffPolicyDistillationCallback(
                args.distil_kwargs, args.distil_kwargs.get('distil_verbose', 0)))

    if args.save_every and args.save_every > 0 and args.specific_seed == args.seed:
        callback_list.append(CheckpointCallback(
            save_freq=int(args.save_every / args.num_envs),
            save_path=os.path.join(args.save_path,
                                   f'{args.env_type}/{args.env_name}/{args.algo_type}'),
            name_prefix=f'{args.save_name}_seed_{args.seed}',
            verbose=1,
            save_vecnormalize=True if args.env_type != 'football' else False))

    if args.no_improvement_kwargs:
        callback_list.append(StopTrainingOnNoImprovementInTraining(
            **args.no_improvement_kwargs, verbose=args.verbose))

    if eval_env is not None:
        stop_train_callback = None
        if args.eval_kwargs and args.eval_kwargs.get('stop_train', False):
            stop_train_callback = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=args.eval_kwargs.get('max_no_improvement_evals', 3),
                min_evals=args.eval_kwargs.get('min_evals', 5),
                verbose=1)

        if args.eval_kwargs.get('record', False):
            video_path = ROOT_PATH / f'videos/{args.env_type}/{args.env_name}/{args.algo_type}'
            os.makedirs(video_path, exist_ok=True)
            video_length = args.eval_kwargs.get('video_length', 200)
            video_freq = args.eval_kwargs.get('video_freq', 100000)
            # Flag-based trigger: armed by the eval callback every video_freq
            # training steps.  Fires at most once per eval round.
            def _video_trigger(step):
                if getattr(_video_trigger, 'armed', False):
                    _video_trigger.armed = False
                    return True
                return False
            _video_trigger.armed = True           # record first eval round
            _video_trigger.video_freq = video_freq
            _video_trigger.last_record_timestep = 0
            eval_env = VecVideoRecorder(
                eval_env,
                video_folder=str(video_path),
                record_video_trigger=_video_trigger,
                name_prefix=f'{args.save_name}_seed_{args.seed}_eval',
                video_length=video_length)

        # Apply VecNormalize AFTER VecVideoRecorder
        if args.wrapper == 'normalize':
            eval_norm_kwargs = dict(args.wrapper_kwargs)
            eval_norm_kwargs['training'] = False
            eval_norm_kwargs['norm_reward'] = False
            eval_env = VecNormalize(eval_env, **eval_norm_kwargs)

        eval_callback_cls = SafetyEvalCallback
        callback_list.append(eval_callback_cls(
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

    return callback_list, eval_env


# ── Algorithm dispatch ───────────────────────────────────────────────────────

def _get_algo_class(algo_type: str):
    """Return the SB3 algorithm class for the given algo_type string."""
    from stable_baselines3.a2c.a2c import A2C
    from stable_baselines3.dqn.dqn import DQN
    from algos.a2c import A2C_GBRL
    from algos.awr import AWR_GBRL
    from algos.awr_nn import AWR
    from algos.safety import PPOLag, CPO, CUP, IPO
    from algos.dqn import DQN_GBRL
    from algos.ppo import PPO_GBRL
    from algos.sac import SAC_GBRL
    from algos.split_rl import SPLIT_RL
    from algos.safety.ppo import VanillaPPO as PPO

    NAME_TO_ALGO = {
        'ppo_gbrl': PPO_GBRL, 'split_rl': SPLIT_RL,
        'a2c_gbrl': A2C_GBRL, 'sac_gbrl': SAC_GBRL,
        'awr_gbrl': AWR_GBRL, 'ppo_nn': PPO,
        'ppo_lag': PPOLag, 'cpo': CPO, 'cup': CUP, 'ipo': IPO,
        'a2c_nn': A2C, 'dqn_gbrl': DQN_GBRL,
        'awr_nn': AWR, 'dqn_nn': DQN,
    }
    if algo_type not in NAME_TO_ALGO:
        raise ValueError(f"Unknown algo_type: {algo_type}. Choose from {list(NAME_TO_ALGO.keys())}")
    return NAME_TO_ALGO[algo_type]


# ============================================================================
# Main training function
# ============================================================================

def train_runner():
    """Run training with checkpoint/resume, wandb integration, and optional ADLR auto-resume."""
    from config.args import parse_args, process_logging, process_policy_kwargs
    from utils.helpers import set_seed

    args = parse_args()

    # ── Add --resume_dir support to argparse ─────────────────────────────
    # parse_args doesn't know about --resume_dir, so we extract it manually
    resume_dir = _pop_extra_arg('--resume_dir')

    sweep_id_env = os.getenv("SWEEP_ID")
    no_resume = bool(os.getenv("NO_RESUME"))
    use_checkpointing = (not no_resume) and (not sweep_id_env or bool(os.getenv("AUTORESUME")))

    # ── Env cleanup ──────────────────────────────────────────────────────
    os.environ.pop('MASTER_ADDR', None)
    os.environ.pop('MASTER_PORT', None)

    # ── Resume detection ─────────────────────────────────────────────────
    auto_resume_details = None
    if use_checkpointing:
        _install_signal_handlers()

        if resume_dir:
            resume_dir_path = Path(resume_dir)
            auto_resume_details = {'LOG_DIR': str(resume_dir_path)}
            wid_file = resume_dir_path / "wandb_run_id.txt"
            if wid_file.exists():
                auto_resume_details['WANDB_RUN_ID'] = wid_file.read_text().strip()
            print(f"Resuming from --resume_dir: {auto_resume_details}")
        elif HAS_ADLR_AUTORESUME:
            print("Setting up ADLR auto-resume")
            AutoResume.init()
            if AutoResume.termination_requested():
                print("AutoResume timer already expired at startup — exiting.")
                sys.exit(0)
            auto_resume_details = AutoResume.get_resume_details()
        elif bool(os.getenv("AUTORESUME")):
            # Only auto-detect local resume state when AUTORESUME is explicitly set.
            # Without it, checkpoints are still saved on interrupt but won't
            # auto-resume on the next run.  Use --resume_dir for manual resume.
            auto_resume_details = _load_local_resume(sweep_id=sweep_id_env)
            if auto_resume_details:
                print(f"Found local resume state: {auto_resume_details}")

    # ── Log directory ────────────────────────────────────────────────────
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    wandb_run_id = os.environ.get('WANDB_RUN_ID')
    if wandb_run_id is None and wandb.run is not None:
        wandb_run_id = wandb.run.id
    is_sweep_run = wandb_run_id is not None

    if auto_resume_details and 'LOG_DIR' in auto_resume_details:
        log_dir = auto_resume_details['LOG_DIR']
        saved_run_id = auto_resume_details.get('WANDB_RUN_ID')
        if saved_run_id:
            wandb_run_id = saved_run_id
            os.environ['WANDB_RUN_ID'] = saved_run_id
        print(f"Resuming with log_dir from previous run: {log_dir}")
    else:
        env_name = getattr(args, 'env_name', 'unknown')
        short_env = env_name.replace('NoFrameskip-v4', '').replace('MiniGrid-', '').lower()
        # Allow environment override for sweeps where sys.argv is rebuilt from wandb config.
        base_log_dir = os.getenv('LOG_DIR') or os.getenv('TRAIN_LOG_DIR') or getattr(args, 'log_dir', None) or 'runs'
        if is_sweep_run:
            log_dir = str(Path(base_log_dir) / f"{short_env}_{args.algo_type}_s{args.seed}_{job_id}_{wandb_run_id}")
        else:
            log_dir = str(Path(base_log_dir) / f"{short_env}_{args.algo_type}_s{args.seed}_{job_id}")
        print(f"First run, log_dir: {log_dir}")

    log_dir_path = Path(log_dir).resolve()
    log_dir = str(log_dir_path)
    log_dir_path.mkdir(exist_ok=True, parents=True)
    args.log_dir = log_dir

    # ── Checkpoint detection ─────────────────────────────────────────────
    checkpoint_path = log_dir_path / "checkpoint.zip"
    args_path = log_dir_path / "checkpoint_args.json"
    is_resume = use_checkpointing and checkpoint_path.exists()
    checkpoint_step = None

    if is_resume:
        print(f"Found checkpoint: {checkpoint_path}")
        if args_path.exists():
            saved_args = json.loads(args_path.read_text())
            checkpoint_step = saved_args.pop('_checkpoint_step', None)
            # Restore key args from checkpoint
            for key, val in saved_args.items():
                if hasattr(args, key) and not key.startswith('_'):
                    setattr(args, key, val)
            print(f"Restored original args: seed={args.seed}, env={args.env_name}, algo={args.algo_type}")
            if checkpoint_step is not None:
                print(f"Checkpoint was at step {checkpoint_step}")

    # ── Resume details (for preemption callback) ─────────────────────────
    resume_details = {}
    if use_checkpointing:
        resume_details = {'LOG_DIR': log_dir}
        if wandb_run_id:
            resume_details['WANDB_RUN_ID'] = wandb_run_id
        if HAS_ADLR_AUTORESUME:
            AutoResume.request_resume(user_dict=resume_details)
        _save_local_resume(resume_details, sweep_id=sweep_id_env)

    # ── Logging setup ────────────────────────────────────────────────────
    callback_list = []

    # wandb resume support
    wandb_run_id_file = log_dir_path / "wandb_run_id.txt"
    effective_wandb_id = wandb_run_id
    if wandb_run_id_file.exists():
        effective_wandb_id = wandb_run_id_file.read_text().strip()
        print(f"Found existing wandb run ID from file: {effective_wandb_id}")
        if is_resume:
            args.wandb = True

    use_wandb = bool(args.wandb) if isinstance(args.wandb, bool) else str(args.wandb).lower() in ("true", "1", "yes")
    wandb_run = None
    tensorboard_log = None

    if use_wandb:
        # Determine resume mode
        if is_resume and effective_wandb_id and checkpoint_step is not None and _HAS_RESUME_FROM:
            resume_mode = None
            resume_from_str = f"{effective_wandb_id}?_step={checkpoint_step}"
        elif is_resume and effective_wandb_id:
            resume_mode = "must"
            resume_from_str = None
        elif effective_wandb_id or is_sweep_run:
            resume_mode = "allow"
            resume_from_str = None
        else:
            resume_mode = None
            resume_from_str = None

        if wandb.run is not None and (
            not effective_wandb_id or wandb.run.id == effective_wandb_id
        ):
            wandb_run = wandb.run
            wandb_run.name = (getattr(args, 'run_name', None) or 'default') + \
                f'_{args.algo_type}_{args.env_type}_{args.env_name}_seed_{args.seed}'
            wandb_run.config.update(
                {k: v for k, v in vars(args).items() if _is_json_serializable(v)},
                allow_val_change=True)
        else:
            if wandb.run is not None:
                wandb.finish(quiet=True)

            group_name = None
            if getattr(args, 'group_name', None):
                group_name = f"{args.group_name}_{args.algo_type}_{args.env_type}_{args.env_name}"

            run_name = (getattr(args, 'run_name', None) or 'default') + \
                f'_{args.algo_type}_{args.env_type}_{args.env_name}_seed_{args.seed}'

            init_kwargs = dict(
                project=getattr(args, 'project', 'gbrl-sb3'),
                entity=getattr(args, 'entity', None),
                group=group_name,
                name=run_name,
                config={k: v for k, v in vars(args).items() if _is_json_serializable(v)},
                id=effective_wandb_id,
                resume=resume_mode,
                save_code=False,
                monitor_gym=True,
                sync_tensorboard=True,
            )
            if resume_from_str is not None:
                init_kwargs["resume_from"] = resume_from_str
            wandb_run = wandb.init(**init_kwargs)

        if not wandb_run_id_file.exists():
            wandb_run_id_file.write_text(wandb_run.id)

        from wandb.integration.sb3 import WandbCallback
        callback_list.append(WandbCallback(gradient_save_freq=0, model_save_path=None, verbose=1))

        tensorboard_log = str(log_dir_path / "tensorboard")
    else:
        tensorboard_log = str(log_dir_path / "tensorboard")

    # ── Build env ────────────────────────────────────────────────────────
    env, eval_env = _build_env(args)

    # ── Build standard callbacks ─────────────────────────────────────────
    std_callbacks, eval_env = _build_callbacks(args, eval_env)
    callback_list.extend(std_callbacks)

    # ── Add preemption checkpoint callback ───────────────────────────────
    if use_checkpointing:
        checkpoint_freq = 0
        if args.eval_kwargs:
            checkpoint_freq = args.eval_kwargs.get('eval_freq', 50000)
        elif args.save_every and args.save_every > 0:
            checkpoint_freq = args.save_every

        callback_list.append(PreemptionCheckpointCallback(
            resume_details=resume_details,
            checkpoint_dir=str(log_dir_path),
            args=args,
            wandb_run=wandb_run,
            use_adlr=HAS_ADLR_AUTORESUME,
            checkpoint_freq=checkpoint_freq,
            sweep_id=sweep_id_env,
        ))

    # ── Build algorithm ──────────────────────────────────────────────────
    set_seed(args.seed)
    algo_kwargs = process_policy_kwargs(args)
    AlgoClass = _get_algo_class(args.algo_type)

    remaining_steps = args.total_n_steps
    if is_resume:
        print(f"Loading model from checkpoint: {checkpoint_path}")
        algo = AlgoClass.load(
            str(checkpoint_path),
            env=env,
            device=args.device,
            tensorboard_log=tensorboard_log,
        )
        if checkpoint_step is not None:
            remaining_steps = max(0, args.total_n_steps - checkpoint_step)
        print(f"Resumed at step {checkpoint_step}, remaining steps: {remaining_steps}")

        # Restore VecNormalize if present
        vec_norm_path = log_dir_path / "checkpoint_vecnormalize.pkl"
        if vec_norm_path.exists():
            from stable_baselines3.common.vec_env import VecNormalize
            vec_norm = VecNormalize.load(str(vec_norm_path), env.unwrapped
                                         if hasattr(env, 'unwrapped') else env)
            algo.set_env(vec_norm)
            print(f"Restored VecNormalize from {vec_norm_path}")
    else:
        print(f"Training with algo_kwargs: {algo_kwargs}")
        algo = AlgoClass(env=env, tensorboard_log=tensorboard_log,
                         _init_setup_model=True, **algo_kwargs)

    # ── Train ────────────────────────────────────────────────────────────
    from stable_baselines3.common.callbacks import CallbackList
    callback = CallbackList(callback_list) if callback_list else None

    print(f"Starting training for {remaining_steps} steps...")
    algo.learn(
        total_timesteps=remaining_steps,
        callback=callback,
        log_interval=args.log_interval,
        progress_bar=False,
        reset_num_timesteps=not is_resume,
    )

    # ── Successful completion ────────────────────────────────────────────
    print("Training completed successfully.")

    # Final model save
    if args.save_every > 0:
        name_prefix = f'{args.env_type}/{args.env_name}/{args.algo_type}'
        model_path = os.path.join(args.save_path,
                                  f"{name_prefix}/{args.save_name}_seed_{args.seed}_{args.total_n_steps}_steps.zip")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        algo.save(model_path)
        print(f"Final model saved to {model_path}")
        if algo.get_vec_normalize_env() is not None:
            vec_normalize_path = os.path.join(
                args.save_path,
                f"{name_prefix}_vecnormalize_{args.total_n_steps}_steps.pkl")
            algo.get_vec_normalize_env().save(vec_normalize_path)

    # Clean up checkpoint artifacts
    if use_checkpointing:
        if HAS_ADLR_AUTORESUME:
            AutoResume.stop_resuming()
        _clear_local_resume(sweep_id=sweep_id_env)

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print("Checkpoint deleted after successful completion.")
        if args_path.exists():
            args_path.unlink()
        vec_norm_ckpt = log_dir_path / "checkpoint_vecnormalize.pkl"
        if vec_norm_ckpt.exists():
            vec_norm_ckpt.unlink()
        if wandb_run_id_file.exists():
            wandb_run_id_file.unlink()

    if wandb_run is not None:
        wandb_run.finish()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pop_extra_arg(flag):
    """Extract a flag+value pair from sys.argv that argparse doesn't know about."""
    for i, arg in enumerate(sys.argv):
        if arg == flag and i + 1 < len(sys.argv):
            val = sys.argv[i + 1]
            del sys.argv[i:i + 2]
            return val
    return None


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    SWEEP_ID = os.getenv("SWEEP_ID", None)
    use_autoresume = bool(os.getenv("AUTORESUME"))
    no_resume = bool(os.getenv("NO_RESUME"))

    # Clear stale resume state when NO_RESUME is set
    if no_resume and SWEEP_ID:
        _clear_local_resume(sweep_id=SWEEP_ID)

    if SWEEP_ID is None:
        # ── Direct run (no sweep) ───────────────────────────────────────
        train_runner()

    else:
        WANDB_PROJECT = os.getenv("WANDB_PROJECT", "gbrl-sb3")
        WANDB_ENTITY = os.getenv("WANDB_ENTITY", None)

        if not WANDB_ENTITY:
            raise RuntimeError("Set WANDB_ENTITY environment variable for sweeps.")

        sweep_id = f"{WANDB_ENTITY}/{WANDB_PROJECT}/{SWEEP_ID}"

        if not use_autoresume:
            # ── Continuous mode ──────────────────────────────────────────
            print(f"Launching wandb agent for sweep (continuous mode): {sweep_id}")

            def _sweep_train():
                try:
                    _inject_args_from_wandb_config()
                    train_runner()
                except SystemExit as e:
                    print(f"\n[train_runner] SystemExit({e.code}) in sweep trial",
                          file=sys.stderr, flush=True)
                    traceback.print_exc(file=sys.stderr)
                    raise
                except Exception as e:
                    print(f"\n[train_runner] Exception in sweep trial: {e}",
                          file=sys.stderr, flush=True)
                    traceback.print_exc(file=sys.stderr)
                    raise

            wandb.agent(sweep_id, function=_sweep_train, count=None)

        else:
            # ── Auto-resume mode ─────────────────────────────────────────
            if HAS_ADLR_AUTORESUME:
                AutoResume.init()
                _resume_details = AutoResume.get_resume_details() or {}
            else:
                _resume_details = _load_local_resume(sweep_id=SWEEP_ID) or {}

            has_unfinished_trial = bool(_resume_details.get("LOG_DIR"))

            if has_unfinished_trial:
                log_dir_env = _resume_details['LOG_DIR']
                print(f"Resuming preempted trial (LOG_DIR={log_dir_env})")

                args_file = Path(log_dir_env) / "checkpoint_args.json"
                if args_file.exists():
                    saved = json.loads(args_file.read_text())
                    _inject_args_from_dict(saved)
                    print("Injected args from checkpoint_args.json")
                else:
                    print(f"No checkpoint_args.json at {log_dir_env}, "
                          "falling through to wandb agent for a fresh trial.")
                    has_unfinished_trial = False

                os.environ.pop("SWEEP_ID", None)
                if has_unfinished_trial:
                    train_runner()

                    # Trial completed — try next trial if time permits
                    for key in [k for k in os.environ
                                if k.startswith("AUTO_RESUME") or k in ("LOG_DIR", "WANDB_RUN_ID")]:
                        os.environ.pop(key, None)
                    _clear_local_resume(sweep_id=SWEEP_ID)

                    if HAS_ADLR_AUTORESUME:
                        AutoResume.init()
                        if AutoResume.termination_requested():
                            print("No SLURM time remaining for another trial.")
                            sys.exit(0)

                    print("Launching wandb agent for next sweep trial...")
                    _install_signal_handlers()
                    if HAS_ADLR_AUTORESUME:
                        AutoResume.request_resume(user_dict={})

                    def _sweep_train_after_resume():
                        try:
                            _inject_args_from_wandb_config()
                            train_runner()
                        except SystemExit as e:
                            print(f"\n[train_runner] SystemExit({e.code}) in sweep trial",
                                  file=sys.stderr, flush=True)
                            traceback.print_exc(file=sys.stderr)
                            raise
                        except Exception as e:
                            print(f"\n[train_runner] Exception in sweep trial: {e}",
                                  file=sys.stderr, flush=True)
                            traceback.print_exc(file=sys.stderr)
                            raise
                    wandb.agent(sweep_id, function=_sweep_train_after_resume, count=1)
                else:
                    for key in [k for k in os.environ
                                if k.startswith("AUTO_RESUME") or k in ("LOG_DIR", "WANDB_RUN_ID")]:
                        os.environ.pop(key, None)
                    _clear_local_resume(sweep_id=SWEEP_ID)

                    print("Launching wandb agent for a fresh sweep trial...")
                    _install_signal_handlers()
                    if HAS_ADLR_AUTORESUME:
                        AutoResume.init()
                        AutoResume.request_resume(user_dict={})

                    def _sweep_train_fresh():
                        _inject_args_from_wandb_config()
                        train_runner()
                    wandb.agent(sweep_id, function=_sweep_train_fresh, count=1)

            else:
                # ── Fresh trial (no unfinished work) ─────────────────────
                print(f"Launching wandb agent for sweep (auto-resume mode): {sweep_id}")
                _install_signal_handlers()

                if HAS_ADLR_AUTORESUME:
                    AutoResume.request_resume(user_dict={})

                def _sweep_train():
                    _inject_args_from_wandb_config()
                    train_runner()
                wandb.agent(sweep_id, function=_sweep_train, count=1)
