#!/usr/bin/env python3.10
"""Generate imitation-learning datasets from trained PPO-GBRL MiniGrid experts.

Loads a trained PPO-GBRL model, rolls out episodes in the MiniGrid env,
and saves successful episodes as .npz files matching the d3rlpy_gbrl format.

Observations stored:
  - observations_image: (N, H, W, 3) uint8 — raw MiniGrid partial-view grid
  - observations_direction: (N,) int32 — agent compass direction (0-3)
  These can be transformed to either GBRL categorical or NN FlatObs downstream.

Also stores:
  - actions: (N,) int64 — discrete actions
  - rewards: (N,) float32 — per-step rewards
  - episode_terminals: (n_eps,) int64 — cumulative end indices
  - episode_returns: (n_eps,) float32 — per-episode return

Only **successful** episodes (terminated=True, truncated=False, reward > 0)
are kept, since we want expert demonstrations.

Usage:
  # Generate datasets for all 3 subtask envs:
  python3.10 scripts/generate_minigrid_dataset.py

  # Single env, custom params:
  python3.10 scripts/generate_minigrid_dataset.py \\
      --env_name MiniGrid-MoveBall-v0 \\
      --model_path saved_models/minigrid/MiniGrid-MoveBall-v0/ppo_gbrl/MiniGrid-MoveBall-v0_seed_0_1000000_steps.zip \\
      --min_episodes 500 --seed 0
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT_PATH = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_PATH))


# ============================================================================
# Environment creation
# ============================================================================

def make_eval_env(env_name: str, seed: int = 0):
    """Create a single MiniGrid env with the GBRL categorical wrapper.

    We need the categorical wrapper so PPO-GBRL can predict, but we
    also extract the raw obs from the underlying env for the dataset.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv
    from env.register_minigrid import register_minigrid_tests
    from env.wrappers import (CategoricalDummyVecEnv,
                              MiniGridCategoricalObservationWrapper)
    from utils.helpers import make_minigrid_vec_env

    register_minigrid_tests()

    env = make_minigrid_vec_env(
        env_name, n_envs=1, seed=seed,
        wrapper_class=MiniGridCategoricalObservationWrapper,
        vec_env_cls=CategoricalDummyVecEnv,
    )
    return env


def get_raw_obs(vec_env) -> Tuple[np.ndarray, int]:
    """Extract the raw MiniGrid image and direction from the underlying env."""
    base_env = vec_env.envs[0]
    # Unwrap through wrappers to reach MiniGridEnv
    env = base_env
    while hasattr(env, 'env'):
        env = env.env
    image = env.gen_obs()['image'].copy()
    direction = int(env.agent_dir)
    return image, direction


# ============================================================================
# Episode collection
# ============================================================================

def collect_expert_episodes(
    vec_env,
    model,
    min_episodes: int,
    max_episodes: int = 0,
    deterministic: bool = True,
    success_only: bool = True,
) -> Tuple[List[Dict], int, int]:
    """Collect episodes from a trained model.

    Returns:
        episodes: list of episode dicts (successful only if success_only=True)
        total_steps: total transitions across all kept episodes
        total_attempted: total episodes attempted (including failures)
    """
    episodes: List[Dict] = []
    total_steps = 0
    total_attempted = 0
    max_attempts = max_episodes if max_episodes > 0 else min_episodes * 20

    obs = vec_env.reset()

    current_images = []
    current_directions = []
    current_actions = []
    current_rewards = []

    # Record initial observation
    img, direction = get_raw_obs(vec_env)
    current_images.append(img)
    current_directions.append(direction)

    t0 = time.time()
    print(f"  Collecting expert episodes (target: {min_episodes})...")

    while len(episodes) < min_episodes and total_attempted < max_attempts:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = vec_env.step(action)

        current_actions.append(int(action[0]))
        current_rewards.append(float(reward[0]))

        if done[0]:
            total_attempted += 1
            ep_return = sum(current_rewards)

            # Check success: positive return and not truncated
            is_truncated = info[0].get('TimeLimit.truncated', False)
            is_success = (ep_return > 0) and (not is_truncated)

            if (not success_only) or is_success:
                episode = {
                    "observations_image": np.array(current_images, dtype=np.uint8),
                    "observations_direction": np.array(current_directions, dtype=np.int32),
                    "actions": np.array(current_actions, dtype=np.int64),
                    "rewards": np.array(current_rewards, dtype=np.float32),
                }
                episodes.append(episode)
                total_steps += len(current_actions)

            # Progress
            if total_attempted % 100 == 0:
                elapsed = time.time() - t0
                success_rate = len(episodes) / max(total_attempted, 1)
                print(f"    [{len(episodes)}/{min_episodes} kept, "
                      f"{total_attempted} attempted, "
                      f"success_rate={success_rate:.1%}, "
                      f"{elapsed:.0f}s]")

            # Reset accumulators
            current_images = []
            current_directions = []
            current_actions = []
            current_rewards = []

            # Record initial obs of new episode
            img, direction = get_raw_obs(vec_env)
            current_images.append(img)
            current_directions.append(direction)
        else:
            img, direction = get_raw_obs(vec_env)
            current_images.append(img)
            current_directions.append(direction)

    elapsed = time.time() - t0
    returns = [float(np.sum(ep["rewards"])) for ep in episodes]
    success_rate = len(episodes) / max(total_attempted, 1)
    print(f"    Done: {len(episodes)} episodes kept out of {total_attempted} attempted "
          f"({success_rate:.1%} success rate)")
    print(f"    {total_steps:,} transitions in {elapsed:.0f}s")
    if returns:
        print(f"    Returns: {np.mean(returns):.3f} +/- {np.std(returns):.3f} "
              f"[{np.min(returns):.3f}, {np.max(returns):.3f}]")
    ep_lens = [len(ep["actions"]) for ep in episodes]
    if ep_lens:
        print(f"    Ep lengths: {np.mean(ep_lens):.0f} +/- {np.std(ep_lens):.0f} "
              f"[{np.min(ep_lens)}, {np.max(ep_lens)}]")

    return episodes, total_steps, total_attempted


# ============================================================================
# .npz save
# ============================================================================

def episodes_to_flat(episodes: List[Dict]) -> Dict[str, np.ndarray]:
    """Flatten episode list into arrays for .npz."""
    all_images = []
    all_dirs = []
    all_acts = []
    all_rews = []
    boundaries = []
    ep_returns = []
    offset = 0

    for ep in episodes:
        T = len(ep["actions"])
        all_images.append(ep["observations_image"])      # (T, H, W, 3)
        all_dirs.append(ep["observations_direction"])     # (T,)
        all_acts.append(ep["actions"])                    # (T,)
        all_rews.append(ep["rewards"])                    # (T,)
        offset += T
        boundaries.append(offset)
        ep_returns.append(float(np.sum(ep["rewards"])))

    return {
        "observations_image": np.concatenate(all_images, axis=0).astype(np.uint8),
        "observations_direction": np.concatenate(all_dirs, axis=0).astype(np.int32),
        "actions": np.concatenate(all_acts, axis=0).astype(np.int64),
        "rewards": np.concatenate(all_rews, axis=0).astype(np.float32),
        "episode_terminals": np.array(boundaries, dtype=np.int64),
        "episode_returns": np.array(ep_returns, dtype=np.float32),
    }


def save_npz(path: str, data: Dict[str, np.ndarray]):
    """Save dataset arrays to compressed .npz."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **data)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    n_steps = len(data["actions"])
    n_eps = len(data["episode_returns"])
    print(f"  Saved {path}")
    print(f"    {n_eps} episodes, {n_steps:,} transitions, {size_mb:.1f} MB")



# ============================================================================
# Main
# ============================================================================

DEFAULT_ENVS = {
    "MiniGrid-MoveBall-v0": "saved_models/minigrid/MiniGrid-MoveBall-v0/ppo_gbrl/MiniGrid-MoveBall-v0_seed_0_1000000_steps.zip",
    "MiniGrid-KeyDoor-v0": "saved_models/minigrid/MiniGrid-KeyDoor-v0/ppo_gbrl/MiniGrid-KeyDoor-v0_seed_0_1000000_steps.zip",
    "MiniGrid-BoxKey-v0": "saved_models/minigrid/MiniGrid-BoxKey-v0/ppo_gbrl/MiniGrid-BoxKey-v0_seed_0_1000000_steps.zip",
}


def generate_dataset(env_name: str, model_path: str, output_dir: str,
                     min_expert_episodes: int,
                     seed: int, deterministic: bool):
    """Generate expert dataset for one environment."""
    from algos.ppo import PPO_GBRL

    env_short = env_name.replace("MiniGrid-", "").replace("-v0", "").lower()
    print(f"\n{'='*60}")
    print(f"Generating dataset for {env_name}")
    print(f"  Model: {model_path}")
    print(f"  Expert episodes: {min_expert_episodes}")
    print(f"{'='*60}")

    vec_env = make_eval_env(env_name, seed=seed)
    model = PPO_GBRL.load(model_path, env=vec_env, device="cuda")
    model.policy.to("cuda")

    expert_episodes, expert_steps, expert_attempted = collect_expert_episodes(
        vec_env, model,
        min_episodes=min_expert_episodes,
        deterministic=deterministic,
        success_only=True,
    )
    vec_env.close()

    if not expert_episodes:
        print(f"  WARNING: No successful expert episodes collected for {env_name}!")
        return

    expert_data = episodes_to_flat(expert_episodes)
    expert_path = os.path.join(output_dir, f"{env_short}_expert.npz")
    save_npz(expert_path, expert_data)
    print(f"  Done with {env_name}!")


def main():
    parser = argparse.ArgumentParser(description="Generate MiniGrid imitation-learning datasets")
    parser.add_argument("--env_name", type=str, default=None,
                        help="Single env to generate for (default: all 3 subtask envs)")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to trained PPO-GBRL model .zip (required if --env_name set)")
    parser.add_argument("--output_dir", type=str, default=str(ROOT_PATH / "datasets" / "minigrid"),
                        help="Output directory for .npz files")
    parser.add_argument("--min_expert_episodes", type=int, default=500,
                        help="Minimum successful expert episodes to collect per env")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic policy for expert")
    parser.add_argument("--no_deterministic", action="store_false", dest="deterministic",
                        help="Use stochastic policy for expert")

    args = parser.parse_args()

    if args.env_name:
        if not args.model_path:
            if args.env_name in DEFAULT_ENVS:
                args.model_path = str(ROOT_PATH / DEFAULT_ENVS[args.env_name])
            else:
                parser.error("--model_path required when --env_name is not a default env")
        generate_dataset(
            args.env_name, args.model_path, args.output_dir,
            args.min_expert_episodes,
            args.seed, args.deterministic,
        )
    else:
        # Generate for all default envs
        for env_name, model_rel in DEFAULT_ENVS.items():
            model_path = str(ROOT_PATH / model_rel)
            if not os.path.exists(model_path):
                print(f"  Skipping {env_name}: model not found at {model_path}")
                continue
            generate_dataset(
                env_name, model_path, args.output_dir,
                args.min_expert_episodes,
                args.seed, args.deterministic,
            )

    print(f"\nAll done! Datasets saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
