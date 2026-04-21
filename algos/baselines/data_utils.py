"""Convert MiniGrid expert .npz files to imitation library format.

Our .npz files contain:
  observations_image: (N, 7, 7, 3) uint8
  observations_direction: (N,) int32
  actions: (N,) int64
  rewards: (N,) float32
  episode_terminals: (n_eps,) int64  (cumulative end indices)
  episode_returns: (n_eps,) float32

The imitation library expects either:
  - List[Trajectory] with obs (T+1, *obs_shape), acts (T,), terminal bool
  - Transitions with obs, acts, infos, next_obs, dones
"""
import numpy as np
from typing import Dict, List

from minigrid.wrappers import FlatObsWrapper
import gymnasium as gym


def _flatten_minigrid_obs(image: np.ndarray, direction: int,
                          flat_dim: int) -> np.ndarray:
    """Replicate FlatObsWrapper's flattening for a single observation.

    FlatObsWrapper concatenates: flattened_image + [direction] + mission_encoding
    The mission encoding pads to fill flat_dim.
    """
    flat_image = image.flatten().astype(np.float32)
    # FlatObsWrapper: image / max_val, then direction, then mission encoding
    # max_val for MiniGrid is typically the max object index
    flat = np.zeros(flat_dim, dtype=np.float32)
    flat[:len(flat_image)] = flat_image
    flat[len(flat_image)] = direction
    return flat


def npz_to_flat_observations(npz_path: str, env: gym.Env) -> Dict[str, np.ndarray]:
    """Load .npz and convert observations to match FlatObsWrapper output.

    Uses the actual env's FlatObsWrapper to ensure exact format match.
    """
    data = np.load(npz_path)
    images = data['observations_image']
    directions = data['observations_direction']
    actions = data['actions']
    rewards = data['rewards']
    ep_terminals = data['episode_terminals']

    n = len(actions)

    # Get the flat observation space dimension from the env
    obs_dim = env.observation_space.shape[0]

    # Use FlatObsWrapper's logic directly: it accesses the 'image' and
    # 'direction' keys from the obs dict and calls self.observation()
    # We replicate that here by creating a temporary wrapper instance
    # and calling its observation method
    if hasattr(env, 'observation'):
        # env is already a FlatObsWrapper
        flat_obs_fn = env.observation
    else:
        # Try to find the FlatObsWrapper in the wrapper chain
        flat_obs_fn = None
        e = env
        while hasattr(e, 'env'):
            if isinstance(e, FlatObsWrapper):
                flat_obs_fn = e.observation
                break
            e = e.env

    observations = np.zeros((n, obs_dim), dtype=np.float32)
    for i in range(n):
        obs_dict = {'image': images[i], 'direction': directions[i], 'mission': ''}
        if flat_obs_fn is not None:
            observations[i] = flat_obs_fn(obs_dict)
        else:
            # Fallback: manual flattening
            flat_img = images[i].flatten().astype(np.float32)
            observations[i, :len(flat_img)] = flat_img
            observations[i, len(flat_img)] = float(directions[i])

    # Build dones from episode_terminals
    dones = np.zeros(n, dtype=bool)
    for end_idx in ep_terminals:
        dones[int(end_idx) - 1] = True

    return {
        'observations': observations,
        'actions': actions[:n],
        'rewards': rewards[:n],
        'dones': dones,
        'episode_terminals': ep_terminals,
    }


def npz_to_transitions(npz_paths: Dict[str, str], env: gym.Env):
    """Convert multiple .npz expert files to imitation Transitions.

    Args:
        npz_paths: dict of {name: path} for each expert dataset
        env: environment with FlatObsWrapper applied

    Returns:
        imitation.data.types.Transitions
    """
    from imitation.data.types import Transitions

    all_obs, all_acts, all_next_obs, all_dones = [], [], [], []

    for name, path in npz_paths.items():
        data = npz_to_flat_observations(path, env)
        obs = data['observations']
        acts = data['actions']
        dones = data['dones']

        # Build next_obs (shift by 1, last in episode wraps to self)
        next_obs = np.zeros_like(obs)
        next_obs[:-1] = obs[1:]
        next_obs[-1] = obs[-1]

        # Fix episode boundaries: at done=True, next_obs = current obs
        done_indices = np.where(dones)[0]
        for idx in done_indices:
            next_obs[idx] = obs[idx]
        # Also fix cross-episode boundaries
        for idx in done_indices:
            if idx + 1 < len(obs):
                # Next transition starts a new episode, its next_obs should
                # be from the new episode (already correct from shift)
                pass

        all_obs.append(obs)
        all_acts.append(acts)
        all_next_obs.append(next_obs)
        all_dones.append(dones)

    obs = np.concatenate(all_obs)
    acts = np.concatenate(all_acts)
    next_obs = np.concatenate(all_next_obs)
    dones = np.concatenate(all_dones)
    infos = np.array([{}] * len(obs))

    return Transitions(obs=obs, acts=acts, infos=infos,
                       next_obs=next_obs, dones=dones)


def npz_to_trajectories(npz_paths: Dict[str, str], env: gym.Env):
    """Convert multiple .npz expert files to list of imitation Trajectory objects.

    Args:
        npz_paths: dict of {name: path} for each expert dataset
        env: environment with FlatObsWrapper applied

    Returns:
        List[imitation.data.types.TrajectoryWithRew]
    """
    from imitation.data.types import TrajectoryWithRew

    trajectories = []

    for name, path in npz_paths.items():
        data = npz_to_flat_observations(path, env)
        obs = data['observations']
        acts = data['actions']
        rewards = data['rewards']
        ep_terminals = data['episode_terminals']

        # Split into per-episode trajectories
        ep_starts = np.concatenate([[0], ep_terminals[:-1]])
        ep_ends = ep_terminals

        for ep_idx in range(len(ep_terminals)):
            start = int(ep_starts[ep_idx])
            end = int(ep_ends[ep_idx])

            # TrajectoryWithRew expects obs to have T+1 entries (including final)
            ep_obs = obs[start:end]
            ep_acts = acts[start:end - 1] if end - start > 1 else acts[start:end]
            ep_rews = rewards[start:end - 1] if end - start > 1 else rewards[start:end]

            # Append final observation (duplicate last)
            final_obs = obs[end - 1:end]
            traj_obs = np.concatenate([ep_obs, final_obs], axis=0)

            infos = np.array([{}] * len(ep_acts))

            traj = TrajectoryWithRew(
                obs=traj_obs,
                acts=ep_acts,
                infos=infos,
                terminal=True,
                rews=ep_rews,
            )
            trajectories.append(traj)

    print(f"Converted {len(trajectories)} expert trajectories "
          f"({sum(len(t.acts) for t in trajectories)} transitions)")
    return trajectories
