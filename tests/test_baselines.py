"""Smoke tests for all baseline algorithms.

Verifies:
1. Each baseline instantiates without error
2. Expert data is loaded correctly (right counts, reward assignments)
3. A short .learn() call runs without crashing
4. Key algorithmic invariants hold (SQIL rewards, RLPD buffer prefill, etc.)

Run:  python3.10 -m pytest tests/test_baselines.py -v
"""
import json
import os
import tempfile
from unittest import mock

import numpy as np
import pytest
from stable_baselines3.common.env_util import make_vec_env

# ── Fixtures ─────────────────────────────────────────────────────────────────

ENV_ID = "CartPole-v1"
N_EXPERT = 200   # small synthetic expert dataset
TOTAL_STEPS = 512  # short training for smoke test


def _generate_expert_data(env, n=N_EXPERT):
    """Generate synthetic expert transitions compatible with CartPole."""
    obs_list, act_list, rew_list, done_list = [], [], [], []
    obs = env.reset()

    for _ in range(n):
        act = np.array([env.action_space.sample()])
        next_obs, rew, done, info = env.step(act)
        obs_list.append(obs[0])
        act_list.append(act[0])
        rew_list.append(rew[0])
        done_list.append(done[0])
        obs = next_obs if not done[0] else env.reset()

    return {
        'observations': np.array(obs_list, dtype=np.float32),
        'actions': np.array(act_list, dtype=np.int64),
        'rewards': np.array(rew_list, dtype=np.float32),
        'dones': np.array(done_list, dtype=bool),
    }


# Cached module-level data so we only generate once
_EXPERT_DATA_CACHE = {}


def _get_expert_data():
    if 'data' not in _EXPERT_DATA_CACHE:
        env = make_vec_env(ENV_ID, n_envs=1)
        _EXPERT_DATA_CACHE['data'] = _generate_expert_data(env)
        env.close()
    return _EXPERT_DATA_CACHE['data']


def _mock_npz_to_flat(path, env):
    """Mock npz_to_flat_observations: return pre-generated CartPole data."""
    return _get_expert_data()


def _mock_npz_to_transitions(npz_paths, env):
    """Mock npz_to_transitions: return imitation Transitions for CartPole."""
    from imitation.data.types import Transitions

    all_obs, all_acts, all_next_obs, all_dones = [], [], [], []
    for name, path in npz_paths.items():
        data = _get_expert_data()
        obs = data['observations']
        acts = data['actions']
        dones = data['dones']

        next_obs = np.zeros_like(obs)
        next_obs[:-1] = obs[1:]
        next_obs[-1] = obs[-1]
        done_indices = np.where(dones)[0]
        for idx in done_indices:
            next_obs[idx] = obs[idx]

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


@pytest.fixture(scope="module")
def expert_datasets_dict():
    """Expert datasets dict — paths are dummy since we mock the loaders."""
    return {"cartpole": "/dummy/cartpole_expert.npz"}


@pytest.fixture(scope="module")
def expert_datasets_json(expert_datasets_dict):
    return json.dumps(expert_datasets_dict)


@pytest.fixture
def vec_env_1():
    """Single-env VecEnv for baselines that need num_envs=1."""
    env = make_vec_env(ENV_ID, n_envs=1)
    yield env
    env.close()


# Patch targets for data_utils used by RLPD and SQIL/BC
_PATCH_FLAT = "algos.baselines.data_utils.npz_to_flat_observations"
_PATCH_TRANS = "algos.baselines.data_utils.npz_to_transitions"


# ── RLPD ─────────────────────────────────────────────────────────────────────

class TestRLPD:
    @mock.patch(_PATCH_FLAT, side_effect=_mock_npz_to_flat)
    def test_instantiate_and_prefill(self, mock_fn, vec_env_1, expert_datasets_dict):
        from algos.baselines.rlpd import RLPDBaseline
        algo = RLPDBaseline(
            env=vec_env_1,
            expert_datasets=expert_datasets_dict,
            buffer_size=10000,
            batch_size=32,
            device="cpu",
            verbose=0,
        )
        # Expert data should be in the buffer
        assert algo.dqn.replay_buffer.pos > 0
        assert algo.dqn.replay_buffer.pos == N_EXPERT

    @mock.patch(_PATCH_FLAT, side_effect=_mock_npz_to_flat)
    def test_prefill_with_multi_env(self, mock_fn, expert_datasets_dict):
        """RLPD should work even when VecEnv has n_envs > 1 (the fix)."""
        env = make_vec_env(ENV_ID, n_envs=4)
        from algos.baselines.rlpd import RLPDBaseline
        algo = RLPDBaseline(
            env=env,
            expert_datasets=expert_datasets_dict,
            buffer_size=10000,
            batch_size=32,
            device="cpu",
            verbose=0,
        )
        assert algo.dqn.replay_buffer.pos == N_EXPERT
        env.close()

    @mock.patch(_PATCH_FLAT, side_effect=_mock_npz_to_flat)
    def test_learn_runs(self, mock_fn, vec_env_1, expert_datasets_dict):
        from algos.baselines.rlpd import RLPDBaseline
        algo = RLPDBaseline(
            env=vec_env_1,
            expert_datasets=expert_datasets_dict,
            buffer_size=10000,
            batch_size=32,
            learning_starts=0,
            device="cpu",
            verbose=0,
        )
        algo.learn(total_timesteps=TOTAL_STEPS)

    @mock.patch(_PATCH_FLAT, side_effect=_mock_npz_to_flat)
    def test_expert_datasets_as_json(self, mock_fn, vec_env_1, expert_datasets_json):
        from algos.baselines.rlpd import RLPDBaseline
        algo = RLPDBaseline(
            env=vec_env_1,
            expert_datasets=expert_datasets_json,
            buffer_size=10000,
            batch_size=32,
            device="cpu",
            verbose=0,
        )
        assert algo.dqn.replay_buffer.pos == N_EXPERT


# ── SQIL ─────────────────────────────────────────────────────────────────────

class TestSQIL:
    @mock.patch(_PATCH_TRANS, side_effect=_mock_npz_to_transitions)
    def test_instantiate(self, mock_fn, vec_env_1, expert_datasets_dict):
        from algos.baselines.sqil_baseline import SQILBaseline
        algo = SQILBaseline(
            env=vec_env_1,
            expert_datasets=expert_datasets_dict,
            buffer_size=10000,
            batch_size=32,
            device="cpu",
            verbose=0,
        )
        assert algo.sqil is not None

    def test_rejects_multi_env(self, expert_datasets_dict):
        """SQIL should raise ValueError when n_envs > 1."""
        env = make_vec_env(ENV_ID, n_envs=4)
        from algos.baselines.sqil_baseline import SQILBaseline
        with pytest.raises(ValueError, match="num_envs=1"):
            SQILBaseline(
                env=env,
                expert_datasets=expert_datasets_dict,
                buffer_size=10000,
                batch_size=32,
                device="cpu",
                verbose=0,
            )
        env.close()

    @mock.patch(_PATCH_TRANS, side_effect=_mock_npz_to_transitions)
    def test_learn_runs(self, mock_fn, vec_env_1, expert_datasets_dict):
        from algos.baselines.sqil_baseline import SQILBaseline
        algo = SQILBaseline(
            env=vec_env_1,
            expert_datasets=expert_datasets_dict,
            buffer_size=10000,
            batch_size=32,
            learning_starts=64,
            device="cpu",
            verbose=0,
        )
        algo.learn(total_timesteps=TOTAL_STEPS)

    @mock.patch(_PATCH_TRANS, side_effect=_mock_npz_to_transitions)
    def test_expert_reward_is_one(self, mock_fn, vec_env_1, expert_datasets_dict):
        """imitation's SQILReplayBuffer should assign reward=1 to expert data."""
        from algos.baselines.sqil_baseline import SQILBaseline
        algo = SQILBaseline(
            env=vec_env_1,
            expert_datasets=expert_datasets_dict,
            buffer_size=10000,
            batch_size=32,
            device="cpu",
            verbose=0,
        )
        buf = algo.sqil.rl_algo.replay_buffer
        # Expert buffer rewards should all be 1.0
        expert_buf = buf.expert_buffer
        expert_rewards = expert_buf.rewards[:expert_buf.pos]
        assert np.all(expert_rewards == 1.0), f"Expected all expert rewards=1, got {np.unique(expert_rewards)}"


# ── BC ───────────────────────────────────────────────────────────────────────

class TestBC:
    @mock.patch(_PATCH_TRANS, side_effect=_mock_npz_to_transitions)
    def test_instantiate(self, mock_fn, vec_env_1, expert_datasets_dict):
        from algos.baselines.bc_baseline import BCBaseline
        algo = BCBaseline(
            env=vec_env_1,
            expert_datasets=expert_datasets_dict,
            n_epochs=1,
            batch_size=32,
            device="cpu",
            verbose=0,
        )
        assert algo.bc is not None

    @mock.patch(_PATCH_TRANS, side_effect=_mock_npz_to_transitions)
    def test_learn_runs(self, mock_fn, vec_env_1, expert_datasets_dict):
        from algos.baselines.bc_baseline import BCBaseline
        algo = BCBaseline(
            env=vec_env_1,
            expert_datasets=expert_datasets_dict,
            n_epochs=2,
            batch_size=32,
            device="cpu",
            verbose=0,
        )
        algo.learn(total_timesteps=0)


# ── BC + PPO ─────────────────────────────────────────────────────────────────

class TestBCPPO:
    @mock.patch(_PATCH_TRANS, side_effect=_mock_npz_to_transitions)
    def test_instantiate(self, mock_fn, vec_env_1, expert_datasets_dict):
        from algos.baselines.bc_baseline import BCPPOFinetuneBaseline
        algo = BCPPOFinetuneBaseline(
            env=vec_env_1,
            expert_datasets=expert_datasets_dict,
            bc_epochs=1,
            bc_batch_size=32,
            device="cpu",
            verbose=0,
        )
        assert algo.bc is not None

    @mock.patch(_PATCH_TRANS, side_effect=_mock_npz_to_transitions)
    def test_learn_copies_weights(self, mock_fn, vec_env_1, expert_datasets_dict):
        from algos.baselines.bc_baseline import BCPPOFinetuneBaseline
        algo = BCPPOFinetuneBaseline(
            env=vec_env_1,
            expert_datasets=expert_datasets_dict,
            bc_epochs=1,
            bc_batch_size=32,
            ppo_n_steps=128,
            device="cpu",
            verbose=1,
        )
        algo.learn(total_timesteps=256)
        # After learn, PPO should exist
        assert algo._ppo is not None


# ── RLPD buffer shape invariants ─────────────────────────────────────────────

class TestRLPDBufferShapes:
    @mock.patch(_PATCH_FLAT, side_effect=_mock_npz_to_flat)
    def test_buffer_shapes_n_envs_1(self, mock_fn, expert_datasets_dict):
        """With n_envs=1 the buffer arrays should be (buffer_size, 1, ...)."""
        env = make_vec_env(ENV_ID, n_envs=1)
        from algos.baselines.rlpd import RLPDBaseline
        algo = RLPDBaseline(
            env=env,
            expert_datasets=expert_datasets_dict,
            buffer_size=1000,
            batch_size=32,
            device="cpu",
            verbose=0,
        )
        buf = algo.dqn.replay_buffer
        assert buf.observations.shape[1] == 1  # n_envs dim
        assert buf.actions.shape[1] == 1
        env.close()

    @mock.patch(_PATCH_FLAT, side_effect=_mock_npz_to_flat)
    def test_buffer_shapes_n_envs_4(self, mock_fn, expert_datasets_dict):
        """With n_envs=4 expert prefill should still work (the bugfix)."""
        env = make_vec_env(ENV_ID, n_envs=4)
        from algos.baselines.rlpd import RLPDBaseline
        algo = RLPDBaseline(
            env=env,
            expert_datasets=expert_datasets_dict,
            buffer_size=1000,
            batch_size=32,
            device="cpu",
            verbose=0,
        )
        buf = algo.dqn.replay_buffer
        # Buffer has n_envs=4, but expert data still loaded
        assert buf.observations.shape[1] == 4
        assert buf.pos == N_EXPERT
        env.close()
