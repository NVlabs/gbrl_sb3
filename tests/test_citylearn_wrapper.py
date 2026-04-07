"""Smoke tests for CityLearn wrapper timestep indexing and signal coherence.

Run:  python3.10 -m pytest tests/test_citylearn_wrapper.py -v
"""
import numpy as np
import pytest


@pytest.fixture(scope="module")
def arb_env():
    from env.citylearn.arbitrage_vs_buffer import make_arbitrage_vs_buffer_env
    env = make_arbitrage_vs_buffer_env()
    yield env
    env.close()


@pytest.fixture(scope="module")
def comfort_env():
    from env.citylearn.cost_vs_comfort import make_cost_vs_comfort_env
    env = make_cost_vs_comfort_env()
    yield env
    env.close()


class TestTimestepIndexing:
    """Verify that the wrapper's reward matches manual consumption × price."""

    def test_reward_matches_manual_computation(self, arb_env):
        """Wrapper reward must equal -(1/N) Σ consumption_i × price_i
        computed directly from the raw building state."""
        env = arb_env
        obs, info = env.reset()

        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            # Manual computation from raw state
            raw = env.unwrapped
            ts = raw.time_step - 1  # already incremented after step
            buildings = raw.buildings
            manual_cost = 0.0
            n_priced = 0
            for b in buildings:
                nec = b.net_electricity_consumption
                if hasattr(nec, "__len__") and 0 <= ts < len(nec):
                    consumption = float(nec[ts])
                else:
                    continue
                ep = b.pricing.electricity_pricing
                if hasattr(ep, "__len__") and 0 <= ts < len(ep):
                    price = float(ep[ts])
                else:
                    continue
                manual_cost += consumption * price
                n_priced += 1

            if n_priced > 0:
                expected_reward = -manual_cost / n_priced
            else:
                expected_reward = 0.0

            assert abs(reward - expected_reward) < 1e-6, (
                f"Reward mismatch at ts={ts}: wrapper={reward:.8f} "
                f"manual={expected_reward:.8f}"
            )

            if term or trunc:
                obs, info = env.reset()


class TestCostCoherence:
    """Verify cost signals are coherent with raw state."""

    def test_arbitrage_cost_matches_soc(self, arb_env):
        """Buffer depletion cost should fire iff mean SOC < safety level."""
        env = arb_env
        obs, info = env.reset()
        safety = env._soc_safety_level

        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            mean_soc = env._mean_electrical_soc()
            cost = info["cost"]

            if mean_soc is None:
                assert cost == 0.0
            elif mean_soc >= safety:
                assert cost == 0.0, (
                    f"Cost should be 0 when mean_soc={mean_soc:.4f} >= "
                    f"safety={safety}, got {cost:.4f}"
                )
            else:
                expected = min(1.0, (safety - mean_soc) / safety)
                assert abs(cost - expected) < 1e-6, (
                    f"Cost mismatch: mean_soc={mean_soc:.4f} "
                    f"expected={expected:.4f} got={cost:.4f}"
                )

            if term or trunc:
                obs, info = env.reset()

    def test_comfort_cost_zero_when_inside_band(self, comfort_env):
        """Cost must be 0 when all buildings are within comfort band."""
        env = comfort_env
        obs, info = env.reset()

        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            # Check each building manually
            all_inside = True
            for b in env._get_buildings():
                lower, upper = env._comfort_bounds(b)
                t_in = env._latest(b.indoor_dry_bulb_temperature)
                if t_in is not None:
                    if upper is not None and t_in > upper:
                        all_inside = False
                    if lower is not None and t_in < lower:
                        all_inside = False

            if all_inside:
                assert info["cost"] == 0.0, (
                    f"Cost should be 0 when all buildings inside band, "
                    f"got {info['cost']:.6f}"
                )

            if term or trunc:
                obs, info = env.reset()


class TestLabelCoherence:
    """Verify label signals match their stated conditions."""

    def test_arbitrage_label_requires_frontier_soc(self, arb_env):
        """Label=1 must only fire when mean SOC ≤ safety + frontier_band."""
        env = arb_env
        obs, info = env.reset()
        frontier_upper = env._soc_safety_level + env._soc_frontier_band

        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            if info["safety_label"] == 1:
                mean_soc = env._mean_electrical_soc()
                assert mean_soc is not None
                assert mean_soc <= frontier_upper + 1e-6, (
                    f"Label=1 but mean_soc={mean_soc:.4f} > "
                    f"frontier_upper={frontier_upper:.4f}"
                )

            if term or trunc:
                obs, info = env.reset()

    def test_comfort_label_excludes_violating_buildings(self, comfort_env):
        """Label=1 must not fire based on already-violating buildings."""
        env = comfort_env
        obs, info = env.reset()

        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            if info["safety_label"] == 1:
                # At least one building must have 0 < headroom < threshold
                has_frontier_building = False
                for b in env._get_buildings():
                    headroom = env._building_headroom(b)
                    if 0 < headroom < env._headroom_thresh:
                        has_frontier_building = True
                assert has_frontier_building, (
                    "Label=1 but no building has positive headroom "
                    "below threshold"
                )

            if term or trunc:
                obs, info = env.reset()


class TestInfoKeys:
    """Both scenarios must emit the required info keys."""

    @pytest.mark.parametrize("env_fixture", ["arb_env", "comfort_env"])
    def test_info_keys_present(self, env_fixture, request):
        env = request.getfixturevalue(env_fixture)
        obs, info = env.reset()
        assert "cost" in info
        assert "safety_label" in info
        assert "original_reward" in info

        obs, reward, term, trunc, info = env.step(env.action_space.sample())
        assert "cost" in info
        assert "safety_label" in info
        assert "original_reward" in info
        assert isinstance(info["cost"], float)
        assert info["safety_label"] in (0, 1)
