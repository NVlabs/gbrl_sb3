"""Smoke tests for CityLearn wrapper timestep indexing and signal coherence.

Run:  python3.10 -m pytest tests/test_citylearn_wrapper.py -v
"""
import numpy as np
import pytest


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def arb_env():
    from env.citylearn.arbitrage_vs_buffer import make_arbitrage_vs_buffer_env
    env = make_arbitrage_vs_buffer_env()
    yield env
    env.close()


@pytest.fixture(scope="module")
def peak_env():
    from env.citylearn.peak_shaving import make_peak_shaving_env
    env = make_peak_shaving_env()
    yield env
    env.close()


@pytest.fixture(scope="module")
def carbon_env():
    from env.citylearn.carbon_aware import make_carbon_aware_env
    env = make_carbon_aware_env()
    yield env
    env.close()


# -----------------------------------------------------------------------
# Reward
# -----------------------------------------------------------------------

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


# -----------------------------------------------------------------------
# Arbitrage vs Buffer — cost & label
# -----------------------------------------------------------------------

class TestArbitrageCost:

    def test_cost_matches_soc(self, arb_env):
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


class TestArbitrageLabel:

    def test_label_requires_frontier_soc(self, arb_env):
        """Label=1 requires mean SOC within [safety, safety + band]."""
        env = arb_env
        obs, info = env.reset()
        frontier_lower = env._soc_safety_level
        frontier_upper = env._soc_safety_level + env._soc_frontier_band

        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            if info["safety_label"] == 1:
                mean_soc = env._mean_electrical_soc()
                assert mean_soc is not None
                assert mean_soc >= frontier_lower - 1e-6, (
                    f"Label=1 but mean_soc={mean_soc:.4f} < "
                    f"frontier_lower={frontier_lower:.4f}"
                )
                assert mean_soc <= frontier_upper + 1e-6, (
                    f"Label=1 but mean_soc={mean_soc:.4f} > "
                    f"frontier_upper={frontier_upper:.4f}"
                )

            if term or trunc:
                obs, info = env.reset()


# -----------------------------------------------------------------------
# Peak Shaving — cost & label
# -----------------------------------------------------------------------

class TestPeakShavingCost:

    def test_cost_zero_below_rolling_peak(self, peak_env):
        """Cost must be 0 when current NEC ≤ rolling daily peak."""
        env = peak_env
        obs, info = env.reset()

        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            current_nec = env._mean_current_nec()
            prev_peak = env._rolling_daily_peak()

            if current_nec is not None and prev_peak is not None and prev_peak > 0:
                if current_nec <= prev_peak:
                    assert info["cost"] == 0.0, (
                        f"Cost should be 0 when nec={current_nec:.4f} <= "
                        f"peak={prev_peak:.4f}, got {info['cost']:.4f}"
                    )

            if term or trunc:
                obs, info = env.reset()

    def test_cost_positive_above_rolling_peak(self, peak_env):
        """Cost must be positive when current NEC exceeds rolling peak."""
        env = peak_env
        obs, info = env.reset()

        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            current_nec = env._mean_current_nec()
            prev_peak = env._rolling_daily_peak()

            if (current_nec is not None and prev_peak is not None
                    and prev_peak > 0 and current_nec > prev_peak):
                assert info["cost"] > 0.0, (
                    f"Cost should be > 0 when nec={current_nec:.4f} > "
                    f"peak={prev_peak:.4f}, got {info['cost']:.4f}"
                )

            if term or trunc:
                obs, info = env.reset()


class TestPeakShavingLabel:

    def test_label_requires_near_peak_and_high_nsl(self, peak_env):
        """Label=1 requires NEC near daily peak AND high non-shiftable load."""
        env = peak_env
        obs, info = env.reset()

        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            if info["safety_label"] == 1:
                current_nec = env._mean_current_nec()
                prev_peak = env._rolling_daily_peak()
                mean_nsl = env._mean_current_nsl()

                assert current_nec is not None
                assert prev_peak is not None and prev_peak > 0
                assert current_nec >= env._peak_proximity * prev_peak - 1e-6, (
                    f"Label=1 but nec={current_nec:.4f} < "
                    f"{env._peak_proximity}×peak={prev_peak:.4f}"
                )
                assert mean_nsl is not None
                assert mean_nsl > env._nsl_threshold - 1e-6, (
                    f"Label=1 but mean_nsl={mean_nsl:.4f} <= "
                    f"nsl_threshold={env._nsl_threshold:.4f}"
                )

            if term or trunc:
                obs, info = env.reset()


# -----------------------------------------------------------------------
# Carbon Aware — cost & label
# -----------------------------------------------------------------------

class TestCarbonAwareCost:

    def test_cost_zero_when_no_grid_import(self, carbon_env):
        """Cost must be 0 when all buildings export (NEC ≤ 0)."""
        env = carbon_env
        obs, info = env.reset()

        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            all_export = True
            for b in env._get_buildings():
                nec = env._at_timestep(b.net_electricity_consumption)
                if nec is not None and nec > 0:
                    all_export = False
                    break

            if all_export:
                assert info["cost"] == 0.0, (
                    f"Cost should be 0 when all buildings export, "
                    f"got {info['cost']:.6f}"
                )

            if term or trunc:
                obs, info = env.reset()


class TestCarbonAwareLabel:

    def test_label_requires_high_carbon_low_price(self, carbon_env):
        """Label=1 requires high carbon AND low price (the disagreement)."""
        env = carbon_env
        obs, info = env.reset()

        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            if info["safety_label"] == 1:
                buildings = env._get_buildings()
                max_carbon = env._max_current_carbon(buildings)
                max_price = env._max_current_price(buildings)

                assert max_carbon is not None
                assert max_carbon > env._carbon_threshold - 1e-6, (
                    f"Label=1 but carbon={max_carbon:.4f} <= "
                    f"threshold={env._carbon_threshold:.4f}"
                )
                assert max_price is not None
                assert max_price <= env._price_threshold + 1e-6, (
                    f"Label=1 but price={max_price:.4f} > "
                    f"threshold={env._price_threshold:.4f}"
                )

            if term or trunc:
                obs, info = env.reset()


# -----------------------------------------------------------------------
# Info keys — all scenarios
# -----------------------------------------------------------------------

class TestInfoKeys:

    @pytest.mark.parametrize("env_fixture", [
        "arb_env", "peak_env", "carbon_env",
    ])
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
