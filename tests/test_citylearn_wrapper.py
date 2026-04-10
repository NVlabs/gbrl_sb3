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
        """Label=1 requires either:
        - mean SOC < safety level (cost recovery), or
        - mean SOC within [safety, safety + band] (frontier).

        Label is computed pre-step, so we snapshot SOC before stepping
        and check the returned label against that snapshot.
        """
        env = arb_env
        obs, info = env.reset()
        frontier_lower = env._soc_safety_level
        frontier_upper = env._soc_safety_level + env._soc_frontier_band

        for _ in range(200):
            # Snapshot pre-step state (what the label was computed from)
            pre_soc = env._mean_electrical_soc()

            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            if info["safety_label"] == 1:
                assert pre_soc is not None
                below_safety = pre_soc < frontier_lower
                in_frontier = (pre_soc >= frontier_lower - 1e-6 and
                               pre_soc <= frontier_upper + 1e-6)
                assert below_safety or in_frontier, (
                    f"Label=1 but pre-step mean_soc={pre_soc:.4f} is "
                    f"neither below safety={frontier_lower:.4f} nor "
                    f"within frontier [{frontier_lower:.4f}, "
                    f"{frontier_upper:.4f}]"
                )

            if term or trunc:
                obs, info = env.reset()


# -----------------------------------------------------------------------
# Peak Shaving — cost & label
# -----------------------------------------------------------------------

class TestPeakShavingCost:

    def test_cost_zero_below_rolling_peak(self, peak_env):
        """Cost must be 0 when current total NEC ≤ rolling daily peak."""
        env = peak_env
        obs, info = env.reset()

        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            current_nec = env._total_current_nec()
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
        """Cost must be positive when current total NEC exceeds rolling peak."""
        env = peak_env
        obs, info = env.reset()

        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            current_nec = env._total_current_nec()
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
        """Label=1 requires total NSL near rolling NSL peak AND high.

        Label is computed pre-step using NSL (exogenous, pre-allocated);
        snapshot the same pre-step values the label used.
        """
        env = peak_env
        obs, info = env.reset()

        for _ in range(200):
            # Snapshot pre-step state (what the label was computed from)
            pre_nsl = env._total_pre_step_nsl()
            nsl_peak = env._rolling_nsl_peak_pre_step()

            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            if info["safety_label"] == 1:
                assert pre_nsl is not None
                assert nsl_peak is not None and nsl_peak > 0
                assert pre_nsl >= env._peak_proximity * nsl_peak - 1e-6, (
                    f"Label=1 but pre-step nsl={pre_nsl:.4f} < "
                    f"{env._peak_proximity}×nsl_peak={nsl_peak:.4f}"
                )
                assert pre_nsl > env._nsl_threshold - 1e-6, (
                    f"Label=1 but pre-step total_nsl={pre_nsl:.4f} <= "
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

    def test_label_requires_high_carbon_low_price_and_import(self, carbon_env):
        """Label=1 requires high carbon, low price, AND district importing.

        Label is computed pre-step; snapshot values before stepping.
        """
        env = carbon_env
        obs, info = env.reset()

        for _ in range(200):
            # Snapshot pre-step state
            buildings = env._get_buildings()
            pre_carbon = env._max_current_carbon(buildings)
            pre_price = env._max_current_price(buildings)
            pre_total_nec = sum(
                env._at_timestep(b.net_electricity_consumption) or 0.0
                for b in buildings
            )

            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            if info["safety_label"] == 1:
                assert pre_carbon is not None
                assert pre_carbon > env._carbon_threshold - 1e-6, (
                    f"Label=1 but pre-step carbon={pre_carbon:.4f} <= "
                    f"threshold={env._carbon_threshold:.4f}"
                )
                assert pre_price is not None
                assert pre_price <= env._price_threshold + 1e-6, (
                    f"Label=1 but pre-step price={pre_price:.4f} > "
                    f"threshold={env._price_threshold:.4f}"
                )
                assert pre_total_nec > 0, (
                    f"Label=1 but pre-step district NEC={pre_total_nec:.4f} "
                    f"<= 0 (no import pressure)"
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


# -----------------------------------------------------------------------
# Action-gap test — Arbitrage vs Buffer
# -----------------------------------------------------------------------

class TestArbitrageActionGap:
    """In label=1 states, discharging battery should improve reward but
    worsen cost relative to holding/charging.

    Uses two identical envs stepped with the same actions up to a label=1
    state, then diverges: env_a discharges, env_b holds.
    """

    def test_discharge_vs_hold_at_frontier(self):
        """At label=1: discharge → higher reward, higher cost (or both 0)."""
        from env.citylearn.arbitrage_vs_buffer import make_arbitrage_vs_buffer_env
        import numpy as np

        # Two identical envs (deterministic — same schema, same weather)
        env_a = make_arbitrage_vs_buffer_env()
        env_b = make_arbitrage_vs_buffer_env()
        obs_a, _ = env_a.reset()
        obs_b, _ = env_b.reset()

        # Action layout per building: [dhw_storage, electrical_storage, cooling_device]
        # 3 buildings → 9 actions.  Index 1, 4, 7 = electrical_storage.
        n_act = env_a.action_space.shape[0]

        # Step both envs identically until we hit label=1
        found = 0
        reward_gaps = []  # reward_discharge - reward_hold
        cost_gaps = []    # cost_discharge - cost_hold

        for step_i in range(718):
            # Use the same random action in both envs
            action = env_a.action_space.sample()
            obs_a, r_a, term_a, trunc_a, info_a = env_a.step(action)
            obs_b, r_b, term_b, trunc_b, info_b = env_b.step(action)

            if info_a["safety_label"] == 1:
                # Diverge at the NEXT step: discharge in A, hold in B
                discharge_action = np.zeros(n_act, dtype=np.float32)
                hold_action = np.zeros(n_act, dtype=np.float32)
                for idx in [1, 4, 7]:  # electrical_storage per building
                    discharge_action[idx] = -1.0  # full discharge
                    hold_action[idx] = 0.0         # hold

                obs_a, r_dis, term_a, trunc_a, info_dis = env_a.step(
                    discharge_action
                )
                obs_b, r_hold, term_b, trunc_b, info_hold = env_b.step(
                    hold_action
                )

                reward_gaps.append(r_dis - r_hold)
                cost_gaps.append(info_dis["cost"] - info_hold["cost"])
                found += 1

                # Re-sync: both envs already have different state now,
                # but we keep going to collect more samples.  For strict
                # causal isolation we'd need to re-create envs, but
                # even drifted samples test the direction of the gap.

            if term_a or trunc_a:
                break

        assert found >= 2, (
            f"Expected at least 2 label=1 states to test, found {found}"
        )

        # Statistical check: discharge should tend to improve reward
        # (higher = less spending because high price × negative NEC)
        mean_reward_gap = np.mean(reward_gaps)
        assert mean_reward_gap > 0, (
            f"Expected positive mean reward gap (discharge > hold), "
            f"got {mean_reward_gap:.6f} from {found} samples: {reward_gaps}"
        )

        # Cost should tend to worsen (higher) with discharge because
        # SOC drops closer to (or below) safety level.
        mean_cost_gap = np.mean(cost_gaps)
        assert mean_cost_gap >= 0, (
            f"Expected non-negative mean cost gap (discharge ≥ hold), "
            f"got {mean_cost_gap:.6f} from {found} samples: {cost_gaps}"
        )

        env_a.close()
        env_b.close()
