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
    from env.citylearn.scenarios import make_arbitrage_vs_buffer_env
    env = make_arbitrage_vs_buffer_env()
    yield env
    env.close()


@pytest.fixture(scope="module")
def demand_env():
    from env.citylearn.scenarios import make_contract_demand_env
    env = make_contract_demand_env()
    yield env
    env.close()


@pytest.fixture(scope="module")
def carbon_env():
    from env.citylearn.scenarios import make_carbon_aware_env
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
        """Buffer depletion cost should fire iff critical SOC < safety level."""
        env = arb_env
        obs, info = env.reset()
        safety = env._soc_safety_level

        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            critical_soc = env._critical_electrical_soc()
            cost = info["cost"]

            if critical_soc is None:
                assert cost == 0.0
            elif critical_soc >= safety:
                assert cost == 0.0, (
                    f"Cost should be 0 when critical_soc={critical_soc:.4f} >= "
                    f"safety={safety}, got {cost:.4f}"
                )
            else:
                expected = min(1.0, (safety - critical_soc) / safety)
                assert abs(cost - expected) < 1e-6, (
                    f"Cost mismatch: critical_soc={critical_soc:.4f} "
                    f"expected={expected:.4f} got={cost:.4f}"
                )

            if term or trunc:
                obs, info = env.reset()


class TestArbitrageLabel:

    def test_label_requires_conflict_frontier(self, arb_env):
        """Label=1 requires all three conditions:
        - critical SOC ≤ safety + band (tight reserve)
        - price > high_price_threshold (discharge temptation)
        - some battery is dischargeable

        Label is computed pre-step, so we snapshot SOC and price before
        stepping and check the returned label against that snapshot.
        """
        env = arb_env
        obs, info = env.reset()
        frontier_upper = env._soc_safety_level + env._soc_frontier_band

        for _ in range(200):
            # Snapshot pre-step state (what the label was computed from)
            pre_critical_soc = env._critical_electrical_soc()
            pre_price = env._max_pre_step_price(env._get_buildings())

            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            if info["safety_label"] == 1:
                assert pre_critical_soc is not None
                assert pre_critical_soc <= frontier_upper + 1e-6, (
                    f"Label=1 but pre-step critical_soc={pre_critical_soc:.4f} "
                    f"> frontier_upper={frontier_upper:.4f}"
                )
                # Must have high price
                assert pre_price is not None
                assert env._price_threshold is not None
                assert pre_price > env._price_threshold - 1e-6, (
                    f"Label=1 but pre-step price={pre_price:.6f} "
                    f"<= threshold={env._price_threshold:.6f}"
                )

            if term or trunc:
                obs, info = env.reset()


# -----------------------------------------------------------------------
# Contract Demand — cost & label
# -----------------------------------------------------------------------

class TestContractDemandCost:

    def test_cost_zero_below_frontier(self, demand_env):
        """Cost must be 0 when district import ≤ frontier."""
        env = demand_env
        obs, info = env.reset()

        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            district_import = env._district_import_post_step()

            if district_import is not None and env._frontier is not None:
                if district_import <= env._frontier:
                    assert info["cost"] == 0.0, (
                        f"Cost should be 0 when import={district_import:.4f} <= "
                        f"frontier={env._frontier:.4f}, got {info['cost']:.4f}"
                    )

            if term or trunc:
                obs, info = env.reset()

    def test_cost_positive_above_frontier(self, demand_env):
        """Cost must be positive when district import > frontier."""
        env = demand_env
        obs, info = env.reset()

        for _ in range(100):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            district_import = env._district_import_post_step()

            if (district_import is not None and env._frontier is not None
                    and district_import > env._frontier):
                assert info["cost"] > 0.0, (
                    f"Cost should be > 0 when import={district_import:.4f} > "
                    f"frontier={env._frontier:.4f}, got {info['cost']:.4f}"
                )

            if term or trunc:
                obs, info = env.reset()


class TestContractDemandLabel:

    def test_label_cheap_charging_trap(self, demand_env):
        """Label=1 iff price <= price_low_threshold AND
        district_nsl >= trap_threshold.

        Label is computed pre-step from the fixed NSL/price schedules.
        """
        env = demand_env
        obs, info = env.reset()

        for _ in range(200):
            # Snapshot pre-step state
            pre_nsl = env._district_pre_step_nsl()
            buildings = env._get_buildings()
            pre_price = env._max_pre_step_price(buildings)

            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            if info["safety_label"] == 1:
                assert (
                    pre_nsl is not None
                    and pre_price is not None
                    and env._trap_threshold is not None
                    and env._price_low_threshold is not None
                ), "Label=1 but missing pre-step state"
                assert pre_price <= env._price_low_threshold + 1e-6, (
                    f"Label=1 but price={pre_price} > "
                    f"threshold={env._price_low_threshold}"
                )
                assert pre_nsl >= env._trap_threshold - 1e-6, (
                    f"Label=1 but nsl={pre_nsl} < "
                    f"trap_threshold={env._trap_threshold}"
                )

            if term or trunc:
                obs, info = env.reset()


# -----------------------------------------------------------------------
# Carbon Aware — cost & label
# -----------------------------------------------------------------------

class TestCarbonAwareCost:

    def test_cost_zero_when_no_grid_import(self, carbon_env):
        """Cost must be 0 when district net export (total NEC ≤ 0)."""
        env = carbon_env
        obs, info = env.reset()

        for _ in range(50):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            # District total NEC
            total_nec = 0.0
            n = 0
            for b in env._get_buildings():
                nec = env._at_timestep(b.net_electricity_consumption)
                if nec is not None:
                    total_nec += nec
                    n += 1

            if n > 0 and total_nec <= 0:
                assert info["cost"] == 0.0, (
                    f"Cost should be 0 when district NEC={total_nec:.6f} "
                    f"<= 0, got {info['cost']:.6f}"
                )

            if term or trunc:
                obs, info = env.reset()


class TestCarbonAwareLabel:

    def test_label_conflict_threshold(self, carbon_env):
        """Label=1 iff dirty × load × cheap ≥ conflict_threshold.

        Cheapness uses price_low_threshold as hard gate.
        Label is computed pre-step from carbon intensity, NSL, and price.
        """
        env = carbon_env
        obs, info = env.reset()

        for _ in range(200):
            # Snapshot pre-step state (same accessors the label uses)
            buildings = env._get_buildings()
            pre_carbon = env._max_pre_step_carbon(buildings)
            pre_nsl = env._district_pre_step_nsl()
            pre_price = env._max_pre_step_price(buildings)

            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            if info["safety_label"] == 1:
                assert pre_carbon is not None and pre_nsl is not None and pre_price is not None
                # Price must be below threshold (hard gate)
                assert pre_price < env._price_low_threshold + 1e-6, (
                    f"Label=1 but price={pre_price} >= "
                    f"price_low_threshold={env._price_low_threshold}"
                )
                vr = env._carbon_max - env._carbon_threshold
                dirty = max(0.0, min(1.0, (pre_carbon - env._carbon_threshold) / vr))
                load = min(1.0, pre_nsl / env._import_scale)
                pr = env._price_low_threshold - env._price_min
                cheap = max(0.0, min(1.0, (env._price_low_threshold - pre_price) / pr))
                conflict = dirty * load * cheap
                assert conflict >= env._conflict_threshold - 1e-6, (
                    f"Label=1 but conflict={conflict:.4f} < "
                    f"threshold={env._conflict_threshold:.4f}"
                )

            if term or trunc:
                obs, info = env.reset()


# -----------------------------------------------------------------------
# Info keys — all scenarios
# -----------------------------------------------------------------------

class TestInfoKeys:

    @pytest.mark.parametrize("env_fixture", [
        "arb_env", "demand_env", "carbon_env",
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
        from env.citylearn.scenarios import make_arbitrage_vs_buffer_env
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

                # Break if either env terminated during the diverge step
                if term_a or trunc_a or term_b or trunc_b:
                    break

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


# -----------------------------------------------------------------------
# Fixtures — new scenarios
# -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def ratchet_env():
    from env.citylearn.scenarios import make_peak_ratchet_env
    env = make_peak_ratchet_env()
    yield env
    env.close()


@pytest.fixture(scope="module")
def dr_env():
    from env.citylearn.scenarios import make_demand_response_env
    env = make_demand_response_env()
    yield env
    env.close()


@pytest.fixture(scope="module")
def solar_env():
    from env.citylearn.scenarios import make_solar_ramp_reserve_env
    env = make_solar_ramp_reserve_env()
    yield env
    env.close()


# -----------------------------------------------------------------------
# Peak Ratchet — cost & label
# -----------------------------------------------------------------------

class TestPeakRatchetObs:

    def test_obs_augmented(self, ratchet_env):
        """Observation should have 55 dims (54 + running peak)."""
        env = ratchet_env
        obs, info = env.reset()
        assert obs.shape == (55,), f"Expected (55,), got {obs.shape}"

        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert obs.shape == (55,), f"Expected (55,), got {obs.shape}"

    def test_obs_peak_clamped(self, ratchet_env):
        """Augmented peak feature must stay in [0, 1] even if
        running_peak exceeds peak_cap."""
        env = ratchet_env
        obs, info = env.reset()

        for _ in range(200):
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            peak_feature = obs[-1]
            assert 0.0 <= peak_feature <= 1.0, (
                f"Peak feature {peak_feature} outside [0,1], "
                f"running_peak={env._running_peak:.2f}, "
                f"peak_cap={env._peak_cap:.2f}"
            )
            if term or trunc:
                obs, info = env.reset()


class TestPeakRatchetCost:

    def test_cost_zero_below_running_peak(self, ratchet_env):
        """Cost must be 0 when district import ≤ running peak."""
        env = ratchet_env
        obs, info = env.reset()

        for _ in range(100):
            pre_peak = env._running_peak
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            di = env._district_import_post_step()
            if di is not None and di <= pre_peak and pre_peak > 0:
                assert info["cost"] == 0.0, (
                    f"Cost should be 0 when import={di:.4f} <= "
                    f"running_peak={pre_peak:.4f}, got {info['cost']:.4f}"
                )

            if term or trunc:
                obs, info = env.reset()

    def test_running_peak_monotonic(self, ratchet_env):
        """Running peak must never decrease within an episode."""
        env = ratchet_env
        obs, info = env.reset()
        prev_peak = 0.0

        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            assert env._running_peak >= prev_peak - 1e-9, (
                f"Running peak decreased: {env._running_peak:.4f} < "
                f"{prev_peak:.4f}"
            )
            prev_peak = env._running_peak

            if term or trunc:
                obs, info = env.reset()
                assert env._running_peak == 0.0, (
                    f"Running peak should reset to 0 on reset, "
                    f"got {env._running_peak}"
                )
                prev_peak = 0.0


class TestPeakRatchetLabel:

    def test_label_requires_cheap_and_potential_above_threshold(
        self, ratchet_env
    ):
        """Label=1 iff price ≤ price_low_threshold AND
        nsl + headroom*power > max(running_peak, peak_target)."""
        env = ratchet_env
        obs, info = env.reset()

        for _ in range(200):
            pre_peak = env._running_peak
            pre_nsl = env._district_pre_step_nsl()
            buildings = env._get_buildings()
            pre_price = env._max_pre_step_price(buildings)

            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            if info["safety_label"] == 1:
                assert pre_price is not None and pre_nsl is not None
                assert pre_price <= env._price_low_threshold + 1e-6, (
                    f"Label=1 but price={pre_price} > "
                    f"threshold={env._price_low_threshold}"
                )
                potential = (
                    pre_nsl + env._charge_headroom_frac
                    * env._total_charge_power
                )
                threshold = max(pre_peak, env._peak_target)
                assert potential > threshold - 1e-6, (
                    f"Label=1 but potential={potential:.4f} <= "
                    f"threshold={threshold:.4f}"
                )

            if term or trunc:
                obs, info = env.reset()


# -----------------------------------------------------------------------
# Demand Response — cost & label
# -----------------------------------------------------------------------

class TestDemandResponseObs:

    def test_obs_augmented(self, dr_env):
        """Observation should have 55 dims (54 + steps-until-event)."""
        env = dr_env
        obs, info = env.reset()
        assert obs.shape == (55,), f"Expected (55,), got {obs.shape}"

        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert obs.shape == (55,), f"Expected (55,), got {obs.shape}"


class TestDemandResponseCost:

    def test_cost_zero_outside_events(self, dr_env):
        """Cost must be 0 when not in an event window."""
        env = dr_env
        obs, info = env.reset()

        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            ts = env._get_time_step() - 1
            if not env._in_event_window(ts):
                assert info["cost"] == 0.0, (
                    f"Cost should be 0 outside event window at ts={ts}, "
                    f"got {info['cost']:.4f}"
                )

            if term or trunc:
                obs, info = env.reset()

    def test_cost_zero_when_soc_above_target(self, dr_env):
        """Cost must be 0 when mean SOC ≥ soc_target, even during events."""
        env = dr_env
        obs, info = env.reset()

        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            ts = env._get_time_step() - 1
            soc = env._mean_electrical_soc()
            if (env._in_event_window(ts) and soc is not None
                    and soc >= env._soc_target):
                assert info["cost"] == 0.0, (
                    f"Cost should be 0 when SOC={soc:.4f} >= "
                    f"target={env._soc_target}, got {info['cost']:.4f}"
                )

            if term or trunc:
                obs, info = env.reset()


class TestDemandResponseLabel:

    def test_label_phase_aware(self, dr_env):
        """Label=1 in three phases:
        1. Prep window + low SOC
        2. Prep window + high price + adequate SOC
        3. In-event + low SOC
        """
        env = dr_env
        obs, info = env.reset()

        for _ in range(300):
            ts = env._get_time_step()
            pre_steps_until = env._steps_until_next_event(ts)
            pre_soc = env._mean_electrical_soc()
            buildings = env._get_buildings()
            pre_price = env._max_pre_step_price(buildings)

            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            if info["safety_label"] == 1:
                in_prep = (0 < pre_steps_until <= env._prep_window)
                in_event = (pre_steps_until == 0)
                low_soc = (
                    pre_soc is not None and pre_soc < env._soc_target
                )
                high_price = (
                    pre_price is not None
                    and env._high_price_threshold is not None
                    and pre_price > env._high_price_threshold - 1e-6
                )

                # Must be in one of the three phases
                phase1 = in_prep and low_soc
                phase2 = in_prep and not low_soc and high_price
                phase3 = in_event and low_soc
                assert phase1 or phase2 or phase3, (
                    f"Label=1 but no phase matched: "
                    f"steps_until={pre_steps_until}, "
                    f"soc={pre_soc}, price={pre_price}"
                )

            if term or trunc:
                obs, info = env.reset()

    def test_event_timing(self, dr_env):
        """Verify event windows and steps_until are consistent."""
        env = dr_env
        # During event → steps_until = 0
        for start in env._event_starts:
            assert env._in_event_window(start)
            assert env._steps_until_next_event(start) == 0
            assert env._in_event_window(
                start + env._event_duration - 1
            )
            assert not env._in_event_window(
                start + env._event_duration
            )

        # Before event → steps_until = start - ts
        ts_before = env._event_starts[0] - 10
        assert env._steps_until_next_event(ts_before) == 10


# -----------------------------------------------------------------------
# Solar Ramp Reserve — cost & label
# -----------------------------------------------------------------------

class TestSolarRampObs:

    def test_obs_augmented(self, solar_env):
        """Observation should have 55 dims (54 + hours-until-buffer)."""
        env = solar_env
        obs, info = env.reset()
        assert obs.shape == (55,), f"Expected (55,), got {obs.shape}"

        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert obs.shape == (55,), f"Expected (55,), got {obs.shape}"


class TestSolarRampCost:

    def test_cost_zero_outside_buffer(self, solar_env):
        """Cost must be 0 when not in a post-sunset buffer window."""
        env = solar_env
        obs, info = env.reset()

        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            ts = env._get_time_step() - 1  # post-step
            if ts not in env._buffer_timesteps:
                assert info["cost"] == 0.0, (
                    f"Cost should be 0 outside buffer at ts={ts}, "
                    f"got {info['cost']:.4f}"
                )

            if term or trunc:
                obs, info = env.reset()

    def test_cost_zero_when_soc_above_reserve(self, solar_env):
        """Cost must be 0 when SOC ≥ soc_reserve, even in buffer."""
        env = solar_env
        obs, info = env.reset()

        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            ts = env._get_time_step() - 1
            soc = env._mean_electrical_soc()
            if (ts in env._buffer_timesteps
                    and soc is not None and soc >= env._soc_reserve):
                assert info["cost"] == 0.0, (
                    f"Cost should be 0 when SOC={soc:.4f} >= "
                    f"reserve={env._soc_reserve}, got {info['cost']:.4f}"
                )

            if term or trunc:
                obs, info = env.reset()


class TestSolarRampLabel:

    def test_label_phase_aware(self, solar_env):
        """Label=1 in three phases:
        1. Pre-sunset + low SOC
        2. Pre-sunset + high price + adequate SOC
        3. In-buffer + low SOC
        """
        env = solar_env
        obs, info = env.reset()

        for _ in range(300):
            ts = env._get_time_step()
            pre_soc = env._mean_electrical_soc()
            buildings = env._get_buildings()
            pre_price = env._max_pre_step_price(buildings)

            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)

            if info["safety_label"] == 1:
                in_prep = ts in env._prep_timesteps
                in_buffer = ts in env._buffer_timesteps
                low_soc = (
                    pre_soc is not None
                    and pre_soc < env._soc_reserve
                )
                high_price = (
                    pre_price is not None
                    and env._high_price_threshold is not None
                    and pre_price > env._high_price_threshold - 1e-6
                )

                phase1 = in_prep and low_soc
                phase2 = in_prep and not low_soc and high_price
                phase3 = in_buffer and low_soc
                assert phase1 or phase2 or phase3, (
                    f"Label=1 but no phase matched: ts={ts}, "
                    f"in_prep={in_prep}, in_buffer={in_buffer}, "
                    f"soc={pre_soc}, price={pre_price}"
                )

            if term or trunc:
                obs, info = env.reset()

    def test_sunset_detection(self, solar_env):
        """Verify that prep and buffer windows were detected."""
        env = solar_env
        env.reset()  # ensure _scenario_init ran
        env.step(env.action_space.sample())
        assert len(env._prep_timesteps) > 0, "No prep timesteps detected"
        assert len(env._buffer_timesteps) > 0, "No buffer timesteps detected"
        assert len(env._buffer_starts) > 0, "No buffer starts detected"
        # Prep and buffer windows should not overlap
        overlap = env._prep_timesteps & env._buffer_timesteps
        assert len(overlap) == 0, (
            f"Prep and buffer windows overlap at {sorted(overlap)[:5]}"
        )


# -----------------------------------------------------------------------
# Info keys — all scenarios (extended)
# -----------------------------------------------------------------------

class TestInfoKeysNewScenarios:

    @pytest.mark.parametrize("env_fixture", [
        "ratchet_env", "dr_env", "solar_env",
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
