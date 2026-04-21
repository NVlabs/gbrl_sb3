#!/usr/bin/env python3.10
"""Diagnostic: step each CityLearn CMDP scenario through one episode with
zero-action (idle battery) and collect label / cost / reward statistics.

This validates scenario mechanics WITHOUT training:
  - Label distribution (fraction of steps at label=0, [0,1], 1)
  - Cost firing rate and magnitude
  - Reward distribution
  - Whether each scenario's conflict conditions actually trigger
  - Per-scenario event window coverage
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import defaultdict

from env.citylearn.scenarios import (
    make_arbitrage_vs_buffer_env,
    make_contract_demand_env,
    make_dirty_window_reserve_env,
    make_peak_ratchet_env,
    make_demand_response_env,
    make_solar_ramp_reserve_env,
)

SCENARIOS = {
    "PeakGuard (arbitrage_vs_buffer)":     make_arbitrage_vs_buffer_env,
    "CapFrontier (contract_demand)":        make_contract_demand_env,
    "DirtyReserve (dirty_window_reserve)":  make_dirty_window_reserve_env,
    "PeakRatchet (peak_ratchet)":           make_peak_ratchet_env,
    "EventReserve (demand_response)":       make_demand_response_env,
    "SunsetBridge (solar_ramp_reserve)":    make_solar_ramp_reserve_env,
}


def run_episode(env, policy="zero"):
    """Step through one full episode. Returns per-step records."""
    obs, info = env.reset()
    records = []
    done = False
    step_i = 0

    while not done:
        if policy == "zero":
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        elif policy == "discharge":
            # CityLearn: negative action = discharge (gentle to avoid assertion)
            action = np.full(env.action_space.shape, -0.15, dtype=np.float32)
        elif policy == "charge":
            # CityLearn: positive action = charge
            action = np.full(env.action_space.shape, 0.3, dtype=np.float32)
        else:
            action = env.action_space.sample()

        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except (AssertionError, Exception):
            break
        done = terminated or truncated

        records.append({
            "step": step_i,
            "reward": reward,
            "cost": info.get("cost", 0.0),
            "label": info.get("safety_label", 0),
        })
        step_i += 1

    return records


def classify_label(label):
    if isinstance(label, list):
        return "both"
    elif label == 1:
        return "cost-only"
    else:
        return "reward-only"


def analyze_records(records):
    """Compute summary statistics from episode records."""
    n = len(records)
    rewards = [r["reward"] for r in records]
    costs = [r["cost"] for r in records]
    labels = [r["label"] for r in records]

    label_counts = defaultdict(int)
    for l in labels:
        label_counts[classify_label(l)] += 1

    nonzero_costs = [c for c in costs if c > 0]

    return {
        "n_steps": n,
        "reward_mean": np.mean(rewards),
        "reward_std": np.std(rewards),
        "reward_min": np.min(rewards),
        "reward_max": np.max(rewards),
        "reward_sum": np.sum(rewards),
        "cost_mean": np.mean(costs),
        "cost_sum": np.sum(costs),
        "cost_nonzero_count": len(nonzero_costs),
        "cost_nonzero_frac": len(nonzero_costs) / n if n > 0 else 0,
        "cost_nonzero_mean": np.mean(nonzero_costs) if nonzero_costs else 0,
        "cost_max": np.max(costs),
        "label_reward_only": label_counts.get("reward-only", 0),
        "label_both": label_counts.get("both", 0),
        "label_cost_only": label_counts.get("cost-only", 0),
        "label_reward_only_frac": label_counts.get("reward-only", 0) / n,
        "label_both_frac": label_counts.get("both", 0) / n,
        "label_cost_only_frac": label_counts.get("cost-only", 0) / n,
    }


def print_analysis(name, stats, policy_name):
    print(f"\n{'='*70}")
    print(f"  {name}  (policy: {policy_name})")
    print(f"{'='*70}")
    print(f"  Episode length:  {stats['n_steps']} steps")
    print(f"  ── Reward ──")
    print(f"    mean={stats['reward_mean']:.4f}  std={stats['reward_std']:.4f}  "
          f"sum={stats['reward_sum']:.1f}  min={stats['reward_min']:.4f}  max={stats['reward_max']:.4f}")
    print(f"  ── Cost ──")
    print(f"    mean={stats['cost_mean']:.4f}  sum={stats['cost_sum']:.1f}  max={stats['cost_max']:.4f}")
    print(f"    nonzero: {stats['cost_nonzero_count']}/{stats['n_steps']} "
          f"({stats['cost_nonzero_frac']:.1%})  "
          f"mean-when-nonzero={stats['cost_nonzero_mean']:.4f}")
    print(f"  ── Labels ──")
    print(f"    reward-only (0):  {stats['label_reward_only']:>4d}  ({stats['label_reward_only_frac']:.1%})")
    print(f"    both [0,1]:       {stats['label_both']:>4d}  ({stats['label_both_frac']:.1%})")
    print(f"    cost-only (1):    {stats['label_cost_only']:>4d}  ({stats['label_cost_only_frac']:.1%})")


def main():
    policies = ["zero", "discharge", "charge"]

    for scenario_name, make_fn in SCENARIOS.items():
        print(f"\n\n{'#'*70}")
        print(f"# {scenario_name}")
        print(f"{'#'*70}")

        for policy in policies:
            env = make_fn()
            records = run_episode(env, policy=policy)
            stats = analyze_records(records)
            print_analysis(scenario_name, stats, policy)

            # Extra scenario-specific diagnostics
            if hasattr(env, '_peak_timesteps') and env._peak_timesteps:
                print(f"  ── Scenario-specific ──")
                print(f"    peak timesteps: {len(env._peak_timesteps)} ({len(env._peak_timesteps)/stats['n_steps']:.1%})")
                if env._high_price_threshold is not None:
                    print(f"    high_price_threshold (within peak): {env._high_price_threshold:.4f}")

            if hasattr(env, '_congestion_timesteps') and env._congestion_timesteps:
                print(f"  ── Scenario-specific ──")
                print(f"    congestion timesteps: {len(env._congestion_timesteps)} ({len(env._congestion_timesteps)/stats['n_steps']:.1%})")
                if env._cap is not None:
                    print(f"    cap={env._cap:.2f}  frontier={env._frontier:.2f}")

            if hasattr(env, '_dirty_timesteps') and env._dirty_timesteps:
                print(f"  ── Scenario-specific ──")
                print(f"    dirty timesteps: {len(env._dirty_timesteps)} ({len(env._dirty_timesteps)/stats['n_steps']:.1%})")
                print(f"    prep timesteps: {len(env._prep_timesteps)} ({len(env._prep_timesteps)/stats['n_steps']:.1%})")

            if hasattr(env, '_running_peak'):
                print(f"  ── Scenario-specific ──")
                print(f"    running_peak={env._running_peak:.2f}  peak_target={env._peak_target:.2f}  peak_cap={env._peak_cap:.2f}")
                if env._baseline_import is not None:
                    bi = env._baseline_import
                    print(f"    baseline_import: max={bi.max():.2f}  Q95={np.quantile(bi, 0.95):.2f}  "
                          f"Q90={np.quantile(bi, 0.90):.2f}  mean={bi.mean():.2f}  charge_power={env._total_charge_power:.2f}")

            if hasattr(env, '_event_starts') and env._event_starts:
                print(f"  ── Scenario-specific ──")
                dur = env._event_duration
                total_event_steps = len(env._event_starts) * dur
                print(f"    events: {len(env._event_starts)} at timesteps {env._event_starts}")
                print(f"    event coverage: {total_event_steps}/{stats['n_steps']} ({total_event_steps/stats['n_steps']:.1%})")

            if hasattr(env, '_buffer_timesteps') and env._buffer_timesteps:
                print(f"  ── Scenario-specific ──")
                print(f"    buffer timesteps: {len(env._buffer_timesteps)} ({len(env._buffer_timesteps)/stats['n_steps']:.1%})")
                print(f"    prep timesteps: {len(env._prep_timesteps)} ({len(env._prep_timesteps)/stats['n_steps']:.1%})")
                print(f"    sunsets detected: {len(env._buffer_starts)}")

            env.close()

    # ── Cross-scenario comparison table ──
    for pol_name in ["zero", "discharge", "charge"]:
        print(f"\n\n{'='*70}")
        print(f"  CROSS-SCENARIO COMPARISON (policy: {pol_name})")
        print(f"{'='*70}")
        print(f"  {'Scenario':<35s} {'Steps':>5s} {'Rew Sum':>9s} {'Cost Sum':>9s} "
              f"{'Cost%':>7s} {'L=1%':>7s} {'L=[0,1]%':>9s}")
        print(f"  {'-'*35} {'-'*5} {'-'*9} {'-'*9} {'-'*7} {'-'*7} {'-'*9}")

        for scenario_name, make_fn in SCENARIOS.items():
            env = make_fn()
            records = run_episode(env, policy=pol_name)
            s = analyze_records(records)
            short = scenario_name.split("(")[0].strip()
            print(f"  {short:<35s} {s['n_steps']:>5d} {s['reward_sum']:>9.1f} "
                  f"{s['cost_sum']:>9.1f} {s['cost_nonzero_frac']:>6.1%} "
                  f"{s['label_cost_only_frac']:>6.1%} {s['label_both_frac']:>8.1%}")
            env.close()


if __name__ == "__main__":
    main()
