#!/usr/bin/env python3.10
"""Measure reward/cost baselines for each (env, scenario) combo.

Runs a max-pressure heuristic controller for N episodes with
clean_episode_prob=0.5, reports ep_rew_clean, ep_rew_event, ep_cost_event.

Usage:
    python3.10 scripts/measure_baselines.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from collections import defaultdict

# Suppress SUMO GUI
os.environ["LIBSUMO_AS_TRACI"] = "1"

# Output file for results (SUMO pollutes stdout)
_OUT = open("/swgwork/bfuhrer/projects/gbrl_project/nvlabs/gbrl_sb3/results/baselines.txt", "w")

def _print(*args, **kwargs):
    """Print to results file."""
    print(*args, file=_OUT, flush=True, **kwargs)

from env.sumo import (
    SumoRewardCostWrapper,
    SUMO_CONFIGS,
    SCENARIO_DEFAULTS,
    make_sumo_raw_env,
)


def run_max_pressure_episodes(
    env_name: str,
    cost_fn: str,
    n_episodes: int = 20,
    clean_episode_prob: float = 0.5,
):
    """Run max-pressure heuristic on a PettingZoo SUMO env.
    
    Max-pressure: at each step, for each agent, pick the phase whose
    served incoming lanes have the highest total queue (density).
    """
    cfg = SUMO_CONFIGS[env_name]
    scenario_kw = SCENARIO_DEFAULTS.get((env_name, cost_fn), {})

    results_clean = {"rew": [], "cost": []}
    results_event = {"rew": [], "cost": []}

    for ep in range(n_episodes):
        raw_env = make_sumo_raw_env(
            net_file=cfg["net_file_fn"](),
            route_file=cfg["route_file_fn"](),
            num_seconds=cfg["num_seconds"],
            delta_time=cfg["delta_time"],
            yellow_time=cfg["yellow_time"],
            min_green=cfg["min_green"],
            max_green=cfg["max_green"],
            reward_fn=cfg["reward_fn"],
            sumo_seed=ep,  # deterministic per episode
            use_gui=False,
        )
        env = SumoRewardCostWrapper(
            raw_env,
            cost_fn=cost_fn,
            clean_episode_prob=clean_episode_prob,
            **scenario_kw,
        )

        obs, infos = env.reset(seed=ep)
        
        # After reset, phase_to_lanes is built. Cache it.
        phase_to_lanes = dict(env._phase_to_lanes)
        
        ep_rewards = {a: 0.0 for a in env.possible_agents}
        ep_costs = {a: 0.0 for a in env.possible_agents}
        is_clean = env._is_clean_episode
        _print(f"  ep {ep}: {'clean' if is_clean else 'event'}", end="")

        while env.agents:
            actions = {}
            for agent in env.agents:
                ts = env._get_traffic_signal(agent)
                if ts is None:
                    actions[agent] = 0
                    continue

                # Get per-lane vehicle count for this agent's incoming lanes
                sumo = env._get_traci()
                lane_vehicles = []
                for lane in ts.lanes:
                    lane_vehicles.append(sumo.lane.getLastStepVehicleNumber(lane))

                # Pick phase with highest total queue on served lanes
                best_phase = 0
                best_score = -1
                p2l = phase_to_lanes.get(agent, {})
                for phase_idx, served_lanes in p2l.items():
                    score = sum(lane_vehicles[li] for li in served_lanes if li < len(lane_vehicles))
                    if score > best_score:
                        best_score = score
                        best_phase = phase_idx

                actions[agent] = best_phase

            obs, rewards, terminations, truncations, infos = env.step(actions)

            for agent, r in rewards.items():
                ep_rewards[agent] += r
                cost = infos.get(agent, {}).get("cost", 0.0)
                ep_costs[agent] += cost

        total_rew = sum(ep_rewards.values())
        total_cost = sum(ep_costs.values())

        if is_clean:
            results_clean["rew"].append(total_rew)
            results_clean["cost"].append(total_cost)
        else:
            results_event["rew"].append(total_rew)
            results_event["cost"].append(total_cost)

        env.close()
        _print(f" rew={total_rew:.0f} cost={total_cost:.1f}")

    return results_clean, results_event


def main():
    env_names = ["sumo-single-vhvh-v0", "sumo-grid4x4-v0", "sumo-arterial4x4-v0"]
    cost_fns = ["bus_priority", "convoy_priority", "premium_priority"]

    _print(f"{'env':<25} {'scenario':<20} {'clean_rew':>12} {'event_rew':>12} {'event_cost':>12} {'ep_rew_mean':>12}")
    _print("-" * 95)

    for env_name in env_names:
        for cost_fn in cost_fns:
            print(f"Running {env_name} / {cost_fn} ...", end="", flush=True, file=sys.stderr)
            try:
                clean, event = run_max_pressure_episodes(
                    env_name, cost_fn, n_episodes=10, clean_episode_prob=0.5
                )
            except Exception as e:
                print(f" ERROR: {e}", file=sys.stderr)
                _print(f"{env_name:<25} {cost_fn:<20} {'ERROR':>12} {'':>12} {'':>12} {'':>12}")
                continue

            clean_rew = np.mean(clean["rew"]) if clean["rew"] else float("nan")
            event_rew = np.mean(event["rew"]) if event["rew"] else float("nan")
            event_cost = np.mean(event["cost"]) if event["cost"] else float("nan")
            
            # Expected ep_rew_mean with 0.5 clean prob
            if clean["rew"] and event["rew"]:
                ep_rew_mean = 0.5 * clean_rew + 0.5 * event_rew
            else:
                ep_rew_mean = float("nan")

            n_clean = len(clean["rew"])
            n_event = len(event["rew"])

            print(f" done ({n_clean} clean, {n_event} event)", file=sys.stderr)
            _print(f"{env_name:<25} {cost_fn:<20} {clean_rew:>12.1f} {event_rew:>12.1f} {event_cost:>12.1f} {ep_rew_mean:>12.1f}")

    _print()
    _print("Notes:")
    _print("  clean_rew    = avg reward on clean episodes (no special vehicles)")
    _print("  event_rew    = avg reward on event episodes (with special vehicles)")
    _print("  event_cost   = avg cost on event episodes")
    _print("  ep_rew_mean  = expected 0.5*clean + 0.5*event (what training sees)")
    _OUT.close()


if __name__ == "__main__":
    main()
