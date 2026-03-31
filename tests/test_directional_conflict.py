#!/usr/bin/env python3.10
"""Empirical validation: directional reward + starvation cost has CC-like structural conflict.

Tests 4 hand-crafted policies on grid4x4:
1. always_priority  – always pick phase 0 (serves priority lanes)
2. always_non_priority – always pick phase 4 (serves non-priority lanes)
3. cycle_5  – cycle through all phases every 5 steps
4. random   – uniform random phase selection

Expected outcome if conflict is genuine (CC analogy):
- always_priority:     HIGH reward, HIGH cost (starvation)
- always_non_priority: LOW reward,  LOW cost
- cycle_5:             MED reward,  MED cost
- random:              MED reward,  MED cost

If reward and cost are positively correlated across policies,
the conflict is structural — you CANNOT optimize both.
"""
import os
import sys

os.environ["LIBSUMO_AS_TRACI"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.sumo import make_sumo_vec_env

N_EPISODES = 3
RESULTS_FILE = "/tmp/directional_conflict.txt"


def run_policy(policy_name, policy_fn, n_episodes=N_EPISODES):
    """Run a hand-crafted policy and collect reward/cost statistics."""
    vec_env = make_sumo_vec_env(
        env_name="sumo-grid4x4-v0",
        n_envs=1,
        seed=0,
        override_reward="directional",
        cost_fn="starvation",
        starvation_threshold=3,
    )
    n_envs = vec_env.num_envs
    act_dim = vec_env.action_space.n

    all_rewards = []
    all_costs = []
    all_starvation = []
    episodes_done = 0
    step_count = 0

    obs = vec_env.reset()
    ep_rewards = np.zeros(n_envs)
    ep_costs = np.zeros(n_envs)
    ep_starvation = np.zeros(n_envs)

    while episodes_done < n_episodes * n_envs:
        actions = np.array([policy_fn(step_count, act_dim) for _ in range(n_envs)])
        obs, rewards, dones, infos = vec_env.step(actions)
        step_count += 1

        ep_rewards += rewards
        for i, info in enumerate(infos):
            ep_costs[i] += info.get("cost", 0.0)
            ep_starvation[i] += info.get("cost_starvation", 0.0)

        for i, done in enumerate(dones):
            if done:
                all_rewards.append(ep_rewards[i])
                all_costs.append(ep_costs[i])
                all_starvation.append(ep_starvation[i])
                ep_rewards[i] = 0.0
                ep_costs[i] = 0.0
                ep_starvation[i] = 0.0
                episodes_done += 1

    vec_env.close()

    return {
        "policy": policy_name,
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "mean_cost": float(np.mean(all_costs)),
        "std_cost": float(np.std(all_costs)),
        "mean_starvation": float(np.mean(all_starvation)),
        "n_episodes": len(all_rewards),
    }


# ── Policy functions ──
def always_priority(step, act_dim):
    return 0  # phase 0 = typically serves one direction

def always_non_priority(step, act_dim):
    return act_dim // 2  # phase in the middle = typically serves cross direction

def cycle_5(step, act_dim):
    return (step // 5) % act_dim

def random_policy(step, act_dim):
    return np.random.randint(act_dim)


if __name__ == "__main__":
    policies = [
        ("always_priority", always_priority),
        ("always_non_priority", always_non_priority),
        ("cycle_5", cycle_5),
        ("random", random_policy),
    ]

    results = []
    for name, fn in policies:
        print(f"\n{'='*60}")
        print(f"Running policy: {name}")
        print(f"{'='*60}")
        r = run_policy(name, fn)
        results.append(r)
        print(f"  mean_reward={r['mean_reward']:.1f}  mean_cost={r['mean_cost']:.1f}  "
              f"mean_starvation={r['mean_starvation']:.1f}")

    # Write results
    with open(RESULTS_FILE, "w") as f:
        f.write("Directional Reward + Starvation Cost: CC-like Conflict Validation\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Policy':<25} {'Mean Reward':>12} {'Mean Cost':>12} {'Mean Starvation':>16}\n")
        f.write("-" * 70 + "\n")
        for r in results:
            f.write(f"{r['policy']:<25} {r['mean_reward']:>12.1f} {r['mean_cost']:>12.1f} "
                    f"{r['mean_starvation']:>16.1f}\n")

        f.write("\n\nConflict Analysis:\n")
        f.write("-" * 70 + "\n")
        rewards = [r["mean_reward"] for r in results]
        costs = [r["mean_cost"] for r in results]
        if len(rewards) > 1:
            corr = np.corrcoef(rewards, costs)[0, 1]
            f.write(f"Pearson correlation(reward, cost) = {corr:.3f}\n")
            if corr > 0.3:
                f.write("POSITIVE correlation -> STRUCTURAL CONFLICT (CC-like)!\n")
                f.write("Higher reward policies also incur higher cost.\n")
                f.write("Agent CANNOT maximize reward without increasing cost.\n")
            elif corr < -0.3:
                f.write("NEGATIVE correlation -> NO conflict (objectives aligned).\n")
            else:
                f.write("WEAK correlation -> Inconclusive.\n")

    print(f"\nResults written to {RESULTS_FILE}")
    with open(RESULTS_FILE) as f:
        print(f.read())
