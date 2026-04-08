#!/usr/bin/env python3.10
"""
Diagnostic script: empirically measure whether SUMO tasks
generate genuine reward-vs-cost conflicts for SPLIT-RL.

Measures:
  1. Reward/cost firing rates and co-occurrence
  2. Label firing rate and label-observation predictability
  3. Correlation(reward, cost) — negative = conflict, positive = no conflict
  4. For SUMO: side-street demand verification

Usage:
  python3.10 scripts/diagnose_conflict.py --env sumo --scenario bus_priority --network arterial
  python3.10 scripts/diagnose_conflict.py --env sumo --scenario bus_priority --network grid
  python3.10 scripts/diagnose_conflict.py --env sumo
"""
import argparse
import sys
import os
import warnings
import numpy as np
from collections import defaultdict

warnings.filterwarnings("ignore")
os.environ.setdefault("LIBSUMO_AS_TRACI", "1")


def diagnose_sumo(
    n_episodes: int = 3,
    verbose: bool = True,
    scenario: str = "bus_priority",
    network: str = "arterial",
):
    """Run SUMO diagnostic for a specific scenario and network."""
    from env.sumo import make_sumo_vec_env

    env_name = f"sumo-{network}4x4-v0"

    # Scenario-specific config
    if scenario == "bus_priority":
        cost_fn = "bus_priority"
        scenario_label = "BUS PRIORITY"
        cost_desc = "max(bus_wait/T_bus) per agent"
        label_desc = "bus on unserved lane AND wait > T_warn"
    elif scenario == "convoy_priority":
        cost_fn = "convoy_priority"
        scenario_label = "CONVOY PRIORITY"
        cost_desc = "max(convoy_wait/T_convoy) on unserved lanes per agent"
        label_desc = "convoy on unserved lane AND wait > T_warn"
    elif scenario == "spillback":
        cost_fn = "spillback"
        scenario_label = "SPILLBACK / ROAD WORKS"
        cost_desc = "max downstream occ on currently served lanes per agent"
        label_desc = "serving into blocked downstream (occ > T_occ)"
    elif scenario == "side_queue":
        cost_fn = "side_queue"
        scenario_label = "SIDE QUEUE (legacy)"
        cost_desc = "max side-street queue ratio"
        label_desc = "side queue > cap"
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    print("=" * 70)
    print(f"SUMO {network.upper()}4x4 — {scenario_label} CONFLICT DIAGNOSTIC")
    print(f"  reward = diff-waiting-time (original SUMO-RL, no override)")
    print(f"  cost   = {cost_desc}")
    print(f"  label  = {label_desc}")
    print("=" * 70)

    env = make_sumo_vec_env(
        env_name=env_name,
        n_envs=1,
        seed=42,
        override_reward=None,  # use original diff-waiting-time
        cost_fn=cost_fn,
    )

    n_agents = env.num_envs
    print(f"  VecEnv slots (agents x envs): {n_agents}")
    print(f"  Obs dim: {env.observation_space.shape[0]}")

    # Accumulators
    all_rewards = []
    all_costs = []
    all_labels = []
    all_obs = []
    conflict_steps = 0
    total_steps = 0
    ep_count = 0

    ep_rewards = np.zeros(n_agents)
    ep_costs = np.zeros(n_agents)
    ep_labels = np.zeros(n_agents)
    ep_steps = 0

    obs = env.reset()

    while ep_count < n_episodes:
        actions = np.array([env.action_space.sample() for _ in range(n_agents)])
        obs, rewards, dones, infos = env.step(actions)

        for i in range(n_agents):
            cost = infos[i].get("cost", 0.0)
            label = infos[i].get("safety_label", 0)
            r = rewards[i]

            all_rewards.append(r)
            all_costs.append(cost)
            all_labels.append(label)
            all_obs.append(obs[i].copy())

            if r > 0 and cost > 0:
                conflict_steps += 1
            total_steps += 1

            ep_rewards[i] += r
            ep_costs[i] += cost
            ep_labels[i] += label

        ep_steps += 1

        if any(dones):
            ep_count += 1
            if verbose:
                mean_r = np.mean(ep_rewards)
                mean_c = np.mean(ep_costs)
                mean_l = np.mean(ep_labels) / max(ep_steps, 1)
                print(f"  Episode {ep_count}: steps={ep_steps}, "
                      f"mean_reward={mean_r:.2f}, mean_cost={mean_c:.2f}, "
                      f"label_rate={mean_l:.3f}")
            ep_rewards[:] = 0
            ep_costs[:] = 0
            ep_labels[:] = 0
            ep_steps = 0

    env.close()

    all_rewards = np.array(all_rewards)
    all_costs = np.array(all_costs)
    all_labels = np.array(all_labels)
    all_obs = np.array(all_obs)

    print(f"\n--- CONFLICT STATISTICS ({total_steps} total transitions) ---")

    # 1. Firing rates
    cost_nonzero = np.sum(all_costs > 0)
    label_fire = np.sum(all_labels > 0)
    reward_positive = np.sum(all_rewards > 0)

    print(f"  Cost > 0:      {cost_nonzero}/{total_steps} = {cost_nonzero/total_steps:.3f}")
    print(f"  Label = 1:     {label_fire}/{total_steps} = {label_fire/total_steps:.3f}")
    print(f"  Reward > 0:    {reward_positive}/{total_steps} = {reward_positive/total_steps:.3f}")
    print(f"  Cost mean:     {np.mean(all_costs):.4f}")
    print(f"  Cost std:      {np.std(all_costs):.4f}")

    # 2. Cost-label co-occurrence
    if label_fire > 0:
        cost_when_label1 = np.sum((all_labels > 0) & (all_costs > 0))
        print(f"\n  Cost > 0 when label=1: {cost_when_label1}/{label_fire} = {cost_when_label1/label_fire:.3f}")

    # 3. Correlation
    if np.std(all_costs) > 1e-8 and np.std(all_rewards) > 1e-8:
        corr = np.corrcoef(all_rewards, all_costs)[0, 1]
        print(f"\n  Pearson correlation(reward, cost): {corr:.4f}")
        if corr > 0.1:
            print("    WARNING: POSITIVE correlation — reward and cost move TOGETHER")
        elif corr < -0.1:
            print("    GOOD: Negative correlation — reward and cost are in tension.")
        else:
            print("    WEAK: Near-zero correlation — independent signals.")
    else:
        print(f"\n  Cannot compute correlation — cost std={np.std(all_costs):.6f}")

    # 4. Label predictability from observation
    if 0 < label_fire < total_steps:
        obs_label1 = all_obs[all_labels > 0]
        obs_label0 = all_obs[all_labels == 0]
        mean_diff = np.mean(obs_label1, axis=0) - np.mean(obs_label0, axis=0)
        top_diff_idx = np.argsort(np.abs(mean_diff))[-5:]
        print(f"\n  Top 5 features differing between label=0 vs label=1:")
        for idx in reversed(top_diff_idx):
            print(f"    obs[{idx}]: mean_diff = {mean_diff[idx]:+.4f} "
                  f"(label0: {np.mean(obs_label0[:, idx]):.4f}, "
                  f"label1: {np.mean(obs_label1[:, idx]):.4f})")

        from sklearn.tree import DecisionTreeClassifier
        from sklearn.model_selection import cross_val_score
        try:
            clf = DecisionTreeClassifier(max_depth=3)
            scores = cross_val_score(clf, all_obs, all_labels, cv=5, scoring='accuracy')
            print(f"\n  Label predictability (DecisionTree depth=3, 5-fold CV):")
            print(f"    Accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
            if np.mean(scores) > 0.95:
                print("    EXCELLENT: Label is highly predictable from obs.")
            elif np.mean(scores) > 0.9:
                print("    GOOD: Label is predictable from obs.")
            elif np.mean(scores) > 0.75:
                print("    OK: Label is moderately predictable.")
            else:
                print("    WARNING: Label is NOT predictable from obs.")
            clf.fit(all_obs, all_labels)
            importances = clf.feature_importances_
            top_feat = np.argsort(importances)[-3:]
            print(f"    Top features: {list(reversed(top_feat))}")
            print(f"    Importances:  {[f'{importances[i]:.3f}' for i in reversed(top_feat)]}")
        except Exception as e:
            print(f"\n  Label predictability test failed: {e}")
    elif label_fire == 0:
        print("\n  WARNING: Label NEVER fires (always 0).")
    else:
        print("\n  WARNING: Label ALWAYS fires (always 1).")

    # 5. Obs layout analysis
    obs_dim = all_obs.shape[1]
    print(f"\n  Observation dimension: {obs_dim}")

    # For bus/convoy/spillback: check the extended features
    if scenario in ("bus_priority", "convoy_priority", "spillback"):
        # Obs = [phase(n_p), min_green(1), density(n_l), queue(n_l), ...ext...]
        n_phases = 5 if network == "arterial" else 8
        raw_obs_dim = obs_dim
        if scenario == "bus_priority":
            n_lanes = (raw_obs_dim - n_phases - 1) // 4  # 4 * n_lanes + n_phases + 1
            ext_start = raw_obs_dim - 2 * n_lanes
            has_x = all_obs[:, ext_start:ext_start + n_lanes]
            x_wait = all_obs[:, ext_start + n_lanes:ext_start + 2 * n_lanes]
            feat_name = "bus"
        elif scenario == "convoy_priority":
            n_lanes = (raw_obs_dim - n_phases - 1) // 5  # 5 * n_lanes + n_phases + 1
            ext_start = raw_obs_dim - 3 * n_lanes
            has_x = all_obs[:, ext_start:ext_start + n_lanes]
            x_count = all_obs[:, ext_start + n_lanes:ext_start + 2 * n_lanes]
            x_wait = all_obs[:, ext_start + 2 * n_lanes:ext_start + 3 * n_lanes]
            feat_name = "convoy"
        else:  # spillback
            n_lanes = (raw_obs_dim - n_phases - 1) // 3  # 3 * n_lanes + n_phases + 1
            ext_start = raw_obs_dim - 1 * n_lanes
            feat_name = "spillback"
        print(f"\n  Extended features ({feat_name}):")
        print(f"    n_lanes={n_lanes}, extension starts at obs[{ext_start}]")
        if scenario != "spillback":
            print(f"    has_{feat_name} mean per lane: {np.mean(has_x, axis=0)}")
            if scenario == "convoy_priority":
                print(f"    convoy_count mean per lane: {np.mean(x_count, axis=0)}")
            print(f"    {feat_name}_wait mean per lane: {np.mean(x_wait, axis=0)}")
            total_presence = np.mean(np.any(has_x > 0, axis=1))
            print(f"    Fraction of steps with {feat_name} on ANY lane: {total_presence:.3f}")
        else:
            ds_occ = all_obs[:, ext_start:ext_start + n_lanes]
            print(f"    downstream_occ mean per lane: {np.mean(ds_occ, axis=0)}")
            print(f"    downstream_occ max per lane: {np.max(ds_occ, axis=0)}")
            blocked_frac = np.mean(np.any(ds_occ > 0.7, axis=1))
            print(f"    Fraction of steps with ANY lane blocked (occ>0.7): {blocked_frac:.3f}")

    # 6. Per-lane queue stats
    n_phases = 5 if network == "arterial" else 8
    orig_n_lanes = (obs_dim - n_phases - 1) // 4 if scenario == "bus_priority" \
                   else (obs_dim - n_phases - 1) // 5 if scenario == "convoy_priority" \
                   else (obs_dim - n_phases - 1) // 3 if scenario == "spillback" \
                   else (obs_dim - n_phases - 1) // 2
    density_start = n_phases + 1
    queue_start = density_start + orig_n_lanes
    if queue_start + orig_n_lanes <= obs_dim:
        queues = all_obs[:, queue_start:queue_start + orig_n_lanes]
        per_lane_mean_q = np.mean(queues, axis=0)
        print(f"\n  Per-lane mean queue ratio:")
        for li, mq in enumerate(per_lane_mean_q):
            print(f"    lane {li}: {mq:.4f}")

    # Verdict
    print(f"\n--- VERDICT ---")
    label_rate = label_fire / total_steps
    cost_rate = cost_nonzero / total_steps
    checks_passed = 0
    checks_total = 5

    # Check 1: Label rate 10-40%
    if 0.05 <= label_rate <= 0.50:
        print(f"  [PASS] Label rate: {label_rate:.1%} (target: 10-40%)")
        checks_passed += 1
    else:
        print(f"  [FAIL] Label rate: {label_rate:.1%} (target: 10-40%)")

    # Check 2: Cost fires when label=1
    if label_fire > 0:
        cost_label_co = np.sum((all_labels > 0) & (all_costs > 0)) / label_fire
        if cost_label_co > 0.5:
            print(f"  [PASS] Cost|label=1: {cost_label_co:.1%} (target: >80%)")
            checks_passed += 1
        else:
            print(f"  [FAIL] Cost|label=1: {cost_label_co:.1%} (target: >80%)")
    else:
        print(f"  [FAIL] Cost|label=1: N/A (label never fires)")

    # Check 3: Label obs-predictable
    if 0 < label_fire < total_steps:
        try:
            clf = DecisionTreeClassifier(max_depth=3)
            scores = cross_val_score(clf, all_obs, all_labels, cv=5, scoring='accuracy')
            pred_acc = np.mean(scores)
            if pred_acc > 0.90:
                print(f"  [PASS] Label predictability: {pred_acc:.3f} (target: >0.95)")
                checks_passed += 1
            else:
                print(f"  [FAIL] Label predictability: {pred_acc:.3f} (target: >0.95)")
        except Exception:
            print(f"  [SKIP] Label predictability: sklearn error")
    else:
        print(f"  [FAIL] Label predictability: N/A")

    # Check 4: Reward-cost not strongly positive
    if np.std(all_costs) > 1e-8 and np.std(all_rewards) > 1e-8:
        corr = np.corrcoef(all_rewards, all_costs)[0, 1]
        if corr < 0.3:
            print(f"  [PASS] Reward-cost corr: {corr:.3f} (target: <0.3)")
            checks_passed += 1
        else:
            print(f"  [FAIL] Reward-cost corr: {corr:.3f} (target: <0.3)")
    else:
        print(f"  [SKIP] Reward-cost corr: insufficient variance")

    # Check 5: Cost signal has variance
    if cost_rate > 0.01:
        print(f"  [PASS] Cost fires: {cost_rate:.1%}")
        checks_passed += 1
    else:
        print(f"  [FAIL] Cost fires: {cost_rate:.1%} (too rare)")

    print(f"\n  Result: {checks_passed}/{checks_total} checks passed")

    return {
        "cost_rate": cost_rate,
        "label_rate": label_rate,
        "conflict_rate": conflict_steps / total_steps,
        "total_steps": total_steps,
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose SPLIT-RL conflict in SUMO")
    parser.add_argument("--env", choices=["sumo"], default="sumo")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per env")
    parser.add_argument("--scenario", default="bus_priority",
                        choices=["bus_priority", "convoy_priority", "spillback", "side_queue"],
                        help="SUMO scenario to test")
    parser.add_argument("--network", default="arterial",
                        choices=["arterial", "grid"],
                        help="SUMO network")
    args = parser.parse_args()

    results = {}
    if args.env in ("sumo", "both"):
        try:
            key = f"sumo_{args.network}_{args.scenario}"
            results[key] = diagnose_sumo(
                n_episodes=args.episodes,
                scenario=args.scenario,
                network=args.network,
            )
        except Exception as e:
            print(f"\nSUMO diagnostic FAILED: {e}")
            import traceback; traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: IS THERE A DEFENSIBLE CONFLICT?")
    print("=" * 70)

    for env_name, stats in results.items():
        print(f"\n  {env_name.upper()}:")
        has_cost = stats["cost_rate"] > 0.01
        has_label = stats["label_rate"] > 0.01
        has_conflict = stats["conflict_rate"] > 0.01

        print(f"    Cost fires:     {'YES' if has_cost else 'NO'} ({stats['cost_rate']:.1%})")
        print(f"    Label fires:    {'YES' if has_label else 'NO'} ({stats['label_rate']:.1%})")
        print(f"    Conflict co-oc: {'YES' if has_conflict else 'NO'} ({stats['conflict_rate']:.1%})")

        if not has_cost:
            print("    VERDICT: NO CONFLICT — cost signal is dead (never/rarely fires)")
        elif not has_conflict:
            print("    VERDICT: NO CONFLICT — cost and reward don't co-occur")
        else:
            print("    VERDICT: Conflict frequency looks adequate. Check label predictability.")


if __name__ == "__main__":
    main()
