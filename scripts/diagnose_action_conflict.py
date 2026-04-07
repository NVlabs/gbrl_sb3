#!/usr/bin/env python3.10
"""
ACTION-LEVEL conflict diagnostic for SUMO Split-RL scenarios.

The hard question: in states where label=1, does the reward-only policy
actually choose actions that hurt cost?  If not, there is no conflict —
no label trick can save the scenario.

Protocol:
  Phase 1 — Train a reward-only PPO (no cost, no label) for N steps.
  Phase 2 — Roll out the trained policy, collecting per-step:
             obs, action, reward, cost, label.
  Phase 3 — In label=1 states, measure:
             (a) mean cost incurred by reward-only policy (should be HIGH)
             (b) mean cost incurred by random policy (baseline)
             (c) action distribution: does reward-only concentrate on a
                 DIFFERENT action than cost would prefer?
             (d) reward_gap: how much reward does the reward-only policy
                 earn in label=1 states vs label=0 states?

Usage:
  python3.10 scripts/diagnose_action_conflict.py --scenario convoy_priority --network arterial
  python3.10 scripts/diagnose_action_conflict.py --scenario spillback --network grid
  python3.10 scripts/diagnose_action_conflict.py --scenario premium_priority --network arterial
  python3.10 scripts/diagnose_action_conflict.py --scenario all --network arterial
"""
import argparse
import os
import sys
import warnings
import numpy as np
from collections import defaultdict

# Ensure project root is on the path (for running from any directory)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

warnings.filterwarnings("ignore")
os.environ.setdefault("LIBSUMO_AS_TRACI", "1")

SCENARIOS = ["convoy_priority", "spillback", "premium_priority"]

SCENARIO_KWARGS = {
    "convoy_priority": {
        "override_reward": None,
        "cost_fn": "convoy_priority",
        "convoy_injection_interval": 120.0,
        "convoy_size_min": 5,
        "convoy_size_max": 8,
        "clean_episode_prob": 0.0,  # no clean episodes for diagnostic
    },
    "spillback": {
        "override_reward": None,
        "cost_fn": "spillback",
        "roadwork_interval_mean": 300.0,
        "roadwork_duration_mean": 400.0,
        "roadwork_speed": 0.3,
        "spillback_occ_threshold": 0.5,
        "clean_episode_prob": 0.0,
    },
    "premium_priority": {
        "override_reward": None,
        "cost_fn": "premium_priority",
        "premium_injection_interval": 40.0,
        "premium_cost_threshold": 15.0,
        "premium_warn_threshold": 5.0,
        "clean_episode_prob": 0.0,
    },
}


def make_env(scenario: str, network: str, seed: int = 42):
    """Create a SUMO vec env for the given scenario."""
    from env.sumo import make_sumo_vec_env

    env_name = f"sumo-{network}4x4-v0"
    kwargs = SCENARIO_KWARGS[scenario].copy()
    return make_sumo_vec_env(env_name=env_name, n_envs=1, seed=seed, **kwargs)


def train_reward_only(scenario: str, network: str, n_steps: int = 200_000,
                      seed: int = 42):
    """Train a reward-only PPO-NN agent (no cost gradient).

    Uses the same VanillaPPO (SB3 PPO wrapper) with MlpPolicy that the
    ppo_nn sweep configs use. This is an NN baseline — no tree structure.
    """
    from stable_baselines3.common.vec_env import VecNormalize
    from algos.safety.ppo import VanillaPPO

    env = make_env(scenario, network, seed=seed)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=0.95)

    model = VanillaPPO(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        n_steps=256,
        batch_size=128,
        n_epochs=10,
        gamma=0.95,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        learning_rate=3e-4,
        normalize_advantage=True,
        device="cuda",
    )

    print(f"  Training reward-only PPO-NN for {n_steps} steps on {network}4x4 / {scenario}...")
    model.learn(total_timesteps=n_steps)
    env.close()
    print(f"  Training complete.")
    return model


def collect_rollouts(model, scenario: str, network: str,
                     n_episodes: int = 5, seed: int = 123,
                     use_random: bool = False):
    """Roll out a policy (or random) and collect (obs, action, reward, cost, label)."""
    from stable_baselines3.common.vec_env import VecNormalize

    env = make_env(scenario, network, seed=seed)
    # Match training normalization (norm_reward=False during eval so raw rewards are visible)
    env = VecNormalize(env, norm_obs=False, norm_reward=False, gamma=0.95)

    n_agents = env.num_envs
    records = []  # list of dicts
    ep_count = 0

    obs = env.reset()

    while ep_count < n_episodes:
        if use_random:
            actions = np.array([env.action_space.sample() for _ in range(n_agents)])
        else:
            actions, _ = model.predict(obs, deterministic=True)

        next_obs, rewards, dones, infos = env.step(actions)

        for i in range(n_agents):
            records.append({
                "obs": obs[i].copy(),
                "action": int(actions[i]) if np.ndim(actions) > 0 else int(actions),
                "reward": float(rewards[i]),
                "cost": float(infos[i].get("cost", 0.0)),
                "label": int(infos[i].get("safety_label", 0)),
            })

        obs = next_obs
        if any(dones):
            ep_count += 1

    env.close()
    return records


def analyze_conflict(records_policy, records_random, scenario: str):
    """Compare reward-only policy vs random in label=1 states."""
    import pandas as pd

    df_pol = pd.DataFrame(records_policy)
    df_rnd = pd.DataFrame(records_random)

    n_total = len(df_pol)
    n_label1 = (df_pol["label"] == 1).sum()
    n_label0 = (df_pol["label"] == 0).sum()
    n_cost_pos = (df_pol["cost"] > 0).sum()

    print(f"\n{'='*70}")
    print(f"ACTION-LEVEL CONFLICT ANALYSIS: {scenario.upper()}")
    print(f"{'='*70}")
    print(f"  Total steps:  {n_total}")
    print(f"  Label=1:      {n_label1} ({n_label1/n_total:.1%})")
    print(f"  Label=0:      {n_label0} ({n_label0/n_total:.1%})")
    print(f"  Cost>0:       {n_cost_pos} ({n_cost_pos/n_total:.1%})")

    if n_label1 == 0:
        print(f"\n  FATAL: Label NEVER fires. Cannot measure conflict.")
        return {"verdict": "NO_LABEL", "conflict_score": 0.0}

    # ── Metric 1: Mean cost in label=1 states ──
    pol_cost_l1 = df_pol.loc[df_pol["label"] == 1, "cost"].mean()
    pol_cost_l0 = df_pol.loc[df_pol["label"] == 0, "cost"].mean()
    rnd_cost_l1 = df_rnd.loc[df_rnd["label"] == 1, "cost"].mean() if (df_rnd["label"] == 1).any() else 0.0
    rnd_cost_l0 = df_rnd.loc[df_rnd["label"] == 0, "cost"].mean() if (df_rnd["label"] == 0).any() else 0.0

    print(f"\n  ── Mean cost by state type ──")
    print(f"  {'':>25s}  {'label=0':>10s}  {'label=1':>10s}")
    print(f"  {'Reward-only policy':>25s}  {pol_cost_l0:>10.4f}  {pol_cost_l1:>10.4f}")
    print(f"  {'Random policy':>25s}  {rnd_cost_l0:>10.4f}  {rnd_cost_l1:>10.4f}")

    # ── Metric 2: Does reward-only policy incur MORE cost than random in label=1? ──
    cost_excess = pol_cost_l1 - rnd_cost_l1
    print(f"\n  Cost excess (policy - random) in label=1 states: {cost_excess:+.4f}")
    if cost_excess > 0.01:
        print(f"    GOOD: Reward-only policy hurts cost MORE than random in conflict states.")
    elif cost_excess > -0.01:
        print(f"    NEUTRAL: Reward-only policy ≈ random in conflict states.")
    else:
        print(f"    BAD: Reward-only policy hurts cost LESS than random. No conflict.")

    # ── Metric 3: Reward comparison in label=1 vs label=0 ──
    pol_rew_l1 = df_pol.loc[df_pol["label"] == 1, "reward"].mean()
    pol_rew_l0 = df_pol.loc[df_pol["label"] == 0, "reward"].mean()
    rnd_rew_l1 = df_rnd.loc[df_rnd["label"] == 1, "reward"].mean() if (df_rnd["label"] == 1).any() else 0.0

    print(f"\n  ── Mean reward by state type ──")
    print(f"  {'':>25s}  {'label=0':>10s}  {'label=1':>10s}")
    print(f"  {'Reward-only policy':>25s}  {pol_rew_l0:>10.4f}  {pol_rew_l1:>10.4f}")
    print(f"  {'Random policy':>25s}  {rnd_cost_l0:>10.4f}  {rnd_rew_l1:>10.4f}")

    # ── Metric 4: Action distribution in label=1 vs label=0 ──
    n_actions = df_pol["action"].max() + 1
    pol_acts_l1 = df_pol.loc[df_pol["label"] == 1, "action"].value_counts(normalize=True).sort_index()
    pol_acts_l0 = df_pol.loc[df_pol["label"] == 0, "action"].value_counts(normalize=True).sort_index()

    print(f"\n  ── Action distribution (reward-only policy) ──")
    print(f"  {'action':>8s}  {'label=0':>10s}  {'label=1':>10s}  {'diff':>10s}")
    for a in range(n_actions):
        p0 = pol_acts_l0.get(a, 0.0)
        p1 = pol_acts_l1.get(a, 0.0)
        diff = p1 - p0
        marker = " <<<" if abs(diff) > 0.1 else ""
        print(f"  {a:>8d}  {p0:>10.3f}  {p1:>10.3f}  {diff:>+10.3f}{marker}")

    # Compute KL divergence between label=0 and label=1 action distributions
    p0_full = np.array([pol_acts_l0.get(a, 1e-8) for a in range(n_actions)])
    p1_full = np.array([pol_acts_l1.get(a, 1e-8) for a in range(n_actions)])
    p0_full = p0_full / p0_full.sum()
    p1_full = p1_full / p1_full.sum()
    kl_div = np.sum(p0_full * np.log(p0_full / p1_full))
    print(f"\n  KL(label=0 || label=1) action distributions: {kl_div:.4f}")
    if kl_div < 0.01:
        print(f"    WARNING: Policy takes SAME actions regardless of label state.")
        print(f"    The policy doesn't distinguish conflict from non-conflict states.")
    elif kl_div > 0.1:
        print(f"    GOOD: Policy takes significantly different actions in conflict states.")

    # ── Metric 5: In label=1, does the chosen action correlate with high cost? ──
    if n_label1 > 10:
        l1_data = df_pol[df_pol["label"] == 1]
        action_cost = l1_data.groupby("action")["cost"].agg(["mean", "count"])
        print(f"\n  ── Per-action cost in label=1 states ──")
        print(f"  {'action':>8s}  {'mean_cost':>10s}  {'count':>8s}")
        for a in action_cost.index:
            print(f"  {a:>8d}  {action_cost.loc[a, 'mean']:>10.4f}  {int(action_cost.loc[a, 'count']):>8d}")

        # Most chosen action's cost vs least chosen action's cost
        most_chosen = l1_data["action"].mode().iloc[0] if len(l1_data) > 0 else 0
        most_chosen_cost = l1_data.loc[l1_data["action"] == most_chosen, "cost"].mean()
        other_actions = l1_data.loc[l1_data["action"] != most_chosen, "cost"]
        other_cost = other_actions.mean() if len(other_actions) > 0 else 0.0

        print(f"\n  Most chosen action in label=1: {most_chosen}")
        print(f"    Its mean cost:      {most_chosen_cost:.4f}")
        print(f"    Other actions' cost: {other_cost:.4f}")
        cost_gap = most_chosen_cost - other_cost
        print(f"    Cost gap (chosen - other): {cost_gap:+.4f}")
        if cost_gap > 0.01:
            print(f"    GOOD: The reward-preferred action is costlier than alternatives.")
        else:
            print(f"    BAD: The reward-preferred action is NOT costlier.")

    # ── VERDICT ──
    print(f"\n{'='*70}")
    print(f"VERDICT: {scenario.upper()}")
    print(f"{'='*70}")

    conflict_score = 0.0

    # Score 1: Cost excess over random
    if cost_excess > 0.05:
        conflict_score += 0.3
        print(f"  [+0.3] Cost excess over random: {cost_excess:.4f} (strong)")
    elif cost_excess > 0.01:
        conflict_score += 0.15
        print(f"  [+0.15] Cost excess over random: {cost_excess:.4f} (moderate)")
    else:
        print(f"  [+0.0] Cost excess over random: {cost_excess:.4f} (weak/none)")

    # Score 2: Action distribution shift
    if kl_div > 0.1:
        conflict_score += 0.3
        print(f"  [+0.3] Action KL divergence: {kl_div:.4f} (strong)")
    elif kl_div > 0.02:
        conflict_score += 0.15
        print(f"  [+0.15] Action KL divergence: {kl_div:.4f} (moderate)")
    else:
        print(f"  [+0.0] Action KL divergence: {kl_div:.4f} (weak)")

    # Score 3: Label actually fires
    label_rate = n_label1 / n_total
    if 0.1 <= label_rate <= 0.5:
        conflict_score += 0.2
        print(f"  [+0.2] Label rate: {label_rate:.1%} (healthy range)")
    elif label_rate > 0.01:
        conflict_score += 0.1
        print(f"  [+0.1] Label rate: {label_rate:.1%} (outside ideal range)")
    else:
        print(f"  [+0.0] Label rate: {label_rate:.1%} (too low)")

    # Score 4: Cost-aware action gap
    if n_label1 > 10:
        if cost_gap > 0.02:
            conflict_score += 0.2
            print(f"  [+0.2] Chosen action costlier by: {cost_gap:.4f} (real conflict)")
        elif cost_gap > 0.005:
            conflict_score += 0.1
            print(f"  [+0.1] Chosen action costlier by: {cost_gap:.4f} (mild)")
        else:
            print(f"  [+0.0] Chosen action costlier by: {cost_gap:.4f} (no gap)")

    print(f"\n  CONFLICT SCORE: {conflict_score:.2f} / 1.00")
    if conflict_score >= 0.6:
        print(f"  → STRONG CONFLICT. Split-RL should have an advantage.")
    elif conflict_score >= 0.3:
        print(f"  → MODERATE CONFLICT. Split-RL may help, worth testing.")
    else:
        print(f"  → WEAK/NO CONFLICT. Split-RL unlikely to beat baselines.")
        print(f"    Consider redesigning this scenario.")

    return {
        "verdict": "STRONG" if conflict_score >= 0.6 else "MODERATE" if conflict_score >= 0.3 else "WEAK",
        "conflict_score": conflict_score,
        "cost_excess": cost_excess,
        "kl_div": kl_div,
        "label_rate": label_rate,
        "cost_gap": cost_gap if n_label1 > 10 else 0.0,
    }


def run_diagnostic(scenario: str, network: str, train_steps: int = 200_000,
                   eval_episodes: int = 5, seed: int = 42,
                   save_dir: str = "saved_models/diagnostic"):
    """Full diagnostic pipeline for one scenario."""
    print(f"\n{'#'*70}")
    print(f"# DIAGNOSTIC: {scenario} on {network}4x4")
    print(f"{'#'*70}")

    # Phase 1: Train reward-only PPO
    print(f"\n── Phase 1: Training reward-only PPO ──")
    model = train_reward_only(scenario, network, n_steps=train_steps, seed=seed)

    # Save the trained model for reuse
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"ppo_nn_reward_only_{scenario}_{network}4x4_seed{seed}")
    model.save(model_path)
    print(f"  Model saved to: {model_path}")

    # Phase 2: Collect rollouts
    print(f"\n── Phase 2: Collecting rollouts ──")
    print(f"  Reward-only policy rollout ({eval_episodes} episodes)...")
    records_policy = collect_rollouts(
        model, scenario, network, n_episodes=eval_episodes, seed=seed + 100
    )
    print(f"  Random policy rollout ({eval_episodes} episodes)...")
    records_random = collect_rollouts(
        model, scenario, network, n_episodes=eval_episodes, seed=seed + 200,
        use_random=True
    )

    # Phase 3: Analyze
    print(f"\n── Phase 3: Conflict analysis ──")
    result = analyze_conflict(records_policy, records_random, scenario)
    result["model_path"] = model_path
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Action-level conflict diagnostic for SUMO Split-RL"
    )
    parser.add_argument("--scenario",
                        choices=SCENARIOS + ["all"],
                        default="all",
                        help="Which scenario to diagnose")
    parser.add_argument("--network", default="arterial",
                        choices=["arterial", "grid"],
                        help="SUMO network")
    parser.add_argument("--train-steps", type=int, default=200000,
                        help="Steps to train reward-only PPO")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="Episodes for evaluation rollout")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str,
                        default="saved_models/diagnostic",
                        help="Directory to save trained reward-only models")
    args = parser.parse_args()

    scenarios = SCENARIOS if args.scenario == "all" else [args.scenario]
    results = {}

    for sc in scenarios:
        results[sc] = run_diagnostic(
            sc, args.network,
            train_steps=args.train_steps,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
            save_dir=args.save_dir,
        )

    # Final summary
    print(f"\n\n{'#'*70}")
    print(f"# FINAL SUMMARY — {args.network.upper()}4x4")
    print(f"{'#'*70}")
    print(f"\n  {'Scenario':<25s}  {'Score':>6s}  {'Verdict':<10s}  {'CostExcess':>11s}  {'KL':>8s}  {'LabelRate':>10s}")
    print(f"  {'-'*25}  {'-'*6}  {'-'*10}  {'-'*11}  {'-'*8}  {'-'*10}")
    for sc, res in results.items():
        print(f"  {sc:<25s}  {res['conflict_score']:>6.2f}  {res['verdict']:<10s}  "
              f"{res.get('cost_excess', 0):>+11.4f}  {res.get('kl_div', 0):>8.4f}  "
              f"{res.get('label_rate', 0):>10.1%}")

    # Recommendation
    strong = [sc for sc, r in results.items() if r["verdict"] == "STRONG"]
    moderate = [sc for sc, r in results.items() if r["verdict"] == "MODERATE"]
    weak = [sc for sc, r in results.items() if r["verdict"] == "WEAK"]

    if strong:
        print(f"\n  KEEP (strong conflict): {', '.join(strong)}")
    if moderate:
        print(f"  KEEP (moderate, worth testing): {', '.join(moderate)}")
    if weak:
        print(f"  DROP or REDESIGN: {', '.join(weak)}")

    print(f"\n  Saved reward-only models:")
    for sc, res in results.items():
        print(f"    {sc}: {res.get('model_path', 'N/A')}")


if __name__ == "__main__":
    main()
