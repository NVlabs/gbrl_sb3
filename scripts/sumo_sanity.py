#!/usr/bin/env python3.10
"""
Sanity-check script: train split_rl and cup on sumo-single-vhvh-v0 with
convoy/bus/premium scenarios. Short runs (100k steps ≈ 5-8 min each).

Outputs per-step debug CSV + episode summary to a local directory.
No wandb. No sweeps. Just raw data for diagnosis.

Usage:
    cd /swgwork/bfuhrer/projects/gbrl_project/nvlabs/gbrl_sb3
    python3.10 scripts/sumo_sanity.py              # all 6 runs (3 scenarios × 2 algos)
    python3.10 scripts/sumo_sanity.py --scenario convoy_priority --algo split_rl  # single run
    python3.10 scripts/sumo_sanity.py --scenario convoy_priority  # both algos, one scenario
    python3.10 scripts/sumo_sanity.py --steps 50000  # shorter run
"""
import os, sys, csv, time, argparse, json
os.environ["LIBSUMO_AS_TRACI"] = "1"
# Ensure project root is on path regardless of cwd
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import warnings
warnings.filterwarnings("ignore")

import numpy as np

# ── scenario definitions ──
SCENARIOS = {
    "convoy_priority": {
        "cost_fn": "convoy_priority",
        "convoy_injection_interval": 80.0,
        "convoy_size_min": 10,
        "convoy_size_max": 12,
        "convoy_headway": 2.0,
        "clean_episode_prob": 0.0,
    },
    "bus_priority": {
        "cost_fn": "bus_priority",
        "bus_injection_interval": 60.0,
        "bus_cost_threshold": 30.0,
        "bus_warn_threshold": 10.0,
        "clean_episode_prob": 0.0,
    },
    "premium_priority": {
        "cost_fn": "premium_priority",
        "premium_injection_interval": 40.0,
        "premium_cost_threshold": 15.0,
        "premium_warn_threshold": 5.0,
        "clean_episode_prob": 0.0,
    },
}

# ── fixed hyperparams (from seed sweep yaml) ──
SPLIT_RL_HP = dict(
    clip_range=0.2,
    normalize_advantage=True,
    n_epochs=10,
    n_steps=512,
    batch_size=256,
    gae_lambda=0.9,
    gamma=0.95,
    ent_coef=0.0010004088576321022,
    vf_coef=0.871923,
    total_n_steps=100_000,
    fixed_std=False,
    log_std_lr="lin_0.0017",
    min_log_std_lr=0.000475,
    device="cuda",
    seed=0,
    verbose=0,
    policy_kwargs={
        "log_std_init": -1.0,
        "squash": False,
        "nn_critic": False,
        "shared_tree_struct": True,
        "tree_struct": {
            "max_depth": 4,
            "n_bins": 256,
            "min_data_in_leaf": 0,
            "par_th": 0,
            "grow_policy": "greedy",
        },
        "tree_optimizer": {
            "params": {
                "split_score_func": "cosine",
                "control_variates": False,
                "generator_type": "Quantile",
                "n_objs": 2,
            },
            "policy_optimizer": {
                "policy_algo": "SGD",
                "policy_lr": 0.012136977141261192,
                "policy_beta_1": 0.9,
                "policy_beta_2": 0.999,
                "policy_eps": 1e-5,
                "policy_shrinkage": 0.0,
            },
            "value_optimizer": {
                "value_algo": "SGD",
                "value_lr": 0.007129579195129064,
                "value_beta_1": 0.9,
                "value_beta_2": 0.999,
                "value_eps": 1e-5,
                "value_shrinkage": 0.0,
            },
            "cost_optimizer": {
                "cost_algo": "SGD",
                "cost_lr": 0.01212642125305808,
                "cost_beta_1": 0.9,
                "cost_beta_2": 0.999,
                "cost_eps": 1e-5,
                "cost_shrinkage": 0.0,
            },
        },
    },
)

CUP_HP = dict(
    learning_rate=0.000029269937233695252,
    n_steps=512,
    batch_size=256,
    n_epochs=10,
    gamma=0.9,
    gae_lambda=0.9,
    clip_range=0.2,
    clip_range_vf=None,
    clip_range_cf=0.3,
    normalize_advantage=True,
    ent_coef=0.00004493884851166288,
    vf_coef=0.25,
    max_grad_norm=0.30721293560142793,
    use_sde=False,
    sde_sample_freq=-1,
    stats_window_size=100,
    verbose=0,
    seed=0,
    device="cuda",
    cost_limit=25,
    lagrangian_multiplier_init=1.0,
    lambda_lr=0.0001827025046620937,
    lambda_optimizer="Adam",
    lagrangian_upper_bound=10.0,
    cf_coef=0.25,
    cup_target_kl=None,
    cup_update_iters=None,
    policy_kwargs=None,
)


def make_env(scenario_name, debug=True):
    """Create sumo-single-vhvh-v0 with the given scenario and debug=True."""
    from env.sumo import make_sumo_vec_env
    kw = SCENARIOS[scenario_name].copy()
    kw["override_reward"] = None
    kw["mainline_direction"] = "ns"
    kw["debug"] = debug
    env = make_sumo_vec_env(env_name="sumo-single-vhvh-v0", n_envs=1, seed=0, **kw)
    return env


class DebugCSVCallback:
    """SB3 callback that writes per-step debug info to a CSV file.

    Hooks into the VecEnv wrapper by reading info dicts after each
    env.step() call inside the rollout collection loop.
    """
    def __init__(self, csv_path, cost_fn):
        self.csv_path = csv_path
        self.cost_fn = cost_fn
        self.fp = None
        self.writer = None
        self.ep_count = 0
        self.step_count = 0
        # Episode accumulators
        self.ep_reward = 0.0
        self.ep_cost = 0.0
        self.ep_label_sum = 0
        self.ep_steps = 0

    def _get_fieldnames(self):
        base = [
            "step", "episode", "raw_reward", "norm_reward", "cost", "label",
            "pressure", "queued", "max_wait",
            "phase", "can_switch", "time_in_phase", "sim_time",
        ]
        if self.cost_fn == "convoy_priority":
            base += ["has_convoy", "convoy_progress_max", "convoy_count_max", "n_active_convoys"]
        elif self.cost_fn == "bus_priority":
            base += ["has_bus", "bus_wait_max"]
        elif self.cost_fn == "premium_priority":
            base += ["has_premium", "premium_wait_max"]
        elif self.cost_fn == "spillback":
            base += ["downstream_occ_max"]
        return base

    def open(self):
        self.fp = open(self.csv_path, "w", newline="")
        self.writer = csv.DictWriter(self.fp, fieldnames=self._get_fieldnames())
        self.writer.writeheader()

    def close(self):
        if self.fp:
            self.fp.close()

    def on_step(self, info, reward):
        # Use the actual reward signal (pre-VecNormalize) from the env wrapper.
        # raw_reward is set in _process_step; original_reward is the sumo-rl
        # reward before override. Fall back to normalized reward as last resort.
        raw_rew = float(info.get("raw_reward", info.get("original_reward", reward)))
        row = {
            "step": self.step_count,
            "episode": self.ep_count,
            "raw_reward": f"{raw_rew:.4f}",
            "norm_reward": f"{reward:.4f}",
            "cost": info.get("cost", 0.0),
            "label": info.get("safety_label", 0),
            "pressure": info.get("pressure", 0),
            "queued": info.get("queued", 0),
            "max_wait": f"{info.get('max_wait', 0.0):.1f}",
            "phase": info.get("dbg_phase", -1),
            "can_switch": info.get("dbg_can_switch", -1),
            "time_in_phase": f"{info.get('dbg_time_in_phase', 0.0):.1f}",
            "sim_time": f"{info.get('dbg_sim_time', 0.0):.1f}",
        }
        if self.cost_fn == "convoy_priority":
            row["has_convoy"] = info.get("dbg_has_convoy", 0)
            row["convoy_progress_max"] = f"{info.get('dbg_convoy_progress_max', 0.0):.3f}"
            row["convoy_count_max"] = f"{info.get('dbg_convoy_count_max', 0.0):.3f}"
            row["n_active_convoys"] = info.get("dbg_n_active_convoys", 0)
        elif self.cost_fn == "bus_priority":
            row["has_bus"] = info.get("dbg_has_bus", 0)
            row["bus_wait_max"] = f"{info.get('dbg_bus_wait_max', 0.0):.3f}"
        elif self.cost_fn == "premium_priority":
            row["has_premium"] = info.get("dbg_has_premium", 0)
            row["premium_wait_max"] = f"{info.get('dbg_premium_wait_max', 0.0):.3f}"
        elif self.cost_fn == "spillback":
            row["downstream_occ_max"] = f"{info.get('dbg_downstream_occ_max', 0.0):.3f}"

        self.writer.writerow(row)
        self.step_count += 1
        self.ep_reward += raw_rew
        self.ep_cost += info.get("cost", 0.0)
        self.ep_label_sum += info.get("safety_label", 0)
        self.ep_steps += 1

    def on_episode_end(self):
        cost_rate = self.ep_cost / max(self.ep_steps, 1) * 100
        label_rate = self.ep_label_sum / max(self.ep_steps, 1) * 100
        print(f"  EP {self.ep_count:3d} | steps={self.ep_steps:4d} "
              f"rew={self.ep_reward:8.1f} cost={self.ep_cost:6.1f} "
              f"cost_rate={cost_rate:5.1f}% label_rate={label_rate:5.1f}%")
        self.ep_count += 1
        self.ep_reward = 0.0
        self.ep_cost = 0.0
        self.ep_label_sum = 0
        self.ep_steps = 0


def _fix_supersuit_dones(vec_env):
    """Patch SuperSuit int dones → bool so VecNormalize doesn't crash.

    SuperSuit returns dones as int (0/1). VecNormalize does
    ``self.returns[dones] = 0`` which uses integer indexing for int dones
    (index 1 → out of bounds on num_envs=1) instead of boolean masking.
    """
    inner = vec_env.venv  # VecCostMonitor under VecNormalize
    original_sw = inner.step_wait

    def fixed_step_wait():
        obs, rewards, dones, infos = original_sw()
        return obs, rewards, np.asarray(dones, dtype=bool), infos

    inner.step_wait = fixed_step_wait


def _patch_env_for_debug(env, callback):
    """Monkey-patch env.step_wait() to intercept per-step info for logging."""
    original_step_wait = env.step_wait

    def patched_step_wait():
        obs, rewards, dones, infos = original_step_wait()
        for i in range(len(infos)):
            callback.on_step(infos[i], float(rewards[i]))
            if dones[i]:
                callback.on_episode_end()
        return obs, rewards, dones, infos

    env.step_wait = patched_step_wait
    return env


def run_training_intercepted(algo_name, scenario_name, total_steps, out_dir):
    """Run one algo on one scenario with step-level debug interception."""
    from stable_baselines3.common.vec_env import VecNormalize

    tag = f"{algo_name}_{scenario_name}"
    csv_path = os.path.join(out_dir, f"{tag}_steps.csv")

    print(f"\n{'='*70}")
    print(f"  {algo_name.upper()} × {scenario_name} — {total_steps} steps")
    print(f"  CSV: {csv_path}")
    print(f"{'='*70}")

    env = make_env(scenario_name, debug=True)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=0.95)
    _fix_supersuit_dones(env)  # SuperSuit int dones → bool

    cost_fn = SCENARIOS[scenario_name]["cost_fn"]
    dbg = DebugCSVCallback(csv_path, cost_fn)
    dbg.open()

    # Monkey-patch step_wait to intercept per-step info
    env = _patch_env_for_debug(env, dbg)

    if algo_name == "split_rl":
        from algos.split_rl import SPLIT_RL
        hp = SPLIT_RL_HP.copy()
        hp["total_n_steps"] = total_steps
        algo = SPLIT_RL(env=env, _init_setup_model=True,
                        safety_mode=True, **hp)
    elif algo_name == "cup":
        from algos.safety import CUP
        from policies.cost_actor_critic import CostActorCriticPolicy
        hp = CUP_HP.copy()
        algo = CUP(policy=CostActorCriticPolicy, env=env, **hp)
    else:
        raise ValueError(f"Unknown algo: {algo_name}")

    t0 = time.time()
    # Use standard learn() — monkey-patched step_wait captures per-step info
    algo.learn(total_timesteps=total_steps, log_interval=1, progress_bar=False)
    elapsed = time.time() - t0

    dbg.close()
    env.close()

    # ── Print summary from CSV ──
    print(f"\n  Completed in {elapsed:.1f}s ({total_steps/max(elapsed,1):.0f} steps/s)")
    _print_csv_summary(csv_path, scenario_name)
    return csv_path


def _print_csv_summary(csv_path, scenario_name):
    """Read the debug CSV and print conflict/label summary."""
    import pandas as pd

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  Could not read CSV: {e}")
        return

    n = len(df)
    if n == 0:
        print("  No data collected!")
        return

    n_cost = (df["cost"] > 0).sum()
    n_label = (df["label"] > 0).sum()
    cost_rate = n_cost / n * 100
    label_rate = n_label / n * 100

    print(f"\n  ── SUMMARY ({n} steps) ──")
    print(f"  Cost>0:  {n_cost:5d} ({cost_rate:.1f}%)")
    print(f"  Label=1: {n_label:5d} ({label_rate:.1f}%)")

    if n_cost > 0:
        r_c1 = df.loc[df["cost"] > 0, "raw_reward"].astype(float).mean()
        r_c0 = df.loc[df["cost"] == 0, "raw_reward"].astype(float).mean()
        delta = r_c1 - r_c0
        eps = 0.05 * max(abs(r_c1), abs(r_c0), 1.0)
        greedy_conflict = delta > eps    # greedy policy → violations (expected)
        inverse_coupling = delta < -eps  # cost states have lower reward
        print(f"  Mean raw_rew (cost>0): {r_c1:.4f}")
        print(f"  Mean raw_rew (cost=0): {r_c0:.4f}")
        print(f"  Δ = {delta:.4f}  (ε = {eps:.4f})")
        if greedy_conflict:
            print(f"  ✓ GREEDY→COST conflict (reward-maximizing causes violations)")
        elif inverse_coupling:
            print(f"  ~ inverse coupling (cost states have low reward — aligned, not conflicting)")
        else:
            print(f"  ✗ no significant conflict (|Δ| < ε)")

    # ── Label quality: horizon-based matching ──
    # Labels are anticipatory (fire T_warn < T_cost), so same-step F1 is
    # wrong — it penalises correct early labels as false positives.
    # Instead: label at t is a TP if cost>0 in [t, t+H].
    #          cost at t is covered if label>0 in [t-H, t].
    if n_cost > 0:
        # Horizon H in steps: (T_cost - T_warn) / delta_time.
        # Convoy: progress-based (~5-10 steps), bus: (30-10)/5=4, premium: (15-5)/5=2.
        HORIZON_MAP = {
            "convoy_priority": 8,
            "bus_priority": 4,
            "premium_priority": 2,
            "spillback": 4,
        }
        H = HORIZON_MAP.get(scenario_name, 4)
        cost_arr = (df["cost"] > 0).values.astype(int)
        label_arr = (df["label"] > 0).values.astype(int)
        N = len(cost_arr)

        # For each label=1 step, check if cost>0 anywhere in [t, t+H]
        label_tp = 0
        label_fp = 0
        for t in range(N):
            if label_arr[t]:
                window = cost_arr[t:min(t + H + 1, N)]
                if window.any():
                    label_tp += 1
                else:
                    label_fp += 1
        # For each cost>0 step, check if label>0 anywhere in [t-H, t]
        cost_covered = 0
        cost_uncovered = 0
        for t in range(N):
            if cost_arr[t]:
                window = label_arr[max(t - H, 0):t + 1]
                if window.any():
                    cost_covered += 1
                else:
                    cost_uncovered += 1

        prec = label_tp / max(label_tp + label_fp, 1)
        rec = cost_covered / max(cost_covered + cost_uncovered, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)
        print(f"  Label (H={H}): P={prec:.3f} R={rec:.3f} F1={f1:.3f}"
              f"  (TP={label_tp} FP={label_fp} covered={cost_covered} uncov={cost_uncovered})")

    # Per-episode summary
    eps = df.groupby("episode").agg(
        steps=("step", "count"),
        total_rew=("raw_reward", lambda x: x.astype(float).sum()),
        total_cost=("cost", "sum"),
        label_rate=("label", "mean"),
    )
    print(f"\n  ── PER-EPISODE ({len(eps)} episodes) ──")
    print(f"  {'EP':>4s} {'Steps':>6s} {'Reward':>9s} {'Cost':>7s} {'CostRate':>9s} {'LabelRate':>10s}")
    for ep_idx, row in eps.iterrows():
        cr = row["total_cost"] / max(row["steps"], 1) * 100
        lr = row["label_rate"] * 100
        print(f"  {int(ep_idx):4d} {int(row['steps']):6d} {row['total_rew']:9.1f} "
              f"{row['total_cost']:7.1f} {cr:8.1f}% {lr:9.1f}%")

    # Learning signal: compare first half vs second half
    half = n // 2
    if half > 100:
        first_rew = df.loc[:half, "raw_reward"].astype(float).mean()
        second_rew = df.loc[half:, "raw_reward"].astype(float).mean()
        first_cost_rate = (df.loc[:half, "cost"] > 0).mean() * 100
        second_cost_rate = (df.loc[half:, "cost"] > 0).mean() * 100
        print(f"\n  ── LEARNING TREND ──")
        print(f"  First half:  avg_rew={first_rew:.4f}  cost_rate={first_cost_rate:.1f}%")
        print(f"  Second half: avg_rew={second_rew:.4f}  cost_rate={second_cost_rate:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="SUMO scenario sanity check")
    parser.add_argument("--scenario", type=str, default=None,
                        choices=list(SCENARIOS.keys()),
                        help="Single scenario to run (default: all 3)")
    parser.add_argument("--algo", type=str, default=None,
                        choices=["split_rl", "cup"],
                        help="Single algo to run (default: both)")
    parser.add_argument("--steps", type=int, default=100_000,
                        help="Total training steps per run (default: 100k)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory (default: /tmp/sumo_sanity_<timestamp>)")
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else list(SCENARIOS.keys())
    algos = [args.algo] if args.algo else ["split_rl", "cup"]

    out_dir = args.out or f"/tmp/sumo_sanity_{int(time.time())}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")
    print(f"Scenarios: {scenarios}")
    print(f"Algos: {algos}")
    print(f"Steps per run: {args.steps}")
    print(f"Total runs: {len(scenarios) * len(algos)}")

    results = {}
    for scenario in scenarios:
        for algo in algos:
            csv_path = run_training_intercepted(algo, scenario, args.steps, out_dir)
            results[f"{algo}_{scenario}"] = csv_path

    print(f"\n\n{'='*70}")
    print(f"  ALL DONE — {len(results)} runs completed")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}")
    for tag, path in results.items():
        print(f"  {tag}: {path}")


if __name__ == "__main__":
    main()
