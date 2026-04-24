"""Load a Split-RL checkpoint and run episodes, logging per-step details."""
import sys, os, argparse, json
import numpy as np

sys.stdout.reconfigure(line_buffering=True)  # Force line-buffered output
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def log(msg):
    print(msg, flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vecnorm_path", type=str, default=None)
    parser.add_argument("--cost_fn", type=str, default="convoy_priority")
    parser.add_argument("--clean_episode_prob", type=float, default=0.5)
    parser.add_argument("--n_episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from algos.split_rl import SPLIT_RL as SplitRL
    from env.sumo import make_sumo_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    # Create env the same way training does
    env = make_sumo_vec_env(
        env_name="sumo-arterial4x4-v0",
        n_envs=1,
        seed=args.seed,
        cost_fn=args.cost_fn,
        clean_episode_prob=args.clean_episode_prob,
    )

    # Load vecnormalize if available
    if args.vecnorm_path and os.path.exists(args.vecnorm_path):
        env = VecNormalize.load(args.vecnorm_path, env)
        env.training = False
        env.norm_reward = False
        log(f"Loaded VecNormalize from {args.vecnorm_path}")

    # Load model
    model = SplitRL.load(args.model_path, env=env, device="cuda")
    log(f"Loaded model from {args.model_path}")
    log(f"Cost fn: {args.cost_fn}, clean_episode_prob: {args.clean_episode_prob}")

    n_agents = env.num_envs  # 16 for grid4x4
    log(f"VecEnv num_envs (agents): {n_agents}")

    episodes_collected = 0
    obs = env.reset()
    step_count = 0

    # Per-agent accumulators
    agent_rewards = np.zeros(n_agents)
    agent_costs = np.zeros(n_agents)
    agent_label_counts = [{0: 0, 1: 0, "multi": 0} for _ in range(n_agents)]
    agent_cost_steps = [[] for _ in range(n_agents)]

    while episodes_collected < args.n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        step_count += 1

        for i in range(n_agents):
            r_i = float(rewards[i])
            c_i = infos[i].get("cost", 0.0)
            lbl = infos[i].get("safety_label", 0)

            agent_rewards[i] += r_i
            agent_costs[i] += c_i

            if isinstance(lbl, list):
                agent_label_counts[i]["multi"] += 1
            elif lbl == 1:
                agent_label_counts[i][1] += 1
            else:
                agent_label_counts[i][0] += 1

            if c_i > 0:
                agent_cost_steps[i].append({"step": step_count, "cost": c_i, "label": str(lbl), "reward": r_i})

        # Print summary every 100 steps (aggregate over agents)
        if step_count % 100 == 0:
            sum_r = float(rewards.sum())
            sum_c = sum(infos[i].get("cost", 0.0) for i in range(n_agents))
            labels_this = [infos[i].get("safety_label", 0) for i in range(n_agents)]
            n_lbl1 = sum(1 for l in labels_this if l == 1)
            n_multi = sum(1 for l in labels_this if isinstance(l, list))
            cum_r = float(agent_rewards.sum())
            cum_c = float(agent_costs.sum())
            log(f"  step={step_count:4d} sum_rew={sum_r:7.1f} sum_cost={sum_c:.1f} "
                f"lbl1={n_lbl1:2d} multi={n_multi:2d} "
                f"cum_rew={cum_r:8.0f} cum_cost={cum_c:6.0f}")

        # Log cost steps in detail (first 5 per episode to avoid spam)
        total_cost_logged = sum(len(cs) for cs in agent_cost_steps)
        if total_cost_logged <= 50:
            for i in range(n_agents):
                c_i = infos[i].get("cost", 0.0)
                if c_i > 0:
                    lbl = infos[i].get("safety_label", 0)
                    log(f"    COST agent={i:2d} step={step_count:4d} cost={c_i:.2f} "
                        f"label={lbl} rew={float(rewards[i]):.1f}")

        # Check for episode end (all agents done simultaneously in SUMO)
        if dones[0]:
            episodes_collected += 1
            # Collect ep_info from agents that have it
            ep_infos = [infos[i].get("episode", {}) for i in range(n_agents)]
            mean_r = np.mean([ei.get("r", 0) for ei in ep_infos if ei])
            mean_c = np.mean([ei.get("c", 0) for ei in ep_infos if ei])
            mean_orig_r = np.mean([ei.get("original_r", 0) for ei in ep_infos if ei])

            total_cost_events = sum(len(cs) for cs in agent_cost_steps)
            total_lbl1 = sum(lc[1] for lc in agent_label_counts)
            total_multi = sum(lc["multi"] for lc in agent_label_counts)
            total_lbl0 = sum(lc[0] for lc in agent_label_counts)

            log(f"\n{'='*60}")
            log(f"Episode {episodes_collected}")
            log(f"  Steps:              {step_count}")
            log(f"  Mean agent reward:  {mean_r:.1f}")
            log(f"  Mean agent cost:    {mean_c:.1f}")
            log(f"  Mean orig reward:   {mean_orig_r:.1f}")
            log(f"  Labels total:       lbl0={total_lbl0} lbl1={total_lbl1} multi={total_multi}")
            log(f"  Cost events total:  {total_cost_events} (across all agents)")
            if ep_infos[0]:
                log(f"  Agent 0 ep_info:    {ep_infos[0]}")
            log(f"{'='*60}\n")

            # Reset accumulators
            step_count = 0
            agent_rewards[:] = 0
            agent_costs[:] = 0
            agent_label_counts = [{0: 0, 1: 0, "multi": 0} for _ in range(n_agents)]
            agent_cost_steps = [[] for _ in range(n_agents)]

    env.close()

if __name__ == "__main__":
    main()
