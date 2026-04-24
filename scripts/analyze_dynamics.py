"""Detailed reward dynamics analysis: clean vs convoy episodes."""
import sys, os, argparse
import numpy as np
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def log(msg):
    print(msg, flush=True)

def run_episodes(model, env, n_episodes, tag):
    n_agents = env.num_envs
    log(f"\n{'#'*60}")
    log(f"# {tag}: {n_episodes} episodes, {n_agents} agents")
    log(f"{'#'*60}")

    for ep in range(n_episodes):
        obs = env.reset()
        step = 0
        # Per-step tracking
        step_rewards = []  # sum across agents each step
        step_costs = []
        step_labels_1 = []  # count of label=1 agents each step
        per_agent_cum_rew = np.zeros(n_agents)

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            step += 1

            sr = float(rewards.sum())
            sc = sum(infos[i].get("cost", 0.0) for i in range(n_agents))
            nl = sum(1 for i in range(n_agents)
                     if infos[i].get("safety_label", 0) == 1)
            step_rewards.append(sr)
            step_costs.append(sc)
            step_labels_1.append(nl)
            per_agent_cum_rew += rewards.flatten()

            # Every 50 steps: show reward trend
            if step % 50 == 0:
                window = step_rewards[-50:]
                avg_r = np.mean(window)
                min_r = np.min(window)
                max_r = np.max(window)
                cum_r = sum(step_rewards)
                cum_c = sum(step_costs)
                n_lbl1_window = sum(step_labels_1[-50:])
                log(f"  [{tag}] ep{ep+1} step={step:4d} "
                    f"avg50={avg_r:7.1f} min50={min_r:7.1f} max50={max_r:7.1f} "
                    f"cum_rew={cum_r:8.0f} cum_cost={cum_c:4.0f} "
                    f"lbl1_50={n_lbl1_window:3d}")

            if dones[0]:
                done = True

        # Episode summary
        ep_infos = [infos[i].get("episode", {}) for i in range(n_agents)]
        mean_r = np.mean([ei.get("r", 0) for ei in ep_infos if ei])
        mean_c = np.mean([ei.get("c", 0) for ei in ep_infos if ei])

        # Per-agent reward distribution
        agent_ep_r = [ei.get("r", 0) for ei in ep_infos if ei]
        log(f"\n  [{tag}] Episode {ep+1} SUMMARY:")
        log(f"    Mean reward:  {mean_r:.1f}  (std={np.std(agent_ep_r):.1f})")
        log(f"    Mean cost:    {mean_c:.1f}")
        log(f"    Steps:        {step}")
        log(f"    Agent rewards (sorted): {sorted([round(r,0) for r in agent_ep_r])}")

        # Identify worst and best 50-step windows
        window_avgs = [np.mean(step_rewards[i:i+50]) for i in range(0, len(step_rewards)-49)]
        worst_idx = np.argmin(window_avgs)
        best_idx = np.argmax(window_avgs)
        log(f"    Worst 50-step window: steps {worst_idx+1}-{worst_idx+50}, avg={window_avgs[worst_idx]:.1f}")
        log(f"    Best  50-step window: steps {best_idx+1}-{best_idx+50}, avg={window_avgs[best_idx]:.1f}")

        # Cost timing
        cost_steps = [i+1 for i, c in enumerate(step_costs) if c > 0]
        log(f"    Cost steps: {cost_steps}")
        log("")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--vecnorm_path", type=str, default=None)
    parser.add_argument("--cost_fn", type=str, default="convoy_priority")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from algos.split_rl import SPLIT_RL
    from env.sumo import make_sumo_vec_env
    from stable_baselines3.common.vec_env import VecNormalize

    # --- CLEAN episodes (no convoy) ---
    log("Creating CLEAN env (clean_episode_prob=1.0)...")
    env_clean = make_sumo_vec_env(
        env_name="sumo-arterial4x4-v0", n_envs=1, seed=args.seed,
        cost_fn=args.cost_fn, clean_episode_prob=1.0,
    )
    if args.vecnorm_path and os.path.exists(args.vecnorm_path):
        env_clean = VecNormalize.load(args.vecnorm_path, env_clean)
        env_clean.training = False
        env_clean.norm_reward = False

    model = SPLIT_RL.load(args.model_path, env=env_clean, device="cuda")
    log(f"Loaded model from {args.model_path}")

    run_episodes(model, env_clean, n_episodes=3, tag="CLEAN")
    env_clean.close()

    # --- CONVOY episodes (always convoy) ---
    log("\nCreating CONVOY env (clean_episode_prob=0.0)...")
    env_convoy = make_sumo_vec_env(
        env_name="sumo-arterial4x4-v0", n_envs=1, seed=args.seed,
        cost_fn=args.cost_fn, clean_episode_prob=0.0,
    )
    if args.vecnorm_path and os.path.exists(args.vecnorm_path):
        env_convoy = VecNormalize.load(args.vecnorm_path, env_convoy)
        env_convoy.training = False
        env_convoy.norm_reward = False
    model.set_env(env_convoy)

    run_episodes(model, env_convoy, n_episodes=3, tag="CONVOY")
    env_convoy.close()

if __name__ == "__main__":
    main()
