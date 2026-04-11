#!/usr/bin/env python3.10
"""
Cross-env × cross-scenario diagnostic for SUMO cost scenarios.

Answers: "Does scenario X induce structural conflict on env Y?"

Stage A — Random-policy smoke test:
  - event prevalence, cost rate, label rate
  - label lead time, horizon-based label P/R/F1
  - Δ = E[r|cost>0] - E[r|cost=0]

Stage B — Scenario-specific probe-policy conflict test:
  - reward_probe:  scenario-specific reward-maximising heuristic
  - safety_probe:  scenario-specific safety-maximising heuristic
  - Reports reward drop, cost drop, and conflict verdict

Probe designs per scenario:
  convoy_priority:  reward=queue_chase, safety=serve convoy lane
  bus_priority:     reward=queue_chase, safety=urgency-serve bus near threshold (sticky)
  premium_priority: reward=queue_chase, safety=urgency-serve premium near threshold (sticky)
  spillback:        reward=queue_chase (ignore ds_occ), safety=avoid high-ds-occ phases

No training. No VecNormalize. Pure scenario measurement.
All output to stdout. Use tee to save:

    python3.10 scripts/sumo_scenario_sanity.py --steps 720 2>&1 | tee /tmp/sanity.log
"""
import os, sys, time, argparse, warnings, traceback
os.environ["LIBSUMO_AS_TRACI"] = "1"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

warnings.filterwarnings("ignore")
import numpy as np

# ── Environments ──
ENVS = {
    "sumo-single-v0":       {"mainline_direction": "ew"},
    "sumo-single-vhvh-v0":  {"mainline_direction": "ns"},
    "sumo-grid4x4-v0":      {"mainline_direction": "auto"},
    "sumo-arterial4x4-v0":  {"mainline_direction": "auto"},
}

# ── Scenarios ──
SCENARIOS = {
    "convoy_priority": {
        "cost_fn": "convoy_priority",
        "convoy_injection_interval": 80.0,
        "convoy_size_min": 10,
        "convoy_size_max": 12,
        "convoy_headway": 2.0,
        "clean_episode_prob": 0.0,
        "label_horizon_steps": 8,
    },
    "bus_priority": {
        "cost_fn": "bus_priority",
        "bus_injection_interval": 30.0,
        "bus_cost_threshold": 15.0,
        "bus_warn_threshold": 6.0,
        "clean_episode_prob": 0.0,
        "label_horizon_steps": 4,
    },
    "premium_priority": {
        "cost_fn": "premium_priority",
        "premium_injection_interval": 15.0,
        "premium_cost_threshold": 8.0,
        "premium_warn_threshold": 3.0,
        "premium_count_per_injection": 1,
        "premium_headway": 2.0,
        "clean_episode_prob": 0.0,
        "label_horizon_steps": 2,
    },
    "spillback": {
        "cost_fn": "spillback",
        "roadwork_interval_mean": 180.0,
        "roadwork_duration_mean": 900.0,
        "roadwork_speed": 0.05,
        "spillback_occ_threshold": 0.30,
        "spillback_warn_threshold": 0.18,
        "clean_episode_prob": 0.0,
        "label_horizon_steps": 4,
    },
}


def _fix_supersuit_dones(vec_env):
    original_sw = vec_env.step_wait
    def fixed_step_wait():
        obs, rewards, dones, infos = original_sw()
        return obs, rewards, np.asarray(dones, dtype=bool), infos
    vec_env.step_wait = fixed_step_wait


def make_env(env_name, scenario_name):
    from env.sumo import make_sumo_vec_env
    env_cfg = ENVS[env_name]
    scn_cfg = {k: v for k, v in SCENARIOS[scenario_name].items()
               if k != "label_horizon_steps"}
    env = make_sumo_vec_env(
        env_name=env_name, n_envs=1, seed=0,
        override_reward=None,
        mainline_direction=env_cfg["mainline_direction"],
        debug=True, **scn_cfg,
    )
    _fix_supersuit_dones(env)
    return env


def collect_rollout(env, n_steps, scenario_name):
    obs = env.reset()
    records = []
    warn_thresh = SCENARIOS[scenario_name].get("spillback_warn_threshold", 0.25)
    for _ in range(n_steps):
        actions = np.array([env.action_space.sample() for _ in range(env.num_envs)])
        obs, rewards, dones, infos = env.step(actions)
        for i in range(env.num_envs):
            info = infos[i]
            if scenario_name == "spillback":
                has_ent = int(info.get("dbg_downstream_occ_max", 0.0) > warn_thresh)
            else:
                has_ent = int(bool(
                    info.get("dbg_has_convoy", 0) or
                    info.get("dbg_has_bus", 0) or
                    info.get("dbg_has_premium", 0)))
            records.append({
                "raw_reward": float(info.get("raw_reward",
                                    info.get("original_reward", rewards[i]))),
                "cost": float(info.get("cost", 0.0)),
                "label": int(info.get("safety_label", 0)),
                "has_entity": has_ent,
            })
    return records


# ── Stage B: Probe-policy infrastructure ──────────────────────────────

def _find_wrapper(env):
    """Walk through VecEnv wrapper layers to find SumoRewardCostWrapper."""
    from env.sumo import SumoRewardCostWrapper
    visited = set()
    queue = [env]
    while queue:
        cur = queue.pop(0)
        cid = id(cur)
        if cid in visited:
            continue
        visited.add(cid)
        if isinstance(cur, SumoRewardCostWrapper):
            return cur
        for attr in ('venv', 'par_env', 'env', 'aec_env', '_env'):
            child = getattr(cur, attr, None)
            if child is not None and id(child) not in visited:
                queue.append(child)
        vec_envs = getattr(cur, 'vec_envs', None)
        if isinstance(vec_envs, list):
            for child in vec_envs:
                if id(child) not in visited:
                    queue.append(child)
    return None


def _build_probe_tables(wrapper):
    """Extract per-agent phase→lane mapping for probe policies."""
    agents = list(wrapper.possible_agents)
    ptl = wrapper._phase_to_lanes
    n_actions = wrapper._agent_n_actions
    agent_obs_dims = wrapper._agent_obs_dims
    raw_max_obs_dim = wrapper._raw_max_obs_dim
    ext_max_n_lanes = getattr(wrapper, '_ext_max_n_lanes', 0)

    tables = {}
    for agent_id in agents:
        n_act = n_actions[agent_id]
        raw_dim = agent_obs_dims[agent_id]
        n_lanes = (raw_dim - n_act - 1) // 2

        phase_lanes = {}
        lane_to_phase = {}
        for ph in range(n_act):
            lanes = ptl.get(agent_id, {}).get(ph, [])
            phase_lanes[ph] = list(lanes)
            for li in lanes:
                if li not in lane_to_phase:
                    lane_to_phase[li] = ph

        tables[agent_id] = {
            "n_act": n_act,
            "n_lanes": n_lanes,
            "phase_lanes": phase_lanes,
            "lane_to_phase": lane_to_phase,
        }

    return agents, tables, raw_max_obs_dim, ext_max_n_lanes


def _queue_chasing_action(obs_i, t):
    """Pick the phase whose lanes have the highest total queue."""
    n_act = t["n_act"]
    n_lanes = t["n_lanes"]
    queue_start = n_act + 1 + n_lanes
    queue = obs_i[queue_start:queue_start + n_lanes]

    best_ph, best_score = 0, -1.0
    for ph, lanes in t["phase_lanes"].items():
        score = sum(queue[li] for li in lanes if li < n_lanes)
        if score > best_score:
            best_score = score
            best_ph = ph
    return best_ph


# ── Scenario-specific safety probes ──────────────────────────────────

def _convoy_safety_action(obs_i, t, raw_max, ext_max):
    """Convoy: serve the lane with a convoy entity; else queue-chase.

    Obs extension: [has_convoy(M), convoy_count(M), convoy_progress(M)]
    """
    if ext_max > 0:
        has_ent = obs_i[raw_max:raw_max + ext_max]
        for li in range(min(t["n_lanes"], len(has_ent))):
            if has_ent[li] > 0 and li in t["lane_to_phase"]:
                return t["lane_to_phase"][li]
    return _queue_chasing_action(obs_i, t)


def _bus_safety_action(obs_i, t, raw_max, ext_max, sticky_state):
    """Bus: urgency-aware — only intervene when bus_wait > warn, pick most urgent.

    Obs extension: [has_bus(M), bus_count(M), bus_wait(M)]
    bus_wait starts at raw_max + 2*ext_max.
    bus_wait normalised by T_cost, warn_norm = T_warn / T_cost.
    Sticky: once committed to serving an urgent bus lane, hold for min_hold steps.
    """
    warn_norm = 6.0 / 15.0  # bus_warn / bus_cost

    if ext_max > 0:
        has_bus = obs_i[raw_max:raw_max + ext_max]
        bus_wait = obs_i[raw_max + 2 * ext_max:raw_max + 3 * ext_max]

        # Check sticky commitment
        committed_lane = sticky_state.get("committed_lane", -1)
        committed_ttl = sticky_state.get("committed_ttl", 0)
        if committed_lane >= 0 and committed_ttl > 0:
            if (committed_lane < len(has_bus) and has_bus[committed_lane] > 0
                    and committed_lane < len(bus_wait)
                    and bus_wait[committed_lane] > warn_norm * 0.5):
                sticky_state["committed_ttl"] = committed_ttl - 1
                if committed_lane in t["lane_to_phase"]:
                    return t["lane_to_phase"][committed_lane]
            else:
                sticky_state["committed_lane"] = -1
                sticky_state["committed_ttl"] = 0

        # Find most urgent unserved bus
        n_act = t["n_act"]
        phase_oh = obs_i[:n_act]
        cur_phase = int(np.argmax(phase_oh))
        served = set(t["phase_lanes"].get(cur_phase, []))

        best_lane, best_urgency = -1, -1.0
        for li in range(min(t["n_lanes"], len(has_bus))):
            if has_bus[li] > 0 and li not in served and li in t["lane_to_phase"]:
                urgency = bus_wait[li] if li < len(bus_wait) else 0.0
                if urgency > warn_norm and urgency > best_urgency:
                    best_urgency = urgency
                    best_lane = li

        if best_lane >= 0:
            sticky_state["committed_lane"] = best_lane
            sticky_state["committed_ttl"] = 3
            return t["lane_to_phase"][best_lane]

    return _queue_chasing_action(obs_i, t)


def _premium_safety_action(obs_i, t, raw_max, ext_max, sticky_state):
    """Premium: urgency-aware — intervene when premium_wait > warn, sticky.

    Obs extension: [has_premium(M), premium_wait(M)]
    premium_wait starts at raw_max + ext_max.
    """
    warn_norm = 3.0 / 8.0  # premium_warn / premium_cost (route-delay based)

    if ext_max > 0:
        has_prem = obs_i[raw_max:raw_max + ext_max]
        prem_wait = obs_i[raw_max + ext_max:raw_max + 2 * ext_max]

        committed_lane = sticky_state.get("committed_lane", -1)
        committed_ttl = sticky_state.get("committed_ttl", 0)
        if committed_lane >= 0 and committed_ttl > 0:
            if (committed_lane < len(has_prem) and has_prem[committed_lane] > 0
                    and committed_lane < len(prem_wait)
                    and prem_wait[committed_lane] > warn_norm * 0.5):
                sticky_state["committed_ttl"] = committed_ttl - 1
                if committed_lane in t["lane_to_phase"]:
                    return t["lane_to_phase"][committed_lane]
            else:
                sticky_state["committed_lane"] = -1
                sticky_state["committed_ttl"] = 0

        n_act = t["n_act"]
        phase_oh = obs_i[:n_act]
        cur_phase = int(np.argmax(phase_oh))
        served = set(t["phase_lanes"].get(cur_phase, []))

        best_lane, best_urgency = -1, -1.0
        for li in range(min(t["n_lanes"], len(has_prem))):
            if has_prem[li] > 0 and li not in served and li in t["lane_to_phase"]:
                urgency = prem_wait[li] if li < len(prem_wait) else 0.0
                if urgency > warn_norm and urgency > best_urgency:
                    best_urgency = urgency
                    best_lane = li

        if best_lane >= 0:
            sticky_state["committed_lane"] = best_lane
            sticky_state["committed_ttl"] = 3
            return t["lane_to_phase"][best_lane]

    return _queue_chasing_action(obs_i, t)


def _spillback_safety_action(obs_i, t, raw_max, ext_max):
    """Spillback: avoid phases with high downstream occupancy.

    Obs extension: [downstream_occ(M)]
    Safety = pick phase with LOWEST max downstream_occ (avoid congestion).
    """
    warn_thresh = 0.18

    if ext_max > 0:
        ds_occ = obs_i[raw_max:raw_max + ext_max]

        max_occ = max(ds_occ[li] for li in range(min(t["n_lanes"], len(ds_occ))))
        if max_occ <= warn_thresh:
            return _queue_chasing_action(obs_i, t)

        # Score phases: prefer low downstream_occ, tiebreak by queue
        n_act = t["n_act"]
        n_lanes = t["n_lanes"]
        queue_start = n_act + 1 + n_lanes
        queue = obs_i[queue_start:queue_start + n_lanes]

        best_ph, best_score = 0, -999.0
        for ph, lanes in t["phase_lanes"].items():
            if not lanes:
                continue
            max_ds = max((ds_occ[li] if li < len(ds_occ) else 0.0) for li in lanes)
            q_score = sum(queue[li] for li in lanes if li < n_lanes)
            score = -max_ds * 10.0 + q_score
            if score > best_score:
                best_score = score
                best_ph = ph
        return best_ph

    return _queue_chasing_action(obs_i, t)


def collect_probe_rollout(env, n_steps, agents, tables,
                          raw_max_obs_dim, ext_max_n_lanes,
                          probe_type, scenario_name,
                          reset_seed=None, wrapper=None):
    """Run rollout with a deterministic probe policy.

    probe_type: "reward" or "safety"
    Dispatches to scenario-specific probe based on scenario_name.
    """
    if reset_seed is not None and wrapper is not None:
        wrapper._episode_count = reset_seed - 1
    obs = env.reset()
    records = []
    n_agents = len(agents)

    # Per-agent sticky state for bus/premium probes
    sticky_states = {aid: {} for aid in agents}

    for _ in range(n_steps):
        actions = np.zeros(env.num_envs, dtype=int)
        for i in range(env.num_envs):
            agent_id = agents[i % n_agents]
            t = tables[agent_id]
            o = obs[i]

            if probe_type == "reward":
                # All scenarios: reward = pure queue-chase (ignore entities/ds_occ)
                actions[i] = _queue_chasing_action(o, t)
            else:
                # Safety probe: scenario-specific
                if scenario_name == "convoy_priority":
                    actions[i] = _convoy_safety_action(o, t, raw_max_obs_dim, ext_max_n_lanes)
                elif scenario_name == "bus_priority":
                    actions[i] = _bus_safety_action(o, t, raw_max_obs_dim, ext_max_n_lanes,
                                                    sticky_states[agent_id])
                elif scenario_name == "premium_priority":
                    actions[i] = _premium_safety_action(o, t, raw_max_obs_dim, ext_max_n_lanes,
                                                        sticky_states[agent_id])
                elif scenario_name == "spillback":
                    actions[i] = _spillback_safety_action(o, t, raw_max_obs_dim, ext_max_n_lanes)
                else:
                    actions[i] = _queue_chasing_action(o, t)

        obs, rewards, dones, infos = env.step(actions)
        for i in range(env.num_envs):
            info = infos[i]
            records.append({
                "raw_reward": float(info.get("raw_reward",
                                    info.get("original_reward", rewards[i]))),
                "cost": float(info.get("cost", 0.0)),
            })
    return records


def analyze_probes(env_name, scenario_name, n_steps):
    """Stage B: run reward and safety probes, compare."""
    result = dict(env=env_name, scenario=scenario_name,
                  rew_reward=0.0, rew_cost_rate=0.0,
                  saf_reward=0.0, saf_cost_rate=0.0,
                  d_reward=0.0, d_cost=0.0,
                  probe_verdict="ERROR", error=None)
    try:
        env = make_env(env_name, scenario_name)
        env.reset()
        wrapper = _find_wrapper(env)
        if wrapper is None:
            env.close()
            result["error"] = "cannot find SumoRewardCostWrapper"
            return result

        agents, tables, raw_max, ext_m = _build_probe_tables(wrapper)

        for aid in agents:
            t = tables[aid]
            print(f"      {aid}: n_act={t['n_act']} n_lanes={t['n_lanes']} "
                  f"phases={list(t['phase_lanes'].keys())} "
                  f"lane→phase={t['lane_to_phase']}")

        probe_seed = 123
        rew_recs = collect_probe_rollout(env, n_steps, agents, tables,
                                          raw_max, ext_m, "reward", scenario_name,
                                          reset_seed=probe_seed, wrapper=wrapper)
        saf_recs = collect_probe_rollout(env, n_steps, agents, tables,
                                          raw_max, ext_m, "safety", scenario_name,
                                          reset_seed=probe_seed, wrapper=wrapper)
        env.close()
    except Exception as e:
        traceback.print_exc()
        result["error"] = str(e)
        return result

    if not rew_recs or not saf_recs:
        result["error"] = "no data from probes"
        return result

    result["rew_reward"] = float(np.mean([r["raw_reward"] for r in rew_recs]))
    result["rew_cost_rate"] = float(np.mean([r["cost"] > 0 for r in rew_recs]))
    result["saf_reward"] = float(np.mean([r["raw_reward"] for r in saf_recs]))
    result["saf_cost_rate"] = float(np.mean([r["cost"] > 0 for r in saf_recs]))

    result["d_reward"] = result["saf_reward"] - result["rew_reward"]
    result["d_cost"] = result["rew_cost_rate"] - result["saf_cost_rate"]

    d_r = result["d_reward"]
    d_c = result["d_cost"]
    rew_cr = result["rew_cost_rate"]
    saf_cr = result["saf_cost_rate"]

    if result["error"]:
        verdict = "ERROR"
    elif rew_cr < 0.01 and saf_cr < 0.01:
        verdict = "NO COST (neither probe triggers cost)"
    elif d_c < 0.01:
        verdict = "NO SEPARATION (safety probe doesn't reduce cost)"
    elif d_r >= 0:
        verdict = "NO TRADEOFF (safety dominates — no reward sacrifice)"
    elif d_c > 0.03 and d_r < -0.005:
        verdict = "CONFLICT CONFIRMED"
    else:
        verdict = "WEAK CONFLICT"

    result["probe_verdict"] = verdict
    return result


def compute_label_lead(records, H):
    costs = [r["cost"] > 0 for r in records]
    labels = [r["label"] > 0 for r in records]
    N = len(records)
    leads = []
    for t in range(N):
        if costs[t] and (t == 0 or not costs[t - 1]):
            earliest = None
            for k in range(min(H + 1, t + 1)):
                if labels[t - k]:
                    earliest = k
            if earliest is not None:
                leads.append(earliest)
    if not leads:
        return 0.0, 0
    return float(np.mean(leads)), len(leads)


def compute_label_quality(records, H):
    costs = np.array([r["cost"] > 0 for r in records], dtype=int)
    labels = np.array([r["label"] > 0 for r in records], dtype=int)
    N = len(records)
    tp, fp = 0, 0
    for t in range(N):
        if labels[t]:
            if costs[t:min(t + H + 1, N)].any():
                tp += 1
            else:
                fp += 1
    covered, uncovered = 0, 0
    for t in range(N):
        if costs[t]:
            if labels[max(t - H, 0):t + 1].any():
                covered += 1
            else:
                uncovered += 1
    prec = tp / max(tp + fp, 1)
    rec = covered / max(covered + uncovered, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return prec, rec, f1


def analyze_pair(env_name, scenario_name, n_steps):
    H = SCENARIOS[scenario_name]["label_horizon_steps"]
    result = dict(env=env_name, scenario=scenario_name, n_steps=0,
                  event_rate=0.0, cost_rate=0.0, label_rate=0.0,
                  mean_rew_cost1=0.0, mean_rew_cost0=0.0, delta=0.0,
                  label_prec=0.0, label_rec=0.0, label_f1=0.0,
                  label_lead=0.0, n_onsets=0, verdict="ERROR", error=None)

    try:
        env = make_env(env_name, scenario_name)
        records = collect_rollout(env, n_steps, scenario_name)
        env.close()
    except Exception as e:
        traceback.print_exc()
        result["error"] = str(e)
        return result

    N = len(records)
    if N == 0:
        result["error"] = "no data"
        return result

    result["n_steps"] = N
    result["event_rate"] = sum(r["has_entity"] for r in records) / N

    n_cost = sum(r["cost"] > 0 for r in records)
    n_label = sum(r["label"] > 0 for r in records)
    result["cost_rate"] = n_cost / N
    result["label_rate"] = n_label / N

    if n_cost > 0 and n_cost < N:
        rews_c1 = [r["raw_reward"] for r in records if r["cost"] > 0]
        rews_c0 = [r["raw_reward"] for r in records if r["cost"] == 0]
        result["mean_rew_cost1"] = float(np.mean(rews_c1))
        result["mean_rew_cost0"] = float(np.mean(rews_c0))
        result["delta"] = result["mean_rew_cost1"] - result["mean_rew_cost0"]

    if n_cost > 0:
        prec, rec, f1 = compute_label_quality(records, H)
        result["label_prec"] = prec
        result["label_rec"] = rec
        result["label_f1"] = f1

    lead, n_onsets = compute_label_lead(records, H)
    result["label_lead"] = lead
    result["n_onsets"] = n_onsets

    cost_rate = result["cost_rate"]
    event_rate = result["event_rate"]

    if result["error"]:
        verdict = "ERROR"
    elif event_rate < 0.01:
        verdict = "no events"
    elif cost_rate < 0.01:
        verdict = "low cost (<1%)"
    elif cost_rate > 0.95:
        verdict = "cost saturated (>95%)"
    elif result["label_f1"] < 0.3:
        verdict = "ok (labels weak)"
    elif result["label_f1"] >= 0.5:
        verdict = "ok"
    else:
        verdict = "ok (labels marginal)"

    result["verdict"] = verdict
    return result


def print_pair(r):
    print(f"    steps={r['n_steps']}  events={r['event_rate']*100:.1f}%  "
          f"cost={r['cost_rate']*100:.1f}%  label={r['label_rate']*100:.1f}%")
    if r["n_steps"] > 0 and r["cost_rate"] > 0:
        print(f"    E[r|c>0]={r['mean_rew_cost1']:+.2f}  "
              f"E[r|c=0]={r['mean_rew_cost0']:+.2f}  Δ={r['delta']:+.2f}")
        print(f"    Label(H): P={r['label_prec']:.3f} R={r['label_rec']:.3f} "
              f"F1={r['label_f1']:.3f}  "
              f"lead={r['label_lead']:.1f}steps ({r['n_onsets']} onsets)")


def print_table(results):
    hdr = (f"  {'Env':<26s} {'Scenario':<18s} {'Evnt%':>6s} {'Cost%':>6s} "
           f"{'Lbl%':>6s} {'Lead':>5s} {'delta':>9s} {'LblF1':>6s} {'Verdict'}")
    sep = "  " + "-" * 108
    print(f"\n{sep}")
    print(hdr)
    print(sep)
    for r in results:
        if r["error"]:
            print(f"  {r['env']:<26s} {r['scenario']:<18s}"
                  f"{'':>42s} ERROR: {r['error'][:40]}")
            continue
        evnt = f"{r['event_rate']*100:5.1f}%"
        cost = f"{r['cost_rate']*100:5.1f}%"
        lbl = f"{r['label_rate']*100:5.1f}%"
        lead = f"{r['label_lead']:4.1f}" if r["n_onsets"] > 0 else " n/a"
        delta = f"{r['delta']:+8.2f}"
        lf1 = f"{r['label_f1']:5.3f}"
        print(f"  {r['env']:<26s} {r['scenario']:<18s} {evnt:>6s} {cost:>6s} "
              f"{lbl:>6s} {lead:>5s} {delta:>9s} {lf1:>6s} {r['verdict']}")
    print(sep)


def print_probe_table(results):
    hdr = (f"  {'Env':<26s} {'Scenario':<18s} "
           f"{'Rew_r':>8s} {'Rew_c%':>6s} "
           f"{'Saf_r':>8s} {'Saf_c%':>6s} "
           f"{'Δrew':>8s} {'Δcost':>7s} {'Probe Verdict'}")
    sep = "  " + "-" * 120
    print(f"\n  ═══ Stage B Summary Table ═══")
    print(sep)
    print(hdr)
    print(sep)
    for r in results:
        if r["error"]:
            print(f"  {r['env']:<26s} {r['scenario']:<18s}"
                  f"{'':>50s} ERROR: {r['error'][:40]}")
            continue
        print(f"  {r['env']:<26s} {r['scenario']:<18s} "
              f"{r['rew_reward']:+8.2f} {r['rew_cost_rate']*100:5.1f}% "
              f"{r['saf_reward']:+8.2f} {r['saf_cost_rate']*100:5.1f}% "
              f"{r['d_reward']:+8.3f} {r['d_cost']*100:+6.1f}% "
              f"{r['probe_verdict']}")
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-env x cross-scenario SUMO conflict diagnostic")
    parser.add_argument("--env", type=str, default=None,
                        choices=list(ENVS.keys()))
    parser.add_argument("--scenario", type=str, default=None,
                        choices=list(SCENARIOS.keys()))
    parser.add_argument("--steps", type=int, default=720,
                        help="Steps per rollout (default: 720 = 1 episode)")
    parser.add_argument("--skip-stage-a", action="store_true")
    parser.add_argument("--skip-stage-b", action="store_true")
    args = parser.parse_args()

    envs = [args.env] if args.env else list(ENVS.keys())
    scenarios = [args.scenario] if args.scenario else list(SCENARIOS.keys())

    print(f"  Envs: {envs}")
    print(f"  Scenarios: {scenarios}")
    print(f"  Steps/rollout: {args.steps}")
    print(f"  Total pairs: {len(envs) * len(scenarios)}")

    # ── Stage A ──
    stage_a_results = []
    if not args.skip_stage_a:
        print(f"\n  ═══ Stage A: Random-Policy Smoke Test ═══")
        for env_name in envs:
            for scenario_name in scenarios:
                print(f"\n  >>> {env_name} x {scenario_name} ...", flush=True)
                t0 = time.time()
                r = analyze_pair(env_name, scenario_name, args.steps)
                elapsed = time.time() - t0
                if r["error"]:
                    print(f"    ERROR: {r['error']}")
                else:
                    print_pair(r)
                print(f"    => {r['verdict']}  ({elapsed:.0f}s)")
                stage_a_results.append(r)

        print_table(stage_a_results)
        ok = [r for r in stage_a_results if r["verdict"].startswith("ok")]
        issues = [r for r in stage_a_results
                  if not r["verdict"].startswith("ok") and r["verdict"] != "ERROR"]
        errors = [r for r in stage_a_results if r["verdict"] == "ERROR"]
        print(f"\n  Stage A summary: ok={len(ok)}  "
              f"issues={len(issues)}  errors={len(errors)}")

    # ── Stage B ──
    stage_b_results = []
    if not args.skip_stage_b:
        print(f"\n  ═══ Stage B: Scenario-Specific Probe-Policy Conflict Test (same-seed) ═══")
        print(f"  reward_probe  = queue-chase (throughput maximiser, ignores entities)")
        print(f"  safety_probe  = scenario-specific safety-maximising heuristic")
        print(f"  Both probes seeded identically → same injection realization")
        print(f"  Probe designs:")
        print(f"    convoy:    safety = serve convoy lane on detection")
        print(f"    bus:       safety = urgency-serve bus near threshold (sticky 3-step hold)")
        print(f"    premium:   safety = urgency-serve premium near threshold (sticky 3-step hold)")
        print(f"    spillback: safety = avoid phases with high downstream occupancy")
        for env_name in envs:
            for scenario_name in scenarios:
                print(f"\n  >>> PROBES: {env_name} x {scenario_name} ...", flush=True)
                t0 = time.time()
                r = analyze_probes(env_name, scenario_name, args.steps)
                elapsed = time.time() - t0
                if r["error"]:
                    print(f"    ERROR: {r['error']}")
                else:
                    print(f"    reward_probe:  E[r]={r['rew_reward']:+.3f}  "
                          f"cost_rate={r['rew_cost_rate']*100:.1f}%")
                    print(f"    safety_probe:  E[r]={r['saf_reward']:+.3f}  "
                          f"cost_rate={r['saf_cost_rate']*100:.1f}%")
                    print(f"    Δ_reward={r['d_reward']:+.4f}  "
                          f"Δ_cost={r['d_cost']*100:+.1f}%")
                print(f"    => {r['probe_verdict']}  ({elapsed:.0f}s)")
                stage_b_results.append(r)

        print_probe_table(stage_b_results)
        confirmed = [r for r in stage_b_results
                     if r["probe_verdict"] == "CONFLICT CONFIRMED"]
        weak = [r for r in stage_b_results
                if r["probe_verdict"] == "WEAK CONFLICT"]
        failed = [r for r in stage_b_results
                  if r["probe_verdict"] not in ("CONFLICT CONFIRMED", "WEAK CONFLICT")]
        print(f"\n  Stage B summary: CONFIRMED={len(confirmed)}  "
              f"WEAK={len(weak)}  FAILED={len(failed)}")


if __name__ == "__main__":
    main()
