#!/usr/bin/env python3.10
"""Quick test to tune bus injection rate for target label rate 10-40%."""
import warnings; warnings.filterwarnings('ignore')
import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ.setdefault('LIBSUMO_AS_TRACI', '1')
import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from env.sumo import make_sumo_vec_env

intervals = [35.0, 25.0, 20.0]
warn_thresholds = [10.0]

for interval in intervals:
    for warn in warn_thresholds:
        print(f"\n--- interval={interval}s, T_warn={warn}s ---")
        env = make_sumo_vec_env(
            env_name='sumo-arterial4x4-v0',
            n_envs=1, seed=42,
            override_reward=None,
            cost_fn='bus_priority',
            bus_injection_interval=interval,
            bus_warn_threshold=warn,
        )
        n = env.num_envs
        obs = env.reset()
        labels, costs, bus_any = [], [], []
        for step in range(720):
            acts = np.array([env.action_space.sample() for _ in range(n)])
            obs, r, d, info = env.step(acts)
            for i in range(n):
                labels.append(info[i].get('safety_label', 0))
                costs.append(info[i].get('cost', 0.0))
                bus_any.append(int(np.any(obs[i, 18:24] > 0)))
            if any(d):
                break
        env.close()
        print(f"  Steps: {len(labels)}")
        print(f"  Label rate: {sum(labels)/len(labels):.3f}")
        print(f"  Cost rate:  {sum(1 for c in costs if c > 0)/len(costs):.3f}")
        print(f"  Bus on any lane: {sum(bus_any)/len(bus_any):.3f}")
