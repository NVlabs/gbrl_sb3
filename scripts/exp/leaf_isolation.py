"""Does the learned partition isolate the guidance-labelled states?

Split-RL's claim is that guidance labels change *where* the tree splits, so that
conflicting updates land in different leaves. This measures that directly, and can be
measured for a model that never saw labels -- the labels come from the environment at
analysis time, never from the model, so the unguided PPO-Lag (GBT) baseline is scored
the same way.

State set. Every reachable (cell, direction) is enumerated and rendered directly. We do
not roll out a policy: a trained safe agent avoids exactly the states where the
partition matters, so trajectory data cannot answer this question.

Measurement. For each tree, find the leaves that any label-1 state reaches, then ask
what else lands in those leaves. If the partition isolates the labelled region, those
leaves are mostly label-1 states (high precision). If it does not, the label-1 states
are spread across leaves dominated by unrelated states, and precision falls to the base
rate. Reported as precision and as enrichment over the base rate, so the numbers are
comparable across models and not inflated by class imbalance.

Usage:
    python3.10 scripts/exp/leaf_isolation.py --n_trees 400
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.exp.corner_density_maps import build_env, _learner
from scripts.exp.leaf_purity import leaf_indices

def ckpts(env_name):
    root = f'saved_models/minigrid/minigrid/{env_name}'
    return {'Split-RL': f'{root}/split_rl/fully_obs_seed_0_300000_steps.zip',
            'PPO-Lag (GBT)': f'{root}/ppo_lag_gbrl/fully_obs_seed_0_300000_steps.zip'}
DIRVEC = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}


def enumerate_fragile(wrapped, unwrapped):
    """FragileCrossing: states are (cell, direction, carrying).

    The label here is `carrying_heavy AND (on_ice OR ice_to_the_east)`. Note two
    differences from Corner. It depends on the agent's inventory, which MiniGrid does
    expose -- gen_obs_grid writes the carried object into the agent's own cell in the
    view -- so the label remains a function of the observation. And `right_pos` is
    absolute east rather than relative to the heading, so the label does not depend on
    direction, though the observation still does.
    """
    from env.safety.utils import Ice
    heavy = None
    for x in range(unwrapped.width):
        for y in range(unwrapped.height):
            c = unwrapped.grid.get(x, y)
            if c is not None and getattr(c, 'type', None) == 'heavy_obj':
                heavy = c
    if heavy is None:
        raise RuntimeError('no heavy_obj found in the grid; cannot enumerate carrying states')

    obs_list, labels, meta = [], [], []
    for x in range(unwrapped.width):
        for y in range(unwrapped.height):
            cell = unwrapped.grid.get(x, y)
            if cell is not None and not isinstance(cell, Ice):
                continue
            east = (unwrapped.grid.get(x + 1, y) if x + 1 < unwrapped.width else None)
            hazard = isinstance(cell, Ice) or isinstance(east, Ice)
            for carrying in (None, heavy):
                for d in range(4):
                    unwrapped.agent_pos = (x, y)
                    unwrapped.agent_dir = d
                    unwrapped.carrying = carrying
                    obs_list.append(wrapped.observation(unwrapped.gen_obs()))
                    labels.append(float(carrying is not None and hazard))
                    meta.append((x, y, d, carrying is not None))
    unwrapped.carrying = None
    return np.asarray(obs_list, dtype=object), np.asarray(labels), meta


def enumerate_states(wrapped, unwrapped):
    """Every (cell, direction) the agent can occupy, with its guidance label."""
    from minigrid.core.world_object import Lava
    obs_list, labels, meta = [], [], []
    for x in range(unwrapped.width):
        for y in range(unwrapped.height):
            cell = unwrapped.grid.get(x, y)
            if cell is not None and not isinstance(cell, Lava):
                continue                      # walls / goals are not standable
            for d in range(4):
                unwrapped.agent_pos = (x, y)
                unwrapped.agent_dir = d
                obs_list.append(wrapped.observation(unwrapped.gen_obs()))
                fx, fy = x + DIRVEC[d][0], y + DIRVEC[d][1]
                fwd = (unwrapped.grid.get(fx, fy)
                       if 0 <= fx < unwrapped.width and 0 <= fy < unwrapped.height else None)
                on_lava = isinstance(cell, Lava)
                facing_lava = isinstance(fwd, Lava)
                labels.append(float(on_lava or facing_lava))
                meta.append((x, y, d))
    return np.asarray(obs_list, dtype=object), np.asarray(labels), meta


def co_occurrence(model, obs, labels, n_trees, stride):
    """Ensemble-level isolation: how often do two states share a leaf?

    A single depth-4 tree has 16 leaves for hundreds of states, so no one tree can
    isolate a small labelled set. The ensemble's effective partition is the
    intersection of all its trees, so the right question is pairwise: across trees, do
    label-1 states land together more often than they land with label-0 states?

    Returns (within, between, ratio) where `within` is the mean fraction of trees in
    which two label-1 states share a leaf, and `between` the same for label-1 against
    label-0. ratio > 1 means the ensemble groups the labelled states together.
    """
    L = _learner(model)
    pos = labels > 0
    n = len(obs)
    same = np.zeros((n, n), dtype=np.float32)
    used = 0
    for ti in range(0, n_trees * stride, stride):
        try:
            t = L.get_tree(ti)
        except Exception:
            break
        leaf = leaf_indices(t, obs)
        if (leaf < 0).all():
            continue
        same += (leaf[:, None] == leaf[None, :]) & (leaf[:, None] >= 0)
        used += 1
    if used == 0:
        return 0.0, 0.0, 0.0
    same /= used
    iu = ~np.eye(n, dtype=bool)
    within = same[np.ix_(pos, pos)][~np.eye(pos.sum(), dtype=bool)].mean()
    between = same[np.ix_(pos, ~pos)].mean()
    return float(within), float(between), float(within / between if between > 0 else 0.0)


def isolation(model, obs, labels, n_trees, stride):
    """Precision of the leaves that label-1 states reach."""
    L = _learner(model)
    prec, cover = [], []
    for ti in range(0, n_trees * stride, stride):
        try:
            t = L.get_tree(ti)
        except Exception:
            break
        leaf = leaf_indices(t, obs)
        ok = leaf >= 0
        if ok.sum() < 10 or labels[ok].sum() < 1:
            continue
        hit = np.unique(leaf[ok & (labels > 0)])       # leaves any label-1 state reaches
        inhit = ok & np.isin(leaf, hit)
        prec.append(labels[inhit].mean())              # what else lands there
        cover.append(len(hit) / max(len(np.unique(leaf[ok])), 1))
    return np.asarray(prec), np.asarray(cover)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env_name', default='MiniGrid-Corner-v0')
    ap.add_argument('--n_trees', type=int, default=400)
    ap.add_argument('--stride', type=int, default=20)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    from algos.split_rl import SPLIT_RL
    from algos.safety.ppo_lag_gbrl import PPOLagGBRL

    vec_env, wrapped = build_env(args.env_name, 0)
    unwrapped = wrapped.unwrapped
    wrapped.reset(seed=0)
    enum = enumerate_fragile if 'Fragile' in args.env_name else enumerate_states
    obs, labels, _ = enum(wrapped, unwrapped)
    base = labels.mean()
    print(f"\n  {len(obs)} enumerated (cell, direction) states, "
          f"{int(labels.sum())} label-1 ({base * 100:.1f}%)\n")

    CKPT = ckpts(args.env_name)
    ref = SPLIT_RL.load(CKPT['Split-RL'], env=vec_env, device=args.device, force_reset=True)
    print(f"  {'model':<16}{'label1-label1':>16}{'label1-label0':>16}{'ratio':>10}")
    for name, path in CKPT.items():
        m = ref if name == 'Split-RL' else PPOLagGBRL.load(
            path, env=vec_env, device=args.device, force_reset=True)
        w, b, r = co_occurrence(m, obs, labels, args.n_trees, args.stride)
        print(f"  {name:<16}{w:>16.4f}{b:>16.4f}{r:>9.2f}x")


if __name__ == '__main__':
    main()
