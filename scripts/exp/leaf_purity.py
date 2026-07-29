"""Do the learned partitions separate the two objectives?

Split-RL's claim is that guidance labels change *where the tree splits*, so that
conflicting gradients land in different regions. This measures that directly, and
crucially it can be measured for a model that never saw labels.

Method. For each tree we recover which leaf every state falls into, using only the
tree structure from ``get_tree`` (oblivious trees share one split per depth, so the
leaf index is the bit-code of the depth-wise decisions). We then look up the
ground-truth guidance label of those states -- taken from the environment, never from
the model -- and ask how mixed each leaf is.

That distinction is what makes the comparison possible. ``predict_densities`` needs
label counts recorded at fit time, so it exists only for Split-RL. Leaf assignment
needs nothing but the tree, so the unguided PPO-Lag (GBT) baseline can be measured the
same way, using labels purely as an analysis instrument.

Metric. Per leaf, |p - 0.5| * 2 where p is the fraction of label-1 states in that leaf:
1.0 means the leaf is pure (all one objective), 0.0 means an even mix. We report the
visit-weighted mean over leaves, and the fraction of states landing in a pure leaf.

Expectation. Split-RL should separate; the unguided baseline has nothing in its
objective rewarding separation, so its leaves should sit near the global label rate.

Usage:
    python3.10 scripts/exp/leaf_purity.py --n_trees 400
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.exp.corner_density_maps import build_env, collect, start_states, _learner

CKPT = {
    'Split-RL': 'saved_models/minigrid/minigrid/MiniGrid-Corner-v0/split_rl/'
                'fully_obs_seed_0_300000_steps.zip',
    'PPO-Lag (GBT)': 'saved_models/minigrid/minigrid/MiniGrid-Corner-v0/ppo_lag_gbrl/'
                     'fully_obs_seed_0_300000_steps.zip',
}


def leaf_indices(tree, obs):
    """Leaf index per state for one oblivious tree, from its structure alone."""
    depth = int(tree['tree_depth'])
    if depth == 0:
        return np.zeros(len(obs), dtype=np.int64)
    feat = np.asarray(tree['feature_indices'])[0][:depth]
    vals = np.asarray(tree['feature_values'])[0][:depth]
    numeric = np.asarray(tree['is_numerics'])[0][:depth]
    cv = np.asarray(tree['categorical_values']).astype(np.uint8)
    per = cv.size // int(tree['max_depth'])

    bits = np.zeros((len(obs), depth), dtype=bool)
    for d in range(depth):
        col = obs[:, feat[d]]
        if numeric[d]:
            bits[:, d] = col.astype(np.float64) > float(vals[d])
        else:
            cat = bytes(cv[d * per:(d + 1) * per]).split(b'\x00')[0].decode('utf-8', 'replace')
            bits[:, d] = (col.astype(str) == cat)

    ineq = np.asarray(tree['inequality_directions']).astype(bool)[:, :depth]
    leaf = np.full(len(obs), -1, dtype=np.int64)
    for li in range(int(tree['n_leaves'])):
        leaf[np.all(bits == ineq[li], axis=1)] = li
    return leaf


def _mi(leaf, labels):
    """Normalised mutual information between leaf assignment and the guidance label.

    Purity is useless at these base rates: the label fires on well under 1% of states,
    so a leaf holding only label-0 states scores as perfectly pure no matter what the
    model learned. MI instead asks how much knowing the leaf tells you about the label,
    which is what "the partition separates the objectives" actually means, and it is
    not inflated by class imbalance.
    """
    n = len(labels)
    py = np.array([1 - labels.mean(), labels.mean()])
    hy = -(py[py > 0] * np.log(py[py > 0])).sum()
    if hy <= 0:
        return 0.0, 0.0
    mi = 0.0
    for li in np.unique(leaf):
        m = leaf == li
        pl = m.mean()
        p1 = labels[m].mean()
        for pc, pyc in ((1 - p1, py[0]), (p1, py[1])):
            if pc > 0 and pyc > 0:
                mi += pl * pc * np.log(pc / pyc)
    return mi / hy, float(max((labels[leaf == li].mean() for li in np.unique(leaf)), default=0.0))


def purity(model, obs, labels, n_trees, stride):
    L = _learner(model)
    nmis, enrich, assigned = [], [], []
    for ti in range(0, n_trees * stride, stride):
        try:
            t = L.get_tree(ti)
        except Exception:
            break
        leaf = leaf_indices(t, obs)
        ok = leaf >= 0
        if ok.sum() < 10 or labels[ok].sum() < 1:
            continue
        assigned.append(ok.mean())
        nmi, mx = _mi(leaf[ok], labels[ok])
        nmis.append(nmi); enrich.append(mx)
    return (np.asarray(nmis), np.asarray(enrich),
            float(np.mean(assigned)) if assigned else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env_name', default='MiniGrid-Corner-v0')
    ap.add_argument('--n_trees', type=int, default=400)
    ap.add_argument('--stride', type=int, default=20)
    ap.add_argument('--episodes', type=int, default=12)
    ap.add_argument('--max_steps', type=int, default=200)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    from algos.split_rl import SPLIT_RL
    from algos.safety.ppo_lag_gbrl import PPOLagGBRL

    vec_env, wrapped = build_env(args.env_name, 0)
    unwrapped = wrapped.unwrapped
    wrapped.reset(seed=0)
    rng = np.random.default_rng(0)
    starts = start_states(unwrapped, 'sweep', args.episodes, True, rng)

    ref = SPLIT_RL.load(CKPT['Split-RL'], env=vec_env, device=args.device, force_reset=True)
    data = collect(ref, wrapped, starts, args.max_steps, 0)
    obs = np.asarray(data['obs'])
    labels = np.asarray(data['label']).astype(float)
    print(f"\n  {len(obs)} states, label-1 rate {labels.mean():.3f}\n")

    print(f"  {'model':<16}{'NMI(leaf; label)':>19}{'best leaf label rate':>23}{'trees':>8}")
    for name, path in CKPT.items():
        m = ref if name == 'Split-RL' else PPOLagGBRL.load(
            path, env=vec_env, device=args.device, force_reset=True)
        pt, pf, asg = purity(m, obs, labels, args.n_trees, args.stride)
        if len(pt) == 0:
            print(f"  {name:<16}{'no trees read':>14}"); continue
        print(f"  {name:<16}{pt.mean():>13.4f} ± {pt.std():<4.4f}"
              f"{pf.mean():>21.3f}{len(pt):>8}")


if __name__ == '__main__':
    main()
