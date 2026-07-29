"""FragileCrossing routing map, plus the per-tree distribution behind it.

Two outputs from the same per-tree density read:

1. The map. Same figure as Corner, but on enumerated (cell, direction, carrying)
   states rather than trajectory data, and restricted to carrying=True, since the
   guidance label there is `carrying_heavy AND (on_ice OR ice_to_the_east)` and cannot
   fire when the agent carries nothing.

2. The per-tree distribution. Averaging leaf densities over the ensemble is what the
   map does, and it is also what hides the mechanism: guided splitting only competes
   with the standard gain where the two gradient streams disagree, which is a minority
   of splits. Reading each tree separately shows whether the routing signal is spread
   thinly across all trees or concentrated in a few that discriminate strongly.
   Per tree we take AUC of that tree's cost-density against the ground-truth label,
   so 0.5 is an uninformative tree and 1.0 perfectly separates the labelled states.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from scripts.exp.corner_density_maps import (CMAP_STOPS, TEXT_PRIMARY, TEXT_SECONDARY,
                                             _desaturate, _learner, build_env,
                                             merge_objectives)
from scripts.exp.corner_density_figure import CAPTION_FS, SUBTITLE_FS, TITLE_FS, _panel_sized
from scripts.exp.leaf_isolation import ckpts, enumerate_fragile


def auc(scores, labels):
    """Rank-based AUC with correct tie handling; 0.5 = uninformative.

    Ties must get *average* ranks. Breaking them by array order makes a tree with
    constant densities score 0 rather than 0.5, and constant densities are common here:
    a tree fit on an all-safe minibatch has no label counts to record.
    """
    from scipy.stats import rankdata
    pos = labels > 0
    if pos.sum() == 0 or (~pos).sum() == 0:
        return 0.5
    r = rankdata(scores, method='average')
    npos, nneg = pos.sum(), (~pos).sum()
    return (r[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def rollout(model, wrapped, n_episodes, max_steps, seed, dump=None):
    """Evaluate the policy and record the episodes it actually produces.

    Mirrors collect() in corner_density_maps.py, minus the Corner-specific
    coins_collected read. Enumerating states is wrong for this environment because much
    of the grid is unreachable in a given carrying state, and unlike Corner the labelled
    states are not avoidable here -- crossing the ice while carrying is required to
    finish the task -- so real trajectories cover them.

    info is computed from the pre-step state, so info['safety_label'] pairs with the
    (x, y, dir, carrying) recorded before the step.
    """
    un = wrapped.unwrapped
    rec = {k: [] for k in ('ep', 'step', 'x', 'y', 'dir', 'carrying', 'label', 'cost', 'reward')}
    obss = []
    ends = []
    for ep in range(n_episodes):
        wrapped.reset(seed=seed + ep)
        obs = wrapped.observation(un.gen_obs())
        for t in range(max_steps):
            x, y, d = int(un.agent_pos[0]), int(un.agent_pos[1]), int(un.agent_dir)
            held = un.carrying is not None
            action, _ = model.predict(obs[None], deterministic=True)
            nxt, reward, term, trunc, info = wrapped.step(int(np.asarray(action).ravel()[0]))
            for k, v in (('ep', ep), ('step', t), ('x', x), ('y', y), ('dir', d),
                         ('carrying', held), ('label', float(info.get('safety_label', 0.0))),
                         ('cost', float(info.get('cost', 0.0))), ('reward', float(reward))):
                rec[k].append(v)
            obss.append(obs)
            obs = nxt
            if term or trunc:
                ends.append(t + 1)
                break
        else:
            ends.append(max_steps)
    out = {k: np.array(v) for k, v in rec.items()}
    out['obs'] = np.stack(obss)
    print(f"  {n_episodes} episodes, lengths min/med/max = "
          f"{min(ends)}/{int(np.median(ends))}/{max(ends)}, "
          f"{sum(e < max_steps for e in ends)}/{n_episodes} terminated")
    if dump:
        with open(dump, 'w') as f:
            f.write('ep step x y dir carrying label cost reward\n')
            for i in range(len(out['x'])):
                f.write(' '.join(str(out[k][i]) for k in
                        ('ep','step','x','y','dir','carrying','label','cost','reward')) + '\n')
        print(f"  wrote {dump}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env_name', default='MiniGrid-FragileCrossing-v0')
    ap.add_argument('--n_trees', type=int, default=250)
    ap.add_argument('--stride', type=int, default=40)
    ap.add_argument('--out', default='results/corner_density_300k_zf/fragile')
    ap.add_argument('--episodes', type=int, default=40)
    ap.add_argument('--max_steps', type=int, default=300)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    from algos.split_rl import SPLIT_RL
    from env.safety.utils import Ice

    vec_env, wrapped = build_env(args.env_name, 0)
    un = wrapped.unwrapped
    wrapped.reset(seed=0)
    m0 = SPLIT_RL.load(ckpts(args.env_name)['Split-RL'], env=vec_env,
                       device=args.device, force_reset=True)
    data = rollout(m0, wrapped, args.episodes, args.max_steps, 0,
                   dump=f'{args.out}_trajectories.txt')
    # The 300k policy learns to fetch and cross but not to release the object, so it
    # parks on the drop zone for the rest of the episode. Those idle steps are ~93% of
    # the buffer and would swamp a per-cell average, so each episode is truncated at
    # the step it first reaches its final position.
    keep = np.zeros(len(data['x']), dtype=bool)
    for ep in np.unique(data['ep']):
        idx = np.where(data['ep'] == ep)[0]
        pos = list(zip(data['x'][idx], data['y'][idx]))
        last = next((i + 1 for i in range(len(pos) - 1, -1, -1) if pos[i] != pos[-1]),
                    len(pos) - 1)
        keep[idx[:last + 1]] = True
    print(f"  trimmed idle tail: {keep.sum()}/{len(keep)} steps kept "
          f"({100*keep.mean():.1f}%)")
    data = {k: v[keep] for k, v in data.items()}
    obs, labels = data['obs'], data['label']
    carrying = data['carrying']; xs, ys = data['x'], data['y']
    print(f"\n  {len(obs)} visited states from {args.episodes} episodes, "
          f"{int(labels.sum())} label-1 ({labels.mean()*100:.1f}%), "
          f"{carrying.sum()} while carrying")

    m = SPLIT_RL.load(ckpts(args.env_name)['Split-RL'], env=vec_env,
                      device=args.device, force_reset=True)
    L = _learner(m)

    # per-tree densities -> cost share per state
    per_tree, aucs = [], []
    for ti in range(0, args.n_trees * args.stride, args.stride):
        try:
            d = L.predict_densities(obs, start_idx=ti, stop_idx=ti + 1)
        except Exception:
            break
        d = np.asarray(d, dtype=np.float64)
        # An all-ones row means the leaf never had its label counts computed: the
        # minibatch was entirely reward-labelled, so no cost samples existed to count.
        # It denotes PURE REWARD, not "mixed". Row-normalising it would read as 0.5 and
        # invert the meaning, so substitute the one-hot before merging.
        allones = np.abs(d.sum(1) - d.shape[1]) < 1e-3
        if allones.any():
            d[allones] = 0.0
            d[allones, 0] = 1.0
        s = merge_objectives(d)
        if not np.isfinite(s).any():
            continue
        s = np.nan_to_num(s)
        per_tree.append(s)
        aucs.append(auc(s, labels))
    per_tree = np.asarray(per_tree); aucs = np.asarray(aucs)
    score = per_tree.mean(0)
    print(f"  {len(aucs)} trees read")
    print(f"  ensemble-mean routing AUC : {auc(score, labels):.3f}")
    print(f"  per-tree AUC   mean {aucs.mean():.3f}   median {np.median(aucs):.3f}   "
          f"max {aucs.max():.3f}")
    for q in (0.60, 0.70, 0.80):
        print(f"    trees with AUC > {q:.2f}: {100*(aucs > q).mean():5.1f}%")

    # ---- map over carrying=True states ----
    h, w = un.height, un.width
    ice = np.array([[isinstance(un.grid.get(x, y), Ice) for x in range(w)] for y in range(h)])
    tile = 32
    bg = _desaturate(un.grid.render(tile, None, None))
    img = un.grid.render(tile, un.agent_pos, un.agent_dir)

    def cellmax(vals, mask):
        """Max over the four headings, within one carrying condition."""
        m = np.full((h, w), -np.inf)
        np.maximum.at(m, (ys[mask], xs[mask]), vals[mask])
        return np.where(np.isfinite(m), m, np.nan)

    def agg(vals):
        """Max over headings per carrying condition, then average the two conditions.

        Averaging over all states at once would let the four headings and the two
        carrying conditions dilute each other. Taking the max within each condition
        answers "does this fire from any heading", and averaging the two conditions
        then reports how much of the state space at that cell is cost-relevant.
        """
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter('ignore', RuntimeWarning)
            return np.nanmean(np.stack([cellmax(vals, carrying),
                                        cellmax(vals, ~carrying)]), axis=0)

    def cellmean(vals):
        tot = np.zeros((h, w)); cnt = np.zeros((h, w))
        np.add.at(tot, (ys, xs), vals); np.add.at(cnt, (ys, xs), 1)
        with np.errstate(invalid='ignore'):
            return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)

    # Ground truth is a property of the environment, so it is computed for every cell:
    # would the label fire here if the object were carried, i.e. on ice or beside ice.
    # Routing is only meaningful where the policy actually went, so it stays restricted
    # to visited cells.
    lab_map = np.full((h, w), np.nan)
    for x in range(w):
        for y in range(h):
            c = un.grid.get(x, y)
            if c is not None and not isinstance(c, Ice):
                continue
            east = un.grid.get(x + 1, y) if x + 1 < w else None
            lab_map[y, x] = float(isinstance(c, Ice) or isinstance(east, Ice))
    sc_map = cellmean(score)
    # Plot routing on its own ABSOLUTE scale. Rescaling to [0,1] made mid-colour read as
    # "mixed" when the underlying density was ~0.17, which misstates what the model does.
    smax = float(np.nanmax(sc_map))
    print(f"  routing | label-1 states {score[labels > 0].mean():.3f}   "
          f"label-0 states {score[labels == 0].mean():.3f}   panel max {smax:.3f}")

    cmap = LinearSegmentedColormap.from_list('routing', CMAP_STOPS)
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 6.8))
    fig.patch.set_facecolor('#fbfbfa')
    axes[0].imshow(img, interpolation='nearest')
    axes[0].set_title('FragileCrossing', fontsize=TITLE_FS, color=TEXT_PRIMARY, pad=9,
                      weight='medium')
    axes[0].text(0.5, -0.035, '(a)', transform=axes[0].transAxes, ha='center', va='top',
                 fontsize=SUBTITLE_FS, color=TEXT_SECONDARY)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for sp in axes[0].spines.values():
        sp.set_visible(False)
    _panel_sized(axes[1], bg, lab_map, ice, tile, cmap, 0.80,
                 'Ground-truth guidance label', '(b)  0 = reward, 1 = cost')
    _panel_sized(axes[2], bg, sc_map, ice, tile, cmap, 0.80,
                 'Learned routing (GBRL leaf density)', '(c)  same units, own range',
                 vmin=0.0, vmax=smax)

    fig.subplots_adjust(left=0.015, right=0.845, top=0.92, bottom=0.34, wspace=0.06)
    cax1 = fig.add_axes([0.862, 0.44, 0.013, 0.40])
    cb1 = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), cax=cax1)
    cb1.set_ticks([0, 0.5, 1]); cb1.set_ticklabels(['0', '0.5', '1'])
    cb1.set_label('(b) label', fontsize=SUBTITLE_FS - 1, color=TEXT_SECONDARY, labelpad=6)
    cax2 = fig.add_axes([0.935, 0.44, 0.013, 0.40])
    cb2 = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, smax)), cax=cax2)
    cb2.set_ticks([0, smax / 2, smax])
    cb2.set_ticklabels([f'{0:.2f}', f'{smax/2:.2f}', f'{smax:.2f}'])
    cb2.set_label('(c) cost-density', fontsize=SUBTITLE_FS - 1, color=TEXT_SECONDARY, labelpad=6)
    for cb in (cb1, cb2):
        cb.ax.tick_params(labelsize=SUBTITLE_FS - 1, colors=TEXT_SECONDARY, length=0)
        cb.outline.set_visible(False)

    import textwrap
    cap = textwrap.fill(
        "(a) The FragileCrossing environment; ice is shown pale blue. (b) The ground-truth "
        "guidance label. (c) Routing recovered from the trained ensemble, by passing each "
        "state through every tree and averaging the label composition of the leaf it "
        "reaches. The label fires when the agent carries the heavy object and is on or "
        "beside ice; (b) marks every cell where it would fire, as a property of the "
        "environment. (c) averages over the states the policy actually visited, so cells "
        "it never enters are left blank. Guidance "
        "labels are never supplied at evaluation time. Note the two colour bars: the label "
        "is 0 or 1, whereas leaf densities are small in absolute terms because most trees "
        f"route most states to the reward objective, so (c) spans 0 to {smax:.2f}. Averaged "
        f"over states, routing is {score[labels > 0].mean():.3f} on labelled states against "
        f"{score[labels == 0].mean():.3f} elsewhere.",
        width=118)
    fig.text(0.5, 0.285, cap, ha='center', va='top', fontsize=CAPTION_FS,
             color=TEXT_SECONDARY, linespacing=1.5)
    out = f'{args.out}_routing.png'
    fig.savefig(out, dpi=180, facecolor=fig.get_facecolor())
    print(f"  wrote {out}")


if __name__ == '__main__':
    main()
