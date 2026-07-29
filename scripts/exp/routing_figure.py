"""Routing figure: does the ensemble recover where the guidance applies?

One script for all three MiniGrid safety environments. Panel (b) is the ground-truth
guidance label, panel (c) is the routing recovered from the trained ensemble by pushing
each state through every tree and averaging the label composition of the leaf it
reaches.

Where the states come from depends on whether the labelled states are avoidable.

In FragileCrossing and DynamicCrossing the agent must cross the hazard to finish, so
evaluated episodes cover the labelled states and are the faithful choice -- much of
those grids is unreachable in a given inventory or hazard configuration anyway.

Corner is the opposite: the lava can be walked around, so a trained policy never faces
it and trajectories contain no labelled states at all (measured: 0 of 5080). Panel (c)
is therefore empty of guidance for Corner and this script has nothing to say about it --
answering that env needs an enumerated state set, which is a different figure.

Panel (b) is always computed for every cell as a property of the environment. For
DynamicCrossing that includes the whole track the ball can occupy, not just where it
happens to sit in one frame.

Usage:
    python3.10 scripts/exp/routing_figure.py --env_name MiniGrid-Corner-v0
"""
import argparse
import os
import sys
import textwrap

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from scripts.exp.corner_density_maps import (CMAP_STOPS, MUTED_INK, TEXT_PRIMARY,
                                             TEXT_SECONDARY, _desaturate, _learner,
                                             build_env, merge_objectives)

TITLE_FS, SUB_FS, CAP_FS = 15.0, 11.0, 11.5
CKPT_STEPS = {'MiniGrid-Corner-v0': 300000, 'MiniGrid-FragileCrossing-v0': 300000,
              'MiniGrid-DynamicCrossing-v0': 200000}
NICE = {'MiniGrid-Corner-v0': 'Corner', 'MiniGrid-FragileCrossing-v0': 'FragileCrossing',
        'MiniGrid-DynamicCrossing-v0': 'DynamicCrossing'}


def rollout(model, wrapped, n_episodes, max_steps, seed, deterministic=True):
    """Evaluate the policy; record position and the env's own label per step.

    Greedy actions, as the paper evaluates. With the layout held fixed (see below) the
    trained policy is near-deterministic, so this traces essentially one path: on
    DynamicCrossing 40 episodes give 8 distinct cells greedily and 9 when sampling
    instead. Panel (c) is therefore sparse by nature -- a trained safe agent visits few
    states -- and that sparsity is the finding, not an artefact of greedy evaluation.
    """
    un = wrapped.unwrapped
    xs, ys, labs, obss, eps, carry, dirs = [], [], [], [], [], [], []
    for ep in range(n_episodes):
        # Same seed every episode: DynamicCrossing redraws its lava opening on each
        # reset, so varying it would collect trajectories from layouts the ground-truth
        # map does not describe. Within an episode the ball still moves, which is the
        # dynamic part that matters here.
        wrapped.reset(seed=seed)
        obs = wrapped.observation(un.gen_obs())
        for t in range(max_steps):
            x, y = int(un.agent_pos[0]), int(un.agent_pos[1])
            car = getattr(un, 'carrying', None) is not None
            dr = int(un.agent_dir)
            act, _ = model.predict(obs[None], deterministic=deterministic)
            nxt, _, term, trunc, info = wrapped.step(int(np.asarray(act).ravel()[0]))
            xs.append(x); ys.append(y); eps.append(ep)
            carry.append(car); dirs.append(dr)
            labs.append(float(info.get('safety_label', 0.0)))
            obss.append(obs)
            obs = nxt
            if term or trunc:
                break
    d = dict(x=np.array(xs), y=np.array(ys), ep=np.array(eps), carry=np.array(carry),
             dir=np.array(dirs),
             label=np.array(labs), obs=np.stack(obss))
    # A policy that finishes early then idles would let one cell dominate the average,
    # so each episode is cut at the step it first reaches its final position.
    keep = np.zeros(len(d['x']), bool)
    for ep in np.unique(d['ep']):
        idx = np.where(d['ep'] == ep)[0]
        pos = list(zip(d['x'][idx], d['y'][idx]))
        last = next((i + 1 for i in range(len(pos) - 1, -1, -1) if pos[i] != pos[-1]),
                    len(pos) - 1)
        keep[idx[:last + 1]] = True
    return {k: v[keep] for k, v in d.items()}


DIRVEC = ((1, 0), (0, 1), (-1, 0), (0, -1))


def ball_track(un):
    """Every cell the moving obstacle can occupy over an episode.

    obstacle_min_y/max_y describe the steady-state cycle only. The ball is *placed*
    one cell above that cycle at reset (opening_y - 2) and leaves it after the first
    step, so reading the bounds alone misses a cell the guidance can fire from. Its
    position at reset is added explicitly; call this straight after a reset.
    """
    cells = {(un.obstacle_start_x, y)
             for y in range(un.obstacle_min_y, un.obstacle_max_y + 1)}
    if getattr(un, 'obstacles', None):
        cells.add(tuple(int(v) for v in un.obstacles[0].cur_pos))
    return sorted(cells)


def label_mean_map(un, env_name):
    """Guidance label averaged over every state the agent can be in at each cell.

    The label is not a property of the cell alone: it depends on heading, and in
    FragileCrossing on whether the object is carried, and in DynamicCrossing on where
    the ball currently is. Averaging over those conditions is what makes a hazard cell
    read as partly cost rather than fully cost -- the guidance fires there in some
    states and not others, and the map should say so.
    """
    from minigrid.core.world_object import Lava
    from env.safety.utils import Ice
    W, H = un.width, un.height
    hz = Ice if 'Fragile' in env_name else Lava
    lava = {(x, y) for x in range(W) for y in range(H)
            if isinstance(un.grid.get(x, y), hz)}
    gt = np.full((H, W), np.nan)
    # The ball blocks its cell only while it is there, and it moves every step, so
    # track cells are standable even though the grid holds an object at reset time.
    track = set(ball_track(un)) if 'Dynamic' in env_name else set()

    for x in range(W):
        for y in range(H):
            c = un.grid.get(x, y)
            if c is not None and not isinstance(c, hz) and (x, y) not in track:
                continue                                   # walls, goals: not standable
            vals = []
            if 'Fragile' in env_name:
                # fires only while carrying: half the (carrying, heading) states
                on_or_east = (x, y) in lava or (x + 1, y) in lava
                for carrying in (False, True):
                    for _ in DIRVEC:
                        vals.append(float(carrying and on_or_east))
            elif 'Dynamic' in env_name:
                # The ball occupies one track cell at a time, so averaging over its
                # positions would divide the label by the track length and understate
                # where guidance can fire. Take the max instead: does guidance fire here
                # in ANY reachable configuration.
                vals.append(max(float((x + dx, y + dy) in lava or (x + dx, y + dy) in track)
                                for dx, dy in DIRVEC))
            else:
                for dx, dy in DIRVEC:                      # on lava, or facing it
                    vals.append(float((x, y) in lava or (x + dx, y + dy) in lava))
            gt[y, x] = float(np.mean(vals))
    return gt


UNVISITED = '#b6b4b0'      # solid grey: absence of data, never confusable with 0.5


def panel(ax, bg, vals, hazard, tile, cmap, title, sub, vmax=1.0, grey_nan=False):
    h_px, w_px = bg.shape[0], bg.shape[1]
    ax.imshow(bg, extent=(0, w_px, h_px, 0), interpolation='nearest')
    if grey_nan and vals is not None:
        # The colour map's midpoint is white, which is close to the desaturated
        # background, so an unvisited cell would read as "mixed". Paint it grey first.
        import matplotlib.colors as mcolors
        miss = np.zeros(vals.shape + (4,))
        miss[np.isnan(vals)] = mcolors.to_rgba(UNVISITED)
        ax.imshow(miss, extent=(0, w_px, h_px, 0), origin='upper',
                  interpolation='nearest', zorder=1.5)
    if vals is not None:
        ax.imshow(np.ma.masked_invalid(vals), extent=(0, w_px, h_px, 0), origin='upper',
                  interpolation='nearest', cmap=cmap, vmin=0.0, vmax=vmax, alpha=0.85,
                  zorder=2)
        for (ly, lx) in np.argwhere(hazard):
            ax.add_patch(mpatches.Rectangle((lx * tile, ly * tile), tile, tile, fill=False,
                                            edgecolor=MUTED_INK, linewidth=0.9,
                                            linestyle=(0, (2, 2)), zorder=3))
    ax.set_title(title, fontsize=TITLE_FS, color=TEXT_PRIMARY, pad=8, weight='medium')
    ax.text(0.5, -0.04, sub, transform=ax.transAxes, ha='center', va='top',
            fontsize=SUB_FS, color=TEXT_SECONDARY)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--env_name', default='MiniGrid-Corner-v0')
    ap.add_argument('--episodes', type=int, default=40)
    ap.add_argument('--max_steps', type=int, default=300)
    ap.add_argument('--n_trees', type=int, default=250)
    ap.add_argument('--stride', type=int, default=40)
    ap.add_argument('--out', default='results/rebuttal_figures')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--sample', action='store_true',
                    help='sample actions instead of acting greedily')
    args = ap.parse_args()

    from algos.split_rl import SPLIT_RL
    from minigrid.core.world_object import Lava
    from env.safety.utils import Ice

    vec_env, wrapped = build_env(args.env_name, 0)
    un = wrapped.unwrapped
    wrapped.reset(seed=0)
    ckpt = (f'saved_models/minigrid/minigrid/{args.env_name}/split_rl/'
            f'fully_obs_seed_0_{CKPT_STEPS[args.env_name]}_steps.zip')
    model = SPLIT_RL.load(ckpt, env=vec_env, device=args.device, force_reset=True)

    model.set_random_seed(0)
    d = rollout(model, wrapped, args.episodes, args.max_steps, 0, not args.sample)
    cells = len(set(zip(d['x'].tolist(), d['y'].tolist())))
    print(f"  {len(d['x'])} states from {args.episodes} episodes, {cells} distinct cells, "
          f"label-1 {d['label'].mean()*100:.1f}%")
    # The rollout mutates the grid, so restore the layout the episodes actually ran on
    # before reading it for the label map and the rendering.
    wrapped.reset(seed=0)

    L = _learner(model)
    acc, n = None, 0
    for ti in range(0, args.n_trees * args.stride, args.stride):
        try:
            dd = np.asarray(L.predict_densities(d['obs'], start_idx=ti, stop_idx=ti + 1),
                            dtype=np.float64)
        except Exception:
            break
        # All-ones rows are leaves whose label counts were never computed because the
        # minibatch was entirely reward-labelled. That means PURE REWARD, not "mixed".
        ao = np.abs(dd.sum(1) - dd.shape[1]) < 1e-3
        if ao.any():
            dd[ao] = 0.0; dd[ao, 0] = 1.0
        s = merge_objectives(dd)
        acc = s if acc is None else acc + s
        n += 1
    score = acc / n

    H, W = un.height, un.width
    tot = np.zeros((H, W)); cnt = np.zeros((H, W))
    np.add.at(tot, (d['y'], d['x']), score); np.add.at(cnt, (d['y'], d['x']), 1)
    with np.errstate(invalid='ignore'):
        sc_map = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)

    lab_map = label_mean_map(un, args.env_name)

    smax = float(np.nanmax(sc_map))
    l1, l0 = score[d['label'] > 0], score[d['label'] == 0]
    print(f"  routing: labelled {l1.mean():.3f}  unlabelled {l0.mean():.3f}  max {smax:.3f}")

    hz = Ice if 'Fragile' in args.env_name else Lava
    hazard = np.array([[isinstance(un.grid.get(x, y), hz) for x in range(W)]
                       for y in range(H)])
    tile = 32
    bg = _desaturate(un.grid.render(tile, None, None))
    img = un.grid.render(tile, un.agent_pos, un.agent_dir)
    cmap = LinearSegmentedColormap.from_list('routing', CMAP_STOPS)

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 6.0))
    fig.patch.set_facecolor('#fbfbfa')
    axes[0].imshow(img, interpolation='nearest')
    axes[0].set_title(NICE[args.env_name], fontsize=TITLE_FS, color=TEXT_PRIMARY, pad=8,
                      weight='medium')
    axes[0].text(0.5, -0.04, '(a)', transform=axes[0].transAxes, ha='center', va='top',
                 fontsize=SUB_FS, color=TEXT_SECONDARY)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for s in axes[0].spines.values():
        s.set_visible(False)
    panel(axes[1], bg, lab_map, hazard, tile, cmap, 'Guidance label', '(b)  all states',
          grey_nan=True)
    # One colour bar for both panels: routing is shown relative to its own maximum,
    # which is stated in the subtitle so the absolute magnitude is not lost.
    panel(axes[2], bg, sc_map / smax, hazard, tile, cmap, 'Learned routing',
          f'(c)  collected trajectories, relative to max {smax:.2f}', grey_nan=True)

    fig.subplots_adjust(left=0.02, right=0.87, top=0.93, bottom=0.26, wspace=0.06)
    cax = fig.add_axes([0.895, 0.36, 0.014, 0.50])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), cax=cax)
    cb.set_ticks([0, 0.5, 1]); cb.set_ticklabels(['reward', '', 'cost'])
    cb.ax.tick_params(labelsize=SUB_FS, colors=TEXT_SECONDARY, length=0)
    cb.outline.set_visible(False)

    cap = textwrap.fill(
        f"(a) The {NICE[args.env_name]} environment. (b) The guidance label, averaged over "
        f"every state the agent can occupy at each cell. (c) Routing recovered from the "
        f"trained ensemble along collected trajectories, scaled by its own maximum; grey "
        f"marks cells with no data. Guidance labels are never supplied at evaluation time.",
        width=112)
    fig.text(0.5, 0.20, cap, ha='center', va='top', fontsize=CAP_FS,
             color=TEXT_SECONDARY, linespacing=1.5)

    out = f"{args.out}/routing_{NICE[args.env_name].lower()}.png"
    fig.savefig(out, dpi=180, facecolor=fig.get_facecolor())
    print(f"  wrote {out}")


if __name__ == '__main__':
    main()
