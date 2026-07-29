#!/usr/bin/env python3.10
"""Objective-routing maps for MiniGrid-Corner from a Split-RL GBRL checkpoint.

Rolls out a deterministic policy from many different start states, records at every
step (a) the global cell the agent occupied, (b) the env's ``safety_label``, and
(c) GBRL's per-leaf density vector for the observation, then renders a three panel
figure:

    1. the Corner grid as-is
    2. the grid overlaid with a heatmap of the env's safety labels
    3. the grid overlaid with a heatmap of GBRL's objective density, visited cells only

POV vs. global view
-------------------
The policy sees a 7x7 *egocentric* observation, but the figure is drawn in the
*global* view. That is well defined because every observation is tagged with the
agent's global ``agent_pos`` / ``agent_dir`` at the moment it was produced, so each
POV observation maps onto exactly one global cell. The mapping is many-to-one --
one cell yields four different observations depending on facing direction (and
more as coins get collected) -- so a cell's value is the mean over all visits.
Per-(cell, direction) means are saved alongside for slicing.

The same many-to-one structure applies to the labels: ``safety_label`` is
``on_lava or facing_lava`` (env/safety/corner.py:148), which is itself direction
dependent, so both heatmaps are averaged the same way and stay comparable.

Density scale
-------------
GBRL returns one density per objective. They are merged into a single score:
0 = the leaves route this state to the reward objective, 1 = to the cost
objective, 0.5 = mixed. With ``n_objs=3`` the blended objective counts as 0.5.

Usage
-----
    python3.10 scripts/exp/corner_density_maps.py --stage both \
        --model_path saved_models/minigrid/minigrid/MiniGrid-Corner-v0/split_rl/fully_obs_seed_0_1000000_steps.zip

    # re-plot without re-rolling
    python3.10 scripts/exp/corner_density_maps.py --stage plot --out_dir results/corner_density
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Blue (reward) -> pale -> red (cost). vmin is pinned to 0 in every mode, so the pale
# midpoint always sits at exactly half of whatever the top of the scale is.
CMAP_STOPS = ['#2a78d6', '#f0efec', '#e34948']
MUTED_INK = '#898781'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'


# --------------------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------------------
def build_env(env_name: str, seed: int):
    """Recreate the exact training env (categorical obs + cost monitor)."""
    from env.register_minigrid import register_minigrid_tests
    register_minigrid_tests()
    from env.wrappers import MiniGridCategoricalObservationWrapper, CategoricalDummyVecEnv
    from utils.helpers import make_cost_vec_env

    vec_env = make_cost_vec_env(
        env_name, n_envs=1, seed=seed,
        wrapper_class=MiniGridCategoricalObservationWrapper,
        vec_env_cls=CategoricalDummyVecEnv,
    )
    # env.envs[0] is the obs wrapper; step it directly so VecEnv auto-reset never
    # fires between our injected start states.
    return vec_env, vec_env.envs[0]


def start_states(unwrapped, mode: str, n_episodes: int, include_lava: bool, rng) -> list:
    """Enumerate (x, y, dir) start states.

    Corner's _gen_grid is fully deterministic -- same layout, same centre start, every
    seed -- so a deterministic policy reseeded N times just replays one trajectory.
    Coverage has to come from varying the start state instead.
    """
    from minigrid.core.world_object import Lava

    cells = []
    for x in range(unwrapped.width):
        for y in range(unwrapped.height):
            cell = unwrapped.grid.get(x, y)
            if cell is None:
                cells.append((x, y))
            elif include_lava and isinstance(cell, Lava):
                cells.append((x, y))  # lava is overlappable; the danger zone is the point

    if mode == 'sweep':
        return [(x, y, d) for (x, y) in cells for d in range(4)]
    idx = rng.integers(0, len(cells), size=n_episodes)
    dirs = rng.integers(0, 4, size=n_episodes)
    return [(cells[i][0], cells[i][1], int(d)) for i, d in zip(idx, dirs)]


def collect(model, wrapped_env, starts, max_steps: int, seed: int) -> dict:
    """Roll out one deterministic episode per start state."""
    unwrapped = wrapped_env.unwrapped
    ep_id, step_i, xs, ys, dirs, labels, costs, rewards, coins, obss = ([] for _ in range(10))

    for ep, (sx, sy, sd) in enumerate(starts):
        wrapped_env.reset(seed=seed)              # regenerate grid + coins
        unwrapped.agent_pos = np.array([sx, sy])  # then inject the start state
        unwrapped.agent_dir = sd
        obs = wrapped_env.observation(unwrapped.gen_obs())

        for t in range(max_steps):
            x, y = int(unwrapped.agent_pos[0]), int(unwrapped.agent_pos[1])
            d, n_coins = int(unwrapped.agent_dir), int(unwrapped.coins_collected)
            action, _ = model.predict(obs[None], deterministic=True)
            next_obs, reward, terminated, truncated, info = wrapped_env.step(int(np.asarray(action).ravel()[0]))

            # info is computed from the pre-step state, so it pairs with obs/x/y/d.
            ep_id.append(ep); step_i.append(t); xs.append(x); ys.append(y); dirs.append(d)
            labels.append(float(info.get('safety_label', 0.0)))
            costs.append(float(info.get('cost', 0.0)))
            rewards.append(float(reward)); coins.append(n_coins); obss.append(obs)

            obs = next_obs
            if terminated or truncated:
                break

        if (ep + 1) % 100 == 0:
            print(f"  episode {ep + 1}/{len(starts)}  ({len(xs)} steps collected)")

    return dict(
        ep_id=np.array(ep_id), step=np.array(step_i), x=np.array(xs), y=np.array(ys),
        dir=np.array(dirs), label=np.array(labels), cost=np.array(costs),
        reward=np.array(rewards), coins_collected=np.array(coins),
        obs=np.stack(obss),
    )


def _learner(model):
    learner = model.policy.model.learner
    if not hasattr(learner, 'predict_densities'):
        raise RuntimeError(
            f"{type(learner).__name__} has no predict_densities; it is defined on GBTLearner "
            "and inherited by the shared-tree learners. Retrain with shared_tree_struct=True."
        )
    return learner


def predict_densities_raw(model, obs: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """Per-objective leaf densities, averaged along each sample's path over every tree."""
    learner = _learner(model)
    out = [np.asarray(learner.predict_densities(obs[i:i + chunk]))
           for i in range(0, len(obs), chunk)]
    return np.concatenate(out, axis=0).astype(np.float64), None


def scan_tree_validity(model):
    """Classify every tree as fully valid / fully all-ones / mixed.

    An all-ones row cannot come from a real label count (each sample increments exactly
    one counter), so it always marks a leaf whose density was never written.
    """
    learner = _learner(model)
    n_trees = model.policy.model.get_num_trees()
    dead = np.zeros(n_trees, dtype=bool)
    mixed = 0
    for t in range(n_trees):
        rows = np.asarray(learner.get_tree(t)['densities'])
        bad = ~np.isclose(rows.sum(axis=1), 1.0, atol=1e-3)
        if bad.all():
            dead[t] = True
        elif bad.any():
            mixed += 1
    return dead, mixed


def predict_densities_zerofill(model, obs: np.ndarray, chunk: int = 8192):
    """Treat an all-ones leaf as "every sample here was objective 0" -> one-hot [1,0,...].

    Once the empty-leaf bug is fixed, an all-ones row can only come from
    gbt_learner.py:169 nulling an all-zero label vector, which means the minibatch
    genuinely had no objective-1 samples. The honest reconstruction is a one-hot on
    objective 0, not "unknown" -- and crucially not the 0.5 that row-normalising
    [1,1] would produce, which reads as "mixed" when the truth is "purely reward".

    When every tree is pure (all leaves dead or all leaves valid) the substitution has
    an exact closed form, so only one prediction is needed:
        corrected[:, 0]  = raw[:, 0]                  (dead adds 1 either way)
        corrected[:, k>0] = raw[:, k] - n_dead/n_trees (dead adds 1, should add 0)
    """
    learner = _learner(model)
    n_trees = model.policy.model.get_num_trees()
    dead, mixed = scan_tree_validity(model)
    n_dead = int(dead.sum())
    print(f"  tree scan: {n_trees - n_dead}/{n_trees} trees valid, {n_dead} all-ones, "
          f"{mixed} partially damaged")

    if mixed == 0:
        raw, _ = predict_densities_raw(model, obs, chunk)
        out = raw.copy()
        out[:, 1:] -= n_dead / float(n_trees)
        votes = np.full(len(obs), n_trees, dtype=np.float64)
        return np.clip(out, 0.0, None), votes

    # Partially damaged trees exist (pre-fix checkpoint): fall back to per-tree work.
    print("  partially damaged trees present -> per-tree substitution "
          "(slower; expected only on pre-fix checkpoints)")
    uniq, inverse = np.unique(obs, axis=0, return_inverse=True)
    acc = None
    for t in range(n_trees):
        d = np.asarray(learner.predict_densities(uniq, start_idx=t, stop_idx=t + 1),
                       dtype=np.float64)
        bad = ~np.isclose(d.sum(axis=1), 1.0, atol=1e-3)
        d[bad] = 0.0
        d[bad, 0] = 1.0
        acc = d if acc is None else acc + d
    return (acc / n_trees)[inverse], np.full(len(obs), n_trees, dtype=np.float64)


def predict_densities_masked(model, obs: np.ndarray, tree_stride: int = 4):
    """Same average, but only over trees whose leaf carries a computed density.

    A leaf that received no samples at fit time stores all-ones (sum == n_objs) on the
    CUDA + oblivious path, and all-ones cannot arise from a real label count -- each
    sample increments exactly one counter, so a computed row always sums to 1. Those
    leaves are therefore identifiable with no false positives, and are excluded from
    the average rather than being allowed to drag it.

    This is the right estimator regardless: a leaf with no data has no opinion.
    See docs/gbrl_cuda_oblivious_density_bug.md.
    """
    learner = _learner(model)
    n_trees = model.policy.model.get_num_trees()
    # Observations repeat heavily across a sweep (same cell+dir+coin state), and the
    # per-tree probe costs one call per tree, so dedupe first.
    uniq, inverse = np.unique(obs, axis=0, return_inverse=True)
    probe = range(0, n_trees, tree_stride)
    print(f"  masked estimator: {len(uniq)} unique observations, "
          f"probing {len(probe)} of {n_trees} trees (stride {tree_stride})")

    acc, cnt = None, np.zeros(len(uniq))
    for i, t in enumerate(probe):
        d = np.asarray(learner.predict_densities(uniq, start_idx=int(t), stop_idx=int(t) + 1),
                       dtype=np.float64)
        if acc is None:
            acc = np.zeros_like(d)
        valid = np.abs(d.sum(axis=1) - 1.0) < 1e-3
        acc[valid] += d[valid]
        cnt += valid
        if (i + 1) % 2000 == 0:
            print(f"    {i + 1}/{len(probe)} trees, mean valid votes/state {cnt.mean():.0f}")

    with np.errstate(invalid='ignore', divide='ignore'):
        dens_u = np.where(cnt[:, None] > 0, acc / np.maximum(cnt, 1)[:, None], np.nan)
    n_dead = int((cnt == 0).sum())
    print(f"  valid votes per state: min {cnt.min():.0f}  median {np.median(cnt):.0f}  "
          f"max {cnt.max():.0f}  ({cnt.mean() / len(probe):.1%} of probed trees)")
    if n_dead:
        print(f"  !! {n_dead}/{len(uniq)} observations had no valid tree at all -> NaN")
    return dens_u[inverse], cnt[inverse]


def merge_objectives(densities: np.ndarray) -> np.ndarray:
    """Collapse the per-objective densities to one score in [0, 1].

    0 -> reward objective, 1 -> cost objective, 0.5 -> mixed. Row-normalising first
    makes this robust to the sum-to-1 invariant being violated (see
    docs/gbrl_cuda_oblivious_density_bug.md).
    """
    n_objs = densities.shape[1]
    norm = densities / np.clip(densities.sum(axis=1, keepdims=True), 1e-8, None)
    if n_objs == 2:
        return norm[:, 1]
    if n_objs == 3:
        return norm[:, 1] + 0.5 * norm[:, 2]  # blended objective sits at the midpoint
    raise ValueError(f"cannot merge n_objs={n_objs} into a binary reward/cost score")


def check_density_invariant(densities: np.ndarray, estimator: str) -> float:
    """Densities must sum to 1 per row. Returns the fraction that do."""
    sums = densities.sum(axis=1)
    finite = sums[np.isfinite(sums)]
    frac_ok = float(np.isclose(finite, 1.0, atol=1e-3).mean()) if finite.size else 0.0
    print(f"  density sum-to-1 check: {frac_ok:.1%} of rows valid "
          f"(min {finite.min():.3f}, max {finite.max():.3f})")
    if frac_ok >= 0.999:
        return frac_ok
    if estimator == 'raw':
        print("  " + "!" * 76)
        print("  !! DENSITIES ARE CORRUPTED -- panel 3 is not meaningful.")
        print("  !! All-ones leaves are being averaged in as if they were real.")
        print("  !! Use --density_estimator zerofill (or masked). "
              "See docs/gbrl_cuda_oblivious_density_bug.md.")
        print("  " + "!" * 76)
    else:
        how = ('read as one-hot on objective 0' if estimator == 'zerofill'
               else 'dropped from the average')
        print(f"  -> all-ones leaves present; {how} by --density_estimator {estimator}"
              " (see docs/gbrl_cuda_oblivious_density_bug.md)")
    return frac_ok


def aggregate(data: dict, score: np.ndarray, width: int, height: int, coins_filter,
              how: str = 'mean'):
    """Mean label / density per global cell, plus a per-(cell, direction) breakdown."""
    keep = np.ones(len(score), dtype=bool)
    if coins_filter is not None:
        keep = data['coins_collected'] == coins_filter
        print(f"  restricting to states with coins_collected == {coins_filter}: "
              f"{keep.sum()}/{len(keep)} steps")

    x, y = data['x'][keep], data['y'][keep]
    lbl, scr, drc = data['label'][keep], score[keep], data['dir'][keep]

    # The masked estimator yields NaN where no tree voted, so score keeps its own counter.
    ok = np.isfinite(scr)
    visits = np.zeros((height, width))
    label_sum = np.zeros((height, width))
    score_n = np.zeros((height, width))
    score_sum = np.zeros((height, width))
    np.add.at(visits, (y, x), 1.0)
    np.add.at(label_sum, (y, x), lbl)
    np.add.at(score_n, (y[ok], x[ok]), 1.0)
    np.add.at(score_sum, (y[ok], x[ok]), scr[ok])

    with np.errstate(invalid='ignore', divide='ignore'):
        label_map = np.where(visits > 0, label_sum / np.maximum(visits, 1), np.nan)
        score_map = np.where(score_n > 0, score_sum / np.maximum(score_n, 1), np.nan)

    visits_d = np.zeros((4, height, width))
    label_d = np.zeros((4, height, width))
    score_nd = np.zeros((4, height, width))
    score_d = np.zeros((4, height, width))
    np.add.at(visits_d, (drc, y, x), 1.0)
    np.add.at(label_d, (drc, y, x), lbl)
    np.add.at(score_nd, (drc[ok], y[ok], x[ok]), 1.0)
    np.add.at(score_d, (drc[ok], y[ok], x[ok]), scr[ok])
    with np.errstate(invalid='ignore', divide='ignore'):
        label_by_dir = np.where(visits_d > 0, label_d / np.maximum(visits_d, 1), np.nan)
        score_by_dir = np.where(score_nd > 0, score_d / np.maximum(score_nd, 1), np.nan)

    if how == 'direction':
        # Average the four per-direction values with EQUAL weight, instead of averaging over
        # visits. Otherwise a cell's value depends on how often the policy happened to walk
        # through it facing each way -- policy trivia, not a property of the state. With equal
        # weighting, "unsafe from exactly one of four headings" is always 0.25.
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN cells are unvisited
            label_map = np.nanmean(label_by_dir, axis=0)
            score_map = np.nanmean(score_by_dir, axis=0)

    if how == 'max':
        # safety_label is a function of (cell, direction), so a cell's mean over visits is
        # "fraction of directions that were unsafe" -- the intermediate shades. 'max' answers
        # the binary question "is this cell unsafe from ANY direction" instead.
        label_max = np.full((height, width), -np.inf)
        score_max = np.full((height, width), -np.inf)
        np.maximum.at(label_max, (y, x), lbl)
        np.maximum.at(score_max, (y[ok], x[ok]), scr[ok])
        label_map = np.where(visits > 0, label_max, np.nan)
        score_map = np.where(score_n > 0, score_max, np.nan)



    return dict(visits=visits, label_map=label_map, score_map=score_map,
                visits_by_dir=visits_d, label_by_dir=label_by_dir, score_by_dir=score_by_dir)


# --------------------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------------------
def _desaturate(img: np.ndarray, lighten: float = 0.45) -> np.ndarray:
    """Grayscale + lighten, so the overlay owns the colour channel and geometry survives."""
    gray = img.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])
    gray = 255.0 - (255.0 - gray) * (1.0 - lighten)
    return np.repeat(gray[:, :, None], 3, axis=2).astype(np.uint8) / 255.0


def _panel(ax, bg, values, lava_mask, tile, cmap, alpha, title, subtitle, vmin=0.0, vmax=1.0):
    import matplotlib.patches as mpatches
    h_px, w_px = bg.shape[0], bg.shape[1]
    ax.imshow(bg, extent=(0, w_px, h_px, 0), interpolation='nearest')
    if values is not None:
        ax.imshow(np.ma.masked_invalid(values), extent=(0, w_px, h_px, 0), origin='upper',
                  interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, zorder=2)
        # Outline lava in neutral ink -- geometry cue that cannot be mistaken for a value.
        for (ly, lx) in np.argwhere(lava_mask):
            ax.add_patch(mpatches.Rectangle((lx * tile, ly * tile), tile, tile, fill=False,
                                            edgecolor=MUTED_INK, linewidth=0.9,
                                            linestyle=(0, (2, 2)), zorder=3))
    ax.set_title(title, fontsize=12, color=TEXT_PRIMARY, pad=9, weight='medium')
    ax.text(0.5, -0.035, subtitle, transform=ax.transAxes, ha='center', va='top',
            fontsize=8.5, color=TEXT_SECONDARY)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def plot(npz_path: str, out_path: str, alpha: float, scale: str = 'absolute',
         debug_banner: bool = False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable

    z = np.load(npz_path, allow_pickle=True)
    img_color = z['img_color']
    bg_gray = _desaturate(z['img_plain'])
    lava_mask, tile = z['lava_mask'], int(z['tile_size'])
    label_map, score_map, visits = z['label_map'], z['score_map'], z['visits']
    score_by_dir = z['score_by_dir']
    frac_ok = float(z['density_frac_valid'])
    estimator = str(z['estimator']) if 'estimator' in z else 'raw'

    cmap = LinearSegmentedColormap.from_list('reward_cost', CMAP_STOPS)
    cmap.set_bad(alpha=0.0)  # unvisited cells stay uncoloured

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.4))
    fig.patch.set_facecolor('#fcfcfb')
    for ax in axes:
        ax.set_facecolor('#fcfcfb')

    n_vis = int((visits > 0).sum())
    n_free = int(np.isfinite(score_map).sum())

    _panel(axes[0], img_color / 255.0, None, lava_mask, tile, cmap, alpha,
           'Corner environment', '')
    _panel(axes[1], bg_gray, label_map, lava_mask, tile, cmap, alpha,
           'Environment safety label', '')
    # The routing score is genuinely small in absolute terms (few cost-labelled samples
    # survive once the agent is safe), so on a true 0..1 axis panel 3 would be uniformly
    # blue. Default is to rescale it by its own maximum, which puts both panels on one
    # 0..1 axis and lets a single colorbar serve the whole figure. Panel 3 then reads as
    # "relative to the most cost-routed cell", which the caption must state.
    finite = score_map[np.isfinite(score_map)]
    smax = float(finite.max()) if finite.size else 1.0
    if scale == 'normalized' and smax > 0:
        score_map = score_map / smax
        dir_max = np.nanmax(score_by_dir) if np.isfinite(score_by_dir).any() else 1.0
        score_by_dir = score_by_dir / (dir_max if dir_max > 0 else 1.0)
    s_vmin, s_vmax = 0.0, 1.0
    s_ticklabels = ['reward', 'mixed', 'cost']

    _panel(axes[2], bg_gray, score_map, lava_mask, tile, cmap, alpha,
           'GBRL leaf-density routing', '', vmin=s_vmin, vmax=s_vmax)

    def _cbar(ax_list, vmin, vmax, ticklabels):
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        cb = fig.colorbar(sm, ax=ax_list, orientation='horizontal', fraction=0.045,
                          pad=0.09, aspect=24)
        cb.set_ticks([vmin, (vmin + vmax) / 2.0, vmax])
        cb.set_ticklabels(ticklabels)
        cb.ax.tick_params(labelsize=10, colors=TEXT_SECONDARY, length=0)
        cb.outline.set_visible(False)
        return cb

    _cbar(list(axes), 0.0, 1.0, ['reward', 'mixed', 'cost'])

    if debug_banner and frac_ok < 0.999:
        fig.text(0.5, 0.965, f'[debug] {frac_ok:.1%} of raw leaf densities sum to 1; '
                             f'estimator={estimator}', ha='center', fontsize=8,
                 color=TEXT_SECONDARY)

    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"  wrote {out_path}")

    # Direction breakdown: same cell yields a different POV obs per facing.
    dir_names = ['heading east  \u2192', 'heading south  \u2193',
                 'heading west  \u2190', 'heading north  \u2191']
    fig2, axes2 = plt.subplots(1, 4, figsize=(17.5, 5.0))
    fig2.patch.set_facecolor('#fcfcfb')
    for d, ax in enumerate(axes2):
        _panel(ax, bg_gray, score_by_dir[d], lava_mask, tile, cmap, alpha,
               dir_names[d], '', vmin=s_vmin, vmax=s_vmax)
    sm2 = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=s_vmin, vmax=s_vmax))
    cbar2 = fig2.colorbar(sm2, ax=axes2, orientation='horizontal', fraction=0.045,
                          pad=0.06, aspect=60)
    cbar2.set_ticks([s_vmin, (s_vmin + s_vmax) / 2.0, s_vmax])
    cbar2.set_ticklabels(s_ticklabels)
    cbar2.ax.tick_params(labelsize=10, colors=TEXT_SECONDARY, length=0)
    cbar2.outline.set_visible(False)
    fig2.suptitle('Cost-routing depends on which way the agent faces', fontsize=13,
                  color=TEXT_PRIMARY, y=1.0)
    fig2.text(0.5, 0.93, 'Same grid in each panel; a cell is coloured by how strongly the model '
                         'routes it to the cost objective when the agent holds that heading.',
              ha='center', fontsize=9.5, color=TEXT_SECONDARY)
    out2 = out_path.replace('.png', '_by_direction.png')
    fig2.savefig(out2, dpi=180, bbox_inches='tight', facecolor=fig2.get_facecolor())
    print(f"  wrote {out2}")


# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', choices=['collect', 'plot', 'both'], default='both')
    p.add_argument('--model_path', type=str,
                   default='saved_models/minigrid/minigrid/MiniGrid-Corner-v0/split_rl/'
                           'fully_obs_seed_0_1000000_steps.zip')
    p.add_argument('--env_name', type=str, default='MiniGrid-Corner-v0')
    p.add_argument('--out_dir', type=str, default='results/corner_density')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--start_mode', choices=['sweep', 'random'], default='sweep',
                   help="'sweep' visits every free cell x 4 directions (uniform coverage); "
                        "'random' samples --n_episodes start states")
    p.add_argument('--n_episodes', type=int, default=400, help='only used by --start_mode random')
    p.add_argument('--max_steps_per_episode', type=int, default=64)
    p.add_argument('--include_lava_starts', action='store_true', default=True,
                   help='lava is overlappable, and it is the region of interest')
    p.add_argument('--no_lava_starts', dest='include_lava_starts', action='store_false')
    p.add_argument('--coins_filter', type=int, default=None,
                   help='restrict aggregation to states with this many coins collected, so the '
                        'heatmap matches the rendered grid exactly (default: all states)')
    p.add_argument('--density_estimator', choices=['zerofill', 'masked', 'raw'], default='zerofill',
                   help="'zerofill' reads an all-ones leaf as one-hot on objective 0 (correct once "
                        'the empty-leaf bug is fixed, since all-ones then only means "the minibatch '
                        'was all label 0"); \'masked\' instead drops those leaves from the average '
                        '(right for pre-fix checkpoints, where all-ones means "unknown"); '
                        "'raw' is plain predict_densities")
    p.add_argument('--tree_stride', type=int, default=4,
                   help='probe every Nth tree for the masked estimator (1 = exact, slower)')
    p.add_argument('--aggregate', choices=['direction', 'visit', 'max'], default='direction',
                   help="how to collapse a cell: 'direction' = mean of the four headings, equally "
                        "weighted (0.25 = unsafe from one heading); 'visit' = mean over visits "
                        "(weighted by how often the policy passed through); 'max' = binary")
    p.add_argument('--debug_banner', action='store_true',
                   help='print the density-invariant diagnostic on the figure '
                        '(off by default so figures are publication-clean)')
    p.add_argument('--tile_size', type=int, default=32)
    p.add_argument('--alpha', type=float, default=0.72)
    p.add_argument('--score_scale', choices=['normalized', 'absolute', 'relative'], default='normalized',
                   help="'absolute' keeps panel 3 on the 0=reward / 1=cost scale; 'relative' "
                        'stretches it to the observed range when the signal is small')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    npz_path = os.path.join(args.out_dir, 'corner_density.npz')
    png_path = os.path.join(args.out_dir, 'corner_density_maps.png')

    if args.stage in ('collect', 'both'):
        from algos.split_rl import SPLIT_RL
        from minigrid.core.world_object import Lava

        print(f"loading {args.model_path}")
        vec_env, wrapped = build_env(args.env_name, args.seed)
        model = SPLIT_RL.load(args.model_path, env=vec_env, device=args.device, force_reset=True)
        unwrapped = wrapped.unwrapped

        wrapped.reset(seed=args.seed)
        img_color = unwrapped.grid.render(args.tile_size, unwrapped.agent_pos, unwrapped.agent_dir)
        img_plain = unwrapped.grid.render(args.tile_size, None, None)
        w, h = unwrapped.width, unwrapped.height
        lava_mask = np.array([[isinstance(unwrapped.grid.get(x, y), Lava) for x in range(w)]
                              for y in range(h)])

        rng = np.random.default_rng(args.seed)
        starts = start_states(unwrapped, args.start_mode, args.n_episodes,
                              args.include_lava_starts, rng)
        print(f"collecting {len(starts)} episodes "
              f"(start_mode={args.start_mode}, max {args.max_steps_per_episode} steps each)")
        data = collect(model, wrapped, starts, args.max_steps_per_episode, args.seed)
        print(f"  {len(data['x'])} steps over {len(starts)} episodes")

        print(f"predicting leaf densities (estimator: {args.density_estimator})")
        raw_densities, _ = predict_densities_raw(model, data['obs'])
        frac_ok = check_density_invariant(raw_densities, args.density_estimator)
        if args.density_estimator == 'zerofill':
            densities, votes = predict_densities_zerofill(model, data['obs'])
        elif args.density_estimator == 'masked':
            densities, votes = predict_densities_masked(model, data['obs'], args.tree_stride)
        else:
            densities, votes = raw_densities, None
        score = merge_objectives(densities)
        agg = aggregate(data, score, w, h, args.coins_filter, args.aggregate)

        # Dynamic range is the readout that matters: a corrupted model is flat everywhere,
        # a healthy one should separate the lava region from the rest.
        finite = agg['score_map'][np.isfinite(agg['score_map'])]
        lava_cells = agg['score_map'][lava_mask & np.isfinite(agg['score_map'])]
        safe_cells = agg['score_map'][~lava_mask & np.isfinite(agg['score_map'])]
        print(f"  routing score over cells: min {finite.min():.3f}  max {finite.max():.3f}  "
              f"spread {finite.max() - finite.min():.3f}")
        if lava_cells.size and safe_cells.size:
            print(f"    mean on lava  {lava_cells.mean():.3f}   mean off lava {safe_cells.mean():.3f}"
                  f"   separation {lava_cells.mean() - safe_cells.mean():+.3f}")
        if finite.max() - finite.min() < 0.02:
            print("    (essentially flat on the absolute scale -- use --score_scale relative)")

        # A well-trained safe agent stops visiting unsafe states, so the cost objective sees
        # few samples and the absolute density stays small. Magnitude therefore understates
        # the result; rank separation against the env's own label is the honest measure.
        ok = np.isfinite(score)
        lbl_bin, scr = data['label'][ok] > 0.5, score[ok]
        if lbl_bin.any() and (~lbl_bin).any():
            pos, neg = scr[lbl_bin], scr[~lbl_bin]
            ranks = np.empty(len(scr), dtype=np.float64)
            ranks[np.argsort(scr, kind='stable')] = np.arange(1, len(scr) + 1)
            auc = (ranks[lbl_bin].mean() - (len(pos) + 1) / 2.0) / len(neg)
            print(f"  vs the env's safety label: AUC {auc:.3f}   "
                  f"unsafe mean {pos.mean():.4f} vs safe {neg.mean():.4f} "
                  f"({pos.mean() / max(neg.mean(), 1e-9):.1f}x)")

        np.savez_compressed(
            npz_path, img_color=img_color, img_plain=img_plain, lava_mask=lava_mask,
            tile_size=args.tile_size, width=w, height=h, densities=densities, score=score,
            raw_densities=raw_densities, density_frac_valid=frac_ok, n_objs=densities.shape[1],
            estimator=args.density_estimator,
            votes=votes if votes is not None else np.zeros(0),
            **{k: v for k, v in data.items() if k != 'obs'}, **agg,
        )
        print(f"  wrote {npz_path}")

    if args.stage in ('plot', 'both'):
        print("plotting")
        plot(npz_path, png_path, args.alpha, args.score_scale, args.debug_banner)


if __name__ == '__main__':
    main()
