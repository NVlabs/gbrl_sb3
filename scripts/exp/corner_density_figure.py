"""Final 2x3 interpretability figure for the rebuttal.

Combines what were previously two separate figures (aggregated maps, and the
per-heading breakdown) into one panel grid with a single vertical colour bar.

Layout:

    row 1   environment | ground-truth guidance label | learned routing
    row 2   heading east | heading south | heading west      (learned routing)

Row 2 shows the learned routing resolved by heading. The guidance label depends
on the agent's orientation as well as its position -- it fires when the cell
directly ahead is lava -- so a per-cell map necessarily aggregates over four
headings, and row 2 is what that aggregation hides.

Two aggregation choices for row 1, produced as separate files:

    --how max     a cell is coloured by the strongest routing over the four
                  headings, i.e. "is this cell cost-relevant from ANY heading".
    --how mean    equal-weighted mean over the four headings, so a cell that is
                  unsafe from exactly one heading reads 0.25.

'mean' is the honest average but washes out to near-white next to lava, which is
why 'max' exists; both are generated so the clearer one can be chosen.

Note on scaling: the label is 0/1 but leaf densities live around 0.01-0.06, since
most trees route most states to the reward objective. On one shared absolute scale
the routing panels collapse to uniform blue, so routing is rescaled by its own
maximum. Centre and right are therefore comparable in PATTERN, not in magnitude --
the claim is that routing recovers the label geometry, not that the values match.

Usage:
    python3.10 scripts/exp/corner_density_figure.py \
        --npz results/corner_density_300k_zf/corner_density.npz \
        --out results/corner_density_300k_zf/figure
"""
import argparse
import os
import sys
import textwrap
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from scripts.exp.corner_density_maps import (CMAP_STOPS, MUTED_INK, TEXT_PRIMARY,
                                             TEXT_SECONDARY, _desaturate, _panel)

HEADINGS = [(0, 'heading east', '→'), (1, 'heading south', '↓'),
            (2, 'heading west', '←'), (3, 'heading north', '↑')]


def reaggregate(z, how: str):
    """Rebuild per-cell maps from the per-sample records stored in the npz."""
    h, w = int(z['height']), int(z['width'])
    x, y, drc = z['x'], z['y'], z['dir']
    lbl, scr = z['label'], z['score']
    ok = np.isfinite(scr)

    visits_d = np.zeros((4, h, w))
    label_d = np.zeros((4, h, w))
    score_nd = np.zeros((4, h, w))
    score_sd = np.zeros((4, h, w))
    np.add.at(visits_d, (drc, y, x), 1.0)
    np.add.at(label_d, (drc, y, x), lbl)
    np.add.at(score_nd, (drc[ok], y[ok], x[ok]), 1.0)
    np.add.at(score_sd, (drc[ok], y[ok], x[ok]), scr[ok])
    with np.errstate(invalid='ignore', divide='ignore'):
        label_by_dir = np.where(visits_d > 0, label_d / np.maximum(visits_d, 1), np.nan)
        score_by_dir = np.where(score_nd > 0, score_sd / np.maximum(score_nd, 1), np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)   # all-NaN cells are unvisited
        if how == 'max':
            label_map = np.nanmax(label_by_dir, axis=0)
            score_map = np.nanmax(score_by_dir, axis=0)
        else:
            # Equal weight per heading, not per visit: otherwise a cell's value
            # reflects how often the policy happened to walk through it facing
            # each way, which is policy trivia rather than a property of the state.
            label_map = np.nanmean(label_by_dir, axis=0)
            score_map = np.nanmean(score_by_dir, axis=0)
    return label_map, score_map, label_by_dir, score_by_dir


TITLE_FS = 15.0
SUBTITLE_FS = 11.0
CBAR_FS = 12.0
CAPTION_FS = 11.5

CAPTION = (
    "(a) The Corner environment. (b) The ground-truth guidance label, which fires when the "
    "agent stands on lava or faces it; a cell is marked if the label fires from any heading, "
    "so this is a property of the environment and is shown for every cell. (c) Routing "
    "recovered from the trained ensemble, by passing each visited state through every tree "
    "and averaging the label composition of the leaf it reaches. The two panels use separate "
    "colour bars: the label is 0 or 1, whereas leaf densities are small in absolute terms "
    "because most trees route most states to the reward objective, so (c) is shown over its "
    "own range. Guidance labels are never supplied at evaluation time."
)


def _panel_sized(ax, bg, values, lava_mask, tile, cmap, alpha, title, subtitle,
                 vmin=0.0, vmax=1.0):
    """_panel with controllable font sizes."""
    import matplotlib.patches as mpatches
    h_px, w_px = bg.shape[0], bg.shape[1]
    ax.imshow(bg, extent=(0, w_px, h_px, 0), interpolation='nearest')
    if values is not None:
        ax.imshow(np.ma.masked_invalid(values), extent=(0, w_px, h_px, 0), origin='upper',
                  interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax,
                  alpha=alpha, zorder=2)
        for (ly, lx) in np.argwhere(lava_mask):
            ax.add_patch(mpatches.Rectangle((lx * tile, ly * tile), tile, tile, fill=False,
                                            edgecolor=MUTED_INK, linewidth=0.9,
                                            linestyle=(0, (2, 2)), zorder=3))
    ax.set_title(title, fontsize=TITLE_FS, color=TEXT_PRIMARY, pad=9, weight='medium')
    ax.text(0.5, -0.035, subtitle, transform=ax.transAxes, ha='center', va='top',
            fontsize=SUBTITLE_FS, color=TEXT_SECONDARY)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def _standable(z, x, y):
    """Cells the agent can occupy: empty, or lava (which is overlappable)."""
    lava = np.asarray(z['lava_mask'])
    if lava[y, x]:
        return True
    return bool(np.isfinite(np.asarray(z['visits'])[y, x]) and np.asarray(z['visits'])[y, x] > 0)


def build(npz_path: str, out_path: str, how: str, alpha: float = 0.80):
    z = np.load(npz_path, allow_pickle=True)
    tile = int(z['tile_size'])
    lava = z['lava_mask']
    bg = _desaturate(z['img_plain'])
    label_map, score_map, _, score_by_dir = reaggregate(z, how)

    # Ground truth over EVERY cell, not just the visited ones: the label is a property
    # of the environment. A cell counts if the guidance fires from any heading, i.e. if
    # the agent stands on lava or faces it. Routing stays restricted to visited cells.
    lava = np.asarray(z['lava_mask'])
    H, W = lava.shape
    gt = np.full((H, W), np.nan)
    for y in range(H):
        for x in range(W):
            if not _standable(z, x, y):
                continue
            nb = [(x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]
            faces = any(0 <= b < W and 0 <= a < H and lava[a, b] for b, a in nb)
            gt[y, x] = float(lava[y, x] or faces)
    label_map = gt

    cmap = LinearSegmentedColormap.from_list('routing', CMAP_STOPS)
    agg_word = 'strongest' if how == 'max' else 'mean'

    # Leaf densities are small in absolute terms -- most trees route most states to
    # the reward objective, so the routing signal lives in roughly 0.01-0.06 while the
    # label is 0/1. Plotted on one shared 0..1 scale the routing panels collapse to
    # uniform blue and the structure is invisible. Rescale routing by its own maximum
    # so the two panels are comparable in PATTERN, and say so on the colour bar: the
    # claim is that routing recovers the label geometry, not that the magnitudes match.
    # Absolute scale for routing: rescaling to [0,1] made mid-colour read as "mixed"
    # when the density was far lower, which misstates the model.
    smax = float(np.nanmax(score_map)) if np.isfinite(score_map).any() else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 6.8))
    fig.patch.set_facecolor('#fbfbfa')

    axes[0].imshow(z['img_color'], interpolation='nearest')
    axes[0].set_title('Corner environment', fontsize=TITLE_FS, color=TEXT_PRIMARY,
                         pad=9, weight='medium')
    axes[0].text(0.5, -0.035, '(a)', transform=axes[0].transAxes, ha='center',
                    va='top', fontsize=SUBTITLE_FS, color=TEXT_SECONDARY)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for s in axes[0].spines.values():
        s.set_visible(False)

    _panel_sized(axes[1], bg, label_map, lava, tile, cmap, alpha,
                 'Ground-truth guidance label', '(b)  0 = reward, 1 = cost')
    _panel_sized(axes[2], bg, score_map, lava, tile, cmap, alpha,
                 'Learned routing (GBRL leaf density)', '(c)  same units, own range',
                 vmin=0.0, vmax=smax)

    # Explicit margins + a dedicated colour-bar axes. Using ax=axes steals space from
    # the axes bounding boxes, but the per-panel subtitles are drawn OUTSIDE those boxes,
    # so they end up underneath the bar. Reserving the right margin by hand avoids that.
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

    finite = np.isfinite(label_map) & np.isfinite(score_map)
    r = np.corrcoef(label_map[finite], score_map[finite])[0, 1] if finite.sum() > 2 else float('nan')

    caption = textwrap.fill(CAPTION.format(r=r, n=int(finite.sum())), width=118)
    fig.text(0.5, 0.285, caption, ha='center', va='top', fontsize=CAPTION_FS,
             color=TEXT_SECONDARY, linespacing=1.5)

    fig.savefig(out_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  wrote {out_path}")
    print(f"    label vs routing correlation ({how}, {int(finite.sum())} cells): {r:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default='results/corner_density_300k_zf/corner_density.npz')
    ap.add_argument('--out', default='results/corner_density_300k_zf/figure')
    ap.add_argument('--alpha', type=float, default=0.80)
    args = ap.parse_args()
    for how in ('max', 'mean'):
        build(args.npz, f'{args.out}_{how}.png', how, args.alpha)


if __name__ == '__main__':
    main()
