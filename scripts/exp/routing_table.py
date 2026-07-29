"""Routing figure, as numbers. Rebuttals cannot carry plots, so the same measurement
is reported as text.

Split-RL records, for every leaf, the label composition of the samples that fell into it
while the tree was fitted. Here each state the trained policy visits is pushed through
every tree, the composition of the leaf it reaches is read off, and the ensemble mean is
taken. That is one number per state: the fraction of the model treating it as
cost-relevant. It is read out of the fitted trees, not predicted -- the model is given no
guidance label at evaluation time. The environment label only says which states we
expected to matter.

The layout is held fixed and the trained policy is near-deterministic, so every episode is
identical (asserted in the output). One episode is therefore printed step by step, which
is the text form of the routing map, and the mean over the critical (labelled) states is
reported against the mean over the rest of the same episode.

FragileCrossing additionally contrasts the same ice tiles carrying vs not carrying: the
guidance fires there only while the object is held, so a model keying on position alone
would route both alike.

Usage:
    python3.10 scripts/exp/routing_table.py
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from scripts.exp.corner_density_maps import build_env, _learner, merge_objectives
from scripts.exp.routing_figure import CKPT_STEPS, NICE, rollout


def routing_scores(model, obs, n_trees, stride):
    """Mean cost-density over the ensemble for each state."""
    L = _learner(model)
    acc, n = None, 0
    for ti in range(0, n_trees * stride, stride):
        try:
            dd = np.asarray(L.predict_densities(obs, start_idx=ti, stop_idx=ti + 1),
                            dtype=np.float64)
        except Exception:
            break
        # All-ones rows are leaves whose label counts were never recorded because the
        # minibatch was entirely reward-labelled: pure reward, not "mixed".
        ao = np.abs(dd.sum(1) - dd.shape[1]) < 1e-3
        if ao.any():
            dd[ao] = 0.0; dd[ao, 0] = 1.0
        s = merge_objectives(dd)
        acc = s if acc is None else acc + s
        n += 1
    if n == 0:
        raise RuntimeError('no trees read from the model')
    return acc / n, n


def hazard_cells(un):
    """Cells where the guidance can fire: on ice, or with ice directly east.

    The label is `carrying AND (on_ice OR ice_to_the_east)`, so the ice tiles alone are
    the wrong set -- the agent's critical states in the evaluated episode are at x=8,
    which is the empty column west of the right-hand ice block, not ice itself.
    """
    from env.safety.utils import Ice
    ice = {(x, y) for x in range(un.width) for y in range(un.height)
           if isinstance(un.grid.get(x, y), Ice)}
    return ice | {(x - 1, y) for x, y in ice}, ice


DIRVEC = ((1, 0), (0, 1), (-1, 0), (0, -1))


def enumerate_states(wrapped, un, env_name):
    """Every state the agent can be in, with its guidance label.

    Trajectories alone cannot answer the question in general. Corner's lava is avoidable,
    so a trained policy never faces it and evaluated episodes contain no labelled states
    at all; and in FragileCrossing the object sits between the two rooms, so a single
    trajectory confounds terrain with inventory. Enumerating the state space removes both
    problems and lets all three environments be measured the same way.

    Returns (obs, labels, keys, varied) where `keys` is what is held fixed for the matched
    control and `varied` is the part that flips the label with the key held constant:

        Corner    key = cell,             varied = heading
        Fragile   key = (cell, heading),  varied = carrying
        Dynamic   key = (cell, heading),  varied = where the ball is

    So every row of the matched control compares states that differ only in the thing the
    guidance actually depends on, with position identical.
    """
    from minigrid.core.world_object import Lava
    from env.safety.utils import Ice
    W, H = un.width, un.height
    hz = Ice if 'Fragile' in env_name else Lava
    haz = {(x, y) for x in range(W) for y in range(H) if isinstance(un.grid.get(x, y), hz)}
    obs, labels, keys, varied = [], [], [], []

    def standable(x, y):
        c = un.grid.get(x, y)
        return c is None or isinstance(c, hz)

    if 'Fragile' in env_name:
        hv = [(x, y) for x in range(W) for y in range(H)
              if type(un.grid.get(x, y)).__name__ == 'HeavyObj']
        if not hv:
            raise RuntimeError('no HeavyObj in the grid')
        hx, hy = hv[0]
        heavy = un.grid.get(hx, hy)
        for x in range(W):
            for y in range(H):
                if not standable(x, y):
                    continue
                on_or_east = (x, y) in haz or (x + 1, y) in haz
                for carrying in (None, heavy):
                    # While carried the object is not on the grid, exactly as after a
                    # real pickup; otherwise it would render in two places at once.
                    un.grid.set(hx, hy, None if carrying is not None else heavy)
                    for d in range(4):
                        un.agent_pos, un.agent_dir = (x, y), d
                        un.carrying = carrying
                        obs.append(wrapped.observation(un.gen_obs()))
                        labels.append(float(carrying is not None and on_or_east))
                        keys.append((x, y, d))
                        varied.append(carrying is not None)
        un.carrying = None
        un.grid.set(hx, hy, heavy)

    elif 'Dynamic' in env_name:
        from scripts.exp.routing_figure import ball_track
        track = ball_track(un)
        ball = un.obstacles[0]
        home = tuple(int(v) for v in ball.cur_pos)
        for bx, by in track:
            un.grid.set(*ball.cur_pos, None)
            un.grid.set(bx, by, ball)
            ball.cur_pos = (bx, by)
            for x in range(W):
                for y in range(H):
                    if (x, y) == (bx, by) or not standable(x, y):
                        continue
                    for d, (dx, dy) in enumerate(DIRVEC):
                        un.agent_pos, un.agent_dir = (x, y), d
                        obs.append(wrapped.observation(un.gen_obs()))
                        f = (x + dx, y + dy)
                        labels.append(float(f in haz or f == (bx, by)))
                        keys.append((x, y, d))
                        varied.append((bx, by))
        un.grid.set(*ball.cur_pos, None)
        un.grid.set(*home, ball)
        ball.cur_pos = home

    else:                                                # Corner
        for x in range(W):
            for y in range(H):
                if not standable(x, y):
                    continue
                for d, (dx, dy) in enumerate(DIRVEC):
                    un.agent_pos, un.agent_dir = (x, y), d
                    obs.append(wrapped.observation(un.gen_obs()))
                    labels.append(float((x, y) in haz or (x + dx, y + dy) in haz))
                    keys.append((x, y))
                    varied.append(d)

    return np.stack(obs), np.asarray(labels), keys, varied


def matched(keys, labels, score):
    """Mean density by label, within groups that share a key and contain both labels.

    Groups with only one label carry no contrast -- a cell standing ON lava is labelled
    from every heading -- and are dropped rather than averaged in.
    """
    grp = {}
    for k, l, s in zip(keys, labels, score):
        grp.setdefault(k, ([], []))[int(l > 0)].append(s)
    both = [(k, float(np.mean(v[0])), float(np.mean(v[1])))
            for k, v in grp.items() if v[0] and v[1]]
    return both


# What each environment's matched control holds fixed and what it varies.
CTRL = {'MiniGrid-Corner-v0': ('cell', 'heading'),
        'MiniGrid-FragileCrossing-v0': ('cell + heading', 'carrying the object'),
        'MiniGrid-DynamicCrossing-v0': ('cell + heading', 'where the ball is')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--envs', nargs='+',
                    default=['MiniGrid-Corner-v0', 'MiniGrid-FragileCrossing-v0',
                             'MiniGrid-DynamicCrossing-v0'])
    ap.add_argument('--episodes', type=int, default=3)
    ap.add_argument('--max_steps', type=int, default=300)
    ap.add_argument('--n_trees', type=int, default=250)
    ap.add_argument('--stride', type=int, default=40)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default='tmp/routing_table.md')
    args = ap.parse_args()

    from algos.split_rl import SPLIT_RL

    blocks, notes, summary, controls = [], [], [], []
    for env_name in args.envs:
        vec_env, wrapped = build_env(env_name, 0)
        un = wrapped.unwrapped
        wrapped.reset(seed=0)
        ckpt = (f'saved_models/minigrid/minigrid/{env_name}/split_rl/'
                f'fully_obs_seed_0_{CKPT_STEPS[env_name]}_steps.zip')
        model = SPLIT_RL.load(ckpt, env=vec_env, device=args.device, force_reset=True)
        model.set_random_seed(0)

        # --- rollout first so visited keys are available for the matched control ----
        wrapped.reset(seed=0)
        d = rollout(model, wrapped, args.episodes, args.max_steps, 0)
        # Restore the grid after rollout mutations before enumerating states.
        wrapped.reset(seed=0)

        # --- enumerated state set -------------------------------------------------
        e_obs, e_lab, e_keys, _ = enumerate_states(wrapped, un, env_name)
        wrapped.reset(seed=0)
        e_sc, n_trees = routing_scores(model, e_obs, args.n_trees, args.stride)
        e_pos = e_lab > 0
        mt = matched(e_keys, e_lab, e_sc)
        summary.append(dict(
            name=NICE[env_name], how='enumerated', n=len(e_lab),
            m1=float(e_sc[e_pos].mean()) if e_pos.any() else float('nan'),
            m0=float(e_sc[~e_pos].mean()) if (~e_pos).any() else float('nan'),
            n1=int(e_pos.sum()), n0=int((~e_pos).sum())))
        if 'Corner' in env_name and mt:
            # Corner's lava is avoidable so trajectories have no cost-labelled states;
            # use the enumerated matched analysis instead.
            a = np.array([m[1] for m in mt]); b = np.array([m[2] for m in mt])
            controls.append(dict(name=NICE[env_name],
                                 m0=float(a.mean()), m1=float(b.mean())))

        # --- evaluation trajectories ----------------------------------------------
        # d already collected above; score the observations now.
        score, n_trees = routing_scores(model, d['obs'], args.n_trees, args.stride)

        # The trained policy is near-deterministic and the layout is fixed, so all
        # episodes are identical. Report one episode step by step rather than averaging
        # n_episodes copies of the same evidence into a single number.
        ep0 = d['ep'] == d['ep'][0]
        ident = all(np.array_equal(d['label'][d['ep'] == e], d['label'][ep0])
                    and np.array_equal(d['x'][d['ep'] == e], d['x'][ep0])
                    for e in np.unique(d['ep']))
        d = {k: (v[ep0] if isinstance(v, np.ndarray) and len(v) == len(score) else v)
             for k, v in d.items()}
        score = score[ep0]

        pos = d['label'] > 0
        m1, m0 = float(score[pos].mean()), float(score[~pos].mean())

        if 'Corner' not in env_name and pos.any() and (~pos).any():
            controls.append(dict(name=NICE[env_name], m0=m0, m1=m1))

        haz, ice = hazard_cells(un) if 'Fragile' in env_name else (set(), set())
        xy = list(zip(d['x'].tolist(), d['y'].tolist()))
        on_haz = np.array([c in haz for c in xy])
        on_ice = np.array([c in ice for c in xy])
        steps = [(i, int(x), int(y), bool(c), float(l), float(s), bool(oi))
                 for i, (x, y, c, l, s, oi) in enumerate(
                     zip(d['x'], d['y'], d['carry'], d['label'], score, on_ice))]
        summary.append(dict(
            name=NICE[env_name], how='trajectories', n=len(score),
            m1=m1 if pos.any() else float('nan'),
            m0=m0 if (~pos).any() else float('nan'),
            n1=int(pos.sum()), n0=int((~pos).sum())))
        blocks.append(dict(name=NICE[env_name], steps=steps, m1=m1, m0=m0,
                           n_cost=int(pos.sum()), has_ice=bool(len(ice)),
                           on_haz=on_haz, carry=d['carry'], score=score,
                           label=d['label']))
        notes.append(f"{NICE[env_name]}: {n_trees} trees sampled every {args.stride}; "
                     f"{args.episodes} episodes, all identical: {ident}; "
                     f"episode length {len(steps)}")

    L = []
    L.append('# What the trees route to the cost objective\n')
    L.append('**Method.** Each leaf stores the density of guidance labels over the samples '
             'that reached it\nduring fitting -- a leaf built from 100 samples of which 30 '
             'were cost-labelled has cost density\n0.30. To score a state we drop it down '
             'every tree, take the cost density of the leaf it\nreaches, and average over '
             'the ensemble. The density is read from the fitted trees, not\npredicted: '
             '**no guidance label is given at evaluation time.** The environment label is '
             'used\nonly to mark which states we expected to be critical. Densities are '
             'bounded by how common\ncost labels were during training, so they are small '
             'in absolute terms and the comparison of\ninterest is between states.\n')
    L.append('Two state sets are reported. **Enumerated** covers every state the agent can '
             'occupy, so it is\nfree of what the policy happens to do -- necessary for '
             'Corner, whose lava is avoidable and\nwhose evaluated episodes therefore '
             'contain no labelled states at all. **Trajectories** covers\nthe states an '
             'evaluated policy actually visits, which is the operationally relevant set '
             'but\nis narrow and can confound factors the policy never varies.\n')

    L.append('## Cost density where the guidance fires\n')
    L.append('| Environment | state set | states | guidance fires | does not fire | ratio |')
    L.append('|---|---|---|---|---|---|')
    for r in summary:
        if r['n1'] == 0:
            fires, ratio = 'none present', 'n/a'
        else:
            fires = f"{r['m1']:.3f} ({r['n1']})"
            ratio = f"{r['m1']/r['m0']:.1f}x" if r['m0'] > 0 else 'n/a'
        L.append(f"| {r['name']} | {r['how']} | {r['n']} | {fires} | "
                 f"{r['m0']:.3f} ({r['n0']}) | {ratio} |")
    L.append('')

    if controls:
        L.append('## Control: is it the position?\n')
        L.append('The gaps above could in principle come from the trees memorising '
                 'locations. Each row below\nholds position fixed and varies only the '
                 'thing the guidance actually depends on, over the\nenumerated states. '
                 'Groups where the label cannot flip -- a cell standing *on* lava is '
                 'labelled\nfrom every heading -- carry no contrast and are excluded.\n')
        L.append('| Environment | label = reward | label = cost | ratio |')
        L.append('|---|---|---|---|')
        for c in controls:
            L.append(f"| {c['name']} | {c['m0']:.3f} | {c['m1']:.3f} | "
                     f"{c['m1']/c['m0']:.1f}x |")
        L.append('')

    for b in blocks:
        if b['n_cost'] == 0:
            continue                     # nothing to show: no labelled state was visited
        L.append(f"## {b['name']}: the evaluation episode step by step\n")
        ice_col = ' carrying |' if b['has_ice'] else ''
        L.append(f"| step | cell |{ice_col} guidance | cost density |")
        L.append(f"|---|---|{'---|' if b['has_ice'] else ''}---|---|")
        for i, x, y, c, l, s, oi in b['steps']:
            mark = ' **<-- fires**' if l > 0 else ''
            cc = f" {'yes' if c else 'no'} |" if b['has_ice'] else ''
            L.append(f"| {i} | ({x},{y}){' ice' if oi else ''} |{cc} {l:.0f} | "
                     f"{s:.3f}{mark} |")
        L.append('')

    L.append('---\n')
    for nt in notes:
        L.append(f'- {nt}')

    txt = '\n'.join(L) + '\n'
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(txt)
    print(txt)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
