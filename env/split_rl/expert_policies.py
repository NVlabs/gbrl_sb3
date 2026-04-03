"""Deterministic BFS-based expert policies for MiniGrid subtask environments.

Each expert operates on the full grid state to compute optimal actions via
BFS pathfinding, phase by phase.  Observations recorded during trajectory
collection are the standard partial (7x7) agent view.

Usage::

    from env.split_rl.multi_room_corridor import MoveBallEnv
    from env.split_rl.expert_policies import MoveBallExpert, collect_trajectories

    env = MoveBallEnv()
    expert = MoveBallExpert(env)
    trajectories = collect_trajectories(env, expert, n_episodes=100)
"""
from __future__ import annotations

from collections import deque
from typing import Optional

from minigrid.core.world_object import Ball, Box, Door, Goal, Key

# Direction vectors: 0=right, 1=down, 2=left, 3=up
_DIR = [(1, 0), (0, 1), (-1, 0), (0, -1)]

# MiniGrid action indices
LEFT, RIGHT, FORWARD, PICKUP, DROP, TOGGLE = 0, 1, 2, 3, 4, 5


class SubtaskExpert:
    """Base class: BFS on the full grid, lazy phase-based planning."""

    def __init__(self, env):
        self.env = env
        self._q: deque[int] = deque()
        self._phase = 0

    def reset(self):
        self._q.clear()
        self._phase = 0

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _fwd(x, y, d):
        return x + _DIR[d][0], y + _DIR[d][1]

    def _passable(self, x, y):
        g = self.env.grid
        if not (0 <= x < g.width and 0 <= y < g.height):
            return False
        c = g.get(x, y)
        return c is None or c.can_overlap()

    def _find(self, cls, color=None):
        g = self.env.grid
        for y in range(g.height):
            for x in range(g.width):
                c = g.get(x, y)
                if isinstance(c, cls) and (color is None or c.color == color):
                    return (x, y)
        return None

    # ---------------------------------------------------------------- BFS

    def _bfs(self, accept, extra_pass=frozenset()):
        """BFS in (x, y, dir) space.  *accept(x, y, d)* → bool."""
        e = self.env
        s0 = (int(e.agent_pos[0]), int(e.agent_pos[1]), int(e.agent_dir))
        vis = {s0}
        q = deque([(s0, [])])
        while q:
            (x, y, d), acts = q.popleft()
            if accept(x, y, d):
                return acts
            for ns, a in [((x, y, (d - 1) % 4), LEFT),
                          ((x, y, (d + 1) % 4), RIGHT)]:
                if ns not in vis:
                    vis.add(ns)
                    q.append((ns, acts + [a]))
            fx, fy = self._fwd(x, y, d)
            if self._passable(fx, fy) or (fx, fy) in extra_pass:
                ns = (fx, fy, d)
                if ns not in vis:
                    vis.add(ns)
                    q.append((ns, acts + [FORWARD]))
        return None

    # -------------------------------------------------------- nav primitives

    def _face(self, pos):
        """Actions to stand adjacent to *pos* and face it."""
        tx, ty = int(pos[0]), int(pos[1])
        return self._bfs(lambda x, y, d: self._fwd(x, y, d) == (tx, ty))

    def _goto(self, pos):
        """Actions to stand ON *pos*."""
        tx, ty = int(pos[0]), int(pos[1])
        return self._bfs(lambda x, y, d: (x, y) == (tx, ty),
                         extra_pass={(tx, ty)})

    def _drop_anywhere(self):
        """Actions + DROP into any empty adjacent cell."""
        g = self.env.grid

        def ok(x, y, d):
            fx, fy = self._fwd(x, y, d)
            return (0 <= fx < g.width and 0 <= fy < g.height
                    and g.get(fx, fy) is None)

        acts = self._bfs(ok)
        return (acts + [DROP]) if acts is not None else None

    def _drop_outside(self, blocked):
        """Actions + DROP into an empty cell not in *blocked*."""
        g = self.env.grid

        def ok(x, y, d):
            fx, fy = self._fwd(x, y, d)
            return (0 <= fx < g.width and 0 <= fy < g.height
                    and g.get(fx, fy) is None
                    and (fx, fy) not in blocked)

        acts = self._bfs(ok)
        return (acts + [DROP]) if acts is not None else None

    # ---------------------------------------------------------------- public

    def get_action(self) -> Optional[int]:
        """Return next expert action (or None if stuck / done)."""
        if not self._q:
            self._plan()
        return self._q.popleft() if self._q else None

    def _plan(self):
        raise NotImplementedError


# =====================================================================
# Concrete experts
# =====================================================================

class MoveBallExpert(SubtaskExpert):
    """MoveBallEnv: pick up ball → drop clear of doorway → reach goal."""

    def _plan(self):
        if self._phase == 0:                            # pick up ball
            p = self._find(Ball)
            if p:
                a = self._face(p)
                if a is not None:
                    self._q.extend(a + [PICKUP])
            self._phase = 1

        elif self._phase == 1:                          # drop away from door
            a = self._drop_outside(self.env.obstructing_cells)
            if a is not None:
                self._q.extend(a)
            self._phase = 2

        elif self._phase == 2:                          # walk to goal
            p = self._find(Goal)
            if p:
                a = self._goto(p)
                if a is not None:
                    self._q.extend(a)
            self._phase = 3


class KeyDoorExpert(SubtaskExpert):
    """KeyDoorEnv: pick up key → unlock door → drop key → reach goal."""

    def _plan(self):
        if self._phase == 0:                            # pick up key
            p = self._find(Key)
            if p:
                a = self._face(p)
                if a is not None:
                    self._q.extend(a + [PICKUP])
            self._phase = 1

        elif self._phase == 1:                          # unlock door
            p = self._find(Door)
            if p:
                a = self._face(p)
                if a is not None:
                    self._q.extend(a + [TOGGLE])
            self._phase = 2

        elif self._phase == 2:                          # drop key
            a = self._drop_anywhere()
            if a is not None:
                self._q.extend(a)
            self._phase = 3

        elif self._phase == 3:                          # walk to goal
            p = self._find(Goal)
            if p:
                a = self._goto(p)
                if a is not None:
                    self._q.extend(a)
            self._phase = 4


class BoxKeyExpert(SubtaskExpert):
    """BoxKeyEnv: open box → pick up key → unlock door → reach goal."""

    def _plan(self):
        if self._phase == 0:                            # open box
            p = self._find(Box)
            if p:
                a = self._face(p)
                if a is not None:
                    self._q.extend(a + [TOGGLE])
            self._phase = 1

        elif self._phase == 1:                          # pick up key
            p = self._find(Key)
            if p:
                a = self._face(p)
                if a is not None:
                    self._q.extend(a + [PICKUP])
            self._phase = 2

        elif self._phase == 2:                          # unlock door
            p = self._find(Door)
            if p:
                a = self._face(p)
                if a is not None:
                    self._q.extend(a + [TOGGLE])
            self._phase = 3

        elif self._phase == 3:                          # drop key
            a = self._drop_anywhere()
            if a is not None:
                self._q.extend(a)
            self._phase = 4

        elif self._phase == 4:                          # walk to goal
            p = self._find(Goal)
            if p:
                a = self._goto(p)
                if a is not None:
                    self._q.extend(a)
            self._phase = 5


# =====================================================================
# Trajectory collection
# =====================================================================

def collect_trajectory(env, expert, seed=None):
    """Run one episode.  Returns (obs_list, act_list, rew_list, solved)."""
    obs, _ = env.reset(seed=seed)
    expert.reset()
    observations, actions, rewards = [obs], [], []
    terminated = truncated = False
    while not (terminated or truncated):
        action = expert.get_action()
        if action is None:
            break
        obs, reward, terminated, truncated, _ = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
    return observations, actions, rewards, terminated


def collect_trajectories(env, expert, n_episodes=100, start_seed=0):
    """Collect *n_episodes* expert trajectories.  Returns list of tuples."""
    return [collect_trajectory(env, expert, seed=start_seed + i)
            for i in range(n_episodes)]


# =====================================================================
# Quick validation
# =====================================================================

if __name__ == "__main__":
    from env.split_rl.multi_room_corridor import MoveBallEnv, KeyDoorEnv, BoxKeyEnv

    configs = [
        ("MoveBall", MoveBallEnv, MoveBallExpert),
        ("KeyDoor", KeyDoorEnv, KeyDoorExpert),
        ("BoxKey",  BoxKeyEnv,  BoxKeyExpert),
    ]
    N = 200
    for name, EnvCls, ExpertCls in configs:
        env = EnvCls()
        expert = ExpertCls(env)
        solved, total_steps = 0, 0
        for seed in range(N):
            obs, _ = env.reset(seed=seed)
            expert.reset()
            done, steps = False, 0
            while not done:
                a = expert.get_action()
                if a is None:
                    break
                _, _, term, trunc, _ = env.step(a)
                steps += 1
                done = term or trunc
            if term and not trunc:
                solved += 1
                total_steps += steps
        avg = total_steps / solved if solved else 0
        print(f"{name}: {solved}/{N} solved  (avg {avg:.1f} steps)")
        env.close()
