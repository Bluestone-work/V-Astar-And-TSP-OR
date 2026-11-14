"""Give the dynamic obstacles a physical motion model, not a lattice walk.

`environment.py` draws `speed = random.randint(1, 2)` cells per step over eight
lattice directions (`environment.py:313-327`). Under the Section V-G time base,
T = 1.42 s, that is 0.70 or 1.41 cells/s, and a speed-2 diagonal covers
2*sqrt(2) cells, i.e. 1.99 cells/s -- twice the robot's v_max of 1.0, against a
robot whose effective speed after the achieved fraction is 0.77 to 0.80. Turns
are instantaneous at cell centres, so lateral acceleration is unbounded. With a
swept body-aware collision test, a share of the resulting contacts is not
something any policy could have avoided, and attributing them to transfer
failure would overstate the cost of continuous execution.

This driver separates the obstacle's PATH from its TIMING:

* the path is exactly the lattice sequence the discrete run produces, pulled
  from `advance_obstacle_script`, so the scenario is the same one;
* the timing is physical -- constant speed along the path's arc length, so an
  axial and a diagonal segment take time in proportion to their length, which
  equal-time-per-sub-move got wrong;
* corners are rounded by a fillet of radius `v**2 / a_lateral`, which bounds
  lateral acceleration the way the robot's limits bound its own.

What this costs: obstacle positions at period boundaries no longer coincide with
the discrete run's, because the obstacle now lags a script that advances faster
than it does. The shared object is the path, not the per-period position. The
alternative -- keeping positions matched -- means keeping speeds that exceed the
robot's maximum, which is the defect being removed.
"""
from __future__ import annotations

import numpy as np

#: Lateral acceleration bound for the corner fillets, in cells/s^2. The robot's
#: own a_max in Section V-G, so the two bodies are bounded by the same number.
DEFAULT_LATERAL_ACCELERATION = 1.


def corner_points(before, corner, after, radius):
    """The rounded replacement for one interior corner of a lattice path.

    Returns the points that replace `corner`: the tangent entry, three interior
    samples of a quadratic Bezier through it, and the tangent exit. The Bezier
    approximates the tangent circular arc to within 3% of the radius for turns
    up to 90 degrees, the largest the eight-direction lattice produces. Returns
    `[corner]` when there is no turn to round.
    """
    before = np.asarray(before, dtype=float)
    corner = np.asarray(corner, dtype=float)
    after = np.asarray(after, dtype=float)
    first, second = corner - before, after - corner
    length_first, length_second = np.linalg.norm(first), np.linalg.norm(second)
    if radius <= 0 or length_first < 1e-12 or length_second < 1e-12:
        return [corner]
    unit_first, unit_second = first / length_first, second / length_second
    turn = float(np.arccos(float(np.clip(unit_first @ unit_second, -1., 1.))))
    if turn < 1e-9:
        return [corner]
    cut = min(radius * np.tan(turn / 2.), .5 * length_first, .5 * length_second)
    start, end = corner - cut * unit_first, corner + cut * unit_second
    interior = [((1 - t) ** 2 * start + 2 * (1 - t) * t * corner + t ** 2 * end)
                for t in (.25, .5, .75)]
    return [start, *interior, end]


def fillet(points, radius):
    """Round every interior corner of a polyline. Used by the tests."""
    points = [np.asarray(p, dtype=float) for p in points]
    if radius <= 0 or len(points) < 3:
        return points
    out = [points[0]]
    for before, corner, after in zip(points, points[1:], points[2:]):
        out.extend(corner_points(before, corner, after, radius))
    out.append(points[-1])
    return out


class ObstacleDriver:
    """Walk every obstacle along its shared lattice path at a physical speed.

    Each obstacle keeps its rounded path as a point array with the cumulative
    arc length alongside it, so a whole period's sample positions come from one
    vectorised lookup per obstacle instead of one Python step per substep. The
    first version walked segment by segment -- 30 obstacles x 71 substeps =
    2130 calls per period -- and could not finish a single Level 2 episode.
    """

    #: Consumed arc length kept behind the cursor before the path is trimmed, so
    #: neither the lookup nor the bookkeeping grows with episode length.
    TRIM_ARC = 8.

    def __init__(self, env, advance_script, speed, lateral_acceleration=None, scale=(1., 1.)):
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError('obstacle speed must be positive and finite')
        self.env = env
        self.advance_script = advance_script
        self.speed = float(speed)
        self.scale = np.asarray(scale, dtype=float)
        acceleration = (DEFAULT_LATERAL_ACCELERATION if lateral_acceleration is None
                        else float(lateral_acceleration))
        self.radius = 0. if acceleration <= 0 else self.speed ** 2 / acceleration
        self.script_index = 0
        count = len(env.dynamic_obstacles)
        #: Lattice points as the discrete script emits them, append-only.
        self.raw = [[np.asarray(o['position'], dtype=float)] for o in env.dynamic_obstacles]
        #: Rounded path points, with the still-unrounded raw head last.
        self.points = [np.asarray([r[0], r[0]], dtype=float) for r in self.raw]
        self.lengths = [np.zeros(2) for _ in self.raw]     # cumulative arc length
        self.filleted = [0] * count                        # corners already rounded
        self.arc = [0.] * count                            # distance travelled on the path
        #: An obstacle whose every neighbour is occupied gets a one-point trace,
        #: so its path never grows. Gating the pre-fetch on all obstacles then
        #: never terminated -- the first period of a 30-obstacle level hung.
        #: Such an obstacle stands still, which `_sample`'s clamp already does.
        self.blocked = [False] * count
        self.position = [r[0].copy() for r in self.raw]
        self.lattice_heads = [tuple(o['position']) for o in env.dynamic_obstacles]

    @staticmethod
    def _cumulative(points):
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        return np.concatenate([[0.], np.cumsum(steps)])

    def _extend(self):
        """Pull one more discrete step of path for every obstacle.

        Only corners the new points make determinable are rounded, and only
        once: the corner at `raw[i]` needs `raw[i+1]`.
        """
        traces = self.advance_script(self.env, self.script_index)
        self.script_index += 1
        for index, trace in enumerate(traces):
            addition = [np.asarray(q, dtype=float) for q in trace[1:]]
            self.blocked[index] = not addition
            if not addition:
                continue
            self.raw[index].extend(addition)
            raw = self.raw[index]
            rounded = list(self.points[index][:-1])        # drop the head sentinel
            while self.filleted[index] + 2 < len(raw):
                corner = self.filleted[index] + 1
                rounded.extend(corner_points(raw[corner - 1], raw[corner],
                                             raw[corner + 1], self.radius))
                self.filleted[index] = corner
            rounded.append(raw[-1])
            self.points[index] = np.asarray(rounded, dtype=float)
            self.lengths[index] = self._cumulative(self.points[index])
            self.lattice_heads[index] = tuple(int(round(v)) for v in raw[-1])
            self._trim(index)

    def _trim(self, index):
        """Drop path far behind the current arc position."""
        if self.arc[index] <= self.TRIM_ARC:
            return
        lengths = self.lengths[index]
        keep = int(np.searchsorted(lengths, self.arc[index] - self.TRIM_ARC, 'right')) - 1
        if keep <= 0:
            return
        self.points[index] = self.points[index][keep:]
        self.arc[index] -= lengths[keep]
        self.lengths[index] = self.lengths[index][keep:] - lengths[keep]
        self.filleted[index] = max(0, self.filleted[index])

    def _remaining(self, index):
        return float(self.lengths[index][-1] - self.arc[index])

    def _sample(self, index, arcs):
        """Positions at the given arc lengths, clamped to the path end."""
        lengths, points = self.lengths[index], self.points[index]
        clamped = np.clip(arcs, 0., lengths[-1])
        upper = np.clip(np.searchsorted(lengths, clamped, 'right'), 1, len(lengths) - 1)
        lower = upper - 1
        span = lengths[upper] - lengths[lower]
        fraction = np.where(span > 1e-12, (clamped - lengths[lower]) / np.maximum(span, 1e-12), 0.)
        return points[lower] + fraction[:, None] * (points[upper] - points[lower])

    def sample_period(self, substeps, dt):
        """Advance one decision period; return positions at every substep edge.

        Shape is (substeps + 1, obstacles, 2), in physical units, so the caller
        can sweep robot and obstacles against each other over the same windows.
        """
        if not self.raw:
            # Level 1 has no dynamic obstacles; keep the shape contract so the
            # caller's vectorised clearance test needs no special case.
            return np.zeros((substeps + 1, 0, 2))
        step = self.speed * dt
        needed = step * substeps + 2.
        while True:
            short = [i for i in range(len(self.raw))
                     if not self.blocked[i] and self._remaining(i) < needed]
            if not short:
                break
            self._extend()
        offsets = np.arange(substeps + 1) * step
        frames = np.empty((substeps + 1, len(self.raw), 2))
        for index in range(len(self.raw)):
            frames[:, index, :] = self._sample(index, self.arc[index] + offsets)
            self.arc[index] += step * substeps
            self.position[index] = frames[-1, index, :].copy()
        return frames * self.scale

    def observed_cells(self):
        """Rounded positions, which is what the robot's occupancy window sees."""
        return [tuple(int(round(v)) for v in p) for p in self.position]
