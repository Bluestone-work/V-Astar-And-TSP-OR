"""Continuous-time geometry shared by the interval audit and the closed loop.

Two primitives that the grid evaluator of Section III-C does not have:

* `moving_pair_min_distance` -- the minimum centre separation between two
  bodies that move *simultaneously* during one period.  `environment.step()`
  compares positions only at period boundaries (`environment.py:577,580,586`),
  so a crossing strictly inside the period leaves no trace in its verdict.
* `box_clearance` -- the clearance to an occupied cell treated as its exact
  rectangle, Eq. (box_clearance) of the new Section V-G.  Note that
  `global_planner.segment_clearance` subtracts the cell *half diagonal*
  instead, i.e. bounds each cell by its circumscribed circle; that is a
  different and strictly more conservative criterion, and conflating the two
  is what made the earlier closed-loop gap unreadable.

Coordinates are physical throughout: a (row, column) pair times `cell_size`.
Callers working in cell units can pass `cell_size=(1., 1.)` and read the
results in cells.
"""
from __future__ import annotations

import numpy as np


def moving_pair_min_distance(p0, p1, q0, q1):
    """Closest approach of two points moving linearly over one period.

    Returns `(distance, t)` with `t` in [0, 1] the time of closest approach.
    Exact: the squared separation is a quadratic in `t`.
    """
    a = np.asarray(p0, dtype=float) - np.asarray(q0, dtype=float)
    b = ((np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float))
         - (np.asarray(q1, dtype=float) - np.asarray(q0, dtype=float)))
    denominator = float(b @ b)
    if denominator == 0.:
        return float(np.linalg.norm(a)), 0.
    t = float(np.clip(-(a @ b) / denominator, 0., 1.))
    return float(np.linalg.norm(a + t * b)), t


def moving_point_polyline_min_distance(p0, p1, trace):
    """Closest approach when the second body follows a polyline in the period.

    `trace` is `_motion_trace` from `environment._update_dynamic_obstacle`: the
    cells an obstacle occupied during one step, start included.  An obstacle
    with `speed` s makes s sub-moves in one step, so each sub-move is given an
    equal 1/s share of the period -- the only reading under which a speed of 2
    means "twice as fast" rather than "teleports".
    """
    points = [np.asarray(x, dtype=float) for x in trace]
    start = np.asarray(p0, dtype=float)
    end = np.asarray(p1, dtype=float)
    if len(points) < 2:
        return moving_pair_min_distance(start, end, points[0], points[0])
    count = len(points) - 1
    best = (float('inf'), 0.)
    for index in range(count):
        first, last = index / count, (index + 1) / count
        # Restricting the robot to the sub-window turns each piece back into
        # the two-point problem above, parameterised on [0, 1].
        window = (start + first * (end - start), start + last * (end - start))
        distance, local = moving_pair_min_distance(*window, points[index], points[index + 1])
        if distance < best[0]:
            best = (distance, first + local * (last - first))
    return best


def box_clearance(point, centres, cell_size=(1., 1.)):
    """Eq. (box_clearance): clearance from a point to occupied cell rectangles.

    Zero at a cell boundary and zero at the shared corner of two diagonally
    adjacent occupied cells, which is why it rejects the corner cut that the
    grid evaluator permits.
    """
    centres = np.asarray(centres, dtype=float).reshape(-1, 2)
    if not len(centres):
        return float('inf')
    half = .5 * np.asarray(cell_size, dtype=float)
    excess = np.maximum(np.abs(np.asarray(point, dtype=float) - centres) - half, 0.)
    return float(np.linalg.norm(excess, axis=1).min())


def segment_box_clearance(p0, p1, centres, cell_size=(1., 1.), resolution=.01):
    """Clearance from a swept segment to occupied cell rectangles.

    `box_clearance` is 1-Lipschitz in the point, so sampling the segment at a
    physical spacing of at most `resolution` overestimates the true minimum by
    at most `resolution / 2`.  Returns `(sampled, lower_bound)` so a caller can
    quote either the estimate or a value it is guaranteed not to be below.
    """
    centres = np.asarray(centres, dtype=float).reshape(-1, 2)
    if not len(centres):
        return float('inf'), float('inf')
    start = np.asarray(p0, dtype=float)
    end = np.asarray(p1, dtype=float)
    length = float(np.linalg.norm(end - start))
    count = max(2, int(np.ceil(length / resolution)) + 1)
    samples = start + np.linspace(0., 1., count)[:, None] * (end - start)
    half = .5 * np.asarray(cell_size, dtype=float)
    excess = np.maximum(np.abs(samples[:, None, :] - centres[None, :, :]) - half, 0.)
    sampled = float(np.linalg.norm(excess, axis=2).min())
    return sampled, sampled - .5 * resolution


def moving_box_clearance(p0, p1, c0, c1, cell_size=(1., 1.), resolution=.01):
    """Clearance over one period between a moving point and a moving cell box.

    In the obstacle's frame the robot traces `p - c`, so this reduces to
    `segment_box_clearance` against a single box at the origin.  This is the
    test Section V-G specifies for dynamic obstacles and the one
    `experiments/kinematic_closed_loop.py` does not apply: that script holds the
    robot at its end-of-period pose and sweeps only the obstacle.
    """
    relative_start = np.asarray(p0, dtype=float) - np.asarray(c0, dtype=float)
    relative_end = np.asarray(p1, dtype=float) - np.asarray(c1, dtype=float)
    return segment_box_clearance(relative_start, relative_end, [[0., 0.]],
                                 cell_size=cell_size, resolution=resolution)


def cuts_corner(position, delta, occupancy):
    """Whether a diagonal grid move squeezes between two occupied cells.

    The rule `environment.py` keeps only inside `_motion_safe()` for route
    validation, restored here as a measurement.  Exact, and the clearance in
    that case is exactly zero for any positive robot radius.
    """
    row, column = int(position[0]), int(position[1])
    step_row, step_column = int(delta[0]), int(delta[1])
    if not (step_row and step_column):
        return False
    rows, columns = np.asarray(occupancy).shape
    if not (0 <= row + step_row < rows and 0 <= column + step_column < columns):
        return False
    return bool(occupancy[row + step_row, column] == 1
                or occupancy[row, column + step_column] == 1)


def moving_polyline_box_clearance(p0, p1, trace, cell_size=(1., 1.), resolution=.01):
    """`moving_box_clearance` for an obstacle that makes several sub-moves.

    Each sub-move takes an equal share of the period, as in
    `moving_point_polyline_min_distance`, and the robot is restricted to the
    matching sub-window so both bodies advance together.
    """
    points = [np.asarray(x, dtype=float) for x in trace]
    start = np.asarray(p0, dtype=float)
    end = np.asarray(p1, dtype=float)
    if len(points) < 2:
        return moving_box_clearance(start, end, points[0], points[0],
                                    cell_size=cell_size, resolution=resolution)
    count = len(points) - 1
    best = (float('inf'), float('inf'))
    for index in range(count):
        first, last = index / count, (index + 1) / count
        sampled, bound = moving_box_clearance(
            start + first * (end - start), start + last * (end - start),
            points[index], points[index + 1],
            cell_size=cell_size, resolution=resolution)
        if sampled < best[0]:
            best = (sampled, bound)
    return best


def nearby_occupied(occupancy, p0, p1, margin=2.):
    """Occupied cell centres that could bound the clearance of one segment.

    A step moves at most sqrt(2) cells, so screening by the segment's bounding
    box grown by `margin` keeps the clearance exact while cutting the candidate
    set from every wall cell in the map to a handful.
    """
    occupied = np.argwhere(np.asarray(occupancy) != 0)
    if not len(occupied):
        return occupied.astype(float)
    corners = np.asarray([p0, p1], dtype=float)
    low = corners.min(axis=0) - margin
    high = corners.max(axis=0) + margin
    inside = np.all((occupied >= low) & (occupied <= high), axis=1)
    return occupied[inside].astype(float)
