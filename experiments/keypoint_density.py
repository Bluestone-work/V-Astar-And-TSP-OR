"""Sub-goal density as a controlled factor for Algorithm 2.

Why this lives outside ``global_planner``
-----------------------------------------
``global_planner.py`` is one of the hashed training sources recorded in every
checkpoint (see ``train.py:source_names``).  Editing it invalidates every
existing checkpoint by design.  The density sweep is a post-training analysis
intervention, so it is implemented here and reuses the unmodified planner
primitives instead.

With ``max_segment=None`` the behaviour is delegated verbatim to
``global_planner.planned_grid_route``, so the released longest-safe-segment
compression stays the reference arm.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from global_planner import (grid_astar_path, planned_grid_route,
                            segment_is_safe, voronoi_astar_path)


def capped_key_points(
    path: Iterable[Sequence[float]],
    occupancy: np.ndarray,
    *,
    max_segment: float,
    robot_radius: float = 0.0,
    cell_size: tuple[float, float] = (1.0, 1.0),
    extra_margin: float = 0.0,
    report: dict | None = None,
) -> list[tuple[float, float]]:
    """``extract_key_points`` with the anchor-to-waypoint span capped.

    The span cap and the safety predicate both govern how far a key point may
    skip, never whether the scan advances, so the adjacent node is always
    taken and forward progress is guaranteed.  An unsafe adjacent edge is an
    edge of the input path; it is counted in ``report['inherited_unsafe_edges']``
    and the sequence is still returned complete.
    """
    if not (np.isfinite(max_segment) and max_segment > 0):
        raise ValueError("max_segment must be a positive finite number")
    geometry = dict(robot_radius=robot_radius, cell_size=cell_size,
                    extra_margin=extra_margin)
    points = [tuple(map(float, p)) for p in path]
    status = report if report is not None else {}
    status.update(scans=0, forced_advances=0, inherited_unsafe_edges=0)
    if not points:
        return []
    if len(points) == 1:
        return points if segment_is_safe(occupancy, points[0], points[0], **geometry) else []

    key_points = [points[0]]
    anchor = 0
    last = len(points) - 1
    while anchor < last:
        status['scans'] += 1
        last_safe = anchor
        for candidate in range(anchor + 1, last + 1):
            span = float(np.hypot(points[candidate][0] - points[anchor][0],
                                  points[candidate][1] - points[anchor][1]))
            if candidate > anchor + 1 and span > max_segment:
                break
            if segment_is_safe(occupancy, points[anchor], points[candidate], **geometry):
                last_safe = candidate
                continue
            break
        if last_safe == anchor:
            # Adjacent edge failed: advance over it rather than stall.
            status['forced_advances'] += 1
            status['inherited_unsafe_edges'] += 1
            last_safe = anchor + 1
        anchor = last_safe
        key_points.append(points[anchor])

    result: list[tuple[float, float]] = []
    for point in key_points:
        if not result or point != result[-1]:
            result.append(point)
    return result


def density_route(grid, start, goal, *, compress=True, max_segment=None, **geometry):
    """``planned_grid_route`` with an optional sub-goal density cap."""
    if max_segment is None:
        return planned_grid_route(grid, start, goal, compress=compress, **geometry)

    start, goal = tuple(start), tuple(goal)
    if any(not (0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1])
           or grid[p] != 0 for p in (start, goal)):
        return [], {'planner': 'invalid_endpoint', 'fallback': False}
    raw, meta = voronoi_astar_path(grid, start, goal, **geometry)
    route = []
    for point in raw:
        cell = tuple(np.rint(point).astype(int))
        if not route or route[-1] != cell:
            route.append(cell)

    def valid(points):
        return bool(points) and all(segment_is_safe(grid, p, p, **geometry) for p in points) and all(
            segment_is_safe(grid, a, b, **geometry) for a, b in zip(points, points[1:]))

    if not valid(route):
        route = grid_astar_path(grid, start, goal, **geometry)
        meta = {'planner': 'grid_astar_fallback', 'fallback': True,
                'reason': 'no_valid_integer_voronoi_route'}
    if not valid(route):
        return [], dict(meta, route_valid=False)
    raw_count = len(route)
    if compress:
        route = capped_key_points(route, grid, max_segment=max_segment, **geometry)
    if not valid(route):
        return [], dict(meta, route_valid=False)
    return [tuple(map(int, p)) for p in route], dict(
        meta, route_valid=True, raw_points=raw_count, key_points=len(route),
        keypoint_max_segment=float(max_segment))
