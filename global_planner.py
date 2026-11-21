"""Geometry and waypoint utilities used by the revised global planner.

The original notebook used the distance from an obstacle-cell centre to an
infinite line.  This module uses the finite navigation segment and subtracts
the half diagonal of a rectangular grid cell.  The latter is conservative for
cell occupancy and makes the robot footprint an explicit parameter.
"""

from __future__ import annotations

from heapq import heappop, heappush
from math import ceil, hypot
from typing import Iterable, Sequence

import numpy as np


def point_to_segment_distance(
    point: Sequence[float], start: Sequence[float], end: Sequence[float]
) -> float:
    """Return the Euclidean distance from ``point`` to the finite segment."""

    p = np.asarray(point, dtype=float)
    s = np.asarray(start, dtype=float)
    e = np.asarray(end, dtype=float)
    d = e - s
    denom = float(np.dot(d, d))
    if denom == 0.0:
        return float(np.linalg.norm(p - s))
    t = float(np.dot(p - s, d) / denom)
    t = min(1.0, max(0.0, t))
    return float(np.linalg.norm(p - (s + t * d)))


def segment_clearance(
    occupancy: np.ndarray,
    start: Sequence[float],
    end: Sequence[float],
    *,
    cell_size: tuple[float, float] = (1.0, 1.0),
) -> float:
    """Return conservative clearance from a segment to occupied cells.

    Coordinates are expressed in the same row/column frame as the occupancy
    map.  Each occupied cell is represented by its centre and its rectangular
    half diagonal is subtracted from the centre-to-segment distance.  A value
    below zero means that the segment intersects the conservative cell box.
    The result is finite-segment based, unlike the previous infinite-line
    approximation.
    """

    grid = np.asarray(occupancy)
    if grid.ndim != 2:
        raise ValueError("occupancy must be a two-dimensional array")
    h, w = map(float, cell_size)
    if not np.isfinite([h, w]).all() or min(h, w) <= 0:
        raise ValueError("cell dimensions must be finite and positive")
    occupied = np.argwhere(grid != 0)
    if len(occupied) == 0:
        return float("inf")
    half_diagonal = 0.5 * hypot(h, w)
    # Grid coordinates use the environment's row/column cell-centre
    # convention (integer indices).  The rectangular half diagonal accounts
    # for the cell extent around each centre.
    scale = np.array([h, w])
    centres = occupied.astype(float) * scale
    s, e = np.asarray(start, dtype=float) * scale, np.asarray(end, dtype=float) * scale
    direction = e - s
    length_squared = float(direction @ direction)
    if length_squared == 0:
        distances = np.linalg.norm(centres - s, axis=1)
    else:
        projections = np.clip(np.sum((centres - s) * direction, axis=1) / length_squared, 0, 1)
        distances = np.linalg.norm(centres - (s + projections[:, None] * direction), axis=1)
    return float(distances.min() - half_diagonal)


def segment_is_safe(
    occupancy: np.ndarray,
    start: Sequence[float],
    end: Sequence[float],
    *,
    robot_radius: float = 0.0,
    cell_size: tuple[float, float] = (1.0, 1.0),
    extra_margin: float = 0.0,
) -> bool:
    """Check a segment against cell geometry, footprint and uncertainty margin."""

    if not np.isfinite([robot_radius, extra_margin]).all() or robot_radius < 0 or extra_margin < 0:
        raise ValueError("robot_radius and extra_margin must be non-negative")
    grid = np.asarray(occupancy)
    scale = np.asarray(cell_size, dtype=float)
    if grid.ndim != 2 or scale.shape != (2,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('invalid map or cell dimensions')
    points = np.asarray([start, end], dtype=float)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError('start and end must be finite two-dimensional points')
    # Cell centres are integer grid coordinates; the map extent is [-.5,n-.5].
    physical = points * scale
    margin = robot_radius + extra_margin
    if (np.any(physical < -.5 * scale + margin) or
            np.any(physical > (np.asarray(grid.shape)-.5)*scale - margin)):
        return False
    clearance = segment_clearance(occupancy, start, end, cell_size=cell_size)
    return clearance > robot_radius + extra_margin


def extract_key_points(
    path: Iterable[Sequence[float]],
    occupancy: np.ndarray,
    *,
    robot_radius: float = 0.0,
    cell_size: tuple[float, float] = (1.0, 1.0),
    extra_margin: float = 0.0,
    report: dict | None = None,
) -> list[tuple[float, float]]:
    """Compress a path while guaranteeing forward progress unconditionally.

    The predicate decides how many nodes a key point may skip, never whether
    the scan advances.  Skipping nothing is not a compression, so the adjacent
    node is taken whatever the predicate says about it: the reference index
    grows by at least one per scan and the procedure terminates after at most
    ``len(path) - 1`` scans.

    Every emitted segment that skips at least one node has passed the
    predicate.  A segment that skips nothing is an edge of ``path`` and
    inherits that path's validity; ``voronoi_astar_path`` admits an edge only
    under this same predicate, so for a path from that planner no such edge is
    unsafe.  If one is, because the path came from elsewhere or the map changed
    after planning, it is counted in ``report['inherited_unsafe_edges']`` for
    the caller to act on and the sequence is still returned complete.
    """

    points = [tuple(map(float, p)) for p in path]
    status = report if report is not None else {}
    status.update(scans=0, forced_advances=0, inherited_unsafe_edges=0)
    if not points:
        return []
    geometry = dict(robot_radius=robot_radius, cell_size=cell_size, extra_margin=extra_margin)
    if len(points) == 1:
        return points if segment_is_safe(occupancy, points[0], points[0], **geometry) else []

    key_points = [points[0]]
    anchor = 0
    last = len(points) - 1
    while anchor < last:
        status['scans'] += 1
        last_safe = anchor
        for candidate in range(anchor + 1, last + 1):
            if segment_is_safe(occupancy, points[anchor], points[candidate], **geometry):
                last_safe = candidate
                continue
            break

        if last_safe == anchor:
            # The adjacent edge itself failed.  Advance over it rather than
            # stall: it is an edge of the input path, so keeping it emits the
            # geometry the input already prescribes, and refusing to move
            # would retest the same pair from the same anchor forever.
            status['forced_advances'] += 1
            status['inherited_unsafe_edges'] += 1
            last_safe = anchor + 1
        anchor = last_safe
        key_points.append(points[anchor])

    # De-duplicate only adjacent repeated points; preserve the terminal point.
    result: list[tuple[float, float]] = []
    for point in key_points:
        if not result or point != result[-1]:
            result.append(point)
    return result


def action_step_distance(start: Sequence[float], goal: Sequence[float], *,
                         model: str = 'grid') -> int:
    """Distance term of Eq. (6), in actions, for the motion model in use.

    ``grid`` is the eight-connected integer grid of the paper: an axial action
    moves 1 and a diagonal action moves sqrt(2).  The obstacle-free minimum
    action count is then the Chebyshev distance, which never exceeds the
    Euclidean distance, so ``ceil`` of the Euclidean distance is a valid budget.

    ``unit`` is the variant in which every action covers distance 1, so a
    diagonal action moves 1/sqrt(2) per axis.  The eight directions are
    unchanged, so an arbitrary direction has to be approximated by mixing two
    of them, and reaching an offset with X = max|.| and Y = min|.| takes
    X + (sqrt(2)-1) Y actions, which is exact rather than conservative.  Using
    the Euclidean distance here would underestimate: the worst direction, 22.5
    degrees off axis, needs d / cos(22.5 deg) = 1.0824 d actions.
    """

    s = np.asarray(start, dtype=float)
    g = np.asarray(goal, dtype=float)
    if s.shape != (2,) or g.shape != (2,):
        raise ValueError("start and goal must be two-dimensional points")
    if model == 'grid':
        return int(ceil(float(np.linalg.norm(g - s))))
    if model == 'unit':
        offset = np.abs(g - s)
        major, minor = float(offset.max()), float(offset.min())
        return int(ceil(major + (2.0 ** 0.5 - 1.0) * minor))
    raise ValueError("model must be 'grid' or 'unit'")


def arrival_budget(
    start: Sequence[float],
    goal: Sequence[float],
    *,
    base_slack: int = 5,
    uncertainty_steps: int = 0,
    model: str = 'grid',
) -> int:
    """Compute a non-premature budget for the nine-action interface."""

    if base_slack < 0 or uncertainty_steps < 0:
        raise ValueError("budget margins must be non-negative")
    return (action_step_distance(start, goal, model=model)
            + int(base_slack) + int(uncertainty_steps))


def grid_astar_path(occupancy: np.ndarray, start: Sequence[float], goal: Sequence[float], *,
                    robot_radius=0.0, cell_size=(1.0, 1.0), extra_margin=0.0):
    """Return an eight-connected collision-free grid path as integer cells."""

    grid = np.asarray(occupancy)
    if grid.ndim != 2:
        raise ValueError("occupancy must be a two-dimensional array")
    s = tuple(int(round(x)) for x in start)
    g = tuple(int(round(x)) for x in goal)
    rows, cols = grid.shape
    if not (0 <= s[0] < rows and 0 <= s[1] < cols and
            0 <= g[0] < rows and 0 <= g[1] < cols):
        raise ValueError("start and goal must be inside the grid")
    if grid[s] != 0 or grid[g] != 0:
        raise ValueError("start and goal must be free cells")

    scale = np.asarray(cell_size, dtype=float)
    geometry = dict(robot_radius=robot_radius, cell_size=cell_size, extra_margin=extra_margin)
    if not segment_is_safe(grid, s, s, **geometry) or not segment_is_safe(grid, g, g, **geometry):
        return []
    directions = [(dr, dc, float(np.linalg.norm(np.asarray([dr, dc])*scale)))
                  for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
    open_set = [(0.0, s)]
    came_from = {}
    g_score = {s: 0.0}
    while open_set:
        _, current = heappop(open_set)
        if current == g:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        for dr, dc, cost in directions:
            nxt = (current[0] + dr, current[1] + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols) or grid[nxt] != 0:
                continue
            if dr and dc and (grid[current[0] + dr, current[1]] != 0 or
                              grid[current[0], current[1] + dc] != 0):
                continue
            if (robot_radius or extra_margin or tuple(cell_size) != (1., 1.)) and not segment_is_safe(grid, current, nxt, **geometry):
                continue
            tentative = g_score[current] + cost
            if tentative < g_score.get(nxt, float("inf")):
                came_from[nxt] = current
                g_score[nxt] = tentative
                heuristic = float(np.linalg.norm((np.asarray(g) - nxt)*scale))
                heappush(open_set, (tentative + heuristic, nxt))
    return []


def _voronoi_sites(occupancy: np.ndarray, max_sites: int | None = None) -> np.ndarray:
    occupied = np.argwhere(np.asarray(occupancy) != 0).astype(float)
    if max_sites is not None and len(occupied) > max_sites:
        indices = np.linspace(0, len(occupied) - 1, max_sites, dtype=int)
        occupied = occupied[np.unique(indices)]
    rows, cols = np.asarray(occupancy).shape
    # Exterior sites bound the finite portion used by the planner and prevent
    # unbounded Voronoi ridges from being silently treated as paths.
    border = np.array([
        [-2.0, -2.0], [-2.0, cols + 1.0],
        [rows + 1.0, -2.0], [rows + 1.0, cols + 1.0],
    ])
    return np.vstack([occupied, border])


def voronoi_astar_path(
    occupancy: np.ndarray,
    start: Sequence[float],
    goal: Sequence[float],
    *,
    robot_radius: float = 0.0,
    cell_size: tuple[float, float] = (1.0, 1.0),
    extra_margin: float = 0.0,
    max_sites: int | None = None,
    connect_k: int = 8,
):
    """Plan on a clearance-filtered finite Voronoi graph.

    The return value is ``(path, metadata)``.  A grid-A* path is used only as a
    deterministic fallback when the finite Voronoi graph is disconnected; the
    metadata records that fallback so it cannot be confused with a Voronoi
    result in the paper's ablation tables.
    """

    grid = np.asarray(occupancy)
    geometry = dict(robot_radius=robot_radius, cell_size=cell_size, extra_margin=extra_margin)
    if not segment_is_safe(grid, start, start, **geometry) or not segment_is_safe(grid, goal, goal, **geometry):
        return [], {'planner': 'invalid_endpoint', 'fallback': False}
    if tuple(start) == tuple(goal):
        return [tuple(start)], {'planner': 'voronoi_astar', 'fallback': False}
    def fallback_result():
        fallback = grid_astar_path(grid, start, goal, robot_radius=robot_radius,
                                  cell_size=cell_size, extra_margin=extra_margin)
        safe = all(segment_is_safe(grid, a, b, robot_radius=robot_radius,
                                  cell_size=cell_size, extra_margin=extra_margin)
                   for a, b in zip(fallback, fallback[1:]))
        if not safe:
            fallback = []
        return fallback, {"planner": "grid_astar_fallback", "fallback": True,
                          "fallback_clearance_valid": bool(safe and fallback)}

    try:
        from scipy.spatial import QhullError, Voronoi
        scale = np.asarray(cell_size, dtype=float)
        vor = Voronoi(_voronoi_sites(grid, max_sites=max_sites)*scale, qhull_options="Qbb Qc Qz")
    except (QhullError, ValueError):
        return fallback_result()

    rows, cols = grid.shape
    vertices = []
    vertex_map = {}
    for original, physical_vertex in enumerate(vor.vertices):
        vertex = physical_vertex / scale
        if not (-1.0 <= vertex[0] <= rows and -1.0 <= vertex[1] <= cols):
            continue
        cell = tuple(int(np.floor(x + .5)) for x in vertex)
        if not (0 <= cell[0] < rows and 0 <= cell[1] < cols) or grid[cell] != 0:
            continue
        if not segment_is_safe(grid, vertex, vertex, robot_radius=robot_radius,
                               cell_size=cell_size, extra_margin=extra_margin):
            continue
        vertex_map[original] = len(vertices)
        vertices.append(np.asarray(vertex, dtype=float))
    if len(vertices) < 2:
        return fallback_result()

    graph = {i: [] for i in range(len(vertices))}
    for ridge in vor.ridge_vertices:
        if len(ridge) != 2 or -1 in ridge or ridge[0] not in vertex_map or ridge[1] not in vertex_map:
            continue
        a, b = vertex_map[ridge[0]], vertex_map[ridge[1]]
        if a == b or not segment_is_safe(grid, vertices[a], vertices[b],
                                         robot_radius=robot_radius,
                                         cell_size=cell_size,
                                         extra_margin=extra_margin):
            continue
        weight = float(np.linalg.norm((vertices[a] - vertices[b])*scale))
        graph[a].append((b, weight))
        graph[b].append((a, weight))

    # Eq. (3): merge only collinear degree-two nodes, preserving junctions.
    removed = 0
    for k in list(graph):
        if len(graph[k]) != 2:
            continue
        (a, wa), (b, wb) = graph[k]
        u, v = (vertices[a]-vertices[k])*scale, (vertices[b]-vertices[k])*scale
        cross = u[0]*v[1] - u[1]*v[0]
        if abs(cross) <= 1e-9 * max(1., np.linalg.norm(u)*np.linalg.norm(v)) and u @ v < 0:
            graph[a] = [(j, w) for j, w in graph[a] if j != k]
            graph[b] = [(j, w) for j, w in graph[b] if j != k]
            if not any(j == b for j, _ in graph[a]):
                graph[a].append((b, wa+wb)); graph[b].append((a, wa+wb))
            graph[k] = []
            removed += 1

    points = vertices + [np.asarray(start, dtype=float), np.asarray(goal, dtype=float)]
    start_idx, goal_idx = len(vertices), len(vertices) + 1
    graph[start_idx] = []
    graph[goal_idx] = []
    for endpoint, endpoint_idx in ((points[start_idx], start_idx), (points[goal_idx], goal_idx)):
        candidates = sorted((i for i in range(len(vertices)) if graph[i]), key=lambda i: np.linalg.norm((vertices[i] - endpoint)*scale))
        for candidate in candidates[:max(1, connect_k)]:
            if not segment_is_safe(grid, endpoint, vertices[candidate],
                                   robot_radius=robot_radius,
                                   cell_size=cell_size,
                                   extra_margin=extra_margin):
                continue
            weight = float(np.linalg.norm((endpoint - vertices[candidate])*scale))
            graph[endpoint_idx].append((candidate, weight))
            graph[candidate].append((endpoint_idx, weight))

    queue = [(0.0, start_idx)]
    came_from = {}
    cost = {start_idx: 0.0}
    while queue:
        _, current = heappop(queue)
        if current == goal_idx:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return [tuple(map(float, points[i])) for i in path], {
                "planner": "voronoi_astar", "fallback": False,
                "vertices": len(vertices), "edges": sum(len(v) for v in graph.values()) // 2,
                "collinear_vertices_removed": removed,
            }
        for neighbour, weight in graph[current]:
            new_cost = cost[current] + weight
            if new_cost < cost.get(neighbour, float("inf")):
                cost[neighbour] = new_cost
                came_from[neighbour] = current
                heuristic = np.linalg.norm((points[neighbour] - points[goal_idx])*scale)
                heappush(queue, (new_cost + heuristic, neighbour))

    return fallback_result()


def plan_key_points(occupancy: np.ndarray, start, goal, **kwargs):
    """Run global planning and apply the finite-segment compressor."""

    raw_path, metadata = voronoi_astar_path(occupancy, start, goal, **kwargs)
    compression_keys = {"robot_radius", "cell_size", "extra_margin"}
    compression_kwargs = {key: value for key, value in kwargs.items() if key in compression_keys}
    compression: dict = {}
    key_points = extract_key_points(raw_path, occupancy, report=compression, **compression_kwargs)
    metadata = dict(metadata)
    metadata.update({"raw_points": len(raw_path), "key_points": len(key_points),
                     "compression": compression})
    return key_points, metadata


def planned_grid_route(grid, start, goal, compress=True, **geometry):
    """Round Voronoi points for grid control and validate every resulting edge.

    Every fallback uses the same footprint and cell geometry. A failed
    adjacent edge never bypasses the clearance test in the no-keypoint arm.
    """
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
    compression: dict = {}
    if compress:
        route = extract_key_points(route, grid, report=compression, **geometry)
    if not valid(route):
        return [], dict(meta, route_valid=False)
    return [tuple(map(int, p)) for p in route], dict(meta, route_valid=True,
                                                   raw_points=raw_count, key_points=len(route),
                                                   compression=compression)
