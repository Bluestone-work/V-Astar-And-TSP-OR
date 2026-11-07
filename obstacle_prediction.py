"""Section IV-B: causal, weighted, one-step multi-hypothesis prediction."""
import numpy as np


def landing_candidates(history, occupancy, mode='history_voting'):
    """Ordered candidates: highest vote, latest nonzero motion, secondary.

    The caller supplies observations through t only. Zero displacements do
    not vote for one of the eight motion directions. All-zero history means
    stationary; a single observation uses all eight neighbours.
    """
    grid = np.asarray(occupancy)
    positions = [np.asarray(p, dtype=int) for p in history]
    if not positions:
        return []
    current = positions[-1]
    if mode == 'none':
        return []
    if mode == 'constant_velocity':
        direction = positions[-1]-positions[-2] if len(positions) > 1 else np.zeros(2, dtype=int)
        point = tuple(current+direction)
        return [point] if (0 <= point[0] < grid.shape[0] and 0 <= point[1] < grid.shape[1]
                           and grid[point] == 0) else [tuple(current)]
    if mode == 'occupancy_aggregation':
        # Explicit new OA definition: historical occupancy frequency, ties
        # broken by recency. The submitted PDF does not specify OA's formula.
        from collections import Counter
        recent = list(dict.fromkeys(tuple(p) for p in reversed(positions)))
        counts = Counter(tuple(p) for p in positions)
        return sorted((p for p in recent if grid[p] == 0), key=counts.get, reverse=True)
    if mode != 'history_voting':
        raise ValueError('unknown prediction mode: ' + str(mode))
    directions = []
    if len(positions) == 1:
        directions = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c]
    else:
        votes = {}
        latest = None
        for j in range(1, min(3, len(positions) - 1) + 1):
            direction = tuple(np.clip(positions[-j] - positions[-j-1], -1, 1))
            if direction == (0, 0):
                continue
            if latest is None:
                latest = direction
            votes[direction] = votes.get(direction, 0.) + 1. / j
        if votes:
            ranked = sorted(votes, key=votes.get, reverse=True)
            directions = [ranked[0], latest]
            if len(ranked) > 1 and votes[ranked[1]] > .5 * votes[ranked[0]]:
                directions.append(ranked[1])
        else:
            directions = [(0, 0)]
    result = []
    for direction in directions:
        p = tuple(current + direction)
        if (0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1]
                and grid[p] == 0 and p not in result):
            result.append(p)
    return result
