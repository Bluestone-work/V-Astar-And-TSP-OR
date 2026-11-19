"""One-step cues and offline candidate coverage for Table VI, Panel B."""
import numpy as np
from obstacle_prediction import landing_candidates


def forecast_top1(history, occupancy, mode='history_voting', horizon=3):
    if len(history) == 0 or horizon < 1:
        raise ValueError('A nonempty observed history and positive horizon are required')
    window = [tuple(p) for p in history][-4:]
    if mode == 'occupancy_aggregation':
        choices = landing_candidates(window, occupancy, mode=mode)
        position = choices[0] if choices else window[-1]
        return np.repeat(np.asarray([position]), horizon, axis=0)
    result = []
    for _ in range(horizon):
        candidates = landing_candidates(window, occupancy, mode=mode)
        point = tuple(candidates[0]) if candidates else window[-1]
        result.append(point)
        window = (window + [point])[-4:]
    return np.asarray(result)


def errors(forecast, future):
    distances = np.linalg.norm(np.asarray(forecast) - np.asarray(future), axis=1)
    return dict(ADE=float(distances.mean()), FDE=float(distances[-1]))


def candidate_coverage(trajectories, future):
    """Score complete candidate paths offline, after observing the future.

    trajectories has shape (K, horizon, 2), future has shape (horizon, 2).
    The minimum mean error and the minimum endpoint error may come from
    different paths. Average these origin-level values across forecast origins.
    """
    paths = np.asarray(trajectories, dtype=float)
    truth = np.asarray(future, dtype=float)
    if truth.ndim != 2 or truth.shape[0] == 0 or truth.shape[1] != 2:
        raise ValueError("future must contain a nonempty sequence of 2D positions")
    if paths.ndim != 3 or paths.shape[0] == 0 or paths.shape[1:] != truth.shape:
        raise ValueError("supply at least one complete path matching the future horizon")
    if not np.isfinite(paths).all() or not np.isfinite(truth).all():
        raise ValueError("candidate and future positions must be finite")
    distances = np.linalg.norm(paths - truth[None, :, :], axis=2)
    return dict(K=len(paths), min_ADE=float(distances.mean(axis=1).min()),
                min_FDE=float(distances[:, -1].min()))


def forecast_candidates(history, occupancy, mode='history_voting', horizon=3):
    """Expand each path from its own four observations using the origin grid."""
    if len(history) != 4 or horizon < 1:
        raise ValueError('Supply four observed cells and a positive horizon')
    window = [tuple(p) for p in history]
    if mode == 'occupancy_aggregation':
        cells = landing_candidates(window, occupancy, mode) or [window[-1]]
        return np.asarray([[cell] * horizon for cell in cells])
    if mode not in ('constant_velocity', 'history_voting'):
        raise ValueError('Unknown predictor: ' + str(mode))
    paths = [[]]
    for _ in range(horizon):
        children = []
        for path in paths:
            observed = (window + path)[-4:]
            choices = landing_candidates(observed, occupancy, mode) or [observed[-1]]
            children.extend(path + [point] for point in choices)
        paths = children
    return np.asarray(paths)


def mean_coverage(origins):
    """Compare all three predictors on exactly the same forecast origins."""
    modes = ('constant_velocity', 'occupancy_aggregation', 'history_voting')
    totals = {mode: np.zeros(3) for mode in modes}
    count = 0
    for origin in origins:
        grid = np.asarray(origin['occupancy'])
        for mode in modes:
            paths = forecast_candidates(origin['history'], grid, mode)
            metrics = candidate_coverage(paths, origin['future'])
            totals[mode] += [metrics['K'], metrics['min_ADE'], metrics['min_FDE']]
        count += 1
    if not count:
        raise ValueError('At least one forecast origin is required')
    labels = ('mean_K', 'min_ADE', 'min_FDE')
    return {'origins': count, 'predictors': {
        mode: dict(zip(labels, sums / count)) for mode, sums in totals.items()
    }}


if __name__ == '__main__':
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('origins', type=Path, help='JSONL, one forecast origin per row')
    args = parser.parse_args()
    with args.origins.open() as handle:
        print(json.dumps(mean_coverage(json.loads(line) for line in handle if line.strip()),
                         indent=2, allow_nan=False))
