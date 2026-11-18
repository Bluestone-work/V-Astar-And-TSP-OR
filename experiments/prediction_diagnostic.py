"""Offline trajectory diagnostics for one forecast origin."""
import numpy as np
from obstacle_prediction import landing_candidates


def forecast_top1(history, occupancy, mode='history_voting', horizon=3):
    if not history or horizon < 1:
        raise ValueError('A nonempty observed history and positive horizon are required')
    window = [tuple(p) for p in history][-4:]
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
