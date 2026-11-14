"""Causal recursive Top-1 forecasts from a supplied observation history."""
import numpy as np
from obstacle_prediction import landing_candidates


def forecast_top1(history, occupancy, mode='history_voting', horizon=3):
    if not history or horizon < 1:
        raise ValueError('A nonempty observed history and positive horizon are required')
    window = [tuple(p) for p in history][-4:]
    if mode == 'occupancy_aggregation':
        position = landing_candidates(window, occupancy, mode=mode)[0]
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
