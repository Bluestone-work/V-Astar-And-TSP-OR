import unittest

import numpy as np

from experiments.prediction_diagnostic import candidate_coverage, forecast_candidates, mean_coverage


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.grid = np.zeros((7, 7), dtype=int)

    def test_branches_use_their_own_histories(self):
        history = [(1, 1), (2, 1), (3, 1), (3, 2)]
        paths = forecast_candidates(history, self.grid)
        self.assertGreater(len(paths), 1)
        self.assertIn([[3, 3], [3, 4], [3, 5]], paths.tolist())
        self.assertIn([[4, 2], [5, 2], [6, 2]], paths.tolist())

    def test_oa_ties_recency_and_hold_each_cell(self):
        history = [(2, 2), (2, 3), (2, 3), (2, 2)]
        paths = forecast_candidates(history, self.grid, 'occupancy_aggregation')
        self.assertEqual(paths.tolist(), [[[2, 2]] * 3, [[2, 3]] * 3])
        self.grid[:, :] = 1
        self.assertEqual(forecast_candidates(history, self.grid, 'occupancy_aggregation').tolist(),
                         [[[2, 2]] * 3])

    def test_cv_stops_at_blocked_cell(self):
        self.grid[1, 5] = 1
        result = forecast_candidates([(1, 1), (1, 2), (1, 3), (1, 4)],
                                     self.grid, 'constant_velocity')
        self.assertEqual(result.tolist(), [[[1, 4]] * 3])

    def test_ade_and_fde_minimize_independently(self):
        scores = candidate_coverage([[[0, 0], [0, 0], [1, 0]], [[3, 0], [3, 0], [0, 0]]],
                                    [[0, 0]] * 3)
        self.assertEqual(scores, dict(K=2, min_ADE=1 / 3, min_FDE=0.))

    def test_mean_k_is_per_origin(self):
        records = [dict(history=[(1, 1)] * 4, occupancy=self.grid, future=[(1, 1)] * 3),
                   dict(history=[(1, 1), (1, 2), (1, 3), (1, 4)],
                        occupancy=self.grid, future=[(1, 4)] * 3)]
        result = mean_coverage(records)
        self.assertEqual(result['origins'], 2)
        self.assertEqual(result['predictors']['occupancy_aggregation']['mean_K'], 2.5)
        self.assertEqual(result['predictors']['constant_velocity']['mean_K'], 1.)
