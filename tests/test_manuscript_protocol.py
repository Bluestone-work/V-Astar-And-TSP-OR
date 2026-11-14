from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from agents import RainbowDQN
from environment import GridWorld
from experiments.prediction_diagnostic import forecast_top1
from experiments.proposed_evaluation import evaluate_scenario, summarize as summarize_grid
import global_planner as planner
from manuscript_replay import dual_priority
from obstacle_prediction import landing_candidates
from revision_experiments.continuous_transfer.tracking_loop import (
    parse_args, summarize as summarize_tracking,
)


def test_double_dqn_uses_online_selection_and_target_value():
    learner = RainbowDQN(SimpleNamespace(), device='cpu')
    learner.q_network = nn.Sequential(nn.Flatten(), nn.Linear(51, 2))
    learner.target_network = nn.Sequential(nn.Flatten(), nn.Linear(51, 2))
    with torch.no_grad():
        learner.q_network[1].weight.zero_()
        learner.q_network[1].bias.copy_(torch.tensor([5., 3.]))
        learner.target_network[1].weight.zero_()
        learner.target_network[1].bias.copy_(torch.tensor([2., 10.]))
    state = np.zeros((1, 51))
    records = [(state, 1, -1., state, done, 0.) for done in (False, True)]
    metrics = learner.compute_bellman_metrics(records)
    np.testing.assert_allclose(metrics['target_q_values'], [.98, -1.], atol=1e-6)
    assert dual_priority(-5., 4.) == pytest.approx(.75 * np.log1p(4.) + 1e-5)


def test_unsafe_rounded_edge_falls_back_before_compression(monkeypatch):
    grid = np.zeros((9, 9), dtype=int)
    grid[4, 4] = 1
    start, goal = (4, 2), (4, 6)
    direct = [start, goal]
    assert planner.extract_key_points(direct, grid, robot_radius=.5) == []
    monkeypatch.setattr(planner, 'voronoi_astar_path', lambda *a, **kw: (direct, {}))
    route, info = planner.planned_grid_route(grid, start, goal, robot_radius=.5)
    assert info['fallback'] and info['route_valid']
    assert route[0] == start and route[-1] == goal
    assert all(planner.segment_is_safe(grid, a, b, robot_radius=.5)
               for a, b in zip(route, route[1:]))
    assert planner.arrival_budget((0, 0), (100, 0)) == 105


def test_occupancy_aggregation_returns_one_candidate_and_holds_it():
    grid = np.zeros((7, 7), dtype=int)
    history = [(2, 1), (2, 2), (2, 2), (2, 1)]
    assert landing_candidates(history, grid, mode='occupancy_aggregation') == [(2, 1)]
    np.testing.assert_array_equal(forecast_top1(history, grid, 'occupancy_aggregation'),
                                  [[2, 1]] * 3)


def test_prediction_reward_uses_the_pre_action_local_window():
    env = GridWorld(np.zeros((15, 15), dtype=int), (7, 7), (12, 12), obs=0, dobs=0)
    env.reset(seed=0)
    env.predicted_landing_points = [(7, 11)]
    # A diagonal move from the observation's edge is not needed here: verify
    # that the annotation and reward use the same pre-action 7x7 footprint.
    original_advance = env._advance
    env._advance = lambda *_: (7, 10)
    try:
        env.step(3)
    finally:
        env._advance = original_advance
    assert env.last_info['reward_components']['prediction'] == 0.


def test_continuous_protocol_and_refusal_denominator():
    for group, speed, no_check, sigma, noise, delay in (
        ('C1', .50, False, 0., 0., 0), ('C1-nc', .50, True, 0., 0., 0),
        ('C1-m', .787, False, 0., 0., 0), ('C1-ncm', .787, True, 0., 0., 0),
        ('C2', .50, False, .25, 0., 0), ('C3', .50, False, 0., .02, 0),
        ('C4', .50, False, 0., 0., 1),
    ):
        args = parse_args(['--group', group, '--training-seed', '0', '--output-dir', 'unused'])
        assert (args.obstacle_speed, args.no_path_check, args.localization_sigma,
                args.sensing_flip_probability, args.action_delay) == (speed, no_check, sigma, noise, delay)
        assert args.max_decisions == 2000 and args.max_sim_time == 9680
        assert args.action_timeout == 4.84 and args.robot_radius == .5
    rows = [dict(termination='success', moves_issued=3, moves_refused=1,
                 moves_completed=3, moves_timed_out=0, mean_tracking_error=.01,
                 mean_arrival_error=.02, navigation_time_s=4, decisions=6,
                 simulated_time_s=4)]
    assert summarize_tracking(rows)['refusal_rate'] == .25


def test_grid_timing_counts_failures_and_excludes_initial_planning_from_cycles():
    common = dict(steps=4, path_length_cells=3, smoothness=.9, decision_count=4,
                  decision_seconds=.04, cycle_count=4, cycle_seconds=.20,
                  global_replans=1, planning_seconds=.2, replanning_seconds=.10,
                  deadlock_events=0, deadlock_recoveries=0)
    rows = [dict(common, success=True, termination='success', wall_time_s=2.),
            dict(common, success=False, termination='collision', wall_time_s=3.)]
    summary = summarize_grid(rows)
    assert summary['task_throughput_per_wall_s'] == .2
    assert summary['mean_wall_time_all_s'] == 2.5
    assert summary['cycle_latency_excluding_replans_ms'] == 25.
    assert summary['cycle_latency_including_replans_ms'] == 50.


def test_grid_smoothness_counts_turns_across_stop_actions():
    grid = np.zeros((12, 12), dtype=int)
    start, goal = (3, 3), (9, 9)
    env = GridWorld(grid, start, goal, obs=0, dobs=0, max_steps=3)
    actions = iter([1, 8, 3])
    policy = SimpleNamespace(select_action=lambda _: next(actions), q_network=nn.Identity(),
                             last_actions=[], device='cpu')
    config = dict(seed=10000, base_grid_map=grid, original_grid_map=grid,
                  start=start, goal=goal, goal_list=[start, goal], dynamic_obstacles=[])
    result = evaluate_scenario(env, config, policy, max_steps=3, controller='fixed_waypoints')
    assert result['smoothness'] == .5


def test_continuous_table_aggregation_keeps_seed_variation_and_pools_outcomes(tmp_path):
    import json

    from tools.seed_statistics import collect_continuous

    directories = []
    for seed in range(3):
        directory = tmp_path / str(seed)
        directory.mkdir()
        directories.append(directory)
        collisions = seed + 1
        result = {
            'protocol': {'training_seed': seed, 'checkpoint': f'seed{seed}.pth',
                         'output_dir': str(directory), 'scenarios': 300, 'group': 'C2'},
            'checkpoint_sha256': str(seed),
            'summary': dict(scenarios=300, success_rate=(300 - collisions) / 300,
                            collision_rate=collisions / 300, planning_failure_rate=0.,
                            refusal_rate=.1, mean_tracking_error=.2, timeout_rate=0.,
                            terminations={'collision_static': collisions,
                                          'success': 300 - collisions}),
        }
        (directory / 'results.json').write_text(json.dumps(result))
    summary = collect_continuous(reversed(directories))
    assert summary['cross_seed']['collision_rate']['sample_sd'] == pytest.approx(1 / 300)
    assert summary['pooled_static_collision_rate'] == pytest.approx(6 / 900)
    assert summary['pooled_false_arrival_rate'] == 0.
