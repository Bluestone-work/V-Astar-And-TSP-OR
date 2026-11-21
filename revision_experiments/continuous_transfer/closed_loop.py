"""Continuous closed-loop evaluator rebuilt to match Section V-G.

`experiments/kinematic_closed_loop.py` departs from the specification it is
supposed to implement in six ways, each of which inflates the apparent cost of
continuous execution and so inflates what the paper would have to attribute to
kinodynamics:

1. Static obstacles are tested with `global_planner.segment_is_safe`, which
   subtracts the cell *half diagonal*, i.e. bounds each cell by its
   circumscribed circle of radius 0.707. Section V-G specifies the exact
   rectangle, Eq. (box_clearance).
2. Dynamic obstacles default to a threshold of `r_robot + half diagonal`,
   0.907 cells, a second and different criterion. V-G states a single one.
3. The robot is frozen at its end-of-period pose while the obstacle sweeps, so
   "neither body is frozen while the other moves" does not hold in the code.
4. The decision period is a free `--substeps` parameter. At the default of 25
   the period is 0.5 s, too short to complete a diagonal action, and the
   obstacle field advances 1 to 2 cells in it: the speed ratio rho reaches 5.66
   rather than 1. Nothing in the script computes or records rho.
5. The arrival budget is computed from the true pose while the controller and
   policy see the measured one, so ground truth leaks into the trigger.
6. The share of periods whose rounded cell repeats is not recorded, although
   V-G says it is reported, and it is what distinguishes an observability
   artefact of the projection from a dynamic effect.

This module fixes all six, and shares the obstacle script with the grid
evaluator so that the discrete and continuous runs of one scenario see the same
obstacle trajectory and the comparison is controlled rather than paired only by
seed. The clearance criterion is selectable so the earlier numbers remain
reproducible, but it defaults to the one the manuscript states.

What is still not matched, and is the next thing to close: the grid evaluator
drives the route through `proposed_evaluation.evaluate_scenario`'s `full`
controller, while this loop uses the environment's own event-triggered
replanning. The protocol records `controller: env_event` so the difference is
not mistaken for a kinodynamic effect.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import random as _random_module
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from revision_experiments.continuous_transfer.geometry import (        # noqa: E402
    box_clearance, moving_point_polyline_min_distance, nearby_occupied)
from revision_experiments.continuous_transfer.obstacle_motion import (  # noqa: E402
    DEFAULT_LATERAL_ACCELERATION, ObstacleDriver)
from revision_experiments.continuous_transfer.timebase import TimeBase  # noqa: E402

#: Obstacles whose closest approach over the period exceeds this cannot bound
#: the clearance, so they are skipped before the per-substep work.
SCREEN_DISTANCE = 2.5


def polyline_at(trace, fraction):
    """Position along an obstacle's sub-moves at a fraction of the period.

    Each sub-move takes an equal share of the period, the same reading the
    interval audit uses, so a speed-2 obstacle is twice as fast rather than
    teleporting at the period boundary.
    """
    if len(trace) < 2:
        return np.asarray(trace[0], dtype=float)
    count = len(trace) - 1
    position = min(max(float(fraction), 0.), 1.) * count
    index = min(int(position), count - 1)
    local = position - index
    start = np.asarray(trace[index], dtype=float)
    return start + local * (np.asarray(trace[index + 1], dtype=float) - start)


def advance_obstacle_script(env, period):
    """Advance every obstacle one period and return their traces.

    The reseeding expression is `SyncedObstacleMixin.step`'s, repeated here
    because this loop drives the environment rather than calling `step()`.
    Obstacle motion does not read the robot pose -- `_is_passable` has that test
    commented out -- so the script is a pure function of (scenario, period) and
    is identical to the one the grid evaluator replays.
    """
    env._obstacle_rng = _random_module.Random(
        (env._episode_motion_seed(), int(period)).__hash__())
    traces = []
    for obstacle in env.dynamic_obstacles:
        start = tuple(obstacle['position'])
        env._update_dynamic_obstacle(obstacle)
        traces.append([np.asarray(p, dtype=float)
                       for p in obstacle.get('_motion_trace', [start])])
    return traces


def polyline_samples(trace, fractions):
    """Vectorised `polyline_at` over many fractions of one period."""
    points = np.asarray(trace, dtype=float)
    if len(points) < 2:
        return np.repeat(points[:1], len(fractions), axis=0)
    count = len(points) - 1
    position = np.clip(np.asarray(fractions, dtype=float), 0., 1.) * count
    index = np.minimum(position.astype(int), count - 1)
    local = (position - index)[:, None]
    return points[index] + local * (points[index + 1] - points[index])


def _clearances(poses, centres, half):
    """Box clearance of every sampled pose against a set of cell centres."""
    excess = np.maximum(np.abs(poses[:, None, :] - centres[None, :, :]) - half, 0.)
    return np.linalg.norm(excess, axis=2).min(axis=1)


def evaluate(env, cfg, agent, args, timebase, trace=None):
    """One scenario through the continuous loop. Returns a record, raises nothing.

    Pass a list as `trace` to collect per-period geometry for rendering; it costs
    nothing when omitted, and the scoring path is identical either way.
    """
    from generate_test_environments import apply_environment_config
    from motion_control import MotionLimits
    from revision_experiments.continuous_transfer.drive import make_tracker
    from navigation_protocol import ACTIONS
    from experiments.proposed_evaluation import reset_policy

    apply_environment_config(env, cfg)
    # The motion stream is deliberately left at the evaluation defaults
    # (train_episode_index 0), because that is what the grid evaluator uses:
    # `ensure_motion_stream` is never called on the evaluation path, so
    # `_episode_motion_seed()` is the same constant there. Pinning it to the
    # scenario seed here would give this loop a different obstacle script and
    # destroy the controlled comparison.
    env.navigation_mode = 'event'
    for key, value in getattr(agent, 'loaded_environment_specification', {}).items():
        if key in ('prediction_mode', 'prediction_enabled', 'avoidance_reward_form',
                   'reward_scales', 'subgoal_tolerance'):
            setattr(env, key, value)
    scale = np.asarray(env.cell_size, dtype=float)
    half = .5 * scale
    radius = float(env.robot_radius)
    threshold = radius if args.clearance_mode == 'box' else radius + .5 * float(np.linalg.norm(scale))
    limits = MotionLimits(max_linear_speed=args.max_linear_speed,
                          max_angular_speed=args.max_angular_speed,
                          max_linear_acceleration=args.max_linear_acceleration,
                          max_angular_acceleration=args.max_angular_acceleration)
    rng = np.random.default_rng(int(cfg['seed']) + 700000)
    wall_start = time.perf_counter()
    env.reset(resample_obstacles=False)
    reset_policy(agent)
    agent.env = env

    heading = 0.
    if args.initial_heading == 'toward_goal':
        offset = (np.asarray(env.goal, dtype=float) - np.asarray(env.start, dtype=float)) * scale
        heading = float(math.atan2(offset[1], offset[0])) if np.any(offset) else 0.
    tracker = make_tracker(args.drive,
                           [*(np.asarray(env.start, dtype=float) * scale), heading], limits)
    driver = ObstacleDriver(env, advance_script=advance_obstacle_script,
                           speed=args.obstacle_speed,
                           lateral_acceleration=args.obstacle_lateral_acceleration,
                           scale=scale)

    pending = deque([8] * args.action_delay)
    termination = 'timeout'
    periods = 0; path_length = 0.; simulated_time = 0.
    peak_linear = 0.; peak_angular = 0.
    repeated_cells = 0; previous_cell = None
    commanded = 0.; achieved = 0.
    obstacle_submoves = []
    decision_times = []
    min_static = float('inf'); min_dynamic = float('inf')
    pending_budget = 0

    def budget():
        """Eq. (6) counted in decision periods, in the form `--budget-form` selects.

        One period is one action, so the budget is the same count the grid
        evaluator uses and no v_max scaling enters. Dividing by the reachable
        displacement per period -- which `experiments/kinematic_closed_loop.py`
        does, and an earlier version of this file did -- makes the budget about
        1.42x tighter at the Section V-G time base and so replans sooner than
        the grid evaluator on the identical geometry. Measured from the pose the
        robot has, not the true one.
        """
        measured_cell = (tracker.pose[:2] + rng.normal(0, args.localization_sigma, 2)) / scale
        distance = float(np.linalg.norm(np.asarray(env.goal, dtype=float) - measured_cell))
        if args.budget_form == 'literal':
            return int(math.ceil(distance)) + 5
        # lambda_scaled: arrival needs about d / lambda periods, so the literal
        # form expires before arrival is possible whenever
        # d > 5 lambda / (1 - lambda); at lambda = 0.84 that is 26.2 cells,
        # below this route's median sub-goal spacing of 36.2.
        return int(math.ceil(distance / args.achieved_fraction)) + 5

    allowed = budget()
    if env.planning_events and not env.planning_events[0]['accepted']:
        return {'scenario_seed': int(cfg['seed']), 'termination': 'initial_planning_failure',
                'success': False, 'periods': 0, 'wall_time_s': time.perf_counter() - wall_start}

    for period in range(args.max_periods):
        periods = period + 1
        measured = tracker.pose.copy()
        measured[:2] += rng.normal(0, args.localization_sigma, 2)
        cell = tuple(np.clip(np.rint(measured[:2] / scale).astype(int),
                             [0, 0], np.asarray(env.grid_map.shape) - 1))
        repeated_cells += int(cell == previous_cell)
        previous_cell = cell
        env.agent_pos = cell
        env.steps = period
        # The robot senses the obstacles where they physically are, rounded to
        # the occupancy grid -- not the lattice head, which the shared path has
        # already raced ahead to. The lattice state is restored immediately so
        # the script keeps generating the same path as the discrete run, while
        # grid_map is left holding the observed positions, which is what the
        # 7x7 planning window should read.
        lattice = [tuple(o['position']) for o in env.dynamic_obstacles]
        for obstacle, observed in zip(env.dynamic_obstacles, driver.observed_cells()):
            obstacle['position'] = observed
        env.rebuild_grid_map()
        state = env.get_state()
        for obstacle, head in zip(env.dynamic_obstacles, lattice):
            obstacle['position'] = head
        difference = np.asarray(env.goal, dtype=float) - measured[:2] / scale
        state[-1, :2] = [np.linalg.norm(difference), np.arctan2(difference[1], difference[0])]
        if args.sensing_flip_probability:
            window = state[:, 2:]
            mask = rng.random(window.shape) < args.sensing_flip_probability
            window[mask] = np.where(window[mask] == 0, 1, 0)
        start_clock = time.perf_counter()
        action = int(agent.select_action(state))
        decision_times.append(time.perf_counter() - start_clock)
        if args.action_delay:
            pending.append(action)
            action = pending.popleft()
        target = None if action == 8 else measured[:2] + np.asarray(ACTIONS[action], dtype=float) * scale

        # Obstacles advance during this period, not after it, at a declared
        # physical speed along the shared path.
        obstacle_frames = driver.sample_period(timebase.substeps, timebase.dt)
        if obstacle_frames.shape[1]:
            # Mean of an empty axis is nan, and Level 1 has no dynamic
            # obstacles, which json.dumps(allow_nan=False) then rejects.
            obstacle_submoves.append(float(np.linalg.norm(
                obstacle_frames[-1] - obstacle_frames[0], axis=1).mean()))

        poses = [tracker.pose[:2].copy()]
        velocity_log = ([np.asarray(tracker.telemetry(), dtype=float)]
                        if trace is not None else None)
        heading_log = [float(tracker.pose[2])] if trace is not None else None
        sagitta_bound = 0.
        for _ in range(timebase.substeps):
            noisy = tracker.pose.copy()
            noisy[:2] += rng.normal(0, args.localization_sigma, 2)
            old, new, sagitta = tracker.advance(target, timebase.dt, noisy,
                                                brake=args.waypoint_mode == 'brake')
            if velocity_log is not None:
                velocity_log.append(np.asarray(tracker.telemetry(), dtype=float))
                heading_log.append(float(tracker.pose[2]))
            simulated_time += timebase.dt
            path_length += float(np.linalg.norm(new - old))
            sagitta_bound = max(sagitta_bound, float(sagitta))
            linear, angular = tracker.telemetry()
            peak_linear = max(peak_linear, abs(linear))
            peak_angular = max(peak_angular, abs(angular))
            poses.append(new.copy())
        poses = np.asarray(poses)
        displacement = float(np.linalg.norm(poses[-1] - poses[0]))
        if target is not None:
            commanded += float(np.linalg.norm(np.asarray(ACTIONS[action], dtype=float) * scale))
            achieved += displacement
        # The sampled minimum can overstate the true one by at most half a
        # sample chord plus the arc-to-chord bound, so that is carried as a
        # margin rather than assumed negligible.
        robot_margin = .5 * float(np.max(np.linalg.norm(np.diff(poses, axis=0), axis=1))) + sagitta_bound

        static = nearby_occupied(env.base_grid_map, poses.min(axis=0) / scale,
                                 poses.max(axis=0) / scale)
        if len(static):
            clear = _clearances(poses, static * scale, half)
            min_static = min(min_static, float(clear.min()))
            if float(clear.min()) - robot_margin < threshold:
                if trace is not None:
                    trace.append({'period': period, 'poses': poses.copy(),
                                  'obstacles': obstacle_frames.copy(),
                                  'route': [tuple(g) for g in env.goal_list],
                                  'goal_index': env.current_goal_index,
                                  'static_clearance': float(clear.min()),
                                  'collision': 'static',
                          'action': action, 'target': None if target is None else target.copy(),
                          'velocity': np.asarray(velocity_log),
                          'heading': np.asarray(heading_log)})
                termination = 'collision_static'
                break

        collided = False
        if obstacle_frames.shape[1]:
            offsets = np.abs(poses[:, None, :] - obstacle_frames)
            clearances = np.linalg.norm(np.maximum(offsets - half, 0.), axis=2)
            min_dynamic = min(min_dynamic, float(clearances.min()))
            obstacle_margin = .5 * float(np.max(np.linalg.norm(
                np.diff(obstacle_frames, axis=0), axis=2)))
            collided = bool((clearances - robot_margin - obstacle_margin < threshold).any())
        if trace is not None:
            trace.append({'period': period, 'poses': poses.copy(),
                          'obstacles': obstacle_frames.copy(),
                          'route': [tuple(g) for g in env.goal_list],
                          'goal_index': env.current_goal_index,
                          'static_clearance': None if not len(static) else float(clear.min()),
                          'collision': 'dynamic' if collided else None,
                          'action': action, 'target': None if target is None else target.copy(),
                          'velocity': np.asarray(velocity_log),
                          'heading': np.asarray(heading_log)})
        if collided:
            termination = 'collision_dynamic'
            break

        # Sub-goal progress is a decision the robot makes on what it measured;
        # final success is declared by the simulator on the true pose. Both are
        # read at the END of the period: `measured` above is the period-start
        # pose the action was chosen from, and the robot travels up to sqrt(2)
        # cells after that, so testing arrival on it reported arrivals that had
        # not happened and, at sigma = 0, a `believed_arrival` that cannot exist.
        measured_end = tracker.pose[:2] + rng.normal(0, args.localization_sigma, 2)
        believed = float(np.linalg.norm(measured_end / scale - np.asarray(env.goal, dtype=float)))
        if believed <= env.subgoal_tolerance:
            if env.current_goal_index + 1 == len(env.goal_list):
                true_distance = float(np.linalg.norm(
                    tracker.pose[:2] / scale - np.asarray(env.goal, dtype=float)))
                termination = 'success' if true_distance <= env.subgoal_tolerance else 'believed_arrival'
                break
            env.current_goal_index += 1
            env.goal = env.goal_list[env.current_goal_index]
            pending_budget = 0
            allowed = budget()
        else:
            pending_budget += 1
        if pending_budget >= allowed:
            if not env._plan_route():
                termination = 'planning_failure'
                break
            pending_budget = 0
            allowed = budget()

    fraction = achieved / commanded if commanded else 1.
    return {'scenario_seed': int(cfg['seed']), 'termination': termination,
            'success': termination == 'success', 'periods': periods,
            'simulated_time_s': simulated_time, 'path_length_physical': path_length,
            'achieved_fraction': fraction,
            # Under cruise the robot can cover more than the commanded one-cell
            # displacement, so lambda exceeds 1; rho's job is to flag a handicap,
            # so it is clamped rather than reported below 1 for that reason.
            'speed_ratio': timebase.speed_ratio(min(max(fraction, 1e-6), 1.)),
            'obstacle_cells_per_period_measured': (float(np.mean(obstacle_submoves))
                                                   if obstacle_submoves else 0.),
            'obstacle_speed': args.obstacle_speed,
            'repeated_cell_share': repeated_cells / periods if periods else None,
            'min_static_clearance': None if not np.isfinite(min_static) else min_static,
            'min_dynamic_clearance': None if not np.isfinite(min_dynamic) else min_dynamic,
            'clearance_threshold': threshold,
            'peak_linear_speed': peak_linear, 'peak_angular_speed': peak_angular,
            'decision_latency_s': float(np.mean(decision_times)) if decision_times else None,
            'global_replans': env.global_replans,
            'planning_seconds': sum(e['seconds'] for e in env.planning_events),
            'wall_time_s': time.perf_counter() - wall_start}


def summarize(rows, timebase):
    """Aggregate one level, keeping the diagnostics V-G says are reported.

    rho comes from the aggregate lambda, not from averaging the per-episode
    rho. An episode that ends in a few periods can have a very small lambda,
    and averaging 1/lambda lets those outliers dominate: an earlier version
    reported rho = 8.41 for a level whose lambda averaged 0.66.
    """
    def mean(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(values)) if values else None

    def percentile(key, q):
        values = [r[key] for r in rows if r.get(key) is not None]
        return float(np.percentile(values, q)) if values else None
    terminations = {}
    for row in rows:
        terminations[row['termination']] = terminations.get(row['termination'], 0) + 1
    return {'scenarios': len(rows),
            'success_rate': sum(r['success'] for r in rows) / len(rows),
            'terminations': terminations,
            'achieved_fraction': mean('achieved_fraction'),
            'achieved_fraction_p10': percentile('achieved_fraction', 10),
            'achieved_fraction_p90': percentile('achieved_fraction', 90),
            'speed_ratio': timebase.speed_ratio(max(mean('achieved_fraction') or 1., 1e-6)),
            'obstacle_displacement_per_period': mean('obstacle_cells_per_period_measured'),
            'repeated_cell_share': mean('repeated_cell_share'),
            'mean_periods': mean('periods'),
            'mean_decision_latency_s': mean('decision_latency_s'),
            'min_static_clearance': min((r['min_static_clearance'] for r in rows
                                         if r.get('min_static_clearance') is not None), default=None),
            'min_dynamic_clearance': min((r['min_dynamic_clearance'] for r in rows
                                          if r.get('min_dynamic_clearance') is not None), default=None)}


def build_parser():
    p = argparse.ArgumentParser(description='Continuous closed-loop transfer test (Section V-G).')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--map-file', default='grid_map2.txt')
    p.add_argument('--device', default='cpu')
    p.add_argument('--scenarios', type=int, default=300)
    p.add_argument('--eval-seed', type=int, default=10000)
    p.add_argument('--max-periods', type=int, default=2000)
    p.add_argument('--levels', type=int, nargs='+', default=[1, 2, 3, 4])
    p.add_argument('--levels-config', default=None)
    p.add_argument('--replay-mode', choices=('manuscript', 'uniform', 'per'), default='manuscript')
    p.add_argument('--architecture', default='bilstm_dueling')
    p.add_argument('--allow-historical-checkpoint', action='store_true')
    p.add_argument('--dt', type=float, default=.02)
    p.add_argument('--substeps', type=int, default=None,
                   help='Integration substeps per decision. Omit for the fewest that can '
                        'complete a diagonal action, which is 71 at the default dt and speed.')
    p.add_argument('--max-linear-speed', type=float, default=1.)
    p.add_argument('--max-angular-speed', type=float, default=2.)
    p.add_argument('--max-linear-acceleration', type=float, default=1.)
    p.add_argument('--max-angular-acceleration', type=float, default=4.)
    p.add_argument('--robot-radius', type=float, default=.2)
    p.add_argument('--cell-size', type=float, nargs=2, default=[1., 1.])
    p.add_argument('--localization-sigma', type=float, default=.25)
    p.add_argument('--sensing-flip-probability', type=float, default=.02)
    p.add_argument('--action-delay', type=int, default=0)
    p.add_argument('--budget-form', choices=('literal', 'lambda_scaled'), default='literal',
                   help="'literal' is Eq. (6) exactly as printed, ceil(d)+5 periods, the same "
                        'trigger the grid evaluator uses. Its validity proof assumes one action '
                        'advances one cell, which continuous execution violates: with an achieved '
                        'fraction lambda the budget expires before arrival is possible for any '
                        'sub-goal farther than 5*lambda/(1-lambda) cells. "lambda_scaled" divides '
                        'by --achieved-fraction to restore the property. Report both; their '
                        'difference is the trigger aperture, not a kinodynamic cost.')
    p.add_argument('--achieved-fraction', type=float, default=None,
                   help='lambda for --budget-form lambda_scaled. Take it from the achieved '
                        'fraction a literal run measured on the same time base and checkpoint, '
                        'so the constant has a stated provenance.')
    p.add_argument('--obstacle-speed', type=float, default=.5,
                   help='Obstacle speed in cells/s. environment.py draws 1 or 2 lattice cells '
                        'per step, which at T = 1.42 s is 0.70 to 1.99 cells/s against a robot '
                        'whose v_max is 1.0 and whose effective speed is 0.77 to 0.80, so part '
                        'of the collisions were not avoidable by any policy. The default of 0.5 '
                        'is half the robot v_max, which keeps avoidance feasible. Obstacles '
                        'follow the same lattice path as the discrete run; only the timing is '
                        'physical, so period-boundary positions no longer coincide with it.')
    p.add_argument('--obstacle-lateral-acceleration', type=float,
                   default=DEFAULT_LATERAL_ACCELERATION,
                   help='Bound in cells/s^2 used for the corner fillet radius v^2/a, so the '
                        'obstacles do not turn instantaneously at cell centres. Pass 0 for the '
                        'unrounded lattice corners.')
    p.add_argument('--clearance-mode', choices=('box', 'circumscribed'), default='box',
                   help="'box' is Eq. (box_clearance) of Section V-G, the exact cell rectangle. "
                        "'circumscribed' reproduces the earlier criterion, which bounded each "
                        'cell by its circumscribed circle and so applied a 0.907-cell threshold.')
    p.add_argument('--drive', choices=('unicycle', 'holonomic'), default='unicycle',
                   help="'unicycle' is the nonholonomic base of Section V-G. 'holonomic' is "
                        'an omnidirectional base, bounded in speed and acceleration the same '
                        'way but with no heading constraint, which matches the action space '
                        'the policy was trained on -- it has no heading state at all. Report '
                        'both: their difference is the cost of nonholonomy.')
    p.add_argument('--waypoint-mode', choices=('brake', 'cruise'), default='brake',
                   help="'brake' keeps the sqrt(2 a_max d) stopping-distance term, active over "
                        'the second half of a one-cell hop. "cruise" treats the one-cell target '
                        'as a point to pass through, which is what the grid action means -- the '
                        'policy issues another action next period and never intended to stop '
                        'there. Holonomic drive only; the unicycle shares motion_control.py.')
    p.add_argument('--initial-heading', choices=('toward_goal', 'zero'), default='toward_goal',
                   help="'zero' points along the row axis regardless of the route, which costs "
                        'the first period to a turn in place.')
    p.add_argument('--allow-short-period', action='store_true',
                   help='Permit a period too short to complete a diagonal action. Such a run '
                        'measures a speed handicap, not kinodynamics; refused without this flag.')
    p.add_argument('--output-dir', required=True)
    return p


def main():
    from experiments import proposed_evaluation as pe
    from generate_test_environments import MAP_NODES, generate_environment_config
    from revision_experiments.legacy_arm.synced_env import SyncedGridWorld
    from revision_experiments.continuous_transfer.timebase import substeps_for
    from train import set_global_seed

    parser = build_parser()
    args = parser.parse_args()
    if (args.dt <= 0 or args.scenarios < 1 or args.max_periods < 1 or args.action_delay < 0
            or args.localization_sigma < 0 or not 0 <= args.sensing_flip_probability <= 1):
        parser.error('positive time and budget, non-negative noise, flip probability in [0,1]')
    if args.budget_form == 'lambda_scaled' and not (args.achieved_fraction
                                                   and 0 < args.achieved_fraction <= 1):
        parser.error('--budget-form lambda_scaled needs --achieved-fraction in (0, 1], '
                     'measured by a literal run on the same time base')
    substeps = args.substeps or substeps_for(args.dt, args.max_linear_speed)
    timebase = TimeBase(dt=args.dt, substeps=substeps,
                        max_linear_speed=args.max_linear_speed,
                        obstacle_cells_per_period=args.obstacle_speed * args.dt * substeps)
    if not timebase.completes_diagonal and not args.allow_short_period:
        parser.error('period %.3f s is shorter than sqrt(2)/v_max = %.3f s, so a diagonal '
                     'action cannot complete and the obstacles are effectively faster than '
                     'the robot. Pass --allow-short-period to run it anyway.'
                     % (timebase.period, timebase.minimum_period))

    out = Path(args.output_dir)
    if (out / 'results.json').exists():
        parser.error('Use a new output directory to preserve existing evidence')
    out.mkdir(parents=True, exist_ok=True)
    set_global_seed(args.eval_seed)
    levels_in_use = pe.load_levels(args.levels_config)
    grid = np.loadtxt(args.map_file, dtype=int)
    nodes = MAP_NODES[os.path.basename(args.map_file)]
    checkpoint = Path(args.checkpoint).resolve()
    base = pe.GridWorld(grid, start=nodes[0], goal=nodes[-1], goal_list=nodes)
    agent = pe.RainbowDQN(env=base, replay_mode=args.replay_mode,
                          architecture=args.architecture, device=args.device)
    agent.load_model(str(checkpoint), allow_historical=args.allow_historical_checkpoint)

    levels = {}
    for level in args.levels:
        static, dynamic, mode = levels_in_use[level]
        template = pe.GridWorld(grid, start=nodes[0], goal=nodes[-1], goal_list=nodes,
                                obs=static, dobs=dynamic, obstacle_mode=mode,
                                block_size_range=(2, 5), max_steps=args.max_periods)
        rows = []
        with (out / ('level%d.jsonl' % level)).open('w', buffering=1) as handle:
            for index in range(args.scenarios):
                cfg = generate_environment_config(template, args.eval_seed + index)
                env = SyncedGridWorld(grid, start=nodes[0], goal=nodes[-1], goal_list=nodes,
                                      obs=static, dobs=dynamic, obstacle_mode=mode,
                                      max_steps=args.max_periods, robot_radius=args.robot_radius,
                                      cell_size=args.cell_size)
                row = evaluate(env, cfg, agent, args, timebase)
                handle.write(json.dumps(row, allow_nan=False) + '\n')
                rows.append(row)
                if (index + 1) % 25 == 0:
                    print('CLOSED-LOOP L%d %d/%d SR=%.3f rho=%.2f lambda=%.2f' % (
                        level, index + 1, args.scenarios,
                        sum(r['success'] for r in rows) / len(rows),
                        timebase.speed_ratio(max(float(np.mean(
                            [r['achieved_fraction'] for r in rows
                             if r.get('achieved_fraction') is not None])), 1e-6)),
                        float(np.mean([r['achieved_fraction'] for r in rows
                                       if r.get('achieved_fraction') is not None]))),
                        flush=True)
        levels['level_%d' % level] = summarize(rows, timebase)

    result = {'protocol': vars(args), 'time_base': timebase.as_dict(),
              'substeps_resolved': substeps,
              'checkpoint': str(checkpoint),
              'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'collision_model': 'swept exact cell rectangle, both bodies moving'
                                 if args.clearance_mode == 'box' else
                                 'swept circumscribed cell circle, both bodies moving',
              'controller': 'env_event',
              'obstacle_script': 'shared with the grid evaluator; pure function of '
                                 '(scenario, period) because obstacle motion never reads the pose',
              'scope': 'simulation only; no wheel slip, field-of-view limit, asynchronous '
                       'sensor timestamps, map-to-pose drift, or hardware in the loop',
              'levels': levels}
    (out / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: {'success_rate': v['success_rate'], 'speed_ratio': v['speed_ratio'],
                          'achieved_fraction': v['achieved_fraction'],
                          'repeated_cell_share': v['repeated_cell_share']}
                      for k, v in levels.items()}, indent=2))


if __name__ == '__main__':
    main()
