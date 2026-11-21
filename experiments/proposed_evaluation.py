"""Auditable evaluation of corrected TSP-OR checkpoints on manuscript L1--L4.

The fixed-waypoint controller is a diagnostic with stored waypoint guidance.
Other controllers execute real online planning calls and are explicitly
post-training controller interventions, not independently retrained ablations.
Raw scenario records are flushed individually; exceptions abort the job.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import random
import platform
import sys
import time

import numpy as np
import torch

from agents import RainbowDQN
from environment import GridWorld
from generate_test_environments import MAP_NODES, apply_environment_config, generate_environment_config
from global_planner import arrival_budget, extract_key_points, grid_astar_path, segment_is_safe, voronoi_astar_path
from train import set_global_seed
from navigation_protocol import CODE_PROTOCOL
from global_planner import planned_grid_route
from experiments.keypoint_density import density_route

LEVELS = {1: (50, 0, 'point'), 2: (50, 30, 'point'),
          3: (50, 30, 'block'), 4: (50, 50, 'block')}


def load_levels(path=None):
    """Load the released L1-L4 definitions; custom files define a separate protocol."""
    if not path:
        return dict(LEVELS)
    spec = json.loads(Path(path).read_text())
    out = {}
    for key, value in spec.items():
        out[int(key)] = (int(value['obs']), int(value['dobs']), str(value['mode']))
    return out
CONTROLLERS = ('fixed_waypoints', 'full', 'no_keypoint', 'direct_goal',
               'no_event', 'fixed_period', 'euclidean_trigger',
               'fixed_waypoints_event')
ACTIONS = np.array([(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1),
                    (-1, 1), (1, -1), (1, 1), (0, 0)])


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


def reset_policy(agent):
    # RecurrentPPO carries an LSTM state across a trajectory; leaving it in place
    # between scenarios would leak the previous episode into the next one.
    if hasattr(agent, 'reset_recurrent_state'):
        agent.reset_recurrent_state()
    agent.epsilon = 0.0
    agent.last_actions.clear()
    agent.stuck_counter = 0
    agent.last_max_q = None
    agent.escape_mode = False
    agent.escape_steps = 0
    agent.q_network.eval()


def planned_route(grid, start, goal, compress, max_segment=None, **geometry):
    # max_segment=None delegates to the released compression.
    return density_route(grid, start, goal, compress=compress,
                         max_segment=max_segment, **geometry)


def set_route(env, route):
    goals = list(route)
    while goals and tuple(goals[0]) == tuple(env.agent_pos):
        goals.pop(0)
    if not goals:
        return False
    env.goal_list = goals
    env.current_goal_index = 0
    env.goal = goals[0]
    env.distance = float(np.linalg.norm(np.asarray(env.goal) - np.asarray(env.agent_pos)))
    return True


def perturbed_state(state, rng, level, localization_sigma=.25):
    if level == 'ideal':
        return state
    state = np.array(state, dtype=np.float32, copy=True)
    if level in ('localization', 'combined'):
        # Public observations contain raw cell distance and radians. The default
        # sigma of 0.25 cell is under 1% of a typical sub-goal distance, so it
        # answers "is the policy robust" with a null result rather than a
        # tolerance. Sweeping sigma gives the magnitude at which it degrades,
        # which is what R2-6 asks for. Azimuth noise scales with it, holding the
        # released 0.02 rad per 0.25 cell ratio.
        state[:, 0] = np.maximum(0, state[:, 0]
                                 + rng.normal(0, localization_sigma, len(state)))
        state[:, 1] += rng.normal(0, .02 * localization_sigma / .25, len(state))
    if level in ('sensing', 'combined'):
        cells = state[:, 2:]
        mask = rng.random(cells.shape) < .02
        cells[mask] = np.where(cells[mask] == 0, 1., 0.)
    return state


def evaluate_scenario(env, cfg, agent, max_steps=2000, controller='fixed_waypoints',
                      keypoint_max_segment=None,
                      fixed_period=20, perturbation='ideal', localization_sigma=.25,
                      dump_paths=False):
    random.seed(int(cfg['seed']))
    np.random.seed(int(cfg['seed']))
    apply_environment_config(env, cfg)
    if hasattr(env, 'ensure_motion_stream'):
        env.ensure_motion_stream(int(cfg['seed']))
    reset_policy(agent)
    agent.env = env
    env.navigation_mode = 'fixed_waypoints'  # This evaluator owns the controller interventions.
    for key, value in getattr(agent, 'loaded_environment_specification', {}).items():
        if key in ('prediction_mode', 'prediction_enabled', 'avoidance_reward_form', 'reward_scales', 'subgoal_tolerance'):
            setattr(env, key, value)
    final_goal = tuple(cfg['goal'])
    rng = np.random.default_rng(int(cfg['seed']) + 700000)
    pending = deque([8])
    start_time = time.perf_counter()
    state = env.reset(resample_obstacles=False)
    path = [tuple(env.agent_pos)]
    goal_history = [tuple(env.goal)]
    events = []
    decisions = []; cycles = []
    replan_count = 0; planning_seconds = 0.0; fallback_count = 0
    initial_points = len(env.goal_list)
    termination = 'timeout'; success = False
    active_goal = tuple(env.goal); goal_age = 0
    budget = arrival_budget(env.agent_pos, active_goal)
    pending_recoveries = []
    recoveries = []
    max_replans = max_steps  # At most one replan per control step; no extra paper-undefined cap.
    steps = 0
    deadlock_events = 0
    deadlock = None
    deadlock_recovery_steps = []

    def plan_now(initial=False):
        nonlocal planning_seconds, fallback_count, replan_count, state
        t0 = time.perf_counter()
        # Initial planning uses only permanent structure. Online calls add
        # occupancy visible in the robot's 7x7 sensor window.
        grid = env.original_grid_map.copy()
        if not initial:
            r, c = env.agent_pos
            lo_r, hi_r = max(0, r-3), min(env.rows, r+4)
            lo_c, hi_c = max(0, c-3), min(env.cols, c+4)
            grid[lo_r:hi_r, lo_c:hi_c] = env.grid_map[lo_r:hi_r, lo_c:hi_c]
        if grid[tuple(env.agent_pos)] or grid[final_goal]:
            route, meta = [], {'fallback': False}
        else:
            route, meta = planned_route(grid, env.agent_pos, final_goal,
                                        compress=controller != 'no_keypoint',
                                        max_segment=keypoint_max_segment,
                                        robot_radius=env.robot_radius, cell_size=env.cell_size,
                                        extra_margin=env.extra_margin)
        elapsed = time.perf_counter() - t0
        planning_seconds += elapsed
        fallback_count += int(meta.get('fallback', False))
        if not initial:
            replan_count += 1
        accepted = set_route(env, route) if route else False
        events.append({'step': steps, 'initial': initial, 'seconds': elapsed,
                       'route_points': len(route), 'accepted': accepted, **meta})
        if accepted:
            # Replace only the goal features in the most recent observation;
            # do not add an extra obstacle-history observation at this timestep.
            state = np.array(state, copy=True)
            delta = np.asarray(env.goal) - np.asarray(env.agent_pos)
            state[-1, :2] = [np.linalg.norm(delta), np.arctan2(delta[1], delta[0])]
        return accepted

    if controller == 'direct_goal':
        set_route(env, [final_goal])
        delta = np.asarray(env.goal) - np.asarray(env.agent_pos)
        state[-1, :2] = [np.linalg.norm(delta), np.arctan2(delta[1], delta[0])]
        initial_points = 1
    elif controller not in ('fixed_waypoints', 'fixed_waypoints_event'):
        if not plan_now(initial=True):
            termination = 'initial_planning_failure'
        initial_points = len(env.goal_list)
    active_goal = tuple(env.goal)
    budget = arrival_budget(env.agent_pos, active_goal)
    if controller == 'euclidean_trigger':
        budget = int(np.ceil(np.linalg.norm(np.asarray(env.agent_pos)-active_goal) / np.sqrt(2))) + 5

    if termination != 'initial_planning_failure':
        for step in range(max_steps):
            cycle_start = time.perf_counter()
            trigger = False
            if controller == 'fixed_period':
                trigger = step > 0 and step % fixed_period == 0
            elif controller in ('full', 'no_keypoint', 'euclidean_trigger',
                                'fixed_waypoints_event'):
                trigger = goal_age >= budget
            if trigger:
                if replan_count >= max_replans:
                    cycles.append(time.perf_counter() - cycle_start)
                    termination = 'replan_limit'; break
                if not plan_now():
                    cycles.append(time.perf_counter() - cycle_start)
                    termination = 'online_planning_failure'; break
                pending_recoveries.append((step, tuple(env.goal)))
                goal_age = 0; active_goal = tuple(env.goal)
                budget = arrival_budget(env.agent_pos, active_goal)
                if controller == 'euclidean_trigger':
                    budget = int(np.ceil(np.linalg.norm(np.asarray(env.agent_pos)-active_goal) / np.sqrt(2))) + 5
            policy_state = perturbed_state(state, rng, perturbation,
                                           localization_sigma=localization_sigma)
            t0 = time.perf_counter()
            action = int(agent.select_action(policy_state))
            if torch.cuda.is_available() and str(agent.device).startswith('cuda'):
                torch.cuda.synchronize(agent.device)
            decisions.append(time.perf_counter() - t0)
            if perturbation in ('delay', 'combined'):
                pending.append(action); action = pending.popleft()
            if perturbation == 'axis_only' and action in (4, 5, 6, 7):
                action = (0, 0, 1, 1)[action - 4]
            current = tuple(env.agent_pos)
            attempted = tuple(np.asarray(current) + ACTIONS[action])
            collision = any(current == tuple(o['position']) for o in env.dynamic_obstacles)
            collision |= not (0 <= attempted[0] < env.rows and 0 <= attempted[1] < env.cols)
            if not collision:
                collision = bool(env.grid_map[attempted] != 0)
            # Obstacle stochasticity is keyed by scenario and control step,
            # independent of training seed, policy RNG, and prior episodes.
            random.seed(int(cfg['seed']) * 1000003 + step)
            state, _, done = env.step(action)
            steps = step + 1
            path.append(tuple(env.agent_pos))
            goal_history.append(tuple(env.goal))
            collision |= bool(env.last_info.get('collision'))
            cycles.append(time.perf_counter() - cycle_start)
            if collision:
                termination = 'collision'; break
            if deadlock is not None:
                since, anchor, stuck_goal, stuck_distance = deadlock
                remaining = np.linalg.norm(np.asarray(stuck_goal) - np.asarray(env.agent_pos))
                displacement = np.linalg.norm(np.asarray(anchor) - np.asarray(env.agent_pos))
                if displacement >= 2 and (remaining <= stuck_distance - 2 or tuple(env.goal) != stuck_goal):
                    deadlock_recovery_steps.append(steps - since)
                    deadlock = None
            if deadlock is None and len(path) >= 8 and len(set(path[-8:])) <= 2:
                deadlock_events += 1
                deadlock = (steps, tuple(env.agent_pos), tuple(env.goal),
                            float(np.linalg.norm(np.asarray(env.goal) - np.asarray(env.agent_pos))))
            if env.last_info.get('success') and np.linalg.norm(np.asarray(env.agent_pos)-final_goal) <= env.subgoal_tolerance:
                success = True; termination = 'success'; break
            if tuple(env.goal) != active_goal:
                active_goal = tuple(env.goal); goal_age = 0
                budget = arrival_budget(env.agent_pos, active_goal)
                if controller == 'euclidean_trigger':
                    budget = int(np.ceil(np.linalg.norm(np.asarray(env.agent_pos)-active_goal) / np.sqrt(2))) + 5
            else:
                goal_age += 1
            unresolved = []
            for since, recovery_goal in pending_recoveries:
                if tuple(env.agent_pos) == recovery_goal:
                    recoveries.append(steps - since)
                else:
                    unresolved.append((since, recovery_goal))
            pending_recoveries = unresolved
            if done:
                termination = env.last_info.get('termination', 'other_terminal')
                break
    wall_seconds = time.perf_counter() - start_time
    moves = np.diff(np.asarray(path, dtype=float), axis=0)
    lengths = np.linalg.norm(moves, axis=1) if len(moves) else np.array([])
    angles = []
    for a, b in zip(moves, moves[1:]):
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm:
            angles.append(float(np.arccos(np.clip(np.dot(a, b) / norm, -1, 1))))
    return {'scenario_seed': int(cfg['seed']), 'success': success, 'termination': termination,
            'steps': steps, 'path_length_cells': float(lengths.sum()),
            'smoothness': 1.0 - float(np.mean(angles) / np.pi) if angles else 1.0,
            'wall_time_s': wall_seconds, 'decision_seconds': float(sum(decisions)),
            'decision_count': len(decisions), 'cycle_seconds': float(sum(cycles)), 'cycle_count': len(cycles),
            'global_replans': replan_count, 'planning_seconds': planning_seconds,
            'initial_planning_calls': sum(e['initial'] for e in events),
            'total_planning_calls': len(events),
            'initial_planning_seconds': sum(e['seconds'] for e in events if e['initial']),
            'replanning_seconds': sum(e['seconds'] for e in events if not e['initial']),
            'fallback_calls': fallback_count, 'initial_keypoints': initial_points,
            'budget_exceedances': sum(not e['initial'] for e in events) if controller != 'fixed_period' else 0,
            'subgoal_recoveries': len(recoveries), 'recovery_steps': recoveries,
            'deadlock_events': deadlock_events,
            'deadlock_recoveries': len(deadlock_recovery_steps),
            'deadlock_recovery_steps': deadlock_recovery_steps,
            'planning_events': events,
            'path_sha256': hashlib.sha256(np.asarray(path, dtype=np.int32).tobytes()).hexdigest(),
            # Only the hash is kept by default: a 2000-step path per scenario would
            # dominate the record. The deadlock counter reports two-cell cycles but
            # the collapsed arms time out while still displacing roughly one cell per
            # step, so the failure is neither a collision nor a two-cell cycle and the
            # geometry has to be looked at directly to say what it is.
            **({'path': [list(map(int, q)) for q in path],
                'goal_path': [list(map(int, g)) for g in goal_history]} if dump_paths else {})}


def summarize(rows):
    successes = [r for r in rows if r['success']]
    elapsed = sum(r['wall_time_s'] for r in rows)
    decisions = sum(r['decision_count'] for r in rows)
    deadlocks = sum(r['deadlock_events'] for r in rows)
    def mean(key, subset):
        return float(np.mean([r[key] for r in subset])) if subset else None
    return {'scenarios': len(rows), 'successes': len(successes),
            'success_rate': len(successes) / len(rows) if rows else None,
            'mean_steps_all': mean('steps', rows),
            'mean_steps_success': mean('steps', successes),
            'mean_path_length_success': mean('path_length_cells', successes),
            'mean_smoothness_success': mean('smoothness', successes),
            'mean_wall_time_success_s': mean('wall_time_s', successes),
            'total_episode_wall_time_s': elapsed,
            'task_throughput_per_wall_s': len(successes) / elapsed if elapsed else None,
            'decision_latency_ms': 1000 * sum(r['decision_seconds'] for r in rows) / decisions if decisions else None,
            'cycle_latency_including_replans_ms': 1000 * sum(r['cycle_seconds'] for r in rows) / sum(r.get('cycle_count', r['decision_count']) for r in rows) if decisions else None,
            'mean_global_replans': mean('global_replans', rows),
            'mean_planning_seconds': mean('planning_seconds', rows),
            'deadlock_events': deadlocks,
            'deadlock_recoveries': sum(r['deadlock_recoveries'] for r in rows),
            'deadlock_recoveries_per_episode': (sum(r['deadlock_recoveries'] for r in rows) / len(rows) if rows else None),
            'deadlock_recovery_rate': sum(r['deadlock_recoveries'] for r in rows) / deadlocks if deadlocks else None,
            'termination_counts': {k: sum(r['termination'] == k for r in rows) for k in sorted({r['termination'] for r in rows})}}


def _world_class(args):
    """Use a policy-independent obstacle stream in paired evaluations."""
    if getattr(args, 'synced_obstacles', False):
        from experiment_protocol import SyncedGridWorld
        return SyncedGridWorld
    return GridWorld


def install_prediction_noise(probability, seed):
    """Replace each predicted landing cell by a random free neighbour with the given probability."""
    import environment as _environment
    original = _environment.landing_candidates
    rng = np.random.default_rng(seed)
    offsets = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c]
    def noisy(history, occupancy, mode='history_voting'):
        candidates = original(history, occupancy, mode=mode)
        if not candidates or not history:
            return candidates
        grid = np.asarray(occupancy); cur = np.asarray(history[-1], dtype=int)
        free = [tuple(cur + o) for o in offsets
                if 0 <= cur[0] + o[0] < grid.shape[0] and 0 <= cur[1] + o[1] < grid.shape[1]
                and grid[tuple(cur + o)] == 0]
        out = []
        for c in candidates:
            if free and rng.random() < probability:
                c = free[int(rng.integers(len(free)))]
            if c not in out:
                out.append(c)
        return out
    _environment.landing_candidates = noisy


def run(args):
    set_global_seed(args.seed)
    if getattr(args, 'perturbation', 'ideal') == 'prediction_noise':
        install_prediction_noise(args.prediction_noise, int(args.eval_seed) + 911)
    levels_in_use = load_levels(getattr(args, 'levels_config', None))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    grid = np.loadtxt(args.map_file, dtype=int)
    nodes = MAP_NODES[os.path.basename(args.map_file)]
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    base = GridWorld(grid, start=nodes[0], goal=nodes[-1], goal_list=nodes)
    agent_kind = getattr(args, 'agent_kind', 'proposed')
    if agent_kind == 'proposed':
        agent = RainbowDQN(env=base, replay_mode=getattr(args, 'replay_mode', 'manuscript'),
                           architecture=getattr(args, 'architecture', 'bilstm_dueling'), device=args.device)
    agent.load_model(str(checkpoint), allow_historical=getattr(args, 'allow_historical_checkpoint', False))
    cpuinfo = Path('/proc/cpuinfo')
    cpu_model = next((line.split(':',1)[1].strip() for line in cpuinfo.read_text().splitlines()
                      if line.startswith('model name')), platform.processor()) if cpuinfo.exists() else platform.processor()
    runtime = dict(platform=platform.platform(), machine=platform.machine(), cpu_model=cpu_model, cpu_count=os.cpu_count(),
        processor=platform.processor(), python=sys.version.split()[0], torch=torch.__version__,
        numpy=np.__version__, device=str(agent.device), torch_threads=torch.get_num_threads(),
        gpu=(torch.cuda.get_device_name(agent.device) if str(agent.device).startswith('cuda') else None))
    results = []
    for level in args.levels:
        static, dynamic, mode = levels_in_use[level]
        template = GridWorld(grid, start=nodes[0], goal=nodes[-1], goal_list=nodes,
                             obs=static, dobs=dynamic, obstacle_mode=mode,
                             block_size_range=(2, 5), max_steps=args.max_steps)
        record_file = out / ('level%d.jsonl' % level)
        protocol = {'code_protocol': CODE_PROTOCOL, 'checkpoint_code_protocol': agent.loaded_code_protocol,
                    'training_source_sha256': agent.training_source_sha256,
                    'environment_specification': agent.loaded_environment_specification,
                    'agent_kind': agent_kind, 'bezier_t': getattr(agent, 'bezier_t', None),
                    'replay_mode': agent.replay_mode, 'architecture': agent.architecture,
                    'robot_radius': args.robot_radius, 'cell_size': args.cell_size, 'extra_margin': args.extra_margin,
                    'map_sha256': hashlib.sha256(Path(args.map_file).read_bytes()).hexdigest(),
                    'map_shape': list(grid.shape), 'map_file': os.path.basename(args.map_file),
                    'training_seed': agent.loaded_training_seed, 'evaluation_rng_seed': args.seed,
                    'learner_specification': agent.loaded_learner_specification,
                    'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_hash,
                    'level': level, 'scenarios': args.scenarios, 'eval_seed': args.eval_seed,
                    'max_steps': args.max_steps, 'controller': args.controller,
                    'keypoint_max_segment': args.keypoint_max_segment,
                    'perturbation': args.perturbation, 'fixed_period': args.fixed_period,
                    'localization_sigma': args.localization_sigma, 'prediction_noise': (args.prediction_noise if args.perturbation == 'prediction_noise' else None),
                    'timing_context': args.timing_context,
                    'runtime': runtime, 'static_obstacles': static,
                    'dynamic_obstacles': dynamic, 'obstacle_mode': mode,
                    'training_source_map': args.training_map,
                    'navigation_time': 'control steps; physical seconds undefined without control period',
                    'collision_model': _world_class(args).collision_protocol,
                    'is_retrained_controller_ablation': False,
                    'max_global_replans': args.max_steps,
                    'deadlock_definition': 'at most 2 distinct poses in 8 consecutive poses',
                    'deadlock_recovery_definition': 'move >=2 cells and reduce distance to stuck goal >=2 cells or move >=2 cells after subgoal change',
                    'source_sha256': {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                      for name in ('agents.py', 'environment.py', 'global_planner.py',
                                                   'experiments/proposed_evaluation.py', 'navigation_protocol.py',
                                                   'obstacle_prediction.py', 'models.py', 'manuscript_replay.py',
                                                   'experiment_protocol.py', 'configs/levels.json')}}
        protocol_file = out / ('level%d_protocol.json' % level)
        if protocol_file.exists() and json.loads(protocol_file.read_text()) != protocol:
            raise RuntimeError('Protocol differs from existing output: ' + str(protocol_file))
        atomic_json(protocol_file, protocol)
        rows = []
        if record_file.exists():
            for line in record_file.read_text().splitlines():
                rows.append(json.loads(line))
            expected = list(range(args.eval_seed, args.eval_seed + len(rows)))
            if [r['scenario_seed'] for r in rows] != expected:
                raise RuntimeError('Corrupt or duplicate scenario records')
        with record_file.open('a', buffering=1) as f:
            for i in range(len(rows), args.scenarios):
                cfg = generate_environment_config(template, args.eval_seed + i)
                env = _world_class(args)(grid, start=nodes[0], goal=nodes[-1], goal_list=nodes,
                                obs=static, dobs=dynamic, obstacle_mode=mode,
                                max_steps=args.max_steps, robot_radius=args.robot_radius,
                                cell_size=args.cell_size, extra_margin=args.extra_margin)
                row = evaluate_scenario(env, cfg, agent, args.max_steps, args.controller,
                                        keypoint_max_segment=args.keypoint_max_segment,
                                        fixed_period=args.fixed_period,
                                        perturbation=args.perturbation,
                                        localization_sigma=args.localization_sigma,
                                        dump_paths=args.dump_paths)
                f.write(json.dumps(row, allow_nan=False) + '\n')
                rows.append(row)
                if (i + 1) % 25 == 0:
                    print('EVAL map=%s seed=%d L%d %d/%d SR=%.3f' % (
                        args.map_file, args.seed, level, i+1, args.scenarios,
                        sum(r['success'] for r in rows) / len(rows)), flush=True)
        result = {'protocol': protocol, 'summary': summarize(rows)}
        atomic_json(out / ('level%d_summary.json' % level), result)
        results.append(result)
    atomic_json(out / 'evaluation_complete.json', results)
    return results


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--map-file', required=True)
    p.add_argument('--training-map', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--seed', type=int, default=0, help='Evaluation RNG seed; training seed is read from checkpoint')
    p.add_argument('--scenarios', type=int, default=300)
    p.add_argument('--eval-seed', type=int, default=10000)
    p.add_argument('--max-steps', type=int, default=2000)
    p.add_argument('--levels', type=int, nargs='+', choices=LEVELS, default=list(LEVELS))
    p.add_argument('--controller', choices=CONTROLLERS, default='full')
    p.add_argument('--robot-radius', type=float, default=0.5)
    p.add_argument('--cell-size', type=float, nargs=2, default=[1., 1.])
    p.add_argument('--extra-margin', type=float, default=0.)
    p.add_argument('--agent-kind', choices=('proposed',), default='proposed')
    p.add_argument('--replay-mode', choices=('manuscript', 'uniform', 'per'), default='manuscript')
    p.add_argument('--architecture', choices=('bilstm_dueling', 'bilstm_plain', 'unilstm_dueling', 'unilstm_plain', 'plain'), default='bilstm_dueling')
    p.add_argument('--allow-historical-checkpoint', action='store_true')
    p.add_argument('--levels-config', default='configs/levels.json',
                   help='JSON overriding the L1-L4 obstacle counts; must be the same '
                        'file the training curriculum was built from')
    p.add_argument('--synced-obstacles', action='store_true', default=True)
    p.add_argument('--fixed-period' , type=int, default=20)
    p.add_argument('--keypoint-max-segment', type=float, default=None,
                   help='Cap the Algorithm 2 anchor-to-waypoint span to sweep sub-goal '
                        'density as a controlled factor; omit for the released '
                        'longest-safe-segment compression')
    p.add_argument('--perturbation', choices=('ideal', 'localization', 'sensing', 'delay', 'combined', 'axis_only',
                                              'prediction_noise'), default='ideal')
    p.add_argument('--prediction-noise', type=float, default=.2,
                   help='Under --perturbation prediction_noise: probability that each predicted '
                        'landing cell is replaced by a uniformly drawn free neighbour of the '
                        'obstacle (a wrong-direction prediction). Evaluation only; the policy '
                        'and its training are unchanged.')
    p.add_argument('--localization-sigma', type=float, default=.25,
                   help='Std dev in cells of the distance-observation noise under the '
                        'localization and combined perturbations. Azimuth noise scales with it.')
    p.add_argument('--dump-paths', action='store_true',
                   help='Record the position and active-sub-goal sequence per scenario. '
                        'Diagnostic only; inflates the record by three orders of magnitude.')
    p.add_argument('--timing-context', choices=('shared_machine', 'sequential'), default='shared_machine')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output-dir', required=True)
    return p


if __name__ == '__main__':
    run(build_parser().parse_args())
