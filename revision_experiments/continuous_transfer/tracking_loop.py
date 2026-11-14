"""Continuous closed loop with a low-level tracking controller (groups C0-C5).

The DRL policy and its weights are unchanged. What changes is how one of its
nine discrete actions is executed.

Grid execution (G0):      policy -> action -> cell update (x+1, y+1)
Event execution (C1-C5):  policy -> action -> target point -> holonomic tracking
                          -> planar velocity vector every 0.02 s -> pose -> feedback

1. The policy outputs one of nine actions.
2. The action becomes a target on the lattice, relative to the robot's current
   cell: up-right is (x+1, y+1). The target is anchored to the cell the robot
   believes it occupies, so the robot stays on the lattice the policy was
   trained on, rather than accumulating sub-cell drift.
3. The controller recomputes a velocity command every substep from the measured
   pose and the target.
4. The state advances every 0.02 s and the robot body is tested for
   intersection at every substep.
5. The next action is requested only when the measured pose is within the
   action tolerance of the target. An action that does not arrive within the
   action timeout is recorded as an execution timeout, and the policy is
   re-queried from wherever the robot is -- the predefined recovery.

Fixed-period execution (C0) is the earlier interface, kept for diagnosis: the
action becomes a target one cell from the measured pose, the controller brakes
to it, and the next action arrives after exactly one period whether or not the
target was reached.

The execution layer is a **simplified holonomic mobile robot model with
bounded translational velocity and acceleration**, not a model of real robot
dynamics. Its scope:

  * circular footprint of radius 0.2 cell;
  * translational speed bounded by 1.0 cell/s, acceleration by 1.0 cell/s^2,
    both enforced on the velocity vector at every substep;
  * continuous pose integrated and solid-body collision checked every 0.02 s;
  * no wheel slip, no motor or drivetrain dynamics, no actuator deadband, no
    sensor field-of-view limit, no hardware in the loop.

An omnidirectional base is the matched choice here, because the policy has no
heading state -- the local window is map-aligned and the goal feature is an
absolute bearing -- so its nine-action space (eight directions plus stop) is
holonomic by construction.

Two principles are borrowed from the ROS 2 Nav2 Regulated Pure Pursuit
controller: linear-velocity regulation by proximity to sensed obstacles, and a
footprint collision check along the commanded motion before committing to it.
The pure-pursuit mechanism itself is NOT implemented and must not be claimed:
there is no path with a sliding lookahead point (the reference is one waypoint),
no pure-pursuit curvature law, no curvature-based speed regulation, no adaptive
lookahead distance and no rotate-to-heading behaviour. For a holonomic base the
curvature law does not apply at all. Describe it as inspired by those two
principles, never as "the Nav2 RPP controller".

Both interfaces share the collision test, the obstacle clock, the observation
pipeline, the noise models and the time limits, so a C0-to-C1 difference is due
to the action interface alone.
"""
from __future__ import annotations

from collections import deque
import math

import numpy as np

from revision_experiments.continuous_transfer.geometry import segment_box_clearance

#: Measured pose within this distance of the target completes an action.
ACTION_TOLERANCE = .1

#: Linear speed is regulated below its maximum only when the body's clearance to
#: a sensed obstacle is under this distance. 0.3 cell is the clearance a robot of
#: radius 0.2 has to a wall in the adjacent cell when it is at its own cell
#: centre, so regulation starts only when the robot is closer than lattice motion
#: would put it.
REGULATION_DISTANCE = .3

#: Floor of the regulated linear speed, cells/s.
REGULATION_MIN_SPEED = .25


def clamp_norm(vector, limit):
    norm = float(np.linalg.norm(vector))
    return vector if norm <= limit or norm == 0. else vector * (limit / norm)


def point_segment_distance(point, start, end):
    direction = end - start
    length = float(direction @ direction)
    if length == 0.:
        return float(np.linalg.norm(point - start))
    t = min(1., max(0., float((point - start) @ direction) / length))
    return float(np.linalg.norm(point - (start + t * direction)))


def disc_clearances(point, centres, half=.5):
    """Distance from a point to each axis-aligned unit cell; 0 inside a cell."""
    if not len(centres):
        return np.empty(0)
    excess = np.maximum(np.abs(np.asarray(centres, dtype=float) - point) - half, 0.)
    return np.linalg.norm(excess, axis=1)


class RegulatedHolonomicController:
    """Bounded holonomic point tracker with proximity speed regulation.

    Not a pure-pursuit controller: the reference is a single waypoint, there is
    no path lookahead and no curvature law. Only the proximity regulation and
    the footprint pre-check are taken from Nav2 Regulated Pure Pursuit.

    Speed command, before the acceleration limit:

        v = min(v_max, sqrt(2 a_max d), v_reg)

    The `sqrt(2 a_max d)` term keeps the robot able to stop at the target. The
    next action may be 'stay', which the controller cannot know in advance, so
    it does not commit beyond the target. It arrives in the tolerance region at
    `sqrt(2 a_max tol)` = 0.45 cell/s and keeps that momentum into the next
    action. `v_reg` scales linearly from `v_max` at `REGULATION_DISTANCE` of
    body clearance down to `REGULATION_MIN_SPEED`.

    The velocity change per substep is bounded by `a_max * dt` as a vector, so
    the base is bounded in speed and acceleration but not in heading.
    """

    def __init__(self, max_speed, max_acceleration, radius, regulate=True,
                 regulation_distance=REGULATION_DISTANCE, min_speed=REGULATION_MIN_SPEED):
        self.max_speed = float(max_speed)
        self.max_acceleration = float(max_acceleration)
        self.radius = float(radius)
        self.regulate = bool(regulate)
        self.regulation_distance = float(regulation_distance)
        self.min_speed = float(min_speed)

    def speed_cap(self, measured, sensed_centres):
        if not self.regulate or not len(sensed_centres):
            return self.max_speed
        clearance = float(disc_clearances(measured, sensed_centres).min()) - self.radius
        if clearance >= self.regulation_distance:
            return self.max_speed
        return max(self.min_speed,
                   self.max_speed * max(clearance, 0.) / self.regulation_distance)

    def desired(self, measured, target, sensed_centres):
        if target is None:
            return np.zeros(2)
        delta = target - measured
        distance = float(np.linalg.norm(delta))
        if distance < 1e-12:
            return np.zeros(2)
        speed = min(self.max_speed,
                    math.sqrt(2. * self.max_acceleration * distance),
                    self.speed_cap(measured, sensed_centres))
        return speed * delta / distance

    def step(self, velocity, measured, target, sensed_centres, dt):
        change = self.desired(measured, target, sensed_centres) - velocity
        return velocity + clamp_norm(change, self.max_acceleration * dt)

    def path_blocked(self, start, target, sensed_centres, resolution=.02):
        """Footprint check along the segment to the target, on sensed occupancy.

        The principle is Nav2 RPP's: project the footprint along the motion
        about to be commanded and refuse it if the body would hit something the
        robot can see. Here that motion is one straight segment rather than an
        arc along a path.

        A cell blocks the segment only if the segment brings the body closer to
        that cell than the radius AND closer than the body already is. Without
        the second condition, a robot whose measured pose sits near a wall --
        which localization error produces -- could never move away from it,
        because every segment starting there would read as blocked.
        """
        if not len(sensed_centres):
            return False
        centres = np.asarray(sensed_centres, dtype=float)
        length = float(np.linalg.norm(np.asarray(target) - np.asarray(start)))
        count = max(2, int(math.ceil(length / resolution)) + 1)
        samples = start + np.linspace(0., 1., count)[:, None] * (np.asarray(target) - start)
        excess = np.maximum(np.abs(samples[:, None, :] - centres[None, :, :]) - .5, 0.)
        per_cell = np.linalg.norm(excess, axis=2)          # (samples, cells)
        closest = per_cell.min(axis=0)
        initial = per_cell[0]
        return bool(np.any((closest < self.radius) & (closest < initial - 1e-9)))


class ObstacleClock:
    """Obstacle positions as a function of physical time only.

    Frames come out of `ObstacleDriver` in chunks and are consumed one substep at
    a time. Splitting execution into actions of different durations therefore
    never adds or removes obstacle motion. The obstacles cannot tell where one
    action ends and the next begins.
    """

    def __init__(self, driver, dt, chunk=50):
        self.driver, self.dt, self.chunk = driver, float(dt), int(chunk)
        frames = driver.sample_period(self.chunk, self.dt)
        self.current = frames[0]
        self.buffer = deque(frames[1:])
        self.ticks = 0

    def tick(self):
        if not self.buffer:
            self.buffer.extend(self.driver.sample_period(self.chunk, self.dt)[1:])
        self.current = self.buffer.popleft()
        self.ticks += 1
        return self.current

    def observed_cells(self):
        return [tuple(int(v) for v in np.rint(p)) for p in self.current]


class LocalizationError:
    """Pose-estimate error as a first-order Gauss-Markov process.

    Stationary standard deviation `sigma` per axis, correlation time `tau`.
    A localization filter's error drifts slowly; it is not white at 50 Hz.
    Injecting white noise at the control rate would make the controller chase
    the noise, and the resulting jitter would come from the error model rather
    than from localization. `tau = 0` selects white noise anyway.
    """

    def __init__(self, sigma, tau, dt, rng):
        self.sigma, self.rng = float(sigma), rng
        self.phi = 0. if tau <= 0 else math.exp(-dt / float(tau))
        self.value = rng.normal(0., self.sigma, 2) if self.sigma > 0 else np.zeros(2)

    def advance(self):
        if self.sigma <= 0:
            return
        self.value = (self.phi * self.value
                      + math.sqrt(1. - self.phi ** 2) * self.rng.normal(0., self.sigma, 2))


class OccupancySensor:
    """The robot's 7x7 occupancy reading, shared by the policy and the controller.

    A reading holds the true occupancy -- static cells and the cells that contain
    the current dynamic obstacle centres -- around the measured cell. Each cell
    is flipped independently with probability `flip` on every reading, the same
    model the grid perturbation table applies to the policy input.
    """

    def __init__(self, static_grid, flip, rng, half=3):
        self.static, self.flip, self.rng, self.half = static_grid, float(flip), rng, half

    def read(self, measured_cell, dynamic_cells):
        rows, cols = self.static.shape
        r, c = measured_cell
        window = np.ones((2 * self.half + 1, 2 * self.half + 1), dtype=bool)
        r0, r1 = max(0, r - self.half), min(rows, r + self.half + 1)
        c0, c1 = max(0, c - self.half), min(cols, c + self.half + 1)
        window[r0 - r + self.half:r1 - r + self.half,
               c0 - c + self.half:c1 - c + self.half] = self.static[r0:r1, c0:c1] != 0
        for dr, dc in dynamic_cells:
            i, j = dr - r + self.half, dc - c + self.half
            if 0 <= i < window.shape[0] and 0 <= j < window.shape[1]:
                window[i, j] = True
        if self.flip > 0:
            window ^= self.rng.random(window.shape) < self.flip
        offsets = np.argwhere(window) - self.half
        return offsets.astype(float) + np.asarray(measured_cell, dtype=float)


def static_hit(position, static_grid, radius):
    """Exact disc-versus-occupied-cell test; outside the map counts as occupied."""
    rows, cols = static_grid.shape
    reach = .5 + radius
    for i in range(int(math.ceil(position[0] - reach)), int(math.floor(position[0] + reach)) + 1):
        for j in range(int(math.ceil(position[1] - reach)), int(math.floor(position[1] + reach)) + 1):
            if 0 <= i < rows and 0 <= j < cols and static_grid[i, j] == 0:
                continue
            excess = np.maximum(np.abs(position - (i, j)) - .5, 0.)
            if float(np.hypot(*excess)) < radius:
                return True
    return False


def dynamic_hit(position, frame, radius):
    """Exact disc-versus-unit-square test against every dynamic obstacle."""
    if not len(frame):
        return False
    excess = np.maximum(np.abs(frame - position) - .5, 0.)
    return bool((np.linalg.norm(excess, axis=1) < radius).any())


def rest_to_rest_time(distance, max_speed, max_acceleration):
    """Shortest time to cover `distance` from rest to rest under the limits."""
    if distance >= max_speed ** 2 / max_acceleration:
        return distance / max_speed + max_speed / max_acceleration
    return 2. * math.sqrt(distance / max_acceleration)


def run_episode(env, cfg, agent, args, trace=None):
    """One scenario. Returns a record; never raises on a navigation outcome.

    Pass a list as `trace` to collect one entry per substep for rendering. The
    scoring path is identical with or without it.
    """
    import time
    from generate_test_environments import apply_environment_config
    from global_planner import arrival_budget
    from navigation_protocol import ACTIONS
    from experiments.proposed_evaluation import reset_policy
    from revision_experiments.continuous_transfer.closed_loop import advance_obstacle_script
    from revision_experiments.continuous_transfer.obstacle_motion import ObstacleDriver

    wall_start = time.perf_counter()
    apply_environment_config(env, cfg)
    env.navigation_mode = 'event'
    for key, value in getattr(agent, 'loaded_environment_specification', {}).items():
        if key in ('prediction_mode', 'prediction_enabled', 'avoidance_reward_form',
                   'reward_scales', 'subgoal_tolerance'):
            setattr(env, key, value)
    env.reset(resample_obstacles=False)
    reset_policy(agent)
    agent.env = env

    seed = int(cfg['seed'])
    dt, radius = args.dt, args.robot_radius
    event = args.interface == 'event'
    # Separate streams so that turning one perturbation on does not shift the
    # draws of another; C5 sees exactly the C2 pose error and the C3 flips.
    localization = LocalizationError(args.localization_sigma, args.localization_tau, dt,
                                     np.random.default_rng(seed + 700001))
    sensor = OccupancySensor(env.base_grid_map, args.sensing_flip_probability,
                             np.random.default_rng(seed + 700002))
    policy_flips = np.random.default_rng(seed + 700003)
    controller = RegulatedHolonomicController(args.max_linear_speed, args.max_linear_acceleration,
                                              radius, regulate=event and not args.no_regulation)
    driver = ObstacleDriver(env, advance_script=advance_obstacle_script, speed=args.obstacle_speed,
                            lateral_acceleration=args.obstacle_lateral_acceleration)
    clock = ObstacleClock(driver, dt)
    grid = env.base_grid_map
    rows, cols = grid.shape

    period_substeps = int(round(args.period / dt))
    timeout_substeps = int(math.ceil(args.action_timeout / dt))
    stay_substeps = int(round(args.stay_duration / dt))
    sensor_substeps = max(1, int(round(1. / (args.sensor_rate * dt))))
    time_cap = int(round(args.max_sim_time / dt))

    position = np.asarray(env.start, dtype=float)
    velocity = np.zeros(2)
    pending = deque([8] * args.action_delay)
    substeps = decisions = 0
    moves = completed = timed_out = refused = 0
    track_sum = track_count = arrival_sum = arrival_count = 0.
    path_length = 0.
    termination = None

    def measured():
        return position + localization.value

    def measured_cell(point):
        return tuple(int(v) for v in np.clip(np.rint(point), [0, 0], [rows - 1, cols - 1]))

    def observe(point):
        cell = measured_cell(point)
        env.agent_pos = cell
        env.steps = decisions
        lattice = [tuple(o['position']) for o in env.dynamic_obstacles]
        for obstacle, seen in zip(env.dynamic_obstacles, clock.observed_cells()):
            obstacle['position'] = seen
        env.rebuild_grid_map()
        state = np.array(env.get_state(), dtype=np.float32, copy=True)
        for obstacle, head in zip(env.dynamic_obstacles, lattice):
            obstacle['position'] = head
        offset = np.asarray(env.goal, dtype=float) - point
        state[-1, :2] = [np.linalg.norm(offset), np.arctan2(offset[1], offset[0])]
        if args.sensing_flip_probability > 0:
            # Same flip model as the grid perturbation table: every cell of the
            # stacked observation, redrawn at each decision, policy input only.
            cells = state[:, 2:]
            mask = policy_flips.random(cells.shape) < args.sensing_flip_probability
            cells[mask] = np.where(cells[mask] == 0, 1., 0.)
        return state, cell

    if env.planning_events and not env.planning_events[0]['accepted']:
        termination = 'initial_planning_failure'
    goal_age = 0
    budget = arrival_budget(np.asarray(env.start, dtype=float), env.goal)

    while termination is None:
        if substeps >= time_cap:
            termination = 'timeout_physical'; break
        if decisions >= args.max_decisions:
            termination = 'timeout_decisions'; break
        estimate = measured()
        state, anchor = observe(estimate)
        action = int(agent.select_action(state))
        decisions += 1
        if args.action_delay:
            pending.append(action)
            action = pending.popleft()

        sensed = sensor.read(anchor, clock.observed_cells())
        if event:
            anchor_point = np.asarray(anchor, dtype=float)
            reference = anchor_point
            if action == 8:
                kind, target, window = 'stay', anchor_point, stay_substeps
            else:
                target = anchor_point + np.asarray(ACTIONS[action], dtype=float)
                if not args.no_path_check and controller.path_blocked(estimate, target, sensed):
                    kind, target, window = 'refused', anchor_point, stay_substeps
                    refused += 1
                else:
                    kind, window = 'move', timeout_substeps
        else:
            reference = estimate.copy()
            window = period_substeps
            if action == 8:
                kind, target = 'stay', None
            else:
                kind, target = 'move', estimate + np.asarray(ACTIONS[action], dtype=float)
        moves += kind == 'move'

        reached = False
        for k in range(window):
            if substeps >= time_cap:
                break
            if k and k % sensor_substeps == 0:
                sensed = sensor.read(measured_cell(measured()), clock.observed_cells())
            velocity = controller.step(velocity, measured(), target, sensed, dt)
            previous = position
            position = position + velocity * dt
            path_length += float(np.linalg.norm(position - previous))
            substeps += 1
            frame = clock.tick()
            localization.advance()
            hit = ('static' if static_hit(position, grid, radius) else
                   'dynamic' if dynamic_hit(position, frame, radius) else None)
            if trace is not None:
                trace.append({'t': substeps * dt, 'position': position.copy(),
                              'velocity': velocity.copy(), 'obstacles': frame.copy(),
                              'target': None if target is None else np.asarray(target).copy(),
                              'kind': kind, 'action': action, 'decision': decisions,
                              'route': [tuple(g) for g in env.goal_list[env.current_goal_index:]],
                              'collision': hit})
            if hit:
                termination = 'collision_' + hit; break
            if kind == 'move':
                track_sum += point_segment_distance(position, reference, target)
                track_count += 1
                if not reached and np.linalg.norm(measured() - target) <= args.action_tolerance:
                    reached = True
                    arrival_sum += float(np.linalg.norm(position - target)); arrival_count += 1
                    if event:
                        break
        if kind == 'move' and termination is None:
            if reached:
                completed += 1
            else:
                timed_out += 1
                arrival_sum += float(np.linalg.norm(position - target)); arrival_count += 1
        if termination is not None:
            break

        # Sub-goal progress and the Eq. (6) budget, both counted in decisions.
        estimate = measured()
        env.agent_pos = measured_cell(estimate)
        env.steps = decisions
        if np.linalg.norm(estimate - np.asarray(env.goal, dtype=float)) <= env.subgoal_tolerance:
            if env.current_goal_index + 1 == len(env.goal_list):
                truth = np.linalg.norm(position - np.asarray(env.goal, dtype=float))
                termination = 'success' if truth <= env.subgoal_tolerance else 'false_arrival'
                break
            env.current_goal_index += 1
            env.goal = env.goal_list[env.current_goal_index]
            goal_age = 0
            budget = arrival_budget(estimate, env.goal)
        else:
            goal_age += 1
            if goal_age >= budget:
                if not env._plan_route():
                    termination = 'planning_failure'; break
                goal_age = 0
                budget = arrival_budget(estimate, env.goal)

    seconds = substeps * dt
    return {'scenario_seed': seed, 'interface': args.interface, 'termination': termination,
            'success': termination == 'success', 'decisions': decisions,
            'simulated_time_s': seconds,
            'navigation_time_s': seconds if termination == 'success' else None,
            'path_length_cells': path_length,
            'moves_issued': moves, 'moves_completed': completed, 'moves_timed_out': timed_out,
            'moves_refused': refused,
            'action_completion_rate': completed / moves if moves else None,
            'mean_tracking_error': track_sum / track_count if track_count else None,
            'mean_arrival_error': arrival_sum / arrival_count if arrival_count else None,
            'global_replans': env.global_replans,
            'planning_seconds': sum(e['seconds'] for e in env.planning_events),
            'wall_time_s': time.perf_counter() - wall_start}


def summarize(rows):
    """The seven reported metrics, plus the splits that explain them."""
    n = len(rows)
    count = lambda *names: sum(r['termination'] in names for r in rows)
    pooled = lambda key: sum(r[key] for r in rows)
    def mean_of(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(values)) if values else None
    moves = pooled('moves_issued')
    return {'scenarios': n,
            'success_rate': count('success') / n,
            'collision_rate': count('collision_static', 'collision_dynamic') / n,
            'collision_static_rate': count('collision_static') / n,
            'collision_dynamic_rate': count('collision_dynamic') / n,
            'timeout_rate': count('timeout_physical', 'timeout_decisions') / n,
            'timeout_physical_rate': count('timeout_physical') / n,
            'timeout_decisions_rate': count('timeout_decisions') / n,
            'planning_failure_rate': count('planning_failure', 'initial_planning_failure') / n,
            'false_arrival_rate': count('false_arrival') / n,
            # Pooled over every issued move, so long episodes weigh by their
            # number of actions rather than counting once each.
            'action_completion_rate': pooled('moves_completed') / moves if moves else None,
            'action_timeout_rate': pooled('moves_timed_out') / moves if moves else None,
            'refused_per_episode': pooled('moves_refused') / n,
            'mean_tracking_error': mean_of('mean_tracking_error'),
            'mean_arrival_error': mean_of('mean_arrival_error'),
            'navigation_time_s_success': mean_of('navigation_time_s'),
            'mean_decisions': mean_of('decisions'),
            'mean_simulated_time_s': mean_of('simulated_time_s'),
            'terminations': {t: count(t) for t in sorted({r['termination'] for r in rows})}}


def build_parser():
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--map-file', default='grid_map2.txt')
    p.add_argument('--levels-config', required=True)
    p.add_argument('--level', type=int, default=2)
    p.add_argument('--scenarios', type=int, default=300)
    p.add_argument('--eval-seed', type=int, default=10000)
    p.add_argument('--interface', choices=('fixed', 'event'), required=True,
                   help="'fixed' is C0, the earlier interface: target one cell from the "
                        "measured pose, braking controller, next action after exactly one period. "
                        "'event' is C1-C5: lattice target, regulated controller, footprint check, "
                        'next action on arrival or on action timeout.')
    p.add_argument('--localization-sigma', type=float, default=0.)
    p.add_argument('--localization-tau', type=float, default=1.,
                   help='Correlation time in s of the pose error; 0 gives white noise.')
    p.add_argument('--sensing-flip-probability', type=float, default=0.)
    p.add_argument('--action-delay', type=int, default=0)
    p.add_argument('--obstacle-speed', type=float, default=.5)
    p.add_argument('--obstacle-lateral-acceleration', type=float, default=1.)
    p.add_argument('--robot-radius', type=float, default=.2)
    p.add_argument('--dt', type=float, default=.02)
    p.add_argument('--period', type=float, default=1.42,
                   help='Decision period of the fixed interface, s; sqrt(2)/v_max rounded up.')
    p.add_argument('--max-linear-speed', type=float, default=1.)
    p.add_argument('--max-linear-acceleration', type=float, default=1.)
    p.add_argument('--action-tolerance', type=float, default=ACTION_TOLERANCE)
    p.add_argument('--action-timeout', type=float, default=None,
                   help='Seconds before an unfinished move is abandoned. Default: twice the '
                        'rest-to-rest time of the longest action, a diagonal of sqrt(2) cells.')
    p.add_argument('--stay-duration', type=float, default=1.,
                   help="Duration of 'stay' and of a refused move under the event interface, s: "
                        'the time to cross one axial cell at v_max. Without a minimum, a stay '
                        'would take no physical time and the policy would be re-queried before '
                        'any obstacle moved.')
    p.add_argument('--sensor-rate', type=float, default=10.,
                   help='Refresh rate of the occupancy reading the controller uses, Hz.')
    p.add_argument('--max-decisions', type=int, default=2000,
                   help='The grid evaluator\'s cap, kept so the decision budget is unchanged.')
    p.add_argument('--max-sim-time', type=float, default=2840.,
                   help='Physical time cap, s, the same for every group: 2000 decisions at the '
                        'fixed period of 1.42 s. Execution time cannot be traded for success.')
    p.add_argument('--no-regulation', action='store_true')
    p.add_argument('--no-path-check', action='store_true')
    p.add_argument('--replay-mode', choices=('manuscript', 'uniform', 'per'), default='manuscript')
    p.add_argument('--architecture', default='bilstm_dueling')
    p.add_argument('--allow-historical-checkpoint', action='store_true')
    p.add_argument('--device', default='cpu')
    p.add_argument('--output-dir', required=True)
    return p


def main():
    import hashlib
    import json
    import os
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from experiments import proposed_evaluation as pe
    from generate_test_environments import MAP_NODES, generate_environment_config
    from revision_experiments.legacy_arm.synced_env import SyncedGridWorld
    from train import set_global_seed

    parser = build_parser()
    args = parser.parse_args()
    if args.action_timeout is None:
        args.action_timeout = 2. * rest_to_rest_time(math.sqrt(2.), args.max_linear_speed,
                                                     args.max_linear_acceleration)
    if args.period * args.max_linear_speed < math.sqrt(2.) - 1e-9:
        parser.error('fixed period too short to complete a diagonal action')
    out = Path(args.output_dir)
    if (out / 'results.json').exists():
        parser.error('use a new output directory to preserve existing evidence')
    out.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.eval_seed)
    levels = pe.load_levels(args.levels_config)
    static, dynamic, mode = levels[args.level]
    grid = np.loadtxt(args.map_file, dtype=int)
    nodes = MAP_NODES[os.path.basename(args.map_file)]
    checkpoint = Path(args.checkpoint).resolve()
    base = pe.GridWorld(grid, start=nodes[0], goal=nodes[-1], goal_list=nodes)
    agent = pe.RainbowDQN(env=base, replay_mode=args.replay_mode,
                          architecture=args.architecture, device=args.device)
    agent.load_model(str(checkpoint), allow_historical=args.allow_historical_checkpoint)
    template = pe.GridWorld(grid, start=nodes[0], goal=nodes[-1], goal_list=nodes,
                            obs=static, dobs=dynamic, obstacle_mode=mode,
                            block_size_range=(2, 5), max_steps=args.max_decisions)
    rows = []
    with (out / ('level%d.jsonl' % args.level)).open('w', buffering=1) as handle:
        for index in range(args.scenarios):
            cfg = generate_environment_config(template, args.eval_seed + index)
            env = SyncedGridWorld(grid, start=nodes[0], goal=nodes[-1], goal_list=nodes,
                                  obs=static, dobs=dynamic, obstacle_mode=mode,
                                  max_steps=args.max_decisions, robot_radius=args.robot_radius)
            row = run_episode(env, cfg, agent, args)
            handle.write(json.dumps(row, allow_nan=False) + '\n')
            rows.append(row)
            if (index + 1) % 25 == 0:
                s = summarize(rows)
                print('L%d %s %d/%d SR=%.3f CR=%.3f TO=%.3f ACR=%.3f' % (
                    args.level, args.interface, index + 1, args.scenarios, s['success_rate'],
                    s['collision_rate'], s['timeout_rate'], s['action_completion_rate'] or 0.),
                    flush=True)
    result = {'protocol': vars(args), 'checkpoint': str(checkpoint),
              'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'collision_model': 'disc of radius r vs unit cell squares, static and dynamic, '
                                 'exact intersection at every %g s substep' % args.dt,
              'obstacle_clock': 'physical time; shared lattice path with the grid evaluator',
              'scope': 'kinematic, omnidirectional, simulation only',
              'summary': summarize(rows)}
    (out / 'results.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
