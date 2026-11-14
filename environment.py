import numpy as np
import random
from collections import deque
import torch
import os
import seaborn as sns
import pickle
from navigation_protocol import ACTION_SETS, ACTIONS, action_transition_matrix
from obstacle_prediction import landing_candidates
from global_planner import arrival_budget, grid_astar_path, segment_is_safe

class GridWorld:
    reward_includes_smoothness = True
    collision_protocol = 'github-endpoint-only-v1'
    def __init__(self, grid_map, start, goal, max_steps=2000, goal_list=None, obs=30, dobs=10,
                 obstacle_mode='point', block_size_range=(2, 5), *,
                 robot_radius=0.5, cell_size=(1.0, 1.0), extra_margin=0.0,
                 subgoal_tolerance=1.0, smoothing_factor=0.5,
                 navigation_mode='fixed_waypoints', prediction_enabled=True, prediction_mode='history_voting',
                 avoidance_reward_form='piecewise', reward_scales=(1., 1., 1.),
                 motion_model='grid', reward_pricing='per_step',
                 avoid_reference='centre'):
        """
            Initialize GridWorld Environment

            Args:
            grid_map: Base map
            start: Starting point
            goal: Goal point
            max_steps: Maximum number of steps
            goal_list: List of path nodes
            obs: Number of static obstacles (count, not pixels)
            dobs: Number of dynamic obstacles
            obstacle_mode: Obstacle mode
            - 'point': Single-point distribution (default)
            - 'block': Block distribution (similar to Tetris)
            block_size_range: Size range of block obstacles (min, max), only effective in block mode
        """
        self.original_grid_map = grid_map.copy()
        self.base_grid_map = grid_map.copy()
        self.grid_map = grid_map.copy()

        self.rows, self.cols = grid_map.shape
        self.max_steps = max_steps
        raw_goal_list = list(goal_list) if goal_list else []
        # Public examples historically passed [start, ..., goal].  The start
        # is an initial pose, not a target; normalize it once so the first
        # learned sub-goal is the first waypoint after the start.
        if raw_goal_list and start is not None and tuple(raw_goal_list[0]) == tuple(start):
            raw_goal_list = raw_goal_list[1:]
        self.goal_list = [tuple(p) for p in raw_goal_list]
        self._reset_goal_list = self.goal_list.copy()
        self.final_goal = tuple(raw_goal_list[-1]) if raw_goal_list else tuple(goal)
        self.robot_radius = float(robot_radius)
        self.cell_size = tuple(map(float, cell_size))
        self.extra_margin = float(extra_margin)
        if (not np.isfinite([self.robot_radius, self.extra_margin, *self.cell_size]).all()
                or min(self.robot_radius, self.extra_margin) < 0 or min(self.cell_size) <= 0):
            raise ValueError('invalid robot footprint or cell dimensions')
        if motion_model not in ACTION_SETS:
            raise ValueError("motion_model must be 'grid' or 'unit'")
        # 'grid': axial action moves 1, diagonal moves sqrt(2), pose is a cell.
        # 'unit': every action moves 1, diagonal moves 1/sqrt(2) per axis and
        # the pose is continuous; the occupancy map stays cell-based.
        self.motion_model = motion_model
        self.action_set = ACTION_SETS[motion_model]
        if reward_pricing not in ('per_step', 'per_period'):
            raise ValueError("reward_pricing must be 'per_step' or 'per_period'")
        if avoid_reference not in ('centre', 'extent'):
            raise ValueError("avoid_reference must be 'centre' or 'extent'")
        # 'per_step' is Eqs. (10)-(16) as published: the rate terms are summed
        # once per action.  'per_period' weights them by the action's duration
        # max(|delta|, 1) in units of 1/v_max, the Section V-G time base, so a
        # route's total does not depend on how finely the path is discretised.
        self.reward_pricing = reward_pricing
        # 'centre' is as published: distance to the occupied cell centre.
        # 'extent' measures to the cell's axis-aligned box, which is the actual
        # clearance and is direction-dependent rather than a constant offset.
        self.avoid_reference = avoid_reference
        self.subgoal_tolerance = float(subgoal_tolerance)
        if not 0 <= self.subgoal_tolerance < 1.5:
            raise ValueError('subgoal_tolerance must be in [0, 1.5) grid cells')
        self.smoothing_factor = float(smoothing_factor)
        self.action_smoothness_penalty = action_transition_matrix()
        self.navigation_mode = navigation_mode
        self.prediction_enabled = bool(prediction_enabled)
        self.prediction_mode = prediction_mode
        self.avoidance_reward_form = avoidance_reward_form
        self.reward_scales = tuple(map(float, reward_scales))
        if avoidance_reward_form not in ('piecewise', 'linear', 'exponential'):
            raise ValueError('unknown avoidance reward form')
        if len(self.reward_scales) != 3 or not np.isfinite(self.reward_scales).all() or min(self.reward_scales) < 0:
            raise ValueError('reward_scales must contain three finite nonnegative values')
        self.last_action = None
        self.last_info = {}
        self._state_step = None
        self._prediction_step = None
        self.history = deque(maxlen=4)
        self.num_obstacles = obs
        self.num_dobs = dobs

        # Obstacle mode configuration
        self.obstacle_mode = obstacle_mode
        self.block_size_range = block_size_range

        # Add dynamic reward decay
        self.reward_decay = 0.99

        self.direction_vectors = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1),
            'up_left': (-1, -1),
            'up_right': (-1, 1),
            'down_left': (1, -1),
            'down_right': (1, 1)
        }
        self.turn_probability = 0.25
        self.behavior_options = ['random_walk', 'patrol', 'wander']

        # Define block obstacle shape templates
        self.block_shapes = {
            'I': [(0, 0), (0, 1), (0, 2), (0, 3)],           # I-shape
            'O': [(0, 0), (0, 1), (1, 0), (1, 1)],           # O-shape
            'T': [(0, 0), (0, 1), (0, 2), (1, 1)],           # T-shape
            'L': [(0, 0), (1, 0), (2, 0), (2, 1)],           # L-shape
            'J': [(0, 1), (1, 1), (2, 1), (2, 0)],           # J-shape
            'Z': [(0, 0), (0, 1), (1, 1), (1, 2)],           # Z-shape
            'S': [(0, 1), (0, 2), (1, 0), (1, 1)],           # S-shape
            'cross': [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)], # Cross-shape
            'small_rect': [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],  # Small rectangle
            'big_rect': [(i, j) for i in range(3) for j in range(3)]  # Large rectangle 3x3
        }

        if start is not None and goal is not None:
            self.start = start
            self.goal = goal
            self.agent_pos = self.start
            self.distance = np.linalg.norm(np.array(self.start) - np.array(self.goal))
        else:
            self.reset_dynamic()

    def _pose_cell(self, pos):
        """The occupancy cell containing a pose.

        Under the ``grid`` model a pose is already a cell index.  Under the
        ``unit`` model a diagonal action moves 1/sqrt(2) per axis, so the pose
        is continuous and the containing cell is its rounding.
        """
        if self.motion_model == 'grid':
            return tuple(int(v) for v in pos)
        return tuple(int(round(float(v))) for v in pos)

    def _advance(self, pos, delta):
        """Apply one action to a pose, keeping integer poses integer."""
        if self.motion_model == 'grid':
            return tuple(np.asarray(pos) + np.asarray(delta))
        return tuple(np.asarray(pos, dtype=float) + np.asarray(delta, dtype=float))

    def _generate_block_obstacle(self, base_pos, safe_zone, occupied_positions):
        """
        Generate a block obstacle

        Args:
            base_pos: Base position of the block (x, y)
            safe_zone: Area where obstacles are not allowed
            occupied_positions: Set of already occupied positions

        Returns:
            list: List of all positions occupied by the block obstacle, empty if placement is not possible
        """
        shape_name = random.choice(list(self.block_shapes.keys()))
        shape_template = self.block_shapes[shape_name]
        rotation = random.choice([0, 90, 180, 270])
        rotated_shape = []
        for dx, dy in shape_template:
            if rotation == 0:
                rotated_shape.append((dx, dy))
            elif rotation == 90:
                rotated_shape.append((-dy, dx))
            elif rotation == 180:
                rotated_shape.append((-dx, -dy))
            else:  # 270
                rotated_shape.append((dy, -dx))
        block_positions = []
        base_x, base_y = base_pos

        for dx, dy in rotated_shape:
            new_x, new_y = base_x + dx, base_y + dy
            if new_x < 1 or new_x >= self.rows - 1 or new_y < 1 or new_y >= self.cols - 1:
                return []
            if self.original_grid_map[new_x, new_y] == 1:
                return []
            if (new_x, new_y) in safe_zone:
                return []
            if (new_x, new_y) in occupied_positions:
                return []
            block_positions.append((new_x, new_y))

        return block_positions

    def _generate_static_obstacles_point_mode(self, valid_positions, count):
        """Point mode: generate count single-point obstacles"""
        if count > len(valid_positions):
            count = len(valid_positions)

        if count > 0:
            return random.sample(valid_positions, count)
        return []

    def _generate_static_obstacles_block_mode(self, valid_positions, count, safe_zone):
        """
        Block mode: generate count block obstacles

        Args:
            valid_positions: List of available positions
            count: Number of block obstacles to generate
            safe_zone: Safe zone

        Returns:
            list: List of all positions occupied by the block obstacles
        """
        all_obstacle_positions = []
        occupied = set()
        attempts_per_block = 50

        for block_idx in range(count):
            placed = False
            for attempt in range(attempts_per_block):
                if not valid_positions:
                    break
                base_pos = random.choice(valid_positions)
                block_positions = self._generate_block_obstacle(base_pos, safe_zone, occupied)
                if block_positions:
                    all_obstacle_positions.extend(block_positions)
                    occupied.update(block_positions)
                    for pos in block_positions:
                        if pos in valid_positions:
                            valid_positions.remove(pos)

                    placed = True
                    break
            if not placed:
                if valid_positions:
                    fallback_pos = random.choice(valid_positions)
                    all_obstacle_positions.append(fallback_pos)
                    occupied.add(fallback_pos)
                    valid_positions.remove(fallback_pos)
        return all_obstacle_positions

    def reset_dynamic(self, seed=None):
        """
        Args:
            seed: Random seed to control the randomness of environment generation
                  - None (default): Completely random generation, suitable for training to enhance environment diversity
                  - Integer: Use a fixed seed for generation, suitable for testing/evaluation to ensure reproducibility

        Usage recommendations:
            Training: reset_dynamic(seed=None)  # Generate different environments each time
            Testing: reset_dynamic(seed=42)    # Fixed environment for fair comparison
        """
        # If a seed is provided, set the random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.base_grid_map = self.original_grid_map.copy()
        self.dynamic_obstacles = []
        def ensure_tuple(pos):
            if isinstance(pos, (list, np.ndarray)):
                pos = tuple(pos)
            if isinstance(pos, tuple) and len(pos) == 2:
                return pos
            else:
                raise ValueError(f"Invalid position format: {pos}, type: {type(pos)}")
        forbidden_positions = set()
        forbidden_positions.add(ensure_tuple(self.start))
        forbidden_positions.add(ensure_tuple(self.goal))
        if self.goal_list:
            for goal_point in self.goal_list:
                forbidden_positions.add(ensure_tuple(goal_point))
        safe_zone = set()
        for fx, fy in forbidden_positions:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    safe_zone.add((fx + dx, fy + dy))
        valid_positions_for_static = [
            (i, j) for i in range(1, self.rows - 1) for j in range(1, self.cols - 1)
            if self.original_grid_map[i, j] == 0 and (i, j) not in safe_zone
        ]
        if self.obstacle_mode == 'point':
            random_obstacles = self._generate_static_obstacles_point_mode(
                valid_positions_for_static.copy(),
                self.num_obstacles
            )
        elif self.obstacle_mode == 'block':
            random_obstacles = self._generate_static_obstacles_block_mode(
                valid_positions_for_static.copy(),
                self.num_obstacles,
                safe_zone
            )
        else:
            raise ValueError(f"Unknown obstacle mode: {self.obstacle_mode}, please use 'point' or 'block'")
        occupied_by_static = set(random_obstacles)
        for x, y in random_obstacles:
            self.base_grid_map[x, y] = 1
        # ========== Generate dynamic obstacles ==========
        valid_positions_for_dynamic = [
            (i, j) for i in range(1, self.rows - 1) for j in range(1, self.cols - 1)
            if self.base_grid_map[i, j] == 0 and (i, j) not in safe_zone
        ]

        for obs_index in range(self.num_dobs):
            if not valid_positions_for_dynamic:
                break

            x, y = random.choice(valid_positions_for_dynamic)
            valid_positions_for_dynamic.remove((x, y))

            behavior = random.choice(self.behavior_options)
            direction = random.choice(list(self.direction_vectors.keys()))
            speed = random.randint(1, 2)

            obstacle = {
                'position': (x, y),
                'direction': direction,
                'speed': speed,
                'history': deque([None] * 4, maxlen=4),
                'behavior': behavior,
                'steps_since_turn': 0,
                'turn_interval': random.randint(4, 10)
            }
            # For patrol behavior, select patrol points (from currently available valid positions)
            if behavior == 'patrol' and len(valid_positions_for_dynamic) >= 2:
                patrol_points = random.sample(valid_positions_for_dynamic, min(2, len(valid_positions_for_dynamic)))
                obstacle['waypoints'] = patrol_points
                obstacle['waypoint_index'] = 0

            self.dynamic_obstacles.append(obstacle)

        self.obstacle_history = {i: deque(maxlen=4) for i in range(len(self.dynamic_obstacles))}

        self.kalman_filters = {}
        for i in range(len(self.dynamic_obstacles)):
            self.kalman_filters[i] = {
                'state': np.zeros(4),
                'covariance': np.eye(4) * 0.1,
                'initialized': False
            }
        self.valid_positions = [
            (i, j) for i in range(1, self.rows - 1) for j in range(1, self.cols - 1)
            if self.base_grid_map[i, j] == 0
        ]
        self.agent_pos = self.start
        self.steps = 0
        self.distance = np.linalg.norm(np.array(self.start) - np.array(self.goal))
        # rebuild the working grid_map that combines base (static) and dynamic obstacles
        self.rebuild_grid_map()

    def reset(self, resample_obstacles=None, seed=None):
        # A fresh environment has no sampled scene yet, even when a caller
        # requests preservation rather than resampling of an existing scene.
        if resample_obstacles or not hasattr(self, 'dynamic_obstacles'):
            self.reset_dynamic(seed=seed)
        self.agent_pos = tuple(self.start)
        self.steps = 0
        self.goal_list = self._reset_goal_list.copy()
        self.goal_list_copy = self.goal_list.copy() if self.goal_list else []
        self.current_goal_index = 0
        if self.goal_list and len(self.goal_list) > 0:
            self.goal = self.goal_list[self.current_goal_index]
        elif self.goal is None:
            raise ValueError("Target point not set, please provide the goal parameter or goal_list during initialization")
        if hasattr(self, 'dynamic_obstacles') and len(self.dynamic_obstacles) > 0:
            def ensure_tuple(pos):
                if isinstance(pos, (list, np.ndarray)):
                    return tuple(pos)
                elif isinstance(pos, tuple):
                    return pos
                else:
                    return tuple(pos)

            forbidden_positions = set()
            forbidden_positions.add(ensure_tuple(self.start))
            forbidden_positions.add(ensure_tuple(self.goal))
            if self.goal_list:
                for goal_point in self.goal_list:
                    forbidden_positions.add(ensure_tuple(goal_point))
            safe_zone = set()
            for fx, fy in forbidden_positions:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        safe_zone.add((fx + dx, fy + dy))
            conflicts = []
            occupied_positions = set()
            for obs in self.dynamic_obstacles:
                pos = tuple(obs['position'])
                occupied_positions.add(pos)
                if pos in safe_zone:
                    conflicts.append(obs)
            if conflicts:
                valid_positions = [
                    (i, j) for i in range(1, self.rows - 1) for j in range(1, self.cols - 1)
                    if self.base_grid_map[i, j] == 0
                    and (i, j) not in safe_zone
                    and (i, j) not in occupied_positions
                ]
                for obs in conflicts:
                    if valid_positions:
                        old_pos = tuple(obs['position'])
                        occupied_positions.discard(old_pos)
                        new_pos = random.choice(valid_positions)
                        obs['position'] = new_pos
                        occupied_positions.add(new_pos)
                        valid_positions.remove(new_pos)
        self.distance = float(np.linalg.norm(np.asarray(self.goal) - self.agent_pos))
        self.history.clear()
        self.obstacle_history = {i: deque(maxlen=4) for i in range(len(self.dynamic_obstacles))}
        self._state_step = None
        self._prediction_step = None
        self.last_action = None
        self.last_info = {}
        self.planning_events = []
        self.global_replans = 0
        self.goal_age = 0
        self._active_goal = tuple(self.goal)
        self.rebuild_grid_map()
        if self.navigation_mode == 'event':
            if not self._plan_route(initial=True):
                self.last_info = {'termination': 'initial_planning_failure', 'success': False}
        elif self.navigation_mode == 'direct_goal':
            # Reviewer 3, comment 6: train with no global guidance at all, so the
            # benefit of Voronoi-A* sub-goals can be attributed independently
            # instead of only as a post-training controller intervention.
            self.goal_list = [self.final_goal]
            self.current_goal_index = 0
            self.goal = self.final_goal
            self._active_goal = tuple(self.goal)
            self.distance = float(np.linalg.norm(np.asarray(self.goal) - self.agent_pos))
        elif self.navigation_mode != 'fixed_waypoints':
            raise ValueError('navigation_mode must be event, fixed_waypoints or direct_goal')
        self.goal_budget = arrival_budget(self.agent_pos, self.goal, model=self.motion_model)
        return self.get_state()

    def get_local_observation(self, position, size=7):
        """
            Get local observation around a specified position

            Args:
            position: (x, y) position tuple
            size: size of the local observation window (default 7x7)

            Returns:
            local_grid: local grid of size x size, where 1 indicates an obstacle and 0 indicates traversable
        """
        # ensure grid_map reflects current dynamic obstacle positions
        self.rebuild_grid_map()
        local_grid = np.zeros((size, size), dtype=int)
        half_size = size // 2

        x_min = max(0, position[0] - half_size)
        x_max = min(self.rows, position[0] + half_size + 1)
        y_min = max(0, position[1] - half_size)
        y_max = min(self.cols, position[1] + half_size + 1)

        r_min = half_size - (position[0] - x_min)
        r_max = half_size + (x_max - position[0])
        c_min = half_size - (position[1] - y_min)
        c_max = half_size + (y_max - position[1])

        local_grid[r_min:r_max, c_min:c_max] = self.grid_map[x_min:x_max, y_min:y_max]

        return local_grid

    def rebuild_grid_map(self):
        # start from base static map
        self.grid_map = self.base_grid_map.copy()
        # overlay dynamic obstacles
        if hasattr(self, 'dynamic_obstacles'):
            for obs in self.dynamic_obstacles:
                x, y = obs['position']
                # only place dynamic obstacle if within bounds
                if 0 <= x < self.rows and 0 <= y < self.cols:
                    self.grid_map[x, y] = 1

    def get_state(self, n_frames=4):
        """Eq. (7)-(8): one causal frame per control step; reads are idempotent.

        At reset the unavailable past frames repeat the initial observation.
        A subgoal update can replace the current target features without
        observing the same obstacle twice or rewriting historical frames.
        """
        if n_frames != 4:
            raise ValueError('Eq. (8) uses exactly four observation frames')
        if self._state_step != self.steps or not self.history:
            self.rebuild_grid_map()
            nearby = np.ones((7, 7), dtype=int)
            r, c = self._pose_cell(self.agent_pos)
            r0, r1 = max(0, r-3), min(self.rows, r+4)
            c0, c1 = max(0, c-3), min(self.cols, c+4)
            nearby[r0-r+3:r1-r+3, c0-c+3:c1-c+3] = self.grid_map[r0:r1, c0:c1]
            self._add_predicted_obstacles_to_grid(nearby)
            delta = np.asarray(self.goal) - self.agent_pos
            frame = np.concatenate(([np.linalg.norm(delta), np.arctan2(delta[1], delta[0])], nearby.ravel()))
            self.history.append(frame)
            while len(self.history) < 4:
                self.history.appendleft(frame.copy())
            self._state_step = self.steps
        else:
            delta = np.asarray(self.goal) - self.agent_pos
            self.history[-1][:2] = [np.linalg.norm(delta), np.arctan2(delta[1], delta[0])]
        return np.asarray(self.history, dtype=np.float32).copy()

    def _motion_safe(self, grid, start, end, *, robot_footprint=True):
        radius = self.robot_radius if robot_footprint else 0.
        margin = self.extra_margin if robot_footprint else 0.
        if not segment_is_safe(grid, start, end, robot_radius=radius,
                               cell_size=self.cell_size, extra_margin=margin):
            return False
        dr, dc = np.asarray(end, dtype=int) - np.asarray(start, dtype=int)
        if dr and dc and abs(dr) == abs(dc) == 1:
            if grid[start[0] + dr, start[1]] or grid[start[0], start[1] + dc]:
                return False
        return True

    def _plan_route(self, initial=False):
        from global_planner import planned_grid_route
        import time
        t0 = time.perf_counter()
        grid = self.original_grid_map.copy()
        if not initial:
            r, c = self._pose_cell(self.agent_pos)
            r0, r1 = max(0, r-3), min(self.rows, r+4)
            c0, c1 = max(0, c-3), min(self.cols, c+4)
            grid[r0:r1, c0:c1] = self.grid_map[r0:r1, c0:c1]
        route, meta = planned_grid_route(grid, self._pose_cell(self.agent_pos), self.final_goal,
                                        robot_radius=self.robot_radius, cell_size=self.cell_size,
                                        extra_margin=self.extra_margin)
        while route and tuple(route[0]) == self._pose_cell(self.agent_pos):
            route.pop(0)
        accepted = bool(route)
        self.planning_events.append({'step': self.steps, 'initial': initial,
                                     'seconds': time.perf_counter()-t0,
                                     'accepted': accepted, **meta})
        if not initial:
            self.global_replans += 1
        if accepted:
            self.goal_list = [tuple(p) for p in route]
            self.current_goal_index = 0
            self.goal = self.goal_list[0]
            self.distance = float(np.linalg.norm(np.asarray(self.goal) - self.agent_pos))
            self._active_goal = tuple(self.goal)
            self.goal_age = 0
            self.goal_budget = arrival_budget(self.agent_pos, self.goal, model=self.motion_model)
        return accepted

    def step(self, action):
        """Execute one grid transition and the six rewards in Eqs. (10)-(16).

        GitHub collision semantics: check the robot's current/next occupied
        cell, then overlap after dynamic obstacles finish moving. Diagonal
        corner cutting and intermediate obstacle crossings are allowed.
        """
        if not 0 <= int(action) < len(self.action_set) or int(action) != action:
            raise ValueError('action must be an integer from 0 through 8')
        action = int(action)
        if self.last_info.get('termination') in ('initial_planning_failure', 'planning_failure'):
            return self.get_state(), 0., True
        self.get_state()
        old_pos = tuple(self.agent_pos)
        target = tuple(self.goal)
        previous_distance = float(np.linalg.norm(np.asarray(target) - old_pos))
        predictions = list(getattr(self, 'predicted_landing_points', []))
        delta = np.asarray(self.action_set[action])
        next_pos = self._advance(old_pos, delta)
        next_cell = self._pose_cell(next_pos)
        collision = any(tuple(o['position']) == self._pose_cell(old_pos) for o in self.dynamic_obstacles)
        collision |= not (0 <= next_cell[0] < self.rows and 0 <= next_cell[1] < self.cols)
        if not collision:
            collision = bool(self.grid_map[next_cell] == 1)
        self.steps += 1
        if not collision:
            self.agent_pos = next_pos
            for obstacle in self.dynamic_obstacles:
                self._update_dynamic_obstacle(obstacle)
            collision = any(
                tuple(o['position']) == self._pose_cell(self.agent_pos) for o in self.dynamic_obstacles)
        self.rebuild_grid_map()
        distance = float(np.linalg.norm(np.asarray(target) - self.agent_pos))
        arrived = not collision and distance <= self.subgoal_tolerance
        components = dict.fromkeys(('goal', 'align', 'avoid', 'prediction', 'step', 'smooth'), 0.)
        components['smooth'] = (0. if self.last_action is None else
            -self.smoothing_factor * float(self.action_smoothness_penalty[self.last_action, action]))
        if collision:
            components['avoid'] = -5.
        else:
            components['goal'] = 10. if arrived else (.6 if distance < previous_distance else -.4)
            goal_vector = np.asarray(target) - old_pos
            norm = np.linalg.norm(delta) * np.linalg.norm(goal_vector)
            components['align'] = .6 * float(delta @ goal_vector / norm) if norm else 0.
            occupied = np.argwhere(self.grid_map != 0)
            if not len(occupied):
                dobs = float('inf')
            elif self.avoid_reference == 'extent':
                # Distance to the cell's axis-aligned box, i.e. the clearance to
                # the obstacle's extent rather than to its centre.
                offset = np.abs(occupied - np.asarray(self.agent_pos, dtype=float))
                dobs = float(np.linalg.norm(np.maximum(offset - .5, 0.), axis=1).min())
            else:
                dobs = float(np.linalg.norm(occupied - self.agent_pos, axis=1).min())
            if self.avoidance_reward_form == 'piecewise':
                components['avoid'] = (-5. if dobs == 0 else -2. if dobs <= 1 else
                                       -1.5 if dobs <= 2 else -.5 if dobs <= 3 else -.4)
            elif self.avoidance_reward_form == 'linear':
                components['avoid'] = -2. * max(0., 3.-dobs) / 3.
            else:
                # Eq. (26): offset the exponential for continuity at 3 cells.
                components['avoid'] = -2.0 * float(np.exp(-dobs / 3.0) - np.exp(-1.0)) if dobs < 3.0 else 0.0
            dpre = min((np.linalg.norm(np.asarray(p) - self.agent_pos) for p in predictions), default=float('inf'))
            components['prediction'] = -1.25 * (2. - float(dpre)) if dpre < 2 else 0.
        if self.reward_pricing == 'per_period':
            # One action occupies max(|delta|, 1) periods of 1/v_max: a diagonal
            # grid action spans sqrt(2) cells and cannot be completed in less
            # (Section V-G), while a stay still consumes one period.  The rate
            # terms are scaled by it; the event terms (arrival, collision,
            # timeout) are not.
            weight = max(float(np.linalg.norm(np.asarray(delta, dtype=float))), 1.)
            if not collision:
                if not arrived:
                    components['goal'] *= weight
                components['align'] *= weight
                components['avoid'] *= weight
                components['prediction'] *= weight
            components['smooth'] *= weight
        self.last_action = action
        success = False
        if arrived:
            if self.current_goal_index + 1 < len(self.goal_list):
                self.current_goal_index += 1
                self.goal = self.goal_list[self.current_goal_index]
            else:
                success = True
        reason = 'collision' if collision else 'success' if success else 'ongoing'
        if reason == 'ongoing' and self.steps >= self.max_steps:
            components['step'] = -10.
            reason = 'timeout'
        if tuple(self.goal) != self._active_goal:
            self._active_goal = tuple(self.goal)
            self.goal_age = 0
            self.goal_budget = arrival_budget(self.agent_pos, self.goal, model=self.motion_model)
        else:
            self.goal_age += 1
        if reason == 'ongoing' and self.navigation_mode == 'event' and self.goal_age >= self.goal_budget:
            if not self._plan_route():
                reason = 'planning_failure'
        self.distance = float(np.linalg.norm(np.asarray(self.goal) - self.agent_pos))
        sg, sa, sm = self.reward_scales
        reward = sg*(components['goal']+components['align']) + sa*(components['avoid']+components['prediction']) + components['step'] + sm*components['smooth']
        self.last_info = {'termination': reason, 'success': reason == 'success',
                          'collision': collision, 'waypoint_arrival': arrived,
                          'reward_components': components.copy()}
        return self.get_state(), float(reward), reason != 'ongoing'

    def _add_predicted_obstacles_to_grid(self, nearby_grid):
        """Record each visible obstacle once at t, then annotate only free cells."""
        if self._prediction_step != self.steps:
            self.predicted_landing_points = []
            for idx, obstacle in enumerate(self.dynamic_obstacles):
                pos = tuple(obstacle['position'])
                cell = self._pose_cell(self.agent_pos)
                if max(abs(pos[0]-cell[0]), abs(pos[1]-cell[1])) > 3:
                    self.obstacle_history[idx].clear()
                    continue
                self.obstacle_history[idx].append(pos)
                if self.prediction_enabled:
                    self.predicted_landing_points.extend(landing_candidates(
                        self.obstacle_history[idx], self.base_grid_map, mode=self.prediction_mode))
            self.predicted_landing_points = list(dict.fromkeys(self.predicted_landing_points))
            self._prediction_step = self.steps
        for position in self.predicted_landing_points:
            self._mark_prediction_in_grid(np.asarray(position), nearby_grid)

    def _mark_prediction_in_grid(self, predicted_pos, nearby_grid):
        """
        Mark predicted position in 7x7 grid

        Args:
            predicted_pos: Predicted position (float or integer array)
            nearby_grid: 7x7 local grid
        """
        pred_pos_int = np.round(predicted_pos).astype(int)

        # Boundary and static obstacle check: only mark positions within map and without static obstacles
        if not (0 <= pred_pos_int[0] < self.rows and 0 <= pred_pos_int[1] < self.cols):
            return  # Out of bounds, do not mark
        if self.base_grid_map[pred_pos_int[0], pred_pos_int[1]] == 1:
            return  # Static obstacle position, do not mark

        agent_cell = self._pose_cell(self.agent_pos)
        dx_obstacle = pred_pos_int[0] - agent_cell[0]
        dy_obstacle = pred_pos_int[1] - agent_cell[1]

        # If within 7x7 range, mark as predicted landing point
        if abs(dx_obstacle) <= 3 and abs(dy_obstacle) <= 3:
            grid_x = int(dx_obstacle + 3)
            grid_y = int(dy_obstacle + 3)
            # Ensure not out of bounds
            if 0 <= grid_x < 7 and 0 <= grid_y < 7:
                if nearby_grid[grid_x, grid_y] == 0:
                    nearby_grid[grid_x, grid_y] = 3

    def _update_dynamic_obstacle(self, obstacle):
        current_pos = obstacle['position']
        obstacle['_motion_trace'] = [tuple(current_pos)]
        behavior = obstacle.get('behavior', 'random_walk')

        if behavior == 'patrol' and obstacle.get('waypoints'):
            target = obstacle['waypoints'][obstacle['waypoint_index']]
            if current_pos == target:
                obstacle['waypoint_index'] = (obstacle['waypoint_index'] + 1) % len(obstacle['waypoints'])
                target = obstacle['waypoints'][obstacle['waypoint_index']]
            direction_vec = np.array(target) - np.array(current_pos)
            direction_vec = np.clip(direction_vec, -1, 1)
            patrol_direction = self._vector_to_direction(tuple(direction_vec))
            if patrol_direction:
                obstacle['direction'] = patrol_direction
        elif behavior == 'wander':
            obstacle['steps_since_turn'] = obstacle.get('steps_since_turn', 0) + 1
            if obstacle['steps_since_turn'] >= obstacle.get('turn_interval', 6):
                obstacle['direction'] = self._choose_new_direction(obstacle, allow_backtrack=False)
                obstacle['steps_since_turn'] = 0
        else:  # random walk
            if random.random() < self.turn_probability:
                obstacle['direction'] = self._choose_new_direction(obstacle)

        direction_vec = self.direction_vectors.get(obstacle['direction'], (0, 0))
        if direction_vec == (0, 0):
            obstacle['direction'] = self._choose_new_direction(obstacle, allow_backtrack=False)
            direction_vec = self.direction_vectors.get(obstacle['direction'], (0, 0))

        speed = max(1, obstacle.get('speed', 1))
        candidate_pos = current_pos
        moved = False

        for _ in range(speed):
            potential = (candidate_pos[0] + direction_vec[0], candidate_pos[1] + direction_vec[1])
            if self._is_passable(potential, ignore=obstacle):
                candidate_pos = potential
                obstacle['_motion_trace'].append(tuple(potential))
                moved = True
            else:
                obstacle['direction'] = self._choose_new_direction(obstacle, allow_backtrack=False)
                direction_vec = self.direction_vectors.get(obstacle['direction'], (0, 0))
                if direction_vec == (0, 0):
                    break

        if moved and candidate_pos != current_pos:
            obstacle['position'] = candidate_pos
        # do NOT directly write into self.grid_map here; rebuild_grid_map() will overlay dynamic obstacles

    def _choose_new_direction(self, obstacle, allow_backtrack=True):
        current_direction = obstacle.get('direction')
        directions = list(self.direction_vectors.keys())
        random.shuffle(directions)

        if not allow_backtrack and current_direction:
            current_vec = self.direction_vectors.get(current_direction, (0, 0))
            opposite_vec = (-current_vec[0], -current_vec[1])
            opposite_direction = self._vector_to_direction(opposite_vec)
            if opposite_direction:
                directions = [d for d in directions if d != opposite_direction]

        for direction in directions:
            vector = self.direction_vectors[direction]
            candidate = (obstacle['position'][0] + vector[0], obstacle['position'][1] + vector[1])
            if self._is_passable(candidate, ignore=obstacle):
                return direction

        return current_direction or random.choice(list(self.direction_vectors.keys()))

    def _is_passable(self, position, ignore=None):
        x, y = position
        if not (0 <= x < self.rows and 0 <= y < self.cols):
            return False
        if self.base_grid_map[x, y] == 1:
            return False
        # Dynamic obstacles should not actively avoid the agent, removed agent position check
        # if tuple(self.agent_pos) == (x, y):
        #     return False
        for other in self.dynamic_obstacles:
            if other is ignore:
                continue
            if tuple(other['position']) == (x, y):
                return False
        return True

    def _direction_to_vector(self, direction):
        return self.direction_vectors.get(direction, (0, 0))

    def _vector_to_direction(self, vector):
        for name, vec in self.direction_vectors.items():
            if vec == vector:
                return name
        return None

    def save_obstacle_config(self):
        """
        Save current obstacle configuration (static + dynamic)

        Returns:
            dict: Environment configuration dictionary containing all necessary information
        """
        # Deep copy dynamic obstacle list to avoid reference issues
        dynamic_obstacles_copy = []
        for obs in self.dynamic_obstacles:
            obs_copy = {
                'position': tuple(obs['position']),
                'direction': obs['direction'],
                'speed': obs['speed'],
                'behavior': obs['behavior'],
                'steps_since_turn': obs['steps_since_turn'],
                'turn_interval': obs['turn_interval']
            }
            # If there are patrol points, also save them
            if 'waypoints' in obs:
                obs_copy['waypoints'] = [tuple(wp) for wp in obs['waypoints']]
                obs_copy['waypoint_index'] = obs['waypoint_index']
            dynamic_obstacles_copy.append(obs_copy)

        config = {
            'base_grid_map': self.base_grid_map.copy(),
            'dynamic_obstacles': dynamic_obstacles_copy,
            'num_obstacles': self.num_obstacles,
            'num_dobs': self.num_dobs,
            'start': tuple(self.start),
            'goal': tuple(self.goal),
            'goal_list': [tuple(g) for g in self.goal_list] if self.goal_list else []
        }

        return config

    def load_obstacle_config(self, config):
        """
        Load previously saved obstacle configuration

        Args:
            config: Configuration dictionary returned from save_obstacle_config()
        """
        # Restore start and goal points
        if 'start' in config:
            self.start = tuple(config['start'])
        if 'goal' in config:
            self.goal = tuple(config['goal'])
        if 'goal_list' in config:
            loaded_goals = [tuple(g) for g in config['goal_list']]
            if loaded_goals and tuple(loaded_goals[0]) == tuple(self.start):
                loaded_goals = loaded_goals[1:]
            self.goal_list = loaded_goals

        self._reset_goal_list = [tuple(p) for p in self.goal_list]
        self.final_goal = self._reset_goal_list[-1] if self._reset_goal_list else tuple(self.goal)
        self._state_step = None
        self._prediction_step = None
        # Restore static obstacles
        self.base_grid_map = config['base_grid_map'].copy()
        # self.base_grid_map = self.original_grid_map.copy()

        # Restore dynamic obstacles
        self.dynamic_obstacles = []
        for obs_data in config['dynamic_obstacles']:
            obstacle = {
                'position': tuple(obs_data['position']),
                'direction': obs_data.get('direction', 'up'),
                'speed': obs_data.get('speed', 1),
                'history': deque([None] * 4, maxlen=4),
                'behavior': obs_data.get('behavior', 'random_walk'),
                'steps_since_turn': obs_data.get('steps_since_turn', 0),
                'turn_interval': obs_data.get('turn_interval', random.randint(4, 10))
            }

            # Restore patrol points (only when waypoints is not None)
            waypoints = obs_data.get('waypoints')
            if waypoints is not None and len(waypoints) > 0:
                obstacle['waypoints'] = [tuple(wp) for wp in waypoints]
                obstacle['waypoint_index'] = obs_data.get('waypoint_index', 0)

            self.dynamic_obstacles.append(obstacle)

        # Reinitialize obstacle history and Kalman filters
        self.obstacle_history = {i: deque(maxlen=4) for i in range(len(self.dynamic_obstacles))}
        self.kalman_filters = {}
        for i in range(len(self.dynamic_obstacles)):
            self.kalman_filters[i] = {
                'state': np.zeros(4),
                'covariance': np.eye(4) * 0.1,
                'initialized': False
            }

        # Rebuild grid
        self.rebuild_grid_map()

    @staticmethod
    def save_configs_to_file(configs, filepath):
        """
        Save multiple environment configurations to file

        Args:
            configs: List of configuration dictionaries
            filepath: Save path
        """
        with open(filepath, 'wb') as f:
            pickle.dump(configs, f)
        print(f"Saved {len(configs)} environment configurations to {filepath}")

    @staticmethod
    def load_configs_from_file(filepath):
        """
        Load environment configurations from file (enhanced version, supports multiple encodings)

        Args:
            filepath: File path

        Returns:
            list: List of configuration dictionaries
        """
        try:
            # First try standard loading
            with open(filepath, 'rb') as f:
                configs = pickle.load(f)
            print(f"Loaded {len(configs)} environment configurations from {filepath}")
            return configs
        except (UnicodeDecodeError, pickle.UnpicklingError) as e:
            print(f"Warning: Standard loading failed ({e}), trying encoding='latin1'...")
            try:
                with open(filepath, 'rb') as f:
                    configs = pickle.load(f, encoding='latin1')
                print(f"✓ Successfully loaded {len(configs)} environment configurations using encoding='latin1'")
                return configs
            except Exception as e2:
                print(f"✗ Loading failed: {e2}")
                raise RuntimeError(f"Unable to load configuration file {filepath}, file may be corrupted") from e2
        return configs
