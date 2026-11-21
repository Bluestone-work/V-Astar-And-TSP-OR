import numpy as np
import pickle
import os
import random
from environment import GridWorld
from tqdm import tqdm
import argparse


MAP_NODES = {
    'grid_map.txt': [(7, 27), (7, 36), (28, 36), (34, 21), (43, 26), (53, 31), (62, 54), (61, 68)],
    'grid_map2.txt': [(80, 110), (71, 98), (35, 94), (42, 78), (82, 69), (75, 25), (67, 14), (31, 19), (24, 27), (13, 35), (13, 94)],
    'grid_map3.txt': [(63, 161), (83, 98), (101, 79), (98, 45), (135, 34), (135, 22)],
    'grid_map4.txt': [(77, 138), (59, 124), (44, 122), (54, 104), (74, 99), (68, 58), (57, 48), (47, 18), (35, 23), (27, 38), (13, 51), (14, 147)],
}


def generate_environment_config(env, seed):
    np.random.seed(seed)
    random.seed(seed)

    env.reset(resample_obstacles=True, seed=seed)

    static_obstacles = []
    for i in range(env.rows):
        for j in range(env.cols):
            if env.base_grid_map[i, j] == 1 and env.original_grid_map[i, j] == 0:
                static_obstacles.append((i, j))

    final_goal = tuple(env.goal_list[-1]) if env.goal_list else tuple(env.goal)

    config = {
        'seed': seed,
        'base_grid_map': env.base_grid_map.copy(),
        'original_grid_map': env.original_grid_map.copy(),
        'static_obstacles': static_obstacles,
        'dynamic_obstacles': [
            {
                'position': tuple(obs['position']),
                'direction': obs['direction'],
                'speed': obs['speed'],
                'behavior': obs.get('behavior', 'random_walk'),
                'steps_since_turn': obs.get('steps_since_turn', 0),
                'turn_interval': obs.get('turn_interval', random.randint(4, 10)),
                'waypoints': ([tuple(wp) for wp in obs['waypoints']]
                             if 'waypoints' in obs and obs['waypoints']
                             else None),
                'waypoint_index': (obs.get('waypoint_index', 0)
                                  if 'waypoints' in obs and obs['waypoints']
                                  else None)
            }
            for obs in env.dynamic_obstacles
        ],
        'start': tuple(env.start),
        'goal': final_goal,
        'goal_list': [tuple(g) for g in env.goal_list],
        'rows': env.rows,
        'cols': env.cols,
        'num_obstacles': env.num_obstacles,
        'num_dynamic_obstacles': env.num_dobs,
        'obstacle_mode': env.obstacle_mode,
        'block_size_range': env.block_size_range
    }

    return config


def generate_test_environments(num_envs=1000,
                               grid_map='grid_map3.txt',
                               obs=100,
                               dobs=50,
                               obstacle_mode='block',
                               block_size_range=(2, 5),
                               output_dir='test_environments',
                               start_seed=0):
    """
        Generate multiple test environments and save them

        Args:
        num_envs: Number of environments to generate
        grid_map: Path to the map file
        obs: Number of static obstacle shapes
        dobs: Number of dynamic obstacles
        obstacle_mode: Obstacle mode ('point' or 'block')
        block_size_range: Range of block sizes (min_size, max_size)
        output_dir: Output directory
        start_seed: Starting random seed
    """

    os.makedirs(output_dir, exist_ok=True)
    mapname = os.path.basename(grid_map)
    file_name, extension = os.path.splitext(mapname)
    grid_map_array = np.loadtxt(grid_map, dtype=int)

    nodes = MAP_NODES.get(os.path.basename(grid_map), MAP_NODES['grid_map4.txt'])
    env = GridWorld(
        grid_map=grid_map_array,
        start=nodes[0],
        goal=nodes[-1],
        goal_list=nodes,
        obs=obs,
        dobs=dobs,
        obstacle_mode=obstacle_mode,
        block_size_range=block_size_range
    )

    print(f"Starting to generate {num_envs} test environments...")
    print(f"Configuration: Number of static obstacle shapes={obs}, Number of dynamic obstacles={dobs}")
    print(f"Obstacle mode: {obstacle_mode}")
    if obstacle_mode == 'block':
        print(f"Block size range: {block_size_range}")
    print(f"Output directory: {output_dir}")

    environments = []

    for i in tqdm(range(num_envs), desc="Generating environments"):
        seed = start_seed + i
        config = generate_environment_config(env, seed)
        environments.append(config)

    # 保存环境配置
    mode_suffix = f'_{obstacle_mode}' if obstacle_mode != 'point' else ''
    output_file = os.path.join(output_dir, f'test_envs_{num_envs}_obs{obs}_dobs{dobs}{mode_suffix}_{file_name}.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(environments, f)

    print(f"\nSuccessfully generated {num_envs} environments")
    print(f"Saved to: {output_file}")

    stats = {
        'num_environments': num_envs,
        'obs': obs,
        'dobs': dobs,
        'obstacle_mode': obstacle_mode,
        'block_size_range': block_size_range if obstacle_mode == 'block' else None,
        'start_seed': start_seed,
        'end_seed': start_seed + num_envs - 1,
        'grid_map': grid_map
    }

    stats_file = os.path.join(output_dir, f'test_envs_{num_envs}_obs{obs}_dobs{dobs}{mode_suffix}_stats_{file_name}.txt')
    with open(stats_file, 'w') as f:
        f.write("Test Environment Set Statistics\n")
        f.write("=" * 50 + "\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")

        total_static_cells = sum(len(env['static_obstacles']) for env in environments)
        total_dynamic_obs = sum(len(env['dynamic_obstacles']) for env in environments)

        f.write("\nObstacle Statistics:\n")
        f.write(f"Average number of static obstacle cells: {total_static_cells / num_envs:.2f}\n")
        f.write(f"Average number of dynamic obstacles: {total_dynamic_obs / num_envs:.2f}\n")

        # Only calculate size distribution if there are dynamic obstacles
        if total_dynamic_obs > 0:
            size_counts = {'1x1': 0, '2x2': 0}
            for env_config in environments:
                for obs in env_config['dynamic_obstacles']:
                    size = obs.get('size', 1)
                    if size == 1:
                        size_counts['1x1'] += 1
                    else:
                        size_counts['2x2'] += 1

            f.write(f"\nDynamic Obstacle Size Distribution:\n")
            f.write(f"1x1: {size_counts['1x1']} ({size_counts['1x1']/total_dynamic_obs*100:.1f}%)\n")
            f.write(f"2x2: {size_counts['2x2']} ({size_counts['2x2']/total_dynamic_obs*100:.1f}%)\n")
        else:
            f.write(f"\nDynamic Obstacle Size Distribution: N/A (no dynamic obstacles)\n")

    print(f"Statistics saved to: {stats_file}")

    return output_file


def load_test_environments(env_file):
    """
    Load test environment configurations

    Args:
        env_file: Path to the environment configuration file

    Returns:
        list: List of environment configurations
    """
    with open(env_file, 'rb') as f:
        environments = pickle.load(f)
    return environments


def apply_environment_config(env, config):
    """
    Apply the saved configuration to the environment

    Args:
    env: GridWorld environment instance
    config: Environment configuration dictionary
    """
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    env.base_grid_map = config['base_grid_map'].copy()
    env.original_grid_map = config['original_grid_map'].copy()
    from collections import deque
    env.dynamic_obstacles = []
    for obs in config['dynamic_obstacles']:
        obstacle = {
            'position': obs['position'],
            'direction': obs['direction'],
            'speed': obs['speed'],
            'behavior': obs.get('behavior', 'random_walk'),
            'history': deque([None] * 4, maxlen=4),
            'steps_since_turn': obs.get('steps_since_turn', 0),
            'turn_interval': obs.get('turn_interval', random.randint(4, 10))
        }
        waypoints = obs.get('waypoints')
        if waypoints is not None and len(waypoints) > 0:
            obstacle['waypoints'] = waypoints
            obstacle['waypoint_index'] = obs.get('waypoint_index', 0)
        env.dynamic_obstacles.append(obstacle)
    env.obstacle_history = {i: deque(maxlen=4) for i in range(len(env.dynamic_obstacles))}
    env.velocity_history = {i: deque(maxlen=4) for i in range(len(env.dynamic_obstacles))}

    if 'obstacle_mode' in config:
        env.obstacle_mode = config['obstacle_mode']
    if 'block_size_range' in config:
        env.block_size_range = config['block_size_range']

    env.start = np.array(config['start'])
    env.goal = np.array(config['goal'])
    env.goal_list = [np.array(g) for g in config['goal_list']]
    if env.goal_list and tuple(env.goal_list[0]) == tuple(env.start):
        env.goal_list = env.goal_list[1:]
    env._reset_goal_list = [tuple(p) for p in env.goal_list]
    env.final_goal = tuple(config['goal'])
    env._state_step = None
    env._prediction_step = None
    env.agent_pos = env.start.copy()
    env.current_goal_index = 0
    env.steps = 0
    env.distance = np.linalg.norm(np.array(env.start) - np.array(env.goal))

    # Mark this environment as loaded from a fixed configuration to prevent resampling on reset
    env._from_fixed_config = True
    env._fixed_config_seed = config['seed']

    # Rebuild the grid map (merge base_grid_map and dynamic obstacles into grid_map)
    env.rebuild_grid_map()

    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate test environment settings for GridWorld')
    parser.add_argument('--num_envs', type=int, default=5, help='Number of environments to generate')
    parser.add_argument('--obs', type=int, default=50, help='Number of static obstacles')
    parser.add_argument('--dobs', type=int, default=0, help='Number of dynamic obstacles')
    parser.add_argument('--obstacle_mode', type=str, default='point',
                       choices=['point', 'block'], help='Obstacle mode')
    parser.add_argument('--block_min', type=int, default=3, help='Minimum block size')
    parser.add_argument('--block_max', type=int, default=5, help='Maximum block size')
    parser.add_argument('--grid_map', type=str, default='grid_map4.txt', help='Map file')
    parser.add_argument('--output_dir', type=str, default='test_environments_g', help='Output directory')
    parser.add_argument('--start_seed', type=int, default=42, help='Starting random seed')

    args = parser.parse_args()

    generate_test_environments(
        num_envs=args.num_envs,
        grid_map=args.grid_map,
        obs=args.obs,
        dobs=args.dobs,
        obstacle_mode=args.obstacle_mode,
        block_size_range=(args.block_min, args.block_max),
        output_dir=args.output_dir,
        start_seed=args.start_seed
    )
