import argparse
import copy
import json
import random
import numpy as np
import os
import time
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from environment import GridWorld
from agents import RainbowDQN
from models import DuelingDQN_with_LSTM
from navigation_protocol import CODE_PROTOCOL
import hashlib
from pathlib import Path


def set_global_seed(seed):
    """Seed every RNG used by the public training entry point."""
    seed = int(seed)
    # The environment loop is small and Python driven; unrestricted BLAS
    # thread pools add substantial scheduling overhead and make seed runs
    # contend with one another.  A single Torch thread keeps the fixed-budget
    # protocol deterministic and is faster for this workload.
    torch.set_num_threads(int(os.environ.get("TSP_OR_TORCH_THREADS", "1")))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed

def build_revision_curriculum(stage_episodes=(300, 450, 750, 1000, 500),
                              epsilon_boosts=(0.0, 0.3, 0.4, 0.3, 0.2)):
    """Manuscript L1--L4 followed by uniformly sampled mixed-level training.

    A separate per-run RNG selects mixed levels, independent of environment,
    replay and exploration random draws. Episode counts may be shortened for
    smoke tests; the defaults implement the requested 3,000-episode budget.
    """
    if len(stage_episodes) != 5 or any(int(n) != n or n < 1 for n in stage_episodes):
        raise ValueError('stage_episodes must contain five positive integers')
    if len(epsilon_boosts) != 5 or any(not 0 <= x <= 1 for x in epsilon_boosts):
        raise ValueError('epsilon_boosts must contain five values in [0, 1]')
    levels = [dict(level=1, obs=50, dobs=0, mode='point'),
              dict(level=2, obs=50, dobs=30, mode='point'),
              dict(level=3, obs=50, dobs=30, mode='block', block_range=(2, 5)),
              dict(level=4, obs=50, dobs=50, mode='block', block_range=(2, 5))]
    stages = [dict(spec, episodes=int(stage_episodes[i]), epsilon_boost=epsilon_boosts[i],
                   description='Level %d' % (i + 1)) for i, spec in enumerate(levels)]
    stages.append(dict(episodes=int(stage_episodes[4]), mode='mixed', levels=levels,
                       epsilon_boost=epsilon_boosts[4], description='Mixed Levels 1-4'))
    return dict(epsilon_reset_strategy='partial', mixed_sampling='uniform_independent_rng', stages=stages)


def train_agent(env, agent, episodes=1000, max_steps=2000, save_path="dqn_model.pth", log_dir="runs",
                curriculum_learning=False, curriculum_config=None, seed=None,
                best_save_path=None, navigation_mode='event', checkpoint_interval=0,
                checkpoint_dir=None):
    """
        Training

        Args:
        curriculum_learning: Whether to use curriculum learning
        curriculum_config: Curriculum learning configuration, format:
        {
            'epsilon_reset_strategy': 'partial', # epsilon reset strategy
            'stages': [
                {'episodes': 300, 'mode': 'point', 'obs': 30,
                    'epsilon_boost': 0.3, 'description': 'Stage 1'},
                {'episodes': 400, 'mode': 'block', 'block_range': (2, 3), 'obs': 40,
                    'epsilon_boost': 0.4, 'description': 'Stage 2'},
                ...
            ]
        }

    """
    if episodes < 1 or max_steps < 1:
        raise ValueError('episodes and max_steps must be positive')
    if int(checkpoint_interval) != checkpoint_interval or checkpoint_interval < 0:
        raise ValueError('checkpoint_interval must be a nonnegative integer')
    curriculum_config = copy.deepcopy(curriculum_config)
    if curriculum_learning:
        if not curriculum_config or not curriculum_config.get('stages'):
            raise ValueError('curriculum_learning requires nonempty stages')
        stages = curriculum_config['stages']
        if any(int(stage['episodes']) != stage['episodes'] or stage['episodes'] < 1 for stage in stages) or sum(stage['episodes'] for stage in stages) != episodes:
            raise ValueError('curriculum stage episodes must be positive and sum to episodes')
        if any(stage['mode'] == 'mixed' and not stage.get('levels') for stage in stages):
            raise ValueError('mixed stages require level specifications')
    curriculum_rng = random.Random(seed)
    checkpoint_records = []
    env.max_steps = int(max_steps)
    env.navigation_mode = navigation_mode
    env.smoothing_factor = agent.smoothing_factor
    agent.env = env
    run_id = time.strftime("%Y%m%d-%H%M%S") + '-%d' % time.time_ns()
    log_dir = os.path.join(log_dir, f"run_{run_id}")
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir else Path(log_dir) / 'checkpoints'
    writer = SummaryWriter(log_dir)
    episode_records = open(os.path.join(log_dir, 'episodes.jsonl'), 'a', buffering=1)
    source_names = ('agents.py', 'environment.py', 'models.py', 'global_planner.py',
                    'train.py', 'navigation_protocol.py', 'obstacle_prediction.py', 'manuscript_replay.py',
                    'experiment_protocol.py', 'generate_test_environments.py',
                    'experiments/proposed_evaluation.py', 'experiments/keypoint_density.py',
                    'configs/method.json', 'configs/curriculum_mazea_1000.json', 'configs/levels.json')
    source_root = Path(__file__).resolve().parent
    snapshot = Path(log_dir) / 'source'
    snapshot.mkdir()
    source_hashes = {}
    for name in source_names:
        contents = (source_root / name).read_bytes()
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)
        source_hashes[name] = hashlib.sha256(contents).hexdigest()
    agent.training_source_sha256 = source_hashes
    agent.training_seed = seed
    agent.training_configuration.update(episodes=episodes, max_steps=max_steps,
        curriculum_config=curriculum_config, navigation_mode=navigation_mode)


    best_reward = -float('inf')
    step_counter = 0
    all_rewards = []
    all_losses = []
    all_epsilons = []
    success_history = []

    if curriculum_learning and curriculum_config:
        print("=" * 80)
        print("Course Study Mode Activated")
        print("=" * 80)
        for i, stage in enumerate(curriculum_config['stages'], 1):
            print(f"Stage {i}: {stage['description']}")
            print(f"  - Episodes: {stage['episodes']}")
            print(f"  - Obstacle Mode: {stage['mode']}")
            print(f"  - Number of Obstacles: {stage.get('obs', env.num_obstacles)}")
            if stage['mode'] == 'block':
                print(f"  - Block Size: {stage.get('block_range', (2, 5))}")
        print("=" * 80)
    else:
        print("Training started (each episode will generate a random environment to enhance diversity)...")

    current_stage_info = None
    current_stage_idx = -1
    if curriculum_learning and curriculum_config:
        episode_counter = 0
        for stage in curriculum_config['stages']:
            stage['start_episode'] = episode_counter
            stage['end_episode'] = episode_counter + stage['episodes']
            episode_counter += stage['episodes']

        # Obtain epsilon reset strategy
        epsilon_reset_strategy = curriculum_config.get('epsilon_reset_strategy', 'partial')
        (Path(log_dir) / 'curriculum_config.json').write_text(json.dumps({
            'curriculum_config': curriculum_config, 'seed': seed,
            'mixed_sampling': 'uniform_independent_rng', 'episodes': episodes,
            'checkpoint_interval_episodes': checkpoint_interval,
        }, indent=2))

    with tqdm(total=episodes, desc="Training Progress", unit="episode") as pbar:
        for episode in range(episodes):
            if curriculum_learning and curriculum_config:
                for stage_idx, stage in enumerate(curriculum_config['stages']):
                    if stage['start_episode'] <= episode < stage['end_episode']:
                        if current_stage_idx != stage_idx:
                            old_epsilon = agent.epsilon
                            current_stage_idx = stage_idx
                            current_stage_info = stage

                            epsilon_boost = stage.get('epsilon_boost', 0.0)

                            if epsilon_reset_strategy == 'full':
                                agent.epsilon = 1.0
                                tqdm.write(f"  Epsilon reset: {old_epsilon:.4f} → 1.0 (Full reset)")
                            elif epsilon_reset_strategy == 'partial':
                                agent.epsilon = min(1.0, agent.epsilon + epsilon_boost)
                                tqdm.write(f"  Epsilon boost: {old_epsilon:.4f} → {agent.epsilon:.4f} (+{epsilon_boost})")
                            elif epsilon_reset_strategy == 'adaptive':
                                # Adaptive: adjust based on difficulty changes
                                if stage_idx > 0:
                                    prev_mode = curriculum_config['stages'][stage_idx-1]['mode']
                                    if prev_mode == 'point' and stage['mode'] == 'block':
                                        # From point to block: significant boost
                                        agent.epsilon = min(1.0, agent.epsilon + 0.4)
                                        tqdm.write(f"  Epsilon adaptive boost: {old_epsilon:.4f} → {agent.epsilon:.4f} (Point→Block)")
                                    else:
                                        # Same type difficulty increase: small boost
                                        agent.epsilon = min(1.0, agent.epsilon + epsilon_boost)
                                        tqdm.write(f"  Epsilon adaptive boost: {old_epsilon:.4f} → {agent.epsilon:.4f}")
                            # else: 'none' - no epsilon adjustment

                            tqdm.write(f"\n{'='*60}")
                            tqdm.write(f"Entering {stage['description']} (Episode {episode})")
                            tqdm.write(f"  Obstacle mode: {stage['mode']}")
                            tqdm.write(f"  Static/dynamic obstacles: {stage.get('obs', 'mixed')}/{stage.get('dobs', 'mixed')}")
                            if stage['mode'] == 'block':
                                tqdm.write(f"  Block size: {stage.get('block_range', (2, 5))}")
                            tqdm.write(f"{'='*60}\n")
                        break

            level_spec = current_stage_info or {}
            if level_spec.get('mode') == 'mixed':
                level_spec = curriculum_rng.choice(level_spec['levels'])
            if level_spec:
                env.obstacle_mode = level_spec['mode']
                env.num_obstacles = level_spec.get('obs', env.num_obstacles)
                env.num_dobs = level_spec.get('dobs', env.num_dobs)
                if env.obstacle_mode == 'block':
                    env.block_size_range = level_spec.get('block_range', (2, 5))
            # Sample the same safe zones for all training arms. reset() selects
            # the final goal or the intermediate route after sampling the scene.
            env.goal_list = env._reset_goal_list.copy()
            env.goal = env.goal_list[0] if env.goal_list else env.final_goal
            env.reset_dynamic(seed=None)
            state = env.reset()
            agent.reset_episode()
            done = bool(env.planning_events and not env.planning_events[0]['accepted'])
            if done:
                env.last_info = {'termination': 'initial_planning_failure', 'success': False}
            total_reward = 0
            episode_steps = 0
            losses = []
            td_errors = []
            reached_goal = False
            # GridWorld advances through the normalized goal list itself.  A
            # prior version iterated over goal pairs while also letting the
            # environment advance its index, which made the start pose an
            # accidental target and did not represent the published protocol.
            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).to(agent.device)
                action = agent.select_action(state_tensor)
                next_state, reward, done = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                learning_metrics = agent.train(return_metrics=True)
                agent.decay_epsilon()
                state = next_state
                total_reward += reward
                episode_steps += 1
                step_counter += 1
                if learning_metrics is not None:
                    losses.append(learning_metrics['loss'])
                    td_errors.append(learning_metrics['abs_td_error_mean'])
                # Per-step Q-value snapshots were never consumed by the
                # evaluator and doubled network inference during long
                # multi-seed runs.  Keep training metrics to reward/loss and
                # interaction count so the fixed-budget protocol is unchanged.
                if done:
                    reached_goal = bool(env.last_info.get('success', False))
            success_history.append(1 if reached_goal else 0)
            recent_window = min(100, len(success_history))
            success_rate = (sum(success_history[-recent_window:]) / recent_window) * 100
            episode_records.write(json.dumps({
                'episode': episode, 'environment_steps': step_counter,
                'stage_index': current_stage_idx + 1 if current_stage_info else None,
                'stage': current_stage_info.get('description') if current_stage_info else None,
                'level': level_spec.get('level'),
                'obstacle_mode': env.obstacle_mode,
                'static_obstacles': env.num_obstacles, 'dynamic_obstacles': env.num_dobs,
                'block_size_range': list(env.block_size_range) if env.obstacle_mode == 'block' else None,
                'episode_steps': episode_steps, 'reward': total_reward,
                'mean_absolute_td_error': float(np.mean(td_errors)) if td_errors else None,
                'success': reached_goal, 'termination': env.last_info.get('termination'),
                'global_replans': env.global_replans,
                'collision_protocol': env.collision_protocol,
                'planning_seconds': sum(e['seconds'] for e in env.planning_events),
                'sampled_event_counts_cumulative': dict(agent.sampled_event_counts),
            }) + '\n')

            writer.add_scalar("Reward/Episode", total_reward, episode)
            writer.add_scalar("Epsilon", agent.epsilon, episode)
            writer.add_scalar("Success_Rate", success_rate, episode)

            if curriculum_learning and current_stage_info:
                stage_name = current_stage_info['description'].replace(' ', '_')
                writer.add_scalar(f"Stage_{stage_name}/Reward", total_reward, episode)
                writer.add_scalar(f"Stage_{stage_name}/Success_Rate", success_rate, episode)

            if losses:
                avg_loss = np.mean(losses)
                writer.add_scalar("Loss/Episode", avg_loss, episode)
                all_losses.append(avg_loss)
            all_rewards.append(total_reward)
            all_epsilons.append(agent.epsilon)

            pbar.set_postfix({
                "Total Reward": f"{total_reward:.1f}",
                "Success Rate": f"{success_rate:.1f}%",
                "Epsilon": f"{agent.epsilon:.3f}"
            })
            pbar.update(1)
            if best_save_path is not None and total_reward > best_reward:
                best_reward = total_reward
                agent.save_model(best_save_path)
            stage_end = bool(current_stage_info and episode + 1 == current_stage_info['end_episode'])
            periodic = bool(checkpoint_interval and (episode + 1) % checkpoint_interval == 0)
            if stage_end or periodic:
                checkpoint_root.mkdir(parents=True, exist_ok=True)
                tag = ('stage%d_end' % (current_stage_idx + 1)) if stage_end else 'periodic'
                intermediate_path = checkpoint_root / ('episode_%04d_%s.pth' % (episode + 1, tag))
                agent.save_model(str(intermediate_path))
                checkpoint_records.append(dict(episode=episode + 1, stage=current_stage_idx + 1,
                                               path=str(intermediate_path.resolve()), reason=tag))
                (Path(log_dir) / 'checkpoint_manifest.json').write_text(json.dumps(checkpoint_records, indent=2))
            if episode % 10 == 0:
                writer.flush()
        # The final checkpoint is the reproducible default.  A best-training-
        # reward checkpoint is optional and must never silently replace it.
        agent.save_model(save_path)
        final_path = Path(save_path).resolve()
        with final_path.open('rb') as handle:
            final_sha256 = hashlib.sha256(handle.read()).hexdigest()
        (Path(log_dir) / 'checkpoint_manifest.json').write_text(json.dumps(
            [*checkpoint_records, dict(episode=episodes, path=str(final_path),
                                       reason='final', sha256=final_sha256)],
            indent=2,
        ) + '\n')
        writer.close()
        episode_records.close()

    print(f"Training complete, the model has been saved to {save_path}")

    np.save(os.path.join(log_dir, "rewards.npy"), all_rewards)
    np.save(os.path.join(log_dir, "losses.npy"), all_losses)
    np.save(os.path.join(log_dir, "epsilons.npy"), all_epsilons)
    with open(os.path.join(log_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "code_protocol": CODE_PROTOCOL,
            "navigation_mode": navigation_mode,
            "collision_protocol": env.collision_protocol,
            "motion_model": getattr(env, 'motion_model', 'grid'),
            "state": "four control timesteps; raw distance cells and bearing radians",
            "reward": "Eqs10-16; TableI; TableII reversal precedence",
            "prediction_mode": env.prediction_mode,
            "prediction_enabled": env.prediction_enabled,
            "avoidance_reward_form": env.avoidance_reward_form,
            "reward_scales": env.reward_scales,
            "subgoal_tolerance_cells": env.subgoal_tolerance,
            "smoothness_weight": env.smoothing_factor,
            "epsilon_decay_unit": "environment_step",
            "target_update_optimizer_interval": agent.update_target_freq,
            "target_tau": agent.target_tau,
            "replay_mode": agent.replay_mode,
            "learner_specification": agent.learner_specification(),
            "training_configuration": agent.training_configuration,
            "per_alpha": agent.alpha if agent.replay_mode == 'per' else None,
            "per_beta_initial": agent.beta if agent.replay_mode == 'per' else None,
            "per_beta_steps": agent.per_beta_steps if agent.replay_mode == 'per' else None,
            "sampled_event_counts": dict(agent.sampled_event_counts),
            "source_sha256": source_hashes,
            "map_shape": list(env.original_grid_map.shape),
            "structural_grid_sha256": hashlib.sha256(np.asarray(env.original_grid_map, dtype=np.int8).tobytes()).hexdigest(),
            "architecture": agent.architecture,
            "online_parameters": sum(p.numel() for p in agent.q_network.parameters()),
            "gamma": agent.gamma,
            "learning_rate": agent.optimizer.param_groups[0]['lr'],
            "replay_capacity": agent.memory.maxlen,
            "robot_radius_physical_units": env.robot_radius,
            "cell_size": env.cell_size,
            "extra_margin_physical_units": env.extra_margin,
            "seed": seed,
            "episodes": episodes,
            "curriculum_learning": curriculum_learning,
            "curriculum_config": curriculum_config if curriculum_learning else None,
            "mixed_level_rng_seed": seed if curriculum_learning else None,
            "checkpoint_interval_episodes": checkpoint_interval,
            "intermediate_checkpoints": checkpoint_records,
            "max_steps_per_episode": max_steps,
            "environment_steps": step_counter,
            "checkpoint": os.path.abspath(save_path),
            "checkpoint_sha256": final_sha256,
            "checkpoint_selection": "final",
        }, f, indent=2)

if __name__ == "__main__":
    from generate_test_environments import MAP_NODES
    parser = argparse.ArgumentParser(description='Train the manuscript-aligned TSP-OR DQN')
    parser.add_argument('--config', default='configs/method.json')
    parser.add_argument('--map-file', default='grid_map2.txt')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--episodes', type=int)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--save-path')
    parser.add_argument('--log-dir', default='outputs/training')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--replay-mode', choices=('manuscript', 'uniform', 'per'), default='manuscript')
    parser.add_argument('--importance-sampling', choices=('auto', 'on', 'off'), default='auto')
    parser.add_argument('--architecture', choices=('bilstm_dueling', 'bilstm_plain', 'unilstm_dueling', 'unilstm_plain', 'plain'), default='bilstm_dueling')
    parser.add_argument('--prediction-mode', choices=('none', 'constant_velocity', 'occupancy_aggregation', 'history_voting'), default='history_voting')
    parser.add_argument('--avoidance-reward-form', choices=('piecewise', 'linear', 'exponential'), default='piecewise')
    parser.add_argument('--reward-scales', type=float, nargs=3, default=[1., 1., 1.])
    parser.add_argument('--smoothing-factor', type=float, default=.5)
    parser.add_argument('--navigation-mode', choices=('event', 'fixed_waypoints', 'direct_goal'), default='event',
                        help='event: BKRT event-triggered global replanning; fixed_waypoints: keep the '
                             'initial waypoint route; direct_goal: no global guidance, the final goal is '
                             'the only sub-goal (Reviewer 3 comment 6 attribution arm)')
    parser.add_argument('--revision-curriculum', action='store_true', help='L1-L4 300/450/750/1000 + 500 mixed')
    parser.add_argument('--curriculum-config', help='JSON curriculum configuration (stages sum must equal episodes)')
    parser.add_argument('--curriculum-epsilon-boosts', type=float, nargs=5, default=[0., .3, .4, .3, .2])
    parser.add_argument('--checkpoint-interval', type=int, default=0, help='Additional checkpoint interval in episodes; stage ends always saved')
    parser.add_argument('--checkpoint-dir')
    parser.add_argument('--no-curriculum', action='store_true', help='Disable the default mixed-level curriculum')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if config.get('code_protocol') != CODE_PROTOCOL:
        raise ValueError('configuration does not identify ' + CODE_PROTOCOL)
    learner = config['learner']
    if learner.get('reward_scaling_applications') != 1:
        raise ValueError('Only single reward scaling is supported')
    if 'bootstrap_power' in learner:
        del learner['bootstrap_power']
        print('Legacy bootstrap_power ignored. Training uses optimizer.gamma once per decision.')
    replay = config['replay']
    from navigation_protocol import REPLAY_PROTOCOL
    if (replay['protocol'], replay['reward_weight'], replay['td_error_weight'],
        replay['priority_floor'], replay['sampling_with_replacement']) != (REPLAY_PROTOCOL,.25,.75,1e-5,True):
        raise ValueError('Replay configuration differs from the implemented released formula')
    set_global_seed(args.seed)
    if args.no_curriculum and (args.revision_curriculum or args.curriculum_config):
        parser.error('--no-curriculum conflicts with a curriculum option')
    if args.revision_curriculum and args.curriculum_config:
        parser.error('choose --revision-curriculum or --curriculum-config')
    curriculum_path = args.curriculum_config or (None if args.no_curriculum or args.revision_curriculum
                                               else config.get('curriculum_file'))
    curriculum = (json.loads(Path(curriculum_path).read_text()) if curriculum_path else
                  build_revision_curriculum(epsilon_boosts=args.curriculum_epsilon_boosts) if args.revision_curriculum else None)
    default_episodes = sum(s['episodes'] for s in curriculum['stages']) if curriculum else config['training_episodes']
    episodes = args.episodes if args.episodes is not None else default_episodes
    cap = args.max_steps if args.max_steps is not None else config['max_steps_per_episode']
    training = config['training_environment']
    safety = config['safety']
    nodes = MAP_NODES[Path(args.map_file).name]
    from experiment_protocol import SyncedGridWorld, layout_seed_base
    env = SyncedGridWorld(np.loadtxt(args.map_file, dtype=int), nodes[0], nodes[-1], goal_list=nodes,
                    obs=training['static_obstacles'], dobs=training['dynamic_obstacles'],
                    obstacle_mode=training['obstacle_mode'], max_steps=cap,
                    robot_radius=safety['robot_radius_physical_units'], cell_size=safety['cell_size'],
                    extra_margin=safety['extra_margin_physical_units'],
                    subgoal_tolerance=config['subgoal_tolerance_cells'],
                    prediction_mode=args.prediction_mode, avoidance_reward_form=args.avoidance_reward_form,
                    reward_scales=args.reward_scales)
    env.train_env_seed_base = layout_seed_base(args.seed)
    agent = RainbowDQN(env, device=args.device, replay_mode=args.replay_mode, architecture=args.architecture,
                       smoothing_factor=args.smoothing_factor,
                       reward_scale=learner['reward_scale'], double_q=learner['double_q'],
                       huber=learner['huber'],
                       importance_sampling=((config['standard_per']['importance_sampling'] if args.replay_mode == 'per'
                                             else config['replay']['importance_sampling'] if args.replay_mode == 'manuscript'
                                             else False) if args.importance_sampling == 'auto'
                                            else args.importance_sampling == 'on'),
                       alpha=config['standard_per']['alpha'], beta=config['standard_per']['beta_initial'],
                       per_beta_steps=config['standard_per']['beta_anneal_optimizer_steps'],
                       **config['optimizer'])
    agent.target_tau = float(config['target_network']['tau'])
    agent.update_target_freq = int(config['target_network']['optimizer_interval'])
    if not 0 < agent.target_tau <= 1 or agent.update_target_freq < 1:
        raise ValueError('Invalid target-network update configuration')
    agent.training_configuration = dict(configuration=config,
        configuration_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        command_line=__import__('sys').argv[1:])
    save_path = args.save_path or ('outputs/%s_%s_is%s_seed%d.pth' %
                                  (Path(args.map_file).stem, args.replay_mode,
                                   'on' if agent.importance_sampling else 'off', args.seed))
    train_agent(env, agent, episodes=episodes, max_steps=cap, save_path=save_path,
                log_dir=args.log_dir, seed=args.seed, navigation_mode=args.navigation_mode,
                curriculum_learning=curriculum is not None, curriculum_config=curriculum,
                checkpoint_interval=args.checkpoint_interval, checkpoint_dir=args.checkpoint_dir)
