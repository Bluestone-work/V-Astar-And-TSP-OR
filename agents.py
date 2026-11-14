"""Released learner: single reward scaling, Double DQN and weighted Huber.

Priority rule and importance sampling are separate experimental factors.
See docs/METHOD.md for the exact released method and historical boundaries.
"""
import inspect
import hashlib
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, Counter
from pathlib import Path
from navigation_protocol import CODE_PROTOCOL, REPLAY_PROTOCOL, action_transition_matrix
from models import (DuelingDQN_with_LSTM, DuelingDQN_with_uniLSTM,
                    PlainDQN_with_LSTM, PlainDQN_with_BiLSTM, PlainDQN)
from manuscript_replay import ManuscriptReplay, StandardPER, UniformReplay

class RainbowDQN:
    def __init__(self, env, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                learning_rate=0.001, batch_size=64, memory_size=200000, device=None, alpha=0.6, beta=0.4,
                n_step=1, shared_replay_buffer=None, smoothing_factor=0.5, action_consistency=0.8,
                replay_mode='manuscript', architecture='bilstm_dueling', per_beta_steps=200000,
                reward_scale=1, double_q=True, huber=True,
                side_buffer=False, importance_sampling=None):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_target_freq = 1
        self.target_tau = 0.01
        self.per_beta_steps = max(1, int(per_beta_steps))
        self.environment_steps = 0
        self.sampled_event_counts = Counter()
        self.replay_events = deque(maxlen=memory_size)
        self.alpha = alpha  # Legacy call compatibility; Eq. (18) has no exponent.
        self.beta = beta
        if n_step != 1:
            raise ValueError("This release stores immediate rewards, not n-step returns; n_step must be 1")
        self.n_step = 1
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_steps = 0
        self.log_offset = 1.0  # Manuscript Eq. (19), lambda=1
        self.w_reward = 0.25
        self.w_TD_error = 0.75
        # Explicit settings for this released method; historical results require their own provenance.
        self.reward_scale = float(reward_scale)
        self.double_q = bool(double_q)
        self.huber = bool(huber)
        if not np.isfinite(self.reward_scale) or self.reward_scale <= 0:
            raise ValueError('reward scale must be finite and positive')
        self.importance_sampling = (replay_mode == 'per' if importance_sampling is None
                                    else bool(importance_sampling))
        if self.importance_sampling and replay_mode == 'uniform':
            raise ValueError('uniform replay uses unweighted loss')
        self.training_seed = None
        self.training_configuration = {}
        self.smoothing_factor = smoothing_factor  # Path smoothing weight
        self.action_consistency = action_consistency  # Action consistency weight
        self.architecture = architecture
        self.last_actions = deque(maxlen=3)  # Recent action history

        # Deadlock detection and escape mechanism
        self.stuck_threshold = 8  # Number of consecutive steps with Q-value change below threshold considered stuck
        self.stuck_counter = 0
        self.last_max_q = None
        self.escape_mode = False
        self.escape_steps = 0

        network_classes = {
            'bilstm_dueling': DuelingDQN_with_LSTM,
            'unilstm_dueling': DuelingDQN_with_uniLSTM,
            'unilstm_plain': PlainDQN_with_LSTM,
            'bilstm_plain': PlainDQN_with_BiLSTM,
            'plain': PlainDQN,
        }
        if architecture not in network_classes:
            raise ValueError(f"unknown architecture: {architecture}")
        network = network_classes[architecture]
        self.q_network = network(51, 9).to(self.device)
        self.target_network = network(51, 9).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        if replay_mode not in {'manuscript', 'uniform', 'per'}:
            raise ValueError("replay_mode must be manuscript, uniform, or per")
        if side_buffer:
            raise ValueError('The released method has one FIFO buffer and no side pool')
        self.replay_mode = replay_mode
        if shared_replay_buffer is not None:
            raise ValueError("Use the single manuscript replay buffer; legacy shared buffers are not supported")
        if replay_mode == 'manuscript':
            self.replay = ManuscriptReplay(memory_size, self.w_reward, self.w_TD_error)
        elif replay_mode == 'uniform':
            self.replay = UniformReplay(memory_size)
        else:
            self.replay = StandardPER(memory_size, alpha=alpha, beta=beta)
        self.memory = self.replay.memory
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.action_smoothness_penalty = action_transition_matrix()

    def reset_episode(self):
        self.last_actions.clear()
        self.stuck_counter = 0
        self.last_max_q = None
        self.escape_mode = False
        self.escape_steps = 0

    def update_target_network(self, tau=0.01):
        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_((1.0 - tau) * target_param.data + tau * local_param.data)

    def select_action(self, state):
        """Algorithm 3: epsilon-greedy action selection, without Q reshaping."""
        if random.random() < self.epsilon:
            action = random.randrange(9)
        else:
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action = int(self.q_network(state_tensor.unsqueeze(0)).argmax(1).item())
        self.last_actions.append(action)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        # Apply the explicit scalar once, at this single storage boundary.
        if self.reward_scale != 1.0:
            reward = self.reward_scale * float(reward)
        # GridWorld already returns all six reward terms. The fallback is only
        # for external legacy environments that do not include SMR.
        adjusted_reward = float(reward)
        if not getattr(self.env, 'reward_includes_smoothness', False) and len(self.last_actions) >= 2:
            previous_action = self.last_actions[-2]
            adjusted_reward -= self.smoothing_factor * self.action_smoothness_penalty[previous_action, action]
        state = state.detach().cpu().numpy() if isinstance(state,torch.Tensor) else np.asarray(state)
        next_state = next_state.detach().cpu().numpy() if isinstance(next_state,torch.Tensor) else np.asarray(next_state)
        with torch.no_grad():
            metrics = self.compute_bellman_metrics(
                [(state, action, adjusted_reward, next_state, done, 0.)])
            td_error = float(abs(metrics['td_errors'][0]))
        self.replay.append(state,action,adjusted_reward,next_state,done,td_error)
        info = getattr(self.env, 'last_info', {})
        category = ('collision' if info.get('collision') else 'success' if info.get('success')
                    else 'waypoint_arrival' if info.get('waypoint_arrival') else 'other')
        self.replay_events.append(category)
        self.environment_steps += 1

    def compute_bellman_metrics(self,batch, importance_weights=None):
        states,actions,rewards,next_states,dones,_ = zip(*batch)
        states=torch.as_tensor(np.array(states),dtype=torch.float32,device=self.device)
        next_states=torch.as_tensor(np.array(next_states),dtype=torch.float32,device=self.device)
        actions=torch.as_tensor(actions,dtype=torch.long,device=self.device)
        rewards=torch.as_tensor(rewards,dtype=torch.float32,device=self.device)
        dones=torch.as_tensor(dones,dtype=torch.float32,device=self.device)
        with torch.no_grad():
            if self.double_q:
                # Double DQN with min clipping: the online network picks the
                # action, the target network values it, and the result is clipped
                # against the target network's own maximum.
                pick = self.q_network(next_states).max(1)[1].unsqueeze(1)
                valued = self.target_network(next_states).gather(1, pick).squeeze(1)
                bootstrap = torch.min(valued, self.target_network(next_states).max(1)[0])
            else:
                # Optional target-network-maximum ablation.
                bootstrap = self.target_network(next_states).max(1).values
            target = rewards + self.gamma * (1. - dones) * bootstrap
        current=self.q_network(states).gather(1,actions[:,None]).flatten()
        td_error=target-current
        if self.huber:
            per_sample = nn.SmoothL1Loss(reduction='none')(current, target)
        else:
            per_sample = td_error.square()
        if importance_weights is not None:
            weights = torch.as_tensor(importance_weights, dtype=per_sample.dtype, device=self.device)
            per_sample = per_sample * weights
        loss = per_sample.mean()
        return {'loss_tensor':loss,'td_errors':td_error.detach().cpu().numpy(),
                'target_q_values':target.detach().cpu().numpy()}

    def train(self,return_metrics=False):
        if len(self.memory) < self.batch_size:
            return None
        indices,batch=self.replay.sample(self.batch_size)
        weights = None
        if self.importance_sampling:
            beta = self.beta + (1. - self.beta) * min(1., self.training_steps / self.per_beta_steps)
            probabilities = self.replay.probabilities()
            weights = (probabilities[np.asarray(indices, dtype=int)] / probabilities.min()) ** (-beta)
        metrics=self.compute_bellman_metrics(batch, weights)
        for i in indices:
            if int(i) < len(self.replay_events):
                self.sampled_event_counts[self.replay_events[int(i)]] += 1
        # Algorithm 3, line 32: refresh with current pre-update TD errors.
        self.replay.refresh(indices,metrics['td_errors'])
        loss=metrics['loss_tensor']
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(),1.)
        self.optimizer.step()
        self.training_steps+=1
        if self.training_steps % self.update_target_freq == 0:
            self.update_target_network(tau=self.target_tau)
        if return_metrics:
            return {'loss':float(loss.detach()),'abs_td_error_mean':float(np.abs(metrics['td_errors']).mean())}
        return loss.detach()

    def update(self):
        self.update_target_network()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, file_path="dqn_model.pth"):
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'code_protocol': CODE_PROTOCOL,
            'training_source_sha256': getattr(self, 'training_source_sha256', {}),
            'learner_specification': self.learner_specification(),
            'training_seed': self.training_seed,
            'training_configuration': self.training_configuration,
            'training_steps': self.training_steps,
            'environment_specification': {key: getattr(self.env, key) for key in
                ('collision_protocol', 'prediction_mode', 'prediction_enabled', 'avoidance_reward_form', 'reward_scales',
                 'subgoal_tolerance', 'robot_radius', 'cell_size', 'extra_margin') if hasattr(self.env, key)},
            'environment_steps': self.environment_steps,
            'gamma': self.gamma,
            'target_tau': self.target_tau,
            'smoothing_factor': self.smoothing_factor,
            'replay_protocol': (
                REPLAY_PROTOCOL if self.replay_mode == 'manuscript'
                else self.replay_mode
            ),
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'action_consistency': self.action_consistency,
            'stuck_threshold': self.stuck_threshold,
            'architecture': self.architecture,
        }, file_path)
        print(f"Model saved to {file_path}")

    def learner_specification(self):
        return dict(reward_scale=self.reward_scale, reward_scaling_applications=1,
                    double_q=self.double_q, huber=self.huber,
                    gamma=self.gamma,
                    replay_mode=self.replay_mode, importance_sampling=self.importance_sampling,
                    alpha=self.alpha, beta=self.beta, per_beta_steps=self.per_beta_steps,
                    batch_size=self.batch_size, architecture=self.architecture)

    def load_model(self, file_path="dqn_model.pth", *, allow_historical=False):
        load_kwargs = {"map_location": self.device}
        # torch.load will flip the default of weights_only in future versions; set explicitly when supported.
        if "weights_only" in inspect.signature(torch.load).parameters:
            load_kwargs["weights_only"] = False

        checkpoint = torch.load(file_path, **load_kwargs)
        if checkpoint.get('code_protocol') != CODE_PROTOCOL and not allow_historical:
            raise ValueError('Historical checkpoint uses a different state/reward protocol. '
                             'Retrain for ' + CODE_PROTOCOL + ', or explicitly set allow_historical=True '
                             'for a separately labelled transfer diagnostic.')
        expected_replay = REPLAY_PROTOCOL if self.replay_mode == 'manuscript' else self.replay_mode
        if checkpoint.get('code_protocol') == CODE_PROTOCOL and checkpoint.get('replay_protocol') != expected_replay:
            raise ValueError('checkpoint replay mode differs from the agent')
        self.loaded_code_protocol = checkpoint.get('code_protocol', 'historical')
        self.loaded_training_seed = checkpoint.get('training_seed')
        self.loaded_learner_specification = checkpoint.get('learner_specification', {})
        if checkpoint.get('code_protocol') == CODE_PROTOCOL:
            spec = self.loaded_learner_specification
            if spec.get('reward_scaling_applications') != 1:
                raise ValueError('Checkpoint does not certify single reward scaling')
            for key in ('reward_scale', 'double_q', 'huber',
                        'importance_sampling', 'alpha', 'beta', 'per_beta_steps'):
                setattr(self, key, spec[key])
        self.loaded_environment_specification = checkpoint.get('environment_specification', {})
        self.training_source_sha256 = checkpoint.get('training_source_sha256', {})
        if not allow_historical:
            root = Path(__file__).resolve().parent
            for name, expected in self.training_source_sha256.items():
                if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
                    raise ValueError('Training source differs from current source: ' + name +
                                     '. Use the frozen training snapshot or retrain.')
        self.environment_steps = checkpoint.get('environment_steps', 0)
        self.gamma = checkpoint.get('gamma', self.gamma)
        self.smoothing_factor = checkpoint.get('smoothing_factor', self.smoothing_factor)
        saved_architecture = checkpoint.get('architecture')
        if saved_architecture is not None and saved_architecture != self.architecture:
            raise ValueError(f"checkpoint architecture {saved_architecture!r} does not match agent {self.architecture!r}")
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        if 'action_consistency' in checkpoint:
            self.action_consistency = checkpoint['action_consistency']
        if 'stuck_threshold' in checkpoint:
            self.stuck_threshold = checkpoint['stuck_threshold']
        print(f"Model loaded from {file_path}")
