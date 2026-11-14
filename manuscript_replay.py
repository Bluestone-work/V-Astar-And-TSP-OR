"""Single-buffer Eq. (19) and reference sampling distributions.

Eq. (19) is directly normalized without an extra exponent or side pool.
Loss/target settings and the independent IS switch live in agents.py.
"""
from collections import deque
import numpy as np


def dual_priority(reward, td_error, reward_weight=.25, td_weight=.75, epsilon=1e-5):
    """Eq. (19), lambda=1; negative rewards contribute zero to the reward term."""
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError('epsilon must be finite and positive')
    if min(reward_weight, td_weight) < 0 or not np.isfinite([reward_weight,td_weight]).all():
        raise ValueError('priority weights must be finite and non-negative')
    reward, td_error = np.asarray(reward,dtype=float), np.asarray(td_error,dtype=float)
    if not np.isfinite(reward).all() or not np.isfinite(td_error).all():
        raise ValueError('reward and TD error must be finite')
    return (reward_weight*np.log1p(np.maximum(reward,0.)) +
            td_weight*np.log1p(np.abs(td_error)) + epsilon)


class ManuscriptReplay:
    def __init__(self,capacity=10000,reward_weight=.25,td_weight=.75,epsilon=1e-5):
        if not isinstance(capacity,int) or capacity < 1:raise ValueError('capacity must be positive')
        dual_priority(0.,0.,reward_weight,td_weight,epsilon)
        self.memory=deque(maxlen=capacity)
        self.reward_weight=reward_weight;self.td_weight=td_weight;self.epsilon=epsilon
    def __len__(self):return len(self.memory)
    def priority(self,reward,error):
        return dual_priority(reward,error,self.reward_weight,self.td_weight,self.epsilon)
    def append(self,state,action,reward,next_state,done,td_error):
        self.memory.append((np.array(state,copy=True),int(action),float(reward),
                            np.array(next_state,copy=True),bool(done),float(self.priority(reward,td_error))))
    def probabilities(self):
        if not self.memory:raise ValueError('cannot sample an empty replay buffer')
        p=np.asarray([item[5] for item in self.memory],dtype=float)
        if not np.isfinite(p).all() or np.any(p<=0):raise ValueError('priorities must be positive and finite')
        return p/p.sum()
    def sample(self,batch_size):
        if not isinstance(batch_size,int) or batch_size<1:raise ValueError('batch_size must be positive')
        indices=np.random.choice(len(self.memory),batch_size,replace=True,p=self.probabilities())
        return indices,[self.memory[int(i)] for i in indices]
    def refresh(self,indices,td_errors):
        if len(indices)!=len(td_errors):raise ValueError('one TD error is required per sampled index')
        for index,error in zip(indices,td_errors):
            index=int(index);item=self.memory[index]
            self.memory[index]=item[:5]+(float(self.priority(item[2],error)),)


class UniformReplay:
    """Uniform replay baseline with the same transition storage interface."""

    def __init__(self, capacity=10000):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be positive")
        from collections import deque
        self.memory = deque(maxlen=capacity)

    def __len__(self):
        return len(self.memory)

    def append(self, state, action, reward, next_state, done, td_error=0.0):
        self.memory.append((np.array(state, copy=True), int(action), float(reward),
                            np.array(next_state, copy=True), bool(done), 1.0))

    def sample(self, batch_size):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be positive")
        indices = np.random.choice(len(self.memory), batch_size, replace=True)
        return indices, [self.memory[int(i)] for i in indices]

    def probabilities(self):
        if not self.memory:
            raise ValueError("cannot sample an empty replay buffer")
        return np.full(len(self.memory), 1.0 / len(self.memory), dtype=float)

    def refresh(self, indices, td_errors):
        # Uniform replay intentionally ignores TD-error refreshes.
        if len(indices) != len(td_errors):
            raise ValueError("one TD error is required per sampled index")


class StandardPER:
    """Conventional proportional PER baseline.

    Sampling uses ``(|delta|+epsilon)^alpha``. The learner applies normalized
    ``(N*P(i))**(-beta)`` weights and anneals beta to one.
    """

    def __init__(self, capacity=10000, alpha=0.6, beta=0.4, epsilon=1e-5):
        if not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be positive")
        if not np.isfinite([alpha, beta, epsilon]).all() or alpha < 0 or not 0 <= beta <= 1 or epsilon <= 0:
            raise ValueError("invalid PER parameters")
        from collections import deque
        self.memory = deque(maxlen=capacity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.epsilon = float(epsilon)

    def __len__(self):
        return len(self.memory)

    def _priority(self, td_error):
        value = abs(float(td_error)) + self.epsilon
        if not np.isfinite(value):
            raise ValueError("TD error must be finite")
        return value ** self.alpha

    def append(self, state, action, reward, next_state, done, td_error=0.0):
        self.memory.append((np.array(state, copy=True), int(action), float(reward),
                            np.array(next_state, copy=True), bool(done),
                            float(self._priority(td_error))))

    def probabilities(self):
        if not self.memory:
            raise ValueError("cannot sample an empty replay buffer")
        p = np.asarray([item[5] for item in self.memory], dtype=float)
        return p / p.sum()

    def sample(self, batch_size):
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be positive")
        indices = np.random.choice(len(self.memory), batch_size, replace=True,
                                   p=self.probabilities())
        return indices, [self.memory[int(i)] for i in indices]

    def importance_weights(self, indices, beta=None):
        beta = self.beta if beta is None else float(beta)
        if not 0 <= beta <= 1:
            raise ValueError('beta must be in [0,1]')
        p = self.probabilities()
        # Normalize against the largest weight in the full buffer, not only
        # the sampled minibatch, so identical items keep identical weights.
        weights = (len(p) * p[np.asarray(indices, dtype=int)]) ** (-beta)
        return weights / ((len(p) * p.min()) ** (-beta))

    def refresh(self, indices, td_errors):
        if len(indices) != len(td_errors):
            raise ValueError("one TD error is required per sampled index")
        for index, error in zip(indices, td_errors):
            index = int(index)
            item = self.memory[index]
            self.memory[index] = item[:5] + (float(self._priority(error)),)
