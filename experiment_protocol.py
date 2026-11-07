"""Paired training layouts and policy-independent obstacle-motion streams."""
import random
import environment as environment_module
from environment import GridWorld


def layout_seed_base(training_seed):
    return 777000 + int(training_seed) * 100000


class SyncedGridWorld(GridWorld):
    """Same training seed gives the same layout/motion prefix across variants."""
    train_env_seed_base = 777000
    train_episode_index = 0
    _obstacle_rng = None

    def __init__(self, *args, **kwargs):
        self.train_episode_index = 0
        super().__init__(*args, **kwargs)

    def reset_dynamic(self, seed=None):
        if seed is None:
            seed = self.train_env_seed_base + self.train_episode_index
            self.train_episode_index += 1
        result = super().reset_dynamic(seed)
        self.ensure_motion_stream(seed)
        return result

    def ensure_motion_stream(self, seed):
        self._motion_base_seed = int(seed) + 10000000
        self._obstacle_rng = random.Random(self._motion_base_seed)

    def step(self, action):
        base = getattr(self, '_motion_base_seed', self.train_env_seed_base + 10000000)
        self._obstacle_rng = random.Random(hash((base, int(self.steps))))
        return super().step(action)

    def _update_dynamic_obstacle(self, obstacle):
        if self._obstacle_rng is None:
            return super()._update_dynamic_obstacle(obstacle)
        previous = environment_module.random
        environment_module.random = self._obstacle_rng
        try:
            return super()._update_dynamic_obstacle(obstacle)
        finally:
            environment_module.random = previous
