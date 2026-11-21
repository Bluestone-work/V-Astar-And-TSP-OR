"""Identical obstacle trajectories across arms, not just identical layouts.

Fixing `reset_dynamic` pins the obstacle layout at the start of each episode, but
the swarm keeps moving, and `_update_dynamic_obstacle` draws from the SAME global
RNG that epsilon-greedy exploration uses (`random.random()` for the turn
probability, `random.shuffle` inside `_choose_new_direction`). Two arms that take
a different number of random actions therefore see the obstacles follow different
paths from the same starting positions.

That this is fixable at all depends on a property of the environment:
`_is_passable` has the agent-position test commented out, with the note
"Dynamic obstacles should not actively avoid the agent". Obstacle motion is a
function of the static map, the swarm's own configuration and its RNG draws --
never of the robot. So the swarm evolution is a script that can be replayed
exactly.

This mixin gives that script its own `random.Random`, reseeded per episode from
the same episode-indexed seed used for the layout. The module-level `random`
reference inside `environment` is swapped only for the duration of one obstacle
update, so no other randomness is touched.

What stays different between arms, correctly: the swarm only advances on steps
where the robot did not collide, so an arm that crashes early simply experiences
a shorter prefix of the same scripted motion. At any step k reached by both arms,
the obstacles are in identical positions.
"""
from __future__ import annotations

import random as _random_module
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import environment as _env_mod                                           # noqa: E402
from environment import GridWorld                                        # noqa: E402
from revision_experiments.legacy_arm.legacy_env import (                 # noqa: E402
    LegacyCollisionGridWorld)
from revision_experiments.legacy_arm.fixed_sequence_env import (         # noqa: E402
    TRAIN_ENV_SEED_BASE, _FixedSequenceMixin)

#: Offset so the obstacle-motion stream never coincides with the layout seed.
MOTION_SEED_OFFSET = 10_000_000


class SyncedObstacleMixin:
    """Route obstacle-motion randomness through a dedicated per-episode stream.

    The counters are class attributes rather than instance state set in
    `reset_dynamic`, because the evaluation path reaches `step()` through
    `apply_environment_config` without going through `reset_dynamic` first. The
    earlier version raised AttributeError on the first obstacle update of every
    level after the first, which silently stopped each evaluation at level 1.
    """

    _obstacle_rng = None
    obstacle_motion_draws = 0

    def reset_dynamic(self, seed=None):
        result = super().reset_dynamic(seed)
        # _FixedSequenceMixin has already incremented the counter, so the episode
        # that was just built is index-1.
        idx = getattr(self, 'train_episode_index', 0)
        base = getattr(self, 'train_env_seed_base', TRAIN_ENV_SEED_BASE)
        self._obstacle_rng = _random_module.Random(base + max(0, idx - 1) + MOTION_SEED_OFFSET)
        self.obstacle_motion_draws = 0
        return result

    def _episode_motion_seed(self):
        idx = getattr(self, 'train_episode_index', 0)
        base = getattr(self, 'train_env_seed_base', TRAIN_ENV_SEED_BASE)
        return base + max(0, idx - 1) + MOTION_SEED_OFFSET

    def step(self, action):
        """Reseed the motion stream from (episode, step) before every update.

        Routing obstacle motion through a dedicated stream was not sufficient on
        its own: the episode still diverged at step 9, because other code reached
        from `step()` -- `get_state()` builds the prediction-augmented grid under
        `history_voting` -- consumes a policy-dependent number of draws, which
        shifts where the motion stream is read from. Reseeding per step makes the
        obstacle update at step k a pure function of (scenario, k), so it cannot
        be perturbed by anything the policy does.
        """
        self._obstacle_rng = _random_module.Random(
            (self._episode_motion_seed(), int(self.steps)).__hash__())
        return super().step(action)

    def ensure_motion_stream(self, seed):
        """Pin the motion stream explicitly, for callers that skip reset_dynamic."""
        self.train_env_seed_base = int(seed)
        self.train_episode_index = 1
        self._obstacle_rng = _random_module.Random(
            int(seed) + MOTION_SEED_OFFSET)
        self.obstacle_motion_draws = 0

    def _update_dynamic_obstacle(self, obstacle):
        rng = getattr(self, '_obstacle_rng', None)
        if rng is None:
            return super()._update_dynamic_obstacle(obstacle)
        saved = _env_mod.random
        _env_mod.random = rng
        self.obstacle_motion_draws += 1
        try:
            return super()._update_dynamic_obstacle(obstacle)
        finally:
            _env_mod.random = saved


class SyncedGridWorld(SyncedObstacleMixin, _FixedSequenceMixin, GridWorld):
    """Fixed layouts, replayable obstacle motion, endpoint-only collision."""


class SyncedLegacyCollisionGridWorld(SyncedObstacleMixin, _FixedSequenceMixin,
                                     LegacyCollisionGridWorld):
    """Same, with the legacy diagonal corner-cut rejection."""
