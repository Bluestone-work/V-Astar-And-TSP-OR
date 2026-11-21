"""Make every arm train on the identical sequence of environments.

`train.train_agent` calls `env.reset_dynamic(seed=None)` once per episode, and
with `seed=None` the obstacle layout is drawn from the global RNG. The same
global RNG also serves epsilon-greedy exploration (`random.random()`,
`random.randrange(9)` in `select_action`), so two arms whose episodes differ in
length -- or that take a different number of random actions -- desynchronise the
stream immediately. Each arm therefore trains on a DIFFERENT sequence of
layouts, and any measured difference between arms is confounded with that.

Evaluation never had this problem: it draws scenarios from `eval_seed + i`.
This class gives training the same property. Passing an integer to
`reset_dynamic` reseeds both `random` and `numpy.random`, so with an
episode-indexed seed every arm sees

  * the identical obstacle layout at episode k, and
  * the identical RNG state at the start of episode k

while the curriculum schedule and mixed-level choices were already identical
(`curriculum_rng` is a separate `Random(seed)`).

Within an episode the streams still diverge, because the policies differ and
consume different numbers of draws -- that is the behaviour under test, not a
confound.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from environment import ACTIONS, GridWorld                               # noqa: E402
from revision_experiments.legacy_arm.legacy_env import (                 # noqa: E402
    LegacyCollisionGridWorld)

#: Disjoint from the evaluation pool (eval_seed 10000 + i) so no training
#: layout can coincide with a held-out evaluation scenario.
TRAIN_ENV_SEED_BASE = 777000

#: Stride between training seeds. The layout sequence has to depend on the
#: training seed, otherwise every seed trains on the identical environments and
#: the cross-seed spread collapses to network initialisation and exploration
#: noise alone -- which would understate exactly the reproducibility variance
#: Reviewer 2 comment 4 and Reviewer 3 comment 4 ask about. The stride exceeds
#: any episode budget used here, so sequences from different seeds never
#: overlap. Within one seed every arm still sees the identical sequence, which
#: is what removes the arm-to-arm confound.
TRAIN_SEED_STRIDE = 100_000


def layout_seed_base(training_seed):
    """Layout seed base for a given training seed."""
    return TRAIN_ENV_SEED_BASE + int(training_seed) * TRAIN_SEED_STRIDE


class _FixedSequenceMixin:
    """reset_dynamic(None) becomes reset_dynamic(base + episode_index)."""

    train_env_seed_base = TRAIN_ENV_SEED_BASE

    def __init__(self, *args, train_env_seed_base=None, **kwargs):
        super().__init__(*args, **kwargs)
        if train_env_seed_base is not None:
            self.train_env_seed_base = int(train_env_seed_base)
        self.train_episode_index = 0

    def reset_dynamic(self, seed=None):
        if seed is None:
            seed = self.train_env_seed_base + self.train_episode_index
            self.train_episode_index += 1
        return super().reset_dynamic(seed)


class FixedSequenceGridWorld(_FixedSequenceMixin, GridWorld):
    """Endpoint-only collision, fixed training sequence."""


class FixedSequenceLegacyCollisionGridWorld(_FixedSequenceMixin,
                                            LegacyCollisionGridWorld):
    """Legacy corner-cut rejection, fixed training sequence."""
