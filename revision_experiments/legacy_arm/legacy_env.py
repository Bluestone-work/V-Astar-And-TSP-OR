"""GridWorld with the pre-alignment collision rule restored.

The public `environment.py` rejects a diagonal move when either orthogonal
neighbour is occupied, i.e. it forbids cutting the corner between two
obstacles. The aligned `environment.py` keeps that test but only inside
`_motion_safe()`, which global planning uses to validate a route segment; the
local policy's `step()` no longer applies it, so at execution time the robot may
squeeze through a diagonal gap. The class attribute
`collision_protocol = 'github-endpoint-only-v1'` therefore overstates the match:
endpoint-only semantics are shared, the corner-cut rule is not.

Restoring it matters beyond reproduction. A diagonal gap between two occupied
cells is not physically traversable by a robot with any footprint, so allowing
it is one of the places where "collision-free" claims outrun the model -- the
R2-1 footprint objection and the R2-6 request to moderate deployment language.
The legacy rule is the more conservative of the two.

Implementation note: rather than duplicate `step()` and risk drifting from its
reward bookkeeping, a corner-cutting move is made to fail the parent's own
static-occupancy branch. `get_state()` is called first so that the `get_state()`
inside `step()` finds `_state_step == steps` and does not rebuild the grid out
from under the injected cell; `step()` rebuilds it at the end regardless, so the
injection never outlives the call.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from environment import ACTIONS, GridWorld                      # noqa: E402


class LegacyCollisionGridWorld(GridWorld):
    """Endpoint-only collision plus the legacy diagonal corner-cut rejection."""

    collision_protocol = 'github-endpoint-and-corner-v1'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.motion_model != 'grid':
            # _cuts_corner reads integer cell offsets from ACTIONS directly.
            raise ValueError('LegacyCollisionGridWorld supports only motion_model="grid"')
        self.corner_cut_rejections = 0

    def _cuts_corner(self, action):
        delta = np.asarray(ACTIONS[int(action)], dtype=int)
        dr, dc = int(delta[0]), int(delta[1])
        if not (dr and dc):
            return False, None
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            return False, None
        if self.grid_map[r + dr, c] == 1 or self.grid_map[r, c + dc] == 1:
            return True, (nr, nc)
        return False, None

    def step(self, action):
        self.get_state()                      # pin _state_step to self.steps
        cuts, target = self._cuts_corner(action)
        if not cuts:
            return super().step(action)
        self.corner_cut_rejections += 1
        restore = self.grid_map[target]
        self.grid_map[target] = 1             # parent's own branch reports it
        try:
            return super().step(action)
        finally:
            if self.grid_map[target] == 1:
                self.grid_map[target] = restore
