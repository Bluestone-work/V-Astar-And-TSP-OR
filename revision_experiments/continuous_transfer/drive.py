"""Drive models for the continuous transfer test.

The policy has no heading state: the 7x7 occupancy window is map-aligned and
indexed by position alone (`environment.py:445-474`), the goal feature is an
absolute bearing (`environment.py:502`), and `heading`, `theta` and
`orientation` appear nowhere in `environment.py`, `agents.py` or `models.py`.
Its action is a displacement in one of eight directions. That action space is
holonomic by construction.

Pairing it with the nonholonomic unicycle of `motion_control.py` therefore
charges the policy for a state it never observed and cannot plan around. The
telemetry rendering makes the cost visible: every decision period is a turn
under the `cos(psi)` gate, an acceleration to `v_max`, and a braked stop, which
is the whole of the measured achieved fraction lambda = 0.77 to 0.83.

`HolonomicTracker` is the matched model: an omnidirectional base, bounded in
speed and acceleration exactly as the unicycle is, with no heading constraint on
translation. It is not an unconstrained robot -- the acceleration bound and the
stopping-distance term both remain, so lambda stays below 1 -- and the footprint
and swept collision test are unchanged.

Both models are reported. Their difference is the cost of nonholonomy, which is
what Reviewer 2's request for "robot motion constraints" actually asks about;
dropping the unicycle entirely would remove the constraint rather than quantify
it.
"""
from __future__ import annotations

import math

import numpy as np

from motion_control import MotionLimits, UnicycleTracker


class HolonomicTracker:
    """Omnidirectional base, bounded speed and acceleration, free heading.

    Interface matches `UnicycleTracker.advance` so the closed loop is unchanged.
    Heading is carried for reporting only: it follows the velocity direction and
    never gates translation.
    """

    def __init__(self, pose, limits=MotionLimits()):
        self.pose = np.asarray(pose, dtype=float).copy()
        if self.pose.shape != (3,) or not np.isfinite(self.pose).all():
            raise ValueError('pose must be (row, column, heading)')
        self.limits = limits
        self.velocity = np.zeros(2)

    def telemetry(self):
        """(linear speed, turn rate) for logging; a holonomic base never turns."""
        return float(np.linalg.norm(self.velocity)), 0.

    #: Under `cruise`, stop commanding speed inside this distance of the target
    #: rather than braking to it. Without a deadband the robot overshoots the
    #: waypoint and oscillates around it for the rest of the period.
    DEADBAND = .15

    def advance(self, target, dt, measured_pose=None, brake=True):
        """Track a point; target=None requests controlled deceleration.

        `brake=True` keeps the unicycle's speed envelope, whose
        `sqrt(2 a_max d)` term is active only inside
        `v_max**2 / (2 a_max)` = 0.5 cells, i.e. over the second half of a
        one-cell hop. `brake=False` treats the waypoint as a point to pass
        through rather than to stop at, which is what the grid action means:
        the policy issues another action next period and never intended to come
        to rest there.

        Returns old position, new position and an arc-to-chord bound, which is
        zero because every substep is a straight segment.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        measured = self.pose if measured_pose is None else np.asarray(measured_pose)
        desired = np.zeros(2)
        if target is not None:
            delta = np.asarray(target, dtype=float) - measured[:2]
            distance = float(np.linalg.norm(delta))
            if distance > (1e-12 if brake else self.DEADBAND):
                # Same speed envelope as the unicycle, minus the cos(heading
                # error) factor, because there is no heading to err.
                speed = (min(self.limits.max_linear_speed,
                             math.sqrt(2 * self.limits.max_linear_acceleration * distance))
                         if brake else self.limits.max_linear_speed)
                desired = speed * delta / distance
        change = desired - self.velocity
        magnitude = float(np.linalg.norm(change))
        limit = self.limits.max_linear_acceleration * dt
        if magnitude > limit:
            change *= limit / magnitude
        self.velocity += change
        old = self.pose[:2].copy()
        self.pose[:2] += self.velocity * dt
        speed = float(np.linalg.norm(self.velocity))
        if speed > 1e-12:
            self.pose[2] = math.atan2(self.velocity[1], self.velocity[0])
        return old, self.pose[:2].copy(), 0.


def make_tracker(kind, pose, limits=MotionLimits()):
    """Build the drive model named by `--drive`."""
    if kind == 'unicycle':
        tracker = UnicycleTracker(pose, limits)
        tracker.telemetry = lambda t=tracker: (float(t.velocity[0]), float(t.velocity[1]))
        # motion_control.py is shared with experiments/kinematic_closed_loop.py,
        # so the unicycle keeps its braking envelope and simply ignores the flag.
        inner = tracker.advance
        tracker.advance = lambda target, dt, measured_pose=None, brake=True: inner(
            target, dt, measured_pose)
        return tracker
    if kind == 'holonomic':
        return HolonomicTracker(pose, limits)
    raise ValueError("drive must be 'unicycle' or 'holonomic'")
