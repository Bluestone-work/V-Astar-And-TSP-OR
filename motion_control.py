"""Optional downstream unicycle tracker for the nine reference displacements.

This is a simulated tracking adapter for Section III-B, not a validated robot
driver. Coordinates are physical (row, column); heading is measured from row.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class MotionLimits:
    max_linear_speed: float = 1.
    max_angular_speed: float = 2.
    max_linear_acceleration: float = 1.
    max_angular_acceleration: float = 4.

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in vars(self).values()):
            raise ValueError('motion limits must be finite and positive')


class UnicycleTracker:
    def __init__(self, pose, limits=MotionLimits()):
        self.pose = np.asarray(pose, dtype=float).copy()
        if self.pose.shape != (3,) or not np.isfinite(self.pose).all():
            raise ValueError('pose must be (row, column, heading)')
        self.limits = limits
        self.velocity = np.zeros(2)

    def advance(self, target, dt, measured_pose=None):
        """Track a physical point; target=None requests controlled deceleration.

        Returns the old and new physical positions and an arc-to-chord bound,
        usable as an additional clearance margin for swept collision checks.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        if dt*self.limits.max_angular_speed > math.pi:
            raise ValueError('dt is too large for the swept arc-to-chord bound')
        measured = self.pose if measured_pose is None else np.asarray(measured_pose)
        desired = np.zeros(2)
        if target is not None:
            delta = np.asarray(target) - measured[:2]
            distance = float(np.linalg.norm(delta))
            error = (math.atan2(delta[1], delta[0]) - measured[2] + math.pi) % (2*math.pi) - math.pi
            desired[0] = min(self.limits.max_linear_speed,
                             math.sqrt(2*self.limits.max_linear_acceleration*distance)) * max(0., math.cos(error))
            desired[1] = float(np.clip(3*error, -self.limits.max_angular_speed, self.limits.max_angular_speed))
        slew = np.array([self.limits.max_linear_acceleration, self.limits.max_angular_acceleration])*dt
        self.velocity += np.clip(desired-self.velocity, -slew, slew)
        v, omega = self.velocity
        old = self.pose[:2].copy()
        theta = self.pose[2]
        if abs(omega) < 1e-10:
            self.pose[:2] += v*dt*np.array([math.cos(theta), math.sin(theta)])
            sagitta = 0.
        else:
            self.pose[:2] += v/omega*np.array([math.sin(theta+omega*dt)-math.sin(theta),
                                             math.cos(theta)-math.cos(theta+omega*dt)])
            sagitta = abs(v/omega)*(1-math.cos(abs(omega)*dt/2))
        self.pose[2] = (theta+omega*dt+math.pi) % (2*math.pi) - math.pi
        return old, self.pose[:2].copy(), sagitta
