"""The one time base the discrete and continuous evaluators are read against.

Section V-G makes two claims about the decision period that the code has to
honour for its numbers to mean what the text says:

* a period shorter than `sqrt(2) / v_max` cannot complete a diagonal action, so
  the robot is issued a fresh action from a pose short of the previous target;
* the obstacle field advances a fixed number of cells per period, so the period
  also sets the obstacle speed. Choosing the period for the robot alone silently
  chooses a speed handicap for the obstacles.

`experiments/kinematic_closed_loop.py` takes `--dt` and `--substeps` as free
parameters and never computes the resulting ratio, which is how a configuration
with obstacles 2 to 8 times faster than the robot was run and initially read as
a kinodynamic effect. This module makes the period a derived quantity and the
ratio a reported one.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math


@dataclass(frozen=True)
class TimeBase:
    """Integration step, decision period and the obstacle speed they imply."""

    dt: float = .02
    substeps: int = 71
    max_linear_speed: float = 1.
    #: Cells the obstacle field advances per decision period. One reproduces
    #: the grid evaluator, where robot and obstacles each take one action.
    obstacle_cells_per_period: float = 1.

    def __post_init__(self):
        values = (self.dt, self.max_linear_speed, self.obstacle_cells_per_period)
        if any(not math.isfinite(v) or v <= 0 for v in values):
            raise ValueError('dt, speed and obstacle displacement must be positive')
        if self.substeps < 1:
            raise ValueError('substeps must be at least 1')

    @property
    def period(self):
        return self.dt * self.substeps

    @property
    def minimum_period(self):
        """Shortest period in which a diagonal action can be completed."""
        return math.sqrt(2.) / self.max_linear_speed

    @property
    def completes_diagonal(self):
        return self.period >= self.minimum_period

    @property
    def obstacle_speed(self):
        """Obstacle speed in cells per second, the physical reading."""
        return self.obstacle_cells_per_period / self.period

    def speed_ratio(self, achieved_fraction=1.):
        """Eq. (speed_ratio): the obstacle-to-robot handicap index.

        Normalised so that the grid evaluator, where both advance one action per
        step, is 1. `achieved_fraction` is lambda, the measured share of the
        commanded displacement the tracker actually covers; pass the value
        measured in the run rather than assuming 1.
        """
        if not math.isfinite(achieved_fraction) or achieved_fraction <= 0:
            raise ValueError('achieved fraction must be positive')
        reachable = min(1., self.period * self.max_linear_speed / math.sqrt(2.))
        return self.obstacle_cells_per_period / (reachable * achieved_fraction)

    def as_dict(self, achieved_fraction=1.):
        record = asdict(self)
        record.update(period=self.period, minimum_period=self.minimum_period,
                      completes_diagonal=self.completes_diagonal,
                      obstacle_speed_cells_per_second=self.obstacle_speed,
                      speed_ratio=self.speed_ratio(achieved_fraction),
                      achieved_fraction=achieved_fraction)
        return record


def substeps_for(dt=.02, max_linear_speed=1.):
    """Fewest substeps whose period can complete a diagonal action.

    At the Section V-G values this returns 71, i.e. T = 1.42 s.
    """
    if not (math.isfinite(dt) and dt > 0 and math.isfinite(max_linear_speed)
            and max_linear_speed > 0):
        raise ValueError('dt and speed must be positive and finite')
    return int(math.ceil(math.sqrt(2.) / (max_linear_speed * dt)))
