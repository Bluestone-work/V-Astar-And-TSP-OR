"""Discrete-to-continuous transfer: interval-collision audit and closed loop.

`geometry` holds the two continuous-time primitives the grid evaluator lacks,
`timebase` fixes the one time base both evaluators are read against, and
`obstacle_script` exposes the scripted obstacle motion as a function of the
period index so the discrete and continuous runs can share a scenario.
"""
