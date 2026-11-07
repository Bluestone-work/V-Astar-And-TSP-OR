"""Executable boundary for the post-review manuscript-aligned implementation."""

CODE_PROTOCOL = 'jas'
REPLAY_PROTOCOL = 'eq19-single-buffer-v3'
ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1),
           (-1, 1), (1, -1), (1, 1), (0, 0))


# Same nine directions, each scaled to unit Euclidean length, so that every
# action covers distance 1 instead of 1 axially and sqrt(2) diagonally.  The
# robot pose becomes continuous: a diagonal action moves 1/sqrt(2) per axis.
_INV_R2 = 1.0 / 2.0 ** 0.5
UNIT_ACTIONS = tuple(
    (float(a), float(b)) if a == 0 or b == 0 else (a * _INV_R2, b * _INV_R2)
    for a, b in ACTIONS
)

ACTION_SETS = {'grid': ACTIONS, 'unit': UNIT_ACTIONS}


def action_transition_matrix():
    """Table II, with reversal taking precedence; unspecified pairs cost .1.

    Axial/diagonal transitions and stop->stop use .1. All eight directed
    reversals, including diagonal reversals, cost .3. This resolves the
    overlap in Table II without silently changing its four stated values.
    """
    import numpy as np
    matrix = np.full((9, 9), .1, dtype=np.float32)
    matrix[:4, :4] = .05
    matrix[:8, 8] = .15
    matrix[8, :8] = .15
    for i, a in enumerate(ACTIONS[:8]):
        for j, b in enumerate(ACTIONS[:8]):
            if a[0] == -b[0] and a[1] == -b[1]:
                matrix[i, j] = .3
    return matrix
