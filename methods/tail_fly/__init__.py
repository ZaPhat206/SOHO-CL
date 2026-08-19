"""TAIL-FLY: tail-aware low-rank state for fixed FlyHash/WTA codes."""

from .learner import TAILFlyLearner, create_learner
from .solver import (
    approximate_gram,
    diagonal_tail,
    solve_diagonal_ridge,
    solve_tail_ridge,
    solve_truncated_svd_ridge,
)
from .streaming_svd import StreamingTruncatedSVD

__all__ = [
    "TAILFlyLearner",
    "StreamingTruncatedSVD",
    "approximate_gram",
    "create_learner",
    "diagonal_tail",
    "solve_diagonal_ridge",
    "solve_tail_ridge",
    "solve_truncated_svd_ridge",
]
