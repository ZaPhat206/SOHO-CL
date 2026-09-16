"""Reusable analytic Ridge state backends."""

from .accounting import persistent_tensor_bytes, tensor_bytes
from .adaptive_upper import AdaptiveCompressedUpper, AdaptiveUpperBlock
from .backends import (
    AnalyticRidgeBackend,
    DenseSquareRootBackend,
    ExactGramBackend,
    SquareRootBackend,
)
from .compressed_upper import CompressedUpper, UpperBlock
from .equal_memory_controls import (
    FrequentDirectionsRidgeBackend,
    PackedExactGramBackend,
    frequent_directions_backend_bytes,
    largest_frequent_directions_rank,
    largest_packed_exact_dimension,
    packed_exact_backend_bytes,
    packed_upper_element_count,
)
from .qr import blocked_qr_rank_update, dense_qr_rank_update

__all__ = [
    "AnalyticRidgeBackend",
    "AdaptiveCompressedUpper",
    "AdaptiveUpperBlock",
    "CompressedUpper",
    "DenseSquareRootBackend",
    "ExactGramBackend",
    "FrequentDirectionsRidgeBackend",
    "PackedExactGramBackend",
    "SquareRootBackend",
    "UpperBlock",
    "blocked_qr_rank_update",
    "dense_qr_rank_update",
    "frequent_directions_backend_bytes",
    "largest_frequent_directions_rank",
    "largest_packed_exact_dimension",
    "packed_exact_backend_bytes",
    "packed_upper_element_count",
    "persistent_tensor_bytes",
    "tensor_bytes",
]
