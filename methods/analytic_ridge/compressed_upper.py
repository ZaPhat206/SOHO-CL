"""Stable generic import surface for the version-1 compressed factor.

The implementation remains in its historical module during M1 so existing
checkpoint imports and locked experiment hashes are not rewritten.  New
backends depend only on this generic surface; a later format version may move
the implementation after an explicit checkpoint-migration gate.
"""

from methods.srq_fly_optimized.storage import CompressedUpper, UpperBlock

__all__ = ["CompressedUpper", "UpperBlock"]
