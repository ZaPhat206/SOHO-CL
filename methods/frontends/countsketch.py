"""Signed-hash feature sketch for fixed explicit analytic features."""

from __future__ import annotations

import torch

from methods.analytic_ridge import AnalyticRidgeBackend, persistent_tensor_bytes


class CountSketchAnalyticLearner:
    """Compose a fixed full-width encoder with a signed-hash Ridge sketch.

    The input to :meth:`encode_codes` is an already materialized explicit
    feature matrix.  Every source coordinate is assigned to one output bucket
    with a fixed Rademacher sign.  The mapping is data-independent and stores
    no sample-level state.
    """

    is_exemplar_free = True

    def __init__(
        self,
        *,
        input_dimension: int,
        backend: AnalyticRidgeBackend,
        seed: int = 2025,
        bucket_indices: torch.Tensor | None = None,
        signs: torch.Tensor | None = None,
    ) -> None:
        if input_dimension <= 0:
            raise ValueError("input_dimension must be positive")
        self.input_dimension = int(input_dimension)
        self.sketch_dimension = int(backend.dimension)
        self.seed = int(seed)
        self.backend = backend
        self.device = backend.device
        if bucket_indices is None or signs is None:
            if bucket_indices is not None or signs is not None:
                raise ValueError("bucket_indices and signs must be supplied together")
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            bucket_indices = torch.randint(
                self.sketch_dimension,
                (self.input_dimension,),
                generator=generator,
                dtype=torch.int32,
            )
            signs = (
                2
                * torch.randint(
                    2,
                    (self.input_dimension,),
                    generator=generator,
                    dtype=torch.int8,
                )
                - 1
            )
        buckets = bucket_indices.to(device=self.device, dtype=torch.int32)
        signed = signs.to(device=self.device, dtype=torch.int8)
        if buckets.shape != (self.input_dimension,) or signed.shape != (
            self.input_dimension,
        ):
            raise ValueError("CountSketch tensor shape mismatch")
        if bool((buckets < 0).any()) or bool((buckets >= self.sketch_dimension).any()):
            raise ValueError("CountSketch bucket is out of range")
        if not bool(((signed == -1) | (signed == 1)).all()):
            raise ValueError("CountSketch signs must be -1 or +1")
        self.bucket_indices = buckets
        self.signs = signed

    @property
    def class_ids(self) -> list[int]:
        return self.backend.class_ids

    def encode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        values = codes.to(
            device=self.device, dtype=self.backend.statistics_dtype
        )
        if values.ndim != 2 or values.shape[1] != self.input_dimension:
            raise ValueError(
                f"codes must have shape (B, {self.input_dimension})"
            )
        if not bool(torch.isfinite(values).all()):
            raise ValueError("codes contain NaN or Inf")
        output = torch.zeros(
            (len(values), self.sketch_dimension),
            device=self.device,
            dtype=values.dtype,
        )
        indices = self.bucket_indices.to(torch.long).expand(len(values), -1)
        signed = values * self.signs.to(values.dtype)
        output.scatter_add_(1, indices, signed)
        return output

    def update_codes(self, codes: torch.Tensor, labels: torch.Tensor) -> None:
        self.backend.update(self.encode_codes(codes), labels)

    def predict_logits_from_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return self.backend.predict_logits(self.encode_codes(codes))

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {
            "countsketch.bucket_indices": self.bucket_indices,
            "countsketch.signs": self.signs,
        }
        tensors.update(self.backend.persistent_tensors())
        return tensors

    def persistent_state_bytes(self) -> int:
        return persistent_tensor_bytes(self.persistent_tensors())

    def assert_exemplar_free_state(self) -> None:
        self.backend.assert_exemplar_free_state()
        if self.bucket_indices.numel() != self.input_dimension:
            raise AssertionError("invalid CountSketch state")


__all__ = ["CountSketchAnalyticLearner"]
