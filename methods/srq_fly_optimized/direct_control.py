"""Certified repaired direct-Gram control for SRQ-FLY Priority 3.

Kept outside the locked optimized learner module so historical P2B and
state-matched source identities remain byte-for-byte reproducible.
"""

from __future__ import annotations

import math

import torch

from .learner import _BaseFLYLearner, _cholesky_solve
from .storage import CompressedUpper

def _quantization_error_metrics(
    reconstructed: torch.Tensor,
    reference: torch.Tensor,
    *,
    row_chunk_size: int,
) -> tuple[float, float]:
    """Return ``(||E||_inf, ||E||_F / max(||reference||_F, 1))``.

    The row-chunked implementation avoids another dense ``m x m`` temporary.
    For a symmetric error matrix, ``||E||_2 <= ||E||_inf``.  The first value
    can therefore certify a lower eigenvalue bound through Weyl's inequality;
    the second value is diagnostic only.
    """
    if (
        reconstructed.ndim != 2
        or reconstructed.shape != reference.shape
        or reconstructed.shape[0] != reconstructed.shape[1]
    ):
        raise ValueError("quantization matrices must be aligned and square")
    if row_chunk_size <= 0:
        raise ValueError("row_chunk_size must be positive")
    device = reconstructed.device
    infinity_norm = torch.zeros((), device=device, dtype=torch.float64)
    squared_error = torch.zeros((), device=device, dtype=torch.float64)
    squared_reference = torch.zeros((), device=device, dtype=torch.float64)
    for start in range(0, len(reference), row_chunk_size):
        end = min(start + row_chunk_size, len(reference))
        difference = reconstructed[start:end] - reference[start:end]
        infinity_norm = torch.maximum(
            infinity_norm,
            difference.abs().sum(1, dtype=torch.float64).amax(),
        )
        squared_error.add_(difference.square().sum(dtype=torch.float64))
        squared_reference.add_(
            reference[start:end].square().sum(dtype=torch.float64)
        )
    relative_error = torch.sqrt(
        squared_error / torch.clamp(squared_reference, min=1.0)
    )
    return float(infinity_norm.item()), float(relative_error.item())


class CertifiedDirectInt8GramLearner(_BaseFLYLearner):
    """Direct-int8 Gram with a deterministic Weyl-certified diagonal repair.

    This is a scientific control, not the proposed SRQ representation.  Let
    ``S_t`` be the pre-quantization symmetric Gram approximation and let
    ``E_t = Q(S_t) - S_t``.  A persistent scalar lower bound is propagated as

    ``ell_t = ell_{t-1} - ||E_t||_inf``.

    Since every task adds a positive-semidefinite ``Z.T @ Z`` and ``E_t`` is
    symmetric, Weyl's inequality gives ``lambda_min(Q(S_t)) >= ell_t``.  The
    solver adds the smallest non-negative diagonal loading implied by this
    certificate and a fixed floating-point margin.  No label, validation
    accuracy, eigensolver, or test data enters the repair decision.
    """

    def __init__(
        self,
        *,
        repair_margin_multiplier: float = 8.0,
        repair_error_chunk_size: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if not torch.isfinite(torch.tensor(repair_margin_multiplier)) or (
            repair_margin_multiplier <= 0
        ):
            raise ValueError("repair_margin_multiplier must be finite and positive")
        if repair_error_chunk_size <= 0:
            raise ValueError("repair_error_chunk_size must be positive")
        self.repair_margin_multiplier = float(repair_margin_multiplier)
        self.repair_error_chunk_size = int(repair_error_chunk_size)
        self.gram: CompressedUpper | None = None
        self.certified_gram_lower_bound = torch.zeros(
            (), device=self.device, dtype=torch.float64
        )
        self.diagonal_loading = torch.zeros(
            (), device=self.device, dtype=torch.float64
        )
        self.diagnostics.update(
            method="certified_direct_int8_gram",
            storage="groupwise_int8",
            repair="weyl_infinity_norm_diagonal_loading",
            repair_margin_multiplier=self.repair_margin_multiplier,
        )

    def update_codes(self, codes: torch.Tensor, labels: torch.Tensor) -> None:
        values = codes.to(device=self.device, dtype=self.statistics_dtype)
        target_labels = labels.to(device=self.device, dtype=torch.long)
        if values.ndim != 2 or values.shape[1] != self.expand_dim:
            raise ValueError(f"codes must have shape (B, {self.expand_dim})")
        if target_labels.ndim != 1 or len(target_labels) != len(values) or not len(values):
            raise ValueError("labels must align with a non-empty code matrix")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("codes contain NaN or Inf")

        class_ids, cross, counts, targets = self._expanded_statistics(target_labels)
        updated = values.T @ values
        if self.gram is not None:
            updated.add_(
                self.gram.reconstruct_symmetric(dtype=self.statistics_dtype)
            )
        # Compression consumes the upper triangle.  Explicit symmetrization
        # makes the PSD-update premise and the error certificate unambiguous.
        updated = (updated + updated.T) * 0.5
        compressed = CompressedUpper.from_upper(
            updated,
            block_size=self.block_size,
            group_size=self.group_size,
            mode="int8",
        )
        reconstructed = compressed.reconstruct_symmetric(dtype=self.solver_dtype)
        local_error_bound, relative_storage_error = _quantization_error_metrics(
            reconstructed.to(updated.dtype),
            updated,
            row_chunk_size=self.repair_error_chunk_size,
        )
        local_error_bound = math.nextafter(
            local_error_bound
            * (1.0 + 4.0 * torch.finfo(torch.float64).eps * self.expand_dim),
            math.inf,
        )
        previous_bound = float(self.certified_gram_lower_bound.item())
        certified_bound = math.nextafter(
            previous_bound - local_error_bound, -math.inf
        )

        scale = max(
            float(reconstructed.diagonal().abs().amax().item()),
            self.ridge_lambda,
            1.0,
        )
        numerical_margin = (
            self.repair_margin_multiplier
            * torch.finfo(self.solver_dtype).eps
            * self.expand_dim
            * scale
        )
        loading = max(
            0.0,
            numerical_margin - (self.ridge_lambda + certified_bound),
        )
        effective_ridge = self.ridge_lambda + loading

        new_cross = cross + values.T @ targets
        new_counts = counts + targets.sum(0)
        work_cross = new_cross.to(self.solver_dtype)
        system = reconstructed
        system.diagonal().add_(effective_ridge)
        try:
            weights, residual = _cholesky_solve(system, work_cross)
        except RuntimeError as error:
            raise RuntimeError(
                "Weyl-certified direct Gram repair failed numerical Cholesky"
            ) from error

        self.gram = compressed
        self.class_ids, self.Q, self.counts = class_ids, new_cross, new_counts
        self.weights = weights
        self.total_rows += len(values)
        self.certified_gram_lower_bound.fill_(certified_bound)
        self.diagonal_loading.fill_(loading)
        self.diagnostics.update(
            solver_relative_residual=residual,
            relative_local_storage_error=relative_storage_error,
            local_quantization_error_infinity_bound=local_error_bound,
            certified_gram_lower_bound=certified_bound,
            diagonal_loading=loading,
            effective_ridge_lambda=effective_ridge,
            certified_system_eigenvalue_floor=numerical_margin,
            total_rows=self.total_rows,
        )
        self.assert_exemplar_free_state()

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = self._base_persistent_tensors()
        tensors.update(
            certified_gram_lower_bound=self.certified_gram_lower_bound,
            diagonal_loading=self.diagonal_loading,
        )
        if self.gram is not None:
            tensors.update(self.gram.persistent_tensors("gram"))
        return tensors

    def state_dict(self) -> dict:
        state = self._configuration_state()
        state.update(
            method="certified_direct_int8_gram",
            repair_margin_multiplier=self.repair_margin_multiplier,
            repair_error_chunk_size=self.repair_error_chunk_size,
            certified_gram_lower_bound=self.certified_gram_lower_bound.detach()
            .cpu()
            .clone(),
            diagonal_loading=self.diagonal_loading.detach().cpu().clone(),
            gram=None if self.gram is None else self.gram.state_dict(),
        )
        return state

    def load_state_dict(self, state: dict) -> None:
        self._load_common(state, "certified_direct_int8_gram")
        if state.get("repair_margin_multiplier") != self.repair_margin_multiplier:
            raise ValueError("checkpoint repair margin multiplier mismatch")
        if state.get("repair_error_chunk_size") != self.repair_error_chunk_size:
            raise ValueError("checkpoint repair chunk size mismatch")
        self.gram = None if state["gram"] is None else CompressedUpper.load_state_dict(
            state["gram"], device=self.device
        )
        lower = state["certified_gram_lower_bound"].to(
            device=self.device, dtype=torch.float64
        )
        loading = state["diagonal_loading"].to(
            device=self.device, dtype=torch.float64
        )
        if lower.shape or loading.shape or not bool(torch.isfinite(lower)) or not bool(
            torch.isfinite(loading)
        ):
            raise ValueError("invalid direct-Gram repair certificate")
        if float(lower) > 0 or float(loading) < 0:
            raise ValueError("invalid direct-Gram repair bounds")
        self.certified_gram_lower_bound.copy_(lower)
        self.diagonal_loading.copy_(loading)
        if self.gram is None:
            if self.total_rows or self.class_ids:
                raise ValueError("non-empty checkpoint is missing its Gram state")
            self.weights = None
        else:
            self._validate_compressed(self.gram, mode="int8")
            reconstructed = self.gram.reconstruct_symmetric(dtype=self.solver_dtype)
            reconstructed.diagonal().add_(
                self.ridge_lambda + float(self.diagonal_loading.item())
            )
            self.weights, residual = _cholesky_solve(
                reconstructed, self.Q.to(self.solver_dtype)
            )
            self.diagnostics.update(
                solver_relative_residual=residual,
                certified_gram_lower_bound=float(
                    self.certified_gram_lower_bound.item()
                ),
                diagonal_loading=float(self.diagonal_loading.item()),
                effective_ridge_lambda=self.ridge_lambda
                + float(self.diagonal_loading.item()),
            )
        self.assert_exemplar_free_state()
