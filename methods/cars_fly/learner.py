"""Exemplar-free CARS-FLY learner with a certified adaptive residual rank."""

from __future__ import annotations

import inspect

import torch

from methods.crt_soho.geometry import anchor_weights
from methods.crt_soho.learner import CRTSOHOLearner
from methods.crt_soho.solver import reconstruct_residual_statistics, solve_block_ridge

from .solver import adaptive_conditional_directions


class CARSFLYLearner(CRTSOHOLearner):
    """Compact fixed WTA anchor plus an adaptive conditional raw correction."""

    is_exemplar_free = True

    def __init__(
        self,
        *,
        raw_dim: int,
        anchor_dim: int,
        synaptic_degree: int,
        coding_level: float,
        anchor_ridge: float,
        residual_ridge: float,
        complement_ridge: float,
        energy_threshold: float,
        max_rank: int,
        min_rank: int = 1,
        minimum_objective_gain: float = 0.0,
        seed: int = 2025,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
        anchor_projection: torch.Tensor | None = None,
    ):
        if not 0 < energy_threshold <= 1:
            raise ValueError("energy_threshold must be in (0, 1]")
        if min_rank <= 0 or max_rank < min_rank:
            raise ValueError("rank bounds must satisfy 1 <= min_rank <= max_rank")
        if minimum_objective_gain < 0:
            raise ValueError("minimum_objective_gain must be non-negative")
        self.energy_threshold = float(energy_threshold)
        self.max_rank = int(max_rank)
        self.min_rank = int(min_rank)
        self.minimum_objective_gain = float(minimum_objective_gain)
        super().__init__(
            method="schur_residual",
            raw_dim=raw_dim,
            anchor_dim=anchor_dim,
            synaptic_degree=synaptic_degree,
            coding_level=coding_level,
            anchor_ridge=anchor_ridge,
            residual_ridge=residual_ridge,
            complement_ridge=complement_ridge,
            requested_rank=max_rank,
            confusion_temperature=1.0,
            scatter_epsilon=1e-4,
            seed=seed,
            device=device,
            dtype=dtype,
            anchor_projection=anchor_projection,
        )
        self.method = "cars_fly"

    def _directions_and_geometry(
        self, raw_anchor_weights: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        del raw_anchor_weights
        correction = adaptive_conditional_directions(
            self.statistics,
            complement_ridge=self.complement_ridge,
            anchor_ridge=self.anchor_ridge,
            residual_ridge=self.residual_ridge,
            energy_threshold=self.energy_threshold,
            max_rank=self.max_rank,
            min_rank=self.min_rank,
            minimum_objective_gain=self.minimum_objective_gain,
        )
        return correction.directions, {
            "geometry": "adaptive_conditional_schur",
            "requested_rank": self.max_rank,
            "effective_rank": correction.effective_rank,
            "energy_threshold": self.energy_threshold,
            "captured_energy": correction.captured_energy,
            "tail_energy": correction.tail_energy,
            "total_correction_energy": correction.total_energy,
            "retained_correction_energy": correction.retained_fraction,
            "energy_threshold_reached": correction.threshold_reached,
            "singular_values": correction.singular_values.detach().clone(),
        }

    def _recompute(self) -> None:
        raw_anchor_weights = anchor_weights(self.statistics, self.anchor_ridge)
        directions, diagnostics = self._directions_and_geometry(
            raw_anchor_weights
        )
        if directions.shape[1] == 0:
            identity = torch.eye(
                self.anchor_dim, device=self.device, dtype=self.dtype
            )
            equation = (
                self.statistics.G_pp + self.anchor_ridge * identity
            ) @ raw_anchor_weights
            solver_residual = float(
                (equation - self.statistics.Q_p).abs().max().item()
            )
            self.anchor_classifier = raw_anchor_weights
            self.directions = None
            self.complement = None
            self.residual_classifier = None
            self.diagnostics = {
                **diagnostics,
                "residualized": True,
                "solver_residual_max": solver_residual,
                "solver_relative_residual_max": solver_residual
                / max(float(self.statistics.Q_p.abs().max().item()), 1.0),
            }
            return

        residual = reconstruct_residual_statistics(
            self.statistics,
            directions,
            self.complement_ridge,
            residualize=True,
        )
        anchor_classifier, residual_classifier, solver_residual = (
            solve_block_ridge(
                self.statistics,
                residual,
                self.anchor_ridge,
                self.residual_ridge,
            )
        )
        self.anchor_classifier = anchor_classifier
        self.directions = directions
        self.complement = residual.C
        self.residual_classifier = residual_classifier
        target_scale = max(
            float(
                torch.cat((self.statistics.Q_p, residual.Q_r), dim=0)
                .abs()
                .max()
                .item()
            ),
            1.0,
        )
        self.diagnostics = {
            **diagnostics,
            "residualized": True,
            "solver_residual_max": solver_residual,
            "solver_relative_residual_max": solver_residual / target_scale,
        }

    def state_dict(self) -> dict:
        """Store fixed configuration and aggregate statistics, not derived state."""
        return {
            "version": 1,
            "method": "cars_fly",
            "raw_dim": self.raw_dim,
            "anchor_dim": self.anchor_dim,
            "synaptic_degree": self.synaptic_degree,
            "coding_level": self.coding_level,
            "anchor_ridge": self.anchor_ridge,
            "residual_ridge": self.residual_ridge,
            "complement_ridge": self.complement_ridge,
            "energy_threshold": self.energy_threshold,
            "max_rank": self.max_rank,
            "min_rank": self.min_rank,
            "minimum_objective_gain": self.minimum_objective_gain,
            "seed": self.seed,
            "anchor_projection": self.anchor.projection_matrix,
            "statistics": self.statistics.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("version") != 1 or state.get("method") != "cars_fly":
            raise ValueError("unsupported CARS-FLY checkpoint")
        fields = (
            "raw_dim",
            "anchor_dim",
            "synaptic_degree",
            "coding_level",
            "anchor_ridge",
            "residual_ridge",
            "complement_ridge",
            "energy_threshold",
            "max_rank",
            "min_rank",
            "minimum_objective_gain",
            "seed",
        )
        for field in fields:
            if field not in state:
                raise ValueError(f"checkpoint missing {field}")
            if state[field] != getattr(self, field):
                raise ValueError(
                    f"checkpoint configuration mismatch for {field}"
                )
        projection = state["anchor_projection"].to(self.device)
        if projection.shape != (self.anchor_dim, self.raw_dim):
            raise ValueError("invalid anchor projection shape")
        if projection.layout != torch.sparse_csc:
            raise ValueError("anchor projection must use sparse CSC layout")
        if not bool(torch.isfinite(projection.values()).all()):
            raise ValueError("anchor projection contains NaN or Inf")
        self.anchor.projection_matrix = projection
        self.statistics.load_state_dict(state["statistics"])
        self._recompute()
        self.assert_exemplar_free_state()


def create_learner(**kwargs) -> CARSFLYLearner:
    return CARSFLYLearner(**kwargs)


assert "task_id" not in inspect.signature(CARSFLYLearner.predict_logits).parameters
