"""Phase-1 MARS-SOHO learner and its exact feature-replay oracle."""

from __future__ import annotations

import inspect

import torch

from .geometry import (
    align_projection_gauge,
    certified_stable_support,
    compute_soho_rotation,
    topk_support_turnover,
)
from .reconstruction import (
    MODEL_MODES,
    SphericalReconstructor,
    allocate_pseudo_budget,
    shuffled_risks,
    wta_statistic_variance,
)
from .statistics import MomentSnapshot, SphericalClassMoments


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _solve_ridge(
    gram: torch.Tensor, cross: torch.Tensor, ridge_lambda: float
) -> tuple[torch.Tensor, float]:
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    system = (gram + gram.T) * 0.5 + ridge_lambda * identity
    factor = torch.linalg.cholesky(system)
    weights = torch.cholesky_solve(cross, factor)
    residual = torch.linalg.vector_norm(system @ weights - cross)
    denominator = torch.linalg.vector_norm(cross).clamp_min(
        torch.finfo(cross.dtype).eps
    )
    return weights, float((residual / denominator).item())


class DynamicSOHOMap:
    """New isolated SOHO map sharing one implementation across Phase-1 controls."""

    def __init__(
        self,
        *,
        feature_dim: int,
        expand_dim: int,
        density: float,
        olda_dim: int,
        coding_level: float,
        use_etf: bool,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if feature_dim <= 0 or expand_dim <= 1:
            raise ValueError("feature_dim must be positive and expand_dim must exceed one")
        if not 0 < density <= 1:
            raise ValueError("density must be in (0, 1]")
        if not 0 < coding_level < 1:
            raise ValueError("coding_level must be in (0, 1)")
        self.feature_dim = int(feature_dim)
        self.expand_dim = int(expand_dim)
        self.olda_dim = min(int(olda_dim), self.feature_dim)
        self.density = float(density)
        self.coding_level = float(coding_level)
        self.use_etf = bool(use_etf)
        self.seed = int(seed)
        self.device = device
        self.dtype = dtype
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed)
        random_values = torch.rand(
            (self.expand_dim, self.olda_dim),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        self.projection = torch.zeros_like(random_values)
        self.projection[random_values < self.density / 2] = 1
        self.projection[
            (random_values >= self.density / 2) & (random_values < self.density)
        ] = -1
        self.rotation = torch.eye(
            self.olda_dim, self.feature_dim, device=device, dtype=dtype
        )
        self.discriminative_rank = 0

    @property
    def k(self) -> int:
        return max(1, int(self.expand_dim * self.coding_level))

    def update_rotation(self, snapshot: MomentSnapshot) -> None:
        result = compute_soho_rotation(
            snapshot, output_dim=self.olda_dim, use_etf=self.use_etf
        )
        self.rotation = align_projection_gauge(
            result.rotation,
            self.rotation,
            discriminative_rank=result.discriminative_rank,
            eigenvalues=result.eigenvalues,
            use_etf=self.use_etf,
        )
        self.discriminative_rank = result.discriminative_rank

    def expanded(self, features: torch.Tensor, *, rotation: torch.Tensor | None = None) -> torch.Tensor:
        values = features.to(device=self.device, dtype=self.dtype)
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (N, {self.feature_dim})")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("features contain NaN or Inf")
        values = torch.nn.functional.normalize(values, p=2, dim=1)
        active_rotation = self.rotation if rotation is None else rotation
        return (values @ active_rotation.T) @ self.projection.T

    def encode(self, features: torch.Tensor, *, rotation: torch.Tensor | None = None) -> torch.Tensor:
        expanded = self.expanded(features, rotation=rotation)
        values, indices = torch.topk(expanded, self.k, dim=1, largest=True)
        output = torch.zeros_like(expanded)
        output.scatter_(1, indices, values)
        return output

    def state_dict(self) -> dict:
        return {
            "projection": self.projection.detach().cpu().clone(),
            "rotation": self.rotation.detach().cpu().clone(),
            "discriminative_rank": self.discriminative_rank,
        }

    def load_state_dict(self, state: dict) -> None:
        projection = state["projection"].to(self.device, self.dtype)
        rotation = state["rotation"].to(self.device, self.dtype)
        if projection.shape != (self.expand_dim, self.olda_dim):
            raise ValueError("invalid MARS projection shape")
        if rotation.shape != (self.olda_dim, self.feature_dim):
            raise ValueError("invalid MARS rotation shape")
        if not bool(torch.isfinite(projection).all() and torch.isfinite(rotation).all()):
            raise ValueError("MARS map contains NaN or Inf")
        self.projection = projection
        self.rotation = rotation
        self.discriminative_rank = int(state["discriminative_rank"])


class MARSSOHOLearner:
    """Exemplar-free reconstruction of old-class SOHO WTA statistics.

    Phase 1 deliberately stores an exact dense Gram. SRQ compression is not
    part of this class, so replay-model error can be measured independently.
    """

    is_exemplar_free = True

    def __init__(
        self,
        *,
        feature_dim: int,
        expand_dim: int,
        density: float,
        olda_dim: int,
        use_etf: bool,
        coding_level: float,
        ridge_lambda: float,
        model_mode: str,
        pseudo_per_class: int,
        pilot_per_class: int,
        covariance_rank: int,
        shrinkage: float,
        minimum_per_class: int = 4,
        risk_floor: float = 1e-3,
        seed: int = 2025,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if model_mode not in MODEL_MODES:
            raise ValueError(f"model_mode must be one of {sorted(MODEL_MODES)}")
        if ridge_lambda <= 0:
            raise ValueError("ridge_lambda must be positive")
        if pseudo_per_class <= 0 or pilot_per_class <= 0:
            raise ValueError("pseudo and pilot counts must be positive")
        if minimum_per_class <= 0 or minimum_per_class > pseudo_per_class:
            raise ValueError("minimum_per_class must lie in [1, pseudo_per_class]")
        self.feature_dim = int(feature_dim)
        self.expand_dim = int(expand_dim)
        self.density = float(density)
        self.olda_dim = min(int(olda_dim), self.feature_dim)
        self.use_etf = bool(use_etf)
        self.coding_level = float(coding_level)
        self.ridge_lambda = float(ridge_lambda)
        self.model_mode = model_mode
        self.pseudo_per_class = int(pseudo_per_class)
        self.pilot_per_class = int(pilot_per_class)
        self.covariance_rank = int(covariance_rank)
        self.shrinkage = float(shrinkage)
        self.minimum_per_class = int(minimum_per_class)
        self.risk_floor = float(risk_floor)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.dtype = dtype
        self.moments = SphericalClassMoments(
            self.feature_dim, device=self.device, dtype=self.dtype
        )
        self.encoder = DynamicSOHOMap(
            feature_dim=self.feature_dim,
            expand_dim=self.expand_dim,
            density=self.density,
            olda_dim=self.olda_dim,
            coding_level=self.coding_level,
            use_etf=self.use_etf,
            seed=self.seed,
            device=self.device,
            dtype=self.dtype,
        )
        self.G = torch.zeros(
            (self.expand_dim, self.expand_dim), device=self.device, dtype=self.dtype
        )
        self.Q = torch.zeros((self.expand_dim, 0), device=self.device, dtype=self.dtype)
        self.weights: torch.Tensor | None = None
        self.diagnostics: dict = {
            "phase": 1,
            "reconstruction": self.model_mode,
            "srq_enabled": False,
        }

    @property
    def class_ids(self) -> list[int]:
        return list(self.moments.class_ids)

    def _targets(self, labels: torch.Tensor) -> torch.Tensor:
        columns = {value: index for index, value in enumerate(self.class_ids)}
        indices = torch.tensor(
            [columns[int(value)] for value in labels.detach().cpu().tolist()],
            device=self.device,
        )
        return torch.nn.functional.one_hot(
            indices, num_classes=len(self.class_ids)
        ).to(self.dtype)

    def _boundary_risks(
        self,
        reconstructor: SphericalReconstructor,
        old_snapshot: MomentSnapshot,
        old_rotation: torch.Tensor,
    ) -> torch.Tensor:
        risks = []
        for class_id in old_snapshot.class_ids:
            pilot = reconstructor.generate(
                class_id, self.pilot_per_class, heterogeneous=True
            )
            old_expanded = self.encoder.expanded(pilot, rotation=old_rotation)
            new_expanded = self.encoder.expanded(pilot)
            stable = certified_stable_support(
                old_expanded, new_expanded, self.encoder.k
            )
            risks.append(1 - stable.to(self.dtype).mean())
        if not risks:
            return torch.zeros(0, device=self.device, dtype=self.dtype)
        return torch.stack(risks)

    def _continuous_risk_diagnostics(
        self,
        reconstructor: SphericalReconstructor,
        old_snapshot: MomentSnapshot,
        old_rotation: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        certificate_failure = []
        support_turnover = []
        statistic_variance = []
        for class_id in old_snapshot.class_ids:
            # The pilot stream is disjoint from the pseudo-statistic stream so
            # allocation does not reuse the samples whose estimator it sizes.
            pilot = reconstructor.generate(
                class_id,
                self.pilot_per_class,
                heterogeneous=True,
                stream_offset=1,
            )
            old_expanded = self.encoder.expanded(pilot, rotation=old_rotation)
            new_expanded = self.encoder.expanded(pilot)
            stable = certified_stable_support(
                old_expanded, new_expanded, self.encoder.k
            )
            certificate_failure.append(1 - stable.to(self.dtype).mean())
            support_turnover.append(
                topk_support_turnover(
                    old_expanded, new_expanded, self.encoder.k
                ).mean()
            )
            statistic_variance.append(
                wta_statistic_variance(self.encoder.encode(pilot))
            )
        empty = torch.zeros(0, device=self.device, dtype=self.dtype)
        return {
            "certificate_failure": torch.stack(certificate_failure)
            if certificate_failure else empty,
            "support_turnover": torch.stack(support_turnover)
            if support_turnover else empty,
            "statistic_variance": torch.stack(statistic_variance)
            if statistic_variance else empty,
        }

    def _allocations(
        self,
        old_snapshot: MomentSnapshot,
        risks: torch.Tensor,
    ) -> dict[int, int]:
        class_ids = list(old_snapshot.class_ids)
        if self.model_mode in {"shared_gaussian", "heterogeneous_spherical"}:
            return {class_id: self.pseudo_per_class for class_id in class_ids}
        active_risks = risks
        if self.model_mode in {
            "shuffled_support",
            "shuffled_turnover",
            "shuffled_statistic_variance",
        }:
            active_risks = shuffled_risks(risks, seed=self.seed + self.moments.total_count)
        return allocate_pseudo_budget(
            class_ids,
            old_snapshot.counts,
            active_risks,
            total_budget=self.pseudo_per_class * len(class_ids),
            minimum_per_class=self.minimum_per_class,
            risk_floor=self.risk_floor,
        )

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        values = features.to(self.device, self.dtype)
        targets = labels.to(self.device, torch.long)
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (N, {self.feature_dim})")
        if targets.ndim != 1 or targets.shape[0] != values.shape[0]:
            raise ValueError("labels must have shape (N,) aligned with features")
        if values.shape[0] == 0:
            return
        if not bool(torch.isfinite(values).all()):
            raise ValueError("features contain NaN or Inf")
        old_snapshot = self.moments.snapshot()
        old_rotation = self.encoder.rotation.clone()
        self.moments.update(values, targets)
        current_snapshot = self.moments.snapshot()
        self.encoder.update_rotation(current_snapshot)
        self.G.zero_()
        self.Q = torch.zeros(
            (self.expand_dim, len(self.class_ids)),
            device=self.device,
            dtype=self.dtype,
        )
        # The arriving batch is encoded exactly once and is discarded afterwards.
        current_codes = self.encoder.encode(values)
        current_targets = self._targets(targets)
        self.G.add_(current_codes.T @ current_codes)
        self.Q.add_(current_codes.T @ current_targets)
        risks = torch.zeros(0, device=self.device, dtype=self.dtype)
        risk_diagnostics: dict[str, torch.Tensor] = {}
        risk_name = "certificate_failure"
        allocations: dict[int, int] = {}
        if old_snapshot.num_classes:
            reconstructor = SphericalReconstructor(
                current_snapshot,
                covariance_rank=self.covariance_rank,
                shrinkage=self.shrinkage,
                seed=self.seed,
            )
            if self.model_mode in {
                "turnover_aware", "shuffled_turnover",
                "statistic_variance_aware", "shuffled_statistic_variance",
            }:
                risk_diagnostics = self._continuous_risk_diagnostics(
                    reconstructor, old_snapshot, old_rotation
                )
                if self.model_mode in {"turnover_aware", "shuffled_turnover"}:
                    risk_name = "support_turnover"
                else:
                    risk_name = "statistic_variance"
                risks = risk_diagnostics[risk_name]
            else:
                risks = self._boundary_risks(
                    reconstructor, old_snapshot, old_rotation
                )
                risk_diagnostics = {"certificate_failure": risks}
            allocations = self._allocations(old_snapshot, risks)
            heterogeneous = self.model_mode != "shared_gaussian"
            class_columns = {
                value: index for index, value in enumerate(self.class_ids)
            }
            pseudo_blocks = []
            pseudo_weights = []
            block_metadata = []
            for old_column, class_id in enumerate(old_snapshot.class_ids):
                pseudo_count = allocations[class_id]
                pseudo = reconstructor.generate(
                    class_id, pseudo_count, heterogeneous=heterogeneous
                )
                weight = old_snapshot.counts[old_column] / pseudo_count
                start = sum(value.shape[0] for value in pseudo_blocks)
                pseudo_blocks.append(pseudo)
                pseudo_weights.append(
                    torch.full(
                        (pseudo_count,), weight,
                        device=self.device, dtype=self.dtype,
                    )
                )
                block_metadata.append(
                    (start, start + pseudo_count, class_columns[class_id], weight)
                )
            all_codes = self.encoder.encode(torch.cat(pseudo_blocks, dim=0))
            all_weights = torch.cat(pseudo_weights)
            weighted_codes = all_codes * all_weights.sqrt().unsqueeze(1)
            self.G.add_(weighted_codes.T @ weighted_codes)
            for start, stop, column, weight in block_metadata:
                self.Q[:, column].add_(weight * all_codes[start:stop].sum(dim=0))
        self.weights, residual = _solve_ridge(self.G, self.Q, self.ridge_lambda)
        self.diagnostics = {
            "phase": 1,
            "reconstruction": self.model_mode,
            "srq_enabled": False,
            "total_count": self.moments.total_count,
            "old_class_count": old_snapshot.num_classes,
            "pseudo_total": sum(allocations.values()),
            "pseudo_allocation": allocations,
            "boundary_risk": {
                class_id: float(risks[index].item())
                for index, class_id in enumerate(old_snapshot.class_ids)
            },
            "allocation_risk_name": risk_name,
            "pilot_risks": {
                name: {
                    class_id: float(values[index].item())
                    for index, class_id in enumerate(old_snapshot.class_ids)
                }
                for name, values in risk_diagnostics.items()
            },
            "solver_relative_residual": residual,
            "discriminative_rank": self.encoder.discriminative_rank,
        }
        self.assert_exemplar_free_state()

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        if self.weights is None:
            raise RuntimeError("update() must be called before prediction")
        return self.encoder.encode(features) @ self.weights

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        columns = self.predict_logits(features).argmax(dim=1).detach().cpu().tolist()
        return torch.tensor([self.class_ids[index] for index in columns], dtype=torch.long)

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {
            "random_projection": self.encoder.projection,
            "active_rotation": self.encoder.rotation,
            "gram": self.G,
            "cross": self.Q,
            **self.moments.persistent_tensors(),
        }
        if self.weights is not None:
            tensors["classifier"] = self.weights
        return tensors

    def persistent_state_bytes(self) -> int:
        return sum(_tensor_bytes(value) for value in self.persistent_tensors().values())

    def assert_exemplar_free_state(self) -> None:
        forbidden_names = ("history", "sample", "image", "dataset", "loader", "cache")
        offending = [
            name for name in self.persistent_tensors()
            if any(token in name.lower() for token in forbidden_names)
        ]
        if offending:
            raise AssertionError(f"forbidden persistent tensors: {offending}")
        total = self.moments.total_count
        structural = {
            self.feature_dim,
            self.expand_dim,
            self.olda_dim,
            self.moments.num_classes,
        }
        if total > max(structural):
            for name, tensor in self.persistent_tensors().items():
                if total in tensor.shape:
                    raise AssertionError(
                        f"{name} has a historical sample-count dimension"
                    )

    def _configuration(self) -> dict:
        return {
            name: getattr(self, name)
            for name in (
                "feature_dim", "expand_dim", "density", "olda_dim", "use_etf",
                "coding_level", "ridge_lambda", "model_mode", "pseudo_per_class",
                "pilot_per_class", "covariance_rank", "shrinkage",
                "minimum_per_class", "risk_floor", "seed",
            )
        }

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "method": "mars_soho_phase1",
            "configuration": self._configuration(),
            "moments": self.moments.state_dict(),
            "encoder": self.encoder.state_dict(),
            "G": self.G.detach().cpu().clone(),
            "Q": self.Q.detach().cpu().clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("version") != 1 or state.get("method") != "mars_soho_phase1":
            raise ValueError("unsupported MARS-SOHO checkpoint")
        if state.get("configuration") != self._configuration():
            raise ValueError("MARS-SOHO checkpoint configuration mismatch")
        self.moments.load_state_dict(state["moments"])
        self.encoder.load_state_dict(state["encoder"])
        self.G = state["G"].to(self.device, self.dtype)
        self.Q = state["Q"].to(self.device, self.dtype)
        if self.G.shape != (self.expand_dim, self.expand_dim):
            raise ValueError("invalid MARS-SOHO Gram shape")
        if self.Q.shape != (self.expand_dim, len(self.class_ids)):
            raise ValueError("invalid MARS-SOHO cross shape")
        self.weights, residual = _solve_ridge(self.G, self.Q, self.ridge_lambda)
        self.diagnostics = {
            "phase": 1,
            "reconstruction": self.model_mode,
            "srq_enabled": False,
            "total_count": self.moments.total_count,
            "solver_relative_residual": residual,
            "resumed": True,
        }
        self.assert_exemplar_free_state()


class MARSExactReplayOracle:
    """Non-exemplar-free exact replay control sharing the MARS map and λ."""

    is_exemplar_free = False

    def __init__(self, **kwargs) -> None:
        allowed = {
            name: kwargs[name]
            for name in (
                "feature_dim", "expand_dim", "density", "olda_dim", "use_etf",
                "coding_level", "seed", "device", "dtype",
            )
            if name in kwargs
        }
        self.ridge_lambda = float(kwargs["ridge_lambda"])
        if self.ridge_lambda <= 0:
            raise ValueError("ridge_lambda must be positive")
        self.feature_dim = int(kwargs["feature_dim"])
        self.expand_dim = int(kwargs["expand_dim"])
        self.device = torch.device(kwargs.get("device", "cpu"))
        self.dtype = kwargs.get("dtype", torch.float64)
        self.moments = SphericalClassMoments(
            self.feature_dim, device=self.device, dtype=self.dtype
        )
        allowed["device"] = self.device
        allowed["dtype"] = self.dtype
        self.encoder = DynamicSOHOMap(**allowed)
        self.feature_history: list[torch.Tensor] = []
        self.label_history: list[torch.Tensor] = []
        self.G = torch.zeros(
            (self.expand_dim, self.expand_dim), device=self.device, dtype=self.dtype
        )
        self.Q = torch.zeros((self.expand_dim, 0), device=self.device, dtype=self.dtype)
        self.weights: torch.Tensor | None = None
        self.diagnostics = {"phase": 1, "oracle": "exact_feature_replay"}

    @property
    def class_ids(self) -> list[int]:
        return list(self.moments.class_ids)

    def update(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        values = features.to(self.device, self.dtype)
        targets = labels.to(self.device, torch.long)
        self.moments.update(values, targets)
        self.encoder.update_rotation(self.moments.snapshot())
        self.feature_history.append(values.detach().clone())
        self.label_history.append(targets.detach().clone())
        historical_values = torch.cat(self.feature_history)
        historical_targets = torch.cat(self.label_history)
        codes = self.encoder.encode(historical_values)
        columns = {value: index for index, value in enumerate(self.class_ids)}
        target_columns = torch.tensor(
            [columns[int(value)] for value in historical_targets.detach().cpu().tolist()],
            device=self.device,
        )
        one_hot = torch.nn.functional.one_hot(
            target_columns, num_classes=len(self.class_ids)
        ).to(self.dtype)
        self.G = codes.T @ codes
        self.Q = codes.T @ one_hot
        self.weights, residual = _solve_ridge(self.G, self.Q, self.ridge_lambda)
        self.diagnostics = {
            "phase": 1,
            "oracle": "exact_feature_replay",
            "retained_sample_count": len(historical_values),
            "solver_relative_residual": residual,
        }

    def predict_logits(self, features: torch.Tensor) -> torch.Tensor:
        if self.weights is None:
            raise RuntimeError("update() must be called before prediction")
        return self.encoder.encode(features) @ self.weights

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        columns = self.predict_logits(features).argmax(dim=1).detach().cpu().tolist()
        return torch.tensor([self.class_ids[index] for index in columns], dtype=torch.long)

    def persistent_state_bytes(self) -> int:
        return sum(_tensor_bytes(value) for value in self.persistent_tensors().values())

    def persistent_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {
            "random_projection": self.encoder.projection,
            "active_rotation": self.encoder.rotation,
            "gram": self.G,
            "cross": self.Q,
            **self.moments.persistent_tensors(),
        }
        tensors.update({
            f"feature_history_{index}": value
            for index, value in enumerate(self.feature_history)
        })
        tensors.update({
            f"label_history_{index}": value
            for index, value in enumerate(self.label_history)
        })
        if self.weights is not None:
            tensors["classifier"] = self.weights
        return tensors


assert "task_id" not in inspect.signature(MARSSOHOLearner.predict_logits).parameters
