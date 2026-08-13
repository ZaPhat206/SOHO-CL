"""Cache-native FLY and current-SOHO controls for matched feature ablations.

These adapters intentionally live outside the original implementations.  They
consume an already extracted fixed feature cache, so backbone/preprocessing is
identical across controls.  `CachedSOHOReplay` is explicitly *not*
exemplar-free: its serialised state includes per-example feature tensors in
order to reproduce the current dynamic-Top-K SOHO algorithm.
"""

from __future__ import annotations

import torch

from methods.sft_cl.statistics import FixedFeatureStatistics
from methods.flycl import select_ridge_parameter as select_fly_ridge_parameter
from methods.sohocl import select_ridge_parameter as select_soho_ridge_parameter
from models.flyhash import FlyHash
from models.soho import SOHO


def _ridge(gram: torch.Tensor, cross: torch.Tensor, ridge_lambda: float) -> torch.Tensor:
    if ridge_lambda <= 0:
        raise ValueError("ridge_lambda must be positive")
    system = gram + ridge_lambda * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    factor = torch.linalg.cholesky(system)
    return torch.cholesky_solve(cross, factor)


def _tensor_storage_bytes(tensor: torch.Tensor) -> int:
    """Count actual tensor storage, including compressed sparse indices."""
    if tensor.layout == torch.strided:
        return tensor.numel() * tensor.element_size()
    if tensor.layout == torch.sparse_csc:
        return (
            tensor.values().numel() * tensor.values().element_size()
            + tensor.ccol_indices().numel() * tensor.ccol_indices().element_size()
            + tensor.row_indices().numel() * tensor.row_indices().element_size()
        )
    if tensor.layout == torch.sparse_coo:
        coalesced = tensor.coalesce()
        return coalesced.values().numel() * coalesced.values().element_size() + coalesced.indices().numel() * coalesced.indices().element_size()
    raise ValueError(f"unsupported tensor layout for accounting: {tensor.layout}")


class CachedFlyCL:
    """Fixed FlyHash/WTA plus exact streaming projected-feature Ridge."""

    is_exemplar_free = True

    def __init__(self, feature_dim, expand_dim, synaptic_degree, coding_level, ridge_lambda, seed=1993, device="cpu", dtype=torch.float32):
        self.feature_dim = int(feature_dim)
        self.expand_dim = int(expand_dim)
        self.synaptic_degree = int(synaptic_degree)
        self.coding_level = float(coding_level)
        self.ridge_lambda = float(ridge_lambda)
        self.seed = int(seed)
        self.device, self.dtype = torch.device(device), dtype
        # Isolate projection randomness from runner/control construction order.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.flyhash = FlyHash(self.feature_dim, self.expand_dim, self.synaptic_degree).to(self.device)
        self.flyhash.to_sparse()
        self.statistics = FixedFeatureStatistics(self.expand_dim, self.device, dtype)
        self.weights: torch.Tensor | None = None
        self.diagnostics: dict = {"projection": "fixed_sparse_gaussian_wta"}

    @property
    def class_ids(self):
        return self.statistics.class_ids

    def _encode(self, features):
        x = features.to(self.device, self.dtype)
        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (B, {self.feature_dim})")
        return self.flyhash(x, self.coding_level, absolute_wta=False)

    def update(self, features, labels):
        self.statistics.update(self._encode(features), labels)
        self.weights = _ridge(self.statistics.G, self.statistics.Q, self.ridge_lambda)
        self.diagnostics["effective_rank"] = self.expand_dim

    def predict_logits(self, features):
        if self.weights is None:
            raise RuntimeError("update() must be called before prediction")
        return self._encode(features) @ self.weights

    def predict(self, features):
        columns = self.predict_logits(features).argmax(1).detach().cpu().tolist()
        return torch.tensor([self.class_ids[column] for column in columns], dtype=torch.long)

    def persistent_tensors(self):
        tensors = {"projection": self.flyhash.projection_matrix, "G": self.statistics.G, "Q": self.statistics.Q, "counts": self.statistics.counts}
        if self.weights is not None:
            tensors["weights"] = self.weights
        return tensors

    def persistent_state_bytes(self):
        return sum(_tensor_storage_bytes(tensor) for tensor in self.persistent_tensors().values())

    def state_dict(self):
        # Projection is deterministically regenerated from the documented seed.
        return {"version": 1, "feature_dim": self.feature_dim, "expand_dim": self.expand_dim, "synaptic_degree": self.synaptic_degree,
                "coding_level": self.coding_level, "ridge_lambda": self.ridge_lambda, "seed": self.seed, "statistics": self.statistics.state_dict()}

    def load_state_dict(self, state):
        expected = (self.feature_dim, self.expand_dim, self.synaptic_degree, self.coding_level, self.ridge_lambda, self.seed)
        actual = (int(state["feature_dim"]), int(state["expand_dim"]), int(state["synaptic_degree"]), float(state["coding_level"]), float(state["ridge_lambda"]), int(state["seed"]))
        if actual != expected:
            raise ValueError("CachedFlyCL checkpoint configuration mismatch")
        self.statistics.load_state_dict(state["statistics"])
        self.weights = _ridge(self.statistics.G, self.statistics.Q, self.ridge_lambda)


class CachedSOHOReplay:
    """Current SOHO re-projection algorithm; deliberately stores feature replay."""

    is_exemplar_free = False

    def __init__(self, feature_dim, expand_dim, density, olda_dim, use_etf, coding_level, ridge_lambda, seed=1993, device="cpu", dtype=torch.float32):
        self.feature_dim, self.expand_dim = int(feature_dim), int(expand_dim)
        self.density, self.olda_dim, self.use_etf = float(density), int(olda_dim), bool(use_etf)
        self.coding_level, self.ridge_lambda, self.seed = float(coding_level), float(ridge_lambda), int(seed)
        self.device, self.dtype = torch.device(device), dtype
        self.soho = self._new_soho()
        self.class_ids: list[int] = []
        self.feature_history: list[torch.Tensor] = []
        self.label_history: list[torch.Tensor] = []
        self.G = torch.zeros((self.expand_dim, self.expand_dim), device=self.device, dtype=self.dtype)
        self.Q = torch.zeros((self.expand_dim, 0), device=self.device, dtype=self.dtype)
        self.weights: torch.Tensor | None = None
        self.diagnostics: dict = {"replay_required": True, "reason": "dynamic_OLDA_projection_plus_sample_dependent_TopK"}

    def _new_soho(self):
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.seed)
            return SOHO(self.feature_dim, self.expand_dim, self.device, density=self.density, olda_dim=self.olda_dim, use_etf=self.use_etf).to(self.device)

    def _refresh_class_ids(self, labels):
        self.class_ids = sorted(set(self.class_ids) | set(map(int, labels.detach().cpu().tolist())))

    def _targets(self, labels):
        columns_for_id = {class_id: column for column, class_id in enumerate(self.class_ids)}
        columns = torch.tensor([columns_for_id[int(label)] for label in labels.detach().cpu().tolist()], device=self.device)
        return torch.nn.functional.one_hot(columns, num_classes=len(self.class_ids)).to(self.dtype)

    def update(self, features, labels):
        x = features.to(self.device, self.dtype)
        y = labels.to(self.device, torch.long)
        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (B, {self.feature_dim})")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("features contain NaN or Inf")
        self._refresh_class_ids(y)
        self.soho.update_stats(x, y)
        # This is the intentional replay state of the legacy SOHO control.
        self.feature_history.append(x.detach().clone())
        self.label_history.append(y.detach().clone())
        historical_x = torch.cat(self.feature_history, dim=0)
        historical_y = torch.cat(self.label_history, dim=0)
        encoded = self.soho(historical_x, self.coding_level, absolute_wta=False)
        targets = self._targets(historical_y)
        self.G = encoded.T @ encoded
        self.Q = encoded.T @ targets
        self.weights = _ridge(self.G, self.Q, self.ridge_lambda)
        self.diagnostics["effective_rank"] = self.soho.R.shape[0]

    def predict_logits(self, features):
        if self.weights is None:
            raise RuntimeError("update() must be called before prediction")
        x = features.to(self.device, self.dtype)
        return self.soho(x, self.coding_level, absolute_wta=False) @ self.weights

    def predict(self, features):
        columns = self.predict_logits(features).argmax(1).detach().cpu().tolist()
        return torch.tensor([self.class_ids[column] for column in columns], dtype=torch.long)

    def persistent_tensors(self):
        tensors = {"G": self.G, "Q": self.Q, "R": self.soho.R, "W": self.soho.W, "olda_global_sum": self.soho.olda.global_sum, "olda_S_w": self.soho.olda.S_w}
        tensors.update({f"feature_history_{index}": value for index, value in enumerate(self.feature_history)})
        tensors.update({f"label_history_{index}": value for index, value in enumerate(self.label_history)})
        tensors.update({f"olda_class_sum_{class_id}": value for class_id, value in self.soho.olda.class_sums.items()})
        if self.weights is not None:
            tensors["weights"] = self.weights
        return tensors

    def persistent_state_bytes(self):
        return sum(tensor.numel() * tensor.element_size() for tensor in self.persistent_tensors().values())

    def state_dict(self):
        return {"version": 1, "feature_dim": self.feature_dim, "expand_dim": self.expand_dim, "density": self.density, "olda_dim": self.olda_dim,
                "use_etf": self.use_etf, "coding_level": self.coding_level, "ridge_lambda": self.ridge_lambda, "seed": self.seed,
                "class_ids": list(self.class_ids), "feature_history": [x.cpu() for x in self.feature_history], "label_history": [y.cpu() for y in self.label_history],
                "G": self.G.cpu(), "Q": self.Q.cpu(), "weights": None if self.weights is None else self.weights.cpu(),
                "soho": {"R": self.soho.R.cpu(), "W": self.soho.W.cpu(), "global_sum": self.soho.olda.global_sum.cpu(), "global_count": self.soho.olda.global_count,
                         "S_w": self.soho.olda.S_w.cpu(), "class_sums": {key: value.cpu() for key, value in self.soho.olda.class_sums.items()}, "class_counts": dict(self.soho.olda.class_counts)}}

    def load_state_dict(self, state):
        expected = (self.feature_dim, self.expand_dim, self.density, self.olda_dim, self.use_etf, self.coding_level, self.ridge_lambda, self.seed)
        actual = (int(state["feature_dim"]), int(state["expand_dim"]), float(state["density"]), int(state["olda_dim"]), bool(state["use_etf"]), float(state["coding_level"]), float(state["ridge_lambda"]), int(state["seed"]))
        if actual != expected:
            raise ValueError("CachedSOHOReplay checkpoint configuration mismatch")
        self.class_ids = [int(class_id) for class_id in state["class_ids"]]
        self.feature_history = [x.to(self.device, self.dtype) for x in state["feature_history"]]
        self.label_history = [y.to(self.device, torch.long) for y in state["label_history"]]
        self.G, self.Q = state["G"].to(self.device, self.dtype), state["Q"].to(self.device, self.dtype)
        self.weights = None if state["weights"] is None else state["weights"].to(self.device, self.dtype)
        packed = state["soho"]
        self.soho.R, self.soho.W = packed["R"].to(self.device, self.dtype), packed["W"].to(self.device, self.dtype)
        self.soho.olda.global_sum, self.soho.olda.global_count = packed["global_sum"].to(self.device, self.dtype), int(packed["global_count"])
        self.soho.olda.S_w = packed["S_w"].to(self.device, self.dtype)
        self.soho.olda.class_sums = {int(key): value.to(self.device, self.dtype) for key, value in packed["class_sums"].items()}
        self.soho.olda.class_counts = {int(key): int(value) for key, value in packed["class_counts"].items()}


class CachedFlyCLFidelity:
    """Feature-cache execution of the original FlyCL update semantics.

    Unlike ``CachedFlyCL``, this adapter retains the original fixed global
    output width and selects Ridge by current-task GCV. It remains
    exemplar-free because the current task matrix is discarded after update.
    """

    is_exemplar_free = True

    def __init__(self, feature_dim, expand_dim, synaptic_degree, coding_level,
                 num_classes, ridge_lower, ridge_upper, seed=1993,
                 device="cpu", dtype=torch.float32):
        self.feature_dim = int(feature_dim)
        self.expand_dim = int(expand_dim)
        self.synaptic_degree = int(synaptic_degree)
        self.coding_level = float(coding_level)
        self.num_classes = int(num_classes)
        self.ridge_lower = float(ridge_lower)
        self.ridge_upper = float(ridge_upper)
        self.seed = int(seed)
        self.device, self.dtype = torch.device(device), dtype
        if self.num_classes <= 0 or self.ridge_lower >= self.ridge_upper:
            raise ValueError("invalid FlyCL fidelity configuration")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.flyhash = FlyHash(
                self.feature_dim, self.expand_dim, self.synaptic_degree
            ).to(self.device)
        self.flyhash.to_sparse()
        self.G = torch.zeros(
            (self.expand_dim, self.expand_dim), device=self.device, dtype=self.dtype
        )
        self.Q = torch.zeros(
            (self.expand_dim, self.num_classes), device=self.device, dtype=self.dtype
        )
        self.counts = torch.zeros(self.num_classes, device=self.device, dtype=self.dtype)
        self.weights: torch.Tensor | None = None
        self.last_ridge: float | None = None
        self.diagnostics = {
            "projection": "fixed_sparse_gaussian_wta",
            "ridge_policy": "original_current_task_gcv",
            "ridge_exponents": list(range(int(self.ridge_lower), int(self.ridge_upper))),
            "global_output_width": self.num_classes,
        }

    @property
    def class_ids(self):
        return list(range(self.num_classes))

    def _encode(self, features):
        x = features.to(self.device, self.dtype)
        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (B, {self.feature_dim})")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("features contain NaN or Inf")
        return self.flyhash(x, self.coding_level, absolute_wta=False)

    def _targets(self, labels):
        y = labels.to(self.device, torch.long)
        if y.ndim != 1 or bool(((y < 0) | (y >= self.num_classes)).any()):
            raise ValueError("labels must be global class IDs in [0, num_classes)")
        return torch.nn.functional.one_hot(y, num_classes=self.num_classes).to(self.dtype)

    def update(self, features, labels):
        encoded = self._encode(features)
        targets = self._targets(labels)
        self.G += encoded.T @ encoded
        self.Q += encoded.T @ targets
        self.counts += torch.bincount(
            labels.to(self.device, torch.long), minlength=self.num_classes
        ).to(self.dtype)
        selected = select_fly_ridge_parameter(
            encoded, targets, self.ridge_lower, self.ridge_upper
        )
        self.last_ridge = float(selected.item())
        self.weights = _ridge(self.G, self.Q, self.last_ridge)
        self.diagnostics.update(
            effective_rank=self.expand_dim,
            selected_ridge=self.last_ridge,
        )

    def predict_logits(self, features):
        if self.weights is None:
            raise RuntimeError("update() must be called before prediction")
        return self._encode(features) @ self.weights

    def predict(self, features):
        return self.predict_logits(features).argmax(1).detach().cpu()

    def persistent_tensors(self):
        tensors = {
            "projection": self.flyhash.projection_matrix,
            "G": self.G,
            "Q": self.Q,
            "counts": self.counts,
        }
        if self.weights is not None:
            tensors["weights"] = self.weights
        return tensors

    def persistent_state_bytes(self):
        return sum(_tensor_storage_bytes(tensor) for tensor in self.persistent_tensors().values())

    def state_dict(self):
        return {
            "version": 1,
            "feature_dim": self.feature_dim,
            "expand_dim": self.expand_dim,
            "synaptic_degree": self.synaptic_degree,
            "coding_level": self.coding_level,
            "num_classes": self.num_classes,
            "ridge_lower": self.ridge_lower,
            "ridge_upper": self.ridge_upper,
            "seed": self.seed,
            "G": self.G.detach().cpu().clone(),
            "Q": self.Q.detach().cpu().clone(),
            "counts": self.counts.detach().cpu().clone(),
            "last_ridge": self.last_ridge,
        }

    def load_state_dict(self, state):
        expected = (
            self.feature_dim, self.expand_dim, self.synaptic_degree,
            self.coding_level, self.num_classes, self.ridge_lower,
            self.ridge_upper, self.seed,
        )
        actual = (
            int(state["feature_dim"]), int(state["expand_dim"]),
            int(state["synaptic_degree"]), float(state["coding_level"]),
            int(state["num_classes"]), float(state["ridge_lower"]),
            float(state["ridge_upper"]), int(state["seed"]),
        )
        if actual != expected:
            raise ValueError("CachedFlyCLFidelity checkpoint configuration mismatch")
        self.G = state["G"].to(self.device, self.dtype)
        self.Q = state["Q"].to(self.device, self.dtype)
        self.counts = state["counts"].to(self.device, self.dtype)
        self.last_ridge = None if state["last_ridge"] is None else float(state["last_ridge"])
        self.weights = None if self.last_ridge is None else _ridge(self.G, self.Q, self.last_ridge)
        self.diagnostics["selected_ridge"] = self.last_ridge


class CachedSOHOReplayFidelity:
    """Feature-cache execution of current SOHO replay and GCV semantics.

    This adapter is deliberately not exemplar-free: its checkpoint retains all
    historical backbone features and labels needed after the OLDA map changes.
    """

    is_exemplar_free = False

    def __init__(self, feature_dim, expand_dim, density, olda_dim, use_etf,
                 coding_level, num_classes, ridge_lower, ridge_upper,
                 seed=1993, device="cpu", dtype=torch.float32,
                 replay_chunk_size=2000, gcv_sample_size=3000):
        self.feature_dim = int(feature_dim)
        self.expand_dim = int(expand_dim)
        self.density = float(density)
        self.olda_dim = int(olda_dim)
        self.use_etf = bool(use_etf)
        self.coding_level = float(coding_level)
        self.num_classes = int(num_classes)
        self.ridge_lower = float(ridge_lower)
        self.ridge_upper = float(ridge_upper)
        self.seed = int(seed)
        self.device, self.dtype = torch.device(device), dtype
        self.replay_chunk_size = int(replay_chunk_size)
        self.gcv_sample_size = int(gcv_sample_size)
        if (
            self.num_classes <= 0 or self.ridge_lower >= self.ridge_upper
            or self.replay_chunk_size <= 0 or self.gcv_sample_size <= 0
        ):
            raise ValueError("invalid SOHO fidelity configuration")
        self.soho, self._gcv_rng_state = self._new_soho_and_rng_state()
        self.feature_history: list[torch.Tensor] = []
        self.label_history: list[torch.Tensor] = []
        self.G = torch.zeros(
            (self.expand_dim, self.expand_dim), device=self.device, dtype=self.dtype
        )
        self.Q = torch.zeros(
            (self.expand_dim, self.num_classes), device=self.device, dtype=self.dtype
        )
        self.weights: torch.Tensor | None = None
        self.last_ridge: float | None = None
        self.diagnostics = {
            "replay_required": True,
            "reason": "dynamic_OLDA_projection_plus_sample_dependent_TopK",
            "ridge_policy": "current_soho_replay_sample_gcv",
            "ridge_exponents": list(range(int(self.ridge_lower), int(self.ridge_upper))),
            "global_output_width": self.num_classes,
        }

    @property
    def class_ids(self):
        return list(range(self.num_classes))

    def _rng_devices(self):
        if self.device.type != "cuda":
            return []
        return [self.device.index if self.device.index is not None else torch.cuda.current_device()]

    def _new_soho_and_rng_state(self):
        devices = self._rng_devices()
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.seed)
            model = SOHO(
                self.feature_dim, self.expand_dim, self.device,
                density=self.density, olda_dim=self.olda_dim,
                use_etf=self.use_etf,
            ).to(self.device)
            state = (
                torch.cuda.get_rng_state(self.device)
                if self.device.type == "cuda" else torch.get_rng_state()
            )
        return model, state

    def _randperm(self, count):
        devices = self._rng_devices()
        with torch.random.fork_rng(devices=devices):
            if self.device.type == "cuda":
                torch.cuda.set_rng_state(self._gcv_rng_state, self.device)
            else:
                torch.set_rng_state(self._gcv_rng_state)
            indices = torch.randperm(count, device=self.device)
            self._gcv_rng_state = (
                torch.cuda.get_rng_state(self.device)
                if self.device.type == "cuda" else torch.get_rng_state()
            )
        return indices

    def _targets(self, labels):
        if labels.ndim != 1 or bool(((labels < 0) | (labels >= self.num_classes)).any()):
            raise ValueError("labels must be global class IDs in [0, num_classes)")
        return torch.nn.functional.one_hot(
            labels, num_classes=self.num_classes
        ).to(self.dtype)

    def update(self, features, labels):
        x = features.to(self.device, self.dtype)
        y = labels.to(self.device, torch.long)
        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(f"features must have shape (B, {self.feature_dim})")
        if not bool(torch.isfinite(x).all()):
            raise ValueError("features contain NaN or Inf")
        self._targets(y)
        self.soho.update_stats(x, y)
        self.feature_history.append(x.detach().clone())
        self.label_history.append(y.detach().clone())
        historical_x = torch.cat(self.feature_history, dim=0)
        historical_y = torch.cat(self.label_history, dim=0)
        targets = self._targets(historical_y)
        self.G.zero_()
        self.Q.zero_()
        for start in range(0, len(historical_x), self.replay_chunk_size):
            stop = min(start + self.replay_chunk_size, len(historical_x))
            encoded = self.soho(
                historical_x[start:stop], self.coding_level, absolute_wta=False
            )
            self.G += encoded.T @ encoded
            self.Q += encoded.T @ targets[start:stop]
        sample_count = min(self.gcv_sample_size, len(historical_x))
        sample = self._randperm(len(historical_x))[:sample_count]
        gcv_features = self.soho(
            historical_x[sample], self.coding_level, absolute_wta=False
        )
        selected = select_soho_ridge_parameter(
            gcv_features, targets[sample], self.ridge_lower, self.ridge_upper
        )
        self.last_ridge = float(selected.item())
        self.weights = _ridge(self.G, self.Q, self.last_ridge)
        self.diagnostics.update(
            effective_rank=int(self.soho.R.shape[0]),
            selected_ridge=self.last_ridge,
            retained_sample_count=int(len(historical_x)),
        )

    def predict_logits(self, features):
        if self.weights is None:
            raise RuntimeError("update() must be called before prediction")
        x = features.to(self.device, self.dtype)
        return self.soho(x, self.coding_level, absolute_wta=False) @ self.weights

    def predict(self, features):
        return self.predict_logits(features).argmax(1).detach().cpu()

    def persistent_tensors(self):
        tensors = {
            "G": self.G,
            "Q": self.Q,
            "R": self.soho.R,
            "W": self.soho.W,
            "olda_global_sum": self.soho.olda.global_sum,
            "olda_S_w": self.soho.olda.S_w,
            "gcv_rng_state": self._gcv_rng_state,
        }
        tensors.update({
            f"feature_history_{index}": value
            for index, value in enumerate(self.feature_history)
        })
        tensors.update({
            f"label_history_{index}": value
            for index, value in enumerate(self.label_history)
        })
        tensors.update({
            f"olda_class_sum_{class_id}": value
            for class_id, value in self.soho.olda.class_sums.items()
        })
        if self.weights is not None:
            tensors["weights"] = self.weights
        return tensors

    def persistent_state_bytes(self):
        return sum(_tensor_storage_bytes(tensor) for tensor in self.persistent_tensors().values())

    def state_dict(self):
        return {
            "version": 1,
            "feature_dim": self.feature_dim,
            "expand_dim": self.expand_dim,
            "density": self.density,
            "olda_dim": self.olda_dim,
            "use_etf": self.use_etf,
            "coding_level": self.coding_level,
            "num_classes": self.num_classes,
            "ridge_lower": self.ridge_lower,
            "ridge_upper": self.ridge_upper,
            "seed": self.seed,
            "replay_chunk_size": self.replay_chunk_size,
            "gcv_sample_size": self.gcv_sample_size,
            "feature_history": [value.detach().cpu().clone() for value in self.feature_history],
            "label_history": [value.detach().cpu().clone() for value in self.label_history],
            "G": self.G.detach().cpu().clone(),
            "Q": self.Q.detach().cpu().clone(),
            "last_ridge": self.last_ridge,
            "gcv_rng_state": self._gcv_rng_state.cpu().clone(),
            "soho": {
                "R": self.soho.R.detach().cpu().clone(),
                "W": self.soho.W.detach().cpu().clone(),
                "global_sum": self.soho.olda.global_sum.detach().cpu().clone(),
                "global_count": self.soho.olda.global_count,
                "S_w": self.soho.olda.S_w.detach().cpu().clone(),
                "class_sums": {
                    key: value.detach().cpu().clone()
                    for key, value in self.soho.olda.class_sums.items()
                },
                "class_counts": dict(self.soho.olda.class_counts),
            },
        }

    def load_state_dict(self, state):
        expected = (
            self.feature_dim, self.expand_dim, self.density, self.olda_dim,
            self.use_etf, self.coding_level, self.num_classes,
            self.ridge_lower, self.ridge_upper, self.seed,
            self.replay_chunk_size, self.gcv_sample_size,
        )
        actual = (
            int(state["feature_dim"]), int(state["expand_dim"]),
            float(state["density"]), int(state["olda_dim"]),
            bool(state["use_etf"]), float(state["coding_level"]),
            int(state["num_classes"]), float(state["ridge_lower"]),
            float(state["ridge_upper"]), int(state["seed"]),
            int(state["replay_chunk_size"]), int(state["gcv_sample_size"]),
        )
        if actual != expected:
            raise ValueError("CachedSOHOReplayFidelity checkpoint configuration mismatch")
        self.feature_history = [value.to(self.device, self.dtype) for value in state["feature_history"]]
        self.label_history = [value.to(self.device, torch.long) for value in state["label_history"]]
        self.G = state["G"].to(self.device, self.dtype)
        self.Q = state["Q"].to(self.device, self.dtype)
        self.last_ridge = None if state["last_ridge"] is None else float(state["last_ridge"])
        # PyTorch CUDA RNG APIs also serialize their state as a CPU ByteTensor.
        self._gcv_rng_state = state["gcv_rng_state"].cpu()
        packed = state["soho"]
        self.soho.R = packed["R"].to(self.device, self.dtype)
        self.soho.W = packed["W"].to(self.device, self.dtype)
        self.soho.olda.global_sum = packed["global_sum"].to(self.device, self.dtype)
        self.soho.olda.global_count = int(packed["global_count"])
        self.soho.olda.S_w = packed["S_w"].to(self.device, self.dtype)
        self.soho.olda.class_sums = {
            int(key): value.to(self.device, self.dtype)
            for key, value in packed["class_sums"].items()
        }
        self.soho.olda.class_counts = {
            int(key): int(value) for key, value in packed["class_counts"].items()
        }
        self.weights = None if self.last_ridge is None else _ridge(
            self.G, self.Q, self.last_ridge
        )
        self.diagnostics.update(
            selected_ridge=self.last_ridge,
            retained_sample_count=sum(len(value) for value in self.feature_history),
        )
