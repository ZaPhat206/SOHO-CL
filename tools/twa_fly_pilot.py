"""Locked train-only Phase A pilot for Two-Way Analytic FLY.

The runner has deliberately no held-out evaluation mode. It physically checks
that ``test.pt`` is absent before opening the train cache. Sample-level WTA
codes are experiment infrastructure on disk, never learner state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.flycl import select_ridge_parameter
from methods.twa_fly import TWAFLYLearner, TWAStatistics, factor_coupled_systems
from methods.twa_fly.solver import solve_one_way, solve_symmetric
from tools.experiment_runner import split, train_validation_indices, validate_cache


CONFIG_KEYS = {
    "schema_version", "study_id", "dataset", "model_name", "checkpoint_sha256",
    "seed", "num_classes", "num_tasks", "validation_fraction", "representation",
    "raw_ridge_lambda", "fly_ridge_lower", "fly_ridge_upper", "rho_candidates",
    "solver_tolerance", "solver_max_iterations", "statistics_dtype", "gate",
}
REPRESENTATION_KEYS = {"expand_dim", "synaptic_degree", "coding_level", "encode_batch_size"}
GATE_KEYS = {
    "minimum_gain_over_fly_pp", "minimum_gain_over_one_way_pp",
    "minimum_gain_over_shuffled_pp", "maximum_state_fraction_of_fly",
    "maximum_solver_relative_residual",
}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tensor_content_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor values and sparse structure without relying on torch.save."""
    digest = hashlib.sha256()
    digest.update(str(tensor.layout).encode("ascii"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    if tensor.layout == torch.strided:
        parts = (tensor,)
    elif tensor.layout == torch.sparse_csc:
        parts = (tensor.ccol_indices(), tensor.row_indices(), tensor.values())
    else:
        raise ValueError(f"unsupported tensor layout {tensor.layout}")
    for part in parts:
        value = part.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _sequence_sha256(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().to(torch.int64).contiguous()
        digest.update(value.numel().to_bytes(8, "little"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _git_provenance() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"runner_git_commit": commit, "runner_git_dirty": dirty}


def _read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError(f"config keys must be exactly {sorted(CONFIG_KEYS)}")
    if config["schema_version"] != 1:
        raise ValueError("unsupported TWA-FLY config schema")
    if set(config["representation"]) != REPRESENTATION_KEYS or set(config["gate"]) != GATE_KEYS:
        raise ValueError("invalid nested config keys")
    if config["num_classes"] <= 1 or config["num_tasks"] <= 0 or config["num_classes"] % config["num_tasks"]:
        raise ValueError("invalid class/task count")
    if not 0 < config["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    if not config["rho_candidates"] or any(float(rho) <= 0 for rho in config["rho_candidates"]):
        raise ValueError("rho_candidates must be a non-empty positive list")
    if len(set(map(float, config["rho_candidates"]))) != len(config["rho_candidates"]):
        raise ValueError("rho_candidates must be unique")
    if config["statistics_dtype"] not in {"float32", "float64"}:
        raise ValueError("statistics_dtype must be float32 or float64")
    if config["raw_ridge_lambda"] <= 0 or config["fly_ridge_lower"] >= config["fly_ridge_upper"]:
        raise ValueError("invalid Ridge configuration")
    if config["solver_tolerance"] <= 0 or config["solver_max_iterations"] <= 0:
        raise ValueError("invalid solver configuration")
    return config


def _projection_identity(config: dict, feature_dim: int) -> dict:
    representation = config["representation"]
    return {
        "raw_dim": int(feature_dim),
        "expand_dim": int(representation["expand_dim"]),
        "synaptic_degree": int(representation["synaptic_degree"]),
        "coding_level": float(representation["coding_level"]),
        "seed": int(config["seed"]),
        "statistics_dtype": config["statistics_dtype"],
    }


def _new_learner(
    config: dict, feature_dim: int, method: str, rho: float, device,
    projection: torch.Tensor | None = None,
) -> TWAFLYLearner:
    representation = config["representation"]
    dtype = {"float32": torch.float32, "float64": torch.float64}[config["statistics_dtype"]]
    return TWAFLYLearner(
        method=method,
        raw_dim=feature_dim,
        fly_dim=int(representation["expand_dim"]),
        num_classes=int(config["num_classes"]),
        synaptic_degree=int(representation["synaptic_degree"]),
        coding_level=float(representation["coding_level"]),
        rho=float(rho),
        raw_ridge=float(config["raw_ridge_lambda"]),
        fly_ridge=1.0,
        solver_tolerance=float(config["solver_tolerance"]),
        solver_max_iterations=int(config["solver_max_iterations"]),
        seed=int(config["seed"]),
        device=device,
        dtype=dtype,
        projection=projection,
    )


def _cache_probe_indices(sample_count: int, seed: int, maximum: int = 16) -> torch.Tensor:
    count = min(int(sample_count), int(maximum))
    if count <= 0:
        raise ValueError("code cache cannot be probed without samples")
    selected = random.Random(int(seed) + 982451653).sample(range(sample_count), count)
    return torch.tensor(sorted(selected), dtype=torch.long)


def _verify_projection_probe(
    *, prototype: TWAFLYLearner, train: dict, indices: torch.Tensor,
    values: torch.Tensor, seed: int,
) -> dict:
    sample_indices = _cache_probe_indices(len(train["features"]), seed)
    cached_indices = indices[sample_indices].to(torch.long)
    cached_values = values[sample_indices]
    features = train["features"][sample_indices].to(
        device=prototype.device, dtype=prototype.flyhash.projection_matrix.dtype
    )
    projection = prototype.flyhash.projection_matrix
    projected = (
        torch.sparse.mm(projection, features.T).T
        if projection.layout == torch.sparse_csc else features @ projection.T
    )
    device_indices = cached_indices.to(projected.device)
    recomputed_values = projected.gather(1, device_indices).detach().cpu()
    cached_values = cached_values.to(recomputed_values.dtype)
    try:
        torch.testing.assert_close(
            recomputed_values, cached_values, atol=5e-5, rtol=1e-5
        )
    except AssertionError as error:
        raise RuntimeError("WTA code cache projection probe value mismatch") from error
    active_size = cached_indices.shape[1]
    observed_values, observed_indices = projected.topk(active_size, dim=1, largest=True)
    cutoff = observed_values[:, -1:]
    tolerance = 5e-5 + 1e-5 * cutoff.abs()
    maximum_membership_violation = float(
        (cutoff - recomputed_values.to(cutoff.device) - tolerance).clamp_min(0).max().item()
    )
    if maximum_membership_violation > 0:
        raise RuntimeError("WTA code cache projection probe Top-K membership mismatch")
    cached_mask = torch.zeros(
        (len(sample_indices), prototype.fly_dim), dtype=torch.bool, device=projected.device
    )
    cached_mask.scatter_(1, device_indices, True)
    overlap = float(cached_mask.gather(1, observed_indices).float().mean().item())
    return {
        "verified": True,
        "verification_semantics": "cached values match projection scores and satisfy tolerant Top-K membership",
        "sample_indices": sample_indices.tolist(),
        "sample_indices_sha256": _tensor_content_sha256(sample_indices),
        "cached_indices_sha256": _tensor_content_sha256(cached_indices),
        "cached_values_sha256": _tensor_content_sha256(cached_values),
        "topk_index_overlap_fraction": overlap,
        "maximum_topk_membership_violation": maximum_membership_violation,
        "atol": 5e-5,
        "rtol": 1e-5,
    }


def _prepare_code_cache(*, train: dict, train_sha256: str, cache_dir: Path, config: dict, device):
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "metadata.json"
    codes_path = cache_dir / "train_codes.pt"
    projection_path = cache_dir / "projection.pt"
    identity = _projection_identity(config, train["features"].shape[1])
    identity["source_train_sha256"] = train_sha256
    identity["sample_count"] = int(len(train["features"]))
    identity_sha256 = _sha256_bytes(json.dumps(identity, sort_keys=True).encode("utf-8"))
    if metadata_path.exists() or codes_path.exists():
        if not (metadata_path.is_file() and codes_path.is_file()):
            raise RuntimeError("incomplete WTA code cache; choose a new code-cache-dir")
        if projection_path.exists():
            projection = torch.load(projection_path, weights_only=True, map_location="cpu")
            prototype = _new_learner(
                config, train["features"].shape[1], "twa_symmetric", 0.0,
                device, projection=projection,
            )
        else:
            # A schema-1 cache can be upgraded only if the configured seed
            # reproduces its rows. Otherwise its projection is irrecoverable.
            prototype = _new_learner(
                config, train["features"].shape[1], "twa_symmetric", 0.0, device
            )
            projection = prototype.flyhash.projection_matrix.detach().cpu()
        projection_sha256 = _tensor_content_sha256(prototype.flyhash.projection_matrix)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("identity_sha256") != identity_sha256:
            raise RuntimeError("stale WTA code cache identity; choose a new code-cache-dir")
        if metadata.get("codes_sha256") != _sha256_file(codes_path):
            raise RuntimeError("WTA code cache SHA-256 mismatch")
        packed = torch.load(codes_path, weights_only=True, map_location="cpu")
        indices, values = packed["indices"], packed["values"]
        expected = (len(train["features"]), max(1, int(identity["expand_dim"] * identity["coding_level"])))
        if indices.shape != expected or values.shape != expected or not bool(torch.isfinite(values).all()):
            raise RuntimeError("WTA code cache tensor validation failed")
        recorded_projection = metadata.get("projection", {}).get("sha256")
        if recorded_projection is not None and recorded_projection != projection_sha256:
            raise RuntimeError("WTA code cache projection SHA-256 mismatch")
        probe = _verify_projection_probe(
            prototype=prototype, train=train, indices=indices, values=values,
            seed=config["seed"],
        )
        if not projection_path.exists():
            temporary_projection = projection_path.with_suffix(projection_path.suffix + ".tmp")
            torch.save(projection, temporary_projection)
            os.replace(temporary_projection, projection_path)
        # Schema-1 caches are upgraded only after their cached rows have been
        # reproduced from the regenerated projection. The 900 MB code tensor is
        # not rewritten.
        metadata.update({
            "schema_version": 2,
            "projection": {
                "sha256": projection_sha256,
                "materialization_torch": torch.__version__,
                "artifact": projection_path.name,
                "disk_bytes": projection_path.stat().st_size,
                "probe": probe,
            },
        })
        _atomic_json(metadata_path, metadata)
        print(f"WTA CACHE restored samples={expected[0]} active={expected[1]} disk={codes_path.stat().st_size}B", flush=True)
        return indices, values, metadata, prototype.flyhash.projection_matrix

    if projection_path.exists():
        raise RuntimeError("incomplete WTA code cache; choose a new code-cache-dir")
    prototype = _new_learner(
        config, train["features"].shape[1], "twa_symmetric", 0.0, device
    )
    projection = prototype.flyhash.projection_matrix.detach().cpu()
    projection_sha256 = _tensor_content_sha256(projection)
    sample_count = len(train["features"])
    active_size = max(1, int(prototype.fly_dim * prototype.coding_level))
    index_dtype = torch.int16 if prototype.fly_dim <= 32767 else torch.int32
    indices = torch.empty((sample_count, active_size), dtype=index_dtype)
    values = torch.empty((sample_count, active_size), dtype=prototype.dtype)
    batch_size = int(config["representation"]["encode_batch_size"])
    started = time.perf_counter()
    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        batch_indices, batch_values = prototype.encode_sparse_fly(train["features"][start:stop])
        indices[start:stop] = batch_indices.detach().cpu().to(index_dtype)
        values[start:stop] = batch_values.detach().cpu()
        elapsed = time.perf_counter() - started
        rate = stop / max(elapsed, 1e-9)
        print(
            f"WTA CACHE {stop}/{sample_count} ({100*stop/sample_count:5.1f}%) "
            f"elapsed={elapsed/60:.1f}m eta={(sample_count-stop)/max(rate,1e-9)/60:.1f}m",
            flush=True,
        )
    temporary = codes_path.with_suffix(codes_path.suffix + ".tmp")
    torch.save({"indices": indices, "values": values}, temporary)
    os.replace(temporary, codes_path)
    temporary_projection = projection_path.with_suffix(projection_path.suffix + ".tmp")
    torch.save(projection, temporary_projection)
    os.replace(temporary_projection, projection_path)
    probe = _verify_projection_probe(
        prototype=prototype, train=train, indices=indices, values=values,
        seed=config["seed"],
    )
    metadata = {
        "schema_version": 2,
        "role": "experiment_cache_not_learner_state",
        "contains_sample_level_codes": True,
        "identity": identity,
        "identity_sha256": identity_sha256,
        "indices_shape": list(indices.shape),
        "indices_dtype": str(indices.dtype),
        "values_shape": list(values.shape),
        "values_dtype": str(values.dtype),
        "finite": bool(torch.isfinite(values).all()),
        "codes_sha256": _sha256_file(codes_path),
        "disk_bytes": codes_path.stat().st_size,
        "projection": {
            "sha256": projection_sha256,
            "materialization_torch": torch.__version__,
            "artifact": projection_path.name,
            "disk_bytes": projection_path.stat().st_size,
            "probe": probe,
        },
    }
    _atomic_json(metadata_path, metadata)
    print(f"WTA CACHE complete shape={tuple(indices.shape)} elapsed={(time.perf_counter()-started)/60:.1f}m", flush=True)
    return indices, values, metadata, prototype.flyhash.projection_matrix


def _dense_codes(indices: torch.Tensor, values: torch.Tensor, fly_dim: int, device, dtype) -> torch.Tensor:
    dense = torch.zeros((indices.shape[0], fly_dim), device=device, dtype=dtype)
    dense.scatter_(1, indices.to(device=device, dtype=torch.long), values.to(device=device, dtype=dtype))
    return dense


def _solve_spd(gram: torch.Tensor, cross: torch.Tensor, ridge: float) -> torch.Tensor:
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    factor = torch.linalg.cholesky((gram + gram.T) * 0.5 + ridge * identity)
    return torch.cholesky_solve(cross, factor)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    if tensor.layout == torch.strided:
        return tensor.numel() * tensor.element_size()
    if tensor.layout == torch.sparse_csc:
        return sum(part.numel() * part.element_size() for part in (
            tensor.ccol_indices(), tensor.row_indices(), tensor.values()
        ))
    raise ValueError(f"unsupported tensor layout {tensor.layout}")


def _accuracy(weights: torch.Tensor, indices: torch.Tensor, values: torch.Tensor, labels: torch.Tensor, fly_dim: int, device, dtype) -> float:
    z = _dense_codes(indices, values, fly_dim, device, dtype)
    predictions = (z @ weights).argmax(1).detach().cpu()
    return float((predictions == labels.cpu()).float().mean().item() * 100)


def _raw_accuracy(weights: torch.Tensor, features: torch.Tensor, labels: torch.Tensor, device, dtype) -> float:
    predictions = (features.to(device=device, dtype=dtype) @ weights).argmax(1).detach().cpu()
    return float((predictions == labels.cpu()).float().mean().item() * 100)


def _state_bytes(statistics: TWAStatistics, projection: torch.Tensor, raw_weights, fly_weights, mode: str) -> int:
    if mode == "fly":
        tensors = (projection, statistics.G_zz, statistics.Q_z, statistics.counts, fly_weights)
    elif mode == "raw":
        tensors = (statistics.G_xx, statistics.Q_x, statistics.counts, raw_weights)
    elif mode == "twa":
        tensors = (
            projection, statistics.G_xx, statistics.G_zz, statistics.R_xz,
            statistics.Q_x, statistics.Q_z, statistics.counts, raw_weights, fly_weights,
        )
    else:
        raise ValueError(mode)
    return sum(_tensor_bytes(tensor) for tensor in tensors)


def _method_key(method: str, rho: float | None = None) -> str:
    return method if rho is None else f"{method}__rho-{str(rho).replace('.', 'p')}"


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    code_cache_dir = Path(args.code_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = _read_config(config_path)
    test_path = feature_cache_dir / "test.pt"
    heldout_hidden = not test_path.exists()
    if args.require_test_hidden and not heldout_hidden:
        raise RuntimeError(f"held-out file is visible: {test_path}; rename it before selection")
    cache_args = argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"])
    train, _, cache_metadata = validate_cache(feature_cache_dir, cache_args, load_test=False)
    if cache_metadata.get("checkpoint_sha256") != config["checkpoint_sha256"]:
        raise ValueError("feature-cache checkpoint hash does not match locked config")
    if sorted(map(int, torch.unique(train["labels"]).tolist())) != list(range(config["num_classes"])):
        raise ValueError("training labels do not match locked global class IDs")
    train_path = feature_cache_dir / "train.pt"
    train_sha256 = _sha256_file(train_path)
    code_indices, code_values, code_metadata, projection = _prepare_code_cache(
        train=train, train_sha256=train_sha256, cache_dir=code_cache_dir,
        config=config, device=args.device,
    )
    class_order = random.Random(config["seed"]).sample(list(range(config["num_classes"])), config["num_classes"])
    tasks = split(train["labels"], class_order, config["num_tasks"])
    train_parts, val_parts = train_validation_indices(
        train["labels"], tasks, config["seed"], config["validation_fraction"]
    )
    dtype = {"float32": torch.float32, "float64": torch.float64}[config["statistics_dtype"]]
    device = torch.device(args.device)
    raw_dim = int(train["features"].shape[1])
    fly_dim = int(config["representation"]["expand_dim"])
    statistics = TWAStatistics(raw_dim, fly_dim, config["num_classes"], device=device, dtype=dtype)
    shuffled_cross = torch.zeros_like(statistics.R_xz)
    result_rows = {
        _method_key("matched_fly"): {"method": "matched_fly", "rho": None, "stage_average_accuracy": [], "solver_relative_residual": []},
        _method_key("raw_ridge"): {"method": "raw_ridge", "rho": None, "stage_average_accuracy": [], "solver_relative_residual": []},
    }
    for rho in map(float, config["rho_candidates"]):
        for method in ("twa_one_way", "twa_symmetric", "twa_shuffled_cross"):
            result_rows[_method_key(method, rho)] = {
                "method": method, "rho": rho, "stage_average_accuracy": [],
                "solver_relative_residual": [], "solver_iterations": [],
            }
    raw_ridge = float(config["raw_ridge_lambda"])
    started = time.perf_counter()
    for task_index, update_indices in enumerate(train_parts):
        task_started = time.perf_counter()
        x = train["features"][update_indices].to(device=device, dtype=dtype)
        z = _dense_codes(code_indices[update_indices], code_values[update_indices], fly_dim, device, dtype)
        y = train["labels"][update_indices].to(device)
        targets = torch.nn.functional.one_hot(y, config["num_classes"]).to(dtype)
        statistics.update(x, z, y)
        permutation = torch.randperm(
            len(update_indices), generator=torch.Generator().manual_seed(config["seed"] + 104729 + task_index)
        ).to(device)
        shuffled_cross.add_(x.T @ z[permutation])
        fly_ridge = float(select_ridge_parameter(
            z, targets, config["fly_ridge_lower"], config["fly_ridge_upper"]
        ).item())
        raw_weights = _solve_spd(statistics.G_xx, statistics.Q_x, raw_ridge)
        fly_weights = _solve_spd(statistics.G_zz, statistics.Q_z, fly_ridge)
        stage_weights = {
            _method_key("matched_fly"): fly_weights,
            _method_key("raw_ridge"): raw_weights,
        }
        for rho in map(float, config["rho_candidates"]):
            factors = factor_coupled_systems(statistics, rho, raw_ridge, fly_ridge)
            one_way = solve_one_way(
                statistics, rho, raw_ridge, fly_ridge,
                raw_teacher=raw_weights, fly_factor=factors.fly_factor,
            )
            symmetric = solve_symmetric(
                statistics, rho, raw_ridge, fly_ridge,
                tolerance=config["solver_tolerance"], max_iterations=config["solver_max_iterations"],
                factors=factors,
            )
            shuffled = solve_symmetric(
                statistics, rho, raw_ridge, fly_ridge,
                tolerance=config["solver_tolerance"], max_iterations=config["solver_max_iterations"],
                cross=shuffled_cross, factors=factors,
            )
            for method, solution in (
                ("twa_one_way", one_way), ("twa_symmetric", symmetric),
                ("twa_shuffled_cross", shuffled),
            ):
                key = _method_key(method, rho)
                stage_weights[key] = solution.fly_weights
                result_rows[key]["solver_relative_residual"].append(solution.relative_residual)
                result_rows[key]["solver_iterations"].append(solution.iterations)
        for key, weights in stage_weights.items():
            indices = torch.cat(val_parts[:task_index + 1])
            if key == "raw_ridge":
                score = _raw_accuracy(
                    raw_weights, train["features"][indices], train["labels"][indices],
                    device, dtype,
                )
            else:
                score = _accuracy(
                    weights, code_indices[indices], code_values[indices],
                    train["labels"][indices], fly_dim, device, dtype,
                )
            result_rows[key]["stage_average_accuracy"].append(score)
        print(
            f"TASK {task_index+1}/{len(train_parts)} ridge_z={fly_ridge:g} "
            f"fly_AA={result_rows['matched_fly']['stage_average_accuracy'][-1]:.4f} "
            f"elapsed={time.perf_counter()-task_started:.1f}s total={(time.perf_counter()-started)/60:.1f}m",
            flush=True,
        )
        del x, z, targets, stage_weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    results = []
    for key, result in result_rows.items():
        result["validation_average_accuracy"] = float(sum(result["stage_average_accuracy"]) / len(result["stage_average_accuracy"]))
        if result["method"] == "matched_fly":
            result["persistent_state_bytes"] = _state_bytes(statistics, projection, raw_weights, fly_weights, "fly")
        elif result["method"] == "raw_ridge":
            result["persistent_state_bytes"] = _state_bytes(statistics, projection, raw_weights, fly_weights, "raw")
        else:
            result["persistent_state_bytes"] = _state_bytes(statistics, projection, raw_weights, fly_weights, "twa")
        result["uses_test_set"] = False
        result["exemplar_free"] = True
        results.append(result)
    fly = result_rows["matched_fly"]
    symmetric = max(
        (row for row in results if row["method"] == "twa_symmetric"),
        key=lambda row: row["validation_average_accuracy"],
    )
    selected_rho = symmetric["rho"]
    one_way = result_rows[_method_key("twa_one_way", selected_rho)]
    shuffled = result_rows[_method_key("twa_shuffled_cross", selected_rho)]
    thresholds = config["gate"]
    max_residual = max(symmetric["solver_relative_residual"])
    state_fraction = symmetric["persistent_state_bytes"] / fly["persistent_state_bytes"]
    gates = {
        "beats_matched_fly": bool(symmetric["validation_average_accuracy"] - fly["validation_average_accuracy"] >= thresholds["minimum_gain_over_fly_pp"]),
        "beats_one_way": bool(symmetric["validation_average_accuracy"] - one_way["validation_average_accuracy"] >= thresholds["minimum_gain_over_one_way_pp"]),
        "beats_shuffled_cross": bool(symmetric["validation_average_accuracy"] - shuffled["validation_average_accuracy"] >= thresholds["minimum_gain_over_shuffled_pp"]),
        "numerical_stability": bool(max_residual <= thresholds["maximum_solver_relative_residual"]),
        "state_budget": bool(state_fraction <= thresholds["maximum_state_fraction_of_fly"]),
        "heldout_test_remained_hidden": bool(heldout_hidden),
    }
    gate = {
        "decision": "REVIEW_FOR_HELDOUT_AUTHORIZATION" if all(gates.values()) else "STOP_TRAIN_ONLY_GATE_FAILED",
        "gates": gates,
        "diagnostics": {
            "selected_rho": selected_rho,
            "twa_minus_fly_pp": symmetric["validation_average_accuracy"] - fly["validation_average_accuracy"],
            "twa_minus_one_way_pp": symmetric["validation_average_accuracy"] - one_way["validation_average_accuracy"],
            "twa_minus_shuffled_pp": symmetric["validation_average_accuracy"] - shuffled["validation_average_accuracy"],
            "maximum_solver_relative_residual": max_residual,
            "state_fraction_of_fly": state_fraction,
        },
        "selected_symmetric": symmetric,
    }
    provenance = {
        **_git_provenance(),
        "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
        "device": str(device), "config_path": str(config_path), "config_sha256": _sha256_file(config_path),
        "feature_cache_dir": str(feature_cache_dir), "feature_cache_metadata": cache_metadata,
        "train_pt_sha256": train_sha256, "code_cache_dir": str(code_cache_dir),
        "code_cache_identity_sha256": code_metadata["identity_sha256"],
        "class_order": class_order,
        "class_order_sha256": _sha256_bytes(",".join(map(str, class_order)).encode("ascii")),
        "training_indices_sha256": _sequence_sha256(train_parts),
        "validation_indices_sha256": _sequence_sha256(val_parts),
        "heldout_test_path_visible": not heldout_hidden,
    }
    payload = {
        "schema_version": 1, "study_id": config["study_id"],
        "selection_protocol": "mean of per-stage validation accuracies on deterministic stratified training split",
        "uses_test_set": False, "held_out_test_authorized": False,
        "config": config, "run_provenance": provenance, "code_cache": code_metadata,
        "candidates": results, "gate": gate,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "selection.json", payload)
    _atomic_json(output_dir / "gate_results.json", gate)
    print(json.dumps(_jsonable(gate), indent=2), flush=True)
    print("TRAIN-ONLY COMPLETE. Held-out evaluation remains unauthorized.", flush=True)
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--code-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-test-hidden", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
