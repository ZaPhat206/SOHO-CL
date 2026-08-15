"""Locked train-only feasibility pilot for ZI-SOHO.

The runner deliberately has no held-out evaluation mode. Hyperparameters are
read only from one versioned JSON config, and ``test.pt`` can be required to be
physically absent before any feature tensor is loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
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

from methods.cached_replay_baselines import CachedFlyCLFidelity
from methods.sft_cl import create_learner as create_sft_learner
from methods.zi_soho import ZISOHOLearner
from tools.experiment_runner import split, train_validation_indices, validate_cache


CONFIG_KEYS = {
    "schema_version", "study_id", "dataset", "model_name",
    "checkpoint_sha256", "seed", "num_classes", "num_tasks",
    "validation_fraction", "representation", "raw_ridge_lambda",
    "fly_ridge_lower", "fly_ridge_upper", "support_alpha", "variance_kappas",
    "variance_epsilon", "statistics_dtype", "gate",
}
REPRESENTATION_KEYS = {
    "expand_dim", "synaptic_degree", "coding_level", "encode_batch_size",
    "score_chunk_size",
}
GATE_KEYS = {
    "minimum_gain_over_raw_pp", "maximum_gap_to_fly_pp",
    "minimum_gain_over_component_pp", "maximum_state_fraction_of_fly",
}
ZI_METHODS = ("wta_ncm", "support_only", "active_gaussian", "hurdle")


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
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sequence_sha256(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().to(torch.int64).contiguous()
        digest.update(value.numel().to_bytes(8, "little"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _read_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != CONFIG_KEYS:
        raise ValueError(
            f"config keys must be exactly {sorted(CONFIG_KEYS)}; got {sorted(payload)}"
        )
    if payload["schema_version"] != 1:
        raise ValueError("unsupported ZI-SOHO pilot schema_version")
    if set(payload["representation"]) != REPRESENTATION_KEYS:
        raise ValueError("invalid representation config keys")
    if set(payload["gate"]) != GATE_KEYS:
        raise ValueError("invalid gate config keys")
    if payload["num_classes"] <= 1 or payload["num_tasks"] <= 0:
        raise ValueError("invalid class/task count")
    if payload["num_classes"] % payload["num_tasks"]:
        raise ValueError("num_classes must be divisible by num_tasks")
    if not 0 < payload["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be in (0,1)")
    if not payload["variance_kappas"] or any(
        float(value) <= 0 for value in payload["variance_kappas"]
    ):
        raise ValueError("variance_kappas must be a non-empty positive list")
    if payload["statistics_dtype"] not in {"float32", "float64"}:
        raise ValueError("statistics_dtype must be float32 or float64")
    return payload


def _git_provenance() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT, text=True, stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"runner_git_commit": commit, "runner_git_dirty": dirty}


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


def _new_zi(config: dict, feature_dim: int, method: str, kappa: float, device) -> ZISOHOLearner:
    representation = config["representation"]
    dtype = {"float32": torch.float32, "float64": torch.float64}[
        config["statistics_dtype"]
    ]
    return ZISOHOLearner(
        raw_dim=feature_dim,
        expand_dim=int(representation["expand_dim"]),
        synaptic_degree=int(representation["synaptic_degree"]),
        coding_level=float(representation["coding_level"]),
        method=method,
        support_alpha=float(config["support_alpha"]),
        variance_kappa=float(kappa),
        variance_epsilon=float(config["variance_epsilon"]),
        score_chunk_size=int(representation["score_chunk_size"]),
        seed=int(config["seed"]),
        device=device,
        dtype=dtype,
    )


def _prepare_code_cache(
    *, train: dict, train_sha256: str, cache_dir: Path, config: dict, device
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path, codes_path = cache_dir / "metadata.json", cache_dir / "train_codes.pt"
    identity = _projection_identity(config, train["features"].shape[1])
    identity["source_train_sha256"] = train_sha256
    identity["sample_count"] = int(len(train["features"]))
    identity_sha256 = _sha256_bytes(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    )
    if metadata_path.exists() or codes_path.exists():
        if not (metadata_path.is_file() and codes_path.is_file()):
            raise RuntimeError("incomplete WTA code cache; choose a new code-cache-dir")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("identity_sha256") != identity_sha256:
            raise RuntimeError("stale WTA code cache identity; choose a new code-cache-dir")
        observed_codes_sha256 = _sha256_file(codes_path)
        if metadata.get("codes_sha256") != observed_codes_sha256:
            raise RuntimeError("WTA code cache SHA-256 mismatch")
        packed = torch.load(codes_path, weights_only=True)
        indices, values = packed["indices"], packed["values"]
        expected = (
            len(train["features"]),
            max(1, int(identity["expand_dim"] * identity["coding_level"])),
        )
        if indices.shape != expected or values.shape != expected:
            raise RuntimeError("WTA code cache shape mismatch")
        if indices.dtype not in {torch.int16, torch.int32, torch.int64}:
            raise RuntimeError("WTA code indices have invalid dtype")
        if values.dtype not in {torch.float32, torch.float64} or not bool(torch.isfinite(values).all()):
            raise RuntimeError("WTA code values are invalid")
        print(
            f"WTA CACHE restored samples={expected[0]} active={expected[1]} "
            f"disk={codes_path.stat().st_size}B",
            flush=True,
        )
        return indices, values, metadata

    prototype = _new_zi(
        config, train["features"].shape[1], "hurdle",
        float(config["variance_kappas"][0]), device,
    )
    sample_count, active_size = len(train["features"]), prototype.active_size
    index_dtype = torch.int16 if prototype.expand_dim <= 32767 else torch.int32
    indices = torch.empty((sample_count, active_size), dtype=index_dtype)
    values = torch.empty((sample_count, active_size), dtype=prototype.dtype)
    batch_size = int(config["representation"]["encode_batch_size"])
    started = time.perf_counter()
    for start in range(0, sample_count, batch_size):
        stop = min(start + batch_size, sample_count)
        batch_indices, batch_values = prototype.encode_sparse(
            train["features"][start:stop]
        )
        indices[start:stop] = batch_indices.detach().cpu().to(index_dtype)
        values[start:stop] = batch_values.detach().cpu()
        elapsed = time.perf_counter() - started
        rate = stop / max(elapsed, 1e-9)
        eta = (sample_count - stop) / max(rate, 1e-9)
        print(
            f"WTA CACHE {stop}/{sample_count} ({100*stop/sample_count:5.1f}%) "
            f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
            flush=True,
        )
    packed = {"indices": indices, "values": values}
    temporary_codes = codes_path.with_suffix(codes_path.suffix + ".tmp")
    torch.save(packed, temporary_codes)
    os.replace(temporary_codes, codes_path)
    metadata = {
        "schema_version": 1,
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
    }
    _atomic_json(metadata_path, metadata)
    print(
        f"WTA CACHE complete shape={tuple(indices.shape)} "
        f"elapsed={(time.perf_counter()-started)/60:.1f}m",
        flush=True,
    )
    return indices, values, metadata


def _tensor_manifest(learner) -> dict:
    result = {}
    for name, tensor in learner.persistent_tensors().items():
        result[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "layout": str(tensor.layout),
        }
    return result


def _predict_class_ids(learner, logits: torch.Tensor) -> torch.Tensor:
    columns = logits.argmax(dim=1).detach().cpu().tolist()
    return torch.tensor([learner.class_ids[column] for column in columns])


def _evaluate_candidate(
    *, learner, method: str, train: dict, train_parts: list[torch.Tensor],
    val_parts: list[torch.Tensor], code_indices: torch.Tensor | None,
    code_values: torch.Tensor | None,
) -> dict:
    started = time.perf_counter()
    scores, stage_means = [], []
    for task, update_indices in enumerate(train_parts):
        if method in ZI_METHODS:
            learner.update_from_sparse(
                code_indices[update_indices], code_values[update_indices],
                train["labels"][update_indices],
            )
        else:
            learner.update(
                train["features"][update_indices], train["labels"][update_indices]
            )
        row = []
        for previous in range(task + 1):
            validation_indices = val_parts[previous]
            if method in ZI_METHODS:
                logits = learner.predict_logits_from_sparse(
                    code_indices[validation_indices], code_values[validation_indices]
                )
            else:
                logits = learner.predict_logits(train["features"][validation_indices])
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError(f"{method} produced non-finite logits")
            predictions = _predict_class_ids(learner, logits)
            accuracy = float(
                (predictions == train["labels"][validation_indices]).float().mean().item()
                * 100
            )
            scores.append(accuracy)
            row.append(accuracy)
        stage_means.append(sum(row) / len(row))
        print(
            f"UPDATE method={method} task={task+1}/{len(train_parts)} "
            f"stage_AA={stage_means[-1]:.4f} elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )
    if method in ZI_METHODS:
        learner.assert_exemplar_free_state()
        if learner.statistics.total_count != sum(len(part) for part in train_parts):
            raise AssertionError("ZI aggregate count does not match train-only stream")
        if "task_id" in inspect.signature(learner.predict_logits).parameters:
            raise AssertionError("ZI inference unexpectedly accepts task_id")
    return {
        "validation_average_accuracy": float(sum(scores) / len(scores)),
        "stage_average_accuracy": stage_means,
        "persistent_state_bytes": int(learner.persistent_state_bytes()),
        "persistent_tensor_manifest": _tensor_manifest(learner),
        # SFT raw Ridge predates the explicit marker but its checkpoint contract
        # is G/Q/counts only; do not mislabel that declared aggregate baseline.
        "exemplar_free": bool(
            getattr(learner, "is_exemplar_free", method == "sft_raw_ridge")
        ),
        "uses_test_set": False,
        "candidate_seconds": float(time.perf_counter() - started),
        "diagnostics": {
            key: _jsonable(value)
            for key, value in getattr(learner, "diagnostics", {}).items()
            if key in {
                "model", "method", "active_size", "seen_classes",
                "total_count", "projection", "effective_rank",
                "selected_ridge", "ridge_policy",
            }
        },
    }


def _candidate_specs(config: dict) -> list[dict]:
    specs = [
        {"method": "sft_raw_ridge", "ridge_lambda": float(config["raw_ridge_lambda"])},
        {
            "method": "cached_flycl_fidelity",
            "ridge_lower": float(config["fly_ridge_lower"]),
            "ridge_upper": float(config["fly_ridge_upper"]),
        },
        {"method": "wta_ncm", "variance_kappa": float(config["variance_kappas"][0])},
        {"method": "support_only", "variance_kappa": float(config["variance_kappas"][0])},
    ]
    for method in ("active_gaussian", "hurdle"):
        specs.extend(
            {"method": method, "variance_kappa": float(kappa)}
            for kappa in config["variance_kappas"]
        )
    return specs


def _candidate_id(spec: dict) -> str:
    suffix = "_".join(
        f"{key}-{str(value).replace('.', 'p')}" for key, value in sorted(spec.items())
        if key != "method"
    )
    return spec["method"] + ("__" + suffix if suffix else "")


def _new_candidate(config: dict, feature_dim: int, spec: dict, device):
    method = spec["method"]
    representation = config["representation"]
    if method == "sft_raw_ridge":
        return create_sft_learner(
            method="raw_ridge", feature_dim=feature_dim,
            ridge_lambda=float(spec["ridge_lambda"]), requested_rank=1,
            seed=int(config["seed"]), device=device, dtype=torch.float64,
        )
    if method == "cached_flycl_fidelity":
        return CachedFlyCLFidelity(
            feature_dim=feature_dim,
            expand_dim=int(representation["expand_dim"]),
            synaptic_degree=int(representation["synaptic_degree"]),
            coding_level=float(representation["coding_level"]),
            num_classes=int(config["num_classes"]),
            ridge_lower=float(spec["ridge_lower"]),
            ridge_upper=float(spec["ridge_upper"]), seed=int(config["seed"]),
            device=device, dtype=torch.float32,
        )
    return _new_zi(
        config, feature_dim, method, float(spec["variance_kappa"]), device
    )


def _best(results: list[dict], method: str) -> dict:
    eligible = [result for result in results if result["method"] == method]
    if not eligible:
        raise RuntimeError(f"missing required result for {method}")
    return max(eligible, key=lambda result: result["validation_average_accuracy"])


def _gate(config: dict, results: list[dict], heldout_hidden: bool) -> dict:
    thresholds = config["gate"]
    proposed, raw, fly = _best(results, "hurdle"), _best(results, "sft_raw_ridge"), _best(results, "cached_flycl_fidelity")
    support, amplitude = _best(results, "support_only"), _best(results, "active_gaussian")
    finite = all(
        all(torch.isfinite(torch.tensor(result[key])).item() for key in (
            "validation_average_accuracy", "persistent_state_bytes", "candidate_seconds"
        ))
        for result in results
    )
    state_fraction = proposed["persistent_state_bytes"] / fly["persistent_state_bytes"]
    gates = {
        "numerical_and_state_audit": bool(finite and proposed["exemplar_free"]),
        "beats_raw": bool(
            proposed["validation_average_accuracy"] - raw["validation_average_accuracy"]
            >= thresholds["minimum_gain_over_raw_pp"]
        ),
        "within_fly": bool(
            fly["validation_average_accuracy"] - proposed["validation_average_accuracy"]
            <= thresholds["maximum_gap_to_fly_pp"]
        ),
        "beats_support": bool(
            proposed["validation_average_accuracy"] - support["validation_average_accuracy"]
            >= thresholds["minimum_gain_over_component_pp"]
        ),
        "beats_active_gaussian": bool(
            proposed["validation_average_accuracy"] - amplitude["validation_average_accuracy"]
            >= thresholds["minimum_gain_over_component_pp"]
        ),
        "state_budget": bool(
            state_fraction <= thresholds["maximum_state_fraction_of_fly"]
        ),
        "heldout_test_remained_hidden": bool(heldout_hidden),
    }
    return {
        "decision": "REVIEW_FOR_HELDOUT_AUTHORIZATION" if all(gates.values()) else "STOP_TRAIN_ONLY_GATE_FAILED",
        "gates": gates,
        "diagnostics": {
            "hurdle_minus_raw_pp": proposed["validation_average_accuracy"] - raw["validation_average_accuracy"],
            "hurdle_minus_fly_pp": proposed["validation_average_accuracy"] - fly["validation_average_accuracy"],
            "hurdle_minus_support_pp": proposed["validation_average_accuracy"] - support["validation_average_accuracy"],
            "hurdle_minus_active_gaussian_pp": proposed["validation_average_accuracy"] - amplitude["validation_average_accuracy"],
            "hurdle_state_fraction_of_fly": state_fraction,
        },
        "selected_hurdle": proposed,
    }


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    code_cache_dir = Path(args.code_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    config = _read_config(config_path)
    test_path = feature_cache_dir / "test.pt"
    heldout_hidden = not test_path.exists()
    if args.require_test_hidden and not heldout_hidden:
        raise RuntimeError(
            f"held-out file is visible: {test_path}; rename it before selection"
        )
    cache_args = argparse.Namespace(
        dataset=config["dataset"], model_name=config["model_name"]
    )
    train, _, cache_metadata = validate_cache(
        feature_cache_dir, cache_args, load_test=False
    )
    if cache_metadata.get("checkpoint_sha256") != config["checkpoint_sha256"]:
        raise ValueError("feature-cache checkpoint hash does not match locked config")
    labels = train["labels"]
    if sorted(map(int, torch.unique(labels).tolist())) != list(range(config["num_classes"])):
        raise ValueError("training labels do not match locked global class IDs")
    train_path = feature_cache_dir / "train.pt"
    train_sha256 = _sha256_file(train_path)
    config_sha256 = _sha256_file(config_path)
    class_order = random.Random(config["seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    tasks = split(labels, class_order, config["num_tasks"])
    train_parts, val_parts = train_validation_indices(
        labels, tasks, config["seed"], config["validation_fraction"]
    )
    code_indices, code_values, code_metadata = _prepare_code_cache(
        train=train, train_sha256=train_sha256, cache_dir=code_cache_dir,
        config=config, device=args.device,
    )
    provenance = {
        **_git_provenance(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(args.device),
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "feature_cache_dir": str(feature_cache_dir),
        "feature_cache_metadata": cache_metadata,
        "train_pt_sha256": train_sha256,
        "code_cache_dir": str(code_cache_dir),
        "code_cache_identity_sha256": code_metadata["identity_sha256"],
        "class_order": class_order,
        "class_order_sha256": _sha256_bytes(
            ",".join(map(str, class_order)).encode("ascii")
        ),
        "training_indices_sha256": _sequence_sha256(train_parts),
        "validation_indices_sha256": _sequence_sha256(val_parts),
        "heldout_test_path_visible": not heldout_hidden,
    }
    run_identity = {
        key: provenance[key]
        for key in (
            "config_sha256", "train_pt_sha256", "code_cache_identity_sha256",
            "class_order_sha256", "training_indices_sha256",
            "validation_indices_sha256",
        )
    }
    run_identity_sha256 = _sha256_bytes(
        json.dumps(run_identity, sort_keys=True).encode("utf-8")
    )
    provenance["run_identity"] = run_identity
    provenance["run_identity_sha256"] = run_identity_sha256
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "run_provenance.json", provenance)
    results = []
    specs = _candidate_specs(config)
    for number, spec in enumerate(specs, 1):
        candidate_id = _candidate_id(spec)
        result_path = output_dir / "candidates" / f"{candidate_id}.json"
        if args.resume and result_path.is_file():
            cached = json.loads(result_path.read_text(encoding="utf-8"))
            if cached.get("run_identity_sha256") != run_identity_sha256:
                raise RuntimeError(f"resume candidate identity mismatch: {candidate_id}")
            results.append(cached)
            print(
                f"SKIP {number}/{len(specs)} method={candidate_id} val_AA="
                f"{cached['validation_average_accuracy']:.4f}", flush=True,
            )
            continue
        print(
            f"START {number}/{len(specs)} method={candidate_id}", flush=True
        )
        learner = _new_candidate(
            config, train["features"].shape[1], spec, args.device
        )
        result = {
            **spec,
            **_evaluate_candidate(
                learner=learner, method=spec["method"], train=train,
                train_parts=train_parts, val_parts=val_parts,
                code_indices=code_indices, code_values=code_values,
            ),
            "candidate_id": candidate_id,
            "config_sha256": config_sha256,
            "run_identity_sha256": run_identity_sha256,
        }
        _atomic_json(result_path, result)
        results.append(result)
        print(
            f"DONE {number}/{len(specs)} method={candidate_id} "
            f"val_AA={result['validation_average_accuracy']:.4f} "
            f"state={result['persistent_state_bytes']}B "
            f"elapsed={result['candidate_seconds']:.1f}s",
            flush=True,
        )
        del learner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    gate = _gate(config, results, heldout_hidden)
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "selection_protocol": "deterministic stratified subset of cached training features only",
        "uses_test_set": False,
        "held_out_test_authorized": False,
        "config": config,
        "run_provenance": provenance,
        "code_cache": code_metadata,
        "candidates": results,
        "best_by_method": {
            method: _best(results, method)
            for method in ("sft_raw_ridge", "cached_flycl_fidelity", *ZI_METHODS)
        },
        "gate": gate,
    }
    _atomic_json(output_dir / "selection.json", payload)
    _atomic_json(output_dir / "gate_results.json", gate)
    print(json.dumps(_jsonable(gate), indent=2), flush=True)
    print(
        "TRAIN-ONLY COMPLETE. Held-out evaluation remains unauthorized.", flush=True
    )
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--code-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-test-hidden", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
