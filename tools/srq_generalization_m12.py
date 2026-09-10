"""Source-locked M12 test confirmation for the controlled RanPAC frontend."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.analytic_ridge import (  # noqa: E402
    ExactGramBackend,
    SquareRootBackend,
    persistent_tensor_bytes,
)
from models.backbone import load_model  # noqa: E402
from tools import srq_generalization_m5 as m5  # noqa: E402
from tools.experiment_runner import split, validate_cache  # noqa: E402
from utils.data_utils import load_dataset  # noqa: E402
from utils.train_utils import feature_extract, random_initialization  # noqa: E402


TOP_KEYS = {
    "schema_version", "study_id", "dataset", "model_name",
    "checkpoint_sha256", "uses_test_set", "test_tuning_allowed",
    "accuracy_based_selection", "seed", "test_use_disclosure", "num_classes",
    "protocol_recovery_disclosure",
    "num_tasks", "expected_train_samples", "expected_test_samples",
    "replicates", "widths", "methods", "excluded_development_method",
    "selected_ridge_by_width", "source_m6", "source_m11", "source_m11b",
    "ranpac", "p2b", "adaptive", "evaluation", "integrity_gates",
}
SOURCE_KEYS = {
    "artifact_filename", "artifact_sha256", "result_member", "result_sha256",
    "required_status",
}
METHODS = ("exact", "p2b_int8", "adaptive_int8_fp16")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _require_clean_git() -> None:
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if dirty:
        raise RuntimeError("M12 requires a clean committed checkout")


def _read_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(config) != TOP_KEYS or config.get("schema_version") != 1:
        raise ValueError("M12 config keys/schema mismatch")
    if not (
        config["uses_test_set"] is True
        and config["test_tuning_allowed"] is False
        and config["accuracy_based_selection"] is False
        and config["seed"] == 2025
        and config["num_classes"] == 100
        and config["num_tasks"] == 10
        and config["expected_train_samples"] == 50000
        and config["expected_test_samples"] == 10000
        and isinstance(config["protocol_recovery_disclosure"], str)
        and "state-byte gate" in config["protocol_recovery_disclosure"]
        and "no method" in config["protocol_recovery_disclosure"].lower()
        and config["widths"] == [10000, 20000]
        and tuple(config["methods"]) == METHODS
        and config["selected_ridge_by_width"]
        == {"10000": 1000000.0, "20000": 1000000.0}
    ):
        raise ValueError("M12 frozen experiment choices changed")
    pairs = [
        (item.get("class_order_seed"), item.get("projection_seed"))
        for item in config["replicates"]
    ]
    if pairs != [(3031 + i, 5031 + i) for i in range(6)]:
        raise ValueError("M12 paired replicate seeds changed")
    if config["excluded_development_method"].get("method") != "scale_refined_int8":
        raise ValueError("M12 must exclude the failed M11b method")
    for name in ("source_m6", "source_m11", "source_m11b"):
        source = config[name]
        if set(source) != SOURCE_KEYS or any(
            len(source[key]) != length
            for key, length in (("artifact_sha256", 64), ("result_sha256", 64))
        ):
            raise ValueError(f"invalid M12 source lock: {name}")
    ranpac = config["ranpac"]
    if not (
        ranpac["path"] == "phase2_no_petl_random_relu"
        and ranpac["feature_dimension"] == 768
        and ranpac["maximum_expand_dimension"] == 20000
        and ranpac["projection_distribution"] == "standard_normal"
        and ranpac["activation"] == "relu"
        and min(ranpac["encode_batch_size"], ranpac["evaluation_batch_size"]) > 0
    ):
        raise ValueError("M12 controlled RanPAC semantics changed")
    p2b = config["p2b"]
    if not (
        p2b["block_size"] == 256
        and p2b["group_size"] == 64
        and p2b["update_panel_size"] == 128
        and p2b["first_update_backend"] == "gram_cholesky"
        and p2b["quantization_backend"] == "streaming"
        and p2b["quantization_batch_blocks"] == 64
    ):
        raise ValueError("M12 P2B identity changed")
    adaptive = config["adaptive"]
    if not (
        adaptive["budget_fraction_between_int8_and_fp16"] == 0.25
        and adaptive["selection_signal"]
        == "current_factor_values_only_no_labels_or_accuracy"
    ):
        raise ValueError("M12 adaptive identity changed")
    evaluation = config["evaluation"]
    if not (
        evaluation["training_split"] == "complete_official_train_split"
        and evaluation["test_split"] == "official_test_split"
        and evaluation["allow_post_test_method_selection"] is False
        and evaluation["allow_post_test_retry"] is False
        and evaluation["confidence_intervals_in_main_table"] is False
    ):
        raise ValueError("M12 evaluation policy changed")
    gates = config["integrity_gates"]
    if not (
        gates.get("accuracy_gate", "missing") is None
        and gates.get("require_fixed_method_taskwise_state_identity_with_sources")
        is True
        and gates.get("require_adaptive_taskwise_budget_conformance") is True
    ):
        raise ValueError("M12 must not have an accuracy gate")
    if float(gates["maximum_solver_relative_residual"]) <= 0:
        raise ValueError("invalid M12 residual gate")
    return config


def _load_source(source: dict, path: Path) -> dict:
    if path.name != source["artifact_filename"]:
        raise ValueError(f"source filename mismatch: {path.name}")
    if _sha256_file(path) != source["artifact_sha256"]:
        raise ValueError(f"source artifact SHA-256 mismatch: {path.name}")
    with zipfile.ZipFile(path) as archive:
        payload = archive.read(source["result_member"])
    if hashlib.sha256(payload).hexdigest() != source["result_sha256"]:
        raise ValueError(f"embedded result SHA-256 mismatch: {path.name}")
    result = json.loads(payload)
    if result.get("status") != source["required_status"]:
        raise ValueError(f"source status mismatch: {path.name}")
    return result


def _load_sources(config: dict, paths: dict[str, Path]) -> dict:
    sources = {
        name: _load_source(config[name], paths[name])
        for name in ("source_m6", "source_m11", "source_m11b")
    }
    m6, m11, m11b = sources.values()
    if m11.get("source_m6") != config["source_m6"]:
        raise ValueError("M11 does not bind the locked M6 source")
    if m11b.get("source_m6") != config["source_m6"]:
        raise ValueError("M11b does not bind the locked M6 source")
    if m11b.get("source_m11") != config["source_m11"]:
        raise ValueError("M11b does not bind the locked M11 source")
    if m11.get("gates", {}).get("adaptive_accuracy_retention") is not True:
        raise ValueError("M11 adaptive method did not pass development")
    if m11b.get("gates", {}).get("accuracy_retention") is not False:
        raise ValueError("M11b exclusion is not supported by its development gate")
    m6_widths = {item["width"]: item for item in m6["width_results"]}
    m11_widths = {item["width"]: item for item in m11["width_results"]}
    for width in config["widths"]:
        if (
            float(m6_widths[width]["selected_ridge_lambda"])
            != float(config["selected_ridge_by_width"][str(width)])
            or float(m11_widths[width]["selected_ridge_lambda"])
            != float(config["selected_ridge_by_width"][str(width)])
        ):
            raise ValueError(f"source Ridge identity mismatch at width {width}")
    return sources


def _cache_args(config: dict):
    class Args:
        pass
    args = Args()
    args.dataset = config["dataset"]
    args.model_name = config["model_name"]
    return args


def _validate_train_cache(config: dict, cache_dir: Path) -> tuple[dict, dict]:
    if (cache_dir / "test.pt").exists():
        raise RuntimeError("test.pt is visible before M12 authorization")
    train, _, metadata = validate_cache(cache_dir, _cache_args(config), load_test=False)
    if not (
        len(train["labels"]) == config["expected_train_samples"]
        and train["features"].shape[1] == config["ranpac"]["feature_dimension"]
        and sorted(torch.unique(train["labels"]).tolist())
        == list(range(config["num_classes"]))
        and metadata.get("checkpoint_sha256") == config["checkpoint_sha256"]
        and metadata.get("test_features_materialized") is False
    ):
        raise ValueError("M12 train cache inventory/identity mismatch")
    return train, metadata


def _authorization_identity(
    config_path: Path, config: dict, train: dict, metadata: dict, sources: dict
) -> dict:
    return {
        "schema_version": 1,
        "study_id": config["study_id"],
        "git_commit": _git_commit(),
        "config_sha256": _sha256_file(config_path),
        "source_artifact_sha256": {
            name: config[name]["artifact_sha256"]
            for name in ("source_m6", "source_m11", "source_m11b")
        },
        "source_status": {name: value["status"] for name, value in sources.items()},
        "train_features_sha256": _tensor_sha256(train["features"]),
        "train_labels_sha256": _tensor_sha256(train["labels"]),
        "train_metadata_sha256": _canonical_sha256(metadata),
        "frozen_choices_sha256": _canonical_sha256(
            {
                key: config[key]
                for key in (
                    "replicates", "widths", "methods", "excluded_development_method",
                    "selected_ridge_by_width", "ranpac", "p2b", "adaptive",
                    "evaluation", "integrity_gates",
                )
            }
        ),
        "test_features_materialized_at_authorization": False,
    }


def authorize(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    if args.require_clean_git or config["integrity_gates"]["require_clean_git"]:
        _require_clean_git()
    paths = _source_paths(args)
    sources = _load_sources(config, paths)
    train, metadata = _validate_train_cache(config, Path(args.feature_cache_dir))
    identity = _authorization_identity(config_path, config, train, metadata, sources)
    payload = {
        "authorization_id": _canonical_sha256(identity),
        "authorized": True,
        "uses_test_set": False,
        "test_tuning_allowed": False,
        "identity": identity,
    }
    destination = Path(args.authorization)
    if destination.exists() and json.loads(destination.read_text()) != payload:
        raise RuntimeError("existing authorization has a different identity")
    _atomic_json(destination, payload)
    print(f"M12 AUTHORIZED: {payload['authorization_id']}", flush=True)
    return payload


def _read_authorization(args, config: dict, require_test_absent: bool) -> dict:
    path = Path(args.authorization)
    if not path.is_file():
        raise FileNotFoundError("M12 authorization is required")
    authorization = json.loads(path.read_text(encoding="utf-8"))
    cache_dir = Path(args.feature_cache_dir)
    if require_test_absent:
        train, metadata = _validate_train_cache(config, cache_dir)
    else:
        train, _, metadata = validate_cache(cache_dir, _cache_args(config), load_test=True)
    sources = _load_sources(config, _source_paths(args))
    expected = _authorization_identity(Path(args.config).resolve(), config, train, metadata, sources)
    # Test extraction changes only explicitly test-related metadata fields.
    if not require_test_absent:
        expected["train_metadata_sha256"] = authorization["identity"]["train_metadata_sha256"]
    expected_id = _canonical_sha256(expected)
    if not (
        authorization.get("authorized") is True
        and authorization.get("authorization_id") == expected_id
        and authorization.get("identity") == expected
    ):
        raise RuntimeError("M12 authorization identity mismatch")
    return authorization


def extract_test(args) -> None:
    config = _read_config(args.config)
    if args.require_clean_git or config["integrity_gates"]["require_clean_git"]:
        _require_clean_git()
    authorization = _read_authorization(args, config, require_test_absent=True)
    random_initialization(config["seed"])
    device = torch.device(args.device)
    loader_args = argparse.Namespace(
        dataset=config["dataset"], root=args.root, num_classes=config["num_classes"],
        num_tasks=config["num_tasks"], batch_size=args.batch_size,
        data_augmentation="vit", num_workers=args.num_workers,
    )
    _, test_loaders = load_dataset(loader_args)
    backbone = load_model(
        config["model_name"], checkpoint_path=args.backbone_checkpoint,
        expected_checkpoint_size=args.backbone_checkpoint_size,
        expected_checkpoint_sha256=config["checkpoint_sha256"],
    ).eval().to(device)
    feature_parts, label_parts = [], []
    for task_id in range(config["num_tasks"]):
        values, labels = feature_extract(backbone, test_loaders[task_id], device)
        feature_parts.append(values.cpu())
        label_parts.append(labels.cpu())
        print(f"TEST FEATURE TASK {task_id + 1}/{config['num_tasks']}", flush=True)
    test = {"features": torch.cat(feature_parts), "labels": torch.cat(label_parts)}
    if not (
        len(test["labels"]) == config["expected_test_samples"]
        and test["features"].shape[1] == config["ranpac"]["feature_dimension"]
        and sorted(torch.unique(test["labels"]).tolist())
        == list(range(config["num_classes"]))
        and bool(torch.isfinite(test["features"]).all())
    ):
        raise RuntimeError("M12 extracted test inventory is invalid")
    cache_dir = Path(args.feature_cache_dir)
    torch.save(test, cache_dir / "test.pt")
    metadata_path = cache_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update({
        "test_shape": list(test["features"].shape),
        "test_labels_shape": list(test["labels"].shape),
        "finite": bool(metadata.get("finite")) and bool(torch.isfinite(test["features"]).all()),
        "test_features_materialized": True,
        "m12_authorization_id": authorization["authorization_id"],
        "test_features_sha256": _tensor_sha256(test["features"]),
        "test_labels_sha256": _tensor_sha256(test["labels"]),
    })
    _atomic_json(metadata_path, metadata)
    print("M12 AUTHORIZED TEST FEATURE CACHE READY", flush=True)


def _source_paths(args) -> dict[str, Path]:
    return {
        "source_m6": Path(args.source_m6_artifact),
        "source_m11": Path(args.source_m11_artifact),
        "source_m11b": Path(args.source_m11b_artifact),
    }


def _source_widths(sources: dict) -> tuple[dict, dict]:
    return (
        {item["width"]: item for item in sources["source_m6"]["width_results"]},
        {item["width"]: item for item in sources["source_m11"]["width_results"]},
    )


def _task_state_contracts(sources: dict, width: int, method: str) -> list[dict]:
    m6_widths, m11_widths = _source_widths(sources)
    if method in {"exact", "p2b_int8"}:
        return [
            {
                "kind": "exact_source_identity",
                "source_reference_bytes": int(
                    record["state"][method]["total_persistent_bytes"]
                ),
                "minimum_bytes": int(
                    record["state"][method]["total_persistent_bytes"]
                ),
                "maximum_bytes": int(
                    record["state"][method]["total_persistent_bytes"]
                ),
            }
            for record in m6_widths[width]["records"]
        ]
    if method != "adaptive_int8_fp16":
        raise ValueError(f"unsupported state contract method: {method}")
    contracts = []
    m6_records = m6_widths[width]["records"]
    m11_records = m11_widths[width]["records"]
    if len(m6_records) != len(m11_records):
        raise ValueError(f"source task-count mismatch at width {width}")
    for m6_record, m11_record in zip(m6_records, m11_records):
        source_total = int(m11_record["total_persistent_bytes"])
        source_factor = int(m11_record["factor_persistent_bytes"])
        factor_ceiling = int(m11_record["factor_budget_ceiling_bytes"])
        total_blocks = int(m11_record["total_blocks"])
        common_bytes = source_total - source_factor
        minimum = (
            int(m6_record["state"]["p2b_int8"]["total_persistent_bytes"])
            + total_blocks
        )
        maximum = common_bytes + factor_ceiling
        if not (0 < source_factor <= factor_ceiling and minimum <= source_total <= maximum):
            raise ValueError(f"invalid adaptive source byte contract at width {width}")
        contracts.append({
            "kind": "locked_adaptive_budget_interval",
            "source_reference_bytes": source_total,
            "minimum_bytes": minimum,
            "maximum_bytes": maximum,
            "factor_budget_ceiling_bytes": factor_ceiling,
            "total_blocks": total_blocks,
        })
    return contracts


def _check_task_state_bytes(
    *, actual: int, contract: dict, method: str, width: int, task: int
) -> None:
    minimum = int(contract["minimum_bytes"])
    maximum = int(contract["maximum_bytes"])
    if not minimum <= int(actual) <= maximum:
        raise AssertionError(
            f"M12 task-wise state contract failed {method}/{width}/task {task}: "
            f"actual={actual}, permitted=[{minimum},{maximum}], "
            f"source_reference={contract['source_reference_bytes']}"
        )
    if contract["kind"] == "exact_source_identity" and minimum != maximum:
        raise AssertionError("fixed-state identity contract is not exact")


def _make_backend(config: dict, width: int, method: str, device: torch.device):
    common = m5._common_backend(
        width, float(config["selected_ridge_by_width"][str(width)]), device
    )
    if method == "exact":
        return ExactGramBackend(**common)
    p2b = config["p2b"]
    shared = {
        "block_size": int(p2b["block_size"]),
        "group_size": int(p2b["group_size"]),
        "update_panel_size": int(p2b["update_panel_size"]),
        "update_trailing_chunk_size": p2b["update_trailing_chunk_size"],
        "first_update_backend": p2b["first_update_backend"],
        "quantization_backend": p2b["quantization_backend"],
        "quantization_batch_blocks": int(p2b["quantization_batch_blocks"]),
    }
    if method == "p2b_int8":
        return SquareRootBackend(storage_mode="int8", **shared, **common)
    if method == "adaptive_int8_fp16":
        return SquareRootBackend(
            storage_mode="adaptive_int8_fp16",
            adaptive_budget_fraction=float(
                config["adaptive"]["budget_fraction_between_int8_and_fp16"]
            ),
            **shared, **common,
        )
    raise ValueError(f"unsupported M12 method: {method}")


def _unit_identity(
    authorization: dict, config: dict, replicate: dict, width: int, method: str
) -> dict:
    return {
        "authorization_id": authorization["authorization_id"],
        "study_id": config["study_id"],
        "class_order_seed": replicate["class_order_seed"],
        "projection_seed": replicate["projection_seed"],
        "width": width,
        "method": method,
        "ridge_lambda": config["selected_ridge_by_width"][str(width)],
    }


def _run_unit(
    *, config: dict, authorization: dict, sources: dict, train: dict, test: dict,
    replicate: dict, width: int, method: str, device: torch.device,
) -> dict:
    order = random.Random(replicate["class_order_seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    training_parts = split(train["labels"], order, config["num_tasks"])
    test_parts_local = split(test["labels"], order, config["num_tasks"])
    features = torch.cat((train["features"], test["features"]), dim=0)
    labels = torch.cat((train["labels"], test["labels"]), dim=0)
    test_parts = [part + len(train["labels"]) for part in test_parts_local]
    generator = torch.Generator(device="cpu").manual_seed(replicate["projection_seed"])
    projection_full = torch.randn(
        config["ranpac"]["feature_dimension"],
        config["ranpac"]["maximum_expand_dimension"],
        generator=generator, dtype=torch.float32,
    ).to(device)
    projection = projection_full[:, :width].contiguous()
    backend = _make_backend(config, width, method, device)
    encoder = lambda values: torch.relu(
        values.to(device=device, dtype=torch.float32) @ projection
    )
    contracts = _task_state_contracts(sources, width, method)
    task_records = []
    update_seconds = 0.0
    encoding_seconds = 0.0
    for index, train_indices in enumerate(training_parts):
        m5._sync(device)
        started = time.perf_counter()
        codes = m5._encode_indices(
            encoder, features, train_indices,
            int(config["ranpac"]["encode_batch_size"]),
        )
        m5._sync(device)
        encoding_seconds += time.perf_counter() - started
        m5._sync(device)
        started = time.perf_counter()
        backend.update(codes, labels[train_indices])
        m5._sync(device)
        update_seconds += time.perf_counter() - started
        seen_test = torch.cat(test_parts[: index + 1])
        accuracy = m5._evaluate(
            encoder=encoder, backends={method: backend}, features=features,
            labels=labels, indices=seen_test,
            batch_size=int(config["ranpac"]["evaluation_batch_size"]),
        )[method]
        actual = persistent_tensor_bytes(
            {"projection": projection, **backend.persistent_tensors()}
        )
        contract = contracts[index]
        _check_task_state_bytes(
            actual=actual, contract=contract, method=method,
            width=width, task=index + 1,
        )
        task_record = {
            "task": index + 1,
            "test_accuracy_percent": float(accuracy),
            "total_persistent_bytes": int(actual),
            "solver_relative_residual": float(
                backend.diagnostics["solver_relative_residual"]
            ),
            "state_contract": contract["kind"],
            "source_reference_total_persistent_bytes": int(
                contract["source_reference_bytes"]
            ),
            "minimum_permitted_total_persistent_bytes": int(
                contract["minimum_bytes"]
            ),
            "maximum_permitted_total_persistent_bytes": int(
                contract["maximum_bytes"]
            ),
        }
        if method == "adaptive_int8_fp16":
            factor_tensors = {
                name: tensor for name, tensor in backend.persistent_tensors().items()
                if name.startswith("factor.")
            }
            factor_bytes = persistent_tensor_bytes(factor_tensors)
            if not (
                factor_bytes == int(backend.diagnostics["factor_persistent_bytes"])
                and int(backend.diagnostics["factor_budget_ceiling_bytes"])
                == int(contract["factor_budget_ceiling_bytes"])
                and int(backend.diagnostics["total_blocks"])
                == int(contract["total_blocks"])
            ):
                raise AssertionError(
                    f"M12 adaptive accounting identity failed {width}/task {index + 1}"
                )
            task_record.update({
                "factor_persistent_bytes": factor_bytes,
                "factor_budget_ceiling_bytes": int(
                    backend.diagnostics["factor_budget_ceiling_bytes"]
                ),
                "selected_fp16_blocks": int(
                    backend.diagnostics["selected_fp16_blocks"]
                ),
                "total_blocks": int(backend.diagnostics["total_blocks"]),
            })
        task_records.append(task_record)
        print(
            f"TASK {index + 1}/{len(training_parts)} {method}={accuracy:.4f} "
            f"state={actual} contract=[{contract['minimum_bytes']},"
            f"{contract['maximum_bytes']}]",
            flush=True,
        )
        del codes
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    accuracies = [item["test_accuracy_percent"] for item in task_records]
    return {
        "identity": _unit_identity(authorization, config, replicate, width, method),
        "uses_test_set": True, "test_tuning_allowed": False,
        "class_order": order, "records": task_records,
        "average_incremental_accuracy_percent": sum(accuracies) / len(accuracies),
        "final_accuracy_percent": accuracies[-1],
        "final_total_persistent_bytes": task_records[-1]["total_persistent_bytes"],
        "maximum_solver_relative_residual": max(
            item["solver_relative_residual"] for item in task_records
        ),
        "representation_encoding_seconds": float(encoding_seconds),
        "analytic_update_seconds": float(update_seconds),
    }


def _validate_unit(
    unit: dict, identity: dict, sources: dict, config: dict
) -> None:
    if unit.get("identity") != identity:
        raise RuntimeError("M12 unit identity mismatch")
    if unit.get("uses_test_set") is not True or unit.get("test_tuning_allowed") is not False:
        raise RuntimeError("M12 unit test-use disclosure mismatch")
    expected_order = random.Random(identity["class_order_seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    if unit.get("class_order") != expected_order:
        raise RuntimeError("M12 unit class-order mismatch")
    records = unit.get("records", [])
    contracts = _task_state_contracts(
        sources, int(identity["width"]), str(identity["method"])
    )
    if len(records) != config["num_tasks"] or len(records) != len(contracts):
        raise RuntimeError("M12 unit task count mismatch")
    for index, (record, contract) in enumerate(zip(records, contracts)):
        values = (
            record.get("test_accuracy_percent"),
            record.get("solver_relative_residual"),
        )
        actual = record.get("total_persistent_bytes")
        if actual is not None:
            _check_task_state_bytes(
                actual=int(actual), contract=contract,
                method=str(identity["method"]), width=int(identity["width"]),
                task=index + 1,
            )
        if (
            record.get("task") != index + 1
            or actual is None
            or record.get("state_contract") != contract["kind"]
            or record.get("source_reference_total_persistent_bytes")
            != contract["source_reference_bytes"]
            or record.get("minimum_permitted_total_persistent_bytes")
            != contract["minimum_bytes"]
            or record.get("maximum_permitted_total_persistent_bytes")
            != contract["maximum_bytes"]
            or any(value is None or not math.isfinite(float(value)) for value in values)
            or not 0.0 <= float(record["test_accuracy_percent"]) <= 100.0
        ):
            raise RuntimeError(f"M12 unit task record mismatch at task {index + 1}")
    accuracies = [float(record["test_accuracy_percent"]) for record in records]
    residual = max(float(record["solver_relative_residual"]) for record in records)
    timings = (
        unit.get("representation_encoding_seconds"),
        unit.get("analytic_update_seconds"),
    )
    if not (
        all(value is not None and math.isfinite(float(value)) and float(value) >= 0 for value in timings)
        and
        math.isclose(
            float(unit.get("average_incremental_accuracy_percent", math.nan)),
            sum(accuracies) / len(accuracies), rel_tol=0.0, abs_tol=1e-12,
        )
        and math.isclose(
            float(unit.get("final_accuracy_percent", math.nan)),
            accuracies[-1], rel_tol=0.0, abs_tol=1e-12,
        )
        and unit.get("final_total_persistent_bytes")
        == records[-1]["total_persistent_bytes"]
        and math.isclose(
            float(unit.get("maximum_solver_relative_residual", math.nan)),
            residual, rel_tol=0.0, abs_tol=1e-12,
        )
    ):
        raise RuntimeError("M12 unit summary mismatch")


def _mean_sd(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def _aggregate(units: list[dict], config: dict) -> list[dict]:
    rows = []
    for width in config["widths"]:
        width_units = [u for u in units if u["identity"]["width"] == width]
        by_method = {}
        exact_by_seed = {
            u["identity"]["class_order_seed"]: u
            for u in width_units if u["identity"]["method"] == "exact"
        }
        for method in METHODS:
            values = [u for u in width_units if u["identity"]["method"] == method]
            values.sort(key=lambda u: u["identity"]["class_order_seed"])
            deltas_aia = [
                u["average_incremental_accuracy_percent"]
                - exact_by_seed[u["identity"]["class_order_seed"]]["average_incremental_accuracy_percent"]
                for u in values
            ]
            deltas_final = [
                u["final_accuracy_percent"]
                - exact_by_seed[u["identity"]["class_order_seed"]]["final_accuracy_percent"]
                for u in values
            ]
            state_values = [int(u["final_total_persistent_bytes"]) for u in values]
            by_method[method] = {
                "average_incremental_accuracy_percent": _mean_sd(
                    [u["average_incremental_accuracy_percent"] for u in values]
                ),
                "final_accuracy_percent": _mean_sd(
                    [u["final_accuracy_percent"] for u in values]
                ),
                "paired_aia_difference_from_exact_pp": _mean_sd(deltas_aia),
                "paired_final_difference_from_exact_pp": _mean_sd(deltas_final),
                "final_total_persistent_bytes": statistics.mean(state_values),
                "final_total_persistent_bytes_sample_standard_deviation": (
                    statistics.stdev(state_values) if len(state_values) > 1 else 0.0
                ),
                "minimum_final_total_persistent_bytes": min(state_values),
                "maximum_final_total_persistent_bytes": max(state_values),
                "analytic_update_seconds": _mean_sd(
                    [u["analytic_update_seconds"] for u in values]
                ),
                "representation_encoding_seconds": _mean_sd(
                    [u["representation_encoding_seconds"] for u in values]
                ),
                "maximum_solver_relative_residual": max(
                    u["maximum_solver_relative_residual"] for u in values
                ),
            }
        rows.append({"width": width, "methods": by_method})
    return rows


def _write_csv(path: Path, aggregate: list[dict]) -> None:
    rows = []
    for width_result in aggregate:
        for method, result in width_result["methods"].items():
            rows.append({
                "width": width_result["width"], "method": method,
                "aia_mean": result["average_incremental_accuracy_percent"]["mean"],
                "aia_sd": result["average_incremental_accuracy_percent"]["sample_standard_deviation"],
                "final_mean": result["final_accuracy_percent"]["mean"],
                "final_sd": result["final_accuracy_percent"]["sample_standard_deviation"],
                "paired_aia_minus_exact_mean_pp": result["paired_aia_difference_from_exact_pp"]["mean"],
                "paired_final_minus_exact_mean_pp": result["paired_final_difference_from_exact_pp"]["mean"],
                "state_bytes_mean": result["final_total_persistent_bytes"],
                "state_bytes_sample_standard_deviation": result[
                    "final_total_persistent_bytes_sample_standard_deviation"
                ],
                "state_bytes_minimum": result["minimum_final_total_persistent_bytes"],
                "state_bytes_maximum": result["maximum_final_total_persistent_bytes"],
                "update_seconds_mean": result["analytic_update_seconds"]["mean"],
                "maximum_solver_relative_residual": result["maximum_solver_relative_residual"],
            })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> dict:
    config = _read_config(args.config)
    if args.require_clean_git or config["integrity_gates"]["require_clean_git"]:
        _require_clean_git()
    sources = _load_sources(config, _source_paths(args))
    authorization = _read_authorization(args, config, require_test_absent=False)
    train, test, metadata = validate_cache(
        Path(args.feature_cache_dir), _cache_args(config), load_test=True
    )
    if not (
        len(train["labels"]) == config["expected_train_samples"]
        and len(test["labels"]) == config["expected_test_samples"]
        and metadata.get("m12_authorization_id") == authorization["authorization_id"]
        and metadata.get("test_features_sha256") == _tensor_sha256(test["features"])
        and metadata.get("test_labels_sha256") == _tensor_sha256(test["labels"])
    ):
        raise RuntimeError("M12 authorized cache identity mismatch")
    output_dir = Path(args.output_dir)
    units_dir = output_dir / "units"
    units_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    units = []
    for replicate in config["replicates"]:
        for width in config["widths"]:
            for method in METHODS:
                identity = _unit_identity(authorization, config, replicate, width, method)
                destination = units_dir / (
                    f"c{replicate['class_order_seed']}_p{replicate['projection_seed']}_"
                    f"w{width}_{method}.json"
                )
                if destination.exists():
                    unit = json.loads(destination.read_text(encoding="utf-8"))
                    if unit.get("identity") != identity:
                        raise RuntimeError(f"refusing mismatched resume unit: {destination}")
                    print(f"RESUME {destination.name}", flush=True)
                else:
                    print(f"START {destination.name}", flush=True)
                    unit = _run_unit(
                        config=config, authorization=authorization, sources=sources,
                        train=train, test=test, replicate=replicate, width=width,
                        method=method, device=device,
                    )
                    _atomic_json(destination, unit)
                    print(f"DONE {destination.name}", flush=True)
                _validate_unit(unit, identity, sources, config)
                units.append(unit)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    aggregate = _aggregate(units, config)
    expected_units = len(config["replicates"]) * len(config["widths"]) * len(METHODS)
    finite = all(
        math.isfinite(float(value))
        for unit in units
        for value in (
            unit["average_incremental_accuracy_percent"],
            unit["final_accuracy_percent"], unit["maximum_solver_relative_residual"],
        )
    )
    maximum_residual = max(u["maximum_solver_relative_residual"] for u in units)
    gates = {
        "source_artifact_identity": True,
        "prior_authorization": True,
        "sample_and_class_inventory": True,
        "all_units_complete": len(units) == expected_units,
        "fixed_method_taskwise_state_identity_with_sources": True,
        "adaptive_taskwise_budget_conformance": True,
        "finite_metrics": finite,
        "solver_residual": maximum_residual
        <= config["integrity_gates"]["maximum_solver_relative_residual"],
        "accuracy_gate": None,
    }
    passed = all(value is True for key, value in gates.items() if key != "accuracy_gate")
    payload = {
        "schema_version": 1, "study_id": config["study_id"],
        "status": "COMPLETE_M12_LOCKED_TEST_CONFIRMATION" if passed else "FAIL_M12_INTEGRITY",
        "uses_test_set": True, "test_tuning_allowed": False,
        "accuracy_based_selection": False,
        "test_use_disclosure": config["test_use_disclosure"],
        "protocol_recovery_disclosure": config["protocol_recovery_disclosure"],
        "scope": {
            "frontend": "controlled RanPAC Phase-2 random-ReLU analytic head",
            "official_ranpac_reproduction": False,
            "excluded_method": config["excluded_development_method"],
        },
        "authorization": authorization,
        "sources": {name: config[name] for name in sources},
        "aggregate": aggregate,
        "units": units,
        "summary": {
            "completed_units": len(units), "expected_units": expected_units,
            "maximum_solver_relative_residual": maximum_residual,
        },
        "gates": gates,
        "provenance": {
            "git_commit": _git_commit(),
            "config_sha256": _sha256_file(Path(args.config)),
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
            "train_pt_sha256": _sha256_file(Path(args.feature_cache_dir) / "train.pt"),
            "test_pt_sha256": _sha256_file(Path(args.feature_cache_dir) / "test.pt"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "m12_results.json", payload)
    _write_csv(output_dir / "m12_summary.csv", aggregate)
    print(json.dumps({"status": payload["status"], "summary": payload["summary"], "gates": gates}, indent=2), flush=True)
    if not passed:
        raise RuntimeError("M12 integrity gate failed; preserve the result without retry")
    return payload


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--source-m6-artifact", required=True)
    parser.add_argument("--source-m11-artifact", required=True)
    parser.add_argument("--source-m11b-artifact", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--require-clean-git", action="store_true")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    authorize_parser = commands.add_parser("authorize")
    _add_common(authorize_parser)
    extract_parser = commands.add_parser("extract-test")
    _add_common(extract_parser)
    extract_parser.add_argument("--root", required=True)
    extract_parser.add_argument("--backbone-checkpoint", required=True)
    extract_parser.add_argument("--backbone-checkpoint-size", type=int)
    extract_parser.add_argument("--device", default="cuda")
    extract_parser.add_argument("--batch-size", type=int, default=128)
    extract_parser.add_argument("--num-workers", type=int, default=2)
    run_parser = commands.add_parser("run")
    _add_common(run_parser)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "authorize":
        authorize(args)
    elif args.command == "extract-test":
        extract_test(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
