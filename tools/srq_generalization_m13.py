"""M13: source-locked, train-only LoRanPAC equal-budget challenger screen."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.analytic_ridge import persistent_tensor_bytes  # noqa: E402
from methods.frontends.loranpac import (  # noqa: E402
    LoRanPACTSVBackend,
    maximum_loranpac_rank_for_backend_budget,
)
from tools import srq_generalization_m4 as m4  # noqa: E402
from tools import srq_generalization_m5 as m5  # noqa: E402
from tools import srq_generalization_m6 as m6  # noqa: E402
from tools.experiment_runner import (  # noqa: E402
    split,
    train_validation_indices,
    validate_cache,
)


TOP_KEYS = {
    "schema_version", "study_id", "dataset", "model_name",
    "checkpoint_sha256", "uses_test_set", "accuracy_based_selection", "seed",
    "num_classes", "num_tasks", "outer_validation_fraction",
    "statistics_dtype", "solver_dtype", "widths", "budget_targets",
    "source_m6", "source_m11", "ranpac", "loranpac", "gates",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(config) != TOP_KEYS or config.get("schema_version") != 1:
        raise ValueError("M13 config keys/schema mismatch")
    if config["uses_test_set"] is not False or config["accuracy_based_selection"] is not False:
        raise ValueError("M13 must remain train-only and nonselective")
    if (
        config["seed"] != 2025
        or config["num_classes"] != 100
        or config["num_tasks"] != 10
        or config["widths"] != [10000, 20000]
        or config["budget_targets"] != ["p2b_int8", "adaptive_int8_fp16"]
        or config["statistics_dtype"] != "float32"
        or config["solver_dtype"] != "float32"
        or not 0 < config["outer_validation_fraction"] < 1
    ):
        raise ValueError("M13 no longer matches the locked M6/M11 stream")
    for key, status in (
        ("source_m6", "FAIL_M6_WIDTH_SWEEP_TRAIN_ONLY"),
        ("source_m11", "PASS_M11_ADAPTIVE_PRECISION_TRAIN_ONLY"),
    ):
        source = config[key]
        if set(source) != {
            "artifact_filename", "artifact_sha256", "result_member",
            "result_sha256", "required_status",
        } or source["required_status"] != status:
            raise ValueError(f"invalid M13 {key} lock")
        if len(source["artifact_sha256"]) != 64 or len(source["result_sha256"]) != 64:
            raise ValueError(f"invalid M13 {key} hashes")
    ranpac = config["ranpac"]
    if set(ranpac) != {
        "upstream_repository", "upstream_commit", "upstream_ranpac_py_sha256",
        "path", "feature_dimension", "maximum_expand_dimension",
        "projection_distribution", "activation", "projection_seed",
        "encode_batch_size", "evaluation_batch_size",
    } or (
        ranpac["path"] != "phase2_no_petl_random_relu"
        or ranpac["feature_dimension"] != 768
        or ranpac["maximum_expand_dimension"] != 20000
        or ranpac["projection_distribution"] != "standard_normal"
        or ranpac["activation"] != "relu"
        or ranpac["projection_seed"] != 2025
        or min(ranpac["encode_batch_size"], ranpac["evaluation_batch_size"]) <= 0
    ):
        raise ValueError("invalid locked M13 RanPAC semantics")
    loranpac = config["loranpac"]
    if set(loranpac) != {
        "upstream_repository", "upstream_commit", "upstream_tsvd_py_sha256",
        "upstream_inc_net_py_sha256", "upstream_cifar_config_sha256",
        "truncate_percent", "rank_rule", "paper_rank_rule",
        "official_ridge_lambda", "matched_ridge_diagnostic",
        "rank_cap_policy", "rank_cap_selection_signal",
    } or (
        loranpac["upstream_commit"] != "32782f9d260e5d722de5702675ed66cca8234883"
        or loranpac["truncate_percent"] != 25.0
        or loranpac["rank_rule"] != "official_code_python_round"
        or loranpac["paper_rank_rule"] != "ceil"
        or loranpac["official_ridge_lambda"] != 0.0
        or loranpac["rank_cap_policy"]
        != "largest_integer_rank_not_exceeding_locked_total_state_budget"
        or not loranpac["rank_cap_selection_signal"].endswith("no_labels_or_accuracy")
    ):
        raise ValueError("M13 LoRanPAC policy is not source-locked")
    gates = config["gates"]
    if not (
        gates.get("accuracy_gate", "missing") is None
        and all(
            gates.get(name) is True
            for name in (
                "require_source_identity_match", "require_all_units_complete",
                "require_rank_caps_derived_from_bytes",
                "require_total_state_not_above_target",
                "maximum_budget_underfill_one_rank_bytes",
            )
        )
        and float(gates["maximum_solver_relative_residual"]) > 0
        and float(gates["maximum_orthogonality_residual"]) > 0
    ):
        raise ValueError("invalid M13 engineering gates")
    return config


def _load_source(source: dict, path: Path) -> dict:
    if path.name != source["artifact_filename"]:
        raise ValueError(f"M13 source filename mismatch: {path.name}")
    if _sha256_file(path) != source["artifact_sha256"]:
        raise ValueError(f"M13 source artifact SHA-256 mismatch: {path.name}")
    with zipfile.ZipFile(path) as archive:
        payload = archive.read(source["result_member"])
    if _sha256_bytes(payload) != source["result_sha256"]:
        raise ValueError(f"M13 embedded result SHA-256 mismatch: {path.name}")
    result = json.loads(payload)
    if result.get("status") != source["required_status"]:
        raise ValueError(f"M13 source status mismatch: {path.name}")
    return result


def _load_sources(config: dict, m6_path: Path, m11_path: Path) -> dict:
    sources = {
        "m6": _load_source(config["source_m6"], m6_path),
        "m11": _load_source(config["source_m11"], m11_path),
    }
    if sources["m11"].get("source_m6") != config["source_m6"]:
        raise ValueError("M11 does not bind the locked M6 artifact")
    if sources["m11"].get("gates", {}).get("adaptive_accuracy_retention") is not True:
        raise ValueError("M11 adaptive source did not pass development")
    m6_widths, m11_widths = _source_widths(sources)
    for width in config["widths"]:
        if width not in m6_widths or width not in m11_widths:
            raise ValueError(f"M13 source width missing: {width}")
        if float(m6_widths[width]["selected_ridge_lambda"]) != float(
            m11_widths[width]["selected_ridge_lambda"]
        ):
            raise ValueError(f"M13 source Ridge mismatch at width {width}")
    return sources


def _source_widths(sources: dict) -> tuple[dict[int, dict], dict[int, dict]]:
    return (
        {int(item["width"]): item for item in sources["m6"]["width_results"]},
        {int(item["width"]): item for item in sources["m11"]["width_results"]},
    )


def _target_total_bytes(sources: dict, width: int, budget: str) -> int:
    m6_widths, m11_widths = _source_widths(sources)
    if budget == "p2b_int8":
        return int(m6_widths[width]["final_total_persistent_bytes"]["p2b_int8"])
    if budget == "adaptive_int8_fp16":
        return int(m11_widths[width]["final_total_persistent_bytes"])
    raise ValueError(f"unknown M13 budget target: {budget}")


def _rank_contract(config: dict, sources: dict, width: int, budget: str) -> dict:
    target_total = _target_total_bytes(sources, width, budget)
    projection_bytes = 4 * int(config["ranpac"]["feature_dimension"]) * width
    target_backend = target_total - projection_bytes
    rank = maximum_loranpac_rank_for_backend_budget(
        dimension=width,
        num_classes=int(config["num_classes"]),
        target_bytes=target_backend,
    )
    if rank <= 0:
        raise ValueError(f"M13 {budget}/{width} cannot fit rank one")
    base = 8 * width * int(config["num_classes"]) + 4 * int(config["num_classes"])
    per_rank = 4 * (width + 1)
    expected_backend = base + rank * per_rank
    expected_total = projection_bytes + expected_backend
    return {
        "budget_target": budget,
        "target_total_persistent_bytes": target_total,
        "projection_bytes": projection_bytes,
        "target_backend_bytes": target_backend,
        "derived_max_rank": rank,
        "bytes_per_additional_rank": per_rank,
        "expected_final_backend_bytes": expected_backend,
        "expected_final_total_persistent_bytes": expected_total,
        "final_budget_underfill_bytes": target_total - expected_total,
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _evaluate_two_heads(
    *, backend: LoRanPACTSVBackend, matched_weights: torch.Tensor,
    encoder, features: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor,
    batch_size: int,
) -> dict[str, float]:
    correct = {"official_ridge0": 0, "matched_m6_ridge": 0}
    total = 0
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        codes = encoder(features[batch_indices])
        targets = labels[batch_indices].cpu()
        official = backend.predict(codes)
        matched_columns = (codes.to(matched_weights.dtype) @ matched_weights).argmax(1)
        matched = torch.tensor([backend.class_ids[index] for index in matched_columns.cpu().tolist()])
        correct["official_ridge0"] += int((official == targets).sum().item())
        correct["matched_m6_ridge"] += int((matched == targets).sum().item())
        total += len(batch_indices)
    return {name: 100.0 * value / total for name, value in correct.items()}


def _run_unit(
    *, config: dict, sources: dict, width: int, budget: str,
    projection: torch.Tensor, train: dict, training_parts: list[torch.Tensor],
    validation_parts: list[torch.Tensor], device: torch.device,
) -> dict:
    contract = _rank_contract(config, sources, width, budget)
    source_m6, _ = _source_widths(sources)
    matched_ridge = float(source_m6[width]["selected_ridge_lambda"])
    backend = LoRanPACTSVBackend(
        dimension=width,
        truncate_percent=float(config["loranpac"]["truncate_percent"]),
        max_rank=int(contract["derived_max_rank"]),
        ridge_lambda=float(config["loranpac"]["official_ridge_lambda"]),
        device=device,
        statistics_dtype=torch.float32,
        solver_dtype=torch.float32,
    )
    encoder = lambda values: torch.relu(
        values.to(device=device, dtype=torch.float32) @ projection
    )
    records = []
    encoding_seconds = 0.0
    update_seconds = 0.0
    for task_id, train_indices in enumerate(training_parts):
        _sync(device)
        started = time.perf_counter()
        codes = m5._encode_indices(
            encoder, train["features"], train_indices,
            int(config["ranpac"]["encode_batch_size"]),
        )
        _sync(device)
        encoding_seconds += time.perf_counter() - started
        _sync(device)
        started = time.perf_counter()
        backend.update(codes, train["labels"][train_indices])
        _sync(device)
        update_seconds += time.perf_counter() - started
        matched_weights, matched_residual = backend.solve_weights(matched_ridge)
        seen_validation = torch.cat(validation_parts[: task_id + 1])
        accuracy = _evaluate_two_heads(
            backend=backend, matched_weights=matched_weights, encoder=encoder,
            features=train["features"], labels=train["labels"],
            indices=seen_validation,
            batch_size=int(config["ranpac"]["evaluation_batch_size"]),
        )
        tensors = {"projection": projection}
        tensors.update(backend.persistent_tensors())
        total_bytes = persistent_tensor_bytes(tensors)
        records.append({
            "task": task_id + 1,
            "accuracy_percent": accuracy,
            "effective_rank": backend.effective_rank,
            "total_persistent_bytes": total_bytes,
            "official_solver_relative_residual": float(
                backend.diagnostics["solver_relative_residual"]
            ),
            "matched_ridge_solver_relative_residual": float(matched_residual),
            "orthogonality_residual": float(
                backend.diagnostics["orthogonality_residual"]
            ),
        })
        print(
            f"M13 width={width} budget={budget} task={task_id + 1}/"
            f"{len(training_parts)} rank={backend.effective_rank} "
            f"official={accuracy['official_ridge0']:.4f} "
            f"matched={accuracy['matched_m6_ridge']:.4f}", flush=True,
        )
        del codes, matched_weights
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    final_actual = records[-1]["total_persistent_bytes"]
    if final_actual != contract["expected_final_total_persistent_bytes"]:
        raise AssertionError(
            f"M13 symbolic/actual state mismatch: {final_actual} != "
            f"{contract['expected_final_total_persistent_bytes']}"
        )
    aia = {
        name: sum(record["accuracy_percent"][name] for record in records) / len(records)
        for name in ("official_ridge0", "matched_m6_ridge")
    }
    return {
        "width": width,
        "budget_target": budget,
        "rank_contract": contract,
        "official_ridge_lambda": 0.0,
        "matched_m6_ridge_lambda": matched_ridge,
        "records": records,
        "validation_aia_percent": aia,
        "final_validation_accuracy_percent": records[-1]["accuracy_percent"],
        "final_total_persistent_bytes": final_actual,
        "representation_encoding_seconds": encoding_seconds,
        "analytic_update_seconds": update_seconds,
        "maximum_solver_relative_residual": max(
            max(record["official_solver_relative_residual"],
                record["matched_ridge_solver_relative_residual"])
            for record in records
        ),
        "maximum_orthogonality_residual": max(
            record["orthogonality_residual"] for record in records
        ),
    }


def _source_comparison(sources: dict, width: int) -> dict:
    m6_widths, m11_widths = _source_widths(sources)
    m6_item, m11_item = m6_widths[width], m11_widths[width]
    return {
        "exact": {
            "validation_aia_percent": m6_item["validation_aia_percent"]["exact"],
            "final_validation_accuracy_percent": m6_item["final_validation_accuracy_percent"]["exact"],
            "final_total_persistent_bytes": m6_item["final_total_persistent_bytes"]["exact"],
        },
        "fp16_square_root": {
            "validation_aia_percent": m6_item["validation_aia_percent"]["fp16_square_root"],
            "final_validation_accuracy_percent": m6_item["final_validation_accuracy_percent"]["fp16_square_root"],
            "final_total_persistent_bytes": m6_item["final_total_persistent_bytes"]["fp16_square_root"],
        },
        "p2b_int8": {
            "validation_aia_percent": m6_item["validation_aia_percent"]["p2b_int8"],
            "final_validation_accuracy_percent": m6_item["final_validation_accuracy_percent"]["p2b_int8"],
            "final_total_persistent_bytes": m6_item["final_total_persistent_bytes"]["p2b_int8"],
        },
        "adaptive_int8_fp16": {
            "validation_aia_percent": m11_item["validation_aia_percent"],
            "final_validation_accuracy_percent": m11_item["final_validation_accuracy_percent"],
            "final_total_persistent_bytes": m11_item["final_total_persistent_bytes"],
        },
    }


def _write_csv(path: Path, units: list[dict], sources: dict) -> None:
    rows = []
    for width in sorted({unit["width"] for unit in units}):
        for method, item in _source_comparison(sources, width).items():
            rows.append({
                "width": width, "method": method, "budget_target": "source",
                "rank": "", "ridge_variant": "source",
                "validation_aia_percent": item["validation_aia_percent"],
                "final_validation_accuracy_percent": item["final_validation_accuracy_percent"],
                "final_total_persistent_bytes": item["final_total_persistent_bytes"],
            })
        for unit in [value for value in units if value["width"] == width]:
            for variant in ("official_ridge0", "matched_m6_ridge"):
                rows.append({
                    "width": width, "method": "loranpac_continual_tsvd",
                    "budget_target": unit["budget_target"],
                    "rank": unit["rank_contract"]["derived_max_rank"],
                    "ridge_variant": variant,
                    "validation_aia_percent": unit["validation_aia_percent"][variant],
                    "final_validation_accuracy_percent": unit["final_validation_accuracy_percent"][variant],
                    "final_total_persistent_bytes": unit["final_total_persistent_bytes"],
                })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args) -> dict:
    config_path = Path(args.config).resolve()
    config = _read_config(config_path)
    if args.require_clean_git and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("M13 requires a clean source checkout")
    cache_dir = Path(args.feature_cache_dir).resolve()
    if (cache_dir / "test.pt").exists():
        raise RuntimeError("M13 refuses a visible test.pt")
    sources = _load_sources(
        config, Path(args.source_m6_artifact), Path(args.source_m11_artifact)
    )
    train, _, metadata = validate_cache(
        cache_dir,
        argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"]),
        load_test=False,
    )
    if (
        metadata.get("checkpoint_sha256") != config["checkpoint_sha256"]
        or train["features"].shape[1] != config["ranpac"]["feature_dimension"]
        or sorted(map(int, torch.unique(train["labels"]).tolist()))
        != list(range(config["num_classes"]))
    ):
        raise ValueError("M13 feature cache identity mismatch")
    class_order = random.Random(config["seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    task_indices = split(train["labels"], class_order, config["num_tasks"])
    training_parts, validation_parts = train_validation_indices(
        train["labels"], task_indices, config["seed"],
        config["outer_validation_fraction"],
    )
    source_provenance = sources["m6"]["provenance"]
    identity_checks = {
        "train_cache": _sha256_file(cache_dir / "train.pt")
        == source_provenance["train_sha256"],
        "class_order": class_order == source_provenance["class_order"],
        "training_indices": m6._sequence_sha256(training_parts)
        == source_provenance["training_indices_sha256"],
        "validation_indices": m6._sequence_sha256(validation_parts)
        == source_provenance["outer_validation_indices_sha256"],
    }
    device = torch.device(args.device)
    generator = torch.Generator(device="cpu").manual_seed(
        int(config["ranpac"]["projection_seed"])
    )
    full_projection = torch.randn(
        config["ranpac"]["feature_dimension"],
        config["ranpac"]["maximum_expand_dimension"],
        generator=generator, dtype=torch.float32,
    ).to(device)
    identity_checks["full_projection"] = (
        m4._tensor_content_sha256(full_projection)
        == source_provenance["full_projection_sha256"]
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    units = []
    for width in config["widths"]:
        projection = full_projection[:, :width].contiguous()
        identity_checks[f"projection_{width}"] = (
            m4._tensor_content_sha256(projection)
            == source_provenance["projection_prefix_sha256"][str(width)]
        )
        for budget in config["budget_targets"]:
            unit_path = output_dir / f"m13_unit_w{width}_{budget}.json"
            contract = _rank_contract(config, sources, width, budget)
            if unit_path.is_file():
                unit = json.loads(unit_path.read_text(encoding="utf-8"))
                if not (
                    unit.get("width") == width
                    and unit.get("budget_target") == budget
                    and unit.get("rank_contract") == contract
                    and len(unit.get("records", [])) == config["num_tasks"]
                    and unit.get("final_total_persistent_bytes")
                    == contract["expected_final_total_persistent_bytes"]
                ):
                    raise RuntimeError(f"invalid cached M13 unit: {unit_path.name}")
                units.append(unit)
                print(f"M13 UNIT REUSED width={width} budget={budget}", flush=True)
                continue
            print(f"M13 UNIT START width={width} budget={budget}", flush=True)
            unit = _run_unit(
                config=config, sources=sources, width=width, budget=budget,
                projection=projection, train=train,
                training_parts=training_parts, validation_parts=validation_parts,
                device=device,
            )
            unit_path.write_text(json.dumps(unit, indent=2) + "\n", encoding="utf-8")
            units.append(unit)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    thresholds = config["gates"]
    gates = {
        "source_identity_match": all(identity_checks.values()),
        "all_units_complete": len(units)
        == len(config["widths"]) * len(config["budget_targets"]),
        "rank_caps_derived_from_bytes": all(
            unit["rank_contract"] == _rank_contract(
                config, sources, unit["width"], unit["budget_target"]
            ) for unit in units
        ),
        "total_state_not_above_target": all(
            unit["final_total_persistent_bytes"]
            <= unit["rank_contract"]["target_total_persistent_bytes"]
            for unit in units
        ),
        "budget_underfill_below_one_rank": all(
            0 <= unit["rank_contract"]["final_budget_underfill_bytes"]
            < unit["rank_contract"]["bytes_per_additional_rank"]
            for unit in units
        ),
        "solver_residual": max(
            unit["maximum_solver_relative_residual"] for unit in units
        ) <= float(thresholds["maximum_solver_relative_residual"]),
        "orthogonality": max(
            unit["maximum_orthogonality_residual"] for unit in units
        ) <= float(thresholds["maximum_orthogonality_residual"]),
    }
    passed = all(gates.values())
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "PASS_M13_LORANPAC_TRAIN_ONLY" if passed else "FAIL_M13_LORANPAC_TRAIN_ONLY",
        "uses_test_set": False,
        "accuracy_based_selection": False,
        "scope": {
            "purpose": "equal-byte LoRanPAC challenger screen",
            "frontend": "M6-locked RanPAC Phase-2 random-ReLU",
            "loranpac_update": "released continual TSVD recurrence",
            "official_end_to_end_loranpac_reproduction": False,
            "petl_reproduced": False,
            "accuracy_gate": None,
        },
        "source_m6": config["source_m6"],
        "source_m11": config["source_m11"],
        "identity_checks": identity_checks,
        "source_comparisons": {
            str(width): _source_comparison(sources, width) for width in config["widths"]
        },
        "units": units,
        "summary": {
            "derived_rank_caps": {
                f"{unit['width']}/{unit['budget_target']}":
                unit["rank_contract"]["derived_max_rank"] for unit in units
            },
            "maximum_solver_relative_residual": max(
                unit["maximum_solver_relative_residual"] for unit in units
            ),
            "maximum_orthogonality_residual": max(
                unit["maximum_orthogonality_residual"] for unit in units
            ),
        },
        "gates": gates,
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "git_dirty": bool(subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()),
            "config_sha256": _sha256_file(config_path),
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
            "loranpac_backend_sha256": _sha256_file(
                ROOT / "methods/frontends/loranpac.py"
            ),
            "train_sha256": _sha256_file(cache_dir / "train.pt"),
            "class_order": class_order,
            "training_indices_sha256": m6._sequence_sha256(training_parts),
            "validation_indices_sha256": m6._sequence_sha256(validation_parts),
        },
    }
    result_path = output_dir / "m13_results.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_dir / "m13_accuracy_state.csv", units, sources)
    print(f"M13 STATUS: {payload['status']}", flush=True)
    if not passed:
        raise RuntimeError("M13 engineering gates failed")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--source-m6-artifact", required=True)
    run_parser.add_argument("--source-m11-artifact", required=True)
    run_parser.add_argument("--feature-cache-dir", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--device", default="cuda")
    run_parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.command == "run":
        run(args)


if __name__ == "__main__":
    main()
