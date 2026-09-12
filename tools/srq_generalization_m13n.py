"""M13-N: train-only numerical audit of the LoRanPAC task-one SVD basis.

This runner deliberately does not open the M13 archive.  The archive is used
only as an immutable, raw-byte SHA-256 identity.  All numerical measurements
are recomputed from the locked train feature cache and contain no accuracy
calculation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.frontends.loranpac import (  # noqa: E402
    maximum_loranpac_rank_for_backend_budget,
    official_loranpac_rank,
)
from tools.experiment_runner import (  # noqa: E402
    split,
    train_validation_indices,
    validate_cache,
)


TOP_KEYS = {
    "schema_version",
    "study_id",
    "dataset",
    "model_name",
    "checkpoint_sha256",
    "uses_test_set",
    "computes_accuracy",
    "uses_accuracy_for_selection",
    "seed",
    "num_classes",
    "num_tasks",
    "outer_validation_fraction",
    "statistics_dtype",
    "solver_dtype",
    "source_m13",
    "train_identity",
    "ranpac",
    "loranpac",
    "units",
    "small_oracle",
    "audit_metrics",
    "gates",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_sha256(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().to(torch.int64).contiguous()
        digest.update(len(value).to_bytes(8, "little"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _read_config(path: str | Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(config) != TOP_KEYS or config.get("schema_version") != 1:
        raise ValueError("M13-N config keys/schema mismatch")
    if (
        config["uses_test_set"] is not False
        or config["computes_accuracy"] is not False
        or config["uses_accuracy_for_selection"] is not False
        or config["seed"] != 2025
        or config["num_classes"] != 100
        or config["num_tasks"] != 10
        or config["statistics_dtype"] != "float32"
        or config["solver_dtype"] != "float32"
        or not 0 < float(config["outer_validation_fraction"]) < 1
    ):
        raise ValueError("M13-N must remain the locked, train-only numerical audit")

    source = config["source_m13"]
    if set(source) != {
        "artifact_filename",
        "artifact_sha256",
        "source_commit",
        "required_status_disclosure",
        "archive_opening_permitted",
    } or (
        source["artifact_filename"]
        != "srq_generalization_m13_loranpac_train_only.zip"
        or len(source["artifact_sha256"]) != 64
        or source["required_status_disclosure"]
        != "FAIL_M13_LORANPAC_TRAIN_ONLY"
        or source["archive_opening_permitted"] is not False
    ):
        raise ValueError("M13-N source identity or disclosure was changed")

    identity = config["train_identity"]
    if set(identity) != {
        "train_pt_sha256",
        "class_order_seed",
        "training_indices_sha256",
        "validation_indices_sha256",
    } or identity["class_order_seed"] != 2025 or any(
        len(identity[name]) != 64
        for name in (
            "train_pt_sha256",
            "training_indices_sha256",
            "validation_indices_sha256",
        )
    ):
        raise ValueError("invalid M13-N train identity")

    ranpac = config["ranpac"]
    if set(ranpac) != {
        "feature_dimension",
        "maximum_expand_dimension",
        "projection_seed",
        "projection_distribution",
        "activation",
        "encode_batch_size",
    } or (
        ranpac["feature_dimension"] != 768
        or ranpac["maximum_expand_dimension"] != 20000
        or ranpac["projection_seed"] != 2025
        or ranpac["projection_distribution"] != "standard_normal"
        or ranpac["activation"] != "relu"
        or int(ranpac["encode_batch_size"]) <= 0
    ):
        raise ValueError("invalid M13-N RanPAC lock")

    loranpac = config["loranpac"]
    if set(loranpac) != {
        "upstream_repository",
        "upstream_commit",
        "upstream_tsvd_py_sha256",
        "upstream_inc_net_py_sha256",
        "truncate_percent",
        "rank_rule",
        "official_ridge_lambda",
        "matched_ridge_lambda",
    } or (
        loranpac["upstream_commit"]
        != "32782f9d260e5d722de5702675ed66cca8234883"
        or loranpac["truncate_percent"] != 25.0
        or loranpac["rank_rule"] != "official_code_python_round"
        or loranpac["official_ridge_lambda"] != 0.0
        or loranpac["matched_ridge_lambda"] != 1_000_000.0
    ):
        raise ValueError("invalid M13-N LoRanPAC lock")

    units = config["units"]
    expected_pairs = {
        (10000, "p2b_int8"),
        (10000, "adaptive_int8_fp16"),
        (20000, "p2b_int8"),
        (20000, "adaptive_int8_fp16"),
    }
    if not isinstance(units, list) or len(units) != len(expected_pairs):
        raise ValueError("M13-N requires exactly four locked units")
    unit_keys = {
        "width",
        "budget_target",
        "maximum_rank",
        "task1_effective_rank",
        "target_total_persistent_bytes",
        "m13_task1_raw_orthogonality",
        "m13_task1_official_solver_residual",
        "m13_task1_matched_solver_residual",
    }
    if {tuple((item["width"], item["budget_target"])) for item in units} != expected_pairs:
        raise ValueError("M13-N unit identities changed")
    for unit in units:
        if set(unit) != unit_keys or min(
            int(unit["maximum_rank"]),
            int(unit["task1_effective_rank"]),
            int(unit["target_total_persistent_bytes"]),
        ) <= 0:
            raise ValueError("invalid M13-N unit")

    oracle = config["small_oracle"]
    if set(oracle) != {
        "feature_dimension",
        "rows",
        "retained_rank",
        "matrix_seed",
        "right_hand_side_columns",
        "right_hand_side_seed",
    } or not (
        0 < oracle["retained_rank"] <= min(oracle["feature_dimension"], oracle["rows"])
        and oracle["right_hand_side_columns"] > 0
    ):
        raise ValueError("invalid M13-N small oracle")

    metrics = config["audit_metrics"]
    if (
        metrics.get("accuracy_fields_permitted") is not False
        or metrics.get("spectral_norm_computation") != "symmetric_eigvalsh_exact"
    ):
        raise ValueError("M13-N metric boundary changed")
    gates = config["gates"]
    required_true = {
        "require_source_artifact_sha256",
        "require_source_archive_unopened",
        "require_train_identity",
        "require_rank_contracts",
        "require_all_metrics_finite",
    }
    if not all(gates.get(name) is True for name in required_true) or (
        gates.get("accuracy_gate", "missing") is not None
        or float(gates["minimum_qr_orthogonality_improvement_factor"]) <= 1
        or float(gates["minimum_qr_solver_residual_improvement_factor"]) <= 1
        or float(gates["maximum_small_oracle_fp64_normalized_orthogonality"]) <= 0
    ):
        raise ValueError("invalid M13-N engineering gates")
    return config


def _verify_source_artifact(config: dict, path: Path) -> dict:
    """Verify only raw bytes; never interpret or open the source container."""

    source = config["source_m13"]
    if path.name != source["artifact_filename"]:
        raise ValueError(f"M13-N source filename mismatch: {path.name}")
    digest = _sha256_file(path)
    if digest != source["artifact_sha256"]:
        raise ValueError("M13-N source artifact SHA-256 mismatch")
    return {
        "filename": path.name,
        "sha256": digest,
        "sha256_matches": True,
        "container_opened": False,
        "reported_source_status": source["required_status_disclosure"],
    }


def _orthogonality_metrics(basis: torch.Tensor) -> dict[str, float]:
    rank = int(basis.shape[1])
    identity = torch.eye(rank, device=basis.device, dtype=basis.dtype)
    error = basis.T @ basis - identity
    symmetric_error = (error + error.T) * 0.5
    eigenvalues = torch.linalg.eigvalsh(symmetric_error)
    frobenius = float(torch.linalg.vector_norm(error).item())
    return {
        "raw_frobenius": frobenius,
        "frobenius_over_sqrt_rank": frobenius / math.sqrt(rank),
        "spectral_norm": float(eigenvalues.abs().max().item()),
        "maximum_absolute_entry": float(error.abs().max().item()),
    }


def _projected_solver_residual(
    basis: torch.Tensor,
    singular_values: torch.Tensor,
    cross: torch.Tensor,
    ridge_lambda: float,
) -> float:
    denominator = singular_values.square() + float(ridge_lambda)
    if bool((denominator <= torch.finfo(denominator.dtype).tiny).any()):
        raise RuntimeError("M13-N encountered a zero retained singular direction")
    projected_cross = basis.T @ cross
    weights = basis @ (projected_cross / denominator.unsqueeze(1))
    residual = denominator.unsqueeze(1) * (basis.T @ weights) - projected_cross
    scale = max(float(torch.linalg.vector_norm(projected_cross).item()), 1.0)
    return float(torch.linalg.vector_norm(residual).item()) / scale


def _audit_basis(
    basis: torch.Tensor,
    singular_values: torch.Tensor,
    cross: torch.Tensor,
    official_ridge: float,
    matched_ridge: float,
) -> dict:
    raw_orthogonality = _orthogonality_metrics(basis)
    raw_solver = {
        "official_formula_relative_residual": _projected_solver_residual(
            basis, singular_values, cross, official_ridge
        ),
        "matched_ridge_relative_residual": _projected_solver_residual(
            basis, singular_values, cross, matched_ridge
        ),
    }
    reorthogonalized, triangular = torch.linalg.qr(basis, mode="reduced")
    del triangular
    qr_orthogonality = _orthogonality_metrics(reorthogonalized)
    qr_solver = {
        "official_formula_relative_residual": _projected_solver_residual(
            reorthogonalized, singular_values, cross, official_ridge
        ),
        "matched_ridge_relative_residual": _projected_solver_residual(
            reorthogonalized, singular_values, cross, matched_ridge
        ),
    }
    epsilon = torch.finfo(basis.dtype).eps
    orthogonality_improvement = raw_orthogonality["raw_frobenius"] / max(
        qr_orthogonality["raw_frobenius"], epsilon
    )
    raw_solver_max = max(raw_solver.values())
    qr_solver_max = max(qr_solver.values())
    solver_improvement = raw_solver_max / max(qr_solver_max, epsilon)
    return {
        "raw": {
            "orthogonality": raw_orthogonality,
            "solver": raw_solver,
        },
        "qr_reorthogonalized_diagnostic": {
            "orthogonality": qr_orthogonality,
            "solver": qr_solver,
            "changes_represented_truncated_system": True,
            "is_proposed_method_change": False,
        },
        "improvement_factor": {
            "orthogonality_raw_frobenius": orthogonality_improvement,
            "maximum_solver_relative_residual": solver_improvement,
        },
    }


def _encode_first_task(
    features: torch.Tensor,
    indices: torch.Tensor,
    projection: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    parts = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        values = features[batch_indices].to(device=device, dtype=torch.float32)
        parts.append(torch.relu(values @ projection))
    return torch.cat(parts, dim=0)


def _task_cross(codes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    class_ids = torch.unique(labels, sorted=True).to(device=codes.device)
    columns = torch.searchsorted(class_ids, labels.to(device=codes.device))
    targets = torch.nn.functional.one_hot(
        columns, num_classes=int(class_ids.numel())
    ).to(dtype=codes.dtype)
    return codes.T @ targets


def _rank_contract(config: dict, unit: dict, task1_rows: int) -> dict:
    width = int(unit["width"])
    classes = int(config["num_classes"])
    projection_bytes = 4 * int(config["ranpac"]["feature_dimension"]) * width
    backend_budget = int(unit["target_total_persistent_bytes"]) - projection_bytes
    derived_maximum_rank = maximum_loranpac_rank_for_backend_budget(
        dimension=width,
        num_classes=classes,
        target_bytes=backend_budget,
    )
    effective_rank = official_loranpac_rank(
        task1_rows,
        dimension=width,
        truncate_percent=float(config["loranpac"]["truncate_percent"]),
        max_rank=derived_maximum_rank,
    )
    return {
        "projection_bytes": projection_bytes,
        "backend_budget_bytes": backend_budget,
        "derived_maximum_rank": derived_maximum_rank,
        "derived_task1_effective_rank": effective_rank,
        "maximum_rank_matches_lock": derived_maximum_rank == int(unit["maximum_rank"]),
        "task1_effective_rank_matches_lock": effective_rank
        == int(unit["task1_effective_rank"]),
    }


def _run_small_oracle(config: dict) -> dict:
    oracle = config["small_oracle"]
    matrix_generator = torch.Generator(device="cpu").manual_seed(
        int(oracle["matrix_seed"])
    )
    rhs_generator = torch.Generator(device="cpu").manual_seed(
        int(oracle["right_hand_side_seed"])
    )
    matrix64 = torch.randn(
        int(oracle["feature_dimension"]),
        int(oracle["rows"]),
        generator=matrix_generator,
        dtype=torch.float64,
    )
    cross64 = torch.randn(
        int(oracle["feature_dimension"]),
        int(oracle["right_hand_side_columns"]),
        generator=rhs_generator,
        dtype=torch.float64,
    )
    results = {}
    for name, dtype in (("float32", torch.float32), ("float64", torch.float64)):
        matrix = matrix64.to(dtype=dtype)
        cross = cross64.to(dtype=dtype)
        left, singular_values, right = torch.linalg.svd(matrix, full_matrices=False)
        del right
        rank = int(oracle["retained_rank"])
        results[name] = _audit_basis(
            left[:, :rank].contiguous(),
            singular_values[:rank].contiguous(),
            cross,
            float(config["loranpac"]["official_ridge_lambda"]),
            float(config["loranpac"]["matched_ridge_lambda"]),
        )
    return {
        "shape": [int(oracle["feature_dimension"]), int(oracle["rows"])],
        "retained_rank": int(oracle["retained_rank"]),
        "results": results,
    }


def _all_finite(value) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _write_csv(path: Path, units: list[dict]) -> None:
    rows = []
    for unit in units:
        for variant in ("raw", "qr_reorthogonalized_diagnostic"):
            audit = unit["audit"][variant]
            rows.append(
                {
                    "width": unit["width"],
                    "budget_target": unit["budget_target"],
                    "effective_rank": unit["effective_rank"],
                    "variant": variant,
                    **audit["orthogonality"],
                    **audit["solver"],
                }
            )
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
        raise RuntimeError("M13-N requires a clean source checkout")

    cache_dir = Path(args.feature_cache_dir).resolve()
    if (cache_dir / "test.pt").exists():
        raise RuntimeError("M13-N refuses a visible test.pt")
    source_identity = _verify_source_artifact(
        config, Path(args.source_m13_artifact).resolve()
    )
    train, _, metadata = validate_cache(
        cache_dir,
        argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"]),
        load_test=False,
    )
    class_order = random.Random(config["seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    task_indices = split(train["labels"], class_order, config["num_tasks"])
    training_parts, validation_parts = train_validation_indices(
        train["labels"],
        task_indices,
        config["seed"],
        config["outer_validation_fraction"],
    )
    identity = config["train_identity"]
    train_identity = {
        "train_pt_sha256": _sha256_file(cache_dir / "train.pt"),
        "training_indices_sha256": _sequence_sha256(training_parts),
        "validation_indices_sha256": _sequence_sha256(validation_parts),
        "checkpoint_sha256": metadata.get("checkpoint_sha256"),
        "feature_dimension": int(train["features"].shape[1]),
        "class_order_seed": config["seed"],
    }
    train_identity_matches = (
        train_identity["train_pt_sha256"] == identity["train_pt_sha256"]
        and train_identity["training_indices_sha256"]
        == identity["training_indices_sha256"]
        and train_identity["validation_indices_sha256"]
        == identity["validation_indices_sha256"]
        and train_identity["checkpoint_sha256"] == config["checkpoint_sha256"]
        and train_identity["feature_dimension"]
        == config["ranpac"]["feature_dimension"]
        and sorted(map(int, torch.unique(train["labels"]).tolist()))
        == list(range(config["num_classes"]))
    )
    if not train_identity_matches:
        raise ValueError("M13-N train feature/split identity mismatch")

    device = torch.device(args.device)
    generator = torch.Generator(device="cpu").manual_seed(
        int(config["ranpac"]["projection_seed"])
    )
    full_projection = torch.randn(
        int(config["ranpac"]["feature_dimension"]),
        int(config["ranpac"]["maximum_expand_dimension"]),
        generator=generator,
        dtype=torch.float32,
    ).to(device=device)
    first_indices = training_parts[0]
    first_labels = train["labels"][first_indices]
    units = []
    rank_contracts_match = True
    for width in (10000, 20000):
        print(f"M13-N SVD START width={width}", flush=True)
        projection = full_projection[:, :width].contiguous()
        codes = _encode_first_task(
            train["features"],
            first_indices,
            projection,
            int(config["ranpac"]["encode_batch_size"]),
            device,
        )
        cross = _task_cross(codes, first_labels)
        left, singular_values, right = torch.linalg.svd(
            codes.T, full_matrices=False
        )
        del right
        width_units = [item for item in config["units"] if item["width"] == width]
        for locked in width_units:
            contract = _rank_contract(config, locked, len(first_indices))
            rank_contracts_match = rank_contracts_match and all(
                contract[name]
                for name in (
                    "maximum_rank_matches_lock",
                    "task1_effective_rank_matches_lock",
                )
            )
            rank = int(locked["task1_effective_rank"])
            audit = _audit_basis(
                left[:, :rank].contiguous(),
                singular_values[:rank].contiguous(),
                cross,
                float(config["loranpac"]["official_ridge_lambda"]),
                float(config["loranpac"]["matched_ridge_lambda"]),
            )
            raw = audit["raw"]
            reference = {
                "raw_orthogonality": float(locked["m13_task1_raw_orthogonality"]),
                "official_solver_relative_residual": float(
                    locked["m13_task1_official_solver_residual"]
                ),
                "matched_solver_relative_residual": float(
                    locked["m13_task1_matched_solver_residual"]
                ),
            }
            units.append(
                {
                    "width": width,
                    "budget_target": locked["budget_target"],
                    "effective_rank": rank,
                    "rank_contract": contract,
                    "audit": audit,
                    "m13_reference": reference,
                    "reference_ratio": {
                        "raw_orthogonality": raw["orthogonality"]["raw_frobenius"]
                        / reference["raw_orthogonality"],
                        "official_solver_relative_residual": raw["solver"][
                            "official_formula_relative_residual"
                        ]
                        / reference["official_solver_relative_residual"],
                        "matched_solver_relative_residual": raw["solver"][
                            "matched_ridge_relative_residual"
                        ]
                        / reference["matched_solver_relative_residual"],
                    },
                }
            )
            print(
                f"M13-N width={width} budget={locked['budget_target']} "
                f"rank={rank} raw_fro={raw['orthogonality']['raw_frobenius']:.6g} "
                f"normalized={raw['orthogonality']['frobenius_over_sqrt_rank']:.6g}",
                flush=True,
            )
        del codes, cross, left, singular_values, projection
        if device.type == "cuda":
            torch.cuda.empty_cache()

    small_oracle = _run_small_oracle(config)
    thresholds = config["gates"]
    gates = {
        "source_artifact_sha256": source_identity["sha256_matches"],
        "source_archive_unopened": source_identity["container_opened"] is False,
        "train_identity": train_identity_matches,
        "rank_contracts": rank_contracts_match,
        "all_metrics_finite": _all_finite(units) and _all_finite(small_oracle),
        "qr_orthogonality_improvement": min(
            unit["audit"]["improvement_factor"]["orthogonality_raw_frobenius"]
            for unit in units
        )
        >= float(thresholds["minimum_qr_orthogonality_improvement_factor"]),
        "qr_solver_residual_improvement": min(
            unit["audit"]["improvement_factor"][
                "maximum_solver_relative_residual"
            ]
            for unit in units
        )
        >= float(thresholds["minimum_qr_solver_residual_improvement_factor"]),
        "small_oracle_fp64": small_oracle["results"]["float64"]["raw"][
            "orthogonality"
        ]["frobenius_over_sqrt_rank"]
        <= float(thresholds["maximum_small_oracle_fp64_normalized_orthogonality"]),
    }
    passed = all(gates.values())
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": (
            "PASS_M13N_NUMERICAL_AUDIT"
            if passed
            else "FAIL_M13N_NUMERICAL_AUDIT"
        ),
        "uses_test_set": False,
        "computes_predictive_metrics": False,
        "scope": {
            "purpose": "diagnose M13 task-one numerical gates",
            "m13_status_remains": config["source_m13"]["required_status_disclosure"],
            "source_archive_semantics_read": False,
            "qr_reorthogonalization_is_diagnostic_only": True,
        },
        "source_identity": source_identity,
        "train_identity": train_identity,
        "units": units,
        "small_oracle": small_oracle,
        "summary": {
            "minimum_qr_orthogonality_improvement_factor": min(
                unit["audit"]["improvement_factor"][
                    "orthogonality_raw_frobenius"
                ]
                for unit in units
            ),
            "minimum_qr_solver_residual_improvement_factor": min(
                unit["audit"]["improvement_factor"][
                    "maximum_solver_relative_residual"
                ]
                for unit in units
            ),
            "maximum_raw_normalized_orthogonality": max(
                unit["audit"]["raw"]["orthogonality"][
                    "frobenius_over_sqrt_rank"
                ]
                for unit in units
            ),
            "maximum_raw_spectral_orthogonality": max(
                unit["audit"]["raw"]["orthogonality"]["spectral_norm"]
                for unit in units
            ),
        },
        "gates": gates,
        "provenance": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "git_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True
                ).strip()
            ),
            "config_sha256": _sha256_file(config_path),
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
        },
    }
    result_path = output_dir / "m13n_results.json"
    result_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_csv(output_dir / "m13n_numerical_metrics.csv", units)
    print(f"M13-N STATUS: {payload['status']}", flush=True)
    if not passed:
        raise RuntimeError("M13-N numerical engineering gates failed")
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--source-m13-artifact", required=True)
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
