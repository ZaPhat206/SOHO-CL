"""Run the source-locked M2 unquantized square-root equivalence audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.analytic_ridge import DenseSquareRootBackend, ExactGramBackend


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = max(float(torch.linalg.vector_norm(expected).item()), 1.0)
    return float(torch.linalg.vector_norm(actual - expected).item()) / denominator


def _case_transform(values: torch.Tensor, case: str) -> torch.Tensor:
    if case == "gaussian":
        return values
    if case == "column_scaled":
        scales = torch.logspace(
            -2,
            2,
            values.shape[1],
            device=values.device,
            dtype=values.dtype,
        )
        return values * scales
    if case == "sparse":
        indices = torch.arange(values.numel(), device=values.device).reshape_as(values)
        mask = ((indices * 17 + 11) % 7 == 0).to(values.dtype)
        return values * mask
    raise ValueError(f"unsupported case: {case}")


def _run_case(config: dict, *, case: str, dtype_name: str) -> dict:
    dtype = getattr(torch, dtype_name)
    seed_offset = {"gaussian": 0, "column_scaled": 1000, "sparse": 2000}[case]
    dtype_offset = 0 if dtype_name == "float64" else 100
    generator = torch.Generator().manual_seed(
        int(config["seed"]) + seed_offset + dtype_offset
    )
    dimension = int(config["dimension"])
    ridge_lambda = float(config["ridge_lambda"])
    common = {
        "dimension": dimension,
        "ridge_lambda": ridge_lambda,
        "statistics_dtype": dtype,
        "solver_dtype": dtype,
    }
    exact = ExactGramBackend(**common)
    dense = DenseSquareRootBackend(update_backend="dense_qr", **common)
    blocked = DenseSquareRootBackend(
        update_backend="blocked_qr",
        update_panel_size=int(config["blocked_qr"]["panel_size"]),
        update_trailing_chunk_size=int(
            config["blocked_qr"]["trailing_chunk_size"]
        ),
        **common,
    )
    probe = _case_transform(
        torch.randn(
            int(config["probe_rows"]),
            dimension,
            generator=generator,
            dtype=dtype,
        ),
        case,
    )
    task_records = []
    for task_id in range(int(config["tasks"])):
        features = _case_transform(
            torch.randn(
                int(config["rows_per_task"]),
                dimension,
                generator=generator,
                dtype=dtype,
            ),
            case,
        )
        first_class = task_id * int(config["classes_per_task"])
        labels = torch.tensor(
            [
                first_class + row % int(config["classes_per_task"])
                for row in range(len(features))
            ],
            dtype=torch.long,
        )
        for backend in (exact, dense, blocked):
            backend.update(features, labels)

        reference_system = exact.gram.clone()
        reference_system.diagonal().add_(ridge_lambda)
        reference_logits = exact.predict_logits(probe)
        reference_prediction = reference_logits.argmax(1)
        methods = {}
        for name, backend in (("dense_qr", dense), ("blocked_qr", blocked)):
            reconstructed = backend.factor.T @ backend.factor
            logits = backend.predict_logits(probe)
            methods[name] = {
                "relative_system_error": _relative_error(
                    reconstructed, reference_system
                ),
                "relative_weight_error": _relative_error(
                    backend.weights, exact.weights
                ),
                "solver_relative_residual": float(
                    backend.diagnostics["solver_relative_residual"]
                ),
                "relative_logit_error": _relative_error(logits, reference_logits),
                "prediction_agreement": float(
                    (logits.argmax(1) == reference_prediction).to(torch.float64).mean()
                ),
            }
        methods["blocked_vs_dense"] = {
            "relative_factor_error": _relative_error(blocked.factor, dense.factor),
            "relative_weight_error": _relative_error(blocked.weights, dense.weights),
        }
        task_records.append({"task_id": task_id + 1, "methods": methods})

    gate = config["gates"][dtype_name]
    maxima = {
        "relative_system_error": max(
            record["methods"][method]["relative_system_error"]
            for record in task_records
            for method in ("dense_qr", "blocked_qr")
        ),
        "relative_weight_error": max(
            record["methods"][method]["relative_weight_error"]
            for record in task_records
            for method in ("dense_qr", "blocked_qr")
        ),
        "solver_relative_residual": max(
            record["methods"][method]["solver_relative_residual"]
            for record in task_records
            for method in ("dense_qr", "blocked_qr")
        ),
        "relative_logit_error": max(
            record["methods"][method]["relative_logit_error"]
            for record in task_records
            for method in ("dense_qr", "blocked_qr")
        ),
        "blocked_vs_dense_factor_error": max(
            record["methods"]["blocked_vs_dense"]["relative_factor_error"]
            for record in task_records
        ),
    }
    minimum_prediction_agreement = min(
        record["methods"][method]["prediction_agreement"]
        for record in task_records
        for method in ("dense_qr", "blocked_qr")
    )
    checks = {
        "system_error": maxima["relative_system_error"]
        <= float(gate["maximum_relative_system_error"]),
        "weight_error": maxima["relative_weight_error"]
        <= float(gate["maximum_relative_weight_error"]),
        "solver_residual": maxima["solver_relative_residual"]
        <= float(gate["maximum_solver_relative_residual"]),
        "prediction_agreement": minimum_prediction_agreement
        >= float(gate["minimum_prediction_agreement"]),
    }
    return {
        "case": case,
        "dtype": dtype_name,
        "maxima": maxima,
        "minimum_prediction_agreement": minimum_prediction_agreement,
        "checks": checks,
        "passed": all(checks.values()),
        "tasks": task_records,
    }


def run(config_path: Path, output_path: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("uses_test_set") is not False:
        raise ValueError("M2 must be feature-free and test-free")
    results = [
        _run_case(config, case=case, dtype_name=dtype_name)
        for case in config["cases"]
        for dtype_name in config["dtypes"]
    ]
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    payload = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": "PASS_M2_UNQUANTIZED_EQUIVALENCE"
        if all(item["passed"] for item in results)
        else "FAIL_M2_UNQUANTIZED_EQUIVALENCE",
        "uses_test_set": False,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "git_commit": git_commit,
        "worktree_dirty": bool(dirty),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run(args.config, args.output)
    print(json.dumps({
        "status": payload["status"],
        "config_sha256": payload["config_sha256"],
        "cases": [
            {
                "case": item["case"],
                "dtype": item["dtype"],
                "passed": item["passed"],
                "maxima": item["maxima"],
                "minimum_prediction_agreement": item[
                    "minimum_prediction_agreement"
                ],
            }
            for item in payload["results"]
        ],
    }, indent=2))
    if payload["status"] != "PASS_M2_UNQUANTIZED_EQUIVALENCE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
