"""Fail-closed held-out evaluation for a train-only-selected Schur proposal."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.crt_soho import create_learner
from methods.crt_soho.geometry import solve_spd, symmetrize
from methods.crt_soho.statistics import DualViewStatistics
from tools.crt_gate_runner import _dtype
from tools.experiment_runner import forgetting_from_matrix, split, validate_cache


LOCK_SCHEMA_VERSION = 1
REQUIRED_GATES = {
    "gate0_numerical_stability",
    "gate1_full_residual_adds_information",
    "gate2_low_rank_approaches_full",
    "gate3_proposal_beats_controls",
}
CONTROL_METHODS = {
    "random_residual",
    "fisher_residual",
    "confusion_residual",
    "shuffled_confusion_residual",
    "confusion_no_residualization",
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _same_candidate(left: dict, right: dict) -> bool:
    return left == right


def _best(candidates: list[dict]) -> dict:
    if not candidates:
        raise ValueError("required candidate family is absent from gate report")
    return max(candidates, key=lambda item: item["validation_average_incremental_accuracy"])


def _semantic_cache_metadata(metadata: dict) -> dict:
    """Exclude provenance-only fields while preserving feature semantics."""
    return {key: value for key, value in metadata.items() if key != "git_commit"}


def authorize_gate_report(path: str | Path, expected_sha256: str) -> tuple[dict, dict]:
    """Verify immutable artifact bytes and recompute every authorization gate."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"gate result does not exist: {path}")
    if len(expected_sha256) != 64:
        raise ValueError("expected gate-results SHA-256 must contain 64 hexadecimal characters")
    try:
        int(expected_sha256, 16)
    except ValueError as error:
        raise ValueError("expected gate-results SHA-256 is not hexadecimal") from error
    actual_hash = sha256(path)
    if actual_hash.lower() != expected_sha256.lower():
        raise ValueError("gate-results SHA-256 mismatch")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("protocol") != "deterministic stratified train-only validation":
        raise ValueError("gate report protocol mismatch")
    if report.get("test_cache_opened") is not False:
        raise ValueError("gate report does not certify train-only selection")
    if report.get("status") != "all_validation_gates_passed":
        raise ValueError("gate report status does not authorize held-out evaluation")
    if report.get("held_out_test_authorized") is not True:
        raise ValueError("gate report explicitly forbids held-out evaluation")
    gates = report.get("gates", {})
    if set(gates) != REQUIRED_GATES or not all(gates[name].get("pass") is True for name in REQUIRED_GATES):
        raise ValueError("all four declared validation gates must pass")

    candidates = report.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("gate report has no candidate inventory")
    if any(candidate.get("uses_test_set") is not False for candidate in candidates):
        raise ValueError("a validation candidate reports test-set access")
    numeric_fields = (
        "validation_average_incremental_accuracy", "solver_relative_residual_max",
    )
    if any(
        not isinstance(candidate.get(field), (int, float))
        or not math.isfinite(float(candidate[field]))
        for candidate in candidates for field in numeric_fields
    ):
        raise ValueError("gate report contains a missing or non-finite candidate metric")
    by_method = {}
    for candidate in candidates:
        by_method.setdefault(candidate.get("method"), []).append(candidate)

    proposal = report.get("selected_proposal", {})
    if proposal.get("method") != "schur_residual" or not _same_candidate(proposal, _best(by_method.get("schur_residual", []))):
        raise ValueError("selected Schur proposal is not the train-validation optimum")
    if proposal.get("final_effective_rank") != proposal.get("rank"):
        raise ValueError("locked proposal requested/effective rank mismatch")
    if not 0 < int(proposal.get("rank", 0)) < int(report["source_gate_cache"]["anchor"]["raw_dim"]):
        raise ValueError("locked proposal is not strict low rank")

    raw = report.get("selected_raw_ridge", {})
    anchor = report.get("selected_anchor", {})
    full = report.get("selected_full_raw_residual", {})
    if not _same_candidate(raw, _best(by_method.get("raw_ridge", []))):
        raise ValueError("selected raw Ridge is not the validation optimum")
    if not _same_candidate(anchor, _best(by_method.get("anchor_only", []))):
        raise ValueError("selected anchor is not the validation optimum")
    if not _same_candidate(full, _best(by_method.get("full_raw_residual", []))):
        raise ValueError("selected full residual is not the validation optimum")

    selected_controls = report.get("selected_controls", [])
    if {control.get("method") for control in selected_controls} != CONTROL_METHODS:
        raise ValueError("selected control inventory mismatch")
    for control in selected_controls:
        if not _same_candidate(control, _best(by_method.get(control["method"], []))):
            raise ValueError(f"selected {control['method']} is not its validation optimum")

    thresholds = report.get("thresholds_percentage_points", {})
    observed_relative = max(candidate["solver_relative_residual_max"] for candidate in candidates)
    full_gain = full["validation_average_incremental_accuracy"] - anchor["validation_average_incremental_accuracy"]
    low_rank_gap = full["validation_average_incremental_accuracy"] - proposal["validation_average_incremental_accuracy"]
    strongest_control = _best(selected_controls)
    proposal_gain = proposal["validation_average_incremental_accuracy"] - strongest_control["validation_average_incremental_accuracy"]
    recomputed = {
        "gate0_numerical_stability": observed_relative <= report["maximum_relative_solver_residual"],
        "gate1_full_residual_adds_information": full_gain >= thresholds["minimum_full_gain"],
        "gate2_low_rank_approaches_full": low_rank_gap <= thresholds["maximum_low_rank_gap"],
        "gate3_proposal_beats_controls": proposal_gain >= thresholds["minimum_proposal_gain"],
    }
    if not all(recomputed.values()):
        raise ValueError("recomputed validation gate failed")
    if abs(gates["gate0_numerical_stability"]["maximum_observed_relative_residual"] - observed_relative) > 1e-12:
        raise ValueError("stored Gate 0 value mismatch")
    if abs(gates["gate1_full_residual_adds_information"]["gain_percentage_points"] - full_gain) > 1e-10:
        raise ValueError("stored Gate 1 value mismatch")
    if abs(gates["gate2_low_rank_approaches_full"]["gap_percentage_points"] - low_rank_gap) > 1e-10:
        raise ValueError("stored Gate 2 value mismatch")
    if abs(gates["gate3_proposal_beats_controls"]["gain_over_strongest_control_percentage_points"] - proposal_gain) > 1e-10:
        raise ValueError("stored Gate 3 value mismatch")

    lock = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "gate_results_path": str(path.resolve()),
        "gate_results_sha256": actual_hash,
        "selected_raw_ridge": raw,
        "selected_anchor": anchor,
        "selected_full_raw_residual": full,
        "selected_proposal": proposal,
        "selected_controls": selected_controls,
        "source_gate_cache": report["source_gate_cache"],
        "authorization_checks": recomputed,
    }
    return report, lock


def _model(args, candidate: dict, projection: torch.Tensor | None = None):
    return create_learner(
        method=candidate["method"],
        raw_dim=args.raw_dim,
        anchor_dim=args.anchor_dim,
        synaptic_degree=args.synaptic_degree,
        coding_level=args.coding_level,
        anchor_ridge=float(candidate["anchor_ridge"]),
        residual_ridge=float(candidate["residual_ridge"]),
        complement_ridge=float(candidate["complement_ridge"]),
        requested_rank=max(int(candidate["rank"]), 1),
        confusion_temperature=float(candidate["temperature"]),
        scatter_epsilon=args.scatter_epsilon,
        seed=args.seed,
        device=args.device,
        dtype=_dtype(args.statistics_dtype),
        anchor_projection=projection,
    )


def _metrics(matrix: list[list[float]]) -> dict:
    stage_means = [sum(row) / len(row) for row in matrix]
    return {
        "accuracy_matrix": matrix,
        "accuracy_after_each_task": stage_means,
        "final_accuracy": sum(matrix[-1]) / len(matrix[-1]),
        "average_incremental_accuracy": sum(stage_means) / len(stage_means),
        "forgetting": forgetting_from_matrix(matrix),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _raw_result(args, snapshots, test, test_indices, candidate) -> dict:
    dtype, device = _dtype(args.statistics_dtype), torch.device(args.device)
    matrix, update_seconds, inference_seconds = [], [], []
    state_bytes = 0
    for stage, snapshot in enumerate(snapshots):
        _synchronize(device)
        started = time.perf_counter()
        gram = snapshot["G_xx"].to(device=device, dtype=dtype)
        cross = snapshot["Q_x"].to(device=device, dtype=dtype)
        counts = snapshot["counts"].to(device=device, dtype=dtype)
        system = symmetrize(gram) + float(candidate["ridge_lambda"]) * torch.eye(args.raw_dim, device=device, dtype=dtype)
        weights = solve_spd(system, cross)
        _synchronize(device)
        update_seconds.append(time.perf_counter() - started)
        class_ids = [int(value) for value in snapshot["class_ids"]]
        row = []
        for task in range(stage + 1):
            started = time.perf_counter()
            logits = test["features"][test_indices[task]].to(device=device, dtype=dtype) @ weights
            predictions = torch.tensor([class_ids[index] for index in logits.argmax(1).cpu().tolist()])
            row.append(float((predictions == test["labels"][test_indices[task]]).float().mean().item() * 100))
            inference_seconds.append(time.perf_counter() - started)
        matrix.append(row)
        state_bytes = sum(tensor.numel() * tensor.element_size() for tensor in (gram, cross, counts, weights))
    return {
        "method": "raw_ridge", **_metrics(matrix),
        "persistent_state_bytes": state_bytes,
        "classifier_solve_seconds": sum(update_seconds),
        "total_inference_seconds": sum(inference_seconds),
    }


def _crt_result(args, snapshots, projection, test, test_indices, candidate) -> dict:
    model = _model(args, candidate, projection)
    device = torch.device(args.device)
    matrix, update_seconds, inference_seconds, effective_ranks = [], [], [], []
    for stage, snapshot in enumerate(snapshots):
        _synchronize(device)
        started = time.perf_counter()
        model.restore_sufficient_statistics(snapshot, projection)
        _synchronize(device)
        update_seconds.append(time.perf_counter() - started)
        effective_ranks.append(model.diagnostics.get("effective_rank"))
        row = []
        for task in range(stage + 1):
            started = time.perf_counter()
            # Include fixed-anchor encoding in method inference time. Reusing a
            # sample-level test-anchor cache here would understate deployment cost.
            logits = model.predict_logits(test["features"][test_indices[task]])
            predictions = torch.tensor([model.class_ids[index] for index in logits.argmax(1).cpu().tolist()])
            row.append(float((predictions == test["labels"][test_indices[task]]).float().mean().item() * 100))
            inference_seconds.append(time.perf_counter() - started)
        matrix.append(row)
    return {
        "method": candidate["method"],
        **_metrics(matrix),
        "requested_rank": candidate["rank"],
        "effective_rank_by_stage": effective_ranks,
        "final_effective_rank": effective_ranks[-1],
        "persistent_state_bytes": model.persistent_state_bytes(),
        "classifier_recompute_seconds": sum(update_seconds),
        "total_inference_seconds": sum(inference_seconds),
        "solver_relative_residual_max": model.diagnostics.get("solver_relative_residual_max"),
        "retained_correction_energy": model.diagnostics.get("retained_correction_energy"),
    }


def run(args) -> dict:
    # Authorization and all train-only integrity checks occur before test.pt is opened.
    report, lock = authorize_gate_report(args.gate_results, args.gate_results_sha256)
    train, _, source_metadata = validate_cache(args.feature_cache_dir, args, load_test=False)
    args.raw_dim = int(train["features"].shape[1])
    gate_manifest = report["source_gate_cache"]
    if _semantic_cache_metadata(source_metadata) != _semantic_cache_metadata(
        gate_manifest["source_cache_metadata"]
    ):
        raise ValueError("gate report/source feature metadata mismatch")
    train_path = Path(args.feature_cache_dir) / "train.pt"
    if train_path.stat().st_size != gate_manifest["source_train"]["bytes"]:
        raise ValueError("source train-cache size mismatch")
    if sha256(train_path) != gate_manifest["source_train"]["sha256"]:
        raise ValueError("source train-cache SHA-256 mismatch")
    expected_anchor = gate_manifest["anchor"]
    expected_protocol = gate_manifest["protocol"]
    for field, value in {
        "anchor_dim": expected_anchor["anchor_dim"],
        "synaptic_degree": expected_anchor["synaptic_degree"],
        "coding_level": expected_anchor["coding_level"],
        # Schema-1 artifacts created before Phase G omitted these two execution
        # values. Their Phase-F notebook used the declared defaults below.
        "scatter_epsilon": expected_anchor.get("scatter_epsilon", 1e-4),
        "seed": expected_anchor["seed"],
        "num_classes": expected_protocol["num_classes"],
        "num_tasks": expected_protocol["num_tasks"],
        "validation_fraction": expected_protocol["validation_fraction"],
        "statistics_dtype": expected_protocol["statistics_dtype"],
        "anchor_batch_size": expected_protocol.get("anchor_batch_size", 1024),
    }.items():
        if getattr(args, field) != value:
            raise ValueError(f"runtime configuration mismatch for {field}")

    # This is the only call in this module that opens test.pt.
    train, test, cache_metadata = validate_cache(args.feature_cache_dir, args, load_test=True)
    if torch.device(args.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    order = gate_manifest["class_order"]
    train_indices = split(train["labels"], order, args.num_tasks)
    test_indices = split(test["labels"], order, args.num_tasks)
    prototype = _model(args, lock["selected_anchor"])
    projection = prototype.anchor.projection_matrix.detach().cpu()
    statistics = DualViewStatistics(
        args.raw_dim, args.anchor_dim, device=args.device, dtype=_dtype(args.statistics_dtype)
    )
    snapshots = []
    stream_started = time.perf_counter()
    for task in range(args.num_tasks):
        started = time.perf_counter()
        indices = train_indices[task]
        for offset in range(0, len(indices), args.anchor_batch_size):
            batch = indices[offset:offset + args.anchor_batch_size]
            raw = train["features"][batch]
            statistics.update(raw, prototype.encode_anchor(raw), train["labels"][batch])
        snapshots.append({
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in statistics.state_dict().items()
        })
        print(
            f"full-stream task {task + 1}/{args.num_tasks}: "
            f"train={len(indices)} test={len(test_indices[task])} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    stream_seconds = time.perf_counter() - stream_started

    candidates = [
        lock["selected_anchor"],
        lock["selected_full_raw_residual"],
        lock["selected_proposal"],
        *lock["selected_controls"],
    ]
    results = [_raw_result(args, snapshots, test, test_indices, lock["selected_raw_ridge"])]
    for candidate in candidates:
        result = _crt_result(args, snapshots, projection, test, test_indices, candidate)
        results.append(result)
        print(
            f"held-out {result['method']}: final={result['final_accuracy']:.4f} "
            f"AA={result['average_incremental_accuracy']:.4f}",
            flush=True,
        )
    output = {
        "protocol": "single locked held-out evaluation after train-only authorization",
        "lock": lock,
        "feature_cache_metadata": cache_metadata,
        "class_order": order,
        "full_training_counts_by_task": [int(len(indices)) for indices in train_indices],
        "full_training_total_count": int(sum(len(indices) for indices in train_indices)),
        "full_stream_statistics_seconds": stream_seconds,
        "test_cache_opened": True,
        "hyperparameter_search_performed": False,
        "feature_cache_disk_bytes": sum(
            path.stat().st_size for path in Path(args.feature_cache_dir).iterdir() if path.is_file()
        ),
        "peak_runtime_memory_bytes": (
            int(torch.cuda.max_memory_allocated(torch.device(args.device)))
            if torch.device(args.device).type == "cuda" else None
        ),
        "results": results,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundled_gate = output_dir / "authorized_gate_results.json"
    bundled_gate.write_bytes(Path(args.gate_results).read_bytes())
    if sha256(bundled_gate) != lock["gate_results_sha256"]:
        raise RuntimeError("bundled gate artifact failed post-copy integrity check")
    lock["bundled_gate_results"] = bundled_gate.name
    dump(output_dir / "heldout_results.json", output)
    dump(output_dir / "locked_manifest.json", lock)
    return output


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-results", required=True)
    parser.add_argument("--gate-results-sha256", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="CIFAR-100")
    parser.add_argument("--model-name", default="vit_base_patch16_224")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--anchor-dim", type=int, default=1024)
    parser.add_argument("--synaptic-degree", type=int, default=300)
    parser.add_argument("--coding-level", type=float, default=0.3)
    parser.add_argument("--statistics-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--scatter-epsilon", type=float, default=1e-4)
    parser.add_argument("--anchor-batch-size", type=int, default=1024)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = run(args)
    print(json.dumps({
        "locked_gate_sha256": output["lock"]["gate_results_sha256"],
        "methods": [result["method"] for result in output["results"]],
    }, indent=2))


if __name__ == "__main__":
    main()
