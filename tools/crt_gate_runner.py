"""Efficient train-only falsification gates for CRT-SOHO.

This tool never opens the held-out ``test.pt`` feature cache. It materializes
fixed-anchor validation features and cumulative sufficient-statistic snapshots
as experiment infrastructure, then reuses them across analytic candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
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
from tools.experiment_runner import split, train_validation_indices, validate_cache


SCHEMA_VERSION = 1


def _dump(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, root: Path) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def _cpu_statistics(state: dict) -> dict:
    return {
        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
        for key, value in state.items()
    }


def _learner(args, method: str, anchor_ridge: float, residual_ridge: float,
             complement_ridge: float, rank: int, temperature: float,
             projection: torch.Tensor | None = None):
    return create_learner(
        method=method,
        raw_dim=args.raw_dim,
        anchor_dim=args.anchor_dim,
        synaptic_degree=args.synaptic_degree,
        coding_level=args.coding_level,
        anchor_ridge=anchor_ridge,
        residual_ridge=residual_ridge,
        complement_ridge=complement_ridge,
        requested_rank=max(rank, 1),
        confusion_temperature=temperature,
        scatter_epsilon=args.scatter_epsilon,
        seed=args.seed,
        device=args.device,
        dtype=_dtype(args.statistics_dtype),
        anchor_projection=projection,
    )


def _encode_in_batches(model, features: torch.Tensor, indices: torch.Tensor, batch_size: int) -> torch.Tensor:
    encoded = []
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        encoded.append(model.encode_anchor(features[batch_indices]).detach().cpu().to(torch.float32))
    return torch.cat(encoded) if encoded else torch.empty((0, model.anchor_dim), dtype=torch.float32)


def _expected_cache_identity(args, train_path: Path, source_metadata: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "crt_train_only_gate_cache",
        "dataset": args.dataset,
        "backbone_model": args.model_name,
        "source_cache_metadata": source_metadata,
        "source_train": {
            "bytes": train_path.stat().st_size,
            "sha256": _sha256(train_path),
        },
        "anchor": {
            "raw_dim": args.raw_dim,
            "anchor_dim": args.anchor_dim,
            "synaptic_degree": args.synaptic_degree,
            "coding_level": args.coding_level,
            "seed": args.seed,
        },
        "protocol": {
            "num_classes": args.num_classes,
            "num_tasks": args.num_tasks,
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
            "statistics_dtype": args.statistics_dtype,
        },
    }


def prepare_cache(args) -> dict:
    """Build anchor views and cumulative train-only statistics exactly once."""
    train, _, source_metadata = validate_cache(args.feature_cache_dir, args, load_test=False)
    args.raw_dim = int(train["features"].shape[1])
    source_train = Path(args.feature_cache_dir) / "train.pt"
    expected = _expected_cache_identity(args, source_train, source_metadata)
    cache_dir = Path(args.gate_cache_dir)
    manifest_path = cache_dir / "metadata.json"
    if manifest_path.is_file():
        manifest = validate_gate_cache(args, train, source_metadata)
        print(f"Using validated CRT gate cache: {cache_dir}", flush=True)
        return manifest
    cache_dir.mkdir(parents=True, exist_ok=True)

    order = random.Random(args.seed).sample(list(range(args.num_classes)), args.num_classes)
    task_indices = split(train["labels"], order, args.num_tasks)
    training, validation = train_validation_indices(
        train["labels"], task_indices, args.seed, args.validation_fraction
    )
    prototype = _learner(args, "anchor_only", 1.0, 1.0, 1.0, 1, 1.0)
    projection_path = cache_dir / "anchor_projection.pt"
    torch.save(prototype.anchor.projection_matrix.detach().cpu(), projection_path)

    statistics = DualViewStatistics(
        args.raw_dim, args.anchor_dim, device=args.device, dtype=_dtype(args.statistics_dtype)
    )
    files = [_file_record(projection_path, cache_dir)]
    started = time.perf_counter()
    for task in range(args.num_tasks):
        task_started = time.perf_counter()
        train_indices = training[task]
        for start in range(0, len(train_indices), args.anchor_batch_size):
            indices = train_indices[start:start + args.anchor_batch_size]
            raw = train["features"][indices]
            phi = prototype.encode_anchor(raw)
            statistics.update(raw, phi, train["labels"][indices])

        snapshot_path = cache_dir / f"statistics_task_{task:02d}.pt"
        torch.save(_cpu_statistics(statistics.state_dict()), snapshot_path)
        files.append(_file_record(snapshot_path, cache_dir))

        validation_path = cache_dir / f"validation_task_{task:02d}.pt"
        validation_payload = {
            "indices": validation[task].cpu(),
            "anchor_features": _encode_in_batches(
                prototype, train["features"], validation[task], args.anchor_batch_size
            ),
        }
        torch.save(validation_payload, validation_path)
        files.append(_file_record(validation_path, cache_dir))
        print(
            f"cache task {task + 1}/{args.num_tasks}: "
            f"train={len(train_indices)} val={len(validation[task])} "
            f"elapsed={time.perf_counter() - task_started:.1f}s",
            flush=True,
        )

    manifest = {
        **expected,
        "class_order": order,
        "training_counts": [int(len(indices)) for indices in training],
        "validation_counts": [int(len(indices)) for indices in validation],
        "files": files,
        "experiment_cache_only": True,
        "allowed_in_learner_checkpoint": False,
        "test_cache_opened": False,
        "preparation_seconds": time.perf_counter() - started,
    }
    _dump(manifest_path, manifest)  # Completion marker is deliberately written last.
    print(f"CRT gate cache prepared in {manifest['preparation_seconds']:.1f}s", flush=True)
    return manifest


def validate_gate_cache(args, train: dict | None = None, source_metadata: dict | None = None) -> dict:
    cache_dir = Path(args.gate_cache_dir)
    manifest_path = cache_dir / "metadata.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing CRT gate cache manifest: {manifest_path}")
    if train is None or source_metadata is None:
        train, _, source_metadata = validate_cache(args.feature_cache_dir, args, load_test=False)
    args.raw_dim = int(train["features"].shape[1])
    expected = _expected_cache_identity(
        args, Path(args.feature_cache_dir) / "train.pt", source_metadata
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"CRT gate cache mismatch for {key}")
    if manifest.get("test_cache_opened") is not False:
        raise ValueError("gate cache does not certify train-only construction")
    expected_names = {"anchor_projection.pt"}
    expected_names |= {f"statistics_task_{task:02d}.pt" for task in range(args.num_tasks)}
    expected_names |= {f"validation_task_{task:02d}.pt" for task in range(args.num_tasks)}
    records = {record["path"]: record for record in manifest.get("files", [])}
    if set(records) != expected_names:
        raise ValueError("CRT gate cache file inventory mismatch")
    for name, record in records.items():
        path = cache_dir / name
        if not path.is_file() or path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError(f"CRT gate cache integrity failure: {name}")
    return manifest


def _csv_values(raw: str, cast=float) -> list:
    values = [cast(value) for value in raw.split(",") if value.strip()]
    if not values:
        raise ValueError("search grid cannot be empty")
    return values


def _evaluate_raw_ridge(args, train: dict, validation: list[dict], snapshots: list[dict], ridge: float) -> dict:
    started = time.perf_counter()
    dtype = _dtype(args.statistics_dtype)
    device = torch.device(args.device)
    matrix = []
    relative_residuals = []
    final_state_bytes = 0
    for stage, snapshot in enumerate(snapshots):
        gram = snapshot["G_xx"].to(device=device, dtype=dtype)
        cross = snapshot["Q_x"].to(device=device, dtype=dtype)
        counts = snapshot["counts"].to(device=device, dtype=dtype)
        system = symmetrize(gram) + ridge * torch.eye(args.raw_dim, device=device, dtype=dtype)
        weights = solve_spd(system, cross)
        residual_max = float((system @ weights - cross).abs().max().item())
        relative_residuals.append(residual_max / max(float(cross.abs().max().item()), 1.0))
        class_ids = [int(class_id) for class_id in snapshot["class_ids"]]
        row = []
        for task in range(stage + 1):
            indices = validation[task]["indices"]
            logits = train["features"][indices].to(device=device, dtype=dtype) @ weights
            predictions = torch.tensor(
                [class_ids[column] for column in logits.argmax(1).detach().cpu().tolist()]
            )
            row.append(float((predictions == train["labels"][indices]).float().mean().item() * 100))
        matrix.append(row)
        final_state_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (gram, cross, counts, weights)
        )
    stage_means = [sum(row) / len(row) for row in matrix]
    result = {
        "method": "raw_ridge",
        "ridge_lambda": ridge,
        "rank": 0,
        "requested_rank": 0,
        "final_effective_rank": None,
        "validation_average_incremental_accuracy": sum(stage_means) / len(stage_means),
        "validation_final_accuracy": sum(matrix[-1]) / len(matrix[-1]),
        "accuracy_matrix": matrix,
        "persistent_state_bytes": final_state_bytes,
        "solver_relative_residual_max": max(relative_residuals),
        "seconds": time.perf_counter() - started,
        "uses_test_set": False,
    }
    print(
        f"candidate raw_ridge lambda={ridge} -> "
        f"val_AA={result['validation_average_incremental_accuracy']:.4f} "
        f"({result['seconds']:.1f}s)",
        flush=True,
    )
    return result


def _evaluate_candidate(args, train: dict, projection: torch.Tensor, validation: list[dict],
                        snapshots: list[dict], candidate: dict) -> dict:
    started = time.perf_counter()
    model = _learner(
        args,
        candidate["method"],
        candidate["anchor_ridge"],
        candidate["residual_ridge"],
        candidate["complement_ridge"],
        candidate["rank"],
        candidate["temperature"],
        projection,
    )
    matrix = []
    solver_residuals = []
    solver_relative_residuals = []
    effective_ranks = []
    for stage, snapshot in enumerate(snapshots):
        model.restore_sufficient_statistics(snapshot, projection)
        row = []
        for task in range(stage + 1):
            indices = validation[task]["indices"]
            logits = model.predict_logits_from_views(
                train["features"][indices], validation[task]["anchor_features"]
            )
            predictions = torch.tensor(
                [model.class_ids[column] for column in logits.argmax(1).detach().cpu().tolist()]
            )
            accuracy = float((predictions == train["labels"][indices]).float().mean().item() * 100)
            row.append(accuracy)
        matrix.append(row)
        solver_residuals.append(model.diagnostics.get("solver_residual_max"))
        solver_relative_residuals.append(model.diagnostics.get("solver_relative_residual_max"))
        effective_ranks.append(model.diagnostics.get("effective_rank"))
    stage_means = [sum(row) / len(row) for row in matrix]
    result = {
        **candidate,
        "requested_rank": candidate["rank"],
        "effective_rank_by_stage": effective_ranks,
        "final_effective_rank": effective_ranks[-1],
        "validation_average_incremental_accuracy": sum(stage_means) / len(stage_means),
        "validation_final_accuracy": sum(matrix[-1]) / len(matrix[-1]),
        "accuracy_matrix": matrix,
        "persistent_state_bytes": model.persistent_state_bytes(),
        "solver_residual_max": max(value for value in solver_residuals if value is not None),
        "solver_relative_residual_max": max(
            value for value in solver_relative_residuals if value is not None
        ),
        "seconds": time.perf_counter() - started,
        "uses_test_set": False,
        "geometry": model.diagnostics.get("geometry"),
        "retained_correction_energy": model.diagnostics.get("retained_correction_energy"),
        "affinity_edge_cv": model.diagnostics.get("affinity_edge_cv"),
        "affinity_normalized_entropy": model.diagnostics.get("affinity_normalized_entropy"),
    }
    print(
        f"candidate {candidate['method']} rank={candidate['rank']} "
        f"lp={candidate['anchor_ridge']} lr={candidate['residual_ridge']} "
        f"eta={candidate['complement_ridge']} tau={candidate['temperature']} -> "
        f"val_AA={result['validation_average_incremental_accuracy']:.4f} "
        f"({result['seconds']:.1f}s)",
        flush=True,
    )
    return result


def _best(results: list[dict]) -> dict:
    return max(results, key=lambda result: result["validation_average_incremental_accuracy"])


def _candidate(method: str, anchor_ridge: float, residual_ridge: float = 1.0,
               complement_ridge: float = 1.0, rank: int = 1,
               temperature: float = 1.0) -> dict:
    return {
        "method": method,
        "anchor_ridge": anchor_ridge,
        "residual_ridge": residual_ridge,
        "complement_ridge": complement_ridge,
        "rank": rank,
        "temperature": temperature,
    }


def _final_subspace_diagnostic(args, projection, snapshot, proposal: dict, controls: list[dict]) -> dict:
    def restored(candidate):
        model = _learner(
            args, candidate["method"], candidate["anchor_ridge"],
            candidate["residual_ridge"], candidate["complement_ridge"],
            candidate["rank"], candidate["temperature"], projection,
        )
        model.restore_sufficient_statistics(snapshot, projection)
        return model

    proposed_model = restored(proposal)
    result = {
        "proposal_method": proposal["method"],
        "proposal_effective_rank": proposed_model.diagnostics.get("effective_rank"),
        "comparisons": [],
    }
    proposed_basis = proposed_model.directions
    for candidate in controls:
        control_model = restored(candidate)
        control_basis = control_model.directions
        if proposed_basis is None or control_basis is None:
            continue
        overlap = torch.linalg.svdvals(proposed_basis.T @ control_basis).clamp(0, 1)
        angles = torch.rad2deg(torch.acos(overlap))
        result["comparisons"].append({
            "control_method": candidate["method"],
            "control_effective_rank": control_model.diagnostics.get("effective_rank"),
            "principal_angle_mean_degrees": float(angles.mean().item()),
            "principal_angle_max_degrees": float(angles.max().item()),
        })
    return result


def run_gates(args) -> dict:
    """Run staged validation gates and stop before any held-out test access."""
    train, _, source_metadata = validate_cache(args.feature_cache_dir, args, load_test=False)
    args.raw_dim = int(train["features"].shape[1])
    manifest = validate_gate_cache(args, train, source_metadata)
    cache_dir = Path(args.gate_cache_dir)
    projection = torch.load(cache_dir / "anchor_projection.pt", weights_only=True)
    snapshots = [
        torch.load(cache_dir / f"statistics_task_{task:02d}.pt", weights_only=True)
        for task in range(args.num_tasks)
    ]
    validation = [
        torch.load(cache_dir / f"validation_task_{task:02d}.pt", weights_only=True)
        for task in range(args.num_tasks)
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    all_results = []
    minimum_proposal_gain = float(
        getattr(args, "minimum_proposal_gain", getattr(args, "minimum_confusion_gain", 0.1))
    )

    raw_results = [
        _evaluate_raw_ridge(args, train, validation, snapshots, ridge)
        for ridge in _csv_values(getattr(args, "raw_ridges", args.anchor_ridges))
    ]
    all_results.extend(raw_results)
    raw_best = _best(raw_results)

    anchor_results = [
        _evaluate_candidate(args, train, projection, validation, snapshots,
                            _candidate("anchor_only", ridge))
        for ridge in _csv_values(args.anchor_ridges)
    ]
    all_results.extend(anchor_results)
    anchor_best = _best(anchor_results)
    locked_anchor_ridge = anchor_best["anchor_ridge"]

    full_results = []
    for residual_ridge in _csv_values(args.residual_ridges):
        for complement_ridge in _csv_values(args.complement_ridges):
            full_results.append(_evaluate_candidate(
                args, train, projection, validation, snapshots,
                _candidate(
                    "full_raw_residual", locked_anchor_ridge, residual_ridge,
                    complement_ridge, args.raw_dim,
                ),
            ))
    all_results.extend(full_results)
    full_best = _best(full_results)
    gate1_gain = (
        full_best["validation_average_incremental_accuracy"]
        - anchor_best["validation_average_incremental_accuracy"]
    )
    gate1 = gate1_gain >= args.minimum_full_gain
    report = {
        "protocol": "deterministic stratified train-only validation",
        "test_cache_opened": False,
        "source_gate_cache": manifest,
        "thresholds_percentage_points": {
            "minimum_full_gain": args.minimum_full_gain,
            "maximum_low_rank_gap": args.maximum_low_rank_gap,
            "minimum_proposal_gain": minimum_proposal_gain,
        },
        "maximum_relative_solver_residual": args.maximum_relative_solver_residual,
        "selected_raw_ridge": raw_best,
        "selected_anchor": anchor_best,
        "selected_full_raw_residual": full_best,
        "gates": {
            "gate1_full_residual_adds_information": {
                "pass": gate1,
                "gain_percentage_points": gate1_gain,
            }
        },
        "candidates": all_results,
    }
    initial_relative_residual = max(
        candidate["solver_relative_residual_max"] for candidate in all_results
    )
    numerical_gate = initial_relative_residual <= args.maximum_relative_solver_residual
    report["gates"]["gate0_numerical_stability"] = {
        "pass": numerical_gate,
        "maximum_observed_relative_residual": initial_relative_residual,
    }
    if not numerical_gate or not gate1:
        stop_status = "stopped_after_numerical_gate" if not numerical_gate else "stopped_after_gate1"
        report.update(status=stop_status, held_out_test_authorized=False)
        report["total_seconds"] = time.perf_counter() - started
        _dump(output_dir / "gate_results.json", report)
        print("STOP: numerical or Gate 1 check failed. Held-out test remains forbidden.", flush=True)
        return report

    proposal_method = getattr(args, "proposal_method", "confusion_residual")
    if proposal_method not in {"confusion_residual", "schur_residual"}:
        raise ValueError("proposal_method must be confusion_residual or schur_residual")
    ranks = _csv_values(args.ranks, int)
    temperatures = _csv_values(args.temperatures)
    proposed_results = []
    for rank in ranks:
        for temperature in (temperatures if proposal_method == "confusion_residual" else [1.0]):
            proposed_results.append(_evaluate_candidate(
                args, train, projection, validation, snapshots,
                _candidate(
                    proposal_method, locked_anchor_ridge,
                    full_best["residual_ridge"], full_best["complement_ridge"],
                    rank, temperature,
                ),
            ))
    all_results.extend(proposed_results)
    proposed_best = _best(proposed_results)
    control_results = []
    for method in ("random_residual", "fisher_residual"):
        for rank in ranks:
            control_results.append(_evaluate_candidate(
                args, train, projection, validation, snapshots,
                _candidate(
                    method, locked_anchor_ridge, full_best["residual_ridge"],
                    full_best["complement_ridge"], rank, 1.0,
                ),
            ))
    confusion_controls = ["shuffled_confusion_residual", "confusion_no_residualization"]
    if proposal_method != "confusion_residual":
        confusion_controls.insert(0, "confusion_residual")
    for method in confusion_controls:
        for rank in ranks:
            for temperature in temperatures:
                control_results.append(_evaluate_candidate(
                    args, train, projection, validation, snapshots,
                    _candidate(
                        method, locked_anchor_ridge, full_best["residual_ridge"],
                        full_best["complement_ridge"], rank, temperature,
                    ),
                ))
    all_results.extend(control_results)
    low_rank_gap = (
        full_best["validation_average_incremental_accuracy"]
        - proposed_best["validation_average_incremental_accuracy"]
    )
    selected_controls = [
        _best([result for result in control_results if result["method"] == method])
        for method in ("random_residual", "fisher_residual", *confusion_controls)
    ]
    strongest_control = _best(selected_controls)
    confusion_gain = (
        proposed_best["validation_average_incremental_accuracy"]
        - strongest_control["validation_average_incremental_accuracy"]
    )
    gate2 = low_rank_gap <= args.maximum_low_rank_gap
    gate3 = confusion_gain >= minimum_proposal_gain
    maximum_relative_residual = max(
        candidate["solver_relative_residual_max"] for candidate in all_results
    )
    numerical_gate = maximum_relative_residual <= args.maximum_relative_solver_residual
    report.update(
        selected_proposal=proposed_best,
        selected_controls=selected_controls,
        gates={
            **report["gates"],
            "gate0_numerical_stability": {
                "pass": numerical_gate,
                "maximum_observed_relative_residual": maximum_relative_residual,
            },
            "gate2_low_rank_approaches_full": {
                "pass": gate2,
                "gap_percentage_points": low_rank_gap,
            },
            "gate3_proposal_beats_controls": {
                "pass": gate3,
                "gain_over_strongest_control_percentage_points": confusion_gain,
                "strongest_control": strongest_control["method"],
            },
        },
        candidates=all_results,
        status="all_validation_gates_passed" if numerical_gate and gate2 and gate3 else "validation_gate_failed",
        held_out_test_authorized=bool(numerical_gate and gate1 and gate2 and gate3),
        total_seconds=time.perf_counter() - started,
    )
    report["final_subspace_diagnostics"] = _final_subspace_diagnostic(
        args, projection, snapshots[-1], proposed_best, selected_controls
    )
    _dump(output_dir / "gate_results.json", report)
    if report["held_out_test_authorized"]:
        print("PASS: all train-only gates passed. Stop and review before held-out test.", flush=True)
    else:
        print("STOP: Gate 2 or Gate 3 failed. Held-out test remains forbidden.", flush=True)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-cache", action="store_true")
    parser.add_argument("--run-gates", action="store_true")
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--gate-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default="CIFAR-100")
    parser.add_argument("--model-name", default="vit_base_patch16_224")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--anchor-dim", type=int, default=2048)
    parser.add_argument("--synaptic-degree", type=int, default=300)
    parser.add_argument("--coding-level", type=float, default=0.3)
    parser.add_argument("--scatter-epsilon", type=float, default=1e-4)
    parser.add_argument("--statistics-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--anchor-batch-size", type=int, default=1024)
    parser.add_argument("--anchor-ridges", default="0.01,0.1,1.0")
    parser.add_argument("--raw-ridges", default="0.01,0.1,1.0")
    parser.add_argument("--residual-ridges", default="0.01,0.1,1.0")
    parser.add_argument("--complement-ridges", default="0.01,0.1,1.0")
    parser.add_argument("--ranks", default="16,32,64,128")
    parser.add_argument("--temperatures", default="0.25,0.5,1.0")
    parser.add_argument("--minimum-full-gain", type=float, default=0.1)
    parser.add_argument("--maximum-low-rank-gap", type=float, default=0.5)
    parser.add_argument("--proposal-method", choices=("confusion_residual", "schur_residual"), default="confusion_residual")
    parser.add_argument("--minimum-proposal-gain", "--minimum-confusion-gain", dest="minimum_proposal_gain", type=float, default=0.1)
    parser.add_argument("--maximum-relative-solver-residual", type=float, default=1e-4)
    args = parser.parse_args(argv)
    if not args.prepare_cache and not args.run_gates:
        parser.error("select --prepare-cache and/or --run-gates")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be in (0, 1)")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.prepare_cache:
        prepare_cache(args)
    if args.run_gates:
        report = run_gates(args)
        _dump(Path(args.output_dir) / "environment.json", {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "cuda_available": torch.cuda.is_available(),
            "argv": sys.argv,
        })
        print(json.dumps({"status": report["status"], "gates": report["gates"]}, indent=2))


if __name__ == "__main__":
    main()
