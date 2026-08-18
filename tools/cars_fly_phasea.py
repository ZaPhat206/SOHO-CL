"""Locked train-only feasibility runner for CARS-FLY Phase A."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
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
from methods.cars_fly import CARSFLYLearner
from methods.crt_soho import CRTSOHOLearner
from methods.sft_cl import create_learner as create_sft_learner
from tools import experiment_runner


SCHEMA_VERSION = 1
REQUIRED_TOP_LEVEL = {
    "schema_version",
    "study_id",
    "dataset",
    "model_name",
    "checkpoint_sha256",
    "seed",
    "num_classes",
    "num_tasks",
    "validation_fraction",
    "statistics_dtype",
    "representation",
    "search",
    "fly_control",
    "gates",
}


def _load_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != REQUIRED_TOP_LEVEL:
        missing = sorted(REQUIRED_TOP_LEVEL - set(payload))
        extra = sorted(set(payload) - REQUIRED_TOP_LEVEL)
        raise ValueError(f"config keys mismatch: missing={missing}, extra={extra}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported CARS-FLY Phase A config schema")
    if payload["seed"] == 1993:
        raise ValueError("new CARS-FLY studies must not reuse historical seed 1993")
    if payload["num_classes"] <= 1 or payload["num_tasks"] <= 0:
        raise ValueError("num_classes and num_tasks must be positive")
    if payload["num_classes"] % payload["num_tasks"]:
        raise ValueError("num_classes must be divisible by num_tasks")
    if not 0 < payload["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be in (0, 1)")
    representation = payload["representation"]
    if set(representation) != {
        "anchor_dim",
        "synaptic_degree",
        "coding_level",
        "encode_batch_size",
    }:
        raise ValueError("representation config keys mismatch")
    search = payload["search"]
    required_search = {
        "anchor_ridges",
        "residual_ridges",
        "complement_ridges",
        "energy_thresholds",
        "max_ranks",
        "min_rank",
        "minimum_objective_gain",
        "raw_ridges",
        "confusion_temperature",
    }
    if set(search) != required_search:
        raise ValueError("search config keys mismatch")
    if any(not values for key, values in search.items() if isinstance(values, list)):
        raise ValueError("search lists must be non-empty")
    if any(value <= 0 for key in ("anchor_ridges", "residual_ridges", "complement_ridges", "raw_ridges") for value in search[key]):
        raise ValueError("all Ridge candidates must be positive")
    if any(not 0 < value <= 1 for value in search["energy_thresholds"]):
        raise ValueError("energy thresholds must be in (0, 1]")
    if search["min_rank"] <= 0 or any(rank < search["min_rank"] for rank in search["max_ranks"]):
        raise ValueError("invalid rank search bounds")
    fly_control = payload["fly_control"]
    if set(fly_control) != {
        "expand_dim",
        "synaptic_degree",
        "coding_level",
        "ridge_lower",
        "ridge_upper",
    }:
        raise ValueError("FLY control config keys mismatch")
    gates = payload["gates"]
    if set(gates) != {
        "maximum_solver_relative_residual",
        "minimum_full_gain_pp",
        "minimum_control_gain_pp",
        "maximum_fly_gap_pp",
        "maximum_state_fraction_of_fly",
    }:
        raise ValueError("gate config keys mismatch")
    return payload


def _dtype(name: str) -> torch.dtype:
    try:
        return {"float32": torch.float32, "float64": torch.float64}[name]
    except KeyError as error:
        raise ValueError("statistics_dtype must be float32 or float64") from error


def _git_identity() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _class_order(config: dict) -> list[int]:
    return random.Random(config["seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )


def _prototype(config: dict, feature_dim: int, device: str) -> CARSFLYLearner:
    representation = config["representation"]
    search = config["search"]
    return CARSFLYLearner(
        raw_dim=feature_dim,
        anchor_dim=representation["anchor_dim"],
        synaptic_degree=representation["synaptic_degree"],
        coding_level=representation["coding_level"],
        anchor_ridge=search["anchor_ridges"][0],
        residual_ridge=search["residual_ridges"][0],
        complement_ridge=search["complement_ridges"][0],
        energy_threshold=search["energy_thresholds"][0],
        max_rank=search["max_ranks"][0],
        min_rank=search["min_rank"],
        minimum_objective_gain=search["minimum_objective_gain"],
        seed=config["seed"],
        device=device,
        dtype=_dtype(config["statistics_dtype"]),
    )


def _encode_anchor(
    prototype: CARSFLYLearner,
    features: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    parts = []
    started = time.perf_counter()
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        parts.append(
            prototype.encode_anchor(features[start:stop]).detach().float().cpu()
        )
        print(
            f"ANCHOR {stop}/{len(features)} elapsed={time.perf_counter()-started:.1f}s",
            flush=True,
        )
    result = torch.cat(parts)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("anchor cache contains NaN or Inf")
    return result


def _predictions(learner, logits: torch.Tensor) -> torch.Tensor:
    columns = logits.argmax(1).detach().cpu().tolist()
    return torch.tensor([learner.class_ids[column] for column in columns])


def _evaluate(
    *,
    name: str,
    learner,
    train: dict,
    anchor_codes: torch.Tensor | None,
    train_parts: list[torch.Tensor],
    validation_parts: list[torch.Tensor],
    rank_schedule: list[int] | None = None,
) -> dict:
    stage_accuracy = []
    diagnostics = []
    started = time.perf_counter()
    for task, indices in enumerate(train_parts):
        task_started = time.perf_counter()
        if rank_schedule is not None and hasattr(learner, "requested_rank"):
            learner.requested_rank = max(int(rank_schedule[task]), 1)
        if anchor_codes is not None and hasattr(learner, "update_from_views"):
            learner.update_from_views(
                train["features"][indices],
                anchor_codes[indices],
                train["labels"][indices],
            )
        else:
            learner.update(train["features"][indices], train["labels"][indices])
        accuracies = []
        for previous in range(task + 1):
            validation = validation_parts[previous]
            if anchor_codes is not None and hasattr(learner, "predict_logits_from_views"):
                logits = learner.predict_logits_from_views(
                    train["features"][validation], anchor_codes[validation]
                )
            else:
                logits = learner.predict_logits(train["features"][validation])
            prediction = _predictions(learner, logits)
            accuracies.append(
                float(
                    (
                        prediction == train["labels"][validation].cpu()
                    ).float().mean().item()
                    * 100
                )
            )
        stage_accuracy.append(sum(accuracies) / len(accuracies))
        current = getattr(learner, "diagnostics", {})
        diagnostics.append(
            {
                "task": task + 1,
                "stage_accuracy": stage_accuracy[-1],
                "effective_rank": current.get("effective_rank"),
                "retained_correction_energy": current.get(
                    "retained_correction_energy"
                ),
                "captured_energy": current.get("captured_energy"),
                "tail_energy": current.get("tail_energy"),
                "energy_threshold_reached": current.get(
                    "energy_threshold_reached"
                ),
                "solver_relative_residual_max": current.get(
                    "solver_relative_residual_max"
                ),
                "seconds": time.perf_counter() - task_started,
            }
        )
        print(
            f"TASK method={name} {task+1}/{len(train_parts)} "
            f"AA={stage_accuracy[-1]:.4f} "
            f"rank={diagnostics[-1]['effective_rank']} "
            f"elapsed={diagnostics[-1]['seconds']:.1f}s",
            flush=True,
        )
    state_bytes = learner.persistent_state_bytes()
    return {
        "method": name,
        "validation_average_accuracy": sum(stage_accuracy) / len(stage_accuracy),
        "stage_accuracy": stage_accuracy,
        "persistent_state_bytes": state_bytes,
        "diagnostics": diagnostics,
        "candidate_seconds": time.perf_counter() - started,
        "uses_test_set": False,
        "exemplar_free": bool(getattr(learner, "is_exemplar_free", True)),
    }


def _cars(
    config: dict,
    feature_dim: int,
    projection: torch.Tensor,
    candidate: dict,
    device: str,
) -> CARSFLYLearner:
    representation = config["representation"]
    search = config["search"]
    return CARSFLYLearner(
        raw_dim=feature_dim,
        anchor_dim=representation["anchor_dim"],
        synaptic_degree=representation["synaptic_degree"],
        coding_level=representation["coding_level"],
        anchor_ridge=candidate["anchor_ridge"],
        residual_ridge=candidate["residual_ridge"],
        complement_ridge=candidate["complement_ridge"],
        energy_threshold=candidate["energy_threshold"],
        max_rank=candidate["max_rank"],
        min_rank=search["min_rank"],
        minimum_objective_gain=search["minimum_objective_gain"],
        seed=config["seed"],
        device=device,
        dtype=_dtype(config["statistics_dtype"]),
        anchor_projection=projection,
    )


def _crt(
    config: dict,
    feature_dim: int,
    projection: torch.Tensor,
    candidate: dict,
    method: str,
    device: str,
) -> CRTSOHOLearner:
    representation = config["representation"]
    return CRTSOHOLearner(
        method=method,
        raw_dim=feature_dim,
        anchor_dim=representation["anchor_dim"],
        synaptic_degree=representation["synaptic_degree"],
        coding_level=representation["coding_level"],
        anchor_ridge=candidate["anchor_ridge"],
        residual_ridge=candidate["residual_ridge"],
        complement_ridge=candidate["complement_ridge"],
        requested_rank=candidate["max_rank"],
        confusion_temperature=config["search"]["confusion_temperature"],
        seed=config["seed"],
        device=device,
        dtype=_dtype(config["statistics_dtype"]),
        anchor_projection=projection,
    )


def _select_raw(
    config: dict,
    train: dict,
    train_parts: list[torch.Tensor],
    validation_parts: list[torch.Tensor],
    device: str,
) -> dict:
    results = []
    for ridge in config["search"]["raw_ridges"]:
        learner = create_sft_learner(
            method="raw_ridge",
            feature_dim=train["features"].shape[1],
            ridge_lambda=ridge,
            seed=config["seed"],
            device=device,
            dtype=_dtype(config["statistics_dtype"]),
        )
        result = _evaluate(
            name="raw_ridge",
            learner=learner,
            train=train,
            anchor_codes=None,
            train_parts=train_parts,
            validation_parts=validation_parts,
        )
        result["ridge_lambda"] = ridge
        results.append(result)
    return max(
        results,
        key=lambda item: (
            item["validation_average_accuracy"],
            -item["persistent_state_bytes"],
            -item["ridge_lambda"],
        ),
    )


def run(args: argparse.Namespace) -> dict:
    config_path = Path(args.config)
    config = _load_config(config_path)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(args.feature_cache_dir)
    if args.require_test_hidden and (cache / "test.pt").exists():
        raise RuntimeError("test.pt must be physically hidden during Phase A")
    cache_args = argparse.Namespace(
        dataset=config["dataset"], model_name=config["model_name"]
    )
    train, test, metadata = experiment_runner.validate_cache(
        cache, cache_args, load_test=False
    )
    if test is not None:
        raise AssertionError("train-only cache validation unexpectedly opened test")
    if metadata.get("checkpoint_sha256") != config["checkpoint_sha256"]:
        raise ValueError("feature cache checkpoint mismatch")
    observed_classes = sorted(set(map(int, train["labels"].tolist())))
    expected_classes = list(range(config["num_classes"]))
    if observed_classes != expected_classes:
        raise ValueError("feature cache class IDs must be exactly [0, num_classes)")

    order = _class_order(config)
    task_parts = experiment_runner.split(
        train["labels"], order, config["num_tasks"]
    )
    train_parts, validation_parts = experiment_runner.train_validation_indices(
        train["labels"],
        task_parts,
        config["seed"],
        config["validation_fraction"],
    )
    feature_dim = int(train["features"].shape[1])
    prototype = _prototype(config, feature_dim, args.device)
    projection = prototype.anchor.projection_matrix.detach().clone()
    anchor_codes = _encode_anchor(
        prototype,
        train["features"],
        config["representation"]["encode_batch_size"],
    )
    del prototype
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    search = config["search"]
    candidates = []
    grid = itertools.product(
        search["anchor_ridges"],
        search["residual_ridges"],
        search["complement_ridges"],
        search["energy_thresholds"],
        search["max_ranks"],
    )
    for index, values in enumerate(grid, start=1):
        candidate = dict(
            zip(
                (
                    "anchor_ridge",
                    "residual_ridge",
                    "complement_ridge",
                    "energy_threshold",
                    "max_rank",
                ),
                values,
            )
        )
        print(f"START candidate={index} config={candidate}", flush=True)
        learner = _cars(config, feature_dim, projection, candidate, args.device)
        result = _evaluate(
            name="cars_fly",
            learner=learner,
            train=train,
            anchor_codes=anchor_codes,
            train_parts=train_parts,
            validation_parts=validation_parts,
        )
        result["config"] = candidate
        candidates.append(result)
        print(
            f"DONE candidate={index} val_AA={result['validation_average_accuracy']:.4f} "
            f"state={result['persistent_state_bytes']}B",
            flush=True,
        )
    selected = max(
        candidates,
        key=lambda item: (
            item["validation_average_accuracy"],
            -item["persistent_state_bytes"],
            -item["config"]["max_rank"],
        ),
    )
    selected_config = selected["config"]
    rank_schedule = [
        int(item["effective_rank"]) for item in selected["diagnostics"]
    ]

    controls = {"cars_fly": selected}
    raw = _select_raw(
        config, train, train_parts, validation_parts, args.device
    )
    controls["raw_ridge"] = raw
    control_methods = {
        "compact_anchor": "anchor_only",
        "full_raw_residual": "full_raw_residual",
        "fixed_rank_schur": "schur_residual",
        "random_residual": "random_residual",
        "fisher_residual": "fisher_residual",
        "confusion_residual": "confusion_residual",
        "shuffled_confusion_residual": "shuffled_confusion_residual",
    }
    for name, method in control_methods.items():
        learner = _crt(
            config,
            feature_dim,
            projection,
            selected_config,
            method,
            args.device,
        )
        schedule = (
            rank_schedule
            if method
            in {
                "random_residual",
                "fisher_residual",
                "confusion_residual",
                "shuffled_confusion_residual",
            }
            else None
        )
        controls[name] = _evaluate(
            name=name,
            learner=learner,
            train=train,
            anchor_codes=anchor_codes,
            train_parts=train_parts,
            validation_parts=validation_parts,
            rank_schedule=schedule,
        )

    fly_config = config["fly_control"]
    fly = CachedFlyCLFidelity(
        feature_dim=feature_dim,
        expand_dim=fly_config["expand_dim"],
        synaptic_degree=fly_config["synaptic_degree"],
        coding_level=fly_config["coding_level"],
        num_classes=config["num_classes"],
        ridge_lower=fly_config["ridge_lower"],
        ridge_upper=fly_config["ridge_upper"],
        seed=config["seed"],
        device=args.device,
    )
    controls["matched_fly"] = _evaluate(
        name="matched_fly",
        learner=fly,
        train=train,
        anchor_codes=None,
        train_parts=train_parts,
        validation_parts=validation_parts,
    )

    maximum_residual = max(
        float(item["solver_relative_residual_max"] or 0.0)
        for result in controls.values()
        for item in result["diagnostics"]
    )
    low_rank_names = (
        "random_residual",
        "fisher_residual",
        "confusion_residual",
        "shuffled_confusion_residual",
    )
    strongest_control = max(
        low_rank_names,
        key=lambda name: controls[name]["validation_average_accuracy"],
    )
    gate_config = config["gates"]
    cars_accuracy = selected["validation_average_accuracy"]
    fly_accuracy = controls["matched_fly"]["validation_average_accuracy"]
    gates = {
        "numerical_stability": maximum_residual
        <= gate_config["maximum_solver_relative_residual"],
        "full_joint_has_headroom": controls["full_raw_residual"][
            "validation_average_accuracy"
        ]
        - controls["compact_anchor"]["validation_average_accuracy"]
        >= gate_config["minimum_full_gain_pp"],
        "cars_beats_low_rank_controls": cars_accuracy
        - controls[strongest_control]["validation_average_accuracy"]
        >= gate_config["minimum_control_gain_pp"],
        "cars_accuracy_gap_to_fly": fly_accuracy - cars_accuracy
        <= gate_config["maximum_fly_gap_pp"],
        "cars_state_fraction_of_fly": selected["persistent_state_bytes"]
        / controls["matched_fly"]["persistent_state_bytes"]
        <= gate_config["maximum_state_fraction_of_fly"],
        "adaptive_rank_not_always_saturated": any(
            rank < selected_config["max_rank"] for rank in rank_schedule
        ),
        "heldout_test_remained_hidden": not (cache / "test.pt").exists(),
    }
    decision = "PASS_REVIEW_HELDOUT_PROTOCOL" if all(gates.values()) else "STOP_CARS_FLY"
    commit, dirty = _git_identity()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "study_id": config["study_id"],
        "status": "train_only_selection_complete",
        "decision": decision,
        "uses_test_set": False,
        "held_out_test_authorized": False,
        "selected_cars_config": selected_config,
        "selected_rank_schedule": rank_schedule,
        "selected_raw_ridge": raw["ridge_lambda"],
        "candidates": candidates,
        "controls": controls,
        "gates": gates,
        "gate_diagnostics": {
            "maximum_solver_relative_residual": maximum_residual,
            "strongest_low_rank_control": strongest_control,
            "gain_over_strongest_control_pp": cars_accuracy
            - controls[strongest_control]["validation_average_accuracy"],
            "gap_to_fly_pp": fly_accuracy - cars_accuracy,
            "state_fraction_of_fly": selected["persistent_state_bytes"]
            / controls["matched_fly"]["persistent_state_bytes"],
        },
        "unavailable_controls": [
            "LoRanPAC-style control is required before a paper claim but is not implemented in this repository"
        ],
        "provenance": {
            "git_commit": commit,
            "git_dirty": dirty,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": args.device,
            "config_path": str(config_path),
            "config_sha256": _sha256(config_path),
            "cache_metadata": metadata,
            "class_order": order,
            "class_order_sha256": hashlib.sha256(
                ",".join(map(str, order)).encode("ascii")
            ).hexdigest(),
            "training_indices_sha256": experiment_runner._sequence_sha256(
                train_parts
            ),
            "validation_indices_sha256": experiment_runner._sequence_sha256(
                validation_parts
            ),
        },
    }
    destination = output / "phasea_results.json"
    destination.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({"decision": decision, "gates": gates}, indent=2), flush=True)
    print(f"RESULT {destination}", flush=True)
    return payload


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--require-test-hidden", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
