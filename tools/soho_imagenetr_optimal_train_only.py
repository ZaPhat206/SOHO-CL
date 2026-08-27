"""Fair train-only SOHO selection against the fixed original FLY control.

Seeds are preregistered replicate factors, never selected by accuracy.  The
runner first selects a fixed Ridge coefficient at the previously locked
ImageNet-R SOHO representation, then refines a small representation
neighbourhood with that Ridge fixed.  An untouched outer partition evaluates
the selected SOHO and FLY fidelity under five paired seeds.  No test feature is
loaded and no outcome authorizes held-out evaluation automatically.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.cached_replay_baselines import (  # noqa: E402
    CachedSOHOReplayFidelity,
    _ridge,
)
from tools import soho_selfcontained as base  # noqa: E402


T_CRITICAL_95_DF4 = 2.7764451051977987


class FixedRidgeSOHO(CachedSOHOReplayFidelity):
    """Exact SOHO replay with a declared fixed Ridge coefficient."""

    def __init__(self, *args, fixed_ridge: float, **kwargs):
        if fixed_ridge <= 0:
            raise ValueError("fixed_ridge must be positive")
        super().__init__(*args, **kwargs)
        self.fixed_ridge = float(fixed_ridge)
        self.diagnostics["ridge_policy"] = "fixed_train_validation_selected"

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
        self.last_ridge = self.fixed_ridge
        self.weights = _ridge(self.G, self.Q, self.fixed_ridge)
        self.diagnostics.update(
            effective_rank=int(self.soho.R.shape[0]),
            selected_ridge=self.fixed_ridge,
            retained_sample_count=int(len(historical_x)),
        )


def _read_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1:
        raise ValueError("protocol schema mismatch")
    if base._sha256_file(ROOT / "tools/soho_selfcontained.py") != protocol.get(
        "base_runner_sha256"
    ):
        raise ValueError("base runner identity mismatch")
    dataset = protocol.get("dataset", {})
    backbone = protocol.get("backbone", {})
    if (
        backbone.get("model_name") != "vit_base_patch16_224"
        or backbone.get("feature_dim") != 768
        or backbone.get("checkpoint_sha256")
        != "32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b"
    ):
        raise ValueError("backbone contract mismatch")
    if (
        dataset.get("key") != "imagenetr"
        or dataset.get("dataset") != "ImageNet-R"
        or dataset.get("num_classes") != 200
        or dataset.get("num_tasks") != 20
        or dataset.get("train_samples") != 23918
    ):
        raise ValueError("ImageNet-R dataset contract mismatch")
    selection = protocol.get("selection", {})
    if (
        selection.get("split_seed") != 2025
        or selection.get("ridge_grid") != [10.0, 100.0, 1000.0, 10000.0]
        or selection.get("density_grid") != [0.1, 0.2, 0.3]
        or selection.get("coding_level_grid") != [0.4, 0.45, 0.5]
        or selection.get("near_tie_tolerance_pp") != 0.05
        or len(selection.get("development_replicates", [])) != 3
    ):
        raise ValueError("selection contract mismatch")
    expected_development = [
        {"class_order_seed": 2025, "projection_seed": 12025},
        {"class_order_seed": 3407, "projection_seed": 13407},
        {"class_order_seed": 4421, "projection_seed": 14421},
    ]
    expected_outer = expected_development + [
        {"class_order_seed": 5501, "projection_seed": 15501},
        {"class_order_seed": 6619, "projection_seed": 16619},
    ]
    if selection["development_replicates"] != expected_development:
        raise ValueError("development seed identity mismatch")
    if protocol.get("outer_confirmation", {}).get("replicates") != expected_outer:
        raise ValueError("outer seed identity mismatch")
    if protocol["outer_confirmation"].get("held_out_test_authorized") is not False:
        raise ValueError("held-out test must remain unauthorized")
    if protocol["outer_confirmation"].get("gate") != {
        "minimum_mean_soho_minus_fly_aia_pp": 0.0,
        "minimum_seed_wins": 4,
        "minimum_paired_ci95_low_pp": -0.5,
    }:
        raise ValueError("outer gate contract mismatch")
    fixed, fly = protocol.get("soho_fixed", {}), protocol.get("fly_fidelity", {})
    if (
        fixed.get("expand_dim") != 10000
        or fixed.get("olda_dim") != 768
        or fixed.get("anchor_density") != 0.2
        or fixed.get("anchor_coding_level") != 0.45
        or fixed.get("use_etf") is not True
    ):
        raise ValueError("SOHO fixed contract mismatch")
    if (
        fly.get("expand_dim") != 10000
        or fly.get("synaptic_degree") != 300
        or fly.get("coding_level") != 0.3
        or fly.get("ridge_lower") != 6
        or fly.get("ridge_upper") != 10
    ):
        raise ValueError("FLY fidelity contract mismatch")
    return protocol


def _verify_method_identity(protocol: dict) -> dict:
    observed = {
        "soho_model_sha256": base._sha256_normalized_source(ROOT / "models/soho.py"),
        "sohocl_sha256": base._sha256_normalized_source(ROOT / "methods/sohocl.py"),
        "cached_baselines_sha256": base._sha256_normalized_source(
            ROOT / "methods/cached_replay_baselines.py"
        ),
        "flyhash_sha256": base._sha256_normalized_source(ROOT / "models/flyhash.py"),
    }
    if observed != protocol["method_identity"]:
        raise ValueError("SOHO/FLY source identity mismatch")
    return observed


def _base_protocol(protocol: dict) -> dict:
    configured = copy.deepcopy(protocol)
    configured["datasets"] = {"imagenetr": protocol["dataset"]}
    configured["fly_fixed"] = copy.deepcopy(protocol["fly_fidelity"])
    configured["soho_fixed"]["ridge_lower"] = -2
    configured["soho_fixed"]["ridge_upper"] = 10
    configured["soho_fixed"]["gcv_sample_size"] = 3000
    return configured


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _evaluate_soho(
    protocol: dict,
    stream: dict,
    training_parts: list[torch.Tensor],
    evaluation_parts: list[torch.Tensor],
    config: dict,
    projection_seed: int,
    device_name: str,
) -> dict:
    device = torch.device(device_name)
    fixed = protocol["soho_fixed"]
    learner = FixedRidgeSOHO(
        protocol["backbone"]["feature_dim"],
        fixed["expand_dim"],
        config["density"],
        fixed["olda_dim"],
        fixed["use_etf"],
        config["coding_level"],
        protocol["dataset"]["num_classes"],
        -2,
        10,
        seed=projection_seed,
        device=device_name,
        replay_chunk_size=fixed["replay_chunk_size"],
        gcv_sample_size=3000,
        fixed_ridge=config["ridge_lambda"],
    )
    matrix, update_seconds, inference_seconds = [], [], []
    retained_rows = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for task, training_indices in enumerate(training_parts):
            _sync(device)
            started = time.perf_counter()
            learner.update(
                stream["features"][training_indices], stream["labels"][training_indices]
            )
            _sync(device)
            update_seconds.append(time.perf_counter() - started)
            retained_rows += len(training_indices)
            audit = base._state_audit(
                learner, "soho_replay_fidelity", retained_rows
            )
            row = []
            for previous in range(task + 1):
                indices = evaluation_parts[previous]
                _sync(device)
                started = time.perf_counter()
                predictions = learner.predict(stream["features"][indices])
                _sync(device)
                inference_seconds.append(time.perf_counter() - started)
                row.append(
                    float(
                        (
                            predictions
                            == stream["labels"][indices].detach().cpu()
                        )
                        .float()
                        .mean()
                        .item()
                        * 100
                    )
                )
            matrix.append(row)
            print(
                f"TASK method=soho_fixed_ridge task={task+1}/{len(training_parts)} "
                f"AA={statistics.fmean(row):.4f} lambda={config['ridge_lambda']:g}",
                flush=True,
            )
        return {
            "status": "complete",
            "uses_test_set": False,
            "method": "soho_fixed_ridge",
            "config": config,
            **base._metrics(matrix),
            "persistent_state_bytes": learner.persistent_state_bytes(),
            "total_update_seconds": sum(update_seconds),
            "total_inference_seconds": sum(inference_seconds),
            "peak_runtime_memory_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None,
            "state_audit": audit,
        }
    finally:
        del learner
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _select_near_tie(
    results: list[dict], tolerance: float, tie_key
) -> tuple[dict, float]:
    valid = [item for item in results if item.get("valid")]
    if not valid:
        raise RuntimeError("no valid candidate")
    best = max(float(item["mean_inner_aia"]) for item in valid)
    eligible = [
        item for item in valid if best - float(item["mean_inner_aia"]) <= tolerance
    ]
    return min(eligible, key=tie_key), best


def _mean_ci(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    half = T_CRITICAL_95_DF4 * std / math.sqrt(len(values)) if len(values) == 5 else None
    return {
        "n": len(values),
        "mean": mean,
        "sample_std": std,
        "ci95_low": None if half is None else mean - half,
        "ci95_high": None if half is None else mean + half,
    }


def _candidate(
    *,
    protocol: dict,
    stream: dict,
    source: dict,
    output_dir: Path,
    phase: str,
    candidate_index: int,
    config: dict,
    replicates: list[dict],
    device_name: str,
) -> dict:
    results = []
    for item in replicates:
        context = {
            **source,
            "phase": phase,
            "candidate": config,
            "replicate": item["replicate"],
            "class_order": item["class_order"],
        }
        result = base._unit(
            output_dir
            / "units"
            / f"{phase}_c{candidate_index}_r{item['index']}.json",
            context,
            lambda item=item, config=config: _evaluate_soho(
                protocol,
                stream,
                item["parts"][0],
                item["parts"][1],
                config,
                item["replicate"]["projection_seed"],
                device_name,
            ),
        )
        results.append(result)
    valid = all(item.get("status") == "complete" for item in results)
    payload = {
        "candidate_index": candidate_index,
        "config": config,
        "valid": valid,
        "mean_inner_aia": statistics.fmean(
            item["average_incremental_accuracy"] for item in results
        )
        if valid
        else None,
        "per_replicate": results,
    }
    print(
        f"CANDIDATE {phase} {candidate_index+1} config={config} "
        f"mean_AIA={payload['mean_inner_aia']}",
        flush=True,
    )
    return payload


def run(
    protocol_path: Path,
    feature_cache_dir: Path,
    output_dir: Path,
    dataset_audit_path: Path,
    device: str,
):
    protocol = _read_protocol(protocol_path)
    identities = _verify_method_identity(protocol)
    dataset_audit = base._validate_dataset_audit(
        dataset_audit_path, "imagenetr", protocol["dataset"]
    )
    cache_protocol = _base_protocol(protocol)
    train, _, metadata = base._validate_cache(
        feature_cache_dir, cache_protocol, "imagenetr", require_test=False
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "units").mkdir(exist_ok=True)
    source = {
        "protocol_sha256": base._sha256_file(protocol_path),
        "runner_sha256": base._sha256_file(Path(__file__).resolve()),
        "train_sha256": base._sha256_file(feature_cache_dir / "train.pt"),
        "dataset_audit_sha256": base._sha256_file(dataset_audit_path),
        "method_identity": identities,
    }
    selection = protocol["selection"]
    development = []
    for index, replicate in enumerate(selection["development_replicates"]):
        class_order = random.Random(replicate["class_order_seed"]).sample(
            range(protocol["dataset"]["num_classes"]),
            protocol["dataset"]["num_classes"],
        )
        development.append(
            {
                "index": index,
                "replicate": replicate,
                "class_order": class_order,
                "parts": base._nested_parts(
                    train["labels"],
                    class_order,
                    protocol["dataset"]["num_tasks"],
                    selection["split_seed"],
                    selection["outer_validation_fraction"],
                    selection["inner_validation_fraction"],
                ),
            }
        )

    anchor = protocol["soho_fixed"]
    ridge_results = []
    for index, ridge in enumerate(selection["ridge_grid"]):
        ridge_results.append(
            _candidate(
                protocol=protocol,
                stream=train,
                source=source,
                output_dir=output_dir,
                phase="inner_ridge",
                candidate_index=index,
                config={
                    "density": anchor["anchor_density"],
                    "coding_level": anchor["anchor_coding_level"],
                    "ridge_lambda": float(ridge),
                    "use_etf": True,
                },
                replicates=development,
                device_name=device,
            )
        )
    selected_ridge, best_ridge_score = _select_near_tie(
        ridge_results,
        selection["near_tie_tolerance_pp"],
        lambda item: -float(item["config"]["ridge_lambda"]),
    )
    ridge = float(selected_ridge["config"]["ridge_lambda"])

    representation_results = []
    configs = [
        {
            "density": float(density),
            "coding_level": float(coding),
            "ridge_lambda": ridge,
            "use_etf": True,
        }
        for density in selection["density_grid"]
        for coding in selection["coding_level_grid"]
    ]
    for index, config in enumerate(configs):
        if (
            config["density"] == anchor["anchor_density"]
            and config["coding_level"] == anchor["anchor_coding_level"]
        ):
            reused = copy.deepcopy(selected_ridge)
            reused["candidate_index"] = index
            reused["config"] = config
            reused["reused_from_phase"] = "inner_ridge"
            representation_results.append(reused)
            print(f"REUSED anchor representation config={config}", flush=True)
        else:
            representation_results.append(
                _candidate(
                    protocol=protocol,
                    stream=train,
                    source=source,
                    output_dir=output_dir,
                    phase="inner_representation",
                    candidate_index=index,
                    config=config,
                    replicates=development,
                    device_name=device,
                )
            )
    selected_representation, best_representation_score = _select_near_tie(
        representation_results,
        selection["near_tie_tolerance_pp"],
        lambda item: (
            float(item["config"]["coding_level"]),
            float(item["config"]["density"]),
        ),
    )
    selected = selected_representation["config"]

    outer_results = []
    for index, replicate in enumerate(protocol["outer_confirmation"]["replicates"]):
        class_order = random.Random(replicate["class_order_seed"]).sample(
            range(protocol["dataset"]["num_classes"]),
            protocol["dataset"]["num_classes"],
        )
        parts = base._nested_parts(
            train["labels"],
            class_order,
            protocol["dataset"]["num_tasks"],
            selection["split_seed"],
            selection["outer_validation_fraction"],
            selection["inner_validation_fraction"],
        )
        shared = {
            **source,
            "phase": "outer_confirmation",
            "replicate": replicate,
            "class_order": class_order,
            "selected_soho": selected,
        }
        soho = base._unit(
            output_dir / "units" / f"outer_soho_r{index}.json",
            {**shared, "method": "soho_fixed_ridge"},
            lambda parts=parts, replicate=replicate: _evaluate_soho(
                protocol,
                train,
                parts[2],
                parts[3],
                selected,
                replicate["projection_seed"],
                device,
            ),
        )
        fly = base._unit(
            output_dir / "units" / f"outer_fly_r{index}.json",
            {**shared, "method": "flycl_fidelity"},
            lambda parts=parts, replicate=replicate: base._evaluate(
                "flycl_fidelity",
                cache_protocol,
                protocol["dataset"],
                replicate["projection_seed"],
                train,
                parts[2],
                parts[3],
                selected,
                1.0,
                device,
                False,
            ),
        )
        if soho.get("status") != "complete" or fly.get("status") != "complete":
            raise RuntimeError("outer confirmation unit failed")
        delta = (
            soho["average_incremental_accuracy"]
            - fly["average_incremental_accuracy"]
        )
        outer_results.append(
            {"replicate": replicate, "soho": soho, "fly": fly, "aia_delta": delta}
        )
        print(
            f"OUTER seed={replicate['class_order_seed']} SOHO_AIA="
            f"{soho['average_incremental_accuracy']:.4f} FLY_AIA="
            f"{fly['average_incremental_accuracy']:.4f} delta={delta:+.4f}",
            flush=True,
        )

    deltas = [item["aia_delta"] for item in outer_results]
    paired = _mean_ci(deltas)
    wins = sum(value > 0 for value in deltas)
    gate_spec = protocol["outer_confirmation"]["gate"]
    test_hidden = not (feature_cache_dir / "test.pt").exists()
    if not test_hidden:
        raise RuntimeError("held-out test cache became visible during train-only selection")
    gates = {
        "mean_soho_beats_fly": paired["mean"]
        > gate_spec["minimum_mean_soho_minus_fly_aia_pp"],
        "seed_wins": wins >= gate_spec["minimum_seed_wins"],
        "paired_ci_floor": paired["ci95_low"]
        >= gate_spec["minimum_paired_ci95_low_pp"],
        "test_remained_hidden": test_hidden,
    }
    payload = {
        "schema_version": 1,
        "study_id": protocol["study_id"],
        "status": "TRAIN_ONLY_GATE_PASS" if all(gates.values()) else "TRAIN_ONLY_GATE_FAIL",
        "uses_test_set": False,
        "held_out_test_authorized": False,
        **source,
        "source_feature_metadata": metadata,
        "dataset_audit": dataset_audit,
        "ridge_results": ridge_results,
        "selected_ridge": selected_ridge,
        "best_ridge_score": best_ridge_score,
        "representation_results": representation_results,
        "selected_soho_config": selected,
        "best_representation_score": best_representation_score,
        "fly_fidelity_config": protocol["fly_fidelity"],
        "outer_results": outer_results,
        "outer_paired_soho_minus_fly_aia": paired,
        "outer_seed_wins": wins,
        "gates": gates,
        "interpretation_guard": (
            "This selects the best SOHO configuration only within the locked train-only "
            "search space. Seeds were preregistered and paired, not optimized. A pass "
            "requires review and does not automatically authorize held-out test use."
        ),
    }
    base._atomic_json(output_dir / "selection_results.json", payload)
    with (output_dir / "candidate_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "phase",
                "density",
                "coding_level",
                "ridge_lambda",
                "use_etf",
                "mean_inner_aia",
            ),
        )
        writer.writeheader()
        for phase, results in (
            ("ridge", ridge_results),
            ("representation", representation_results),
        ):
            for item in results:
                writer.writerow(
                    {
                        "phase": phase,
                        **item["config"],
                        "mean_inner_aia": item["mean_inner_aia"],
                    }
                )
    print(
        f"COMPLETE status={payload['status']} selected={selected} "
        f"outer_delta={paired['mean']:+.4f} wins={wins}/5; test remains unauthorized",
        flush=True,
    )
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-audit", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(
        Path(args.protocol).resolve(),
        Path(args.feature_cache_dir).resolve(),
        Path(args.output_dir).resolve(),
        Path(args.dataset_audit).resolve(),
        args.device,
    )


if __name__ == "__main__":
    main()
