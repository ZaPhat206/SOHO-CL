"""Train-only diagnosis of the ImageNet-R SOHO AIA collapse.

The previously consumed three-dataset artifact suggested that SOHO's replay-
wide GCV often selects 0.1 or 1.0 at stages with large accuracy collapses. This
runner does not open ``test.pt`` and does not tune a replacement method. It
replays the exact six historical class/projection seed pairs on a stratified
training-validation split and evaluates one explicitly post-hoc fixed-lambda
counterfactual from the same SOHO state.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
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
    CachedFlyCLFidelity,
    CachedSOHOReplayFidelity,
    _ridge,
)
from tools import soho_selfcontained as base  # noqa: E402


METHODS = (
    "soho_current_gcv",
    "soho_fixed_1000_posthoc",
    "flycl_fidelity",
)
T_CRITICAL_95_DF5 = 2.570581835636305


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _read_protocol(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset = payload.get("dataset", {})
    diagnostic = payload.get("diagnostic", {})
    soho = payload.get("soho", {})
    fly = payload.get("fly_original", {})
    if payload.get("schema_version") != 1:
        raise ValueError("diagnostic protocol schema mismatch")
    if (
        dataset.get("key") != "imagenetr"
        or dataset.get("dataset") != "ImageNet-R"
        or dataset.get("num_classes") != 200
        or dataset.get("num_tasks") != 20
        or dataset.get("train_samples") != 23918
    ):
        raise ValueError("ImageNet-R dataset contract mismatch")
    if (
        diagnostic.get("seed") != 2025
        or diagnostic.get("validation_fraction") != 0.2
        or diagnostic.get("uses_test_set") is not False
        or diagnostic.get("held_out_test_authorized") is not False
        or diagnostic.get("probe_rows") != 256
    ):
        raise ValueError("train-only diagnostic contract mismatch")
    expected_replicates = [
        {"class_order_seed": seed, "projection_seed": seed + 2000}
        for seed in range(3031, 3037)
    ]
    if diagnostic.get("replicates") != expected_replicates:
        raise ValueError("historical paired replicate identity mismatch")
    if (
        soho.get("expand_dim") != 10000
        or soho.get("olda_dim") != 768
        or soho.get("density") != 0.2
        or soho.get("coding_level") != 0.45
        or soho.get("use_etf") is not True
        or soho.get("gcv_ridge_lower") != -2
        or soho.get("gcv_ridge_upper") != 10
        or soho.get("posthoc_fixed_ridge") != 1000.0
    ):
        raise ValueError("locked SOHO diagnostic configuration mismatch")
    if (
        fly.get("expand_dim") != 10000
        or fly.get("synaptic_degree") != 300
        or fly.get("coding_level") != 0.3
        or fly.get("ridge_lower") != 6
        or fly.get("ridge_upper") != 10
    ):
        raise ValueError("original FLY configuration mismatch")
    backbone = payload.get("backbone", {})
    if (
        backbone.get("model_name") != "vit_base_patch16_224"
        or backbone.get("feature_dim") != 768
        or backbone.get("checkpoint_sha256")
        != "32aa17d6e17b43500f531d5f6dc9bc93e56ed8841b8a75682e1bb295d722405b"
    ):
        raise ValueError("backbone contract mismatch")
    return payload


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


def _cache_protocol(protocol: dict) -> dict:
    return {
        "backbone": protocol["backbone"],
        "datasets": {"imagenetr": protocol["dataset"]},
    }


def _validation_parts(labels: torch.Tensor, class_order: list[int], protocol: dict):
    diagnostic = protocol["diagnostic"]
    grouped = base._nested_parts(
        labels,
        class_order,
        protocol["dataset"]["num_tasks"],
        diagnostic["seed"],
        diagnostic["validation_fraction"],
        0.2,
    )
    # Development (80%) fits the learners; the untouched outer 20% evaluates
    # the diagnostic. Inner partitions are deliberately unused: lambda=1000
    # is a post-hoc causal control, not a newly selected method.
    return grouped[2], grouped[3]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _accuracy(encoded: torch.Tensor, weights: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = (encoded @ weights).argmax(1).detach().cpu()
    return float((predictions == labels.cpu()).float().mean().item() * 100)


def _probe_indices(parts: list[torch.Tensor], task: int, count: int, seed: int) -> torch.Tensor:
    historical = torch.cat(parts[:task])
    generator = torch.Generator().manual_seed(seed * 1000 + task)
    order = torch.randperm(len(historical), generator=generator)
    return historical[order[: min(count, len(historical))]]


def _soho_dual_evaluation(
    protocol: dict,
    stream: dict,
    fit_parts: list[torch.Tensor],
    validation_parts: list[torch.Tensor],
    projection_seed: int,
    device_name: str,
) -> dict:
    device = torch.device(device_name)
    dataset, config = protocol["dataset"], protocol["soho"]
    learner = CachedSOHOReplayFidelity(
        protocol["backbone"]["feature_dim"],
        config["expand_dim"],
        config["density"],
        config["olda_dim"],
        config["use_etf"],
        config["coding_level"],
        dataset["num_classes"],
        config["gcv_ridge_lower"],
        config["gcv_ridge_upper"],
        seed=projection_seed,
        device=device_name,
        replay_chunk_size=config["replay_chunk_size"],
        gcv_sample_size=config["gcv_sample_size"],
    )
    matrices = {"soho_current_gcv": [], "soho_fixed_1000_posthoc": []}
    selected_ridges, stage_diagnostics, update_seconds = [], [], []
    retained_rows = 0
    try:
        for task, indices in enumerate(fit_parts):
            previous_r = learner.soho.R.detach().clone()
            old_probe = None
            probe_indices = None
            if task:
                probe_indices = _probe_indices(
                    fit_parts,
                    task,
                    protocol["diagnostic"]["probe_rows"],
                    protocol["diagnostic"]["seed"] + projection_seed,
                )
                with torch.no_grad():
                    old_probe = learner.soho(
                        stream["features"][probe_indices].to(device),
                        config["coding_level"],
                        absolute_wta=False,
                    )
            _sync(device)
            started = time.perf_counter()
            learner.update(stream["features"][indices], stream["labels"][indices])
            _sync(device)
            update_seconds.append(time.perf_counter() - started)
            retained_rows += len(indices)
            selected_ridges.append(learner.last_ridge)
            gcv_weights = learner.weights
            fixed_weights = _ridge(learner.G, learner.Q, config["posthoc_fixed_ridge"])

            support_turnover = None
            code_cosine = None
            if task:
                with torch.no_grad():
                    new_probe = learner.soho(
                        stream["features"][probe_indices].to(device),
                        config["coding_level"],
                        absolute_wta=False,
                    )
                    old_support, new_support = old_probe.ne(0), new_probe.ne(0)
                    intersection = (old_support & new_support).sum(1).to(torch.float32)
                    support_turnover = float(
                        (1.0 - intersection / old_support.sum(1).clamp_min(1)).mean().item()
                    )
                    code_cosine = float(
                        torch.nn.functional.cosine_similarity(
                            old_probe, new_probe, dim=1, eps=1e-12
                        ).mean().item()
                    )
                del old_probe, new_probe, old_support, new_support

            rows = {method: [] for method in matrices}
            for previous in range(task + 1):
                validation_indices = validation_parts[previous]
                with torch.no_grad():
                    encoded = learner.soho(
                        stream["features"][validation_indices].to(device),
                        config["coding_level"],
                        absolute_wta=False,
                    )
                labels = stream["labels"][validation_indices]
                rows["soho_current_gcv"].append(_accuracy(encoded, gcv_weights, labels))
                rows["soho_fixed_1000_posthoc"].append(
                    _accuracy(encoded, fixed_weights, labels)
                )
                del encoded
            for method in matrices:
                matrices[method].append(rows[method])
            r_drift = float(
                torch.linalg.norm(learner.soho.R - previous_r)
                / torch.linalg.norm(previous_r).clamp_min(1e-12)
            )
            stage_diagnostics.append(
                {
                    "task": task + 1,
                    "selected_gcv_ridge": learner.last_ridge,
                    "r_relative_drift": r_drift,
                    "probe_support_turnover": support_turnover,
                    "probe_code_cosine": code_cosine,
                    "gcv_stage_accuracy": statistics.fmean(rows["soho_current_gcv"]),
                    "fixed_stage_accuracy": statistics.fmean(rows["soho_fixed_1000_posthoc"]),
                }
            )
            print(
                f"SOHO TASK {task+1}/{len(fit_parts)} ridge={learner.last_ridge:g} "
                f"gcv_AA={stage_diagnostics[-1]['gcv_stage_accuracy']:.4f} "
                f"fixed_AA={stage_diagnostics[-1]['fixed_stage_accuracy']:.4f} "
                f"support_turnover={support_turnover if support_turnover is not None else 'NA'}",
                flush=True,
            )
            del fixed_weights
        audit = base._state_audit(learner, "soho_replay_fidelity", retained_rows)
        results = {}
        for method, matrix in matrices.items():
            results[method] = {
                "status": "complete",
                "uses_test_set": False,
                "method": method,
                **base._metrics(matrix),
                "persistent_state_bytes": learner.persistent_state_bytes(),
                "total_update_seconds_shared": sum(update_seconds),
                "state_audit": audit,
            }
        results["soho_current_gcv"]["selected_ridge_by_task"] = selected_ridges
        results["soho_fixed_1000_posthoc"]["fixed_ridge"] = config[
            "posthoc_fixed_ridge"
        ]
        return {"methods": results, "stage_diagnostics": stage_diagnostics}
    finally:
        del learner
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _fly_evaluation(
    protocol: dict,
    stream: dict,
    fit_parts: list[torch.Tensor],
    validation_parts: list[torch.Tensor],
    projection_seed: int,
    device_name: str,
) -> dict:
    dataset, config = protocol["dataset"], protocol["fly_original"]
    learner = CachedFlyCLFidelity(
        protocol["backbone"]["feature_dim"],
        config["expand_dim"],
        config["synaptic_degree"],
        config["coding_level"],
        dataset["num_classes"],
        config["ridge_lower"],
        config["ridge_upper"],
        seed=projection_seed,
        device=device_name,
    )
    device = torch.device(device_name)
    matrix, selected_ridges, update_seconds = [], [], []
    try:
        for task, indices in enumerate(fit_parts):
            _sync(device)
            started = time.perf_counter()
            learner.update(stream["features"][indices], stream["labels"][indices])
            _sync(device)
            update_seconds.append(time.perf_counter() - started)
            selected_ridges.append(learner.last_ridge)
            row = []
            for previous in range(task + 1):
                validation_indices = validation_parts[previous]
                row.append(
                    float(
                        (
                            learner.predict(stream["features"][validation_indices])
                            == stream["labels"][validation_indices]
                        )
                        .float()
                        .mean()
                        .item()
                        * 100
                    )
                )
            matrix.append(row)
            print(
                f"FLY TASK {task+1}/{len(fit_parts)} ridge={learner.last_ridge:g} "
                f"AA={statistics.fmean(row):.4f}",
                flush=True,
            )
        return {
            "status": "complete",
            "uses_test_set": False,
            "method": "flycl_fidelity",
            **base._metrics(matrix),
            "persistent_state_bytes": learner.persistent_state_bytes(),
            "total_update_seconds": sum(update_seconds),
            "selected_ridge_by_task": selected_ridges,
            "state_audit": base._state_audit(learner, "flycl_fidelity", 0),
        }
    finally:
        del learner
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _mean_ci(values: list[float]) -> dict:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    half = T_CRITICAL_95_DF5 * std / math.sqrt(len(values)) if len(values) == 6 else None
    return {
        "n": len(values),
        "mean": mean,
        "sample_std": std,
        "ci95_low": None if half is None else mean - half,
        "ci95_high": None if half is None else mean + half,
    }


def _correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)
    )
    return None if denominator == 0 else numerator / denominator


def _summarize(replicates: list[dict], protocol: dict) -> dict:
    summary = {}
    for method in METHODS:
        summary[method] = {
            metric: _mean_ci(
                [item["methods"][method][metric] for item in replicates]
            )
            for metric in ("average_incremental_accuracy", "final_accuracy", "forgetting")
        }
    paired = {}
    for left, right in (
        ("soho_current_gcv", "flycl_fidelity"),
        ("soho_fixed_1000_posthoc", "soho_current_gcv"),
        ("soho_fixed_1000_posthoc", "flycl_fidelity"),
    ):
        paired[f"{left}_minus_{right}"] = {
            metric: _mean_ci(
                [
                    item["methods"][left][metric] - item["methods"][right][metric]
                    for item in replicates
                ]
            )
            for metric in ("average_incremental_accuracy", "final_accuracy")
        }

    ridge_groups: dict[str, list[float]] = {}
    turnover, gcv_minus_fly = [], []
    for item in replicates:
        gcv = item["methods"]["soho_current_gcv"]
        fly = item["methods"]["flycl_fidelity"]
        for task, ridge in enumerate(gcv["selected_ridge_by_task"]):
            delta = gcv["stage_accuracy"][task] - fly["stage_accuracy"][task]
            ridge_groups.setdefault(str(float(ridge)), []).append(delta)
            diagnostic = item["stage_diagnostics"][task]
            if diagnostic["probe_support_turnover"] is not None:
                turnover.append(diagnostic["probe_support_turnover"])
                gcv_minus_fly.append(delta)
    return {
        "status": "POSTHOC_DIAGNOSTIC_COMPLETE",
        "uses_test_set": False,
        "held_out_test_authorized": False,
        "methods": summary,
        "paired_differences": paired,
        "ridge_conditioned_gcv_minus_fly_stage_accuracy": {
            ridge: _mean_ci(values) for ridge, values in sorted(ridge_groups.items())
        },
        "support_turnover_vs_gcv_minus_fly_stage_accuracy_correlation": _correlation(
            turnover, gcv_minus_fly
        ),
        "interpretation_guard": (
            "The fixed-1000 row is a post-hoc causal diagnostic motivated by an already "
            "consumed artifact. It is not validation-selected and cannot be reported as a "
            "new method or used to authorize ImageNet-R test evaluation."
        ),
        "protocol": {
            "study_id": protocol["study_id"],
            "replicates": protocol["diagnostic"]["replicates"],
            "soho": protocol["soho"],
            "fly_original": protocol["fly_original"],
        },
    }


def run(protocol_path: Path, feature_cache_dir: Path, output_dir: Path, device: str) -> dict:
    protocol = _read_protocol(protocol_path)
    identities = _verify_method_identity(protocol)
    train, _, metadata = base._validate_cache(
        feature_cache_dir, _cache_protocol(protocol), "imagenetr", require_test=False
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    units = output_dir / "units"
    units.mkdir(exist_ok=True)
    source = {
        "protocol_sha256": base._sha256_file(protocol_path),
        "runner_sha256": base._sha256_file(Path(__file__).resolve()),
        "train_sha256": base._sha256_file(feature_cache_dir / "train.pt"),
        "method_identity": identities,
    }
    replicate_results = []
    for index, replicate in enumerate(protocol["diagnostic"]["replicates"]):
        class_order = random.Random(replicate["class_order_seed"]).sample(
            range(protocol["dataset"]["num_classes"]),
            protocol["dataset"]["num_classes"],
        )
        fit_parts, validation_parts = _validation_parts(
            train["labels"], class_order, protocol
        )
        context = {
            **source,
            "replicate": replicate,
            "class_order": class_order,
            "phase": "train_only_posthoc_diagnostic",
        }
        path = units / f"replicate_{index}.json"
        context_sha = hashlib.sha256(
            json.dumps(context, sort_keys=True).encode()
        ).hexdigest()
        if path.is_file():
            unit = json.loads(path.read_text(encoding="utf-8"))
            if unit.get("context_sha256") != context_sha:
                raise RuntimeError(f"resume context mismatch: {path}")
            result = unit["result"]
            print(f"RESTORED replicate={index+1}/6", flush=True)
        else:
            print(
                f"START replicate={index+1}/6 class_seed={replicate['class_order_seed']} "
                f"projection_seed={replicate['projection_seed']}",
                flush=True,
            )
            started = time.perf_counter()
            soho_result = _soho_dual_evaluation(
                protocol,
                train,
                fit_parts,
                validation_parts,
                replicate["projection_seed"],
                device,
            )
            fly_result = _fly_evaluation(
                protocol,
                train,
                fit_parts,
                validation_parts,
                replicate["projection_seed"],
                device,
            )
            result = {
                "status": "complete",
                "uses_test_set": False,
                "replicate": replicate,
                "class_order": class_order,
                "methods": {
                    **soho_result["methods"],
                    "flycl_fidelity": fly_result,
                },
                "stage_diagnostics": soho_result["stage_diagnostics"],
            }
            _atomic_json(
                path,
                {
                    "context_sha256": context_sha,
                    "unit_seconds": time.perf_counter() - started,
                    "result": result,
                },
            )
            print(
                f"DONE replicate={index+1}/6 "
                f"SOHO_GCV_AIA={result['methods']['soho_current_gcv']['average_incremental_accuracy']:.4f} "
                f"SOHO_FIXED_AIA={result['methods']['soho_fixed_1000_posthoc']['average_incremental_accuracy']:.4f} "
                f"FLY_AIA={fly_result['average_incremental_accuracy']:.4f}",
                flush=True,
            )
        replicate_results.append(result)
    if not all(item.get("status") == "complete" for item in replicate_results):
        raise RuntimeError("incomplete diagnostic replicate")
    summary = _summarize(replicate_results, protocol)
    payload = {
        "schema_version": 1,
        **source,
        "source_feature_metadata": metadata,
        "replicates": replicate_results,
        "summary": summary,
    }
    _atomic_json(output_dir / "diagnostic_results.json", payload)

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method", "aia_mean", "aia_std", "final_mean", "final_std"),
        )
        writer.writeheader()
        for method, metrics in summary["methods"].items():
            writer.writerow(
                {
                    "method": method,
                    "aia_mean": metrics["average_incremental_accuracy"]["mean"],
                    "aia_std": metrics["average_incremental_accuracy"]["sample_std"],
                    "final_mean": metrics["final_accuracy"]["mean"],
                    "final_std": metrics["final_accuracy"]["sample_std"],
                }
            )
    print(
        "DIAGNOSTIC COMPLETE - test.pt remained absent; return the ZIP for audit.",
        flush=True,
    )
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run(
        Path(args.protocol).resolve(),
        Path(args.feature_cache_dir).resolve(),
        Path(args.output_dir).resolve(),
        args.device,
    )


if __name__ == "__main__":
    main()
