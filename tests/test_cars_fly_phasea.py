import argparse
import json

import pytest
import torch

from tools import cars_fly_phasea, experiment_runner


def config_payload():
    return {
        "schema_version": 1,
        "study_id": "cars-fly-tiny-train-only",
        "dataset": "tiny-cars",
        "model_name": "synthetic",
        "checkpoint_sha256": "synthetic-checkpoint",
        "seed": 2025,
        "num_classes": 4,
        "num_tasks": 2,
        "validation_fraction": 0.25,
        "statistics_dtype": "float64",
        "representation": {
            "anchor_dim": 9,
            "synaptic_degree": 3,
            "coding_level": 0.34,
            "encode_batch_size": 8,
        },
        "search": {
            "anchor_ridges": [0.1],
            "residual_ridges": [0.2],
            "complement_ridges": [0.3],
            "energy_thresholds": [0.8, 0.95],
            "max_ranks": [2],
            "min_rank": 1,
            "minimum_objective_gain": 0.0,
            "raw_ridges": [0.1],
            "confusion_temperature": 1.0,
        },
        "fly_control": {
            "expand_dim": 11,
            "synaptic_degree": 3,
            "coding_level": 0.34,
            "ridge_lower": -1,
            "ridge_upper": 2,
        },
        "gates": {
            "maximum_solver_relative_residual": 1e-5,
            "minimum_full_gain_pp": -100.0,
            "minimum_control_gain_pp": -100.0,
            "maximum_fly_gap_pp": 100.0,
            "maximum_state_fraction_of_fly": 100.0,
        },
    }


def prepare(tmp_path):
    cache = tmp_path / "cache"
    config = tmp_path / "config.json"
    output = tmp_path / "output"
    payload = config_payload()
    config.write_text(json.dumps(payload), encoding="utf-8")
    generator = torch.Generator().manual_seed(91)
    features = torch.randn(48, 6, generator=generator)
    labels = torch.tensor([0, 1, 2, 3] * 12)
    cache_args = argparse.Namespace(
        dataset=payload["dataset"],
        model_name=payload["model_name"],
        data_augmentation="none",
    )
    experiment_runner.save_train_cache(
        cache,
        features,
        labels,
        cache_args,
        expected_test_samples=16,
        checkpoint_hash=payload["checkpoint_sha256"],
    )
    args = argparse.Namespace(
        config=str(config),
        feature_cache_dir=str(cache),
        output_dir=str(output),
        device="cpu",
        require_test_hidden=True,
    )
    return args, cache, config, output


def test_phasea_runner_is_train_only_and_emits_controls_and_certificates(tmp_path):
    args, _, _, output = prepare(tmp_path)
    result = cars_fly_phasea.run(args)

    assert result["uses_test_set"] is False
    assert result["held_out_test_authorized"] is False
    assert (output / "phasea_results.json").is_file()
    assert len(result["candidates"]) == 2
    assert {
        "cars_fly",
        "raw_ridge",
        "compact_anchor",
        "full_raw_residual",
        "fixed_rank_schur",
        "random_residual",
        "fisher_residual",
        "confusion_residual",
        "shuffled_confusion_residual",
        "matched_fly",
    } == set(result["controls"])
    assert len(result["selected_rank_schedule"]) == 2
    assert all(
        item["uses_test_set"] is False for item in result["controls"].values()
    )
    assert all(
        item["captured_energy"] is not None
        for item in result["controls"]["cars_fly"]["diagnostics"]
    )


def test_phasea_runner_fails_if_heldout_cache_is_visible(tmp_path):
    args, cache, _, _ = prepare(tmp_path)
    torch.save({"features": torch.randn(4, 6), "labels": torch.arange(4)}, cache / "test.pt")
    with pytest.raises(RuntimeError, match="physically hidden"):
        cars_fly_phasea.run(args)


def test_new_phase_config_rejects_historical_seed(tmp_path):
    args, _, config, _ = prepare(tmp_path)
    payload = json.loads(config.read_text())
    payload["seed"] = 1993
    config.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="historical seed"):
        cars_fly_phasea.run(args)


def test_phase_config_rejects_incomplete_class_inventory(tmp_path):
    args, cache, _, _ = prepare(tmp_path)
    train = torch.load(cache / "train.pt", weights_only=True)
    keep = train["labels"] != 3
    torch.save(
        {"features": train["features"][keep], "labels": train["labels"][keep]},
        cache / "train.pt",
    )
    with pytest.raises(ValueError, match="exactly"):
        cars_fly_phasea.run(args)
