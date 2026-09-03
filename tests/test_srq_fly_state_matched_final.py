import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import pytest
import torch

from tools import srq_fly_state_matched_final as runner
from tools.srq_fly_d2_state_match import exact_fly_state_bytes


CONFIG = Path("configs/srq_fly_state_matched_final.json")


def test_locked_config_derives_width_from_bytes_without_accuracy():
    config = runner._read_config(CONFIG)
    expected = {"cifar100": 4409, "cub200": 4518, "imagenetr": 4518}
    for key, width in expected.items():
        match = runner.closest_non_exceeding_width(
            target_bytes=config["state_matching"]["p2b_target_bytes"][key],
            feature_dim=768,
            synaptic_degree=300,
            num_classes=config["datasets"][key]["num_classes"],
            maximum_width=9999,
        )
        assert set(match) == {
            "width", "exact_fly_state_bytes", "target_p2b_state_bytes",
            "relative_byte_gap",
        }
        assert match["width"] == width
        assert match["exact_fly_state_bytes"] <= match["target_p2b_state_bytes"]
        assert match["relative_byte_gap"] < 0.001
        next_state = exact_fly_state_bytes(
            feature_dim=768, expand_dim=width + 1, synaptic_degree=300,
            num_classes=config["datasets"][key]["num_classes"],
        )
        assert next_state > match["target_p2b_state_bytes"]


def test_width_selector_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="positive"):
        runner.closest_non_exceeding_width(
            target_bytes=0, feature_dim=3, synaptic_degree=2,
            num_classes=2, maximum_width=5,
        )
    with pytest.raises(ValueError, match="width-one"):
        runner.closest_non_exceeding_width(
            target_bytes=1, feature_dim=3, synaptic_degree=2,
            num_classes=2, maximum_width=5,
        )


def _fake_reference(tmp_path: Path):
    artifact = tmp_path / "reference.zip"
    prefix = "export/results"
    replicates = [
        {"class_order_seed": 3031 + index, "projection_seed": 5031 + index}
        for index in range(6)
    ]
    summary = {
        "study_id": "reference-study",
        "status": "CONFIRMATION_REPORTED_WITHOUT_ACCURACY_GATE",
        "uses_test_set": True,
        "test_tuning_allowed": False,
    }
    summary_bytes = json.dumps(summary).encode()
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr(f"{prefix}/summary.json", summary_bytes)
        for key in runner.DATASET_KEYS:
            payload = {
                "study_id": "reference-study", "dataset_key": key,
                "status": "CONFIRMATION_COMPLETE", "uses_test_set": True,
                "test_tuning_allowed": False,
                "seed_results": [
                    {
                        **replicate, "methods": {
                            name: {"status": "complete"}
                            for name in runner.REFERENCE_METHODS
                        },
                    }
                    for replicate in replicates
                ],
            }
            archive.writestr(f"{prefix}/{key}.json", json.dumps(payload))
    config = {
        "p2b_reference": {
            "artifact_sha256": runner._sha256_file(artifact),
            "summary_member": f"{prefix}/summary.json",
            "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
            "results_member_template": f"{prefix}/{{dataset_key}}.json",
            "study_id": "reference-study",
        },
        "final_evaluation": {"replicates": replicates},
    }
    return artifact, config


def test_reference_artifact_is_content_addressed(tmp_path):
    artifact, config = _fake_reference(tmp_path)
    loaded = runner.read_reference_artifact(config, artifact)
    assert set(loaded["results"]) == set(runner.DATASET_KEYS)
    damaged = tmp_path / "damaged.zip"
    damaged.write_bytes(artifact.read_bytes() + b"x")
    with pytest.raises(ValueError, match="SHA-256"):
        runner.read_reference_artifact(config, damaged)


def test_selection_refuses_visible_test_tensor(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "metadata.json").write_text("{}", encoding="utf-8")
    (cache / "train.pt").write_bytes(b"train")
    (cache / "test.pt").write_bytes(b"test")
    protocol = {
        "datasets": {"cifar100": {}},
        "backbone": {},
    }
    config = {"datasets": {"cifar100": {}}}
    monkeypatch.setattr(runner, "_read_config", lambda _: config)
    monkeypatch.setattr(runner, "_base_protocol", lambda _: protocol)
    monkeypatch.setattr(runner.base, "_validate_dataset_audit", lambda *a, **k: None)
    args = SimpleNamespace(
        config=str(tmp_path / "config.json"), dataset_key="cifar100",
        feature_cache_dir=str(cache), dataset_audit=None,
    )
    with pytest.raises(RuntimeError, match="visible test.pt"):
        runner.select_dataset(args)


def test_train_unit_resume_is_context_bound(tmp_path):
    path = tmp_path / "unit.json"
    calls = []

    def evaluate():
        calls.append(True)
        return {"status": "complete", "uses_test_set": False, "value": 7}

    first = runner._run_train_unit(path, "abc", "test", evaluate)
    second = runner._run_train_unit(path, "abc", "test", evaluate)
    assert first["value"] == second["value"] == 7
    assert len(calls) == 1
    with pytest.raises(RuntimeError, match="stale unit artifact"):
        runner._run_train_unit(path, "changed", "test", evaluate)


def test_train_exact_adapter_runs_two_task_stream(monkeypatch):
    protocol = {
        "representation": {
            "synaptic_degree": 1, "coding_level": 0.5,
            "encode_batch_size": 4, "evaluation_batch_size": 4,
            "block_size": 2, "group_size": 2,
        }
    }
    config = {
        "state_matching": {"selected_widths": {"tiny": 2}},
        "datasets": {"tiny": {"num_classes": 2}},
    }
    monkeypatch.setattr(runner, "_base_protocol", lambda _: protocol)
    train = {
        "features": torch.zeros(4, 2),
        "labels": torch.tensor([0, 0, 1, 1]),
    }
    indices = torch.tensor([[0], [0], [1], [1]])
    values = torch.ones(4, 1)
    projection = torch.eye(2).to_sparse_csc()
    result = runner._evaluate_train_exact(
        config=config, dataset_key="tiny", ridge=1.0, seed=7,
        train=train, cache=(indices, values, {}, projection),
        fit_parts=[torch.tensor([0]), torch.tensor([2])],
        validation_parts=[torch.tensor([1]), torch.tensor([3])],
        device=torch.device("cpu"),
    )
    assert result["status"] == "complete"
    assert result["validation_average_accuracy"] == 100.0
    assert result["uses_test_set"] is False


def test_config_has_disjoint_development_and_final_replicates():
    config = runner._read_config(CONFIG)
    development = {
        (row["class_order_seed"], row["projection_seed"])
        for row in config["selection"]["development_replicates"]
    }
    final = {
        (row["class_order_seed"], row["projection_seed"])
        for row in config["final_evaluation"]["replicates"]
    }
    assert len(development) == 3
    assert len(final) == 6
    assert development.isdisjoint(final)
    assert config["final_evaluation"]["test_tuning_allowed"] is False
    assert config["final_evaluation"]["accuracy_based_early_stop"] is False


def test_colab_notebook_is_parseable_and_source_locked():
    notebook = json.loads(
        Path("notebooks/srq_fly_state_matched_final_colab.ipynb").read_text(
            encoding="utf-8"
        )
    )
    assert notebook["nbformat"] == 4
    sources = []
    for index, cell in enumerate(notebook["cells"], 1):
        source = "".join(cell.get("source", []))
        sources.append(source)
        if cell.get("cell_type") == "code":
            compile(source, f"notebook-cell-{index}", "exec")
    joined = "\n".join(sources)
    assert "EXPECTED_CONFIG_SHA256" in joined
    assert "EXPECTED_RUNNER_SHA256" in joined
    assert "EXPECTED_REFERENCE_SHA256" in joined
    assert "test.pt" in joined
    assert "--require-clean-git" in joined
    assert "paired_p2b_minus_state_matched_fly_aia" in joined
