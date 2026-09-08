"""M7 gates for task-wise quantization-error diagnosis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import zipfile

import pytest
import torch

from tools import srq_generalization_m7 as m7


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "srq_generalization_m7_error_trajectory_train_only.json"
NOTEBOOK = ROOT / "notebooks" / "srq_generalization_m7_error_trajectory_colab.ipynb"


def _fake_m6(tmp_path: Path, config: dict) -> Path:
    source = config["source_m6"]
    result = {
        "status": source["required_status"],
        "uses_test_set": False,
        "width_results": [
            {"width": width, "selected_ridge_lambda": config["ridge_by_width"][str(width)]}
            for width in config["diagnostic_widths"]
        ],
        "summary": {"maximum_p2b_validation_aia_loss_pp": 0.255845},
    }
    result_bytes = (json.dumps(result) + "\n").encode()
    path = tmp_path / source["artifact_filename"]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(source["result_member"], result_bytes)
    source["artifact_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    source["result_sha256"] = hashlib.sha256(result_bytes).hexdigest()
    return path


def test_m7_config_is_train_only_diagnostic_and_not_selection():
    config = m7._read_config(CONFIG)
    assert config["uses_test_set"] is False
    assert config["accuracy_based_selection"] is False
    assert config["diagnostic_widths"] == [10000, 20000]
    assert config["diagnostics"]["system_probe_count"] == 16


def test_m7_rejects_test_use_and_changed_widths(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["uses_test_set"] = True
    path = tmp_path / "test.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="train-only"):
        m7._read_config(path)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["diagnostic_widths"] = [15000]
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="10k and 20k"):
        m7._read_config(path)


def test_m7_verifies_failed_m6_by_content_identity(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    artifact = _fake_m6(tmp_path, config)
    verified = m7._verify_m6_artifact(config, artifact)
    assert verified["status"] == "FAIL_M6_WIDTH_SWEEP_TRAIN_ONLY"
    damaged = tmp_path / "damaged.zip"
    damaged.write_bytes(artifact.read_bytes() + b"x")
    config["source_m6"]["artifact_filename"] = damaged.name
    with pytest.raises(ValueError, match="SHA-256"):
        m7._verify_m6_artifact(config, damaged)


def test_margin_certificate_is_sufficient_for_unchanged_prediction():
    reference_logits = torch.tensor([[4.0, 1.0], [2.0, 1.9], [3.0, 0.0]])
    candidate_logits = torch.tensor([[3.9, 1.1], [1.8, 2.1], [2.8, 0.1]])
    reference_prediction, reference_margin = m7._prediction_and_margin(
        reference_logits, [0, 1]
    )
    candidate_prediction, candidate_margin = m7._prediction_and_margin(
        candidate_logits, [0, 1]
    )
    backend = type("Backend", (), {})()
    backend.class_ids = [0, 1]
    backend.weights = torch.eye(2)
    evaluation = {
        "logits": candidate_logits,
        "prediction": candidate_prediction,
        "margin": candidate_margin,
        "accuracy_percent": 2 / 3 * 100,
    }
    reference = {
        "logits": reference_logits,
        "prediction": reference_prediction,
        "margin": reference_margin,
        "accuracy_percent": 100.0,
        "class_ids": [0, 1],
        "weights": torch.eye(2),
    }
    result = m7._compare_snapshot(evaluation, backend, reference)
    assert result["prediction_agreement"] == pytest.approx(2 / 3)
    assert result["margin_certified_fraction"] == pytest.approx(2 / 3)
    certified = 2 * (candidate_logits - reference_logits).abs().amax(1) < reference_margin
    assert torch.equal(candidate_prediction[certified], reference_prediction[certified])


def test_m7_summary_has_only_integrity_and_numerical_gates():
    config = m7._read_config(CONFIG)
    width_results = []
    for width in config["diagnostic_widths"]:
        exact = [
            {
                "task": task,
                "accuracy_percent": 90.0,
                "solver_relative_residual": 1e-6,
                "exact_probe_identity_error": 1e-7,
            }
            for task in range(1, 11)
        ]
        compressed = [
            {
                "task": task,
                "accuracy_percent": 89.9,
                "solver_relative_residual": 2e-6,
                "relative_system_action_error": task * 1e-4,
                "relative_weight_error": task * 2e-4,
                "relative_logit_error": task * 3e-4,
                "prediction_agreement": 0.99,
                "margin_certified_fraction": 0.95,
                "accuracy_gap_exact_minus_method_pp": 0.1,
            }
            for task in range(1, 11)
        ]
        width_results.append(
            {
                "width": width,
                "records": {
                    "exact": exact,
                    "fp16_square_root": compressed,
                    "p2b_int8": compressed,
                },
            }
        )
    summary = m7._summarize(width_results, config, True)
    assert all(summary["gates"].values())
    assert "p2b_accuracy_retention" not in summary["gates"]
    assert summary["per_width"]["20000"][
        "p2b_final_relative_system_action_error"
    ] == pytest.approx(0.001)


def test_m7_small_stream_records_factor_system_weight_logit_and_margin_errors():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["ranpac"]["encode_batch_size"] = 4
    config["ranpac"]["evaluation_batch_size"] = 4
    config["p2b"]["block_size"] = 2
    config["p2b"]["group_size"] = 2
    config["p2b"]["update_panel_size"] = 2
    features = torch.randn(12, 3, generator=torch.Generator().manual_seed(4))
    labels = torch.tensor([0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 2, 3])
    training_parts = [torch.tensor([0, 1, 2, 3]), torch.tensor([4, 5, 6, 7])]
    validation_parts = [torch.tensor([8, 9]), torch.tensor([10, 11])]
    projection = torch.randn(3, 4, generator=torch.Generator().manual_seed(5))
    probes = torch.randn(4, 2, generator=torch.Generator().manual_seed(6))
    references, exact = m7._run_exact(
        config=config,
        width=4,
        ridge=10.0,
        projection=projection,
        probes=probes,
        features=features,
        labels=labels,
        training_parts=training_parts,
        validation_parts=validation_parts,
        device=torch.device("cpu"),
    )
    p2b = m7._run_compressed(
        config=config,
        width=4,
        ridge=10.0,
        mode="p2b_int8",
        projection=projection,
        probes=probes,
        references=references,
        features=features,
        labels=labels,
        training_parts=training_parts,
        validation_parts=validation_parts,
        device=torch.device("cpu"),
    )
    assert len(exact) == len(p2b) == 2
    assert max(item["exact_probe_identity_error"] for item in exact) < 1e-6
    assert all(0 <= item["prediction_agreement"] <= 1 for item in p2b)
    assert all(0 <= item["margin_certified_fraction"] <= 1 for item in p2b)
    assert all(item["relative_system_action_error"] >= 0 for item in p2b)


def test_m7_notebook_is_source_locked_train_only_and_compiles():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "--extract-train-only" in code
    assert "test.pt').exists()" in code
    assert "--m6-artifact" in code
    assert "--require-clean-git" in code
    assert "m7_results.json" in code
    assert "error_trajectory.csv" in code
    assert "m7_error_trajectory.svg" in code
    assert "files.download(archive)" in code
    locked = dict(re.findall(r"'([^']+)':'([0-9a-f]{64})'", code))
    assert "tools/srq_generalization_m7.py" in locked
    for relative_path, expected in locked.items():
        canonical = (ROOT / relative_path).read_text(encoding="utf-8")
        actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert actual == expected, relative_path
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")
