"""Protocol and numerical gates for the locked M16 Cars experiment."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from tools import srq_generalization_m16 as m16


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/srq_generalization_m16_cars_phase2_locked.json"
NOTEBOOK = ROOT / "notebooks/srq_generalization_m16_cars_phase2_colab.ipynb"


def test_m16_config_pins_official_cars_phase2_scope():
    config = m16._read_config(CONFIG)
    assert config["dataset"] == "Stanford Cars"
    assert config["class_increments"] == [16] + [20] * 9
    assert config["ranpac"]["expand_dimension"] == 10000
    assert config["backbone"]["architecture"] == "resnet50"
    assert config["backbone"]["weights"] == "IMAGENET1K_V2"
    assert config["ranpac_source"]["cars_publish_row"] == 10
    assert config["methods"] == list(m16.METHODS)
    assert config["integrity_gates"]["accuracy_gate"] is None
    assert len(config["replicates"]) == 6


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("test_tuning_allowed", True),
        ("accuracy_based_method_selection", True),
        ("class_increments", [196]),
        ("methods", ["exact"]),
    ],
)
def test_m16_rejects_changed_frozen_scope(tmp_path, field, value):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config[field] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        m16._read_config(path)


def test_m16_unequal_schedule_assigns_every_sample_once():
    labels = torch.tensor([0, 1, 2, 3, 4, 5] * 3)
    parts = m16._split_by_increments(labels, [4, 1, 5, 0, 3, 2], [2, 1, 3])
    assert [len(part) for part in parts] == [6, 3, 9]
    merged = torch.cat(parts)
    assert sorted(merged.tolist()) == list(range(len(labels)))


def test_m16_dual_ridge_matches_primal_and_uses_smallest_tie():
    generator = torch.Generator().manual_seed(17)
    fit = torch.randn(12, 7, generator=generator, dtype=torch.float64)
    validation = torch.randn(8, 7, generator=generator, dtype=torch.float64)
    fit_labels = torch.tensor([0, 1, 2] * 4)
    validation_labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    candidates = [0.1, 1.0, 10.0]
    selected, records = m16._select_ridge_dual(
        fit, fit_labels, validation, validation_labels, candidates
    )
    target = torch.nn.functional.one_hot(fit_labels, num_classes=3).double()
    validation_target = torch.nn.functional.one_hot(
        validation_labels, num_classes=3
    ).double()
    primal_losses = []
    for ridge in candidates:
        weights = torch.linalg.solve(
            fit.T @ fit + ridge * torch.eye(7, dtype=torch.float64),
            fit.T @ target,
        )
        primal_losses.append(float(torch.mean((validation @ weights - validation_target) ** 2)))
    assert [record["validation_mse"] for record in records] == pytest.approx(
        primal_losses, rel=1e-11, abs=1e-12
    )
    assert selected == candidates[min(range(3), key=primal_losses.__getitem__)]


def _small_config():
    config = copy.deepcopy(json.loads(CONFIG.read_text(encoding="utf-8")))
    config.update({"num_classes": 4, "class_increments": [2, 2]})
    config["backbone"]["feature_dimension"] = 5
    config["ranpac"].update({
        "expand_dimension": 64,
        "encode_batch_size": 8,
        "evaluation_batch_size": 8,
    })
    config["p2b"].update({
        "block_size": 32,
        "group_size": 16,
        "update_panel_size": 32,
        "quantization_batch_blocks": 2,
    })
    return config


def test_m16_small_unit_runs_all_locked_backends():
    config = _small_config()
    replicate = {
        "class_order_seed": 5,
        "projection_seed": 7,
        "ridge_split_seed": 11,
    }
    generator = torch.Generator().manual_seed(19)
    train = {
        "features": torch.randn(24, 5, generator=generator),
        "labels": torch.tensor([0, 1, 2, 3] * 6),
    }
    test = {
        "features": torch.randn(12, 5, generator=generator),
        "labels": torch.tensor([0, 1, 2, 3] * 3),
    }
    results = {
        method: m16._run_unit(
            config, train, test, replicate, 10.0, method, torch.device("cpu")
        )
        for method in m16.METHODS
    }
    assert all(len(result["records"]) == 2 for result in results.values())
    assert all(result["uses_test_set"] is True for result in results.values())
    assert results["p2b_int8"]["final_total_persistent_bytes"] < results["exact"][
        "final_total_persistent_bytes"
    ]
    assert results["adaptive_int8_fp16"]["final_total_persistent_bytes"] < results[
        "exact"
    ]["final_total_persistent_bytes"]


def test_m16_aggregate_reports_paired_differences_and_sample_sd():
    units = []
    for seed, exact in ((1, 80.0), (2, 82.0)):
        replicate = {
            "class_order_seed": seed,
            "projection_seed": seed,
            "ridge_split_seed": seed,
        }
        for method, delta, state in (
            ("exact", 0.0, 400.0),
            ("p2b_int8", -0.2, 100.0),
            ("adaptive_int8_fp16", -0.1, 120.0),
        ):
            units.append({
                "identity": {"replicate": replicate, "method": method},
                "average_incremental_accuracy_percent": exact + delta,
                "final_accuracy_percent": exact - 1 + delta,
                "final_total_persistent_bytes": state,
                "analytic_update_seconds": 2.0,
                "maximum_solver_relative_residual": 1e-6,
            })
    summary = m16._aggregate(units)
    assert summary["exact"]["average_incremental_accuracy_percent"]["mean"] == 81.0
    assert summary["exact"]["average_incremental_accuracy_percent"][
        "sample_standard_deviation"
    ] == pytest.approx(2**0.5)
    assert summary["p2b_int8"]["paired_aia_difference_from_exact_pp"][
        "mean"
    ] == pytest.approx(-0.2)
    assert summary["adaptive_int8_fp16"]["paired_final_difference_from_exact_pp"][
        "mean"
    ] == pytest.approx(-0.1)


def test_m16_resumable_unit_validation_rejects_changed_identity():
    config = _small_config()
    replicate = {
        "class_order_seed": 5,
        "projection_seed": 7,
        "ridge_split_seed": 11,
    }
    generator = torch.Generator().manual_seed(23)
    train = {
        "features": torch.randn(24, 5, generator=generator),
        "labels": torch.tensor([0, 1, 2, 3] * 6),
    }
    test = {
        "features": torch.randn(12, 5, generator=generator),
        "labels": torch.tensor([0, 1, 2, 3] * 3),
    }
    unit = m16._run_unit(
        config, train, test, replicate, 10.0, "exact", torch.device("cpu")
    )
    m16._validate_unit(unit, config, replicate, 10.0, "exact")
    changed = copy.deepcopy(unit)
    changed["identity"]["ridge_lambda"] = 1.0
    with pytest.raises(RuntimeError, match="identity"):
        m16._validate_unit(changed, config, replicate, 10.0, "exact")


def test_m16_runner_compiles():
    compile(
        (ROOT / "tools/srq_generalization_m16.py").read_text(encoding="utf-8"),
        "m16",
        "exec",
    )


def test_m16_notebook_locks_source_and_authorizes_before_test():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    assert "REPO_COMMIT='209f8f901fcf248530d25d0ca36f4f7bd87f05fd'" in code
    assert code.index("'select-ridge'") < code.index("'authorize'") < code.index("'extract-test'") < code.index("'run'")
    assert "eduardo4jesus/stanford-cars-dataset" in code
    assert "m16_handoff_{count:03d}_units.zip" in code
    assert "--max-new-units','6'" in code
    assert "accuracy_gate" not in code
    for relative in (
        "configs/srq_generalization_m16_cars_phase2_locked.json",
        "tools/srq_generalization_m16.py",
    ):
        digest = hashlib.sha256(
            (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        assert f"'{relative}':'{digest}'" in code


def test_m16_notebook_code_cells_compile():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"m16-cell-{index}", "exec")
