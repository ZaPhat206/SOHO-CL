"""M12 locked-confirmation protocol and implementation gates."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import zipfile

import pytest
import torch

from methods.analytic_ridge import persistent_tensor_bytes
from tools import srq_generalization_m12 as m12


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/srq_generalization_m12_locked_test_confirmation.json"
NOTEBOOK = ROOT / "notebooks/srq_generalization_m12_locked_test_colab.ipynb"


def test_m12_config_freezes_test_confirmation_without_accuracy_gate():
    config = m12._read_config(CONFIG)
    assert config["uses_test_set"] is True
    assert config["test_tuning_allowed"] is False
    assert config["accuracy_based_selection"] is False
    assert config["seed"] == 2025
    assert config["widths"] == [10000, 20000]
    assert config["methods"] == list(m12.METHODS)
    assert config["excluded_development_method"]["method"] == "scale_refined_int8"
    assert config["integrity_gates"]["accuracy_gate"] is None
    assert len(config["replicates"]) == 6


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("test_tuning_allowed", True),
        ("accuracy_based_selection", True),
        ("widths", [10000]),
        ("methods", ["exact", "scale_refined_int8"]),
    ],
)
def test_m12_rejects_post_test_selection_or_changed_scope(tmp_path, field, value):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config[field] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        m12._read_config(path)


def test_m12_source_loader_verifies_outer_and_embedded_hashes(tmp_path):
    payload = json.dumps({"status": "LOCKED"}, separators=(",", ":")).encode()
    artifact = tmp_path / "source.zip"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("result.json", payload)
    source = {
        "artifact_filename": artifact.name,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "result_member": "result.json",
        "result_sha256": hashlib.sha256(payload).hexdigest(),
        "required_status": "LOCKED",
    }
    assert m12._load_source(source, artifact)["status"] == "LOCKED"
    broken = copy.deepcopy(source)
    broken["result_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="embedded result"):
        m12._load_source(broken, artifact)


def _unit(width, method, seed, aia, final, state):
    return {
        "identity": {
            "width": width,
            "method": method,
            "class_order_seed": seed,
        },
        "average_incremental_accuracy_percent": aia,
        "final_accuracy_percent": final,
        "final_total_persistent_bytes": state,
        "analytic_update_seconds": 2.0,
        "representation_encoding_seconds": 1.0,
        "maximum_solver_relative_residual": 1e-6,
    }


def test_m12_aggregate_uses_paired_differences_and_sample_sd():
    config = {"widths": [10]}
    units = []
    for seed, exact in ((1, 90.0), (2, 92.0)):
        units.extend(
            [
                _unit(10, "exact", seed, exact, exact - 1, 400),
                _unit(10, "p2b_int8", seed, exact - 0.2, exact - 1.3, 100),
                _unit(10, "adaptive_int8_fp16", seed, exact - 0.1, exact - 1.1, 120),
            ]
        )
    result = m12._aggregate(units, config)[0]["methods"]
    assert result["exact"]["average_incremental_accuracy_percent"]["mean"] == 91.0
    assert result["exact"]["average_incremental_accuracy_percent"]["sample_standard_deviation"] == pytest.approx(2**0.5)
    assert result["p2b_int8"]["paired_aia_difference_from_exact_pp"]["mean"] == pytest.approx(-0.2)
    assert result["adaptive_int8_fp16"]["paired_final_difference_from_exact_pp"]["mean"] == pytest.approx(-0.1)


def test_m12_authorization_id_is_canonical_and_sensitive():
    identity = {"b": [2, 1], "a": {"x": False}}
    assert m12._canonical_sha256(identity) == m12._canonical_sha256(
        {"a": {"x": False}, "b": [2, 1]}
    )
    assert m12._canonical_sha256(identity) != m12._canonical_sha256(
        {"a": {"x": True}, "b": [2, 1]}
    )


def test_m12_small_locked_unit_runs_all_three_backends_with_taskwise_bytes():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config.update({"num_classes": 4, "num_tasks": 2, "widths": [12]})
    config["selected_ridge_by_width"] = {"12": 10.0}
    config["ranpac"].update(
        {"feature_dimension": 5, "maximum_expand_dimension": 12,
         "encode_batch_size": 8, "evaluation_batch_size": 8}
    )
    config["p2b"].update(
        {"block_size": 4, "group_size": 2, "update_panel_size": 4,
         "quantization_batch_blocks": 2}
    )
    replicate = {"class_order_seed": 11, "projection_seed": 17}
    labels = torch.tensor([0, 1, 2, 3] * 5)
    test_labels = torch.tensor([0, 1, 2, 3] * 2)
    generator = torch.Generator().manual_seed(3)
    train = {"features": torch.randn(20, 5, generator=generator), "labels": labels}
    test = {"features": torch.randn(8, 5, generator=generator), "labels": test_labels}

    order = __import__("random").Random(11).sample(list(range(4)), 4)
    parts = m12.split(labels, order, 2)
    expected = {}
    projection = torch.empty(5, 12)
    for method in m12.METHODS:
        backend = m12._make_backend(config, 12, method, torch.device("cpu"))
        method_bytes = []
        for indices in parts:
            backend.update(torch.randn(len(indices), 12, generator=generator), labels[indices])
            method_bytes.append(
                persistent_tensor_bytes(
                    {"projection": projection, **backend.persistent_tensors()}
                )
            )
        expected[method] = method_bytes
    sources = {
        "source_m6": {"width_results": [{
            "width": 12,
            "records": [
                {"state": {
                    "exact": {"total_persistent_bytes": expected["exact"][task]},
                    "p2b_int8": {"total_persistent_bytes": expected["p2b_int8"][task]},
                }} for task in range(2)
            ],
        }]},
        "source_m11": {"width_results": [{
            "width": 12,
            "records": [
                {"total_persistent_bytes": expected["adaptive_int8_fp16"][task]}
                for task in range(2)
            ],
        }]},
    }
    authorization = {"authorization_id": "a" * 64}
    for method in m12.METHODS:
        result = m12._run_unit(
            config=config, authorization=authorization, sources=sources,
            train=train, test=test, replicate=replicate, width=12,
            method=method, device=torch.device("cpu"),
        )
        assert len(result["records"]) == 2
        assert result["final_total_persistent_bytes"] == expected[method][-1]
        assert result["uses_test_set"] is True
        m12._validate_unit(result, result["identity"], sources, config)


def test_m12_notebook_enforces_authorize_before_test_and_exports_locked_result():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert code.index("'authorize'") < code.index("'extract-test'") < code.index("'run'")
    assert "srq_generalization_m12_locked_test_confirmation.zip" in code
    assert "scale_refined_int8" not in code
    assert "m12_results.json" in code
    for relative in (
        "configs/srq_generalization_m12_locked_test_confirmation.json",
        "tools/srq_generalization_m12.py",
    ):
        digest = hashlib.sha256(
            (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        assert f"'{relative}':'{digest}'" in code


def test_m12_runner_and_notebook_compile():
    compile((ROOT / "tools/srq_generalization_m12.py").read_text(encoding="utf-8"), "m12", "exec")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{index}", "exec")
