import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from tools import srq_fly_selfcontained as final


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/srq_fly_selfcontained_final.json"


def test_protocol_locks_grid_and_independent_random_sources():
    protocol = final._read_protocol(PROTOCOL)
    final._verify_method_identity(protocol)
    assert protocol["selection"]["split_seed"] == 2025
    assert len(protocol["selection"]["ridge_grid"]) == 12
    development = protocol["selection"]["development_replicates"]
    heldout = protocol["final_evaluation"]["replicates"]
    assert {item["class_order_seed"] for item in development}.isdisjoint(
        {item["class_order_seed"] for item in heldout}
    )
    assert all(item["class_order_seed"] != item["projection_seed"] for item in development + heldout)
    assert protocol["final_evaluation"]["methods"] == list(final.METHODS)


def test_nested_split_is_stratified_disjoint_and_complete():
    labels = torch.tensor([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10)
    parts = final._nested_parts(labels, [2, 0, 3, 1], 2, 2025, 0.2, 0.2)
    inner_fit, inner_val, outer_fit, outer_val = parts
    for task in range(2):
        assert not set(inner_fit[task].tolist()) & set(inner_val[task].tolist())
        assert not set(outer_fit[task].tolist()) & set(outer_val[task].tolist())
        assert set(inner_fit[task].tolist()) | set(inner_val[task].tolist()) == set(outer_fit[task].tolist())
    all_indices = set(torch.cat(outer_fit + outer_val).tolist())
    assert all_indices == set(range(40))
    assert sum(len(part) for part in inner_fit) == 24
    assert sum(len(part) for part in inner_val) == 8
    assert sum(len(part) for part in outer_val) == 8


def test_train_only_cache_refuses_visible_test(tmp_path):
    protocol = copy.deepcopy(final._read_protocol(PROTOCOL))
    protocol["backbone"]["feature_dim"] = 3
    protocol["datasets"]["cifar100"].update({"num_classes": 2, "train_samples": 4, "test_samples": 2})
    cache = tmp_path / "cache"; cache.mkdir()
    (cache / "metadata.json").write_text(json.dumps({
        "dataset": "CIFAR-100", "backbone_model": "vit_base_patch16_224",
        "checkpoint_sha256": protocol["backbone"]["checkpoint_sha256"], "preprocessing": "vit",
    }), encoding="utf-8")
    torch.save({"features": torch.randn(4, 3), "labels": torch.tensor([0, 0, 1, 1])}, cache / "train.pt")
    torch.save({"features": torch.randn(2, 3), "labels": torch.tensor([0, 1])}, cache / "test.pt")
    with pytest.raises(RuntimeError, match="refuses a visible test.pt"):
        final._validate_train_cache(cache, protocol, "cifar100", require_test=False)


def test_selection_wrapper_uses_only_train_and_selects_interior_candidate(tmp_path, monkeypatch):
    protocol = copy.deepcopy(final._read_protocol(PROTOCOL))
    protocol["backbone"]["feature_dim"] = 3
    protocol["representation"].update({"expand_dim": 6, "synaptic_degree": 2, "coding_level": 1 / 3})
    protocol["selection"]["ridge_grid"] = [0.1, 1.0, 10.0]
    protocol["selection"]["development_replicates"] = [
        {"class_order_seed": 2025, "projection_seed": 4201},
        {"class_order_seed": 2026, "projection_seed": 4202},
        {"class_order_seed": 2027, "projection_seed": 4203},
    ]
    protocol["datasets"]["cifar100"].update({"num_classes": 2, "num_tasks": 2, "train_samples": 20})
    train = {"features": torch.randn(20, 3), "labels": torch.tensor([0] * 10 + [1] * 10)}
    monkeypatch.setattr(final, "_read_protocol", lambda _: protocol)
    monkeypatch.setattr(final, "_verify_method_identity", lambda _: {})
    monkeypatch.setattr(final, "_validate_dataset_audit", lambda *_: None)
    monkeypatch.setattr(final, "_validate_train_cache", lambda *_args, **_kwargs: (train, None, {"dataset": "CIFAR-100"}))
    monkeypatch.setattr(final, "_sha256_file", lambda _: "source")
    monkeypatch.setattr(final, "_tensor_content_sha256", lambda _: "projection")
    cache = (
        torch.zeros((20, 2), dtype=torch.int16), torch.ones((20, 2)),
        {"identity_sha256": "codes"}, torch.randn(6, 3).to_sparse_csc(),
    )
    monkeypatch.setattr(final, "_prepare_code_cache", lambda **_: cache)

    def paired(**kwargs):
        ridge = kwargs["config"]["ridge_lambda"]
        score = 90.0 - abs(ridge - 1.0)
        result = {"status": "complete", "validation_average_accuracy": score, "maximum_solver_relative_residual": 0.0}
        return {"status": "complete", "exact": dict(result), "srq": dict(result), "uses_test_set": False}

    def raw(**kwargs):
        ridge = kwargs["config"]["raw_ridge_lambda"]
        return {"status": "complete", "validation_average_accuracy": 80.0 - abs(ridge - 1.0), "maximum_solver_relative_residual": 0.0, "uses_test_set": False}

    monkeypatch.setattr(final.d1, "_evaluate_paired_exact_srq", paired)
    monkeypatch.setattr(final.d0, "_evaluate_raw", raw)
    result = final.select_dataset(
        protocol_path=tmp_path / "protocol.json", dataset_key="cifar100",
        feature_cache_dir=tmp_path / "cache", code_cache_root=tmp_path / "codes",
        output_root=tmp_path / "output", dataset_audit_path=None, device_name="cpu",
    )
    assert result["status"] == "SELECTION_COMPLETE"
    assert result["selected_fly_family_lambda"] == 1.0
    assert result["selected_raw_ridge_lambda"] == 1.0
    assert result["uses_test_set"] is False


def test_lock_binds_all_selection_files_and_rejects_mutation(tmp_path, monkeypatch):
    protocol = final._read_protocol(PROTOCOL)
    monkeypatch.setattr(final, "_verify_method_identity", lambda _: {})
    for key in final.DATASET_KEYS:
        path = tmp_path / "selection" / key / "selection.json"; path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "status": "SELECTION_COMPLETE", "uses_test_set": False,
            "held_out_test_authorized": False, "protocol_sha256": final._sha256_file(PROTOCOL),
            "runner_sha256": final._sha256_file(Path(final.__file__)),
            "selected_fly_family_lambda": 1.0, "selected_raw_ridge_lambda": 0.1,
        }), encoding="utf-8")
    boundary_path = tmp_path / "selection" / "cifar100" / "selection.json"
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    boundary["status"] = "STOP_BOUNDARY_SELECTION"
    boundary_path.write_text(json.dumps(boundary), encoding="utf-8")
    with pytest.raises(ValueError, match="selection contract mismatch"):
        final.lock_selection(PROTOCOL, tmp_path / "selection", tmp_path / "output", False)
    boundary["status"] = "SELECTION_COMPLETE"
    boundary_path.write_text(json.dumps(boundary), encoding="utf-8")
    record = final.lock_selection(PROTOCOL, tmp_path / "selection", tmp_path / "output", False)
    assert record["test_tuning_allowed"] is False
    changed = tmp_path / "selection" / "cifar100" / "selection.json"
    changed.write_text(changed.read_text() + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="selection changed"):
        final._validate_authorization(tmp_path / "output" / "authorization.json", PROTOCOL, tmp_path / "selection")


def test_summary_emits_plot_ready_tables(tmp_path):
    for key in final.DATASET_KEYS:
        seed_results = []
        for replicate in range(6):
            methods = {}
            for method in final.METHODS:
                methods[method] = {
                    "status": "complete", "final_accuracy": 80.0,
                    "average_incremental_accuracy": 85.0 + (method == "srq_fly_10000"),
                    "forgetting": 3.0, "persistent_state_bytes": 1024,
                    "total_update_seconds": 1.0, "total_inference_seconds": 2.0,
                    "stage_accuracy": [90.0, 80.0],
                }
            seed_results.append({"replicate_index": replicate, "methods": methods})
        path = tmp_path / key / "heldout_results.json"; path.parent.mkdir()
        path.write_text(json.dumps({"uses_test_set": True, "test_tuning_allowed": False, "seed_results": seed_results}), encoding="utf-8")
    summary = final.summarize(PROTOCOL, tmp_path)
    assert summary["status"] == "REPORTED_WITHOUT_ACCURACY_GATE"
    assert (tmp_path / "metrics_summary.csv").is_file()
    assert (tmp_path / "task_curves.csv").is_file()


def test_colab_notebook_cells_compile_and_bind_source_hashes():
    notebook_path = ROOT / "notebooks/srq_fly_selfcontained_final_colab.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell_{index}", "exec")
    text = notebook_path.read_text(encoding="utf-8")
    for path in (PROTOCOL, ROOT / "tools/srq_fly_selfcontained.py"):
        assert hashlib.sha256(path.read_bytes()).hexdigest() in text
    assert "01_accuracy_by_task.png" in text
    assert "04_accuracy_memory_pareto.png" in text
