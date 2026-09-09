"""M10 gates for the controlled GACL generalized-stream adapter."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import torch

from tools import srq_generalization_m10 as m10


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/srq_generalization_m10_gacl_train_only.json"
NOTEBOOK = ROOT / "notebooks/srq_generalization_m10_gacl_colab.ipynb"


def _small_config() -> dict:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["num_classes"] = 6
    config["development_subset"].update(
        fit_samples_per_class=10,
        validation_samples_per_class=3,
    )
    config["stream"].update(
        num_tasks=3,
        disjoint_class_ratio_percent=50,
        blurry_sample_ratio_percent=10,
        mini_batch_size=7,
    )
    config["gacl"].update(
        expansion_dim=24,
        encode_batch_size=11,
    )
    config["p2b"].update(
        block_size=6,
        group_size=5,
        update_panel_size=7,
        quantization_batch_blocks=3,
    )
    config["gates"].update(
        maximum_inverse_primal_weight_relative_error=2e-4,
        maximum_inverse_primal_logit_relative_error=2e-4,
        minimum_inverse_primal_prediction_agreement=0.99,
        minimum_p2b_total_state_reduction_fraction=0.0,
        maximum_p2b_validation_aia_loss_pp=100.0,
        maximum_solver_relative_residual=1e-4,
    )
    return config


def _small_data() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(771)
    labels = torch.arange(6).repeat_interleave(18)
    features = torch.randn(len(labels), 9, generator=generator)
    return features, labels


def test_m10_config_locks_controlled_not_official_scope():
    config = m10._read_config(CONFIG)
    assert config["uses_test_set"] is False
    assert config["accuracy_based_selection"] is False
    assert config["stream"]["online_iter"] == 1
    assert config["stream"]["mini_batch_size"] == 64
    assert config["gacl"]["gamma"] == 100.0
    assert config["gacl"]["expansion_dim"] == 5000
    assert config["gacl"]["scope"] == (
        "cached_vit_b_features_not_official_deit_checkpoint"
    )


def test_m10_rejects_test_use_and_silent_frequency_change(tmp_path):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["uses_test_set"] = True
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="train-only"):
        m10._read_config(path)
    config["uses_test_set"] = False
    config["stream"]["online_iter"] = 3
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Si-Blurry"):
        m10._read_config(path)


def test_fixed_si_blurry_port_is_deterministic_exact_and_generalized():
    features, labels = _small_data()
    del features
    allowed = torch.cat([torch.where(labels == class_id)[0][:10] for class_id in range(6)])
    first, first_metadata = m10.fixed_si_blurry_tasks(
        labels,
        allowed,
        num_tasks=3,
        disjoint_ratio_percent=50,
        blurry_sample_ratio_percent=10,
        seed=2025,
    )
    second, second_metadata = m10.fixed_si_blurry_tasks(
        labels,
        allowed,
        num_tasks=3,
        disjoint_ratio_percent=50,
        blurry_sample_ratio_percent=10,
        seed=2025,
    )
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    assert first_metadata == second_metadata
    assert first_metadata["partition_exact"] is True
    assert first_metadata["repeated_class_count"] > 0
    flattened = torch.cat(first)
    assert len(flattened) == len(allowed)
    assert set(flattened.tolist()) == set(allowed.tolist())


def test_m10_small_generalized_stream_passes_equivalence_and_integrity():
    config = _small_config()
    features, labels = _small_data()
    fit, validation = m10.development_subset_indices(
        labels,
        num_classes=6,
        fit_per_class=10,
        validation_per_class=3,
        seed=3035,
    )
    tasks, metadata = m10.fixed_si_blurry_tasks(
        labels,
        fit,
        num_tasks=3,
        disjoint_ratio_percent=50,
        blurry_sample_ratio_percent=10,
        seed=2025,
    )
    result = m10._run_stream(
        config=config,
        features=features,
        labels=labels,
        fit_indices=fit,
        validation_indices=validation,
        task_indices=tasks,
        stream_metadata=metadata,
        device=torch.device("cpu"),
    )
    assert all(result["gates"].values())
    assert result["summary"]["total_mini_batch_updates"] == sum(
        record["mini_batch_updates"] for record in result["records"]
    )
    assert result["summary"]["mixed_exposed_unexposed_mini_batches"] > 0
    assert result["summary"]["maximum_inverse_primal_weight_relative_error"] < 2e-4


def test_m10_quadratic_bytes_counts_dense_and_compressed_factor_storage():
    dense = m10.DenseSquareRootBackend(
        dimension=12,
        ridge_lambda=3.0,
        update_backend="blocked_qr",
        device=torch.device("cpu"),
    )
    compressed = m10.SquareRootBackend(
        dimension=12,
        ridge_lambda=3.0,
        storage_mode="int8",
        block_size=4,
        group_size=3,
        update_panel_size=5,
        quantization_batch_blocks=2,
        device=torch.device("cpu"),
    )
    generator = torch.Generator().manual_seed(91)
    codes = torch.randn(18, 12, generator=generator)
    labels = torch.arange(3).repeat(6)
    dense.update(codes, labels)
    compressed.update(codes, labels)

    assert m10._quadratic_bytes(dense) == 12 * 12 * 4
    compressed_tensors = compressed.persistent_tensors()
    expected = m10.persistent_tensor_bytes(
        {
            name: tensor
            for name, tensor in compressed_tensors.items()
            if name.startswith("factor.")
        }
    )
    assert expected > 0
    assert m10._quadratic_bytes(compressed) == expected


def test_m10_notebook_is_source_locked_train_only_and_compiles():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "--extract-train-only" in code
    assert "test.pt').exists()" in code
    assert "--require-clean-git" in code
    assert "m10_results.json" in code
    assert "files.download(archive)" in code
    assert "PASS_M10_GACL_CONTROLLED_TRAIN_ONLY" in code
    locked_paths = (
        "configs/srq_generalization_m10_gacl_train_only.json",
        "tools/srq_generalization_m10.py",
        "methods/frontends/gacl.py",
        "methods/frontends/__init__.py",
        "methods/analytic_ridge/backends.py",
        "tools/experiment_runner.py",
        "models/backbone.py",
        "utils/data_utils.py",
        "utils/train_utils.py",
    )
    for relative_path in locked_paths:
        # Source locks are portable across Windows and Linux checkouts. Git's
        # canonical content uses LF, whereas a Windows working tree may expose
        # CRLF or mixed endings without changing the committed source.
        canonical = (ROOT / relative_path).read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha256(canonical).hexdigest()
        assert f"'{relative_path}':'{digest}'" in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")
