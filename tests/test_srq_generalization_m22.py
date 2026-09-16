import copy
import json
from pathlib import Path

import pytest
import torch

from methods.analytic_ridge.selection_controls import (
    SelectionControlSquareRootBackend,
    block_layout,
    random_select,
)
from tools import srq_generalization_m4 as m4
from tools import srq_generalization_m6 as m6
from tools import srq_generalization_m22 as m22


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/srq_generalization_m22_selection_controls_train_only.json"


def _config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_locked_config_is_valid_train_only_and_has_48_units():
    config = m22.read_config(CONFIG)
    assert config["uses_test_set"] is False
    assert config["accuracy_based_selection"] is False
    assert config["gates"]["accuracy_gate"] is None
    plan = m22.unit_plan(config)
    assert len(plan) == len({item["unit_id"] for item in plan}) == 48
    assert sum(item["method"] == "exact" for item in plan) == 3
    assert sum(item["method"] == "p2b_int8" for item in plan) == 3
    assert sum(item["method"] == "adaptive" for item in plan) == 15
    assert sum(item["method"] == "static" for item in plan) == 9
    assert sum(item["method"] == "random" for item in plan) == 18


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.__setitem__("uses_test_set", True),
        lambda c: c.__setitem__("accuracy_based_selection", True),
        lambda c: c.__setitem__("seed", 2026),
        lambda c: c.__setitem__("width", 10000),
        lambda c: c["selection_controls"].__setitem__("comparison_budgets", [0.05]),
        lambda c: c["selection_controls"].__setitem__("random_allocation_seeds", [1, 2]),
        lambda c: c["gates"].__setitem__("accuracy_gate", 0.1),
        lambda c: c["streams"].reverse(),
        lambda c: c.__setitem__("unexpected", 1),
    ],
)
def test_locked_config_mutations_are_rejected(mutate):
    config = _config()
    mutate(config)
    with pytest.raises(ValueError):
        m22.validate_config(config)


def test_train_only_guard_rejects_test_cache(tmp_path):
    from tools import srq_generalization_m20 as m20

    m20.assert_train_only_cache(tmp_path)
    (tmp_path / "test.pt").write_bytes(b"forbidden")
    with pytest.raises(RuntimeError):
        m20.assert_train_only_cache(tmp_path)


def test_random_selector_is_deterministic_budgeted_and_seed_sensitive():
    _, _, costs = block_layout(80, block_size=16, group_size=8)
    first, used_first = random_select(costs, extra_budget=1200, seed=202501)
    repeat, used_repeat = random_select(costs, extra_budget=1200, seed=202501)
    second, used_second = random_select(costs, extra_budget=1200, seed=202502)
    assert first == repeat and used_first == used_repeat <= 1200
    assert used_second <= 1200
    assert first != second


def _backend(policy, seed=2025):
    return SelectionControlSquareRootBackend(
        dimension=64,
        ridge_lambda=2.0,
        device="cpu",
        statistics_dtype=torch.float32,
        solver_dtype=torch.float32,
        selection_policy=policy,
        allocation_seed=seed,
        adaptive_budget_fraction=0.5,
        block_size=16,
        group_size=8,
        update_panel_size=8,
        update_trailing_chunk_size=None,
        first_update_backend="gram_cholesky",
        quantization_backend="streaming",
        quantization_batch_blocks=4,
    )


def test_static_reuses_mask_while_random_control_is_reproducible():
    generator = torch.Generator().manual_seed(42)
    x1, x2 = torch.randn(30, 64, generator=generator), torch.randn(30, 64, generator=generator)
    y1, y2 = torch.arange(3).repeat_interleave(10), torch.arange(3, 6).repeat_interleave(10)
    static = _backend("static")
    static.update(x1, y1)
    first_mask = static.factor.precision_mask.clone()
    static.update(x2, y2)
    assert torch.equal(static.factor.precision_mask, first_mask)
    assert static.diagnostics["mask_source"] == "task1_factor_mse_mask"

    random_a, random_b = _backend("random", 9), _backend("random", 9)
    for backend in (random_a, random_b):
        backend.update(x1, y1)
        backend.update(x2, y2)
    assert torch.equal(random_a.factor.precision_mask, random_b.factor.precision_mask)
    assert random_a.diagnostics["used_extra_bytes"] <= random_a.diagnostics["extra_budget_bytes"]


def _tiny_config():
    config = copy.deepcopy(_config())
    config["num_classes"] = 4
    config["num_tasks"] = 2
    config["width"] = 48
    config["ridge_lambda"] = 1.0
    config["required_device_name_substring"] = ""
    config["ranpac"]["feature_dimension"] = 6
    config["ranpac"]["maximum_expand_dimension"] = 48
    config["ranpac"]["encode_batch_size"] = 7
    config["ranpac"]["evaluation_batch_size"] = 5
    config["p2b"]["block_size"] = 16
    config["p2b"]["group_size"] = 8
    config["p2b"]["update_panel_size"] = 8
    config["p2b"]["quantization_batch_blocks"] = 3
    config["selection_controls"]["comparison_budgets"] = [0.5]
    config["selection_controls"]["adaptive_connector_budgets"] = []
    config["selection_controls"]["random_allocation_seeds"] = [11, 12]
    config["streams"] = [{
        "stream_id": "s2025", "class_order_seed": 2025, "split_seed": 2025,
        "projection_seed": 2025, "m6_identity_locked": True,
    }]
    return config


def _tiny_train():
    generator = torch.Generator().manual_seed(0)
    labels = torch.arange(4).repeat_interleave(30)
    means = torch.randn(4, 6, generator=generator) * 2
    features = means[labels] + torch.randn(len(labels), 6, generator=generator)
    return {"features": features, "labels": labels}


def _source(config, prepared, exact=None, p2b=None):
    width = config["width"]
    return {
        "provenance": {
            "class_order": prepared["class_order"],
            "training_indices_sha256": m6._sequence_sha256(prepared["training_parts"]),
            "outer_validation_indices_sha256": m6._sequence_sha256(prepared["validation_parts"]),
            "full_projection_sha256": m4._tensor_content_sha256(prepared["full_projection"]),
            "projection_prefix_sha256": {
                str(width): m4._tensor_content_sha256(prepared["full_projection"][:, :width].contiguous())
            },
        },
        "width_results": [{
            "width": width,
            "validation_aia_percent": {
                "exact": 0 if exact is None else exact["validation_aia_percent"],
                "p2b_int8": 0 if p2b is None else p2b["validation_aia_percent"],
            },
            "final_validation_accuracy_percent": {
                "exact": 0 if exact is None else exact["final_validation_accuracy_percent"],
                "p2b_int8": 0 if p2b is None else p2b["final_validation_accuracy_percent"],
            },
            "final_total_persistent_bytes": {
                "exact": 0 if exact is None else exact["final_total_persistent_bytes"],
                "p2b_int8": 0 if p2b is None else p2b["final_total_persistent_bytes"],
            },
        }],
    }


def test_tiny_end_to_end_pipeline_passes_all_structural_gates(tmp_path):
    config, train = _tiny_config(), _tiny_train()
    device = torch.device("cpu")
    prepared = m22.m20.prepare_stream(config, config["streams"][0], train["labels"], device)
    placeholder = _source(config, prepared)
    units = []
    for item in m22.unit_plan(config):
        units.append(m22.run_unit(
            config, item, train=train, source=placeholder, output_dir=tmp_path, device=device
        ))
    by_id = {unit["unit_id"]: unit for unit in units}
    source = _source(config, prepared, by_id["s2025__exact"], by_id["s2025__p2b_int8"])
    result = m22.summarize(config, units, source=source)
    assert result["status"] == m22.STATUS_PASS
    assert all(result["gates"].values())
    static = by_id["s2025__static_b0.50"]
    adaptive = by_id["s2025__adaptive_b0.50"]
    assert static["records"][0]["precision_mask_sha256"] == adaptive["records"][0]["precision_mask_sha256"]
    assert len({row["precision_mask_sha256"] for row in static["records"]}) == 1
    assert "relative_logit_frobenius_error_vs_exact" in static["records"][-1]
    assert result["maximum_actual_used_byte_gap_under_equal_ceiling"] >= 0


def test_non_exact_unit_requires_exact_cache(tmp_path):
    config = _tiny_config()
    train = _tiny_train()
    prepared = m22.m20.prepare_stream(
        config, config["streams"][0], train["labels"], torch.device("cpu")
    )
    item = next(row for row in m22.unit_plan(config) if row["method"] == "static")
    with pytest.raises(RuntimeError):
        m22.run_unit(
            config, item, train=train, source=_source(config, prepared), output_dir=tmp_path,
            device=torch.device("cpu"),
        )
