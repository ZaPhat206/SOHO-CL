import copy
import json
from pathlib import Path

import pytest
import torch

from tools import srq_generalization_m20 as m20


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/srq_generalization_m20_adaptive_budget_criterion_train_only.json"


def _config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_locked_config_is_valid_and_train_only():
    config = m20.read_config(CONFIG)
    assert config["uses_test_set"] is False
    assert config["accuracy_based_selection"] is False
    assert config["gates"]["accuracy_gate"] is None
    assert config["streams"][0]["m6_identity_locked"] is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.__setitem__("seed", 2026),
        lambda c: c.__setitem__("uses_test_set", True),
        lambda c: c.__setitem__("accuracy_based_selection", True),
        lambda c: c["budget_sweep"].__setitem__("factor_mse_budgets", [0.0, 0.5, 0.1, 0.25]),
        lambda c: c["budget_sweep"].__setitem__("factor_mse_budgets", [0.0, 0.1, 0.5]),
        lambda c: c["budget_sweep"].__setitem__("row_weighted_system_budgets", [0.05, 1.5]),
        lambda c: c["gates"].__setitem__("accuracy_gate", 0.25),
        lambda c: c["streams"].reverse(),
        lambda c: c["p2b"].__setitem__("update_panel_size", 1024),
        lambda c: c.__setitem__("width", 10000),
        lambda c: c["interpretation"].__setitem__("criterion_comparison_budgets", [0.0]),
        lambda c: c["scoring_benchmark"].__setitem__("criterion", "row_weighted_system"),
        lambda c: c.__setitem__("extra_key", 1),
    ],
)
def test_config_mutations_are_rejected(mutate):
    config = _config()
    mutate(config)
    with pytest.raises(ValueError):
        m20.validate_config(config)


def test_unit_plan_orders_exact_first_and_includes_one_benchmark():
    config = m20.read_config(CONFIG)
    plan = m20.unit_plan(config)
    ids = [item["unit_id"] for item in plan]
    assert len(ids) == len(set(ids)) == 3 * (2 + 6 + 4) + 1
    for stream in config["streams"]:
        sid = stream["stream_id"]
        stream_ids = [item["unit_id"] for item in plan if item["stream_id"] == sid]
        assert stream_ids[0] == f"{sid}__exact"
    benchmarks = [item for item in plan if item["method"] == "adaptive_criterion_benchmark"]
    assert len(benchmarks) == 1 and benchmarks[0]["stream_id"] == "s2025"
    assert "s2025__factor_mse_b0.25" in ids and "s2027__row_weighted_system_b0.10" in ids


def test_train_only_guard_refuses_test_cache(tmp_path):
    m20.assert_train_only_cache(tmp_path)
    (tmp_path / "test.pt").write_bytes(b"x")
    with pytest.raises(RuntimeError):
        m20.assert_train_only_cache(tmp_path)


# ---------------------------------------------------------------------------
# a tiny end-to-end CPU pipeline through the real unit runner
# ---------------------------------------------------------------------------
def _tiny_config():
    config = _config()
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
    config["budget_sweep"]["factor_mse_budgets"] = [0.0, 0.1, 0.25, 1.0]
    config["budget_sweep"]["row_weighted_system_budgets"] = [0.1, 0.25]
    config["interpretation"]["criterion_comparison_budgets"] = [0.1]
    config["scoring_benchmark"]["batch_blocks"] = 2
    config["streams"] = [
        {"stream_id": "a", "class_order_seed": 2025, "split_seed": 2025,
         "projection_seed": 2025, "m6_identity_locked": True},
        {"stream_id": "b", "class_order_seed": 7, "split_seed": 7,
         "projection_seed": 7, "m6_identity_locked": False},
    ]
    config["scoring_benchmark"]["stream_id"] = "a"
    return config


def _tiny_train():
    generator = torch.Generator().manual_seed(0)
    labels = torch.arange(4).repeat_interleave(30)
    means = torch.randn(4, 6, generator=generator) * 2.0
    features = means[labels] + torch.randn(len(labels), 6, generator=generator)
    return {"features": features, "labels": labels}


def _tiny_source(config, prepared, exact_unit, p2b_unit):
    from tools import srq_generalization_m4 as m4
    from tools import srq_generalization_m6 as m6
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
                "exact": exact_unit["validation_aia_percent"] if exact_unit else 0.0,
                "p2b_int8": p2b_unit["validation_aia_percent"] if p2b_unit else 0.0,
            },
            "final_validation_accuracy_percent": {
                "exact": exact_unit["final_validation_accuracy_percent"] if exact_unit else 0.0,
                "p2b_int8": p2b_unit["final_validation_accuracy_percent"] if p2b_unit else 0.0,
            },
            "final_total_persistent_bytes": {
                "exact": exact_unit["final_total_persistent_bytes"] if exact_unit else 0,
                "p2b_int8": p2b_unit["final_total_persistent_bytes"] if p2b_unit else 0,
            },
        }],
    }


def test_tiny_pipeline_runs_every_unit_and_passes_structural_gates(tmp_path):
    config = _tiny_config()
    train = _tiny_train()
    device = torch.device("cpu")
    stream = config["streams"][0]
    prepared = m20.prepare_stream(config, stream, train["labels"], device)
    placeholder = _tiny_source(config, prepared, None, None)

    units = []
    for item in m20.unit_plan(config):
        units.append(
            m20.run_unit(config, item, train=train, source=placeholder,
                         output_dir=tmp_path, device=device)
        )
    by_id = {unit["unit_id"]: unit for unit in units}
    source = _tiny_source(config, prepared, by_id["a__exact"], by_id["a__p2b_int8"])
    result = m20.summarize(config, units, source=source, config_path=None)

    gates = result["gates"]
    assert gates["all_units_complete"]
    assert gates["m6_identity_for_locked_stream"]
    assert gates["exact_and_p2b_reproduce_m6"]
    assert gates["budget_conformance"]
    assert gates["state_monotone_in_budget"]
    assert gates["criterion_backend_fidelity"]
    assert gates["batched_scoring_decision_agreement"]
    assert gates["solver_residual_within_tolerance"]
    assert result["criterion_backend_fidelity"]["weights_bitwise_identical"]
    assert result["status"] == m20.STATUS_PASS

    exact = by_id["a__exact"]
    assert all(r["relative_classifier_error_vs_exact"] == 0.0 for r in exact["records"])
    full = by_id["a__factor_mse_b1.00"]["records"][-1]
    none = by_id["a__factor_mse_b0.00"]["records"][-1]
    assert none["selected_fp16_blocks"] == 0
    assert full["relative_classifier_error_vs_exact"] <= none["relative_classifier_error_vs_exact"]
    assert (tmp_path / "cache" / "a_exact_weights.pt").is_file()
    assert set(result["criterion_comparison"]) == {"0.10"}
    m20.write_curve_csv(result, tmp_path / "curve.csv")
    assert "factor_mse_b0.25" in (tmp_path / "curve.csv").read_text()


def test_non_exact_unit_requires_exact_cache(tmp_path):
    config = _tiny_config()
    train = _tiny_train()
    item = next(u for u in m20.unit_plan(config) if u["unit_id"] == "b__p2b_int8")
    with pytest.raises(RuntimeError):
        m20.run_unit(config, item, train=train, source={}, output_dir=tmp_path,
                     device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# interpretation rules on synthetic unit summaries
# ---------------------------------------------------------------------------
def _fake_units(config, aia, werr, *, fidelity_ok=True, agreement_ok=True):
    units = []
    for item in m20.unit_plan(config):
        key = item["unit_id"].split("__", 1)[1]
        record = {
            "task": 1,
            "validation_accuracy_percent": aia[key],
            "total_persistent_bytes": int(1000 + 1000 * (item["budget"] or 0)),
            "factor_persistent_bytes": 10,
            "solver_relative_residual": 1e-7,
            "update_seconds": 1.0,
            "class_ids": [0],
            "weights_sha256": "w",
            "relative_classifier_error_vs_exact": werr[key],
        }
        if item["method"].startswith("adaptive"):
            record.update({
                "precision_mask_sha256": "m", "factor_budget_ceiling_bytes": 20,
                "extra_budget_bytes": 5, "used_extra_bytes": 5,
                "selected_fp16_blocks": 1, "total_blocks": 2,
            })
        unit = {
            "status": "complete", **item, "stream": {},
            "m6_identity_checks": {"x": True} if item["stream_id"] == config["streams"][0]["stream_id"] else None,
            "records": [record],
            "validation_aia_percent": aia[key],
            "final_validation_accuracy_percent": aia[key],
            "final_total_persistent_bytes": record["total_persistent_bytes"],
            "analytic_update_seconds": 1.0,
            "maximum_solver_relative_residual": 1e-7,
            "scoring_benchmark": None,
            "environment": {"device_name": "Tesla T4"},
        }
        if item["method"] == "adaptive_criterion_benchmark":
            unit["scoring_benchmark"] = [{
                "task": 1, "per_block_scoring_seconds": 2.0, "batched_scoring_seconds": 0.5,
                "costs_identical": True, "selections_identical": agreement_ok,
                "differing_selected_blocks": 0 if agreement_ok else 1,
                "maximum_relative_benefit_difference": 0.0,
            }]
            if not fidelity_ok:
                record["precision_mask_sha256"] = "different"
        units.append(unit)
    return units


def _fake_source(config, aia):
    width = config["width"]
    return {"width_results": [{
        "width": width,
        "validation_aia_percent": {"exact": aia["exact"], "p2b_int8": aia["p2b_int8"]},
        "final_validation_accuracy_percent": {"exact": aia["exact"], "p2b_int8": aia["p2b_int8"]},
        "final_total_persistent_bytes": {"exact": 1000, "p2b_int8": 1000},
    }]}


def _tables(config, *, rws_better):
    aia = {"exact": 90.0, "p2b_int8": 89.7}
    werr = {"exact": 0.0, "p2b_int8": 0.05}
    for budget in config["budget_sweep"]["factor_mse_budgets"]:
        key = f"factor_mse_b{budget:.2f}"
        aia[key] = 90.0 - (0.3 if budget == 0.0 else 0.05 if budget < 0.25 else 0.0)
        werr[key] = 0.05 / (1 + 10 * budget)
    for budget in config["budget_sweep"]["row_weighted_system_budgets"]:
        key = f"row_weighted_system_b{budget:.2f}"
        mse = f"factor_mse_b{budget:.2f}"
        aia[key] = aia[mse] + (0.01 if rws_better else -0.01)
        werr[key] = werr[mse] * (0.5 if rws_better else 1.5)
    aia["benchmark_factor_mse_b0.25"] = aia["factor_mse_b0.25"]
    werr["benchmark_factor_mse_b0.25"] = werr["factor_mse_b0.25"]
    return aia, werr


def test_summary_rules_sufficient_budget_and_supported_criterion():
    config = m20.read_config(CONFIG)
    aia, werr = _tables(config, rws_better=True)
    result = m20.summarize(config, _fake_units(config, aia, werr), source=_fake_source(config, aia))
    assert result["gates"]["all_units_complete"]
    assert result["smallest_sufficient_factor_mse_budget"] == 0.25
    assert result["row_weighted_criterion_supported"] is True
    assert result["scoring_benchmark"]["median_speedup"] == 4.0
    assert result["status"] == m20.STATUS_PASS


def test_summary_rules_reject_worse_criterion_and_flag_fidelity():
    config = m20.read_config(CONFIG)
    aia, werr = _tables(config, rws_better=False)
    result = m20.summarize(
        config, _fake_units(config, aia, werr, fidelity_ok=False, agreement_ok=False),
        source=_fake_source(config, aia),
    )
    assert result["row_weighted_criterion_supported"] is False
    assert result["gates"]["criterion_backend_fidelity"] is False
    assert result["gates"]["batched_scoring_decision_agreement"] is False
    assert result["status"] == m20.STATUS_WARNING
