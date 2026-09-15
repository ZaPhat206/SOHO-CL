import copy
import json
from pathlib import Path

import pytest
import torch

from methods.analytic_ridge import ExactGramBackend, SquareRootBackend
from tools import srq_fly_priority3_direct_control as p3
from tools import srq_generalization_m4 as m4
from tools import srq_generalization_m5 as m5
from tools import srq_generalization_m6 as m6
from tools import srq_generalization_m21 as m21
from tools.experiment_runner import split, train_validation_indices


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/srq_generalization_m21_accumulation_gram_load_train_only.json"


def _config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_locked_config_is_valid_train_only_and_pins_priority3():
    config = m21.read_config(CONFIG)
    assert config["uses_test_set"] is False and config["accuracy_based_selection"] is False
    assert config["gates"]["accuracy_gate"] is None
    p3_config = m21.read_priority3_config(config)
    assert p3_config["representation"]["expand_dim"] == 10000
    assert p3_config["fly_ridge_lambda"] == 1.0e6


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.__setitem__("seed", 2026),
        lambda c: c.__setitem__("uses_test_set", True),
        lambda c: c.__setitem__("accuracy_based_selection", True),
        lambda c: c["gates"].__setitem__("accuracy_gate", 0.1),
        lambda c: c["gram_load"]["methods"].reverse(),
        lambda c: c["gram_load"]["minimal_load"].__setitem__("uses_labels_or_accuracy", True),
        lambda c: c["gram_load"]["minimal_load"].__setitem__("load_margin_multiplier", 4.0),
        lambda c: c["gram_load"]["interpretation"].__setitem__("practical_equivalence_pp", 0.6),
        lambda c: c["accumulation"]["streams"].reverse(),
        lambda c: c["accumulation"]["p2b"].__setitem__("quantization_backend", "eager"),
        lambda c: c["accumulation"]["reference"].pop("acc_w10000_s2025"),
        lambda c: c["accumulation"]["interpretation"].__setitem__("dominant_minimum_ratio", 1.05),
        lambda c: c["accumulation"]["interpretation"].__setitem__("primary_width", 15000),
        lambda c: c["accumulation"].__setitem__("ridge_lambda", 1.0e5),
        lambda c: c.__setitem__("extra_key", 1),
    ],
)
def test_config_mutations_are_rejected(mutate):
    config = _config()
    mutate(config)
    with pytest.raises(ValueError):
        m21.validate_config(config)


def test_unit_plan_runs_fast_gram_units_first():
    plan = m21.unit_plan(m21.read_config(CONFIG))
    ids = [item["unit_id"] for item in plan]
    assert ids == [
        "gram_exact_fly_10000", "gram_srq_int8_p2b", "gram_direct_int8_gram_weyl_repair",
        "gram_direct_int8_gram_minimal_load", "acc_w20000_s2025", "acc_w20000_s2026",
        "acc_w20000_s2027", "acc_w10000_s2025",
    ]


def test_relative_error_has_no_floor_on_the_reference_norm():
    reference = torch.full((3, 2), 0.001)
    assert m21.relative_error(reference * 1.5, reference) == pytest.approx(0.5)
    assert m21.relative_error(torch.zeros(2), torch.zeros(2)) == 0.0


# ---------------------------------------------------------------------------
# accumulation: a tiny end-to-end CPU unit
# ---------------------------------------------------------------------------
def _tiny_config():
    config = _config()
    config["num_classes"] = 4
    config["num_tasks"] = 2
    config["required_device_name_substring"] = ""
    accumulation = config["accumulation"]
    accumulation["ridge_lambda"] = 1.0
    accumulation["ranpac"]["feature_dimension"] = 6
    accumulation["ranpac"]["maximum_expand_dimension"] = 40
    accumulation["ranpac"]["encode_batch_size"] = 7
    accumulation["ranpac"]["evaluation_batch_size"] = 5
    accumulation["p2b"].update(block_size=16, group_size=8, update_panel_size=8,
                               quantization_batch_blocks=3)
    accumulation["system_probe_count"] = 4
    accumulation["streams"] = [
        {"stream_id": "a", "class_order_seed": 2025, "split_seed": 2025,
         "projection_seed": 2025, "m6_identity_locked": True},
        {"stream_id": "b", "class_order_seed": 7, "split_seed": 7,
         "projection_seed": 7, "m6_identity_locked": False},
    ]
    accumulation["units"] = [
        {"unit_id": "acc_w40_a", "width": 40, "stream_id": "a"},
        {"unit_id": "acc_w40_b", "width": 40, "stream_id": "b"},
        {"unit_id": "acc_w24_a", "width": 24, "stream_id": "a"},
    ]
    placeholder = {
        "source": "test", "exact_aia_percent": 0.0, "exact_final_percent": 0.0, "exact_bytes": 0,
        "p2b_int8_aia_percent": 0.0, "p2b_int8_final_percent": 0.0, "p2b_int8_bytes": 0,
        "m7_p2b_int8_relative_logit_error": None,
    }
    accumulation["reference"] = {unit["unit_id"]: dict(placeholder) for unit in accumulation["units"]}
    accumulation["interpretation"]["primary_width"] = 40
    return config


def _tiny_train():
    generator = torch.Generator().manual_seed(0)
    labels = torch.arange(4).repeat_interleave(30)
    means = torch.randn(4, 6, generator=generator) * 2.0
    features = means[labels] + torch.randn(len(labels), 6, generator=generator)
    return {"features": features, "labels": labels}


def _tiny_source(config, train):
    stream = config["accumulation"]["streams"][0]
    prepared = m21.prepare_ranpac_stream(config, stream, train["labels"], 40, torch.device("cpu"))
    return {"provenance": {
        "class_order": prepared["class_order"],
        "training_indices_sha256": m6._sequence_sha256(prepared["training_parts"]),
        "outer_validation_indices_sha256": m6._sequence_sha256(prepared["validation_parts"]),
        "full_projection_sha256": m4._tensor_content_sha256(prepared["full_projection"]),
        "projection_prefix_sha256": {
            str(width): m4._tensor_content_sha256(prepared["full_projection"][:, :width].contiguous())
            for width in (24, 40)
        },
    }}


def _standalone(config, train, width, stream_id, mode):
    """The M20-style single-backend run the recursive arm must reproduce."""
    accumulation = config["accumulation"]
    device = torch.device("cpu")
    stream = m21._stream_by_id(config, stream_id)
    prepared = m21.prepare_ranpac_stream(config, stream, train["labels"], width, device)
    projection = prepared["projection"]
    encoder = lambda values: torch.relu(values.to(dtype=torch.float32) @ projection)
    common = m5._common_backend(width, accumulation["ridge_lambda"], device)
    backend = (
        ExactGramBackend(**common) if mode == "exact"
        else SquareRootBackend(storage_mode="int8", **common, **m21.m20._p2b_kwargs({"p2b": accumulation["p2b"]}))
    )
    accuracies, weights = [], []
    for task_id, indices in enumerate(prepared["training_parts"]):
        codes = m5._encode_indices(encoder, train["features"], indices, accumulation["ranpac"]["encode_batch_size"])
        backend.update(codes, train["labels"][indices])
        seen = torch.cat(prepared["validation_parts"][: task_id + 1])
        accuracies.append(m5._evaluate(
            encoder=encoder, backends={"b": backend}, features=train["features"],
            labels=train["labels"], indices=seen, batch_size=accumulation["ranpac"]["evaluation_batch_size"],
        )["b"])
        weights.append(backend.weights.clone())
    return accuracies, weights


def test_tiny_accumulation_unit_reproduces_standalone_backends(tmp_path):
    config = _tiny_config()
    train = _tiny_train()
    source = _tiny_source(config, train)
    unit = next(u for u in m21.unit_plan(config) if u["unit_id"] == "acc_w40_a")
    result = m21.run_accumulation_unit(config, unit, train=train, source=source,
                                       output_dir=tmp_path, device=torch.device("cpu"))
    assert result["status"] == "complete" and all(result["m6_identity_checks"].values())
    assert (tmp_path / "units" / "acc_w40_a.json").is_file()
    records = result["records"]
    assert len(records) == config["num_tasks"]
    assert records[0]["task1_one_shot_minus_recursive_relative_weight_difference"] == 0.0
    assert records[0]["one_shot_int8"]["relative_logit_error"] == records[0]["recursive_int8"]["relative_logit_error"]
    assert records[1]["task1_one_shot_minus_recursive_relative_weight_difference"] is None
    for name in ("exact", "recursive_int8"):
        mode = "exact" if name == "exact" else "int8"
        accuracies, _ = _standalone(config, train, 40, "a", mode)
        assert [record[name]["accuracy_percent"] for record in records] == accuracies
    for record in records:
        for arm in ("recursive_int8", "one_shot_int8"):
            assert 0.0 <= record[arm]["prediction_agreement"] <= 1.0
            assert record[arm]["relative_system_action_error"] >= 0.0


def test_one_shot_quantizes_the_exact_factor_of_the_current_system():
    config = _tiny_config()
    generator = torch.Generator().manual_seed(3)
    codes = torch.relu(torch.randn(50, 40, generator=generator))
    gram = codes.T @ codes
    cross = codes.T @ torch.nn.functional.one_hot(torch.arange(50) % 3, 3).float()
    factor, weights, local_error, residual = m21.one_shot_int8(
        gram, cross, ridge=1.0, p2b=config["accumulation"]["p2b"]
    )
    system = gram.clone()
    system.diagonal().add_(1.0)
    exact = torch.linalg.solve(system.double(), cross.double()).float()
    assert torch.equal(factor, torch.triu(factor)) and bool((factor.diagonal() > 0).all())
    assert 0.0 < local_error < 0.05 and residual < 1e-4
    assert m21.relative_error(weights, exact) < 0.5


# ---------------------------------------------------------------------------
# gram_load: tiny synthetic FLY stream through the real Priority-3 loops
# ---------------------------------------------------------------------------
def _tiny_fly(config, directory):
    p3_config = json.loads((ROOT / config["gram_load"]["priority3_config"]).read_text(encoding="utf-8"))
    p3_config.update(num_classes=4, num_tasks=2, feature_dim=6, fly_ridge_lambda=1e-3)
    p3_config["representation"].update(expand_dim=24, synaptic_degree=3, coding_level=0.25,
                                       encode_batch_size=5, evaluation_batch_size=5)
    p3_config["storage"].update(block_size=8, group_size=4)
    path = directory / "tiny_priority3.json"
    path.write_text(json.dumps(p3_config), encoding="utf-8")
    return p3._read_config(path)


def _tiny_code_cache(p3_config, train):
    from methods.srq_fly_optimized.minimal_load_control import MinimalLoadDirectInt8GramLearner
    kwargs = p3._common_kwargs(p3_config, 6, None, torch.device("cpu"))
    encoder = MinimalLoadDirectInt8GramLearner(**kwargs)
    codes = encoder.encode(train["features"])
    active = max(1, int(p3_config["representation"]["expand_dim"] * p3_config["representation"]["coding_level"]))
    values, indices = codes.topk(active, dim=1)
    return indices, values, None, encoder.flyhash.projection_matrix


def test_tiny_gram_load_arms_run_and_summarize(tmp_path):
    config = _config()
    config["num_classes"], config["num_tasks"] = 4, 2
    config["required_device_name_substring"] = ""
    p3_config = _tiny_fly(config, tmp_path)
    train = _tiny_train()
    class_order = list(range(4))
    task_indices = split(train["labels"], class_order, 2)
    training_parts, validation_parts = train_validation_indices(train["labels"], task_indices, 2025, 0.2)
    code_cache = _tiny_code_cache(p3_config, train)
    stream = (train, class_order, training_parts, validation_parts, code_cache)
    units = []
    for item in m21.unit_plan(config)[:4]:
        units.append(m21.run_gram_load_unit(config, item, p3_config=p3_config, stream=stream,
                                            output_dir=tmp_path, device=torch.device("cpu")))
    by_id = {unit["unit_id"]: unit for unit in units}
    minimal = by_id["gram_direct_int8_gram_minimal_load"]["result"]
    assert minimal["status"] == "complete" and len(minimal["task_diagnostics"]) == 2
    assert all(row["diagonal_loading"] >= 0.0 for row in minimal["task_diagnostics"])
    summary = m21.summarize_gram_load(config, by_id)
    assert summary["minimal_load_search_consistent"] is True
    assert summary["verdict"] is not None
    assert len(summary["observations"]["weyl_loading_to_ridge_ratio_by_task"]) == 2


# ---------------------------------------------------------------------------
# interpretation rules on synthetic summaries
# ---------------------------------------------------------------------------
def _fake_accumulation_unit(unit, *, rec_err, one_err, rec_aia, one_aia):
    def arm(error, accuracy):
        return {"accuracy_percent": accuracy, "relative_logit_error": error,
                "relative_system_action_error": error / 10, "relative_classifier_error": 1.2,
                "relative_local_factor_error": 0.01, "prediction_agreement": 0.99,
                "margin_certified_fraction": 0.8, "solver_relative_residual": 1e-7}
    records = []
    for task in (1, 2):
        records.append({
            "task": task,
            "exact": {"accuracy_percent": 95.0, "solver_relative_residual": 1e-7, "total_persistent_bytes": 10},
            # Both arms coincide at the first task; only the recursive error grows.
            "recursive_int8": {**arm(one_err if task == 1 else rec_err, rec_aia), "total_persistent_bytes": 5},
            "one_shot_int8": arm(one_err, one_aia),
            "task1_one_shot_minus_recursive_relative_weight_difference": 0.0 if task == 1 else None,
        })
    return {
        "status": "complete", "unit_id": unit["unit_id"], "part": "accumulation",
        "width": unit["width"], "stream_id": unit["stream_id"], "records": records,
        "m6_identity_checks": {"x": True},
        "validation_aia_percent": {"exact": 95.0, "recursive_int8": rec_aia, "one_shot_int8": one_aia},
        "final_validation_accuracy_percent": {"exact": 95.0, "recursive_int8": rec_aia, "one_shot_int8": one_aia},
        "final_total_persistent_bytes": {"exact": 10, "recursive_int8": 5},
        "environment": {"device_name": "Tesla T4"},
    }


@pytest.mark.parametrize(
    "rec_err, expected",
    [
        (0.6, "ACCUMULATION_DOMINATES_FINAL_LOGIT_ERROR"),
        (0.25, "ACCUMULATION_CONTRIBUTES_TO_FINAL_LOGIT_ERROR"),
        (0.21, "ACCUMULATION_NOT_SUPPORTED_BY_FINAL_LOGIT_ERROR"),
    ],
)
def test_accumulation_verdict_follows_the_preregistered_ratio(rec_err, expected):
    config = _tiny_config()
    units = [
        _fake_accumulation_unit(item, rec_err=rec_err, one_err=0.2, rec_aia=94.0, one_aia=94.5)
        for item in m21.unit_plan(config) if item["part"] == "accumulation"
    ]
    by_id = {unit["unit_id"]: unit for unit in units}
    summary = m21.summarize_accumulation(config, by_id, units)
    assert summary["logit_error_verdict"] == expected
    assert summary["accuracy_verdict"] == "ONE_SHOT_HIGHER_AIA_ON_EVERY_STREAM"
    assert summary["task1_one_shot_equals_recursive"] is True
    assert summary["primary"]["final_logit_error_ratio"]["n"] == 2


def _fake_gram_units(aia):
    units = {}
    for method in m21.GRAM_LOAD_METHODS:
        rows = [{"diagonal_loading": 5e6, "loading_to_ridge_ratio": 5.0, "cholesky_attempts": 9}] * 10
        units[f"gram_{method}"] = {
            "unit_id": f"gram_{method}", "status": "complete", "ridge_lambda": 1e6,
            "environment": {"device_name": "Tesla T4"},
            "result": {"status": "complete", "validation_average_accuracy": aia[method],
                       "maximum_solver_relative_residual": 1e-6, "persistent_state_bytes": 1,
                       "total_update_seconds": 1.0, "task_diagnostics": rows},
        }
    return units


@pytest.mark.parametrize(
    "minimal_aia, expected",
    [
        (91.5, "SQUARE_ROOT_ADVANTAGE_MATERIAL_AGAINST_MINIMAL_LOAD"),
        (92.0, "SQUARE_ROOT_ADVANTAGE_MODEST_AGAINST_MINIMAL_LOAD"),
        (92.25, "MINIMAL_LOAD_PRACTICALLY_EQUIVALENT_TO_SQUARE_ROOT"),
        (92.5, "MINIMAL_LOAD_OUTPERFORMS_SQUARE_ROOT"),
    ],
)
def test_gram_load_verdict_and_priority3_reproduction(minimal_aia, expected):
    config = m21.read_config(CONFIG)
    reference = config["gram_load"]["priority3_reference"]["validation_average_accuracy"]
    aia = {**reference, "direct_int8_gram_minimal_load": minimal_aia}
    summary = m21.summarize_gram_load(config, _fake_gram_units(aia))
    assert summary["verdict"] == expected
    assert summary["reproduces_priority3"] is True
    shifted = copy.deepcopy(aia)
    shifted["srq_int8_p2b"] += 0.01
    assert m21.summarize_gram_load(config, _fake_gram_units(shifted))["reproduces_priority3"] is False
