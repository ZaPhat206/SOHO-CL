import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/srq_generalization_m22_selection_controls_colab.ipynb"
SOURCE_COMMIT = "92374f7703acd0640ab37fab1e3bf06dfd865840"
LOCKED = (
    "configs/srq_generalization_m22_selection_controls_train_only.json",
    "tools/srq_generalization_m22.py",
    "methods/analytic_ridge/selection_controls.py",
    "tests/test_srq_generalization_m22.py",
    "docs/research/SRQ_GENERALIZATION_M22_PROTOCOL.md",
    "methods/analytic_ridge/adaptive_selection.py",
    "methods/analytic_ridge/adaptive_upper.py",
    "methods/analytic_ridge/backends.py",
    "methods/analytic_ridge/qr.py",
    "tools/srq_generalization_m4.py",
    "tools/srq_generalization_m5.py",
    "tools/srq_generalization_m6.py",
    "tools/srq_generalization_m11.py",
    "tools/srq_generalization_m20.py",
    "tools/experiment_runner.py",
    "models/backbone.py",
    "utils/data_utils.py",
    "utils/train_utils.py",
)


def _code() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_m22_notebook_pins_source_and_normalized_hashes():
    code = _code()
    assert f"REPO_COMMIT='{SOURCE_COMMIT}'" in code
    for relative in LOCKED:
        digest = hashlib.sha256(
            (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        assert f"'{relative}':'{digest}'" in code


def test_m22_notebook_is_train_only_atomic_and_exports_complete_artifact():
    code = _code()
    assert "--extract-train-only" in code
    assert "--extract-test" not in code
    assert "load_test=True" not in code
    assert "TOTAL_UNITS=48" in code
    assert "--max-new-units','1'" in code
    assert "--require-clean-git" in code
    assert "m22_results.json" in code
    assert "m22_selection_controls_train_only.zip" in code
    assert "relative.parts[0]!='cache'" in code
    assert "do not relax gates or discard unfavorable outcomes" in code


def test_m22_notebook_preflight_precedes_feature_extraction_and_long_run():
    code = _code()
    pytest_call = code.index("tests/test_srq_generalization_m22.py")
    feature_call = code.index("--extract-features-only")
    long_call = code.index("M22 START/RESUME")
    assert pytest_call < feature_call < long_call
