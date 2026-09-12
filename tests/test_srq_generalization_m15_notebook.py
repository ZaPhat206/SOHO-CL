from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "srq_generalization_m15_loranpac_task1_closure_colab.ipynb"
)
PINNED_COMMIT = "13b0bd063c471ebddfd7e054d5f1a02a50eeeed4"


def _canonical_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _code() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_m15_notebook_pins_immutable_runner_and_protocol_sources():
    code = _code()
    assert f"REPO_COMMIT='{PINNED_COMMIT}'" in code
    for relative in (
        "configs/srq_generalization_m15_loranpac_task1_closure.json",
        "tools/srq_generalization_m15.py",
        "methods/frontends/loranpac.py",
        "tests/test_srq_generalization_m15.py",
        "docs/research/SRQ_GENERALIZATION_M15_PROTOCOL.md",
    ):
        assert f"'{relative}':'{_canonical_sha(ROOT / relative)}'" in code
    assert "git','checkout','--detach',REPO_COMMIT" in code
    assert "M15 SOURCE LOCK: PASS" in code


def test_m15_notebook_is_train_only_nonpredictive_and_source_locked():
    code = _code()
    assert "--extract-train-only" in code
    assert "load-test" not in code
    assert ".argmax(" not in code
    assert "predict_logits" not in code
    assert "srq_generalization_m14_loranpac_multiseed_train_only.zip" in code
    assert "4beb726bf7f29f8569e5c6de630d90284abdf7ae53928471c27bba178d148b0c" in code
    assert "assert (cache/'train.pt').is_file() and not (cache/'test.pt').exists()" in code


def test_m15_notebook_runs_tests_before_audit_and_exports_resumable_evidence():
    code = _code()
    assert code.index("tests/test_srq_generalization_m15.py") < code.index(
        "'run','--config'"
    )
    assert "COMPLETED WIDTH UNITS" in code
    assert "M15 UNIT" not in code  # runner, rather than notebook, owns unit semantics
    assert "m15_results.json" in code
    assert "m15_numerical_metrics.csv" in code
    assert "m15_task1_numerical_closure.svg" in code
    assert "srq_generalization_m15_loranpac_task1_closure.zip" in code
    assert "'m14_status_remains':'FAIL_M14_LORANPAC_MULTISEED_TRAIN_ONLY'" in code
    assert "PASS_M15_LORANPAC_TASK1_CLOSURE" in code
