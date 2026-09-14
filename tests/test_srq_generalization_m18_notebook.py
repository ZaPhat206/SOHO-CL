import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/srq_generalization_m18_fly20k_adaptive_colab.ipynb"


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _code() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_m18_notebook_pins_committed_scientific_source():
    code = _code()
    assert "REPO_COMMIT='6464950c3834ff9a1f81faed1ba9a40a746e9f89'" in code
    for relative in (
        "configs/srq_generalization_m18_fly20k_adaptive_locked_test.json",
        "tools/srq_generalization_m18.py",
        "tests/test_srq_generalization_m18.py",
        "docs/research/SRQ_GENERALIZATION_M18_PROTOCOL.md",
    ):
        digest = _source_sha256(ROOT / relative)
        assert f"'{relative}':'{digest}'" in code


def test_m18_notebook_enforces_authorization_before_test_and_run():
    code = _code()
    authorize = code.index("RUNNER,'authorize'")
    extract_test = code.index("RUNNER,'extract-test'")
    run = code.index("RUNNER,'run'")
    assert authorize < extract_test < run
    assert "assert not (cache_dir/'test.pt').exists()" in code
    assert "--max-new-replicates','1'" in code
    assert "assert after==before+1" in code


def test_m18_notebook_has_cross_account_handoffs_without_wta_cache():
    code = _code()
    assert "IMPORT_HANDOFF=False" in code
    assert "m18_handoff_{completed:03d}_replicates.zip" in code
    assert "after in (2,4)" in code
    assert "finished WTA cache removed" in code
    assert "CODE_CACHE_ROOT" not in code[code.index("def create_handoff"):code.index("print('M18 HANDOFF HELPER READY')")]


def test_m18_notebook_exports_locked_complete_result():
    code = _code()
    assert "srq_generalization_m18_fly20k_adaptive_locked_test.zip" in code
    assert "m18_results.json" in code
    assert "PASS_M18_FLY20K_ADAPTIVE_LOCKED_TEST" in code
    assert "do not filter a seed or relax a gate" in code
    assert "accuracy_gate" not in code.lower().replace(
        "result['status']=='pass_m18_fly20k_adaptive_locked_test'", ""
    )
