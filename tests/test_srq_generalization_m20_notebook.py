import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/srq_generalization_m20_adaptive_budget_colab.ipynb"
PINNED_COMMIT = "3308a0360ef8b541cfdacc7709d545a37a0efbc9"


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _code() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"] if cell["cell_type"] == "code"
    )


def test_m20_notebook_pins_commit_and_current_source_hashes():
    code = _code()
    assert f"REPO_COMMIT='{PINNED_COMMIT}'" in code
    pinned = re.findall(r"'([^']+\.(?:py|json|md))':'([0-9a-f]{64})'", code)
    paths = {path for path, _ in pinned}
    for required in (
        "configs/srq_generalization_m20_adaptive_budget_criterion_train_only.json",
        "tools/srq_generalization_m20.py",
        "methods/analytic_ridge/adaptive_selection.py",
        "methods/analytic_ridge/adaptive_upper.py",
        "methods/analytic_ridge/backends.py",
        "docs/research/SRQ_GENERALIZATION_M20_PROTOCOL.md",
        "tests/test_srq_generalization_m20.py",
        "tests/test_adaptive_selection.py",
    ):
        assert required in paths
    for relative, expected in pinned:
        assert _source_sha256(ROOT / relative) == expected, relative


def test_m20_notebook_is_train_only_t4_and_atomic():
    code = _code()
    assert "'T4' in gpu_name" in code
    assert "'--extract-train-only'" in code
    assert "not (cache/'test.pt').exists()" in code
    assert "extract-test" not in code
    assert "'--max-new-units','1'" in code
    assert "after==before+1" in code
    assert "'--require-clean-git'" in code
    assert "TOTAL_UNITS=37" in code


def test_m20_notebook_exports_results_without_cache_and_keeps_warnings():
    code = _code()
    assert "srq_generalization_m20_adaptive_budget_criterion_t4.zip" in code
    assert "relative.parts[0]!='cache'" in code
    assert "m20_results.json" in code
    assert "preserve the artifact; do not relax gates" in code
    assert "files.download(FINAL_EXPORT)" in code
