import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/srq_generalization_m21_accumulation_gram_load_colab.ipynb"
PINNED_COMMIT = "beb94cb0ba5f3cf98f93dd8594575a87eb4f0bd3"


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _code() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"] if cell["cell_type"] == "code"
    )


def test_m21_notebook_pins_commit_and_current_source_hashes():
    code = _code()
    assert f"REPO_COMMIT='{PINNED_COMMIT}'" in code
    pinned = re.findall(r"'([^']+\.(?:py|json|md))':'([0-9a-f]{64})'", code)
    paths = {path for path, _ in pinned}
    for required in (
        "configs/srq_generalization_m21_accumulation_gram_load_train_only.json",
        "tools/srq_generalization_m21.py",
        "methods/srq_fly_optimized/minimal_load_control.py",
        "docs/research/SRQ_GENERALIZATION_M21_PROTOCOL.md",
        "tests/test_srq_generalization_m21.py",
        "tests/test_minimal_load_control.py",
        "configs/srq_fly_priority3_direct_control_cifar100_train_only.json",
        "tools/srq_fly_priority3_direct_control.py",
        "methods/srq_fly_optimized/direct_control.py",
        "methods/analytic_ridge/backends.py",
    ):
        assert required in paths
    for relative, expected in pinned:
        assert _source_sha256(ROOT / relative) == expected, relative


def test_m21_notebook_is_train_only_t4_and_atomic():
    code = _code()
    assert "'T4' in gpu_name" in code
    assert "'--extract-train-only'" in code
    assert "not (cache/'test.pt').exists()" in code
    assert "extract-test" not in code
    assert "'--max-new-units','1'" in code
    assert "after==before+1" in code
    assert "'--require-clean-git'" in code
    assert "'--code-cache-dir',WTA_CACHE_DIR" in code
    assert "TOTAL_UNITS=8" in code


def test_m21_notebook_exports_results_without_sample_caches_and_keeps_warnings():
    code = _code()
    assert "srq_generalization_m21_accumulation_gram_load_t4.zip" in code
    assert "Path(OUTPUT_DIR).rglob('*')" in code
    assert "WTA_CACHE_DIR=RUN_ROOT+'/wta_10000'" in code and "OUTPUT_DIR=RUN_ROOT+'/output'" in code
    assert "m21_results.json" in code
    assert "preserve the artifact; do not relax gates" in code
    assert "files.download(FINAL_EXPORT)" in code
