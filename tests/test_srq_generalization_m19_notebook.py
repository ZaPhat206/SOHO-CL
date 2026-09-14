import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/srq_generalization_m19_panel_system_benchmark_colab.ipynb"


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _code() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"] if cell["cell_type"] == "code"
    )


def test_m19_notebook_pins_clean_portable_source():
    code = _code()
    assert "REPO_COMMIT='17c1b734e8d825b832517ae5ac463b648512528e'" in code
    for relative in (
        "configs/srq_generalization_m19_panel_system_benchmark.json",
        "tools/srq_generalization_m19.py",
        "tests/test_srq_generalization_m19.py",
        "docs/research/SRQ_GENERALIZATION_M19_PROTOCOL.md",
    ):
        assert f"'{relative}':'{_source_sha256(ROOT / relative)}'" in code


def test_m19_notebook_is_synthetic_t4_only_and_atomic():
    code = _code()
    assert "'T4' in gpu_name" in code
    assert "--max-new-units','1'" in code
    assert "after==before+1" in code
    assert "10 units complete" in code
    forbidden = ("extract-test", "extract-train", "test.pt", "train.pt", "files.upload")
    assert all(token not in code for token in forbidden)


def test_m19_notebook_exports_warning_artifacts_instead_of_discarding_them():
    code = _code()
    assert "srq_generalization_m19_panel_system_benchmark_t4.zip" in code
    assert "m19_results.json" in code
    assert "preserve the artifact; do not relax gates" in code
    assert "files.download(FINAL_EXPORT)" in code
