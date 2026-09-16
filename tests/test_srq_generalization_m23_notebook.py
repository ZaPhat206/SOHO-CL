"""Source-lock and train-only checks for the M23 Colab notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "srq_generalization_m23_equal_budget_multistream_colab.ipynb"


def _source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_m23_notebook_is_pinned_train_only_and_compiles():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert "REPO_COMMIT='b23dd37'" in code
    assert "srq_generalization_m23_equal_budget_multistream_train_only.zip" in code
    assert "M5_NAME='srq_generalization_m5_equal_budget_train_only.zip'" in code
    assert "test.pt').exists()" in code
    assert "--extract-train-only" in code
    assert "--original-m5-artifact" in code
    assert "M23 START" in code
    assert "m23_results.json" in code
    assert "files.download(archive)" in code
    locked = dict(re.findall(r"'([^']+)':'([0-9a-f]{64})'", code))
    for relative in (
        "configs/srq_generalization_m23_equal_budget_multistream_train_only.json",
        "tools/srq_generalization_m23.py",
        "tools/srq_generalization_m5.py",
        "tools/srq_generalization_m4.py",
        "tools/experiment_runner.py",
        "methods/analytic_ridge/backends.py",
        "methods/frontends/ranpac.py",
        "methods/frontends/countsketch.py",
        "methods/frontends/__init__.py",
    ):
        assert relative in locked
        assert locked[relative] == _source_sha(ROOT / relative), relative
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), str(NOTEBOOK), "exec")
