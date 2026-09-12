"""Source-lock and Kaggle boundary checks for the M14 notebook."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/srq_generalization_m14_loranpac_multiseed_kaggle.ipynb"


def test_m14_kaggle_notebook_is_pinned_train_only_and_byte_preserving():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    pinned = re.search(r"REPO_COMMIT='([0-9a-f]{40})'", code)
    assert pinned is not None
    commit = pinned.group(1)
    assert "--extract-train-only" in code
    assert "not (cache/'test.pt').exists()" in code
    assert "--source-m6-artifact" in code
    assert "--source-m11-artifact" in code
    assert "--source-m13n-artifact" in code
    assert "name+'.bin'" in code
    assert "60 units" in code
    assert "PRIOR M14 UNITS IMPORTED" in code
    assert "Conflicting prior M14 unit" in code
    assert "PASS_M14_LORANPAC_MULTISEED_TRAIN_ONLY" in code
    assert "srq_generalization_m14_loranpac_multiseed_train_only.zip" in code
    locked_paths = (
        "configs/srq_generalization_m14_loranpac_multiseed_train_only.json",
        "tools/srq_generalization_m14.py",
        "methods/frontends/loranpac.py",
        "tools/srq_generalization_m12.py",
        "tools/srq_generalization_m13.py",
        "tools/srq_generalization_m6.py",
        "tools/srq_generalization_m5.py",
        "tools/srq_generalization_m4.py",
        "tools/experiment_runner.py",
        "models/backbone.py",
        "utils/data_utils.py",
        "utils/train_utils.py",
        "tests/test_srq_generalization_m14.py",
        "docs/research/SRQ_GENERALIZATION_M14_PLAN.md",
    )
    for relative in locked_paths:
        payload = subprocess.check_output(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT
        ).replace(b"\r\n", b"\n")
        digest = hashlib.sha256(payload).hexdigest()
        assert f"'{relative}':'{digest}'" in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell.get("source", [])), str(NOTEBOOK), "exec")
