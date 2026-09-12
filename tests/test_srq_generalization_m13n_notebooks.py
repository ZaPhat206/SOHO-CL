"""Source-lock and boundary checks for M13-N Colab/Kaggle notebooks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = (
    ROOT / "notebooks/srq_generalization_m13n_numerical_audit_colab.ipynb",
    ROOT / "notebooks/srq_generalization_m13n_numerical_audit_kaggle.ipynb",
)
LOCKED_PATHS = (
    "configs/srq_generalization_m13n_numerical_audit.json",
    "tools/srq_generalization_m13n.py",
    "methods/frontends/loranpac.py",
    "tools/experiment_runner.py",
    "models/backbone.py",
    "utils/data_utils.py",
    "utils/train_utils.py",
    "tests/test_srq_generalization_m13n.py",
    "docs/research/SRQ_GENERALIZATION_M13N_PROTOCOL.md",
)


def _code(path: Path) -> tuple[dict, str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    return notebook, code


@pytest.mark.parametrize("path", NOTEBOOKS)
def test_m13n_notebook_is_pinned_compilable_and_source_locked(path):
    notebook, code = _code(path)
    pinned = re.search(r"REPO_COMMIT='([0-9a-f]{40})'", code)
    assert pinned is not None
    for relative in LOCKED_PATHS:
        assert subprocess.run(
            ["git", "cat-file", "-e", f"{pinned.group(1)}:{relative}"],
            cwd=ROOT,
        ).returncode == 0
        digest = hashlib.sha256(
            (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        ).hexdigest()
        assert f"'{relative}':'{digest}'" in code
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell.get("source", [])), str(path), "exec")


@pytest.mark.parametrize("path", NOTEBOOKS)
def test_m13n_notebook_preserves_train_only_nonpredictive_boundary(path):
    _, code = _code(path)
    assert "--extract-train-only" in code
    assert "not (cache/'test.pt').exists()" in code
    assert "--source-m13-artifact" in code
    assert "--require-clean-git" in code
    assert "m13n_results.json" in code
    assert "srq_generalization_m13n_numerical_audit.zip" in code
    assert "PASS_M13N_NUMERICAL_AUDIT" in code
    assert "FAIL_M13_LORANPAC_TRAIN_ONLY" in code
    assert "predict_logits" not in code
    assert ".argmax(" not in code


def test_m13n_kaggle_notebook_requires_byte_preserved_source_bin():
    _, code = _code(NOTEBOOKS[1])
    assert "SOURCE_NAME+'.bin'" in code
    assert "shutil.copyfile(matches[0],source_copy)" in code
    assert "sha_raw(source_copy)==SOURCE_SHA" in code
    assert "zipfile.ZipFile(source_copy" not in code
    assert "'/kaggle/working/srq_generalization_m13n_numerical_audit.zip'" in code


def test_m13n_colab_exports_before_enforcing_pass_status():
    _, code = _code(NOTEBOOKS[0])
    assert code.index("files.download(str(export))") < code.index(
        "result['status']=='PASS_M13N_NUMERICAL_AUDIT'"
    )
