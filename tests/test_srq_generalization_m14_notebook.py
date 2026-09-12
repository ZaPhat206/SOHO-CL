"""Source-lock and platform-boundary checks for the M14 notebooks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
KAGGLE_NOTEBOOK = (
    ROOT / "notebooks/srq_generalization_m14_loranpac_multiseed_kaggle.ipynb"
)
COLAB_NOTEBOOK = (
    ROOT / "notebooks/srq_generalization_m14_loranpac_multiseed_colab.ipynb"
)

LOCKED_PATHS = (
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


def _read_code(notebook_path: Path) -> tuple[dict, str, str]:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    pinned = re.search(r"REPO_COMMIT='([0-9a-f]{40})'", code)
    assert pinned is not None
    return notebook, code, pinned.group(1)


def _assert_source_lock(code: str, commit: str) -> None:
    for relative in LOCKED_PATHS:
        payload = subprocess.check_output(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT
        ).replace(b"\r\n", b"\n")
        digest = hashlib.sha256(payload).hexdigest()
        assert f"'{relative}':'{digest}'" in code


def _assert_cells_compile(notebook: dict, notebook_path: Path) -> None:
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell.get("source", [])), str(notebook_path), "exec")


def test_m14_kaggle_notebook_is_pinned_train_only_and_byte_preserving():
    notebook, code, commit = _read_code(KAGGLE_NOTEBOOK)
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
    _assert_source_lock(code, commit)
    _assert_cells_compile(notebook, KAGGLE_NOTEBOOK)


def test_m14_colab_notebook_is_pinned_train_only_and_session_local():
    notebook, code, commit = _read_code(COLAB_NOTEBOOK)
    assert "drive.mount(" not in code
    assert "PERSIST_ROOT='/content/srq_m14_local'" in code
    assert "--extract-train-only" in code
    assert "not (cache/'test.pt').exists()" in code
    assert "--source-m6-artifact" in code
    assert "--source-m11-artifact" in code
    assert "--source-m13n-artifact" in code
    assert "60 units" in code
    assert "COMPLETED UNITS BEFORE RUN" in code
    assert "resumable only while this runtime remains alive" in code
    assert "m14_runner.log" in code
    assert "subprocess.Popen(command" in code
    assert "stderr=subprocess.STDOUT" in code
    assert "LAST 200 RUNNER LOG LINES" in code
    assert code.index("os.chdir(source_dir)") < code.index("files.upload()")
    assert "accidental=Path(WORK_DIR)/name" in code
    assert "Checkout dirty after artifact upload" in code
    assert "IMPORT_HANDOFF=False" in code
    assert "HANDOFF_NAME='m14_handoff_checkpoint.zip'" in code
    assert "def create_handoff():" in code
    assert "HANDOFF_MANIFEST.json" in code
    assert "source_sha256':SOURCE_SHA" in code
    assert "archive.extract(relative,path=PERSIST_ROOT)" in code
    assert "HANDOFF READY:" in code
    assert "HANDOFF TRAIN CACHE FOUND: raw CIFAR and checkpoint download skipped" in code
    assert "PASS_M14_LORANPAC_MULTISEED_TRAIN_ONLY" in code
    assert "srq_generalization_m14_loranpac_multiseed_train_only.zip" in code
    assert "files.download(str(export))" in code
    assert code.index("os.chdir('/content')") < code.index("shutil.rmtree(repo)")
    _assert_source_lock(code, commit)
    _assert_cells_compile(notebook, COLAB_NOTEBOOK)
