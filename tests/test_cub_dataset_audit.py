from pathlib import Path

import pytest

from tools.cub_dataset_audit import audit_cub_dataset, resolve_cub_root


def _write_image(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _tiny_cub(root: Path):
    for split in ("train", "test"):
        for class_name in ("001.One", "002.Two"):
            for index in range(2):
                _write_image(
                    root / "cub" / split / class_name / f"{split}_{index}.jpg",
                    f"{split}-{class_name}-{index}".encode(),
                )


def test_resolve_and_audit_processed_cub_is_deterministic(tmp_path):
    _tiny_cub(tmp_path)

    first = audit_cub_dataset(
        tmp_path, expected_classes=2, expected_train_images=4,
        expected_test_images=4, progress_every=0,
    )
    second = audit_cub_dataset(
        tmp_path / "cub", expected_classes=2, expected_train_images=4,
        expected_test_images=4, progress_every=0,
    )

    assert resolve_cub_root(tmp_path) == (tmp_path / "cub").resolve()
    assert first["dataset_identity_sha256"] == second["dataset_identity_sha256"]
    assert first["class_names"] == ["001.One", "002.Two"]
    assert first["cross_split_duplicate_content_count"] == 0


def test_audit_rejects_train_test_content_overlap(tmp_path):
    _tiny_cub(tmp_path)
    duplicate = (tmp_path / "cub" / "train" / "001.One" / "train_0.jpg").read_bytes()
    (tmp_path / "cub" / "test" / "002.Two" / "test_0.jpg").write_bytes(duplicate)

    with pytest.raises(ValueError, match="content-identical train/test"):
        audit_cub_dataset(
            tmp_path, expected_classes=2, expected_train_images=4,
            expected_test_images=4, progress_every=0,
        )


def test_audit_rejects_class_mapping_mismatch(tmp_path):
    _tiny_cub(tmp_path)
    source = tmp_path / "cub" / "test" / "002.Two"
    source.rename(tmp_path / "cub" / "test" / "003.Three")

    with pytest.raises(ValueError, match="class mappings differ"):
        audit_cub_dataset(
            tmp_path, expected_classes=2, expected_train_images=4,
            expected_test_images=4, progress_every=0,
        )
