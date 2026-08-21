from pathlib import Path

import pytest

from tools.imagenetr_dataset_audit import (
    DatasetOverlapError,
    audit_imagenetr_dataset,
    resolve_imagenetr_root,
)


def _write_image(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _tiny_imagenetr(root: Path):
    for split in ("train", "test"):
        for class_name in ("n00000001", "n00000002"):
            for index in range(2):
                _write_image(
                    root
                    / "imagenet-r"
                    / "imagenet-r"
                    / split
                    / class_name
                    / f"{split}_{index}.jpg",
                    f"{split}-{class_name}-{index}".encode(),
                )


def test_resolve_and_audit_processed_imagenetr_is_deterministic(tmp_path):
    _tiny_imagenetr(tmp_path)

    first = audit_imagenetr_dataset(
        tmp_path,
        expected_classes=2,
        expected_train_images=4,
        expected_test_images=4,
        progress_every=0,
    )
    second = audit_imagenetr_dataset(
        tmp_path / "imagenet-r" / "imagenet-r",
        expected_classes=2,
        expected_train_images=4,
        expected_test_images=4,
        progress_every=0,
    )

    expected_root = (tmp_path / "imagenet-r" / "imagenet-r").resolve()
    assert resolve_imagenetr_root(tmp_path) == expected_root
    assert first["dataset_identity_sha256"] == second["dataset_identity_sha256"]
    assert first["class_names"] == ["n00000001", "n00000002"]
    assert first["cross_split_duplicate_content_count"] == 0
    assert first["train"]["within_split_duplicate_content_count"] == 0
    assert first["test"]["within_split_duplicate_content_count"] == 0
    assert first["audit_semantics"].endswith("no image decoding")


def test_audit_rejects_train_test_content_overlap(tmp_path):
    _tiny_imagenetr(tmp_path)
    base = tmp_path / "imagenet-r" / "imagenet-r"
    duplicate = (base / "train" / "n00000001" / "train_0.jpg").read_bytes()
    (base / "test" / "n00000002" / "test_0.jpg").write_bytes(duplicate)

    with pytest.raises(DatasetOverlapError, match="content-identical train/test") as error:
        audit_imagenetr_dataset(
            tmp_path,
            expected_classes=2,
            expected_train_images=4,
            expected_test_images=4,
            progress_every=0,
        )
    assert error.value.report["cross_split_duplicate_content_count"] == 1
    assert error.value.report["cross_split_conflicting_label_duplicate_count"] == 1


def test_diagnostic_mode_reports_overlap_without_decoding(tmp_path):
    _tiny_imagenetr(tmp_path)
    base = tmp_path / "imagenet-r" / "imagenet-r"
    duplicate = (base / "train" / "n00000001" / "train_0.jpg").read_bytes()
    (base / "test" / "n00000001" / "test_0.jpg").write_bytes(duplicate)

    report = audit_imagenetr_dataset(
        tmp_path,
        expected_classes=2,
        expected_train_images=4,
        expected_test_images=4,
        progress_every=0,
        workers=2,
        reject_cross_split_duplicates=False,
    )

    assert report["cross_split_duplicate_content_count"] == 1
    assert report["cross_split_same_label_duplicate_count"] == 1
    assert report["cross_split_conflicting_label_duplicate_count"] == 0


def test_audit_rejects_class_mapping_mismatch(tmp_path):
    _tiny_imagenetr(tmp_path)
    test = tmp_path / "imagenet-r" / "imagenet-r" / "test"
    (test / "n00000002").rename(test / "n00000003")

    with pytest.raises(ValueError, match="class mappings differ"):
        audit_imagenetr_dataset(
            tmp_path,
            expected_classes=2,
            expected_train_images=4,
            expected_test_images=4,
            progress_every=0,
        )


def test_audit_rejects_wrong_expected_counts(tmp_path):
    _tiny_imagenetr(tmp_path)

    with pytest.raises(ValueError, match="expected 5 train images"):
        audit_imagenetr_dataset(
            tmp_path,
            expected_classes=2,
            expected_train_images=5,
            expected_test_images=4,
            progress_every=0,
        )


def test_audit_counts_within_split_duplicate_content(tmp_path):
    _tiny_imagenetr(tmp_path)
    base = tmp_path / "imagenet-r" / "imagenet-r" / "train" / "n00000001"
    duplicate = (base / "train_0.jpg").read_bytes()
    (base / "train_1.jpg").write_bytes(duplicate)

    report = audit_imagenetr_dataset(
        tmp_path,
        expected_classes=2,
        expected_train_images=4,
        expected_test_images=4,
        progress_every=0,
    )

    assert report["train"]["within_split_duplicate_content_count"] == 1
    assert report["test"]["within_split_duplicate_content_count"] == 0
