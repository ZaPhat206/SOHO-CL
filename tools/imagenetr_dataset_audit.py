"""Deterministic, feature-free identity audit for processed ImageNet-R."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_CLASSES = 200
EXPECTED_TRAIN_IMAGES = 23918
EXPECTED_TEST_IMAGES = 6082


class DatasetOverlapError(ValueError):
    def __init__(self, report: dict):
        self.report = report
        examples = report["cross_split_duplicate_examples"][:3]
        super().__init__(
            "ImageNet-R has content-identical train/test images: "
            f"{examples}"
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_imagenetr_root(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    candidates = (
        root,
        root / "imagenet-r",
        root / "imagenet-r" / "imagenet-r",
    )
    matches = []
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            resolved = candidate.resolve()
            if resolved not in matches:
                matches.append(resolved)
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one processed ImageNet-R train/test root below "
            f"{root}; found {matches}"
        )
    return matches[0]


def _class_directories(split: Path) -> list[Path]:
    directories = sorted(path for path in split.iterdir() if path.is_dir())
    unexpected = sorted(path.name for path in split.iterdir() if not path.is_dir())
    if unexpected:
        raise ValueError(f"unexpected files at split root {split}: {unexpected}")
    return directories


def _audit_split(
    split_name: str,
    split: Path,
    progress_every: int,
    workers: int,
) -> tuple[dict, dict[str, list[str]]]:
    class_directories = _class_directories(split)
    pending: list[tuple[Path, str, int]] = []
    digests: dict[str, list[str]] = {}
    counts_by_class: dict[str, int] = {}
    for class_directory in class_directories:
        images = sorted(path for path in class_directory.rglob("*") if path.is_file())
        invalid = [path for path in images if path.suffix.lower() not in IMAGE_SUFFIXES]
        if invalid:
            raise ValueError(
                f"non-image files found in {class_directory}: {invalid[:3]}"
            )
        if not images:
            raise ValueError(f"empty ImageNet-R class: {class_directory}")
        counts_by_class[class_directory.name] = len(images)
        for image in images:
            relative = image.relative_to(split).as_posix()
            byte_count = image.stat().st_size
            pending.append((image, relative, byte_count))

    paths = [item[0] for item in pending]
    if workers == 1:
        digest_iterator = map(sha256_file, paths)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        digest_iterator = executor.map(sha256_file, paths)
    records: list[tuple[str, int, str]] = []
    try:
        for index, ((_, relative, byte_count), digest) in enumerate(
            zip(pending, digest_iterator), start=1
        ):
            records.append((relative, byte_count, digest))
            digests.setdefault(digest, []).append(f"{split_name}/{relative}")
            if progress_every and index % progress_every == 0:
                print(
                    f"[ImageNet-R audit] split={split_name} hashed={index}",
                    flush=True,
                )
    finally:
        if workers != 1:
            executor.shutdown(wait=True)
    manifest = hashlib.sha256()
    for relative, byte_count, digest in records:
        manifest.update(f"{relative}\0{byte_count}\0{digest}\n".encode("utf-8"))
    return {
        "image_count": len(records),
        "class_count": len(class_directories),
        "class_names": [path.name for path in class_directories],
        "counts_by_class": counts_by_class,
        "content_manifest_sha256": manifest.hexdigest(),
        "total_bytes": sum(record[1] for record in records),
        "within_split_duplicate_content_count": sum(
            1 for paths_for_digest in digests.values()
            if len(paths_for_digest) > 1
        ),
    }, digests


def _class_from_audit_path(path: str) -> str:
    parts = path.split("/")
    if len(parts) < 3:
        raise ValueError(f"invalid audited image path: {path}")
    return parts[1]


def audit_imagenetr_dataset(
    root: str | Path,
    expected_classes: int = EXPECTED_CLASSES,
    expected_train_images: int = EXPECTED_TRAIN_IMAGES,
    expected_test_images: int = EXPECTED_TEST_IMAGES,
    progress_every: int = 2000,
    workers: int = 1,
    reject_cross_split_duplicates: bool = True,
) -> dict:
    if workers <= 0:
        raise ValueError("workers must be positive")
    imagenetr_root = resolve_imagenetr_root(root)
    train, train_digests = _audit_split(
        "train", imagenetr_root / "train", progress_every, workers
    )
    test, test_digests = _audit_split(
        "test", imagenetr_root / "test", progress_every, workers
    )
    if train["class_names"] != test["class_names"]:
        raise ValueError("ImageNet-R train/test class mappings differ")
    if train["class_count"] != expected_classes:
        raise ValueError(
            f"expected {expected_classes} ImageNet-R classes, "
            f"observed {train['class_count']}"
        )
    if train["image_count"] != expected_train_images:
        raise ValueError(
            f"expected {expected_train_images} train images, "
            f"observed {train['image_count']}"
        )
    if test["image_count"] != expected_test_images:
        raise ValueError(
            f"expected {expected_test_images} test images, "
            f"observed {test['image_count']}"
        )
    cross_split_digests = sorted(set(train_digests) & set(test_digests))
    cross_split_examples = [
        train_digests[digest] + test_digests[digest]
        for digest in cross_split_digests[:20]
    ]
    conflicting_label_digests = []
    same_label_digests = []
    for digest in cross_split_digests:
        train_classes = {
            _class_from_audit_path(path) for path in train_digests[digest]
        }
        test_classes = {
            _class_from_audit_path(path) for path in test_digests[digest]
        }
        if train_classes == test_classes and len(train_classes) == 1:
            same_label_digests.append(digest)
        else:
            conflicting_label_digests.append(digest)

    class_mapping_bytes = json.dumps(
        train["class_names"], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    identity_fields = {
        "schema_version": 1,
        "dataset": "ImageNet-R",
        "processed_layout": "ImageFolder/train+test",
        "class_mapping_sha256": hashlib.sha256(class_mapping_bytes).hexdigest(),
        "train": {
            key: train[key]
            for key in (
                "image_count",
                "class_count",
                "content_manifest_sha256",
                "total_bytes",
                "within_split_duplicate_content_count",
            )
        },
        "test": {
            key: test[key]
            for key in (
                "image_count",
                "class_count",
                "content_manifest_sha256",
                "total_bytes",
                "within_split_duplicate_content_count",
            )
        },
        "cross_split_duplicate_content_count": len(cross_split_digests),
        "cross_split_same_label_duplicate_count": len(same_label_digests),
        "cross_split_conflicting_label_duplicate_count": len(
            conflicting_label_digests
        ),
    }
    identity_bytes = json.dumps(
        identity_fields, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    report = {
        **identity_fields,
        "dataset_identity_sha256": hashlib.sha256(identity_bytes).hexdigest(),
        "resolved_root": str(imagenetr_root),
        "class_names": train["class_names"],
        "train_counts_by_class": train["counts_by_class"],
        "test_counts_by_class": test["counts_by_class"],
        "cross_split_duplicate_examples": cross_split_examples,
        "cross_split_duplicate_sha256": cross_split_digests,
        "audit_semantics": "raw-byte/path identity only; no image decoding",
    }
    if cross_split_digests and reject_cross_split_duplicates:
        raise DatasetOverlapError(report)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--expected-identity-sha256")
    parser.add_argument("--progress-every", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--diagnose-cross-split-duplicates",
        action="store_true",
        help="write a failing overlap report instead of aborting before output",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = audit_imagenetr_dataset(
        args.root,
        progress_every=args.progress_every,
        workers=args.workers,
        reject_cross_split_duplicates=not args.diagnose_cross_split_duplicates,
    )
    if (
        args.expected_identity_sha256
        and report["dataset_identity_sha256"] != args.expected_identity_sha256
    ):
        raise ValueError(
            "ImageNet-R dataset identity mismatch: expected "
            f"{args.expected_identity_sha256}, observed "
            f"{report['dataset_identity_sha256']}"
        )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": (
                    "FAIL_CROSS_SPLIT_DUPLICATES"
                    if report["cross_split_duplicate_content_count"]
                    else "PASS"
                ),
                "resolved_root": report["resolved_root"],
                "classes": report["train"]["class_count"],
                "train_images": report["train"]["image_count"],
                "test_images": report["test"]["image_count"],
                "dataset_identity_sha256": report["dataset_identity_sha256"],
                "cross_split_duplicate_content_count": report[
                    "cross_split_duplicate_content_count"
                ],
                "cross_split_conflicting_label_duplicate_count": report[
                    "cross_split_conflicting_label_duplicate_count"
                ],
                "decoded_images": 0,
                "extracted_features": 0,
            },
            indent=2,
        )
    )
    if report["cross_split_duplicate_content_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
