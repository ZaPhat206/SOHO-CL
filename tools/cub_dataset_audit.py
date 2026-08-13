"""Deterministic identity audit for the processed CUB-200-2011 split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_CLASSES = 200
EXPECTED_TRAIN_IMAGES = 5994
EXPECTED_TEST_IMAGES = 5794


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_cub_root(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    candidates = (
        root,
        root / "cub",
        root / "cub-200-2011",
        root / "cub-200-2011" / "cub",
    )
    matches = [
        candidate for candidate in candidates
        if (candidate / "train").is_dir() and (candidate / "test").is_dir()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one processed CUB train/test root below {root}; "
            f"found {matches}"
        )
    return matches[0]


def _class_directories(split: Path) -> list[Path]:
    directories = sorted(path for path in split.iterdir() if path.is_dir())
    unexpected = sorted(path.name for path in split.iterdir() if not path.is_dir())
    if unexpected:
        raise ValueError(f"unexpected files at split root {split}: {unexpected}")
    return directories


def _audit_split(split_name: str, split: Path, progress_every: int) -> tuple[dict, dict[str, list[str]]]:
    class_directories = _class_directories(split)
    records: list[tuple[str, int, str]] = []
    digests: dict[str, list[str]] = {}
    counts_by_class: dict[str, int] = {}
    for class_directory in class_directories:
        images = sorted(path for path in class_directory.rglob("*") if path.is_file())
        invalid = [path for path in images if path.suffix.lower() not in IMAGE_SUFFIXES]
        if invalid:
            raise ValueError(f"non-image files found in {class_directory}: {invalid[:3]}")
        if not images:
            raise ValueError(f"empty CUB class: {class_directory}")
        counts_by_class[class_directory.name] = len(images)
        for image in images:
            relative = image.relative_to(split).as_posix()
            digest = sha256_file(image)
            records.append((relative, image.stat().st_size, digest))
            digests.setdefault(digest, []).append(f"{split_name}/{relative}")
            if progress_every and len(records) % progress_every == 0:
                print(
                    f"[CUB audit] split={split_name} hashed={len(records)}",
                    flush=True,
                )
    manifest = hashlib.sha256()
    for relative, byte_count, digest in records:
        manifest.update(
            f"{relative}\0{byte_count}\0{digest}\n".encode("utf-8")
        )
    return {
        "image_count": len(records),
        "class_count": len(class_directories),
        "class_names": [path.name for path in class_directories],
        "counts_by_class": counts_by_class,
        "content_manifest_sha256": manifest.hexdigest(),
        "total_bytes": sum(record[1] for record in records),
    }, digests


def audit_cub_dataset(
    root: str | Path,
    expected_classes: int = EXPECTED_CLASSES,
    expected_train_images: int = EXPECTED_TRAIN_IMAGES,
    expected_test_images: int = EXPECTED_TEST_IMAGES,
    progress_every: int = 1000,
) -> dict:
    cub_root = resolve_cub_root(root)
    train, train_digests = _audit_split("train", cub_root / "train", progress_every)
    test, test_digests = _audit_split("test", cub_root / "test", progress_every)
    if train["class_names"] != test["class_names"]:
        raise ValueError("CUB train/test class mappings differ")
    if train["class_count"] != expected_classes:
        raise ValueError(
            f"expected {expected_classes} CUB classes, observed {train['class_count']}"
        )
    if train["image_count"] != expected_train_images:
        raise ValueError(
            f"expected {expected_train_images} train images, observed {train['image_count']}"
        )
    if test["image_count"] != expected_test_images:
        raise ValueError(
            f"expected {expected_test_images} test images, observed {test['image_count']}"
        )
    cross_split_digests = sorted(set(train_digests) & set(test_digests))
    if cross_split_digests:
        examples = [
            train_digests[digest] + test_digests[digest]
            for digest in cross_split_digests[:3]
        ]
        raise ValueError(f"CUB has content-identical train/test images: {examples}")
    class_mapping_bytes = json.dumps(
        train["class_names"], ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    identity_fields = {
        "schema_version": 1,
        "dataset": "CUB-200-2011",
        "processed_layout": "ImageFolder/train+test",
        "class_mapping_sha256": hashlib.sha256(class_mapping_bytes).hexdigest(),
        "train": {
            key: train[key]
            for key in ("image_count", "class_count", "content_manifest_sha256", "total_bytes")
        },
        "test": {
            key: test[key]
            for key in ("image_count", "class_count", "content_manifest_sha256", "total_bytes")
        },
        "cross_split_duplicate_content_count": 0,
    }
    identity_bytes = json.dumps(
        identity_fields, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        **identity_fields,
        "dataset_identity_sha256": hashlib.sha256(identity_bytes).hexdigest(),
        "resolved_root": str(cub_root),
        "class_names": train["class_names"],
        "train_counts_by_class": train["counts_by_class"],
        "test_counts_by_class": test["counts_by_class"],
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output")
    parser.add_argument("--expected-identity-sha256")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = audit_cub_dataset(args.root, progress_every=args.progress_every)
    if (
        args.expected_identity_sha256
        and report["dataset_identity_sha256"] != args.expected_identity_sha256
    ):
        raise ValueError(
            "CUB dataset identity mismatch: expected "
            f"{args.expected_identity_sha256}, observed "
            f"{report['dataset_identity_sha256']}"
        )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "resolved_root": report["resolved_root"],
        "classes": report["train"]["class_count"],
        "train_images": report["train"]["image_count"],
        "test_images": report["test"]["image_count"],
        "dataset_identity_sha256": report["dataset_identity_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
