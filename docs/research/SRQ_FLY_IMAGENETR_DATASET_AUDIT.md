# SRQ-FLY ImageNet-R dataset identity audit

Status: **FAIL — processed train/test artifact is not content-disjoint.** This
is a metadata/raw-byte audit, not a model experiment. It decoded zero images,
extracted zero features, and created no `test.pt`.

## Exact command

Run from the repository root:

```powershell
$auditOutput=Join-Path $env:TEMP 'srq_fly_imagenetr_dataset_audit_v2.json'
python -u tools/imagenetr_dataset_audit.py --root ..\processed_datasets --output $auditOutput --progress-every 8000 --workers 8 --diagnose-cross-split-duplicates
```

The diagnostic intentionally returned exit code `2` because duplicate content
crossed the train/test boundary. The JSON report was 16,982 bytes with
SHA-256
`366f5c4dbf69c1e27eb4f383539df03f2960d0e8a664a6397c5a0e452571ec0b`.
The audit source `tools/imagenetr_dataset_audit.py` had SHA-256
`b850728058e06c4253f4beb8ca167f02f2405f05dfaf1738a0e5af8fdcc919b7`.

## Stable identity

- resolved local root: `D:\lab\FLY\processed_datasets\imagenet-r`;
- layout: `ImageFolder/train+test`;
- classes: `200` in both splits;
- train images: `23,918`;
- test images: `6,082`;
- dataset identity SHA-256:
  `3f3d963b2b0c245ceabc0166c8b1c64d624c2ea31df07ee6ffdbf4cab5f7739d`;
- class mapping SHA-256:
  `dd62e9413ab14ffe4f41035d8dc0c9ba29f07d7c2463bed436a1d1056e6ad385`;
- train manifest SHA-256:
  `98e87cec5777a9b84fc16a27c2f700c9126bcd5872b88d385ef1fc08c7a2b0c5`;
- test manifest SHA-256:
  `c24d414ffb410e77943100f249a7b24f7a5c48917b63831b8cb27c1747356609`.

The manifests hash each relative path, byte count, and file SHA-256 in sorted
order. Dataset identity excludes the machine-specific resolved path.

## Duplicate findings

- within-train duplicate-content groups: `38`;
- within-test duplicate-content groups: `2`;
- unique content hashes crossing train/test: `19`;
- cross-split duplicates under the same class: `1`;
- cross-split duplicates under conflicting classes: `18`.

All 19 crossing pairs are:

| Train path | Test path |
|---|---|
| `n01806143/graphic_9.jpg` | `n01860187/graphic_3.jpg` |
| `n03888257/videogame_0.jpg` | `n03773504/videogame_1.jpg` |
| `n04133789/videogame_1.jpg` | `n04118538/videogame_8.jpg` |
| `n02951358/videogame_4.jpg` | `n02950826/videogame_7.jpg` |
| `n02113799/sketch_11.jpg` | `n02113624/sketch_22.jpg` |
| `n03888257/videogame_1.jpg` | `n03773504/videogame_2.jpg` |
| `n03773504/videogame_13.jpg` | `n03888257/videogame_9.jpg` |
| `n02950826/graphic_0.jpg` | `n04389033/graphic_1.jpg` |
| `n07718472/embroidery_1.jpg` | `n03372029/embroidery_1.jpg` |
| `n02128385/toy_11.jpg` | `n02130308/toy_10.jpg` |
| `n03888257/videogame_3.jpg` | `n03773504/videogame_4.jpg` |
| `n02950826/videogame_9.jpg` | `n02951358/videogame_5.jpg` |
| `n04389033/art_2.jpg` | `n02950826/art_0.jpg` |
| `n02950826/videogame_12.jpg` | `n02951358/videogame_7.jpg` |
| `n02128757/sculpture_3.jpg` | `n02128385/sculpture_4.jpg` |
| `n02007558/deviantart_4.jpg` | `n02007558/deviantart_8.jpg` |
| `n03773504/videogame_14.jpg` | `n03888257/videogame_10.jpg` |
| `n03888257/videogame_2.jpg` | `n03773504/videogame_3.jpg` |
| `n04325704/origami_0.jpg` | `n03775071/origami_0.jpg` |

The sole same-class pair is the `n02007558` pair. Every other pair has
conflicting directory labels. This is leakage and label-conflict evidence,
not a numerical issue in SRQ-FLY.

## Decision and next gate

Do not extract or evaluate held-out ImageNet-R features from this split. Do
not silently delete files or report results from a post-hoc filtered test set.

The next protocol review must choose one option before implementation:

1. obtain a separately verified content-disjoint processed artifact; or
2. preregister a deterministic content-hash exclusion/index policy, preserve
   an immutable manifest, and rerun train-only selection if its training view
   changes.

Only metadata-based exclusion is defensible at this point because no model
accuracy has been observed. The resulting view must be audited again and
assigned new train/test manifest and dataset identity hashes before a held-out
runner is created.

## Subsequent project decision

On 2026-08-22 the project elected to retain the unchanged processed artifact
for comparability and avoid a post-audit resplit. This does not convert the
audit to a pass. Any future result from this artifact must be labeled
**legacy processed-split evaluation**, disclose all 19 crossing hashes, and
must not be called content-disjoint or an untouched held-out benchmark. All
methods still receive the identical split, backbone, preprocessing, class
order and seed.
