# SRQ generalization M9: repeated whole-process systems audit

Status: `PASS_M9_REPEATED_SYSTEMS_TRAIN_ONLY`; the four-repetition CIFAR-100
train-only measurement is complete. M9 does not open a test split and does
not select a model, precision, width, or threshold from accuracy.

## Question

Are the memory and timing observations from the one-run Priority-5 audit
repeatable when each method is executed from a fresh process and method order
is balanced?

## Locked design

- dataset view: all 50,000 CIFAR-100 training images; no test dataset;
- backbone: the same frozen ViT-B/16 checkpoint and preprocessing as
  Priority 5;
- methods: width-10,000 Exact FLY and locked P2B INT8/FP32 SRQ;
- four paired repetitions, giving eight fresh worker processes;
- balanced order: Exact/SRQ, SRQ/Exact, Exact/SRQ, SRQ/Exact;
- every worker independently loads the backbone, extracts all train features,
  releases the backbone, runs ten analytic updates, evaluates the fixed
  512-row probe, and temporarily serializes `learner.state_dict()`;
- the temporary checkpoint is deleted after its byte size is recorded;
- the parent polls NVML every 20 ms and attributes the primary process metric
  to the worker PID.

Repeated feature extraction is intentional. Reusing a feature cache would no
longer measure the complete deployment path and would make the process peak
incomparable with Priority 5.

## Quantities kept separate

For each method and repetition, M9 records:

1. persistent learner tensor bytes;
2. temporary serialized checkpoint bytes;
3. PyTorch peak allocated and reserved bytes for the analytic stage;
4. process-attributed NVML peak for the analytic stage and whole process;
5. device-wide NVML diagnostics;
6. time for backbone load, feature extraction, analytic update, final probe,
   checkpoint serialization, and their measured total;
7. solver residual and paired fixed-probe predictions/logits.

The report aggregates mean, sample standard deviation, minimum, and maximum.
It also reports paired SRQ/Exact ratios per repetition. It must never merge
persistent state, allocator peak, and process NVML peak into one ambiguous
number.

## Gates

`PASS_M9_REPEATED_SYSTEMS_TRAIN_ONLY` requires:

- at least three repetitions and all four locked repetitions complete;
- every underlying Priority-5 correctness/memory gate passes independently;
- all workers remain train-only and produce a nonempty checkpoint byte count;
- source identity is stable across workers;
- persistent state is deterministic within each method;
- the balanced method order is executed exactly;
- every worker is observed for enough NVML samples.

No gate constrains the variance or forces a favorable whole-process ratio.
Those are empirical results to report, not outcomes to guarantee after seeing
the measurements.

## Expected runtime and output

On a Tesla T4, expect roughly 35--50 minutes because the frozen feature
extraction is repeated eight times. The notebook exports:

- `m9_results.json`;
- `m9_repetitions.csv`;
- `m9_summary.csv`;
- `m9_memory_summary.svg`;
- `m9_stage_time_summary.svg`;
- eight worker JSON files and their stage markers;
- the locked config, this runbook, and a manifest.

The ZIP excludes the dataset, backbone checkpoint, scratch views, extracted
features, and temporary learner checkpoints.

## Allowed conclusion

A PASS supports only a repeatability claim for the reported CIFAR-100,
ViT-B/16, software stack, and GPU. Means and sample standard deviations may be
used in the paper. It is not evidence that every GPU or deployment realizes
the same reduction, and it is not an accuracy benchmark.

## Recorded result

Artifact SHA-256:
`71e84eeddc24066d250d93328b87561ccad0f0b9cac09a619f66c128b3f7167f`.
Result SHA-256:
`43725ef3d1c9c9faff903117848a812ce6e7b917451c81bfb205a04c2812cf8f`.
The archive passes CRC, embeds the locked config byte-for-byte, reports source
commit `22787d1`, and has `uses_test_set=false`. Its eight repetition rows and
24 aggregate rows match the result JSON exactly, and both SVG files parse as
valid XML. All nine M9 gates and every nested Priority-5 gate pass. The maximum
solver relative residual is `2.85e-6`; each worker was sampled by NVML at
least 29,729 times.

| Metric, mean $\pm$ sample SD over four runs | Exact FLY | SRQ P2B | SRQ/Exact |
|---|---:|---:|---:|
| Persistent state (MiB) | 423.44 $\pm$ 0 | 92.66 $\pm$ 0 | 0.2188 |
| Serialized checkpoint (MiB) | 446.33 $\pm$ 0 | 89.29 $\pm$ 0 | 0.2001 |
| PyTorch analytic peak allocated (MiB) | 1802.76 $\pm$ 0 | 1405.30 $\pm$ 0 | 0.7795 |
| PyTorch analytic peak reserved (MiB) | 2422 $\pm$ 0 | 2062 $\pm$ 0 | 0.8514 |
| NVML analytic worker peak (MiB) | 2586 $\pm$ 0 | 2228 $\pm$ 0 | 0.8616 |
| NVML whole-process worker peak (MiB) | 2588 $\pm$ 0 | 2228 $\pm$ 0 | 0.8609 |
| Analytic-stage time (s) | 12.72 $\pm$ 0.20 | 23.07 $\pm$ 0.17 | 1.8138 $\pm$ 0.0182 |
| Total measured-stage time (s) | 621.69 $\pm$ 6.22 | 629.60 $\pm$ 0.91 | 1.0128 $\pm$ 0.0110 |

The byte counts and peak-memory plateaus are identical across the four
repetitions because the tensor shapes, allocator path, and isolated peak are
deterministic in this workload. This zero observed variance is not a claim of
zero uncertainty on other hardware. Timing does vary: SRQ's analytic stage is
about 81.4% slower, but the sum of measured stages is only about 1.3% slower
because feature extraction takes roughly 600 seconds per worker and dominates
the end-to-end path.
