# SRQ generalization M9: repeated whole-process systems audit

Status: implementation ready; the real four-repetition CIFAR-100 train-only
measurement has not yet been run. M9 does not open a test split and does not
select a model, precision, width, or threshold from accuracy.

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
