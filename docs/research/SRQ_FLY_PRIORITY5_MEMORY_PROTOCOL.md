# SRQ-FLY Priority 5: whole-process memory protocol

## Question

Does SRQ-FLY reduce actual GPU memory, rather than only the byte count of its
persistent tensors, when compared with Exact FLY at the same width 10,000?

Priority 2B already measured isolated analytic workers on synthetic streams.
Priority 5 measures the full train-only path on CIFAR-100: frozen ViT loading,
training-feature extraction, backbone release, ten analytic updates, and one
fixed probe. Exact FLY and the locked P2B SRQ backend run in fresh processes.
During the shared extraction stage, each pooled 768D batch is copied to CPU
immediately. This breaks the CLS view's alias to the full ViT token storage and
prevents an implementation artifact from retaining every historical token
tensor on GPU. Exact FLY and SRQ use the identical extraction path.

## Locked pairing

Both methods use the same:

- frozen ViT-B/16 checkpoint and preprocessing;
- 50,000 CIFAR-100 training images; the test dataset is never instantiated;
- seed 2025, class order, width 10,000 sparse projection, WTA coding level 0.3;
- synaptic degree 300 and Ridge lambda `1e6`;
- ten task groups.

The only intervention is analytic-state representation and its update:

- `exact_fly_10000`: dense float32 Gram and Cholesky solve;
- `srq_fly_p2b_10000`: mixed INT8/FP32 upper square-root factor, blocked QR,
  and streaming block quantization batch 64.

## Memory quantities

The report keeps four quantities separate:

1. `persistent_state_bytes`: tensors retained by the learner after update;
2. PyTorch `max_memory_allocated` and `max_memory_reserved`, by stage;
3. NVML peak memory attributed to the isolated worker PID, by stage and over
   the whole process;
4. NVML device-wide peak and baseline-adjusted device peak as diagnostics.

Process-attributed NVML is the primary whole-process metric. Device-wide
memory can include the Colab kernel or unrelated GPU clients. PyTorch counters
cover allocator-managed tensors but not every CUDA library allocation.

The protocol deliberately does not require the whole-process peak to fall.
Frozen-backbone extraction is common and may dominate both methods. It instead
requires a reduced analytic-stage peak and reports the end-to-end result even
if the shared extraction peak hides the state saving.

## Gates and interpretation

The run stops unless both workers complete, use identical data/projection,
remain train-only, are observed sufficiently often by NVML, retain at least
98% fixed-probe prediction agreement, satisfy the solver tolerance, retain at
most 25% of Exact FLY's persistent state, and reduce both PyTorch and NVML
analytic-stage peaks by the preregistered ratios.

`PASS_PRIORITY5_MEMORY` supports a bounded claim about CIFAR-100 on the
reported GPU/software stack. It is not an accuracy result, a held-out result,
or proof that every deployment will realize the same peak reduction.
