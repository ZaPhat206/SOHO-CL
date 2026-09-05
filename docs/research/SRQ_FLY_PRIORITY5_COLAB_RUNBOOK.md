# SRQ-FLY Priority 5 Colab runbook

Use `notebooks/srq_fly_priority5_memory_colab.ipynb` on one Colab T4 GPU.

1. Select **Runtime -> Disconnect and delete runtime**, then reconnect with
   **Runtime -> Change runtime type -> T4 GPU**. Colab may still assign roughly
   100 MiB of CUDA context to the notebook kernel; the runner records this in
   the device baseline and attributes the primary peak to each worker PID.
2. Open a fresh copy of the notebook from the repository; do not reuse cells
   manually copied from an older Colab tab.
3. Run every cell from top to bottom without changing the config, seed, model,
   batch size, method order, gates, or polling interval.
4. If the Kaggle dataset is private, authenticate Kaggle before the download
   cell. The public artifact `zaphat206/cifar-100` normally needs no token.
5. The input cell loads one CPU batch through the exact CIFAR transform and
   checks NVML before starting the expensive work. It must print both
   `TRAIN LOADER/TRANSFORM PREFLIGHT: PASS` and `NVML PREFLIGHT: PASS`.
6. The long benchmark runs Exact FLY first and SRQ second. Each prints
   `STAGE`, then ten `TASK` lines. Feature extraction is repeated intentionally
   so each method has an independent whole-process peak.
7. Keep the browser connected. This phase is not resumable across a runtime
   reset because a partial NVML trace is not valid evidence.
8. Download `srq_fly_priority5_whole_process_memory.zip` from the last cell and
   return it for audit.

Expected runtime on a T4 is roughly 20--45 minutes, depending mainly on frozen
ViT extraction and the SRQ blocked-QR updates. No held-out image, label, or
feature is read. The ZIP excludes the dataset, checkpoint, and temporary
training features.
