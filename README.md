# SOHO-CL

SOHO-CL is an advanced continual learning framework.

## Project Structure

- `data/`: Contains datasets (will be downloaded automatically by the script).
- `models/`: Source code for the backbone network, `flyhash.py`, and `soho.py`.
- `methods/`: Training logic and update mechanisms (e.g., `base_cl.py`, `flycl.py`, `sohocl.py`).
- `utils/`: Evaluation metrics, data utilities, and training helpers.
- `configs/`: Configuration files (YAML or Python) for reproducible experiments.
- `notebooks/`: Kaggle notebooks or exploratory scripts (e.g., `run_flycl_kaggle.ipynb`).
- `main.py`: Main CLI entry point for training and evaluation.

## Getting Started

## T-SOHO research prototype

Run synthetic CPU validation with `python -m pytest -q` and `python tools/tsoho_runner.py --tiny-synthetic`.
For the Kaggle Phase-4 workflow, open `notebooks/phase4_tsoho_kaggle.ipynb` and follow `docs/research/KAGGLE_PHASE4_RUNBOOK.md`.

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Training:**
   ```bash
   python main.py --config configs/flycl_cifar100.yaml
   ```

3. **Kaggle Execution:**
   Upload or sync this repository to Kaggle and run the `notebooks/run_flycl_kaggle.ipynb` notebook.
