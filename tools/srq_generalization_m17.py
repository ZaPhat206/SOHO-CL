"""M17: train-only conditioning audit of the Ridge perturbation bound.

M7 measured a randomized *action* of the cumulative system error and said so
explicitly: it is not an estimate of

    epsilon_t = || A_t^{-1} Delta_t ||_2,

which is the quantity the M8 Ridge-solution bound is conditioned on. The
manuscript therefore states a bound it never evaluates. M17 closes that hole.

For every task of the locked M6/M7 stream, at widths 10,000 and 20,000, it
measures

    lambda_max(A_t), lambda_min(A_t), kappa_2(A_t),
    epsilon_t = || A_t^{-1} Delta_t ||_2,
    bound_t   = epsilon_t / (1 - epsilon_t),

and checks bound_t against the relative weight error that M7 already recorded
for the same width and task. Passing means the derived bound really does
dominate the observed classifier perturbation; failing would indicate an error
in the derivation or in the estimator and must be reported, not tuned away.

M17 is prediction-free. It computes no accuracy, opens no test split, and
cannot change the recorded status of M6 or M7.

Cost control
------------
Delta_t is never materialised. It is applied matrix-free as

    Delta_t v = Rhat_t^T (Rhat_t v) - A_t v,

and A_t itself is released as soon as it has been factored: with A_t = L L^T,
every later product is L (L^T v). The run holds only L and the decoded
compressed factor -- never the dense exact system or the reconstructed
quantized system -- which keeps a float64 audit at width 20,000 near 9.6 GB,
inside a 16 GB T4. Both A_t and Delta_t are symmetric, so with
M = A_t^{-1} Delta_t,

    M^T M v = Delta_t ( A_t^{-1} ( A_t^{-1} ( Delta_t v ) ) ),

which costs two triangular solve pairs and two Delta-products per step.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import traceback
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.analytic_ridge import ExactGramBackend, SquareRootBackend  # noqa: E402
from tools.experiment_runner import split, train_validation_indices, validate_cache  # noqa: E402


STUDY_STATUS_PASS = "PASS_M17_CONDITIONING_AUDIT_TRAIN_ONLY"
STUDY_STATUS_FAIL = "FAIL_M17_CONDITIONING_AUDIT_TRAIN_ONLY"

TOP_KEYS = {
    "schema_version",
    "study_id",
    "dataset",
    "model_name",
    "checkpoint_sha256",
    "uses_test_set",
    "accuracy_based_selection",
    "prediction_free",
    "seed",
    "num_classes",
    "num_tasks",
    "outer_validation_fraction",
    "statistics_dtype",
    "solver_dtype",
    "audit_dtype",
    "diagnostic_widths",
    "ridge_by_width",
    "source_m6",
    "source_m7",
    "ranpac",
    "conditioning",
    "p2b",
    "gates",
}


# ---------------------------------------------------------------------------
# hashing and config
# ---------------------------------------------------------------------------
def _lock_precision() -> dict:
    """Disable reduced-precision matmul paths and record what was locked.

    The stream replay runs in float32. On Ampere and later, PyTorch may route
    float32 matmuls through TF32, which carries a 10-bit mantissa. That would
    perturb A_t by order 1e-3 while the quantization signal Delta_t is order
    1e-2, so TF32 would contaminate roughly ten percent of the measurement and
    would also make the replay disagree with the T4-class hardware the upstream
    milestones were produced on. Turing has no TF32 path, so locking it off is
    what makes the audit hardware-portable rather than merely hardware-lucky.
    """
    locked = {}
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
        locked["cuda_matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        locked["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        locked["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    try:
        torch.backends.cuda.preferred_linalg_library("magma")
    except Exception:  # noqa: BLE001 - optional backend, not all builds have it
        pass
    try:
        torch.set_float32_matmul_precision("highest")
        locked["float32_matmul_precision"] = "highest"
    except Exception:  # noqa: BLE001 - older torch
        pass
    locked["torch_version"] = torch.__version__
    locked["cuda_version"] = getattr(torch.version, "cuda", None)
    if torch.cuda.is_available():
        locked["device_name"] = torch.cuda.get_device_name(0)
        locked["device_capability"] = ".".join(
            str(x) for x in torch.cuda.get_device_capability(0)
        )
    return locked


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    return _sha256_bytes(tensor.detach().cpu().contiguous().numpy().tobytes())


def _validate_config(config: dict) -> dict:
    """Pure validation so the contract is testable without touching the disk."""
    missing = TOP_KEYS - set(config)
    if missing:
        raise ValueError(f"M17 config is missing keys: {sorted(missing)}")
    unexpected = set(config) - TOP_KEYS
    if unexpected:
        raise ValueError(f"M17 config has unexpected keys: {sorted(unexpected)}")
    if config["uses_test_set"] or config["accuracy_based_selection"]:
        raise ValueError("M17 must be train-only and free of accuracy-based selection")
    if not config.get("prediction_free"):
        raise ValueError("M17 must declare prediction_free")
    if int(config["seed"]) != 2025:
        raise ValueError("M17 uses the protocol seed 2025")
    return config


def _read_config(path: str | Path) -> dict:
    return _validate_config(json.loads(Path(path).read_text(encoding="utf-8")))


def _verify_source_artifacts(config: dict, m6_path, m7_path) -> dict:
    """Both upstream artifacts must be byte-identical to the recorded ones."""
    identity = {}
    for key, path in (("source_m6", m6_path), ("source_m7", m7_path)):
        spec = config[key]
        if path is None:
            raise ValueError(f"M17 requires the {key} artifact")
        digest = _sha256_file(path)
        if digest != spec["artifact_sha256"]:
            raise ValueError(
                f"{key} artifact SHA-256 mismatch: expected "
                f"{spec['artifact_sha256']}, found {digest}"
            )
        with zipfile.ZipFile(path) as archive:
            payload = archive.read(spec["result_member"])
        result = json.loads(payload.decode("utf-8"))
        if result.get("status") != spec["required_status"]:
            raise ValueError(
                f"{key} status is {result.get('status')!r}, expected "
                f"{spec['required_status']!r}"
            )
        if "result_sha256" in spec and _sha256_bytes(payload) != spec["result_sha256"]:
            raise ValueError(f"{key} result member SHA-256 mismatch")
        identity[key] = {
            "artifact_sha256": digest,
            "result_member_sha256": _sha256_bytes(payload),
            "status": result.get("status"),
        }
    return identity


def _load_m7_records(config: dict, m7_path) -> dict[tuple[int, int], dict]:
    """M7's recorded P2B trajectory, keyed by (width, task).

    Two fields matter. `relative_weight_error` is what the derived bound has to
    dominate. `relative_local_factor_quantization_error` is a property of the
    replay itself, so reproducing it proves that this run regenerated the same
    stream M7 did -- which is what makes the comparison valid regardless of
    which GPU either run used.
    """
    member = config["source_m7"]["trajectory_member"]
    with zipfile.ZipFile(m7_path) as archive:
        text = archive.read(member).decode("utf-8")
    recorded: dict[tuple[int, int], dict] = {}
    for row in csv.DictReader(io.StringIO(text)):
        if row["method"] != "p2b_int8":
            continue
        weight = row.get("relative_weight_error", "")
        local = row.get("relative_local_factor_quantization_error", "")
        if weight == "":
            continue
        recorded[(int(row["width"]), int(row["task"]))] = {
            "relative_weight_error": float(weight),
            "relative_local_factor_error": float(local) if local != "" else None,
        }
    if not recorded:
        raise ValueError("M17 found no P2B records in the M7 trajectory")
    return recorded


# ---------------------------------------------------------------------------
# linear-algebra diagnostics
# ---------------------------------------------------------------------------
def _cholesky(system: torch.Tensor) -> torch.Tensor:
    symmetric = (system + system.T) * 0.5
    factor, info = torch.linalg.cholesky_ex(symmetric)
    if int(info.max().item()) != 0:
        raise RuntimeError("exact Ridge system is not positive definite")
    return factor


def _solve(factor: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    return torch.cholesky_solve(rhs.unsqueeze(-1), factor).squeeze(-1)


def _power_iteration(apply, dimension, device, dtype, *, seed, max_steps, tolerance,
                     restarts):
    """Largest eigenvalue of a symmetric PSD operator, by power iteration.

    Returns (value, converged, steps, relative_change). Restarts use fresh
    random starts and the largest estimate is kept, which guards against a
    start vector that is nearly orthogonal to the leading eigenvector.
    """
    best = -math.inf
    best_converged = False
    best_steps = 0
    best_change = math.inf
    for restart in range(max(1, int(restarts))):
        generator = torch.Generator(device="cpu").manual_seed(int(seed) + restart)
        vector = torch.randn(dimension, generator=generator, dtype=torch.float64)
        vector = vector.to(device=device, dtype=dtype)
        vector = vector / torch.linalg.vector_norm(vector)
        value = 0.0
        converged = False
        change = math.inf
        steps = 0
        for step in range(int(max_steps)):
            product = apply(vector)
            norm = float(torch.linalg.vector_norm(product.double()))
            steps = step + 1
            if not math.isfinite(norm):
                break
            if norm == 0.0:
                # The operator annihilates the iterate. For a PSD operator this
                # means the leading eigenvalue is zero, which is a converged
                # answer rather than a failure -- it is what an unperturbed
                # factor produces, and must not be reported as non-convergence.
                value, converged, change = 0.0, True, 0.0
                break
            new_value = norm
            change = abs(new_value - value) / max(new_value, 1e-30)
            value = new_value
            vector = product / norm
            if change < float(tolerance):
                converged = True
                break
        if value > best:
            best, best_converged, best_steps, best_change = value, converged, steps, change
    if not math.isfinite(best):
        return 0.0, False, 0, math.inf
    return best, best_converged, best_steps, best_change


def _audit_task(*, gram, ridge, compressed_factor, dimension, device, dtype, settings):
    """All conditioning quantities for one task at one width."""
    # Form A_t only long enough to factor it, then drop it: A_t = L L^T means
    # every later product A_t v can be taken as L (L^T v). Holding one matrix
    # instead of two is what lets a float64 audit at width 20,000 fit on a 16 GB
    # card at all -- the dense pair would need about 6.4 GB on its own.
    workspace = gram.to(dtype).clone()
    workspace.diagonal().add_(ridge)
    workspace = (workspace + workspace.T) * 0.5
    factor = _cholesky(workspace)
    del workspace
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def system_product(vector):
        return factor @ (factor.T @ vector)

    seed = int(settings["power_iteration_seed"])
    max_steps = int(settings["power_iteration_max_steps"])
    tolerance = float(settings["power_iteration_tolerance"])
    restarts = int(settings["power_iteration_restarts"])

    # Cholesky fidelity: || A x - b || / || b || on one random right-hand side.
    probe_generator = torch.Generator(device="cpu").manual_seed(seed + 977)
    probe = torch.randn(dimension, generator=probe_generator, dtype=torch.float64)
    probe = probe.to(device=device, dtype=dtype)
    solved = _solve(factor, probe)
    residual = float(
        torch.linalg.vector_norm((system_product(solved) - probe).double())
        / torch.linalg.vector_norm(probe.double())
    )

    # lambda_max(A_t): power iteration on A_t itself.
    lambda_max, max_ok, max_steps_used, max_change = _power_iteration(
        system_product, dimension, device, dtype,
        seed=seed, max_steps=max_steps, tolerance=tolerance, restarts=restarts,
    )
    # lambda_min(A_t): power iteration on A_t^{-1}, then invert.
    inverse_max, min_ok, min_steps_used, min_change = _power_iteration(
        lambda v: _solve(factor, v), dimension, device, dtype,
        seed=seed + 13, max_steps=max_steps, tolerance=tolerance, restarts=restarts,
    )
    lambda_min = 1.0 / inverse_max if inverse_max > 0 else float("nan")

    # epsilon_t = || A_t^{-1} Delta_t ||_2, matrix-free in Delta_t.
    upper = compressed_factor.to(dtype)

    def delta(vector):
        return upper.T @ (upper @ vector) - system_product(vector)

    def normal_operator(vector):
        return delta(_solve(factor, _solve(factor, delta(vector))))

    epsilon_squared, eps_ok, eps_steps, eps_change = _power_iteration(
        normal_operator, dimension, device, dtype,
        seed=seed + 29, max_steps=max_steps, tolerance=tolerance, restarts=restarts,
    )
    epsilon = math.sqrt(max(epsilon_squared, 0.0))
    bound = epsilon / (1.0 - epsilon) if epsilon < 1.0 else float("inf")

    # Delta_t is formed as Rhat^T Rhat - A_t, so its own rounding error is of
    # order eps_machine * ||A_t||. After applying A_t^{-1} that becomes a floor
    # of roughly eps_machine * kappa(A_t) on epsilon_t. Below that floor the
    # power iteration is chasing arithmetic noise: it will not converge, and any
    # value it returns is meaningless. Report the floor so the measurement can
    # be judged, and treat a sub-floor result as a resolved zero rather than a
    # convergence failure.
    condition_number = lambda_max / lambda_min if lambda_min > 0 else float("inf")
    noise_floor = float(torch.finfo(dtype).eps) * condition_number
    at_noise_floor = bool(epsilon <= noise_floor)
    if at_noise_floor:
        eps_ok = True

    del factor, upper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "lambda_max": lambda_max,
        "lambda_min": lambda_min,
        "condition_number": condition_number,
        "epsilon": epsilon,
        "epsilon_noise_floor": noise_floor,
        "epsilon_at_noise_floor": at_noise_floor,
        "epsilon_over_noise_floor": (
            epsilon / noise_floor if noise_floor > 0 else float("inf")
        ),
        "ridge_perturbation_bound": bound,
        "cholesky_relative_residual": residual,
        "power_iteration_converged": bool(max_ok and min_ok and eps_ok),
        "power_iteration_steps": {
            "lambda_max": max_steps_used,
            "lambda_min": min_steps_used,
            "epsilon": eps_steps,
        },
        "power_iteration_relative_change": {
            "lambda_max": max_change,
            "lambda_min": min_change,
            "epsilon": eps_change,
        },
    }


# ---------------------------------------------------------------------------
# stream replay
# ---------------------------------------------------------------------------
def _backend(config: dict, width: int, ridge: float, mode: str, device):
    common = {
        "dimension": width,
        "ridge_lambda": ridge,
        "device": device,
        "statistics_dtype": torch.float32,
        "solver_dtype": torch.float32,
    }
    if mode == "exact":
        return ExactGramBackend(**common)
    p2b = config["p2b"]
    return SquareRootBackend(
        storage_mode="int8",
        block_size=int(p2b["block_size"]),
        group_size=int(p2b["group_size"]),
        update_panel_size=int(p2b["update_panel_size"]),
        update_trailing_chunk_size=p2b["update_trailing_chunk_size"],
        first_update_backend=p2b["first_update_backend"],
        quantization_backend=p2b["quantization_backend"],
        quantization_batch_blocks=int(p2b["quantization_batch_blocks"]),
        **common,
    )


def _encode_indices(encoder, features, indices, batch_size: int) -> torch.Tensor:
    parts = []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        parts.append(encoder(features[chunk]))
    return torch.cat(parts) if parts else encoder(features[indices])


def _run_width(*, config, width, ridge, projection, features, labels, training_parts,
               device, audit_dtype, recorded_m7):
    exact = _backend(config, width, ridge, "exact", device)
    compressed = _backend(config, width, ridge, "p2b", device)
    encoder = lambda values: torch.relu(  # noqa: E731
        values.to(device=device, dtype=torch.float32) @ projection
    )
    settings = config["conditioning"]
    records = []
    for task_id, train_indices in enumerate(training_parts, start=1):
        codes = _encode_indices(
            encoder, features, train_indices,
            int(config["ranpac"]["encode_batch_size"]),
        )
        exact.update(codes, labels[train_indices])
        compressed.update(codes, labels[train_indices])
        decoded = compressed.factor.reconstruct_upper(dtype=torch.float32)
        audit = _audit_task(
            gram=exact.gram,
            ridge=ridge,
            compressed_factor=decoded,
            dimension=width,
            device=device,
            dtype=audit_dtype,
            settings=settings,
        )
        reference = recorded_m7.get((width, task_id), {})
        measured = reference.get("relative_weight_error")
        # Replay fidelity: the local factor error is a property of this replay,
        # so agreeing with M7 proves the same stream was regenerated.
        replayed_local = float(compressed.diagnostics["relative_local_factor_error"])
        m7_local = reference.get("relative_local_factor_error")
        local_drift = (
            None if m7_local in (None, 0.0)
            else abs(replayed_local - m7_local) / abs(m7_local)
        )
        audit.update(
            task=task_id,
            width=width,
            m7_relative_weight_error=measured,
            replay_local_factor_error=replayed_local,
            m7_local_factor_error=m7_local,
            replay_local_factor_relative_drift=local_drift,
            bound_dominates_measured=(
                None if measured is None
                else bool(audit["ridge_perturbation_bound"] >= measured)
            ),
            bound_over_measured=(
                None if measured in (None, 0.0)
                else audit["ridge_perturbation_bound"] / measured
            ),
        )
        records.append(audit)
        print(
            f"M17 w={width} t={task_id} "
            f"kappa={audit['condition_number']:.4e} "
            f"eps={audit['epsilon']:.6e} "
            f"bound={audit['ridge_perturbation_bound']:.6e} "
            f"m7_weight_err={measured} "
            f"replay_drift={local_drift}",
            flush=True,
        )
        del codes, decoded
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del exact, compressed
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


# ---------------------------------------------------------------------------
# summary and gates
# ---------------------------------------------------------------------------
def _finite(value) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _summarize(width_results: list[dict], config: dict, identity: dict) -> dict:
    gates_config = config["gates"]
    expected_tasks = int(config["num_tasks"])
    all_records = [record for block in width_results for record in block["records"]]

    complete = len(width_results) == len(config["diagnostic_widths"]) and all(
        len(block["records"]) == expected_tasks for block in width_results
    )
    converged = all(record["power_iteration_converged"] for record in all_records)
    spd = all(math.isfinite(record["lambda_min"]) and record["lambda_min"] > 0
              for record in all_records)
    max_residual = max(record["cholesky_relative_residual"] for record in all_records)
    epsilon_below_one = all(record["epsilon"] < 1.0 for record in all_records)
    # Every reported epsilon must sit above the arithmetic noise floor, or the
    # audit has measured rounding error rather than quantization error.
    epsilon_resolved = all(
        not record.get("epsilon_at_noise_floor", False) for record in all_records
    )
    # Replay fidelity. If this run did not regenerate M7's stream, comparing a
    # bound computed here against a weight error recorded there is meaningless,
    # whatever hardware either run used.
    drifts = [r.get("replay_local_factor_relative_drift") for r in all_records]
    drifts = [d for d in drifts if d is not None]
    max_drift = max(drifts) if drifts else None
    replay_faithful = bool(drifts) and max_drift <= float(
        gates_config["maximum_replay_local_factor_relative_drift"]
    )
    comparable = [r for r in all_records if r["m7_relative_weight_error"] is not None]
    slack = float(gates_config.get("bound_slack_tolerance", 0.0))
    bound_ok = bool(comparable) and all(
        record["ridge_perturbation_bound"] + slack >= record["m7_relative_weight_error"]
        for record in comparable
    )

    gates = {
        "source_m6_identity": identity.get("source_m6") is not None,
        "source_m7_identity": identity.get("source_m7") is not None,
        "all_widths_and_tasks_complete": complete,
        "power_iteration_convergence": converged,
        "spd_exact_system": spd,
        "cholesky_residual": max_residual
        <= float(gates_config["maximum_cholesky_relative_residual"]),
        "epsilon_below_one": epsilon_below_one,
        "epsilon_resolved_above_noise_floor": epsilon_resolved,
        "replay_matches_m7": replay_faithful,
        "bound_dominates_measured_weight_error": bound_ok,
        "finite_diagnostics": _finite(width_results),
    }

    per_width = {}
    for block in width_results:
        records = block["records"]
        ratios = [r["bound_over_measured"] for r in records
                  if r.get("bound_over_measured") is not None]
        per_width[str(block["width"])] = {
            "condition_number_first_task": records[0]["condition_number"],
            "condition_number_final_task": records[-1]["condition_number"],
            "condition_number_growth": (
                records[-1]["condition_number"] / records[0]["condition_number"]
                if records[0]["condition_number"] > 0 else float("inf")
            ),
            "epsilon_first_task": records[0]["epsilon"],
            "epsilon_final_task": records[-1]["epsilon"],
            "epsilon_growth": (
                records[-1]["epsilon"] / records[0]["epsilon"]
                if records[0]["epsilon"] > 0 else float("inf")
            ),
            "bound_final_task": records[-1]["ridge_perturbation_bound"],
            "m7_weight_error_final_task": records[-1]["m7_relative_weight_error"],
            "minimum_bound_over_measured": min(ratios) if ratios else None,
            "maximum_bound_over_measured": max(ratios) if ratios else None,
        }

    return {
        "gates": gates,
        "status": STUDY_STATUS_PASS if all(gates.values()) else STUDY_STATUS_FAIL,
        "summary": {
            "maximum_cholesky_relative_residual": max_residual,
            "maximum_replay_local_factor_relative_drift": max_drift,
            "maximum_epsilon": max(r["epsilon"] for r in all_records),
            "per_width": per_width,
        },
    }


def _write_csv(path: Path, width_results: list[dict]) -> None:
    columns = [
        "width", "task", "lambda_max", "lambda_min", "condition_number",
        "epsilon", "epsilon_noise_floor", "epsilon_over_noise_floor",
        "epsilon_at_noise_floor", "ridge_perturbation_bound",
        "m7_relative_weight_error", "bound_over_measured",
        "bound_dominates_measured", "cholesky_relative_residual",
        "replay_local_factor_error", "m7_local_factor_error",
        "replay_local_factor_relative_drift", "power_iteration_converged",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for block in width_results:
            for record in block["records"]:
                writer.writerow({key: record.get(key) for key in columns})


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def run(args) -> dict:
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = _read_config(config_path)
    precision_lock = _lock_precision()

    if args.require_clean_git and subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip():
        raise RuntimeError("M17 requires a clean source checkout")

    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    if (feature_cache_dir / "test.pt").exists():
        raise RuntimeError("M17 refuses a visible test.pt")

    identity = _verify_source_artifacts(config, args.m6_artifact, args.m7_artifact)
    recorded_m7 = _load_m7_records(config, args.m7_artifact)

    train, _, metadata = validate_cache(
        feature_cache_dir,
        argparse.Namespace(dataset=config["dataset"], model_name=config["model_name"]),
        load_test=False,
    )
    if metadata.get("checkpoint_sha256") != config["checkpoint_sha256"]:
        raise ValueError("feature-cache checkpoint SHA-256 mismatch")
    if int(train["features"].shape[1]) != config["ranpac"]["feature_dimension"]:
        raise ValueError("M17 feature dimension mismatch")

    class_order = random.Random(config["seed"]).sample(
        list(range(config["num_classes"])), config["num_classes"]
    )
    task_indices = split(train["labels"], class_order, config["num_tasks"])
    training_parts, _ = train_validation_indices(
        train["labels"], task_indices, config["seed"],
        config["outer_validation_fraction"],
    )

    device = torch.device(args.device)
    audit_dtype = getattr(torch, args.audit_dtype)
    generator = torch.Generator(device="cpu").manual_seed(
        int(config["ranpac"]["projection_seed"])
    )
    full_projection_cpu = torch.randn(
        int(config["ranpac"]["feature_dimension"]),
        int(config["ranpac"]["maximum_expand_dimension"]),
        generator=generator,
        dtype=torch.float32,
    )
    projection_hashes = {
        str(width): _tensor_sha256(full_projection_cpu[:, :width])
        for width in config["diagnostic_widths"]
    }
    full_projection = full_projection_cpu.to(device)
    del full_projection_cpu

    features = train["features"]
    labels = train["labels"]
    width_results = []
    numerical_failure = None
    try:
        for width in config["diagnostic_widths"]:
            print(f"M17 WIDTH START {width}", flush=True)
            projection = full_projection[:, :width].contiguous()
            ridge = float(config["ridge_by_width"][str(width)])
            records = _run_width(
                config=config,
                width=width,
                ridge=ridge,
                projection=projection,
                features=features,
                labels=labels,
                training_parts=training_parts,
                device=device,
                audit_dtype=audit_dtype,
                recorded_m7=recorded_m7,
            )
            width_results.append({"width": width, "ridge_lambda": ridge,
                                  "records": records})
            del projection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception as error:  # noqa: BLE001 - recorded, then re-raised below
        numerical_failure = {
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }

    verdict = _summarize(width_results, config, identity)
    results = {
        "schema_version": 1,
        "study_id": config["study_id"],
        "status": verdict["status"] if numerical_failure is None else STUDY_STATUS_FAIL,
        "uses_test_set": False,
        "accuracy_based_selection": False,
        "prediction_free": True,
        "scope": {
            "frontend": "RanPAC Phase-2 random-ReLU analytic head",
            "comparison": "conditioning audit of the M8 Ridge perturbation bound",
            "diagnostic_not_method_selection": True,
            "epsilon_estimator": "power_iteration_on_normal_operator",
            "cannot_change_m6_or_m7": True,
        },
        "provenance": {
            "config_sha256": _sha256_file(config_path),
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
            "audit_dtype": str(audit_dtype),
            "device": str(device),
            "precision_lock": precision_lock,
            "projection_prefix_sha256": projection_hashes,
            **identity,
        },
        "numerical_failure": numerical_failure,
        "width_results": width_results,
        **{key: verdict[key] for key in ("summary", "gates")},
    }

    results_path = output_dir / "m17_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_csv(output_dir / "conditioning_audit.csv", width_results)
    print(f"M17 STATUS {results['status']}", flush=True)
    if numerical_failure is not None:
        raise RuntimeError(f"M17 numerical failure: {numerical_failure['error']}")
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="SRQ generalization M17")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "srq_generalization_m17_conditioning_audit.json"),
    )
    parser.add_argument("--feature-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--m6-artifact", required=True)
    parser.add_argument("--m7-artifact", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--audit-dtype", default="float32",
                        choices=["float32", "float64"])
    parser.add_argument("--require-clean-git", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
