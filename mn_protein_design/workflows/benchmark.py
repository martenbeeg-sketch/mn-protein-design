from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from mn_protein_design.core.candidates import STAGE_COMPLEX_REFOLDING, STAGE_BENCHMARK, read_candidates, write_candidates
from mn_protein_design.core.gpu import docker_gpu_args, gpu_queue_resource, normalize_gpu_device
from mn_protein_design.core.jobs import (
    JobPaths,
    create_job,
    finish_job,
    mark_internal_job,
    prepare_job_for_resume,
    read_json,
    update_status,
    utc_now,
    write_json,
)
from mn_protein_design.core.structures import filter_pdb_text
from mn_protein_design.runtime import runs_root
from mn_protein_design.workflows import esm_binder as esm_binder_workflow
from mn_protein_design.workflows import chain_roles
from mn_protein_design.workflows import refolding as refolding_workflow
from mn_protein_design.workflows import target_msa as target_msa_workflow


BENCHMARK_GROUP = "benchmark"
ESMFOLD2_IMAGE = "mnprot-biohub-esm-cu128:latest"
SCORING_SCRIPTS_IMAGE = "ovo-python-structure:latest"
PYROSETTA_METRICS_IMAGE = "ovo-bindcraft:latest"
PYMOL_PYTHON = Path("/home/user/mambaforge/envs/mn-protein-design/bin/python")
AF2_INITIAL_GUESS_IMAGE = "ovo-bindcraft:latest"
COLABFOLD_IMAGE = "mnprot-colabfold-cuda12:1.6.1"
COLABFOLD_CACHE_DIR = Path("/mnt/db/reference_files/alphafold_models")
MSA_REPOSITORY_DIR = target_msa_workflow.BOLTZ_MSA_REPOSITORY_DIR
BOLTZ2_IMAGE = "ovoex-boltz2:latest"
ALPHAFAST_IMAGE = "alphafast:latest"
ALPHAFAST_DB_DIR = Path("/mnt/db/reference_files/alignment")
ALPHAFAST_WEIGHTS_DIR = Path("/mnt/db/reference_files/alphafold3")
REPO_ROOT = Path(__file__).resolve().parents[2]
BIOHUB_ESM_ROOT = Path("/mnt/db/reference_files/biohub-esm")
DE_NOVO_BINDER_SCORING_DIR = REPO_ROOT / "tools_to_implement" / "de_novo_binder_scoring"
PUBLISHED_DATASET = DE_NOVO_BINDER_SCORING_DIR / "analysis" / "data" / "prepared_training_dataset.csv"
TARGET_MSA_REQUIRED_MIN_LENGTH = 30
CHAIN_INDEXED_CONFIDENCE_FEATURE_RE = re.compile(
    r"(?:^|_)(?:"
    r"chain_\d+_(?:iptm|ptm|pae(?:_min)?|plddt)|"
    r"chain_pair_\d+_\d+_(?:iptm|ptm|pae(?:_min)?)"
    r")$",
    re.IGNORECASE,
)
FEATURE_RANKING_COLUMNS = [
    "feature",
    "direction",
    "count",
    "positive_count",
    "negative_count",
    "auroc",
    "average_precision",
    "inverse_auroc",
    "inverse_average_precision",
    "best_auroc",
    "best_auroc_direction",
    "best_average_precision",
    "positive_mean",
    "negative_mean",
]

BENCHMARK_ENGINE_PREFIXES: dict[str, tuple[str, ...]] = {
    "AF3": ("af3_", "alphafast_af3_"),
    "AF2-IG": ("af2_",),
    "Boltz-2": ("boltz2_",),
    "BoltzGen Fold": ("boltzgen_fold_",),
    "ColabFold": ("colab_",),
    "ESMFold2": ("esmfold2_",),
    "OpenFold-3": ("openfold3_",),
    "Protenix v0.5": ("protenix_",),
    "Protenix v1": ("protenix_v1_",),
    "Protenix v2": ("protenix_v2_",),
    "RF3": ("rf3_",),
    "Input": ("input_",),
}
BENCHMARK_ENGINE_ALIASES = {
    "Protenix": "Protenix v0.5",
}

BENCHMARK_ENGINE_ARTIFACTS: dict[str, tuple[str, str, str]] = {
    "AF3": ("alphafast_af3", "alphafast_af3_metrics.csv", "af3"),
    "AF2-IG": ("af2_initial_guess", "af2_initial_guess_metrics.csv", "af2"),
    "Boltz-2": ("boltz2", "boltz2_initial_guess_metrics.csv", "boltz2"),
    "BoltzGen Fold": ("boltzgen_fold", "boltzgen_fold_metrics.csv", "boltzgen_fold"),
    "ColabFold": ("colabfold", "colabfold_metrics.csv", "colab"),
    "ESMFold2": ("esmfold2", "esmfold2_metrics.csv", "esmfold2"),
    "OpenFold-3": ("openfold3", "openfold3_metrics.csv", "openfold3"),
    "Protenix v0.5": ("protenix", "protenix_metrics.csv", "protenix"),
    "Protenix v1": ("protenix_v1", "protenix_v1_metrics.csv", "protenix_v1"),
    "Protenix v2": ("protenix_v2", "protenix_v2_metrics.csv", "protenix_v2"),
    "RF3": ("rf3", "rf3_metrics.csv", "rf3"),
}


def _canonical_benchmark_engine_label(engine: object) -> str:
    text = str(engine or "").strip()
    return BENCHMARK_ENGINE_ALIASES.get(text, text)


def _is_chain_indexed_confidence_feature(column: object) -> bool:
    return bool(CHAIN_INDEXED_CONFIDENCE_FEATURE_RE.search(str(column or "")))


def _hydrate_legacy_af2_initial_guess_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add modern AF2-IG aliases that are derivable from legacy one-model runs."""
    if df.empty or not any(str(col).startswith("af2_") for col in df.columns):
        return df
    out = df.copy()

    def fill_from(target: str, source: str) -> None:
        if source not in out.columns:
            return
        source_values = out[source]
        if target in out.columns:
            out[target] = out[target].combine_first(source_values)
        else:
            out[target] = source_values

    alias_pairs = {
        "af2_average_binder_pae": "af2_binder_pae",
        "af2_model_1_binder_pae": "af2_binder_pae",
        "af2_average_binder_plddt": "af2_binder_plddt",
        "af2_model_1_binder_plddt": "af2_binder_plddt",
        "af2_average_con_loss": "af2_con_loss",
        "af2_model_1_con_loss": "af2_con_loss",
        "af2_average_i_con_loss": "af2_i_con_loss",
        "af2_model_1_i_con_loss": "af2_i_con_loss",
        "af2_average_ipae": "af2_ipae",
        "af2_model_1_ipae": "af2_ipae",
        "af2_average_iptm": "af2_iptm",
        "af2_model_1_iptm": "af2_iptm",
        "af2_average_ptm": "af2_ptm",
        "af2_model_1_ptm": "af2_ptm",
        "af2_average_target_aligned_binder_rmsd": "af2_target_aligned_binder_rmsd",
        "af2_model_1_target_aligned_binder_rmsd": "af2_target_aligned_binder_rmsd",
        "af2_average_binder_rmsd": "af2_monomer_refolding_rmsd",
        "af2_model_1_binder_rmsd": "af2_monomer_refolding_rmsd",
        "af2_model_1_interface_target_residues": "af2_interface_target_residues",
    }
    for target, source in alias_pairs.items():
        fill_from(target, source)

    legacy_mask = pd.Series(False, index=out.index)
    for source in ["af2_iptm", "af2_ptm", "af2_ipae", "af2_binder_pae", "af2_binder_plddt"]:
        if source in out.columns:
            legacy_mask = legacy_mask | out[source].notna()
    if legacy_mask.any():
        for target, value in {
            "af2_model_count": 1,
            "af2_representative_model": 1,
        }.items():
            if target not in out.columns:
                out[target] = np.nan
            out.loc[legacy_mask & out[target].isna(), target] = value
    return out


BENCHMARK_DATASET_PATH_KWARGS = {
    "input_zip",
    "input_csv",
    "input_pdb_dir",
    "candidates_jsonl",
    "source_run_dir",
    "rf3_checkpoint_path",
    "openfold3_checkpoint_path",
    "colabfold_cache_dir",
    "msa_repository_dir",
    "alphafast_db_dir",
    "alphafast_weights_dir",
}

BENCHMARK_RESUME_STAGES: tuple[tuple[str, str, str], ...] = (
    ("run_esmfold2", "ESMFold2", "esmfold2_metrics.csv"),
    ("run_af2_initial_guess", "AF2 initial guess", "af2_initial_guess_metrics.csv"),
    ("run_boltz2_initial_guess", "Boltz-2", "boltz2_initial_guess_metrics.csv"),
    ("run_rf3", "RF3", "rf3_metrics.csv"),
    ("run_openfold3", "OpenFold-3", "openfold3_metrics.csv"),
    ("run_protenix", "Protenix v0.5", "protenix_metrics.csv"),
    ("run_boltzgen_fold", "BoltzGen target-template fold", "boltzgen_fold_metrics.csv"),
    ("run_colabfold", "ColabFold", "colabfold_metrics.csv"),
    ("run_alphafast_af3", "AlphaFast AF3", "alphafast_af3_metrics.csv"),
)


def _repo_and_runs_mounts() -> list[str]:
    """Mount the repo plus the active run root when it lives outside the repo."""
    mounts = ["-v", f"{REPO_ROOT}:{REPO_ROOT}"]
    try:
        run_root = runs_root().resolve()
        repo_root = REPO_ROOT.resolve()
    except Exception:
        return mounts
    try:
        run_root.relative_to(repo_root)
    except ValueError:
        mounts.extend(["-v", f"{run_root}:{run_root}"])
    return mounts


def _csv_data_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open(newline="") as handle:
            return max(0, sum(1 for _ in csv.reader(handle)) - 1)
    except OSError:
        return 0


def _csv_binder_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["binder_id"])
    except Exception:
        return set()
    return {str(value).strip() for value in df["binder_id"].dropna() if str(value).strip()}


def _prepared_run_csv_matches_staged_input(prepared_run_csv: Path, staged_csv: Path | None) -> bool:
    if staged_csv is None or not staged_csv.exists() or not prepared_run_csv.exists():
        return False
    prepared_ids = _csv_binder_ids(prepared_run_csv)
    staged_ids = _csv_binder_ids(staged_csv)
    return bool(prepared_ids) and prepared_ids == staged_ids


def _completed_benchmark_child_runs(parent_run_dir: Path, engine: str) -> list[str]:
    matches: list[tuple[str, str]] = []
    root = runs_root()
    for metadata_path in root.glob("*/*/metadata.json"):
        metadata = read_json(metadata_path)
        if str(metadata.get("parent_run_id") or "") != parent_run_dir.name:
            continue
        if str(metadata.get("benchmark_engine") or "") != engine:
            continue
        if str(metadata.get("status") or "") != "completed":
            continue
        matches.append((str(metadata.get("updated_at") or ""), str(metadata_path.parent)))
    return [run_dir for _, run_dir in sorted(matches)]


def _recoverable_benchmark_child_job(parent_run_dir: Path, engine: str) -> JobPaths | None:
    expected_ids = _csv_binder_ids(parent_run_dir / "artifacts" / "raw" / "de_novo_binder_scoring" / "output" / "run.csv")
    matches: list[tuple[str, Path]] = []
    for metadata_path in runs_root().glob("*/*/metadata.json"):
        metadata = read_json(metadata_path)
        if str(metadata.get("parent_run_id") or "") != parent_run_dir.name:
            continue
        if str(metadata.get("benchmark_engine") or "") != engine:
            continue
        if str(metadata.get("status") or "") == "completed":
            continue
        if expected_ids:
            input_payload = read_json(metadata_path.parent / "input.json")
            inputs = input_payload.get("inputs") if isinstance(input_payload.get("inputs"), dict) else {}
            candidates_path = Path(str(inputs.get("candidates_jsonl") or ""))
            try:
                child_ids = {str(candidate.get("candidate_id") or "").strip() for candidate in read_candidates(candidates_path)}
            except Exception:
                child_ids = set()
            child_ids.discard("")
            if child_ids and child_ids != expected_ids:
                continue
        matches.append((str(metadata.get("updated_at") or ""), metadata_path.parent))
    if not matches:
        return None
    run_dir = sorted(matches)[-1][1]
    return JobPaths(task_group=run_dir.parent.name, run_id=run_dir.name, run_dir=run_dir)


def inspect_benchmark_run(run_dir: Path) -> dict[str, Any]:
    """Inspect durable artifacts and report the earliest incomplete benchmark stage."""
    run_dir = Path(run_dir).expanduser().resolve()
    metadata = read_json(run_dir / "metadata.json")
    result = read_json(run_dir / "result.json")
    request = read_json(run_dir / "worker_request.json")
    kwargs = request.get("kwargs") if isinstance(request.get("kwargs"), dict) else {}
    benchmark_dir = run_dir / "artifacts" / "benchmark"
    run_csv = run_dir / "artifacts" / "raw" / "de_novo_binder_scoring" / "output" / "run.csv"
    record_count = _csv_data_row_count(run_csv)
    stages: list[dict[str, Any]] = [
        {
            "stage": "Prepared inputs",
            "requested": True,
            "complete": record_count > 0,
            "records": record_count,
            "expected_records": record_count or None,
            "artifact": str(run_csv),
        }
    ]
    first_incomplete = ""
    for option, label, filename in BENCHMARK_RESUME_STAGES:
        requested = bool(kwargs.get(option))
        path = benchmark_dir / filename
        rows = _csv_data_row_count(path)
        complete = bool(requested and record_count > 0 and rows == record_count)
        incomplete_reason = ""
        if complete and option == "run_af2_initial_guess" and not _benchmark_fragment_child_inputs_complete(
            run_dir,
            "af2_initial_guess",
            run_csv,
        ):
            complete = False
            incomplete_reason = "Existing AF2 child inputs are missing declared target fragments."
        if complete and option == "run_boltz2_initial_guess" and not _benchmark_fragment_child_inputs_complete(
            run_dir,
            "boltz2",
            run_csv,
        ):
            complete = False
            incomplete_reason = "Existing Boltz-2 child inputs are missing declared target fragments."
        engine_key_by_option = {
            "run_rf3": "rf3",
            "run_openfold3": "openfold3",
            "run_protenix": "protenix",
            "run_protenix_v1": "protenix_v1",
            "run_protenix_v2": "protenix_v2",
            "run_boltzgen_fold": "boltzgen_fold",
        }
        engine_key = engine_key_by_option.get(option)
        if complete and engine_key and not _benchmark_fragment_child_inputs_complete(run_dir, engine_key, run_csv):
            complete = False
            incomplete_reason = f"Existing {label} child inputs are missing declared target fragments."
        if not requested:
            complete = True
        stages.append(
            {
                "stage": label,
                "requested": requested,
                "complete": complete,
                "records": rows,
                "expected_records": record_count or None,
                "artifact": str(path),
                "incomplete_reason": incomplete_reason,
            }
        )
        if requested and not complete and not first_incomplete:
            first_incomplete = label
    merged_metrics = benchmark_dir / "merged_benchmark_metrics.csv"
    merged_ranking = benchmark_dir / "merged_benchmark_feature_ranking.csv"
    postprocess_complete = (
        record_count > 0
        and _csv_data_row_count(merged_metrics) == record_count
        and merged_ranking.exists()
    )
    stages.append(
        {
            "stage": "Final metric postprocessing",
            "requested": True,
            "complete": postprocess_complete,
            "records": _csv_data_row_count(merged_metrics),
            "expected_records": record_count or None,
            "artifact": str(merged_metrics),
        }
    )
    if not postprocess_complete and not first_incomplete:
        first_incomplete = "Final metric postprocessing"
    updated_at = str(metadata.get("updated_at") or metadata.get("created_at") or "")
    age_seconds: float | None = None
    if updated_at:
        try:
            age_seconds = (datetime.now(timezone.utc) - datetime.fromisoformat(updated_at)).total_seconds()
        except ValueError:
            pass
    free_bytes = shutil.disk_usage(run_dir).free
    required_free_bytes = max(20 * 1024**3, int(record_count) * 120 * 1024**2)
    requested_stages = [stage for stage in stages if stage["requested"]]
    status_text = str(metadata.get("status") or "")
    failed_result_while_active = bool(status_text in {"running", "preparing", "queued"} and result.get("success") is False)
    stale = bool(
        status_text in {"running", "preparing", "queued"}
        and (failed_result_while_active or (age_seconds or 0) > 6 * 3600)
    )
    return {
        "run_id": run_dir.name,
        "job_code": metadata.get("job_code"),
        "status": metadata.get("status"),
        "updated_at": updated_at,
        "age_seconds": age_seconds,
        "stale": stale,
        "stale_reason": "active metadata still has failed result" if failed_result_while_active else "",
        "record_count": record_count,
        "completed_stage_count": sum(1 for stage in requested_stages if stage["complete"]),
        "requested_stage_count": len(requested_stages),
        "first_incomplete_stage": first_incomplete or None,
        "finished_correctly": bool(requested_stages and all(stage["complete"] for stage in requested_stages)),
        "can_resume": bool(record_count > 0 and first_incomplete),
        "free_bytes": free_bytes,
        "required_free_bytes": required_free_bytes,
        "stages": stages,
    }


def prepare_benchmark_run_resume(run_dir: Path, gpu_device: object | None = None) -> dict[str, Any]:
    """Mark a benchmark for artifact-aware resumption by the durable worker."""
    report = inspect_benchmark_run(run_dir)
    if not report["can_resume"]:
        raise ValueError("This benchmark has no resumable prepared dataset or is already complete.")
    request_path = Path(run_dir) / "worker_request.json"
    request = read_json(request_path)
    if not request:
        raise ValueError("worker_request.json is missing; the original benchmark settings cannot be restored.")
    kwargs = request.get("kwargs") if isinstance(request.get("kwargs"), dict) else {}
    kwargs["resume"] = True
    normalized_gpu = normalize_gpu_device(gpu_device) if gpu_device is not None else ""
    if normalized_gpu:
        kwargs["gpu_device"] = normalized_gpu
        kwargs["alphafast_gpu_device"] = normalized_gpu
        kwargs["colabfold_gpu_device"] = normalized_gpu
    request["kwargs"] = kwargs
    write_json(request_path, request)
    if normalized_gpu:
        input_path = Path(run_dir) / "input.json"
        input_payload = read_json(input_path)
        params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
        params["gpu_device"] = normalized_gpu
        params["alphafast_gpu_device"] = normalized_gpu
        params["colabfold_gpu_device"] = normalized_gpu
        params.pop("queue_resource", None)
        input_payload["params"] = params
        write_json(input_path, input_payload)
    for metadata_path in runs_root().glob("*/*/metadata.json"):
        child_metadata = read_json(metadata_path)
        if str(child_metadata.get("parent_run_id") or "") != Path(run_dir).name:
            continue
        if str(child_metadata.get("status") or "") not in {"running", "preparing", "queued"}:
            continue
        update_status(
            metadata_path.parent,
            "failed",
            worker_error="Parent benchmark was recovered after its worker stopped.",
            recovery_superseded=True,
        )
    prepare_job_for_resume(Path(run_dir))
    if normalized_gpu:
        metadata_path = Path(run_dir) / "metadata.json"
        metadata = read_json(metadata_path)
        queue_resource = gpu_queue_resource(normalized_gpu)
        if queue_resource:
            metadata["queue_resource"] = queue_resource
        else:
            metadata.pop("queue_resource", None)
        metadata["updated_at"] = utc_now()
        write_json(metadata_path, metadata)
    return report


def _jsonable_worker_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return {"__bytes__": True, "size": len(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable_worker_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_worker_value(item) for item in value]
    return value


def _stage_de_novo_benchmark_worker_inputs(job: JobPaths, kwargs: dict[str, Any]) -> dict[str, Any]:
    staged_kwargs = dict(kwargs)
    staged_dir = job.run_dir / "artifacts" / "queued_inputs"
    staged_dir.mkdir(parents=True, exist_ok=True)

    input_zip_bytes = staged_kwargs.pop("input_zip_bytes", None)
    if input_zip_bytes is not None:
        staged_zip = staged_dir / "input.zip"
        staged_zip.write_bytes(input_zip_bytes)
        staged_kwargs["input_zip"] = staged_zip

    input_csv_text = staged_kwargs.pop("input_csv_text", None)
    if input_csv_text is not None:
        staged_csv = staged_dir / "input.csv"
        staged_csv.write_text(str(input_csv_text))
        staged_kwargs["input_csv"] = staged_csv

    return staged_kwargs


def _enqueue_benchmark_worker_job(
    *,
    job_type: str,
    tool_name: str,
    input_payload: dict[str, Any],
    params_payload: dict[str, Any],
    kwargs: dict[str, Any],
    kind: str = "de_novo_binder_scoring_dataset",
    task_group: str = BENCHMARK_GROUP,
) -> Path:
    job = create_job(task_group, job_type, tool_name, input_payload, params_payload)
    staged_kwargs = _stage_de_novo_benchmark_worker_inputs(job, kwargs)
    worker_kwargs = {key: _jsonable_worker_value(value) for key, value in staged_kwargs.items()}
    write_json(
        job.run_dir / "worker_request.json",
        {
            "kind": kind,
            "kwargs": worker_kwargs,
            "path_kwargs": sorted(BENCHMARK_DATASET_PATH_KWARGS),
        },
    )
    write_json(
        job.run_dir / "command.json",
        {
            "mode": "local_worker",
            "command": ["python", "-m", "mn_protein_design.core.local_worker", "--run-dir", str(job.run_dir)],
        },
    )
    return job.run_dir


def enqueue_de_novo_binder_scoring_dataset(**kwargs: Any) -> Path:
    """Create a durable queued benchmark job for the local worker.

    The heavy benchmark workflow is intentionally not executed here. This keeps
    Streamlit reruns from interrupting Docker/GPU work while still preserving the
    same final run folder layout as a directly executed benchmark.
    """

    return _enqueue_benchmark_worker_job(
        job_type="de_novo_binder_scoring_dataset",
        tool_name="de_novo_binder_scoring_scripts",
        input_payload={
            "queued_worker_request": True,
            "input_zip": str(kwargs.get("input_zip")) if kwargs.get("input_zip") else None,
            "input_zip_uploaded": kwargs.get("input_zip_bytes") is not None,
            "input_csv": str(kwargs.get("input_csv")) if kwargs.get("input_csv") else None,
            "input_csv_uploaded": kwargs.get("input_csv_text") is not None,
            "input_pdb_dir": str(kwargs.get("input_pdb_dir")) if kwargs.get("input_pdb_dir") else None,
        },
        params_payload={
            "queued_worker": "local",
            "mode": kwargs.get("mode", "pdb_only"),
            "models": list(kwargs.get("models") or []),
            "max_records": int(kwargs.get("max_records") or 0),
            "gpu_device": normalize_gpu_device(kwargs.get("gpu_device", kwargs.get("alphafast_gpu_device", "0"))),
            "run_alphafast_af3": bool(kwargs.get("run_alphafast_af3")),
            "alphafast_use_target_templates": bool(kwargs.get("alphafast_use_target_templates", True)),
            "run_colabfold": bool(kwargs.get("run_colabfold")),
            "run_af2_initial_guess": bool(kwargs.get("run_af2_initial_guess")),
            "run_boltz2_initial_guess": bool(kwargs.get("run_boltz2_initial_guess")),
            "run_esmfold2": bool(kwargs.get("run_esmfold2")),
            "num_loops": int(kwargs.get("num_loops") or 0),
            "num_sampling_steps": int(kwargs.get("num_sampling_steps") or 0),
            "esmfold2_modes": list(kwargs.get("esmfold2_modes") or []),
            "esmfold2_use_target_msa": bool(kwargs.get("esmfold2_use_target_msa")),
            "run_rf3": bool(kwargs.get("run_rf3")),
            "rf3_use_target_msa": bool(kwargs.get("rf3_use_target_msa", True)),
            "rf3_use_target_template": bool(kwargs.get("rf3_use_target_template", True)),
            "run_openfold3": bool(kwargs.get("run_openfold3")),
            "openfold3_checkpoint_path": str(kwargs.get("openfold3_checkpoint_path")) if kwargs.get("openfold3_checkpoint_path") else None,
            "openfold3_use_target_msa": bool(kwargs.get("openfold3_use_target_msa", True)),
            "openfold3_num_diffusion_samples": int(kwargs.get("openfold3_num_diffusion_samples") or 0),
            "openfold3_num_model_seeds": int(kwargs.get("openfold3_num_model_seeds") or 0),
            "openfold3_num_recycles": int(kwargs.get("openfold3_num_recycles") or 0),
            "openfold3_use_msa_server": bool(kwargs.get("openfold3_use_msa_server")),
            "run_protenix": bool(kwargs.get("run_protenix")),
            "protenix_use_msa": bool(kwargs.get("protenix_use_msa", True)),
            "run_protenix_v1": bool(kwargs.get("run_protenix_v1")),
            "protenix_v1_model_name": str(kwargs.get("protenix_v1_model_name") or refolding_workflow.PROTENIX_V1_MODEL),
            "protenix_v1_use_msa": bool(kwargs.get("protenix_v1_use_msa", True)),
            "protenix_v1_use_template": bool(kwargs.get("protenix_v1_use_template")),
            "protenix_v1_use_default_params": bool(kwargs.get("protenix_v1_use_default_params", True)),
            "protenix_v1_cycle": int(kwargs.get("protenix_v1_cycle") or 0),
            "protenix_v1_diffusion_steps": int(kwargs.get("protenix_v1_diffusion_steps") or 0),
            "protenix_v1_samples": int(kwargs.get("protenix_v1_samples") or 0),
            "run_protenix_v2": bool(kwargs.get("run_protenix_v2")),
            "protenix_v2_model_name": str(kwargs.get("protenix_v2_model_name") or refolding_workflow.PROTENIX_V2_MODEL),
            "protenix_v2_use_msa": bool(kwargs.get("protenix_v2_use_msa", True)),
            "protenix_v2_use_template": bool(kwargs.get("protenix_v2_use_template")),
            "protenix_v2_use_default_params": bool(kwargs.get("protenix_v2_use_default_params", True)),
            "protenix_v2_cycle": int(kwargs.get("protenix_v2_cycle") or 0),
            "protenix_v2_diffusion_steps": int(kwargs.get("protenix_v2_diffusion_steps") or 0),
            "protenix_v2_samples": int(kwargs.get("protenix_v2_samples") or 0),
            "run_boltzgen_fold": bool(kwargs.get("run_boltzgen_fold")),
        },
        kwargs=kwargs,
    )


def _safe_id(value: object, fallback: str = "benchmark") -> str:
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value or "").strip())
    return text.strip("_") or fallback


def _safe_int(value: object) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _json_clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_clean(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if pd.isna(value):
            return None
        return float(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _truthy_label(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        if float(value) == 1.0:
            return 1
        if float(value) == 0.0:
            return 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "binder", "binds", "positive", "pos"}:
        return 1
    if text in {"0", "false", "no", "n", "nonbinder", "non-binder", "negative", "neg", "unknown"}:
        return 0 if text != "unknown" else None
    return None


def _split_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            payload = json.loads(text)
            if isinstance(payload, list):
                return [str(item).strip() for item in payload if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [token.strip() for token in text.replace(";", ",").split(",") if token.strip()]


def _row_declared_target_fragment_chains(row: pd.Series) -> list[str]:
    chains = _split_list(row.get("target_chains"))
    declared: list[str] = []
    for chain in chains:
        value = row.get(f"target_subchain_{chain}_seq")
        if value is None or pd.isna(value):
            continue
        if str(value).strip() and chain not in declared:
            declared.append(chain)
    return declared


def _benchmark_child_input_dir(parent_run_dir: Path, engine: str) -> Path | None:
    if engine == "af2_initial_guess":
        path = parent_run_dir / "artifacts" / "engines" / "af2_initial_guess" / "artifacts" / "raw" / "af2_initial_guess" / "inputs"
    elif engine == "boltz2":
        path = parent_run_dir / "artifacts" / "engines" / "boltz2" / "artifacts" / "raw" / "boltz2_initial_guess" / "input_pdbs"
    elif engine in {"rf3", "openfold3", "protenix", "protenix_v1", "protenix_v2", "boltzgen_fold"}:
        path = parent_run_dir / "artifacts" / "engines" / engine / "artifacts" / "raw" / engine / "inputs"
    else:
        return None
    return path if path.exists() else None


def _benchmark_fragment_child_inputs_complete(parent_run_dir: Path, engine: str, run_csv: Path) -> bool:
    """Return False when old child inputs dropped declared target fragments."""
    input_dir = _benchmark_child_input_dir(parent_run_dir, engine)
    if input_dir is None or not run_csv.exists():
        return True
    try:
        df = pd.read_csv(run_csv)
    except Exception:
        return True
    for _, row in df.iterrows():
        expected_chains = _row_declared_target_fragment_chains(row)
        if len(expected_chains) <= 1:
            continue
        safe_id = _safe_id(row.get("binder_id") or row.get("candidate_id"))
        target_chains_path = input_dir / f"{safe_id}.target_chains.txt"
        observed_chains = set(_split_list(target_chains_path.read_text(errors="ignore") if target_chains_path.exists() else ""))
        chain_map_path = input_dir / f"{safe_id}.chain_map.json"
        if chain_map_path.exists():
            chain_map = read_json(chain_map_path)
            fragments = chain_map.get("target_fragments")
            if isinstance(fragments, list):
                observed_chains.update(
                    str(fragment.get("engine_chain") or "")
                    for fragment in fragments
                    if isinstance(fragment, dict) and fragment.get("engine_chain")
                )
        pdb_path = input_dir / f"{safe_id}.pdb"
        if pdb_path.exists():
            observed_chains.update(refolding_workflow._structure_chains(pdb_path))
        if not set(expected_chains).issubset(observed_chains):
            return False
    return True


def _benchmark_fragment_resume_complete(parent_run_dir: Path, engine: str, run_csv: Path) -> bool:
    if _benchmark_fragment_child_inputs_complete(parent_run_dir, engine, run_csv):
        return True
    return False


def _resolve_input_path(value: object, base_dir: Path) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path if path.exists() else None


def _copy_input(path: Path | None, input_dir: Path, candidate_id: str, suffix: str) -> Path | None:
    if path is None or not path.exists():
        return None
    target = input_dir / f"{_safe_id(candidate_id)}_{suffix}{path.suffix}"
    if path.resolve() != target.resolve():
        shutil.copy2(path, target)
    return target


def _row_value(row: dict[str, Any], *keys: str) -> Any:
    lowered = {str(key).lower(): value for key, value in row.items()}
    for key in keys:
        if key.lower() in lowered:
            value = lowered[key.lower()]
            if value is not None and str(value).strip() != "":
                return value
    return None


def _relative_candidate_path(source_run_dir: Path, value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = source_run_dir / path
    return path if path.exists() else None


def _parse_structure_for_alignment(path: Path):
    from Bio.PDB import MMCIFParser, PDBParser

    if path.suffix.lower() in {".cif", ".mmcif"}:
        return MMCIFParser(QUIET=True).get_structure(path.stem, str(path))
    return PDBParser(QUIET=True).get_structure(path.stem, str(path))


def _ca_atoms_for_alignment(structure: object, chains: list[str]) -> list[object]:
    atoms: list[object] = []
    wanted = {str(chain) for chain in chains}
    for model in structure:
        for chain in model:
            if chain.id not in wanted:
                continue
            for residue in chain:
                residue_id = residue.id
                if residue_id[0] != " " or "CA" not in residue:
                    continue
                atoms.append(residue["CA"])
        break
    return atoms


def _chain_ca_sequence_atoms_for_alignment(chain: object) -> tuple[str, list[object]]:
    from Bio.SeqUtils import seq1

    sequence = ""
    atoms: list[object] = []
    for residue in chain:
        residue_id = residue.id
        if residue_id[0] != " " or "CA" not in residue:
            continue
        try:
            sequence += seq1(residue.resname)
        except Exception:  # noqa: BLE001 - unknown residues still keep alignment positions.
            sequence += "X"
        atoms.append(residue["CA"])
    return sequence, atoms


def _sequence_aligned_ca_atoms_for_alignment(
    fixed_structure: object,
    moving_structure: object,
    fixed_chains: list[str],
    moving_chains: list[str],
) -> tuple[list[object], list[object], str]:
    from Bio.Align import PairwiseAligner

    fixed_model = next(fixed_structure.get_models(), None)
    moving_model = next(moving_structure.get_models(), None)
    if fixed_model is None or moving_model is None:
        return [], [], "missing model"

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -0.5
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5

    fixed_atoms: list[object] = []
    moving_atoms: list[object] = []
    notes: list[str] = []
    for fixed_chain_id, moving_chain_id in zip(fixed_chains, moving_chains):
        if fixed_chain_id not in fixed_model or moving_chain_id not in moving_model:
            notes.append(f"{moving_chain_id}->{fixed_chain_id}: missing chain")
            continue
        fixed_seq, fixed_chain_atoms = _chain_ca_sequence_atoms_for_alignment(fixed_model[fixed_chain_id])
        moving_seq, moving_chain_atoms = _chain_ca_sequence_atoms_for_alignment(moving_model[moving_chain_id])
        if not fixed_seq or not moving_seq:
            notes.append(f"{moving_chain_id}->{fixed_chain_id}: no CA sequence")
            continue
        alignment = aligner.align(fixed_seq, moving_seq)[0]
        before = len(fixed_atoms)
        for fixed_block, moving_block in zip(alignment.aligned[0], alignment.aligned[1]):
            fixed_start, fixed_end = int(fixed_block[0]), int(fixed_block[1])
            moving_start, moving_end = int(moving_block[0]), int(moving_block[1])
            count = min(fixed_end - fixed_start, moving_end - moving_start)
            for offset in range(count):
                fixed_atoms.append(fixed_chain_atoms[fixed_start + offset])
                moving_atoms.append(moving_chain_atoms[moving_start + offset])
        notes.append(f"{moving_chain_id}->{fixed_chain_id}: {len(fixed_atoms) - before} sequence-matched CA")
    return fixed_atoms, moving_atoms, "; ".join(notes)


def _align_target_override_to_candidate_frame(
    *,
    source_complex: Path,
    source_target_chains: list[str],
    target_pdb: Path,
    target_chains: list[str],
    output_path: Path,
) -> tuple[Path, str]:
    """Place a target override in the same coordinate frame as the candidate target."""

    if not source_target_chains or not target_chains:
        return target_pdb, "target override alignment skipped: missing source or override target chains"
    try:
        from Bio.PDB import PDBIO, Superimposer

        fixed_structure = _parse_structure_for_alignment(source_complex)
        moving_structure = _parse_structure_for_alignment(target_pdb)
        fixed_atoms, moving_atoms, pairing_note = _sequence_aligned_ca_atoms_for_alignment(
            fixed_structure,
            moving_structure,
            source_target_chains,
            target_chains,
        )
        if not fixed_atoms or not moving_atoms:
            fixed_atoms = _ca_atoms_for_alignment(fixed_structure, source_target_chains)
            moving_atoms = _ca_atoms_for_alignment(moving_structure, target_chains)
            pairing_note = "positional CA fallback"
        common_count = min(len(fixed_atoms), len(moving_atoms))
        if common_count < 3:
            return (
                target_pdb,
                f"target override alignment skipped: only {common_count} common CA atoms",
            )
        superimposer = Superimposer()
        superimposer.set_atoms(fixed_atoms[:common_count], moving_atoms[:common_count])
        superimposer.apply(moving_structure.get_atoms())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = PDBIO()
        writer.set_structure(moving_structure)
        writer.save(str(output_path))
        return (
            output_path,
            (
                f"target override aligned to candidate target on {common_count} CA atoms "
                f"({pairing_note}); RMSD {superimposer.rms:.3f} A"
            ),
        )
    except Exception as exc:  # noqa: BLE001 - keep staging usable and record why alignment was skipped.
        return target_pdb, f"target override alignment skipped: {exc}"


def _complex_chain_contact_summary(
    path: Path,
    binder_chains: list[str],
    target_chains: list[str],
    *,
    cutoff: float = 8.0,
) -> tuple[bool, float | None]:
    from Bio.PDB import NeighborSearch

    structure = _parse_structure_for_alignment(path)
    model = next(structure.get_models(), None)
    if model is None:
        return False, None
    binder_atoms = [
        atom
        for chain_id in binder_chains
        if chain_id in model
        for atom in model[chain_id].get_atoms()
        if str(getattr(atom, "element", "")).upper() != "H"
    ]
    target_atoms = [
        atom
        for chain_id in target_chains
        if chain_id in model
        for atom in model[chain_id].get_atoms()
        if str(getattr(atom, "element", "")).upper() != "H"
    ]
    if not binder_atoms or not target_atoms:
        return False, None
    neighbor_search = NeighborSearch(target_atoms)
    for binder_atom in binder_atoms:
        if neighbor_search.search(binder_atom.coord, cutoff, level="A"):
            return True, 0.0
    return False, None


def _stage_candidate_set_as_repo_dataset(
    *,
    source_run_dir: Path,
    candidates_jsonl: Path,
    staged_dir: Path,
    max_candidates: int = 0,
    selected_candidate_ids: list[str] | None = None,
    target_override_pdb: Path | None = None,
    target_override_chains: list[str] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    source_run_dir = Path(source_run_dir).expanduser().resolve()
    candidates = read_candidates(Path(candidates_jsonl).expanduser())
    selected_set = {str(candidate_id) for candidate_id in (selected_candidate_ids or []) if str(candidate_id).strip()}
    if selected_set:
        candidates = [
            candidate
            for candidate in candidates
            if str(candidate.get("candidate_id") or "").strip() in selected_set
        ]
    if max_candidates and int(max_candidates) > 0:
        candidates = candidates[: int(max_candidates)]
    if not candidates:
        raise ValueError("No candidates are available for refolding evaluation.")

    input_pdb_dir = staged_dir / "input_pdbs"
    target_pdb_dir = staged_dir / "target_pdbs"
    aligned_target_dir = staged_dir / "aligned_target_overrides"
    job_run_dir = staged_dir.parents[2]
    input_pdb_dir.mkdir(parents=True, exist_ok=True)
    target_pdb_dir.mkdir(parents=True, exist_ok=True)
    csv_path = staged_dir / "input.csv"
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    chain_role_findings: list[dict[str, Any]] = []
    chain_role_warnings: list[dict[str, Any]] = []

    def _csv_safe_metric_value(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        try:
            return json.dumps(value, sort_keys=True)
        except TypeError:
            return str(value)

    def _add_candidate_metric_columns(row: dict[str, Any], metrics: dict[str, Any]) -> None:
        protected = set(row)
        for key, value in metrics.items():
            column = str(key or "").strip()
            if not column or column in protected:
                column = f"source_metric_{column}" if column else ""
            if not column:
                continue
            row[column] = _csv_safe_metric_value(value)
        ranking_score = metrics.get("boltzbio_ranking_score")
        ranking_rank = metrics.get("boltzbio_rank")
        if ranking_score not in {None, ""} and "source_ranking_score" not in row:
            row["source_ranking_score"] = _csv_safe_metric_value(ranking_score)
        if ranking_rank not in {None, ""} and "source_rank" not in row:
            row["source_rank"] = _csv_safe_metric_value(ranking_rank)

    source_input_payload = read_json(source_run_dir / "input.json")
    source_inputs = source_input_payload.get("inputs") if isinstance(source_input_payload.get("inputs"), dict) else {}
    source_params = source_input_payload.get("params") if isinstance(source_input_payload.get("params"), dict) else {}
    upstream_source_run_dir = None
    upstream_source_text = str(source_inputs.get("source_run_dir") or source_params.get("source_run_dir") or "").strip()
    if upstream_source_text:
        upstream_source_candidate = Path(upstream_source_text)
        upstream_source_run_dir = upstream_source_candidate if upstream_source_candidate.is_absolute() else source_run_dir / upstream_source_candidate
        upstream_source_run_dir = upstream_source_run_dir.resolve()

    for index, candidate in enumerate(candidates, start=1):
        candidate_id = str(candidate.get("candidate_id") or f"candidate_{index}").strip()
        binder_id = re.sub(
            r"[^a-z0-9_]",
            "_",
            _safe_id(candidate_id, fallback=f"candidate_{index}").lower(),
        )
        target_pdb = Path(target_override_pdb).expanduser().resolve() if target_override_pdb else refolding_workflow._target_pdb_for_candidate(source_run_dir, candidate)
        if target_pdb is None or not target_pdb.exists():
            skipped.append({"candidate_id": candidate_id, "reason": "missing_target_pdb"})
            continue
        binder_sequence = str(candidate.get("binder_sequence") or "").strip()
        if not binder_sequence:
            try:
                binder_sequence = refolding_workflow._candidate_binder_sequence(source_run_dir, candidate)
            except ValueError:
                binder_sequence = ""
        if not binder_sequence:
            skipped.append({"candidate_id": candidate_id, "reason": "missing_binder_sequence"})
            continue
        source_complex = _relative_candidate_path(
            source_run_dir,
            candidate.get("complex_pdb") or candidate.get("binder_pdb"),
        )
        declared_binder_chains = _split_list(candidate.get("binder_chains"))
        declared_target_chains = _split_list(candidate.get("target_chains"))
        metrics = dict(candidate.get("metrics") or {})
        raw_metadata = dict(candidate.get("raw_metadata") or {})
        has_source_binder_structure = source_complex is not None and source_complex.exists() and bool(declared_binder_chains)
        if has_source_binder_structure:
            source_complex_sequences = refolding_workflow._sequences_by_chain(source_complex)
            source_binder_sequence = "".join(
                source_complex_sequences.get(chain, "")
                for chain in declared_binder_chains
            )
            if source_binder_sequence != binder_sequence:
                if source_binder_sequence:
                    raw_metadata.setdefault(
                        "binder_sequence_warning",
                        "input binder_sequence did not match declared binder chain; using structure-derived sequence",
                    )
                    raw_metadata.setdefault("input_binder_sequence_length", len(binder_sequence))
                    binder_sequence = source_binder_sequence
                else:
                    skipped.append(
                        {
                            "candidate_id": candidate_id,
                            "reason": "binder_structure_sequence_does_not_match_csv",
                            "csv_length": len(binder_sequence),
                            "structure_length": len(source_binder_sequence),
                        }
                    )
                    continue

        legacy_capacity_chains = _split_list(candidate.get("legacy_capacity_binder_chains")) or _split_list(
            raw_metadata.get("legacy_capacity_binder_chains")
        )
        parent_source_candidate = raw_metadata.get("source_candidate") if isinstance(raw_metadata.get("source_candidate"), dict) else {}
        parent_source_complex = (
            _relative_candidate_path(upstream_source_run_dir, parent_source_candidate.get("complex_pdb"))
            if upstream_source_run_dir is not None and parent_source_candidate
            else None
        )
        parent_source_binder_chains = _split_list(parent_source_candidate.get("binder_chains")) if parent_source_candidate else []
        parent_source_target_chains = _split_list(parent_source_candidate.get("target_chains")) if parent_source_candidate else []
        capacity_target_only = bool(raw_metadata.get("capacity_target_only"))
        if capacity_target_only and source_complex is not None:
            biological_target_chains = (
                _split_list(raw_metadata.get("staged_target_chains"))
                or declared_target_chains
                or _split_list(raw_metadata.get("biological_target_chains"))
                or legacy_capacity_chains
                or declared_binder_chains
            )
            preserve_target_fragments = len(biological_target_chains) > 1
            staged_target_chains = list(biological_target_chains if preserve_target_fragments else ["A"])
            next_atom = 1
            staged_lines: list[str] = []
            if preserve_target_fragments:
                for chain in biological_target_chains:
                    chain_lines, next_atom = refolding_workflow._renumber_structure_chain(
                        source_complex,
                        chain,
                        next_atom,
                        {chain},
                    )
                    if chain_lines:
                        staged_lines.extend(chain_lines + ["TER"])
            else:
                chain_lines, next_atom = refolding_workflow._renumber_structure_chain(
                    source_complex,
                    "A",
                    next_atom,
                    set(declared_binder_chains),
                )
                if chain_lines:
                    staged_lines.extend(chain_lines + ["TER"])
            if not staged_lines:
                skipped.append({"candidate_id": candidate_id, "reason": "target_only_chain_staging_failed"})
                continue
            canonical_complex_pdb = input_pdb_dir / f"{binder_id}.pdb"
            canonical_complex_pdb.write_text("\n".join(staged_lines + ["END", ""]))
            staged_sequences = refolding_workflow._sequences_by_chain(canonical_complex_pdb)
            refolding_workflow._write_engine_chain_map(
                input_pdb_dir,
                binder_id,
                binder_source_chains=[],
                target_source_chains=biological_target_chains,
                target_engine_chains=staged_target_chains,
                target_fragments=[
                    {
                        "source_chain": chain,
                        "engine_chain": chain,
                        "start": 1,
                        "end": len(staged_sequences.get(chain, "")),
                        "is_fragment": preserve_target_fragments,
                    }
                    for chain in staged_target_chains
                ],
            )
            row = {
                "binder_id": binder_id,
                "original_binder_id": candidate_id,
                "target_id": str(raw_metadata.get("target_id") or metrics.get("target_id") or source_complex.stem),
                "target_source": "capacity_target_only",
                "capacity_target_only": True,
                "binder": "",
                "label": "",
                "source": str(candidate.get("source_tool") or raw_metadata.get("import_name") or "candidate_set"),
                "binder_chain": staged_target_chains[0],
                "binder_chains": json.dumps(staged_target_chains),
                "target_chains": json.dumps(staged_target_chains),
                "target_only_chains": json.dumps(staged_target_chains),
                "segment_ids": json.dumps(staged_target_chains),
                "target_chain_range": json.dumps([]),
                "msa_info": json.dumps([f"{chain}:run_msa" for chain in staged_target_chains]),
                "complex_pdb": str(canonical_complex_pdb),
                "target_pdb": str(canonical_complex_pdb),
                "source_target_pdb": str(target_pdb),
                "source_input_run_dir": os.path.relpath(source_run_dir, job_run_dir),
                "source_input_complex_pdb": os.path.relpath(source_complex, job_run_dir),
                "source_input_binder_chains": json.dumps(declared_binder_chains),
                "source_input_target_chains": json.dumps(biological_target_chains),
                "refolding_target_source_chains": json.dumps(biological_target_chains),
                "engine_target_chains": json.dumps(staged_target_chains),
                "chain_role_schema": chain_roles.CHAIN_ROLE_SCHEMA_EXPLICIT_V2,
                "aligned_target_pdb": "",
                "target_alignment": "capacity target-only fragments" if preserve_target_fragments else "capacity target-only monomer",
                "target_alignment_reference_pdb": "",
                "target_alignment_reference_chains": json.dumps(staged_target_chains),
                "source_target_chains": json.dumps(biological_target_chains),
                "binder_sequence": binder_sequence,
            }
            for chain in staged_target_chains:
                sequence = staged_sequences.get(chain, "")
                row[f"{chain}_seq"] = sequence
                row[f"{chain}_length"] = len(sequence)
            _add_candidate_metric_columns(row, metrics)
            findings = chain_roles.validate_chain_roles(
                candidate_id=candidate_id,
                binder_chains=[],
                target_chains=staged_target_chains,
                structure_chains=refolding_workflow._structure_chains(canonical_complex_pdb),
                schema=chain_roles.CHAIN_ROLE_SCHEMA_EXPLICIT_V2,
                target_only=True,
            )
            warnings = chain_roles.problem_findings(findings)
            if findings:
                row["chain_role_finding_count"] = len(findings)
                chain_role_findings.extend(findings)
            if warnings:
                row["chain_role_warning_count"] = len(warnings)
                chain_role_warnings.extend(warnings)
            rows.append(row)
            continue

        candidate_target_chains = _split_list(candidate.get("target_chains"))
        available_target_chains = refolding_workflow._structure_chains(target_pdb)
        declared_target_chains = list(target_override_chains or []) or candidate_target_chains
        source_target_chains = [
            chain for chain in declared_target_chains if chain in available_target_chains
        ] or available_target_chains
        if not source_target_chains:
            skipped.append({"candidate_id": candidate_id, "reason": "target_pdb_has_no_protein_chains"})
            continue
        target_staging_pdb = target_pdb
        target_alignment_note = ""
        if target_override_pdb and source_complex is not None and candidate_target_chains:
            target_staging_pdb, target_alignment_note = _align_target_override_to_candidate_frame(
                source_complex=source_complex,
                source_target_chains=candidate_target_chains,
                target_pdb=target_pdb,
                target_chains=source_target_chains,
                output_path=aligned_target_dir / f"{binder_id}.pdb",
            )
        elif target_override_pdb:
            target_alignment_note = "target override used without input-complex alignment"

        source_target_fragment_specs = refolding_workflow._target_fragment_specs(
            target_staging_pdb,
            source_target_chains,
        )
        if not source_target_fragment_specs:
            source_target_fragment_specs = [
                {
                    "source_chain": source_chain,
                    "engine_chain": source_chain,
                    "start": 0,
                    "end": 0,
                    "is_fragment": False,
                }
                for source_chain in source_target_chains
            ]
        target_engine_chain_order = list(chain_roles.TARGET_CHAIN_ORDER)
        if len(source_target_fragment_specs) > len(target_engine_chain_order):
            skipped.append({"candidate_id": candidate_id, "reason": "too_many_target_chains_for_engine_staging"})
            continue
        target_fragment_specs = [
            {
                **spec,
                "engine_chain": engine_chain,
            }
            for spec, engine_chain in zip(source_target_fragment_specs, target_engine_chain_order, strict=False)
        ]
        engine_target_chains = [str(spec["engine_chain"]) for spec in target_fragment_specs]
        staged_target_pdb = target_pdb_dir / f"{binder_id}.pdb"
        target_lines: list[str] = []
        next_atom = 1
        for spec in target_fragment_specs:
            residue_range = (
                (int(spec["start"]), int(spec["end"]))
                if spec.get("is_fragment")
                else None
            )
            chain_lines, next_atom = refolding_workflow._renumber_structure_chain(
                target_staging_pdb,
                str(spec["engine_chain"]),
                next_atom,
                {str(spec["source_chain"])},
                residue_range,
            )
            if chain_lines:
                target_lines.extend(chain_lines + ["TER"])
        if not target_lines:
            skipped.append({"candidate_id": candidate_id, "reason": "target_chain_staging_failed"})
            continue
        staged_target_pdb.write_text("\n".join(target_lines + ["END", ""]))

        target_sequences = refolding_workflow._sequences_by_chain(staged_target_pdb)
        if any(not sequence for sequence in target_sequences.values()):
            skipped.append({"candidate_id": candidate_id, "reason": "target_sequence_extraction_failed"})
            staged_target_pdb.unlink(missing_ok=True)
            continue

        canonical_complex_pdb = input_pdb_dir / f"{binder_id}.pdb"
        binder_lines: list[str] = []
        binder_engine_chains = chain_roles.assign_binder_engine_chains(
            declared_binder_chains or ["sequence"],
            reserved=engine_target_chains,
        )
        binder_engine_chain = binder_engine_chains[0] if binder_engine_chains else "Z"
        if has_source_binder_structure and source_complex is not None:
            binder_lines, next_atom = refolding_workflow._append_role_chains(
                source_complex,
                source_chains=declared_binder_chains,
                engine_chains=binder_engine_chains,
                next_atom=1,
            )
            if not binder_lines:
                skipped.append({"candidate_id": candidate_id, "reason": "binder_chain_staging_failed"})
                staged_target_pdb.unlink(missing_ok=True)
                continue
        else:
            next_atom = 1
        canonical_target_lines: list[str] = []
        for spec in target_fragment_specs:
            residue_range = (
                (int(spec["start"]), int(spec["end"]))
                if spec.get("is_fragment")
                else None
            )
            chain_lines, next_atom = refolding_workflow._renumber_structure_chain(
                target_staging_pdb,
                str(spec["engine_chain"]),
                next_atom,
                {str(spec["source_chain"])},
                residue_range,
            )
            if chain_lines:
                canonical_target_lines.extend(chain_lines + ["TER"])
        canonical_lines = binder_lines + canonical_target_lines + ["END", ""]
        canonical_complex_pdb.write_text("\n".join(canonical_lines))
        has_target_contact, minimum_target_distance = (True, None)
        if has_source_binder_structure:
            has_target_contact, minimum_target_distance = _complex_chain_contact_summary(
                canonical_complex_pdb,
                binder_engine_chains,
                engine_target_chains,
            )
        if target_override_pdb and has_source_binder_structure and not has_target_contact:
            skipped.append(
                {
                    "candidate_id": candidate_id,
                    "reason": "override_target_not_in_contact_with_input_binder_after_alignment",
                    "minimum_atom_distance": minimum_target_distance,
                    "target_alignment": target_alignment_note,
                }
            )
            canonical_complex_pdb.unlink(missing_ok=True)
            staged_target_pdb.unlink(missing_ok=True)
            continue

        refolding_workflow._write_engine_chain_map(
            input_pdb_dir,
            binder_id,
            binder_source_chains=declared_binder_chains if has_source_binder_structure else ["sequence"],
            target_source_chains=[str(spec["source_chain"]) for spec in target_fragment_specs],
            target_engine_chains=engine_target_chains,
            binder_engine_chains=binder_engine_chains,
            target_fragments=target_fragment_specs,
        )
        row = {
            "binder_id": binder_id,
            "original_binder_id": candidate_id,
            "target_id": str(raw_metadata.get("target_id") or metrics.get("target_id") or target_pdb.stem),
            "target_source": "override" if target_override_pdb else "candidate",
            "binder": "",
            "label": "",
            "source": str(candidate.get("source_tool") or raw_metadata.get("import_name") or "candidate_set"),
            "binder_chain": binder_engine_chain,
            "binder_chains": json.dumps(binder_engine_chains),
            "target_chains": json.dumps(engine_target_chains),
            "segment_ids": json.dumps(engine_target_chains),
            "target_chain_range": json.dumps(
                [f"1:{len(target_sequences[chain])}" for chain in engine_target_chains]
            ),
            "msa_info": json.dumps([f"{chain}:no_msa" for chain in binder_engine_chains] + [f"{chain}:run_msa" for chain in engine_target_chains]),
            "complex_pdb": str(canonical_complex_pdb),
            "target_pdb": str(staged_target_pdb),
            "source_target_pdb": str(target_pdb),
            "source_input_run_dir": os.path.relpath(source_run_dir, job_run_dir),
            "source_input_complex_pdb": os.path.relpath(source_complex, job_run_dir) if source_complex is not None else "",
            "source_parent_run_dir": os.path.relpath(upstream_source_run_dir, job_run_dir) if upstream_source_run_dir is not None else "",
            "source_parent_complex_pdb": os.path.relpath(parent_source_complex, job_run_dir) if parent_source_complex is not None else "",
            "source_parent_binder_chains": json.dumps(parent_source_binder_chains),
            "source_parent_target_chains": json.dumps(parent_source_target_chains),
            "source_input_binder_chains": json.dumps(declared_binder_chains),
            "source_input_target_chains": json.dumps(candidate_target_chains),
            "refolding_target_source_chains": json.dumps(source_target_chains),
            "engine_target_chains": json.dumps(engine_target_chains),
            "chain_role_schema": chain_roles.CHAIN_ROLE_SCHEMA_EXPLICIT_V2,
            "target_fragment_specs": json.dumps(target_fragment_specs),
            "aligned_target_pdb": str(target_staging_pdb) if target_staging_pdb != target_pdb else "",
            "target_alignment": target_alignment_note,
            "target_alignment_reference_pdb": str(source_complex) if target_override_pdb and source_complex is not None else "",
            "target_alignment_reference_chains": json.dumps(candidate_target_chains) if target_override_pdb else "[]",
            "source_target_chains": json.dumps(source_target_chains),
            "binder_sequence": binder_sequence,
            f"{binder_engine_chain}_seq": binder_sequence,
            f"{binder_engine_chain}_length": len(binder_sequence),
        }
        for chain in engine_target_chains:
            row[f"target_subchain_{chain}_seq"] = target_sequences[chain]
            row[f"target_subchain_{chain}_len"] = len(target_sequences[chain])
            row.setdefault(f"{chain}_seq", target_sequences[chain])
            row.setdefault(f"{chain}_length", len(target_sequences[chain]))
        _add_candidate_metric_columns(row, metrics)
        findings = chain_roles.validate_chain_roles(
            candidate_id=candidate_id,
            binder_chains=binder_engine_chains,
            target_chains=engine_target_chains,
            structure_chains=refolding_workflow._structure_chains(canonical_complex_pdb),
            schema=chain_roles.CHAIN_ROLE_SCHEMA_EXPLICIT_V2,
            target_only=False,
        )
        warnings = chain_roles.problem_findings(findings)
        if findings:
            row["chain_role_finding_count"] = len(findings)
            chain_role_findings.extend(findings)
        if warnings:
            row["chain_role_warning_count"] = len(warnings)
            chain_role_warnings.extend(warnings)
        rows.append(row)
    if not rows:
        raise ValueError("None of the selected candidates had a usable binder sequence and target PDB.")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        staged_dir / "candidate_staging_summary.json",
        {
            "source_run_dir": str(source_run_dir),
            "candidates_jsonl": str(candidates_jsonl),
            "selected_candidate_ids": list(selected_set),
            "candidate_count": len(candidates),
            "staged_count": len(rows),
            "skipped_count": len(skipped),
            "skipped": skipped[:100],
            "chain_role_finding_count": len(chain_role_findings),
            "chain_role_findings": chain_role_findings[:100],
            "chain_role_warning_count": len(chain_role_warnings),
            "chain_role_warnings": chain_role_warnings[:100],
            "target_override_pdb": str(target_override_pdb) if target_override_pdb else None,
            "target_override_chains": list(target_override_chains or []),
        },
    )
    return csv_path, input_pdb_dir, {
        "candidate_count": len(candidates),
        "staged_count": len(rows),
        "skipped_count": len(skipped),
        "chain_role_finding_count": len(chain_role_findings),
        "chain_role_warning_count": len(chain_role_warnings),
    }


def enqueue_candidate_refolding_evaluation(
    *,
    source_run_dir: Path,
    candidates_jsonl: Path,
    max_candidates: int = 0,
    selected_candidate_ids: list[str] | None = None,
    evaluation_name: str = "Refolding evaluation",
    job_type: str = "refolding_evaluation",
    tool_name: str = "refolding_evaluation_engines",
    **kwargs: Any,
) -> Path:
    """Queue a benchmark-style engine evaluation for an unlabeled candidate set."""
    queued_candidate_count = len(selected_candidate_ids or [])
    queued_total_residues: int | None = None
    try:
        candidates = read_candidates(candidates_jsonl)
        if selected_candidate_ids:
            selected_set = {str(candidate_id) for candidate_id in selected_candidate_ids}
            candidates = [
                candidate
                for candidate in candidates
                if str(candidate.get("candidate_id") or "") in selected_set
            ]
        elif max_candidates and int(max_candidates) > 0:
            candidates = candidates[: int(max_candidates)]
        queued_candidate_count = len(candidates) if candidates else queued_candidate_count
        queued_total_residues = _candidate_total_residues(candidates)
    except Exception:
        if not queued_candidate_count:
            queued_candidate_count = int(max_candidates or 0)

    input_payload = {
        "queued_worker_request": True,
        "source_run_dir": str(source_run_dir),
        "candidates_jsonl": str(candidates_jsonl),
        "selected_candidate_ids": list(selected_candidate_ids or []),
        "target_override_pdb": str(kwargs.get("target_override_pdb")) if kwargs.get("target_override_pdb") else None,
        "target_override_chains": list(kwargs.get("target_override_chains") or []),
        "target_refolding": bool(kwargs.get("target_refolding")),
    }
    params_payload = {
        "queued_worker": "local",
        "evaluation_name": evaluation_name,
        "evaluation_mode": "refolding_validation",
        "target_refolding": bool(kwargs.get("target_refolding")),
        "max_candidates": int(max_candidates or 0),
        "selected_candidate_count": queued_candidate_count,
        "candidate_count": queued_candidate_count,
        "total_residues": queued_total_residues,
        "gpu_device": normalize_gpu_device(kwargs.get("gpu_device", kwargs.get("alphafast_gpu_device", "0"))),
        "target_override_pdb": str(kwargs.get("target_override_pdb")) if kwargs.get("target_override_pdb") else None,
        "target_override_chains": list(kwargs.get("target_override_chains") or []),
        "models": list(kwargs.get("models") or []),
        "run_alphafast_af3": bool(kwargs.get("run_alphafast_af3")),
        "alphafast_use_target_templates": bool(kwargs.get("alphafast_use_target_templates", True)),
        "run_colabfold": bool(kwargs.get("run_colabfold")),
        "run_af2_initial_guess": bool(kwargs.get("run_af2_initial_guess")),
        "af2_use_initial_guess": bool(kwargs.get("af2_use_initial_guess")),
        "run_boltz2_initial_guess": bool(kwargs.get("run_boltz2_initial_guess")),
        "boltz2_use_target_msa": bool(kwargs.get("boltz2_use_target_msa", True)),
        "run_esmfold2": bool(kwargs.get("run_esmfold2")),
        "num_loops": int(kwargs.get("num_loops") or 0),
        "num_sampling_steps": int(kwargs.get("num_sampling_steps") or 0),
        "esmfold2_modes": list(kwargs.get("esmfold2_modes") or []),
        "esmfold2_use_target_msa": bool(kwargs.get("esmfold2_use_target_msa")),
        "run_rf3": bool(kwargs.get("run_rf3")),
        "run_openfold3": bool(kwargs.get("run_openfold3")),
        "run_protenix": bool(kwargs.get("run_protenix")),
        "protenix_use_msa": bool(kwargs.get("protenix_use_msa", True)),
        "run_protenix_v1": bool(kwargs.get("run_protenix_v1")),
        "protenix_v1_model_name": str(kwargs.get("protenix_v1_model_name") or refolding_workflow.PROTENIX_V1_MODEL),
        "protenix_v1_use_msa": bool(kwargs.get("protenix_v1_use_msa", True)),
        "protenix_v1_use_template": bool(kwargs.get("protenix_v1_use_template")),
        "protenix_v1_use_default_params": bool(kwargs.get("protenix_v1_use_default_params", True)),
        "protenix_v1_cycle": int(kwargs.get("protenix_v1_cycle") or 0),
        "protenix_v1_diffusion_steps": int(kwargs.get("protenix_v1_diffusion_steps") or 0),
        "protenix_v1_samples": int(kwargs.get("protenix_v1_samples") or 0),
        "run_protenix_v2": bool(kwargs.get("run_protenix_v2")),
        "protenix_v2_model_name": str(kwargs.get("protenix_v2_model_name") or refolding_workflow.PROTENIX_V2_MODEL),
        "protenix_v2_use_msa": bool(kwargs.get("protenix_v2_use_msa", True)),
        "protenix_v2_use_template": bool(kwargs.get("protenix_v2_use_template")),
        "protenix_v2_use_default_params": bool(kwargs.get("protenix_v2_use_default_params", True)),
        "protenix_v2_cycle": int(kwargs.get("protenix_v2_cycle") or 0),
        "protenix_v2_diffusion_steps": int(kwargs.get("protenix_v2_diffusion_steps") or 0),
        "protenix_v2_samples": int(kwargs.get("protenix_v2_samples") or 0),
        "run_boltzgen_fold": bool(kwargs.get("run_boltzgen_fold")),
        "require_real_target_msa": bool(kwargs.get("require_real_target_msa", True)),
        "shared_msa_source": str(kwargs.get("colabfold_msa_source") or "msa_repository_then_alphafast_mmseqs_gpu"),
        "alphafast_query_only_msa": bool(kwargs.get("alphafast_query_only_msa")),
    }
    worker_kwargs = {
        **kwargs,
        "source_run_dir": Path(source_run_dir),
        "candidates_jsonl": Path(candidates_jsonl),
        "max_records": int(max_candidates or 0),
        "selected_candidate_ids": list(selected_candidate_ids or []),
        "mode": "seq_only_csv",
        "generate_inputs": True,
        "job_type": job_type,
        "tool_name": tool_name,
    }
    return _enqueue_benchmark_worker_job(
        job_type=job_type,
        tool_name=tool_name,
        input_payload=input_payload,
        params_payload=params_payload,
        kwargs=worker_kwargs,
        kind="candidate_refolding_evaluation",
        task_group=str(kwargs.get("task_group") or BENCHMARK_GROUP),
    )


def _candidate_total_residues(candidates: list[dict[str, Any]]) -> int | None:
    total = 0
    for candidate in candidates:
        metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
        raw_metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
        value = (
            _safe_int(metrics.get("total_length"))
            or _safe_int(metrics.get("sequence_length"))
            or (
                (_safe_int(metrics.get("binder_length")) or 0)
                + (_safe_int(metrics.get("target_length")) or 0)
            )
            or _safe_int(candidate.get("binder_length"))
            or _safe_int(raw_metadata.get("sequence_length"))
        )
        total += int(value or 0)
    return total or None


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _stage_colabfold_target_templates(run_csv: Path, output_dir: Path, max_records: int = 0) -> tuple[Path | None, int]:
    input_pdb_dir = run_csv.parent / "input_pdbs"
    if not input_pdb_dir.exists():
        return None, 0
    template_dir = output_dir / "ColabFold" / "target_templates"
    if template_dir.exists():
        shutil.rmtree(template_dir)
    template_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(run_csv)
    if max_records > 0:
        df = df.head(int(max_records))
    staged_keys: set[str] = set()
    count = 0
    for row in df.to_dict(orient="records"):
        binder_id = str(_row_value(row, "binder_id", "candidate_id", "id") or "").strip()
        if not binder_id:
            continue
        source = input_pdb_dir / f"{_safe_id(binder_id)}.pdb"
        if not source.exists():
            source = input_pdb_dir / f"{binder_id}.pdb"
        if not source.exists():
            continue
        target_chains = _split_list(_row_value(row, "target_chains", "target_chain"))
        if not target_chains:
            continue
        target_id = str(_row_value(row, "target_id") or binder_id).strip() or binder_id
        source_key = f"{_safe_id(target_id)}__{'_'.join(_safe_id(chain) for chain in target_chains)}"
        if source_key in staged_keys:
            continue
        pdb_text = source.read_text(errors="ignore")
        target_text = filter_pdb_text(
            pdb_text,
            keep_chains=set(target_chains),
            remove_waters=True,
            remove_hetero=True,
        )
        if "ATOM" not in target_text:
            continue
        template_key = f"t{count + 1:03d}"
        (template_dir / f"{template_key}.pdb").write_text(target_text)
        staged_keys.add(source_key)
        count += 1
    return (template_dir if count else None), count


def _run_csv_rows_by_binder_id(run_csv: Path, max_records: int = 0) -> dict[str, dict[str, Any]]:
    if not run_csv.exists():
        return {}
    df = pd.read_csv(run_csv)
    if max_records > 0:
        df = df.head(int(max_records))
    rows: dict[str, dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        binder_id = str(_row_value(row, "binder_id", "candidate_id", "id") or "").strip()
        if binder_id:
            rows[_safe_id(binder_id)] = row
            rows[binder_id] = row
    return rows


def _protein_ids(protein: dict[str, Any]) -> list[str]:
    value = protein.get("id")
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _stage_af3_target_templates(
    *,
    payload: dict[str, Any],
    source_name: str,
    row: dict[str, Any] | None,
    input_pdb_dir: Path,
    template_dir: Path,
) -> dict[str, int]:
    metrics = {
        "protein_count": 0,
        "templated_protein_count": 0,
        "template_count": 0,
        "missing_source_pdb_count": 0,
        "missing_target_chain_count": 0,
        "template_build_failed_count": 0,
    }
    if not row:
        return metrics
    binder_id = str(_row_value(row, "binder_id", "candidate_id", "id") or source_name).strip() or source_name
    source_pdb = input_pdb_dir / f"{_safe_id(binder_id)}.pdb"
    if not source_pdb.exists():
        source_pdb = input_pdb_dir / f"{binder_id}.pdb"
    if not source_pdb.exists():
        metrics["missing_source_pdb_count"] += 1
        return metrics
    target_chains = _split_list(_row_value(row, "target_chains", "target_chain"))
    if not target_chains:
        metrics["missing_target_chain_count"] += 1
        return metrics

    template_source_pdb = source_pdb
    source_chains = set(refolding_workflow._structure_chains(source_pdb))
    missing_source_chains = [chain for chain in target_chains if chain not in source_chains]
    if missing_source_chains:
        declared_sequences = {
            chain: str(_row_value(row, f"target_subchain_{chain}_seq", f"{chain}_seq") or "").strip()
            for chain in target_chains
        }
        fragment_specs = refolding_workflow._target_fragment_specs_from_declared_sequences(
            source_pdb,
            declared_sequences,
            target_chains,
        )
        if fragment_specs:
            template_dir.mkdir(parents=True, exist_ok=True)
            fragmented_source = template_dir / f"{_safe_id(source_name)}__fragmented_target.pdb"
            fragment_lines: list[str] = []
            next_atom = 1
            for spec in fragment_specs:
                residue_range = (int(spec["start"]), int(spec["end"]))
                chain_lines, next_atom = refolding_workflow._renumber_structure_chain(
                    source_pdb,
                    str(spec["engine_chain"]),
                    next_atom,
                    {str(spec["source_chain"])},
                    residue_range,
                )
                if chain_lines:
                    fragment_lines.extend(chain_lines + ["TER"])
            if fragment_lines:
                fragmented_source.write_text("\n".join(fragment_lines + ["END", ""]))
                template_source_pdb = fragmented_source

    for sequence in payload.get("sequences", []):
        protein = sequence.get("protein") if isinstance(sequence, dict) else None
        if not isinstance(protein, dict):
            continue
        metrics["protein_count"] += 1
        ids = _protein_ids(protein)
        template_chains = [chain for chain in target_chains if chain in ids] or (
            target_chains if len(target_chains) == 1 and len(ids) == 1 and ids[0] in target_chains else []
        )
        if not template_chains:
            continue
        query_sequence = str(protein.get("sequence") or "")
        template_path = template_dir / f"{_safe_id(source_name)}__{'_'.join(_safe_id(chain) for chain in template_chains)}.json"
        info = refolding_workflow._pdb_chains_to_protenix_template_json(
            pdb_path=template_source_pdb,
            chains=template_chains,
            query_sequence=query_sequence,
            output_path=template_path,
            entry_id=f"af3_template_{_safe_id(source_name)}",
        )
        if not info:
            metrics["template_build_failed_count"] += 1
            continue
        try:
            template_payload = read_json(template_path)
        except Exception:
            metrics["template_build_failed_count"] += 1
            continue
        if not isinstance(template_payload, list) or not template_payload:
            metrics["template_build_failed_count"] += 1
            continue
        protein["templates"] = template_payload
        metrics["templated_protein_count"] += 1
        metrics["template_count"] += len(template_payload)
    return metrics


def _prepare_benchmark_records(
    input_csv: Path,
    input_dir: Path,
    max_records: int = 0,
    base_dir: Path | None = None,
) -> list[dict[str, Any]]:
    rows = list(csv.DictReader(input_csv.open(newline="", encoding="utf-8-sig", errors="replace")))
    if max_records > 0:
        rows = rows[: int(max_records)]
    records: list[dict[str, Any]] = []
    base_dir = base_dir or input_csv.parent
    for index, row in enumerate(rows, start=1):
        candidate_id = _safe_id(_row_value(row, "binder_id", "candidate_id", "id") or f"candidate_{index:05d}")
        target_pdb = _copy_input(
            _resolve_input_path(_row_value(row, "target_pdb", "target_path"), base_dir),
            input_dir,
            candidate_id,
            "target",
        )
        complex_pdb = _copy_input(
            _resolve_input_path(_row_value(row, "complex_pdb", "complex_path", "pdb"), base_dir),
            input_dir,
            candidate_id,
            "complex",
        )
        binder_pdb = _copy_input(
            _resolve_input_path(_row_value(row, "binder_pdb", "binder_path"), base_dir),
            input_dir,
            candidate_id,
            "binder",
        )
        msa_paths: dict[str, str] = {}
        for key, value in row.items():
            if not key.startswith("msa_path_"):
                continue
            chain = key.removeprefix("msa_path_")
            raw_msa = str(value or "").strip()
            if not chain or not raw_msa or raw_msa.lower() == "no_msa":
                continue
            resolved_msa = _resolve_input_path(raw_msa, base_dir)
            msa_paths[chain] = str(resolved_msa or raw_msa)
        label = _truthy_label(_row_value(row, "label", "binder", "is_binder", "binds"))
        records.append(
            {
                "candidate_id": candidate_id,
                "target_id": str(_row_value(row, "target_id", "target") or ""),
                "source": str(_row_value(row, "source", "dataset") or ""),
                "label": label,
                "label_raw": str(_row_value(row, "label", "binder", "is_binder", "binds") or ""),
                "binder_sequence": str(_row_value(row, "binder_sequence", "sequence", "binder_seq") or "").strip(),
                "target_pdb": str(target_pdb) if target_pdb else None,
                "complex_pdb": str(complex_pdb) if complex_pdb else None,
                "binder_pdb": str(binder_pdb) if binder_pdb else None,
                "binder_chains": _split_list(_row_value(row, "binder_chains", "binder_chain")) or ["A"],
                "target_chains": _split_list(_row_value(row, "target_chains", "target_chain")) or [],
                "target_source": str(_row_value(row, "target_source") or ""),
                "capacity_target_only": str(_row_value(row, "capacity_target_only") or "").strip().lower()
                in {"1", "true", "yes", "y"},
                "msa_paths": msa_paths,
                "hotspots": _split_list(_row_value(row, "hotspots", "target_hotspots")),
                "raw_input": row,
            }
        )
    if not records:
        raise ValueError("Benchmark CSV did not contain any rows.")
    return records


def _load_esmfold2_msa(msa_cls: Any, msa_path: object, expected_sequence: str) -> tuple[Any | None, str | None]:
    raw_path = str(msa_path or "").strip()
    if not raw_path or raw_path.lower() == "no_msa":
        return None, None
    path = Path(raw_path)
    if not path.exists():
        return None, f"missing:{raw_path}"
    try:
        msa = msa_cls.from_a3m(path)
    except Exception as exc:
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".a3m", delete=True) as handle:
                refolding_workflow._copy_a3m_match_columns_only(path, Path(handle.name))
                handle.flush()
                msa = msa_cls.from_a3m(Path(handle.name))
        except Exception as sanitized_exc:
            return None, f"invalid:{path.name}:{exc}; sanitized_invalid:{sanitized_exc}"
        sanitized_note = f"sanitized:{path.name}"
    else:
        sanitized_note = None
    query = "".join(str(getattr(msa, "query", "") or "").upper().split())
    expected = "".join(str(expected_sequence or "").upper().split())
    if query and expected and query != expected:
        return None, f"query_mismatch:{path.name}"
    return msa, sanitized_note


def _manual_auroc(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _manual_average_precision(labels: list[int], scores: list[float]) -> float | None:
    if not any(label == 1 for label in labels) or not any(label == 0 for label in labels):
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    positives_seen = 0
    precision_sum = 0.0
    total_positives = sum(1 for label in labels if label == 1)
    for rank, (_score, label) in enumerate(ordered, start=1):
        if label == 1:
            positives_seen += 1
            precision_sum += positives_seen / rank
    return precision_sum / total_positives if total_positives else None


def _benchmark_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [
        row
        for row in rows
        if row.get("label") in {0, 1} and row.get("esmfold2_benchmark_score") is not None
    ]
    labels = [int(row["label"]) for row in labeled]
    scores = [float(row["esmfold2_benchmark_score"]) for row in labeled]
    summary: dict[str, Any] = {
        "record_count": len(rows),
        "labeled_count": len(labeled),
        "positive_count": sum(1 for label in labels if label == 1),
        "negative_count": sum(1 for label in labels if label == 0),
        "auroc": _manual_auroc(labels, scores),
        "average_precision": _manual_average_precision(labels, scores),
    }
    if rows:
        ranked = sorted(rows, key=lambda row: row.get("esmfold2_benchmark_score") or 0.0, reverse=True)
        summary["top_candidate_id"] = ranked[0].get("candidate_id")
        summary["top_score"] = ranked[0].get("esmfold2_benchmark_score")
    return summary


def _metric_column_summary(df: pd.DataFrame, label_col: str, max_columns: int = 250) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if label_col not in df.columns:
        raise ValueError(f"Label column not found: {label_col}")
    label_values = df.loc[:, label_col]
    if isinstance(label_values, pd.DataFrame):
        label_values = label_values.iloc[:, 0]
    if not isinstance(label_values, pd.Series):
        label_values = pd.Series([label_values] * len(df), index=df.index)
    labels_raw = [_truthy_label(value) for value in label_values.tolist()]
    positive_count = sum(1 for label in labels_raw if label == 1)
    negative_count = sum(1 for label in labels_raw if label == 0)
    if positive_count == 0 or negative_count == 0:
        return [], {
            "record_count": int(len(df)),
            "labeled_count": int(positive_count + negative_count),
            "positive_count": int(positive_count),
            "negative_count": int(negative_count),
            "numeric_feature_count": 0,
            "scored_feature_count": 0,
            "top_feature": None,
            "top_feature_average_precision": None,
            "top_feature_auroc": None,
        }

    def _numeric_column_values(frame: pd.DataFrame, column: object) -> pd.Series:
        extracted = frame.loc[:, column]
        if isinstance(extracted, pd.DataFrame):
            extracted = extracted.iloc[:, 0]
        if not isinstance(extracted, pd.Series):
            extracted = pd.Series([extracted] * len(frame), index=frame.index)
        return pd.to_numeric(extracted, errors="coerce")

    def _is_label_leakage_column(column: object) -> bool:
        text = str(column)
        if text in {"label", "binder", "is_binder", "binds"}:
            return True
        for prefixes in BENCHMARK_ENGINE_PREFIXES.values():
            for prefix in prefixes:
                if not text.startswith(prefix):
                    continue
                remainder = text[len(prefix) :]
                if remainder == "binder":
                    return True
                parts = [part for part in remainder.split("_") if part]
                if len(parts) == 2 and parts[1] == "binder":
                    return True
        return False

    numeric_cols: list[str] = []
    label_like_cols = {label_col, "label", "binder", "is_binder", "binds"}
    for column in df.columns:
        if (
            str(column) in label_like_cols
            or _is_label_leakage_column(column)
            or _is_chain_indexed_confidence_feature(column)
        ):
            continue
        values = _numeric_column_values(df, column)
        if values.notna().sum() >= 2:
            numeric_cols.append(str(column))
    rows: list[dict[str, Any]] = []
    for column in numeric_cols[:max_columns]:
        values = _numeric_column_values(df, column)
        valid_labels: list[int] = []
        valid_scores: list[float] = []
        for label, score in zip(labels_raw, values.tolist()):
            if label in {0, 1} and pd.notna(score):
                valid_labels.append(int(label))
                valid_scores.append(float(score))
        if not valid_scores:
            continue
        auroc = _manual_auroc(valid_labels, valid_scores)
        ap = _manual_average_precision(valid_labels, valid_scores)
        inverse_auroc = _manual_auroc(valid_labels, [-score for score in valid_scores])
        inverse_ap = _manual_average_precision(valid_labels, [-score for score in valid_scores])
        best_auroc_direction = "higher"
        best_auroc = auroc
        if inverse_auroc is not None and (best_auroc is None or inverse_auroc > best_auroc):
            best_auroc_direction = "lower"
            best_auroc = inverse_auroc
        best_direction = "higher"
        best_ap = ap
        if inverse_ap is not None and (best_ap is None or inverse_ap > best_ap):
            best_direction = "lower"
            best_ap = inverse_ap
        positives = [score for label, score in zip(valid_labels, valid_scores) if label == 1]
        negatives = [score for label, score in zip(valid_labels, valid_scores) if label == 0]
        rows.append(
            {
                "feature": column,
                "direction": best_direction,
                "count": len(valid_scores),
                "positive_count": len(positives),
                "negative_count": len(negatives),
                "auroc": auroc,
                "average_precision": ap,
                "inverse_auroc": inverse_auroc,
                "inverse_average_precision": inverse_ap,
                "best_auroc": best_auroc,
                "best_auroc_direction": best_auroc_direction,
                "best_average_precision": best_ap,
                "positive_mean": float(np.mean(positives)) if positives else None,
                "negative_mean": float(np.mean(negatives)) if negatives else None,
            }
        )
    rows.sort(
        key=lambda row: (
            row.get("best_average_precision") or 0.0,
            row.get("best_auroc") or 0.0,
        ),
        reverse=True,
    )
    positive_count = sum(1 for label in labels_raw if label == 1)
    negative_count = sum(1 for label in labels_raw if label == 0)
    ranking_unavailable_reason = None
    if not positive_count or not negative_count:
        ranking_unavailable_reason = "Feature ranking requires at least one binder and one nonbinder."
    elif not rows:
        ranking_unavailable_reason = "No numeric feature had enough labeled values for ranking."
    summary = {
        "record_count": int(len(df)),
        "label_column": label_col,
        "labeled_count": sum(1 for label in labels_raw if label in {0, 1}),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "numeric_feature_count": len(numeric_cols),
        "scored_feature_count": len(rows),
        "ranking_available": bool(rows),
        "ranking_unavailable_reason": ranking_unavailable_reason,
        "top_feature": rows[0]["feature"] if rows else None,
        "top_feature_average_precision": rows[0]["best_average_precision"] if rows else None,
        "top_feature_auroc": rows[0]["best_auroc"] if rows else None,
        "top_feature_direction": rows[0]["direction"] if rows else None,
        "ranking_metric": "average_precision",
    }
    return rows, summary


def _write_feature_ranking(path: Path, feature_rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(feature_rows, columns=FEATURE_RANKING_COLUMNS).to_csv(path, index=False)


def _collection_merge_keys(df: pd.DataFrame) -> list[str]:
    return ["target_id", "binder_id"] if {"target_id", "binder_id"}.issubset(df.columns) else ["binder_id"]


def _collection_metadata_columns(df: pd.DataFrame) -> list[str]:
    metadata_names = {
        "binder_id",
        "original_binder_id",
        "target_id",
        "binder",
        "label",
        "source",
        "binder_chain",
        "binder_chains",
        "target_chains",
        "target_only_chains",
        "legacy_capacity_binder_chains",
        "source_input_binder_chains",
        "source_input_target_chains",
        "source_parent_binder_chains",
        "source_parent_target_chains",
        "refolding_target_source_chains",
        "engine_target_chains",
        "source_target_chains",
        "chain_role_schema",
        "target_fragment_specs",
        "A_seq",
        "A_length",
        "B_length",
        "target_chain_range",
        "segment_ids",
    }
    return [col for col in df.columns if col in metadata_names]


def _benchmark_engine_for_column(column: str) -> str | None:
    engine_prefixes = [
        (engine, prefix)
        for engine, prefixes in BENCHMARK_ENGINE_PREFIXES.items()
        for prefix in prefixes
    ]
    for engine, prefix in sorted(engine_prefixes, key=lambda item: len(item[1]), reverse=True):
        if column.startswith(prefix):
            return engine
    return None


ESMFOLD2_COLLECTION_VARIANTS: dict[tuple[int, int], str] = {
    (3, 50): "fast",
    (10, 68): "standard",
    (20, 68): "careful",
    (10, 200): "high_diffusion",
    (3, 200): "design_rank",
}


def _collection_variant_token(text: object) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "").strip().lower()).strip("_")
    return token or "variant"


def _source_esmfold2_variant_token(source_run_dir: Path, metadata: dict[str, Any], order: int) -> str:
    input_payload = read_json(source_run_dir / "input.json")
    params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
    worker_payload = read_json(source_run_dir / "worker_request.json")
    worker_kwargs = worker_payload.get("kwargs") if isinstance(worker_payload.get("kwargs"), dict) else {}
    loops = _safe_int(params.get("num_loops", worker_kwargs.get("num_loops")))
    steps = _safe_int(params.get("num_sampling_steps", worker_kwargs.get("num_sampling_steps")))
    if loops is not None and steps is not None:
        return ESMFOLD2_COLLECTION_VARIANTS.get((loops, steps), f"custom_{loops}_{steps}")
    job_code = str(metadata.get("job_code") or "").strip()
    return _collection_variant_token(job_code or f"run_{order}")


def _source_engine_variant_token(source: dict[str, Any], engine: str) -> str:
    engine = _canonical_benchmark_engine_label(engine)
    metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
    order = int(source.get("order") or 0)
    if engine == "ESMFold2":
        return _source_esmfold2_variant_token(Path(str(source.get("source_run_dir") or "")), metadata, order)
    source_run_dir = Path(str(source.get("source_run_dir") or ""))
    input_payload = read_json(source_run_dir / "input.json")
    params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
    worker_payload = read_json(source_run_dir / "worker_request.json")
    worker_kwargs = worker_payload.get("kwargs") if isinstance(worker_payload.get("kwargs"), dict) else {}

    def get_value(*names: str, default: object = "") -> object:
        for name in names:
            if name in params and params.get(name) not in (None, ""):
                return params.get(name)
            if name in worker_kwargs and worker_kwargs.get(name) not in (None, ""):
                return worker_kwargs.get(name)
        return default

    setting_parts: list[str] = []
    if engine == "AF2-IG":
        setting_parts = [
            f"recycles={get_value('af2_num_recycles')}",
            f"multimer={bool(get_value('af2_multimer', default=True))}",
            f"complex_ig={bool(get_value('af2_use_initial_guess', default=False))}",
            f"binder_template={bool(get_value('af2_use_binder_template', default=False))}",
            f"interface_template={bool(get_value('af2_use_interface_template', default=False))}",
        ]
    elif engine == "Boltz-2":
        setting_parts = [
            f"target_template={bool(get_value('boltz2_use_target_template', default=True))}",
            f"target_msa={bool(get_value('boltz2_use_target_msa', default=True))}",
            f"recycles={get_value('boltz2_recycling_steps')}",
            f"sampling={get_value('boltz2_sampling_steps')}",
            f"samples={get_value('boltz2_diffusion_samples')}",
        ]
    elif engine == "BoltzGen Fold":
        setting_parts = [
            f"recycles={get_value('boltzgen_recycling_steps')}",
            f"sampling={get_value('boltzgen_sampling_steps')}",
            f"samples={get_value('boltzgen_diffusion_samples')}",
        ]
    elif engine == "ColabFold":
        setting_parts = [
            f"msa={get_value('colabfold_msa_source')}",
            f"recycles={get_value('colabfold_num_recycles')}",
            f"models={get_value('colabfold_num_models')}",
            f"templates={bool(get_value('colabfold_use_target_templates', default=False))}",
        ]
    elif engine == "AF3":
        setting_parts = [
            f"templates={bool(get_value('alphafast_use_target_templates', default=True))}",
            f"target_msa={not bool(get_value('alphafast_query_only_msa', default=False))}",
            f"recycles={get_value('alphafast_num_recycles')}",
            f"batch={get_value('alphafast_batch_size')}",
        ]
    elif engine == "RF3":
        checkpoint = Path(str(get_value("rf3_checkpoint_path"))).name
        setting_parts = [
            f"templates={bool(get_value('rf3_use_target_template', default=True))}",
            f"target_msa={bool(get_value('rf3_use_target_msa', default=True))}",
            f"recycles={get_value('rf3_recycles')}",
            f"steps={get_value('rf3_num_steps')}",
            f"batch={get_value('rf3_diffusion_batch_size')}",
            f"checkpoint={checkpoint}",
        ]
    elif engine == "OpenFold-3":
        checkpoint = Path(str(get_value("openfold3_checkpoint_path"))).name
        setting_parts = [
            f"target_msa={bool(get_value('openfold3_use_target_msa', default=True))}",
            f"samples={get_value('openfold3_num_diffusion_samples')}",
            f"seeds={get_value('openfold3_num_model_seeds')}",
            f"recycles={get_value('openfold3_num_recycles')}",
            f"msa_server={bool(get_value('openfold3_use_msa_server', default=False))}",
            f"checkpoint={checkpoint}",
        ]
    elif engine == "Protenix v0.5":
        setting_parts = [
            f"msa={bool(get_value('protenix_use_msa', default=True))}",
            f"cycles={get_value('protenix_cycle')}",
            f"steps={get_value('protenix_diffusion_steps')}",
            f"samples={get_value('protenix_samples')}",
        ]
    elif engine == "Protenix v1":
        setting_parts = [
            f"model={get_value('protenix_v1_model_name', default=refolding_workflow.PROTENIX_V1_MODEL)}",
            f"msa={bool(get_value('protenix_v1_use_msa', default=True))}",
            f"template={bool(get_value('protenix_v1_use_template', default=False))}",
            f"cycles={get_value('protenix_v1_cycle')}",
            f"steps={get_value('protenix_v1_diffusion_steps')}",
            f"samples={get_value('protenix_v1_samples')}",
        ]
    elif engine == "Protenix v2":
        setting_parts = [
            f"model={get_value('protenix_v2_model_name', default=refolding_workflow.PROTENIX_V2_MODEL)}",
            f"msa={bool(get_value('protenix_v2_use_msa', default=True))}",
            f"template={bool(get_value('protenix_v2_use_template', default=False))}",
            f"cycles={get_value('protenix_v2_cycle')}",
            f"steps={get_value('protenix_v2_diffusion_steps')}",
            f"samples={get_value('protenix_v2_samples')}",
        ]
    token = _collection_variant_token(";".join(str(part) for part in setting_parts if str(part)))
    return token or "settings"


def _rename_collection_variant_columns(
    columns: list[str],
    *,
    engine: str,
    variant_token: str,
) -> dict[str, str]:
    engine = _canonical_benchmark_engine_label(engine)
    prefixes = BENCHMARK_ENGINE_PREFIXES.get(engine, ())
    renamed: dict[str, str] = {}
    for column in columns:
        for prefix in prefixes:
            if column.startswith(prefix):
                renamed[column] = f"{prefix}{variant_token}_{column[len(prefix):]}"
                break
    return renamed


def benchmark_engines_in_metrics(metrics_csv: Path) -> list[str]:
    if not metrics_csv.exists():
        return []
    try:
        columns = pd.read_csv(metrics_csv, nrows=0).columns
    except Exception:
        return []
    engines = {
        engine
        for column in columns
        if (engine := _benchmark_engine_for_column(str(column))) is not None
    }
    return [engine for engine in BENCHMARK_ENGINE_PREFIXES if engine in engines]


def _benchmark_run_metrics_path(run_id: str) -> Path | None:
    path = runs_root() / BENCHMARK_GROUP / str(run_id) / "artifacts" / "benchmark" / "merged_benchmark_metrics.csv"
    return path if path.exists() else None


def _benchmark_sidecar_metric_paths(benchmark_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for path in sorted(benchmark_dir.glob("*metrics.csv")):
        if path.name == "merged_benchmark_metrics.csv":
            continue
        if path.name.endswith("_feature_ranking.csv"):
            continue
        paths.append(path)
    return paths


def _merge_metrics_sidecar(base: pd.DataFrame, sidecar: pd.DataFrame) -> pd.DataFrame:
    if base.empty or sidecar.empty or "binder_id" not in base.columns or "binder_id" not in sidecar.columns:
        return base
    keys = ["binder_id"]
    if "target_id" in base.columns and "target_id" in sidecar.columns:
        keys = ["target_id", "binder_id"]
    base = base.copy()
    sidecar = sidecar.copy()
    for key in keys:
        base[key] = base[key].astype(str)
        sidecar[key] = sidecar[key].astype(str)
    sidecar = sidecar[
        sidecar["binder_id"].notna()
        & ~sidecar["binder_id"].isin({"", "nan", "None"})
    ].copy()
    if sidecar.empty:
        return base
    sidecar = sidecar.drop_duplicates(keys).copy()
    value_columns = [col for col in sidecar.columns if col not in keys]
    if not value_columns:
        return base
    duplicate_columns = [col for col in value_columns if col in base.columns]
    if duplicate_columns:
        sidecar = sidecar.rename(columns={col: f"{col}__sidecar" for col in duplicate_columns})
    merged = base.merge(sidecar, on=keys, how="left")
    def first_series(frame: pd.DataFrame, column: str) -> pd.Series:
        values = frame[column]
        return values.iloc[:, 0] if isinstance(values, pd.DataFrame) else values

    for column in duplicate_columns:
        sidecar_column = f"{column}__sidecar"
        if sidecar_column not in merged.columns:
            continue
        merged[column] = first_series(merged, column).combine_first(first_series(merged, sidecar_column))
        merged = merged.drop(columns=[sidecar_column])
    return merged


def _read_benchmark_metrics_with_sidecars(metrics_path: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_path)
    benchmark_dir = metrics_path.parent
    for sidecar_path in _benchmark_sidecar_metric_paths(benchmark_dir):
        try:
            sidecar = pd.read_csv(sidecar_path)
        except Exception:
            continue
        for _, filename, prefix in BENCHMARK_ENGINE_ARTIFACTS.values():
            if sidecar_path.name == filename:
                sidecar = _prefix_engine_metric_columns(
                    sidecar,
                    prefix=prefix,
                    passthrough={"binder_id", "target_id", "label"},
                )
                break
        df = _merge_metrics_sidecar(df, sidecar)
    return df


def _reuse_cached_input_rosetta_metrics(
    *,
    run_csv: Path,
    out_csv: Path,
    current_run_dir: Path,
) -> dict[str, Any]:
    if out_csv.exists() or not run_csv.exists():
        return {}
    try:
        run_df = pd.read_csv(run_csv, usecols=["binder_id"])
    except Exception:
        return {}
    if run_df.empty or "binder_id" not in run_df.columns:
        return {}
    wanted = [str(value) for value in run_df["binder_id"].tolist()]
    wanted_set = set(wanted)
    if not wanted_set:
        return {}

    current_run_dir = current_run_dir.resolve()
    cached_rows: dict[str, pd.Series] = {}
    source_rows: dict[str, int] = {}
    source_files: list[str] = []
    for metrics_path in sorted(
        runs_root().glob(
            f"{BENCHMARK_GROUP}/*/artifacts/raw/de_novo_binder_scoring/output/input_rosetta_metrics.csv"
        )
    ):
        try:
            if current_run_dir in metrics_path.resolve().parents:
                continue
        except Exception:
            pass
        try:
            metrics_df = pd.read_csv(metrics_path)
        except Exception:
            continue
        if metrics_df.empty or "binder_id" not in metrics_df.columns:
            continue
        metrics_df["binder_id"] = metrics_df["binder_id"].astype(str)
        subset = metrics_df[metrics_df["binder_id"].isin(wanted_set - set(cached_rows))]
        if subset.empty:
            continue
        added = 0
        for _, row in subset.drop_duplicates("binder_id", keep="first").iterrows():
            binder_id = str(row["binder_id"])
            if binder_id in cached_rows:
                continue
            cached_rows[binder_id] = row
            added += 1
        if added:
            try:
                source_files.append(str(metrics_path.relative_to(runs_root())))
            except ValueError:
                source_files.append(str(metrics_path))
            source_rows[str(metrics_path)] = added
        if wanted_set.issubset(cached_rows):
            break

    if not wanted_set.issubset(cached_rows):
        return {
            "input_rosetta_cache_available_rows": int(len(cached_rows)),
            "input_rosetta_cache_missing_rows": int(len(wanted_set - set(cached_rows))),
        }

    merged_df = pd.DataFrame([cached_rows[binder_id] for binder_id in wanted])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(out_csv, index=False)
    return {
        "input_rosetta_reused_from_benchmark_cache": True,
        "input_rosetta_cache_row_count": int(len(merged_df)),
        "input_rosetta_cache_source_count": int(len(source_files)),
        "input_rosetta_cache_sources": source_files[:50],
        "input_rosetta_cache_rows_by_source": source_rows,
    }


def _engine_metric_prefix(engine: str) -> str | None:
    artifact_info = BENCHMARK_ENGINE_ARTIFACTS.get(str(engine))
    return artifact_info[2] if artifact_info else None


def benchmark_missing_pyrosetta_rows(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Report selected benchmark matrix cells missing predicted PyRosetta metrics."""

    grouped: dict[str, dict[str, set[str]]] = {}
    for cell in _json_clean(cells):
        run_id = str(cell.get("run_id") or "").strip()
        engine = str(cell.get("engine") or "").strip()
        target = str(cell.get("target_id") or "").strip()
        if not run_id or not engine:
            continue
        grouped.setdefault(run_id, {}).setdefault(engine, set())
        if target:
            grouped[run_id][engine].add(target)

    rows: list[dict[str, Any]] = []
    for run_id, engine_targets in sorted(grouped.items()):
        metrics_path = _benchmark_run_metrics_path(run_id)
        if metrics_path is None:
            rows.append(
                {
                    "run_id": run_id,
                    "engine": "",
                    "status": "missing_source_metrics",
                    "records": 0,
                    "existing_rosetta_records": 0,
                    "missing_rosetta_records": 0,
                    "targets_missing": "",
                }
            )
            continue
        source_run_dir = metrics_path.parents[2]
        metadata = read_json(source_run_dir / "metadata.json")
        try:
            df = _read_benchmark_metrics_with_sidecars(metrics_path)
        except Exception as exc:
            rows.append(
                {
                    "run_id": run_id,
                    "job_code": str(metadata.get("job_code") or ""),
                    "engine": "",
                    "status": f"read_error:{exc}",
                    "records": 0,
                    "existing_rosetta_records": 0,
                    "missing_rosetta_records": 0,
                    "targets_missing": "",
                }
            )
            continue
        for engine, targets in sorted(engine_targets.items()):
            prefix = _engine_metric_prefix(engine)
            if not prefix:
                continue
            subset = df
            if targets and "target_id" in df.columns:
                subset = df[df["target_id"].astype(str).isin(targets)].copy()
            records = int(len(subset))
            rosetta_col = f"{prefix}_rosetta_interface_dG"
            existing = int(subset[rosetta_col].notna().sum()) if rosetta_col in subset.columns else 0
            missing = max(0, records - existing)
            target_missing = ""
            if missing and "target_id" in subset.columns:
                missing_df = subset
                if rosetta_col in missing_df.columns:
                    missing_df = missing_df[missing_df[rosetta_col].isna()]
                target_missing = ", ".join(sorted(missing_df["target_id"].dropna().astype(str).unique().tolist()))
            rows.append(
                {
                    "run_id": run_id,
                    "job_code": str(metadata.get("job_code") or ""),
                    "engine": engine,
                    "metric_column": rosetta_col,
                    "targets": ", ".join(sorted(targets)),
                    "records": records,
                    "existing_rosetta_records": existing,
                    "missing_rosetta_records": missing,
                    "targets_missing": target_missing,
                    "status": "missing" if missing else "complete",
                }
            )
    return rows


def _combine_metric_tables_by_binder(existing_path: Path, update_path: Path) -> int:
    if not update_path.exists():
        return 0
    update_df = pd.read_csv(update_path)
    if update_df.empty or "binder_id" not in update_df.columns:
        return 0
    keys = ["binder_id"]
    if existing_path.exists():
        existing_df = pd.read_csv(existing_path)
        if "binder_id" not in existing_df.columns:
            existing_df = pd.DataFrame(columns=keys)
        merged = existing_df.copy()
    else:
        merged = pd.DataFrame(columns=keys)
    for key in keys:
        if key not in merged.columns:
            merged[key] = pd.Series(dtype=object)
    merged = merged.drop_duplicates(keys).set_index(keys, drop=False)
    update_df = update_df.drop_duplicates(keys).set_index(keys, drop=False)
    for column in update_df.columns:
        if column in keys:
            continue
        if column not in merged.columns:
            merged[column] = np.nan
        merged[column] = update_df[column].combine_first(merged[column])
    missing_keys = update_df.index.difference(merged.index)
    if len(missing_keys):
        merged = pd.concat([merged, update_df.loc[missing_keys]], axis=0, sort=False)
    out_df = merged.reset_index(drop=True)
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(existing_path, index=False)
    return int(len(update_df))


def _stage_predicted_metric_pdbs_for_engines(
    *,
    source_run_dir: Path,
    engines: list[str],
) -> tuple[list[tuple[str, Path]], dict[str, int]]:
    run_csv = source_run_dir / "artifacts" / "raw" / "de_novo_binder_scoring" / "output" / "run.csv"
    output_dir = source_run_dir / "artifacts" / "raw" / "de_novo_binder_scoring" / "output"
    benchmark_dir = source_run_dir / "artifacts" / "benchmark"
    group_roles = _benchmark_group_roles(run_csv)
    predicted_pdb_dirs: list[tuple[str, Path]] = []
    counts: dict[str, int] = {}
    engine_set = {str(engine) for engine in engines}

    def existing_metric_dir(metric_prefix: str) -> tuple[Path | None, int]:
        folder = benchmark_dir / "predicted_metric_pdbs" / metric_prefix
        if not folder.exists():
            return None, 0
        count = len(list(folder.glob("*.pdb")))
        return (folder, count) if count else (None, 0)

    if "AF3" in engine_set:
        pdb_dir, count = _stage_alphafast_pdbs(
            output_dir / "AF3" / "alphafast_output",
            benchmark_dir / "predicted_metric_pdbs" / "af3",
            group_roles,
        )
        if pdb_dir is None:
            pdb_dir, count = existing_metric_dir("af3")
        if pdb_dir is not None:
            predicted_pdb_dirs.append(("af3", pdb_dir))
            counts["AF3"] = count
    if "ColabFold" in engine_set:
        pdb_dir, count = _stage_colabfold_pdbs(
            output_dir,
            benchmark_dir / "predicted_metric_pdbs" / "colab",
            group_roles,
        )
        if pdb_dir is None:
            pdb_dir, count = existing_metric_dir("colab")
        if pdb_dir is not None:
            predicted_pdb_dirs.append(("colab", pdb_dir))
            counts["ColabFold"] = count

    child_engine_map = {
        "AF2-IG": ("af2_initial_guess", "af2"),
        "Boltz-2": ("boltz2", "boltz2"),
        "BoltzGen Fold": ("boltzgen_fold", "boltzgen_fold"),
        "ESMFold2": ("esmfold2", "esmfold2"),
        "OpenFold-3": ("openfold3", "openfold3"),
        "Protenix v0.5": ("protenix", "protenix"),
        "RF3": ("rf3", "rf3"),
    }
    for engine, (child_key, metric_prefix) in child_engine_map.items():
        if engine not in engine_set:
            continue
        child_runs = _completed_benchmark_child_runs(source_run_dir, child_key)
        pdb_dir, count = _stage_child_candidate_pdbs(
            child_runs,
            benchmark_dir / "predicted_metric_pdbs" / metric_prefix,
            source_label=metric_prefix,
        )
        if pdb_dir is None:
            engine_dir = source_run_dir / "artifacts" / "engines" / child_key
            pdb_dir, count = _stage_engine_artifact_candidate_pdbs(
                source_run_dir,
                engine_dir,
                benchmark_dir / "predicted_metric_pdbs" / metric_prefix,
                source_label=metric_prefix,
            )
        if pdb_dir is None:
            pdb_dir, count = existing_metric_dir(metric_prefix)
        if pdb_dir is not None:
            predicted_pdb_dirs.append((metric_prefix, pdb_dir))
            counts[engine] = count
    return predicted_pdb_dirs, counts


def _refresh_merged_benchmark_metrics(source_run_dir: Path) -> tuple[Path | None, Path | None]:
    run_csv = source_run_dir / "artifacts" / "raw" / "de_novo_binder_scoring" / "output" / "run.csv"
    benchmark_dir = source_run_dir / "artifacts" / "benchmark"
    merged_path = benchmark_dir / "merged_benchmark_metrics.csv"
    if not run_csv.exists() or not merged_path.exists():
        return None, None
    df = _read_benchmark_metrics_with_sidecars(merged_path)
    df = _hydrate_legacy_af2_initial_guess_columns(df)
    df.to_csv(merged_path, index=False)
    label_col = "binder" if "binder" in df.columns else "label" if "label" in df.columns else ""
    ranking_path: Path | None = None
    if label_col:
        feature_rows, summary = _metric_column_summary(df, label_col, max_columns=2000)
        summary_path = benchmark_dir / "merged_benchmark_feature_summary.json"
        ranking_path = benchmark_dir / "merged_benchmark_feature_ranking.csv"
        write_json(summary_path, summary)
        _write_feature_ranking(ranking_path, feature_rows)
    return merged_path, ranking_path


def enqueue_missing_pyrosetta_benchmark_metrics(
    *,
    missing_rows: list[dict[str, Any]],
    pyrosetta_nprocs: int = 32,
) -> Path:
    clean_rows = [
        row
        for row in _json_clean(missing_rows)
        if int(row.get("missing_rosetta_records") or 0) > 0 and str(row.get("run_id") or "").strip()
    ]
    if not clean_rows:
        raise ValueError("No selected benchmark cells are missing PyRosetta metrics.")
    engines_by_run: dict[str, set[str]] = {}
    for row in clean_rows:
        engines_by_run.setdefault(str(row["run_id"]), set()).add(str(row.get("engine") or ""))
    grouped = {
        run_id: sorted(engine for engine in engines if engine)
        for run_id, engines in sorted(engines_by_run.items())
    }
    job = create_job(
        BENCHMARK_GROUP,
        "benchmark_pyrosetta_backfill",
        "benchmark_pyrosetta_backfill",
        {
            "source_runs": sorted(grouped),
        },
        {
            "source_runs": sorted(grouped),
            "engines_by_run": grouped,
            "pyrosetta_nprocs": max(1, int(pyrosetta_nprocs)),
            "missing_cells": clean_rows,
        },
    )
    write_json(
        job.run_dir / "worker_request.json",
        {
            "kind": "benchmark_pyrosetta_backfill",
            "kwargs": {
                "engines_by_run": grouped,
                "pyrosetta_nprocs": max(1, int(pyrosetta_nprocs)),
            },
            "path_kwargs": [],
        },
    )
    write_json(
        job.run_dir / "command.json",
        {
            "mode": "local_worker",
            "command": ["python", "-m", "mn_protein_design.core.local_worker", "--run-dir", str(job.run_dir)],
        },
    )
    return job.run_dir


def run_missing_pyrosetta_benchmark_metrics(
    run_dir: Path,
    *,
    engines_by_run: dict[str, list[str]],
    pyrosetta_nprocs: int = 32,
) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    update_status(run_dir, "running", current_phase="Recalculating missing PyRosetta metrics")
    commands: list[list[str]] = []
    source_summaries: list[dict[str, Any]] = []
    total_scored = 0
    job_artifacts = run_dir / "artifacts" / "benchmark"
    job_artifacts.mkdir(parents=True, exist_ok=True)

    for run_id, engines in sorted((engines_by_run or {}).items()):
        source_run_dir = runs_root() / BENCHMARK_GROUP / str(run_id)
        run_csv = source_run_dir / "artifacts" / "raw" / "de_novo_binder_scoring" / "output" / "run.csv"
        benchmark_dir = source_run_dir / "artifacts" / "benchmark"
        if not run_csv.exists():
            source_summaries.append({"run_id": run_id, "status": "missing_run_csv", "engines": ", ".join(engines or [])})
            continue
        predicted_pdb_dirs, staged_counts = _stage_predicted_metric_pdbs_for_engines(
            source_run_dir=source_run_dir,
            engines=list(engines or []),
        )
        if not predicted_pdb_dirs:
            source_summaries.append(
                {
                    "run_id": run_id,
                    "status": "no_predicted_pdbs",
                    "engines": ", ".join(engines or []),
                    "staged_counts": staged_counts,
                }
            )
            continue
        update_csv = benchmark_dir / f"predicted_rosetta_metrics_backfill_{run_dir.name}.csv"
        rosetta_cmd = [
            "docker",
            "run",
            "--rm",
            *_repo_and_runs_mounts(),
            "-w",
            str(DE_NOVO_BINDER_SCORING_DIR),
            PYROSETTA_METRICS_IMAGE,
            "python",
            "./scripts/compute_rosetta_metrics.py",
            "--run-csv",
            str(run_csv),
            "--out-csv",
            str(update_csv),
            "--nprocs",
            str(max(1, int(pyrosetta_nprocs))),
            "--dalphaball-path",
            "./functions/DAlphaBall.gcc",
        ]
        for prefix, folder in predicted_pdb_dirs:
            rosetta_cmd.extend(["--folder", f"{prefix}:{folder}"])
        commands.append(rosetta_cmd)
        update_status(
            run_dir,
            "running",
            current_phase="Recalculating missing PyRosetta metrics",
            current_engine=", ".join(engines or []),
        )
        rc = _run_docker_command(run_dir, rosetta_cmd)
        canonical_rosetta = benchmark_dir / "predicted_rosetta_metrics.csv"
        rows_scored = 0
        if rc == 0 and update_csv.exists():
            rows_scored = _combine_metric_tables_by_binder(canonical_rosetta, update_csv)
            total_scored += rows_scored
            merged_path, ranking_path = _refresh_merged_benchmark_metrics(source_run_dir)
        else:
            merged_path, ranking_path = None, None
        source_summaries.append(
            {
                "run_id": run_id,
                "engines": ", ".join(engines or []),
                "return_code": rc,
                "staged_counts": staged_counts,
                "rows_scored": rows_scored,
                "rosetta_metrics": str(canonical_rosetta) if canonical_rosetta.exists() else "",
                "merged_metrics": str(merged_path) if merged_path else "",
                "ranking": str(ranking_path) if ranking_path else "",
                "status": "completed" if rc == 0 and rows_scored else "failed",
            }
        )

    source_summary_path = job_artifacts / "pyrosetta_backfill_sources.csv"
    pd.DataFrame(source_summaries).to_csv(source_summary_path, index=False)
    write_json(run_dir / "command.json", {"mode": "docker", "commands": commands})
    success = bool(source_summaries) and all(str(row.get("status")) == "completed" for row in source_summaries)
    finish_job(
        run_dir,
        success,
        {
            "outputs": {
                "pyrosetta_backfill_sources": str(source_summary_path.relative_to(run_dir)),
            },
            "metrics": {
                "source_run_count": len(source_summaries),
                "rows_scored": total_scored,
                "pyrosetta_nprocs": max(1, int(pyrosetta_nprocs)),
            },
        },
    )
    if not success:
        failed = [row for row in source_summaries if str(row.get("status")) != "completed"]
        raise RuntimeError(f"PyRosetta backfill failed for {len(failed)} source run(s).")
    return run_dir


def _replace_path_with_link_or_copy(source_path: Path, target_path: Path) -> None:
    if target_path.exists() or target_path.is_symlink():
        if target_path.is_dir() and not target_path.is_symlink():
            shutil.rmtree(target_path)
        else:
            target_path.unlink()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        target_path.symlink_to(source_path, target_is_directory=source_path.is_dir())
    except OSError:
        if source_path.is_dir():
            shutil.copytree(source_path, target_path)
        else:
            shutil.copy2(source_path, target_path)


def _merge_or_link_collection_csv(source_path: Path, target_path: Path) -> None:
    if not target_path.exists() and not target_path.is_symlink():
        _replace_path_with_link_or_copy(source_path, target_path)
        return
    try:
        existing = pd.read_csv(target_path)
        incoming = pd.read_csv(source_path)
    except Exception:
        _replace_path_with_link_or_copy(source_path, target_path)
        return
    if "binder_id" not in existing.columns or "binder_id" not in incoming.columns:
        _replace_path_with_link_or_copy(source_path, target_path)
        return
    existing["binder_id"] = existing["binder_id"].astype(str)
    incoming["binder_id"] = incoming["binder_id"].astype(str)
    merged = pd.concat([existing, incoming], ignore_index=True, sort=False).drop_duplicates("binder_id", keep="last")
    if target_path.is_symlink() or target_path.is_file():
        target_path.unlink()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(target_path, index=False)


def _merge_or_link_collection_tree(source_path: Path, target_path: Path) -> None:
    if not target_path.exists() and not target_path.is_symlink():
        _replace_path_with_link_or_copy(source_path, target_path)
        return
    if target_path.is_symlink() or target_path.is_file():
        previous = target_path.resolve()
        target_path.unlink()
        target_path.mkdir(parents=True, exist_ok=True)
        if previous.exists() and previous.is_dir():
            _merge_or_link_collection_tree(previous, target_path)
    else:
        target_path.mkdir(parents=True, exist_ok=True)
    for source_file in source_path.rglob("*"):
        if source_file.is_dir():
            continue
        target_file = target_path / source_file.relative_to(source_path)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        if target_file.exists() or target_file.is_symlink():
            target_file.unlink()
        try:
            target_file.symlink_to(source_file)
        except OSError:
            shutil.copy2(source_file, target_file)


def _stage_collection_engine_artifacts(
    *,
    source_run_dir: Path,
    collection_run_dir: Path,
    benchmark_dir: Path,
    engines: list[str],
) -> None:
    source_benchmark = source_run_dir / "artifacts" / "benchmark"
    for engine in engines:
        engine = _canonical_benchmark_engine_label(engine)
        artifact_info = BENCHMARK_ENGINE_ARTIFACTS.get(engine)
        if not artifact_info:
            continue
        engine_key, metrics_csv, metric_pdb_key = artifact_info
        source_metrics = source_benchmark / metrics_csv
        if source_metrics.exists():
            _merge_or_link_collection_csv(source_metrics, benchmark_dir / metrics_csv)
        source_engine_dir = source_run_dir / "artifacts" / "engines" / engine_key
        if source_engine_dir.exists():
            _merge_or_link_collection_tree(
                source_engine_dir,
                collection_run_dir / "artifacts" / "engines" / engine_key,
            )
        source_metric_pdbs = source_benchmark / "predicted_metric_pdbs" / metric_pdb_key
        if source_metric_pdbs.exists():
            _merge_or_link_collection_tree(
                source_metric_pdbs,
                benchmark_dir / "predicted_metric_pdbs" / metric_pdb_key,
            )
        source_viewer_structures = source_benchmark / "predicted_viewer_structures" / metric_pdb_key
        if source_viewer_structures.exists():
            _merge_or_link_collection_tree(
                source_viewer_structures,
                benchmark_dir / "predicted_viewer_structures" / metric_pdb_key,
            )


def _stage_collection_reference_artifacts(source_run_dir: Path, collection_run_dir: Path) -> None:
    for relative in [
        Path("artifacts") / "raw" / "de_novo_binder_scoring" / "output" / "input_pdbs",
        Path("artifacts") / "benchmark" / "input_pdbs",
    ]:
        source_path = source_run_dir / relative
        if source_path.exists():
            _replace_path_with_link_or_copy(source_path, collection_run_dir / relative)


def create_benchmark_collection(
    *,
    name: str,
    selections: list[dict[str, Any]],
    target_ids: list[str] | None = None,
    include_input_columns: bool = True,
) -> Path:
    clean_name = str(name or "Benchmark collection").strip() or "Benchmark collection"
    selections = _json_clean(selections)
    include_input_columns = bool(_json_clean(include_input_columns))
    selected_targets = {str(target) for target in (target_ids or []) if str(target).strip()}
    job = create_job(
        BENCHMARK_GROUP,
        "benchmark_collection",
        "benchmark_collection_merge",
        {
            "source_runs": [str(item.get("run_id") or "") for item in selections],
        },
        {
            "collection_name": clean_name,
            "target_ids": sorted(selected_targets),
            "include_input_columns": include_input_columns,
            "selections": selections,
        },
    )
    update_status(job.run_dir, "running")
    benchmark_dir = job.run_dir / "artifacts" / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    base: pd.DataFrame | None = None
    keys: list[str] = ["binder_id"]
    provenance_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    replacement_rows: list[dict[str, Any]] = []
    loaded_sources: list[dict[str, Any]] = []

    for order, selection in enumerate(selections, start=1):
        run_id = str(selection.get("run_id") or "").strip()
        engines = [_canonical_benchmark_engine_label(engine) for engine in (selection.get("engines") or []) if str(engine).strip()]
        raw_engine_targets = selection.get("engine_targets") if isinstance(selection.get("engine_targets"), dict) else {}
        engine_targets = {
            _canonical_benchmark_engine_label(engine): {str(target) for target in (targets or []) if str(target).strip()}
            for engine, targets in raw_engine_targets.items()
        }
        metrics_path = _benchmark_run_metrics_path(run_id)
        if metrics_path is None:
            source_rows.append({"order": order, "run_id": run_id, "status": "missing_metrics", "engines": ",".join(engines)})
            continue
        source_run_dir = metrics_path.parents[2]
        metadata = read_json(source_run_dir / "metadata.json")
        result = read_json(source_run_dir / "result.json")
        try:
            raw_df = pd.read_csv(metrics_path)
            df = _read_benchmark_metrics_with_sidecars(metrics_path)
            df = _hydrate_legacy_af2_initial_guess_columns(df)
        except Exception as exc:
            source_rows.append({"order": order, "run_id": run_id, "status": f"read_error:{exc}", "engines": ",".join(engines)})
            continue
        hydrated_column_count = max(0, len(df.columns) - len(raw_df.columns))
        source_targets = set(selected_targets)
        for targets in engine_targets.values():
            source_targets.update(targets)
        if source_targets and "target_id" in df.columns:
            df = df[df["target_id"].astype(str).isin(source_targets)].copy()
        if df.empty:
            source_rows.append({"order": order, "run_id": run_id, "status": "empty_after_target_filter", "engines": ",".join(engines)})
            continue
        loaded_sources.append(
            {
                "order": order,
                "run_id": run_id,
                "engines": engines,
                "metrics_path": metrics_path,
                "source_run_dir": source_run_dir,
                "metadata": metadata,
                "result": result,
                "df": df,
                "raw_column_count": int(len(raw_df.columns)),
                "hydrated_column_count": int(hydrated_column_count),
                "keys": _collection_merge_keys(df),
                "engine_targets": engine_targets,
            }
        )

    if loaded_sources:
        base_source = max(loaded_sources, key=lambda item: int(len(item["df"])))
        keys = list(base_source["keys"])
        metadata_cols = _collection_metadata_columns(base_source["df"])
        for source in loaded_sources:
            if list(source["keys"]) != keys:
                continue
            for col in _collection_metadata_columns(source["df"]):
                if col not in metadata_cols:
                    metadata_cols.append(col)
        if not set(keys).issubset(metadata_cols):
            metadata_cols = [col for col in keys if col in base_source["df"].columns] + metadata_cols
            metadata_cols = list(dict.fromkeys(metadata_cols))
        base_frames: list[pd.DataFrame] = []
        for source in loaded_sources:
            if list(source["keys"]) != keys:
                continue
            source_df: pd.DataFrame = source["df"]
            source_cols = [col for col in metadata_cols if col in source_df.columns]
            if source_cols:
                base_frames.append(source_df[source_cols].copy())
        if base_frames:
            base = pd.concat(base_frames, ignore_index=True, sort=False).drop_duplicates(keys).copy()
        else:
            base = base_source["df"][metadata_cols].drop_duplicates(keys).copy()
        _stage_collection_reference_artifacts(base_source["source_run_dir"], job.run_dir)

    engine_counts: dict[str, int] = {}
    engine_setting_tokens: dict[str, set[str]] = {}
    for source in loaded_sources:
        for engine in source["engines"]:
            if engine == "Input":
                continue
            engine_counts[engine] = engine_counts.get(engine, 0) + 1
            engine_setting_tokens.setdefault(engine, set()).add(_source_engine_variant_token(source, engine))
    for source in loaded_sources:
        engine_variants: dict[str, str] = {}
        for engine in source["engines"]:
            if engine == "Input" or engine_counts.get(engine, 0) <= 1:
                continue
            if engine == "ESMFold2":
                continue
            token = _source_engine_variant_token(source, engine)
            if len(engine_setting_tokens.get(engine, set())) <= 1:
                continue
            engine_variants[engine] = token
        source["engine_variants"] = engine_variants

    for source in loaded_sources:
        order = int(source["order"])
        run_id = str(source["run_id"])
        engines = list(source["engines"])
        df: pd.DataFrame = source["df"]
        metadata: dict[str, Any] = source["metadata"]
        result: dict[str, Any] = source["result"]
        current_keys = list(source["keys"])
        engine_targets = source.get("engine_targets") if isinstance(source.get("engine_targets"), dict) else {}
        if current_keys != keys:
            source_rows.append({"order": order, "run_id": run_id, "status": "incompatible_merge_keys", "engines": ",".join(engines)})
            continue

        selected_columns: list[str] = []
        if include_input_columns:
            selected_columns.extend(
                col
                for col in df.columns
                if _benchmark_engine_for_column(str(col)) == "Input"
            )
        for engine in engines:
            if engine == "Input":
                continue
            selected_columns.extend(
                col
                for col in df.columns
                if _benchmark_engine_for_column(str(col)) == engine
            )
        selected_columns = [col for col in dict.fromkeys(selected_columns) if col not in keys]
        if not selected_columns:
            source_rows.append({"order": order, "run_id": run_id, "status": "no_selected_engine_columns", "engines": ",".join(engines)})
            continue
        assert base is not None
        source_columns = list(selected_columns)
        column_renames: dict[str, str] = {}
        engine_variants = source.get("engine_variants") if isinstance(source.get("engine_variants"), dict) else {}
        for engine, variant_token in engine_variants.items():
            engine_columns = [col for col in source_columns if _benchmark_engine_for_column(str(col)) == engine]
            column_renames.update(
                _rename_collection_variant_columns(
                    engine_columns,
                    engine=engine,
                    variant_token=str(variant_token),
                )
            )
        selected_columns = [column_renames.get(col, col) for col in source_columns]
        incoming = df[keys + source_columns].drop_duplicates(keys).rename(columns=column_renames).copy()
        if engine_targets and "target_id" in incoming.columns:
            for engine, targets in engine_targets.items():
                if not targets:
                    continue
                engine_columns = [
                    column_renames.get(col, col)
                    for col in source_columns
                    if _benchmark_engine_for_column(str(col)) == engine
                ]
                engine_columns = [col for col in engine_columns if col in incoming.columns]
                if engine_columns:
                    incoming[engine_columns] = incoming[engine_columns].astype("object")
                    outside_targets = ~incoming["target_id"].astype(str).isin(set(targets))
                    incoming.loc[outside_targets, engine_columns] = np.nan
        source_key_index = set(tuple(row) for row in incoming[keys].itertuples(index=False, name=None))
        base_key_index = set(tuple(row) for row in base[keys].itertuples(index=False, name=None))
        records_not_in_base = len(source_key_index - base_key_index)
        matched_records = len(source_key_index & base_key_index)
        coverage_records = int(incoming[selected_columns].notna().any(axis=1).sum())
        replaced = [col for col in selected_columns if col in base.columns]
        if replaced:
            incoming = incoming.rename(columns={column: f"{column}__incoming" for column in replaced})
        base = base.merge(incoming, on=keys, how="left")
        for column in replaced:
            incoming_column = f"{column}__incoming"
            if incoming_column not in base.columns:
                continue
            existing = base[column] if column in base.columns else pd.Series([np.nan] * len(base), index=base.index)
            base[column] = existing.combine_first(base[incoming_column])
            filled_count = int(base[incoming_column].notna().sum())
            replacement_rows.append({"column": column, "merged_from_run_id": run_id, "order": order, "filled_values": filled_count})
            base = base.drop(columns=[incoming_column])
        _stage_collection_engine_artifacts(
            source_run_dir=source["source_run_dir"],
            collection_run_dir=job.run_dir,
            benchmark_dir=benchmark_dir,
            engines=engines,
        )
        for column in selected_columns:
            provenance_rows.append(
                {
                    "feature": column,
                    "engine": _benchmark_engine_for_column(column) or "Other",
                    "source_run_id": run_id,
                    "source_job_code": metadata.get("job_code"),
                    "source_created_at": metadata.get("created_at"),
                    "source_status": metadata.get("status"),
                    "source_top_feature": (result.get("metrics") or {}).get("merged_benchmark_top_feature"),
                    "selection_order": order,
                    "source_engine_variant": engine_variants.get(_benchmark_engine_for_column(column) or ""),
                }
            )
        source_rows.append(
            {
                "order": order,
                "run_id": run_id,
                "status": "included",
                "engines": ",".join(engines),
                "selected_feature_count": len(selected_columns),
                "raw_column_count": int(source.get("raw_column_count") or 0),
                "hydrated_column_count": int(source.get("hydrated_column_count") or 0),
                "engine_variants": ",".join(f"{engine}:{variant}" for engine, variant in sorted(engine_variants.items())),
                "engine_targets": json.dumps({engine: sorted(targets) for engine, targets in engine_targets.items()}),
                "records": int(len(df)),
                "collection_records": int(len(base)),
                "matched_records": int(matched_records),
                "coverage_records": int(coverage_records),
                "records_not_in_collection": int(records_not_in_base),
                "is_collection_backbone": bool(run_id == base_source["run_id"]) if loaded_sources else False,
            }
        )

    if base is None or base.empty:
        finish_job(
            job.run_dir,
            False,
            {
                "metrics": {"error": "No compatible benchmark metric tables were selected."},
                "outputs": {},
            },
        )
        raise ValueError("No compatible benchmark metric tables were selected.")

    merged_path = benchmark_dir / "merged_benchmark_metrics.csv"
    base.to_csv(merged_path, index=False)
    provenance_path = benchmark_dir / "benchmark_collection_provenance.csv"
    pd.DataFrame(provenance_rows).to_csv(provenance_path, index=False)
    sources_path = benchmark_dir / "benchmark_collection_sources.csv"
    pd.DataFrame(source_rows).to_csv(sources_path, index=False)
    replacements_path = benchmark_dir / "benchmark_collection_replacements.csv"
    pd.DataFrame(replacement_rows).to_csv(replacements_path, index=False)

    label_col = "binder" if "binder" in base.columns else "label" if "label" in base.columns else ""
    metrics: dict[str, Any] = {
        "collection_name": clean_name,
        "record_count": int(len(base)),
        "target_count": int(base["target_id"].nunique()) if "target_id" in base.columns else None,
        "source_run_count": len({row["source_run_id"] for row in provenance_rows}),
        "selected_feature_count": len(provenance_rows),
        "replacement_count": len(replacement_rows),
        "merged_benchmark_metrics": str(merged_path.relative_to(job.run_dir)),
        "benchmark_collection_provenance": str(provenance_path.relative_to(job.run_dir)),
        "benchmark_collection_sources": str(sources_path.relative_to(job.run_dir)),
        "benchmark_collection_replacements": str(replacements_path.relative_to(job.run_dir)),
    }
    if loaded_sources:
        metrics["collection_backbone_run_id"] = str(base_source["run_id"])
        metrics["collection_backbone_records"] = int(len(base_source["df"]))
    if label_col:
        feature_rows, summary = _metric_column_summary(base, label_col, max_columns=2000)
        summary_path = benchmark_dir / "merged_benchmark_feature_summary.json"
        feature_table = benchmark_dir / "merged_benchmark_feature_ranking.csv"
        write_json(summary_path, summary)
        _write_feature_ranking(feature_table, feature_rows)
        metrics.update(
            {
                "label_column": label_col,
                "positive_count": summary.get("positive_count"),
                "negative_count": summary.get("negative_count"),
                "merged_benchmark_feature_summary": str(summary_path.relative_to(job.run_dir)),
                "merged_benchmark_feature_ranking": str(feature_table.relative_to(job.run_dir)),
                "merged_benchmark_top_feature": summary.get("top_feature"),
                "merged_benchmark_top_feature_average_precision": summary.get("top_feature_average_precision"),
                "merged_benchmark_top_feature_auroc": summary.get("top_feature_auroc"),
            }
        )
    write_json(
        benchmark_dir / "collection.json",
        {
            "name": clean_name,
            "target_ids": sorted(selected_targets),
            "include_input_columns": include_input_columns,
            "selections": selections,
            "merge_keys": keys,
            "backbone_run_id": str(base_source["run_id"]) if loaded_sources else None,
        },
    )
    finish_job(
        job.run_dir,
        True,
        {
            "metrics": metrics,
            "outputs": {
                "merged_benchmark_metrics": str(merged_path.relative_to(job.run_dir)),
                "merged_benchmark_feature_ranking": metrics.get("merged_benchmark_feature_ranking"),
                "benchmark_collection_provenance": str(provenance_path.relative_to(job.run_dir)),
                "benchmark_collection_sources": str(sources_path.relative_to(job.run_dir)),
                "benchmark_collection_replacements": str(replacements_path.relative_to(job.run_dir)),
                "collection": "artifacts/benchmark/collection.json",
            },
        },
    )
    return job.run_dir


def create_benchmark_matrix_workspace(
    *,
    name: str,
    selections: list[dict[str, Any]],
) -> Path:
    clean_name = str(name or "Benchmark matrix workspace").strip() or "Benchmark matrix workspace"
    selections = _json_clean(selections)
    source_rows: list[dict[str, Any]] = []
    targets: set[str] = set()
    engines: set[str] = set()
    for order, selection in enumerate(selections, start=1):
        run_id = str(selection.get("run_id") or "").strip()
        selected_engines = [_canonical_benchmark_engine_label(engine) for engine in (selection.get("engines") or []) if str(engine).strip()]
        engine_targets = selection.get("engine_targets") if isinstance(selection.get("engine_targets"), dict) else {}
        for engine in selected_engines:
            raw_targets = engine_targets.get(engine)
            if raw_targets is None and engine == "Protenix v0.5":
                raw_targets = engine_targets.get("Protenix")
            target_values = [str(target) for target in (raw_targets or []) if str(target).strip()]
            if not target_values:
                source_rows.append(
                    {
                        "order": order,
                        "run_id": run_id,
                        "engine": engine,
                        "target_id": "",
                    }
                )
                engines.add(engine)
                continue
            for target_id in target_values:
                source_rows.append(
                    {
                        "order": order,
                        "run_id": run_id,
                        "engine": engine,
                        "target_id": target_id,
                    }
                )
                engines.add(engine)
                targets.add(target_id)

    job = create_job(
        BENCHMARK_GROUP,
        "benchmark_matrix_workspace",
        "benchmark_matrix_workspace",
        {
            "source_runs": sorted({str(row["run_id"]) for row in source_rows if str(row.get("run_id") or "")}),
        },
        {
            "workspace_name": clean_name,
            "selections": selections,
        },
    )
    update_status(job.run_dir, "running")
    benchmark_dir = job.run_dir / "artifacts" / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    sources_path = benchmark_dir / "benchmark_matrix_workspace_sources.csv"
    pd.DataFrame(source_rows).to_csv(sources_path, index=False)
    workspace_path = benchmark_dir / "matrix_workspace.json"
    write_json(
        workspace_path,
        {
            "name": clean_name,
            "selections": selections,
            "source_count": len({str(row["run_id"]) for row in source_rows if str(row.get("run_id") or "")}),
            "cell_count": len(source_rows),
            "engines": sorted(engines),
            "targets": sorted(targets),
        },
    )
    finish_job(
        job.run_dir,
        True,
        {
            "metrics": {
                "workspace_name": clean_name,
                "source_run_count": len({str(row["run_id"]) for row in source_rows if str(row.get("run_id") or "")}),
                "cell_count": len(source_rows),
                "engine_count": len(engines),
                "target_count": len(targets),
                "benchmark_matrix_workspace_sources": str(sources_path.relative_to(job.run_dir)),
                "matrix_workspace": str(workspace_path.relative_to(job.run_dir)),
            },
            "outputs": {
                "benchmark_matrix_workspace_sources": str(sources_path.relative_to(job.run_dir)),
                "matrix_workspace": str(workspace_path.relative_to(job.run_dir)),
            },
        },
    )
    return job.run_dir


def run_precomputed_metric_benchmark(
    *,
    input_csv: Path | None = None,
    input_csv_text: str | None = None,
    use_published_dataset: bool = False,
    label_column: str = "binder",
    max_rows: int = 0,
    max_columns: int = 250,
) -> Path:
    source_csv = PUBLISHED_DATASET if use_published_dataset else input_csv
    job = create_job(
        BENCHMARK_GROUP,
        "metric_dataset_benchmark",
        "de_novo_binder_scoring_metrics",
        {"input_csv": str(source_csv) if source_csv else None, "use_published_dataset": use_published_dataset},
        {"label_column": label_column, "max_rows": max_rows, "max_columns": max_columns},
    )
    raw_dir = job.run_dir / "artifacts" / "raw" / "metric_dataset"
    raw_dir.mkdir(parents=True, exist_ok=True)
    staged_csv = raw_dir / "input_metrics.csv"
    if input_csv_text is not None:
        staged_csv.write_text(input_csv_text)
    elif source_csv is not None:
        shutil.copy2(source_csv, staged_csv)
    else:
        raise ValueError("Provide a metric CSV or select the published dataset.")
    update_status(job.run_dir, "running")
    df = pd.read_csv(staged_csv)
    if max_rows > 0:
        df = df.head(int(max_rows))
    feature_rows, summary = _metric_column_summary(df, label_column, max_columns=max_columns)
    out_dir = job.run_dir / "artifacts" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_table = out_dir / "metric_feature_benchmark.csv"
    summary_path = out_dir / "metric_feature_summary.json"
    _write_feature_ranking(feature_table, feature_rows)
    write_json(summary_path, summary)
    finish_job(
        job.run_dir,
        True,
        {
            "outputs": {
                "feature_table": str(feature_table.relative_to(job.run_dir)),
                "summary": str(summary_path.relative_to(job.run_dir)),
                "staged_csv": str(staged_csv.relative_to(job.run_dir)),
            },
            "metrics": summary,
            "downstream_artifacts": {"benchmark_table": str(feature_table.relative_to(job.run_dir))},
        },
    )
    return job.run_dir


def _score_record_metrics(metrics: dict[str, Any]) -> float:
    iptm = float(metrics.get("iptm") or 0.0)
    binder_plddt = float(metrics.get("binder_plddt") or 0.0)
    if binder_plddt > 1.5:
        binder_plddt /= 100.0
    ipae = float(metrics.get("ipae") or 31.0)
    ipsae = float(metrics.get("ipsae") or 0.0)
    contact_fraction = float(metrics.get("hotspot_contact_fraction") or 0.0)
    return (0.40 * iptm) + (0.25 * binder_plddt) + (0.20 * max(0.0, 1.0 - ipae / 31.0)) + (0.10 * ipsae) + (0.05 * contact_fraction)


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                continue
            archive.extract(member, out_dir)


def _stage_repo_format_inputs(
    *,
    raw_dir: Path,
    input_zip: Path | None,
    input_zip_bytes: bytes | None,
    input_csv: Path | None,
    input_csv_text: str | None,
    input_pdb_dir: Path | None,
) -> tuple[Path | None, Path | None]:
    dataset_dir = raw_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if input_zip_bytes is not None:
        staged_zip = raw_dir / "dataset_upload.zip"
        staged_zip.write_bytes(input_zip_bytes)
        _extract_zip(staged_zip, dataset_dir)
    elif input_zip is not None:
        _extract_zip(Path(input_zip).expanduser().resolve(), dataset_dir)
    staged_csv = None
    staged_pdb_dir = None
    if input_csv_text is not None:
        staged_csv = dataset_dir / "input.csv"
        staged_csv.write_text(input_csv_text)
    elif input_csv is not None:
        staged_csv = dataset_dir / "input.csv"
        shutil.copy2(Path(input_csv).expanduser().resolve(), staged_csv)
    else:
        for candidate in [dataset_dir / "input.csv", *dataset_dir.glob("**/input.csv")]:
            if candidate.exists():
                staged_csv = candidate
                break
    if input_pdb_dir is not None:
        staged_pdb_dir = dataset_dir / "input_pdbs"
        staged_pdb_dir.mkdir(parents=True, exist_ok=True)
        selected_ids: set[str] | None = None
        if staged_csv is not None and staged_csv.exists():
            try:
                staged_df = pd.read_csv(staged_csv, usecols=["binder_id"])
            except Exception:
                selected_ids = None
            else:
                selected_ids = {str(value).strip() for value in staged_df["binder_id"].dropna() if str(value).strip()}
        pdb_sources = sorted(Path(input_pdb_dir).expanduser().glob("*.pdb"))
        if selected_ids:
            selected_names = {f"{binder_id}.pdb" for binder_id in selected_ids}
            pdb_sources = [pdb for pdb in pdb_sources if pdb.name in selected_names]
        for pdb in pdb_sources:
            shutil.copy2(pdb, staged_pdb_dir / pdb.name)
            chain_map = pdb.with_suffix(".chain_map.json")
            if chain_map.exists():
                shutil.copy2(chain_map, staged_pdb_dir / chain_map.name)
    else:
        for candidate in [dataset_dir / "input_pdbs", *dataset_dir.glob("**/input_pdbs")]:
            if candidate.exists() and candidate.is_dir():
                staged_pdb_dir = candidate
                break
    return staged_csv, staged_pdb_dir


def _apply_staged_csv_metadata_to_run_csv(run_csv: Path, staged_csv: Path | None) -> dict[str, int]:
    if staged_csv is None or not staged_csv.exists() or not run_csv.exists():
        return {"staged_metadata_rows": 0, "staged_metadata_matched_rows": 0}
    try:
        staged_df = pd.read_csv(staged_csv)
        run_df = pd.read_csv(run_csv)
    except Exception:
        return {"staged_metadata_rows": 0, "staged_metadata_matched_rows": 0}
    if "binder_id" not in staged_df.columns or "binder_id" not in run_df.columns:
        return {"staged_metadata_rows": int(len(staged_df)), "staged_metadata_matched_rows": 0}

    base_metadata_cols = {
        "binder",
        "label",
        "source",
        "target_id",
        "binder_chain",
        "binder_chains",
        "target_chains",
        "target_only_chains",
        "legacy_capacity_binder_chains",
        "source_input_binder_chains",
        "source_input_target_chains",
        "source_parent_binder_chains",
        "source_parent_target_chains",
        "refolding_target_source_chains",
        "engine_target_chains",
        "source_target_chains",
        "chain_role_schema",
        "target_fragment_specs",
        "target_chain_range",
        "segment_ids",
    }
    technical_cols = {
        "complex_pdb",
        "target_pdb",
        "binder_pdb",
        "source_target_pdb",
        "source_input_run_dir",
        "source_input_complex_pdb",
        "aligned_target_pdb",
        "target_alignment_reference_pdb",
    }
    metadata_cols = []
    for col in staged_df.columns:
        if col in {"binder_id"} or col.startswith("msa_path_"):
            continue
        if col in base_metadata_cols or col not in run_df.columns:
            if col not in technical_cols:
                metadata_cols.append(col)
    if not metadata_cols:
        return {"staged_metadata_rows": int(len(staged_df)), "staged_metadata_matched_rows": 0}

    staged = staged_df.copy()
    staged["_safe_binder_id"] = staged["binder_id"].map(lambda value: _safe_id(value).lower())
    run_df["_safe_binder_id"] = run_df["binder_id"].map(lambda value: _safe_id(value).lower())
    meta = staged.drop_duplicates("_safe_binder_id").set_index("_safe_binder_id")
    duplicate_safe_rows = int(run_df.duplicated("_safe_binder_id", keep=False).sum())
    dropped_duplicate_rows = 0
    if duplicate_safe_rows:
        preferred_ids = meta["binder_id"].astype(str).to_dict()
        run_df["_row_order"] = range(len(run_df))
        run_df["_preferred_case"] = run_df.apply(
            lambda row: str(row.get("binder_id") or "") == preferred_ids.get(str(row.get("_safe_binder_id") or "")),
            axis=1,
        )
        before = len(run_df)
        run_df = (
            run_df.sort_values(["_safe_binder_id", "_preferred_case", "_row_order"], ascending=[True, False, True])
            .drop_duplicates("_safe_binder_id", keep="first")
            .sort_values("_row_order")
            .drop(columns=["_row_order", "_preferred_case"])
        )
        dropped_duplicate_rows = before - len(run_df)
    matched = run_df["_safe_binder_id"].isin(meta.index)
    for col in metadata_cols:
        mapped = run_df["_safe_binder_id"].map(meta[col])
        if col in {"target_id", "binder_chain", "target_chains", "target_chain_range"} and col in run_df.columns:
            run_df[col] = mapped.where(mapped.notna(), run_df[col])
        else:
            run_df[col] = mapped
    original_id_column = "original_binder_id" if "original_binder_id" in meta.columns else "binder_id"
    run_df["original_binder_id"] = run_df["_safe_binder_id"].map(meta[original_id_column])
    run_df = run_df.drop(columns=["_safe_binder_id"])
    run_df.to_csv(run_csv, index=False)
    return {
        "staged_metadata_rows": int(len(staged_df)),
        "staged_metadata_matched_rows": int(matched.sum()),
        "casefold_duplicate_input_rows": duplicate_safe_rows,
        "casefold_duplicate_input_rows_dropped": int(dropped_duplicate_rows),
    }


def _run_docker_command(run_dir: Path, command: list[str]) -> int:
    with (run_dir / "stdout.log").open("a") as stdout, (run_dir / "stderr.log").open("a") as stderr:
        stdout.write(f"$ {' '.join(command)}\n")
        stdout.flush()
        proc = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    return int(proc.returncode)


def _run_local_command(run_dir: Path, command: list[str], *, cwd: Path | None = None) -> int:
    with (run_dir / "stdout.log").open("a") as stdout, (run_dir / "stderr.log").open("a") as stderr:
        stdout.write(f"$ {' '.join(command)}\n")
        stdout.flush()
        proc = subprocess.run(command, cwd=cwd, stdout=stdout, stderr=stderr, check=False)
    return int(proc.returncode)


def _run_csv_chain_sequence(row: pd.Series, chain: str, binder_chain: str) -> str:
    candidates: list[str] = []
    if chain == binder_chain:
        candidates.extend([f"{chain}_seq", f"target_subchain_{chain}_seq", "binder_sequence", "sequence"])
        if chain == "A":
            candidates.append("A_seq")
    else:
        candidates.extend([f"target_subchain_{chain}_seq", f"{chain}_seq"])
    for column in candidates:
        value = row.get(column)
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _run_csv_is_capacity_target_only(row: pd.Series) -> bool:
    target_source = str(row.get("target_source") or "").strip().lower()
    if target_source == "capacity_target_only":
        return True
    raw = row.get("capacity_target_only")
    if raw is None or pd.isna(raw):
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def _validate_run_csv_chain_roles(run_csv: Path, *, max_records: int = 0) -> dict[str, Any]:
    if not run_csv.exists():
        return {"chain_role_validation_count": 0, "chain_role_finding_count": 0, "chain_role_warning_count": 0}
    try:
        df = pd.read_csv(run_csv)
    except Exception as exc:
        return {
            "chain_role_validation_count": 0,
            "chain_role_finding_count": 1,
            "chain_role_warning_count": 1,
            "chain_role_validation_error": f"{type(exc).__name__}: {exc}",
        }
    if max_records and max_records > 0:
        df = df.head(int(max_records))
    input_pdb_dir = run_csv.parent / "input_pdbs"
    findings: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        binder_id = str(row.get("binder_id") or "").strip()
        if not binder_id:
            continue
        target_only = _run_csv_is_capacity_target_only(row)
        binder_chain = str(row.get("binder_chain") or "A").strip() or "A"
        binder_chains = _split_list(row.get("binder_chains")) or [binder_chain]
        target_chains = _split_list(row.get("target_chains"))
        if target_only and target_chains:
            binder_chains = []
        structure_chains: list[str] = []
        pdb_path = input_pdb_dir / f"{binder_id}.pdb"
        if pdb_path.exists():
            structure_chains = refolding_workflow._structure_chains(pdb_path)
        schema = ""
        chain_map = pdb_path.with_suffix(".chain_map.json")
        if chain_map.exists():
            schema = str(read_json(chain_map).get("chain_role_schema") or "")
        findings.extend(
            chain_roles.validate_chain_roles(
                candidate_id=binder_id,
                binder_chains=binder_chains,
                target_chains=target_chains,
                structure_chains=structure_chains,
                schema=schema,
                target_only=target_only,
            )
        )
    warnings = chain_roles.problem_findings(findings)
    finding_path = run_csv.parent / "chain_role_findings.json"
    warning_path = run_csv.parent / "chain_role_warnings.json"
    if findings:
        write_json(finding_path, findings)
    if warnings:
        write_json(warning_path, warnings)
    return {
        "chain_role_validation_count": int(len(df)),
        "chain_role_finding_count": len(findings),
        "chain_role_finding_path": str(finding_path) if findings else None,
        "chain_role_warning_count": len(warnings),
        "chain_role_warning_path": str(warning_path) if warnings else None,
    }


def _run_csv_target_msa_chains(row: pd.Series, binder_chain: str) -> list[str]:
    if _run_csv_is_capacity_target_only(row):
        return (
            _split_list(row.get("target_chains"))
            or _split_list(row.get("target_only_chains"))
            or _split_list(row.get("legacy_capacity_binder_chains"))
            or _split_list(row.get("binder_chains"))
            or [binder_chain]
        )
    return [chain for chain in _split_list(row.get("target_chains")) if chain and chain != binder_chain]


def _repo_run_csv_to_esmfold2_csv(run_dir: Path, run_csv: Path) -> str:
    df = pd.read_csv(run_csv)
    rows: list[dict[str, Any]] = []
    input_pdb_dir = run_csv.parent / "input_pdbs"
    for _, row in df.iterrows():
        binder_id = str(row.get("binder_id") or "").strip()
        if not binder_id:
            continue
        complex_pdb = input_pdb_dir / f"{binder_id}.pdb"
        if not complex_pdb.exists():
            continue
        target_chains = _split_list(row.get("target_chains"))
        if not target_chains and not _run_csv_is_capacity_target_only(row):
            try:
                parsed = json.loads(str(row.get("target_chains") or "[]"))
                target_chains = [str(item) for item in parsed]
            except Exception:
                target_chains = ["B"]
        binder_chain = str(row.get("binder_chain") or "A").strip() or "A"
        capacity_target_only = bool(_run_csv_is_capacity_target_only(row))
        target_only_chains = _run_csv_target_msa_chains(row, binder_chain) if capacity_target_only else []
        binder_chains = target_only_chains if capacity_target_only else (_split_list(row.get("binder_chains")) or [binder_chain])
        if capacity_target_only:
            target_chains = target_only_chains
        binder_sequence = (
            "".join(_run_csv_chain_sequence(row, chain, binder_chain) for chain in binder_chains)
            if capacity_target_only
            else _run_csv_chain_sequence(row, binder_chain, binder_chain)
        )
        target_pdb_value = str(row.get("target_pdb") or "").strip()
        target_pdb = Path(target_pdb_value).expanduser() if target_pdb_value else complex_pdb
        if not target_pdb.is_absolute():
            target_pdb = (run_csv.parent / target_pdb).resolve()
        if not target_pdb.exists():
            target_pdb = complex_pdb
        msa_columns = {
            str(col): str(row.get(col) or "").strip()
            for col in df.columns
            if str(col).startswith("msa_path_") and str(row.get(col) or "").strip()
        }
        target_fragment_columns = {
            str(col): row.get(col)
            for col in df.columns
            if str(col).startswith("target_subchain_")
        }
        rows.append(
            {
                "candidate_id": binder_id,
                "target_id": row.get("target_id") or "",
                "source": row.get("source") or "",
                "label": row.get("binder") if "binder" in df.columns else row.get("label"),
                "target_pdb": str(target_pdb.resolve()),
                "complex_pdb": str(complex_pdb.resolve()),
                "binder_sequence": binder_sequence,
                "binder_chains": ",".join(binder_chains if capacity_target_only else [binder_chain]),
                "target_chains": ",".join(target_chains),
                "target_source": row.get("target_source") or "",
                "capacity_target_only": capacity_target_only,
                **msa_columns,
                **target_fragment_columns,
            }
        )
    out = run_dir / "artifacts" / "benchmark" / "esmfold2_input.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return out.read_text()


def _repo_run_csv_to_candidates(run_dir: Path, run_csv: Path, max_records: int = 0) -> Path:
    df = pd.read_csv(run_csv)
    if max_records > 0:
        df = df.head(int(max_records))
    rows: list[dict[str, Any]] = []
    input_pdb_dir = run_csv.parent / "input_pdbs"
    for _, row in df.iterrows():
        binder_id = str(row.get("binder_id") or "").strip()
        if not binder_id:
            continue
        complex_pdb = input_pdb_dir / f"{binder_id}.pdb"
        if not complex_pdb.exists():
            continue
        target_pdb_value = str(row.get("target_pdb") or "").strip()
        target_pdb = Path(target_pdb_value).expanduser() if target_pdb_value else None
        if target_pdb is not None and not target_pdb.is_absolute():
            target_pdb = (run_csv.parent / target_pdb).resolve()
        if target_pdb is None or not target_pdb.exists():
            target_pdb = complex_pdb
        binder_chain = str(row.get("binder_chain") or "A").strip() or "A"
        target_chains = _split_list(row.get("target_chains"))
        if not target_chains and not _run_csv_is_capacity_target_only(row):
            try:
                parsed = json.loads(str(row.get("target_chains") or "[]"))
                target_chains = [str(item) for item in parsed if str(item)]
            except Exception:
                target_chains = ["B"]
        capacity_target_only = bool(_run_csv_is_capacity_target_only(row))
        target_only_chains = _run_csv_target_msa_chains(row, binder_chain) if capacity_target_only else []
        binder_chains = target_only_chains if capacity_target_only else (_split_list(row.get("binder_chains")) or [binder_chain])
        if capacity_target_only:
            target_chains = target_only_chains
        binder_sequence = (
            "".join(_run_csv_chain_sequence(row, chain, binder_chain) for chain in binder_chains)
            if capacity_target_only
            else _run_csv_chain_sequence(row, binder_chain, binder_chain)
        )
        label = _truthy_label(row.get("binder") if "binder" in df.columns else row.get("label"))
        rows.append(
            {
                "candidate_id": binder_id,
                "stage": STAGE_COMPLEX_REFOLDING,
                "source_tool": "de_novo_binder_scoring_dataset",
                "target_id": str(row.get("target_id") or ""),
                "target_pdb": str(target_pdb),
                "complex_pdb": str(complex_pdb.relative_to(run_dir)),
                "binder_sequence": binder_sequence,
                "binder_chains": [] if capacity_target_only else [binder_chain],
                "legacy_capacity_binder_chains": binder_chains if capacity_target_only else [],
                "target_chains": target_chains,
                "target_only": bool(capacity_target_only),
                "metrics": {
                    "label": label,
                    "binder": label,
                    "target_id": row.get("target_id") or "",
                    "source": row.get("source") or "",
                },
                "raw_metadata": {
                    "repo_run_csv_row": {str(key): value for key, value in row.to_dict().items()},
                    "capacity_target_only": capacity_target_only,
                    "biological_target_chains": target_chains if capacity_target_only else [],
                    "staged_target_chains": target_chains if capacity_target_only else [],
                    "legacy_capacity_binder_chains": binder_chains if capacity_target_only else [],
                },
            }
        )
    normalized = write_candidates(run_dir, "de_novo_binder_scoring_dataset", rows)
    if not normalized:
        raise ValueError("No repo dataset candidates with PDB inputs could be prepared for AF2 initial guess.")
    return run_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl"


def _summarize_child_candidate_metrics(
    *,
    parent_run_dir: Path,
    child_run_dir: Path,
    output_prefix: str,
) -> tuple[Path | None, Path | None, dict[str, Any]]:
    candidates = read_candidates(child_run_dir)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        metrics = dict(candidate.get("metrics") or {})
        parents = candidate.get("parents") or []
        binder_id = str(parents[0]) if parents else str(candidate.get("candidate_id") or "")
        raw_source = (candidate.get("raw_metadata") or {}).get("source_candidate") or {}
        if isinstance(raw_source, dict) and raw_source.get("candidate_id"):
            binder_id = str(raw_source.get("candidate_id"))
        row = {
            "binder_id": binder_id,
            "candidate_id": candidate.get("candidate_id"),
            "label": metrics.get("label", metrics.get("binder")),
            "complex_pdb": candidate.get("complex_pdb"),
        }
        row.update(metrics)
        rows.append(row)
    if not rows:
        return None, None, {"record_count": 0, "scored_feature_count": 0}
    out_dir = parent_run_dir / "artifacts" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = out_dir / f"{output_prefix}_metrics.csv"
    summary_path = out_dir / f"{output_prefix}_feature_summary.json"
    df = pd.DataFrame(rows)
    if output_prefix == "esmfold2":
        df = _prefix_engine_metric_columns(
            df,
            prefix="esmfold2",
            passthrough={"binder_id", "candidate_id", "label", "complex_pdb"},
        )
    elif output_prefix == "af2_initial_guess":
        df = _prefix_engine_metric_columns(
            df,
            prefix="af2",
            passthrough={"binder_id", "candidate_id", "label", "complex_pdb"},
        )
    df.to_csv(table, index=False)
    feature_rows, summary = _metric_column_summary(df, "label", max_columns=500)
    summary.update(
        {
            "engine_artifact_dir": f"artifacts/engines/{'boltz2' if output_prefix == 'boltz2_initial_guess' else output_prefix}",
            "scored_feature_count": len(feature_rows),
            "top_feature": feature_rows[0]["feature"] if feature_rows else None,
            "top_feature_average_precision": feature_rows[0]["best_average_precision"] if feature_rows else None,
            "top_feature_auroc": feature_rows[0]["best_auroc"] if feature_rows else None,
        }
    )
    write_json(summary_path, summary)
    _write_feature_ranking(out_dir / f"{output_prefix}_feature_benchmark.csv", feature_rows)
    return table, summary_path, summary


def _rename_metric_prefix(csv_path: Path, old_prefix: str, new_prefix: str) -> None:
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    rename = {
        column: f"{new_prefix}{str(column)[len(old_prefix):]}"
        for column in df.columns
        if str(column).startswith(old_prefix)
    }
    if rename:
        df = df.rename(columns=rename)
        df.to_csv(csv_path, index=False)


def _prefix_engine_metric_columns(
    df: pd.DataFrame,
    *,
    prefix: str,
    passthrough: set[str] | None = None,
) -> pd.DataFrame:
    keep = passthrough or set()
    rename: dict[str, str] = {}
    for column in df.columns:
        column_text = str(column)
        if column_text in keep or column_text.startswith(f"{prefix}_"):
            continue
        rename[column_text] = f"{prefix}_{column_text}"
    return df.rename(columns=rename) if rename else df


def _parse_json_list(value: Any) -> list[str]:
    if value is None:
        return []
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [token.strip() for token in re.split(r"[,;\s]+", text) if token.strip()]


def _structure_ca_atoms(structure: object, chains: list[str]) -> list[object]:
    model = next(structure.get_models(), None)
    if model is None:
        return []
    atoms: list[object] = []
    for chain_id in chains:
        if chain_id not in model:
            continue
        for residue in model[chain_id]:
            if getattr(residue, "id", ("",))[0] == " " and "CA" in residue:
                atoms.append(residue["CA"])
    return atoms


def _target_aligned_binder_pose_metrics(
    reference_pdb: Path,
    prediction_pdb: Path,
    *,
    reference_binder_chains: list[str],
    reference_target_chains: list[str],
    prediction_binder_chains: list[str],
    prediction_target_chains: list[str],
) -> dict[str, Any]:
    from Bio.PDB import PDBParser, Superimposer

    parser = PDBParser(QUIET=True)
    reference = parser.get_structure("reference", str(reference_pdb))
    prediction = parser.get_structure("prediction", str(prediction_pdb))
    reference_target_atoms = _structure_ca_atoms(reference, reference_target_chains)
    prediction_target_atoms = _structure_ca_atoms(prediction, prediction_target_chains)
    target_count = min(len(reference_target_atoms), len(prediction_target_atoms))
    if target_count < 3:
        return {
            "target_aligned_binder_rmsd": None,
            "target_alignment_rmsd": None,
            "target_alignment_ca_count": target_count,
            "binder_ca_count": 0,
            "pose_rmsd_status": f"target alignment skipped: {target_count} shared target CA atoms",
        }
    superimposer = Superimposer()
    superimposer.set_atoms(reference_target_atoms[:target_count], prediction_target_atoms[:target_count])
    superimposer.apply(prediction.get_atoms())
    reference_binder_atoms = _structure_ca_atoms(reference, reference_binder_chains)
    prediction_binder_atoms = _structure_ca_atoms(prediction, prediction_binder_chains)
    binder_count = min(len(reference_binder_atoms), len(prediction_binder_atoms))
    if binder_count < 3:
        return {
            "target_aligned_binder_rmsd": None,
            "target_alignment_rmsd": float(superimposer.rms),
            "target_alignment_ca_count": target_count,
            "binder_ca_count": binder_count,
            "pose_rmsd_status": f"binder RMSD skipped: {binder_count} shared binder CA atoms",
        }
    total = 0.0
    for fixed_atom, moving_atom in zip(reference_binder_atoms[:binder_count], prediction_binder_atoms[:binder_count]):
        delta = fixed_atom.coord - moving_atom.coord
        total += float((delta * delta).sum())
    return {
        "target_aligned_binder_rmsd": float((total / binder_count) ** 0.5),
        "target_alignment_rmsd": float(superimposer.rms),
        "target_alignment_ca_count": target_count,
        "binder_ca_count": binder_count,
        "pose_rmsd_status": "ok",
    }


def _write_target_aligned_binder_rmsd_table(
    *,
    job_run_dir: Path,
    run_csv: Path,
    predicted_pdb_dirs: list[tuple[str, Path]],
    output_path: Path,
) -> tuple[Path | None, dict[str, int]]:
    if not predicted_pdb_dirs or not run_csv.exists():
        return None, {"pose_rmsd_rows": 0, "pose_rmsd_ok": 0}
    run_df = pd.read_csv(run_csv)
    if "binder_id" not in run_df.columns:
        return None, {"pose_rmsd_rows": 0, "pose_rmsd_ok": 0}
    rows: list[dict[str, Any]] = []
    for _, source_row in run_df.iterrows():
        binder_id = str(source_row.get("binder_id") or "").strip()
        if not binder_id:
            continue
        safe_id = _safe_id(binder_id)
        reference_value = str(
            source_row.get("source_parent_complex_pdb")
            or source_row.get("complex_pdb")
            or source_row.get("source_input_complex_pdb")
            or ""
        ).strip()
        if not reference_value:
            continue
        reference_pdb = Path(reference_value)
        if not reference_pdb.is_absolute():
            reference_pdb = job_run_dir / reference_pdb
        if not reference_pdb.exists():
            continue
        reference_binder_chains = (
            _parse_json_list(source_row.get("source_parent_binder_chains"))
            or _parse_json_list(source_row.get("binder_chains"))
            or _parse_json_list(source_row.get("binder_chain"))
            or _parse_json_list(source_row.get("source_input_binder_chains"))
            or ["A"]
        )
        reference_target_chains = (
            _parse_json_list(source_row.get("source_parent_target_chains"))
            or _parse_json_list(source_row.get("target_chains"))
            or _parse_json_list(source_row.get("source_input_target_chains"))
            or ["B"]
        )
        prediction_binder_chains = _parse_json_list(source_row.get("binder_chains")) or _parse_json_list(source_row.get("binder_chain")) or ["A"]
        prediction_target_chains = _parse_json_list(source_row.get("target_chains")) or ["B"]
        row: dict[str, Any] = {"binder_id": binder_id}
        for engine_prefix, folder in predicted_pdb_dirs:
            prediction_pdb = folder / f"{safe_id}.pdb"
            if not prediction_pdb.exists():
                prediction_pdb = folder / f"{safe_id.lower()}.pdb"
            if not prediction_pdb.exists():
                row[f"{engine_prefix}_pose_rmsd_status"] = "missing prediction PDB"
                continue
            try:
                metrics = _target_aligned_binder_pose_metrics(
                    reference_pdb,
                    prediction_pdb,
                    reference_binder_chains=reference_binder_chains,
                    reference_target_chains=reference_target_chains,
                    prediction_binder_chains=prediction_binder_chains,
                    prediction_target_chains=prediction_target_chains,
                )
            except Exception as exc:
                metrics = {
                    "target_aligned_binder_rmsd": None,
                    "target_alignment_rmsd": None,
                    "target_alignment_ca_count": 0,
                    "binder_ca_count": 0,
                    "pose_rmsd_status": f"failed: {type(exc).__name__}: {exc}",
                }
            for key, value in metrics.items():
                row[f"{engine_prefix}_{key}"] = value
        rows.append(row)
    if not rows:
        return None, {"pose_rmsd_rows": 0, "pose_rmsd_ok": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output_path, index=False)
    ok_count = sum(
        1
        for row in rows
        for key, value in row.items()
        if key.endswith("_pose_rmsd_status") and value == "ok"
    )
    return output_path, {"pose_rmsd_rows": len(rows), "pose_rmsd_ok": ok_count}


def _ca_bfactor_plddt(pdb_path: Path) -> list[float]:
    values: list[float] = []
    seen: set[tuple[str, str, str]] = set()
    for line in pdb_path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        chain_id = line[21:22].strip()
        residue_number = line[22:26].strip()
        insertion_code = line[26:27].strip()
        key = (chain_id, residue_number, insertion_code)
        if key in seen:
            continue
        seen.add(key)
        try:
            values.append(float(line[60:66]))
        except ValueError:
            values.append(0.0)
    return values


def _write_af2_adapter_json(pae_path: Path, pdb_path: Path, target_path: Path) -> bool:
    try:
        payload = json.loads(pae_path.read_text())
    except json.JSONDecodeError:
        return False
    plddt = _ca_bfactor_plddt(pdb_path)
    if plddt:
        payload["plddt"] = plddt
    target_path.write_text(json.dumps(payload))
    return True


def _esmfold2_chain_role_map(pae_path: Path) -> dict[str, str]:
    try:
        payload = json.loads(pae_path.read_text())
    except json.JSONDecodeError:
        return {}
    binder_chain = str(payload.get("binder_chain") or "").strip()
    target_chains = [
        str(chain).strip()
        for chain in payload.get("target_chains", [])
        if str(chain).strip()
    ]
    chain_map: dict[str, str] = {}
    if binder_chain:
        chain_map[binder_chain] = "A"
    available_target_ids = [chr(code) for code in range(ord("B"), ord("Z") + 1)]
    for target_chain, mapped_chain in zip(target_chains, available_target_ids):
        chain_map[target_chain] = mapped_chain
    return chain_map


def _write_pdb_with_chain_map(source_path: Path, target_path: Path, chain_map: dict[str, str]) -> None:
    lines: list[str] = []
    for line in source_path.read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")) and len(line) > 21:
            mapped_chain = chain_map.get(line[21].strip())
            if mapped_chain:
                line = f"{line[:21]}{mapped_chain[:1]}{line[22:]}"
        lines.append(line)
    target_path.write_text("\n".join(lines) + "\n")


def _link_or_copy(source_path: Path, target_path: Path) -> None:
    _replace_path_with_link_or_copy(source_path, target_path)


def _scale_fractional_pdb_confidence(path: Path) -> bool:
    """Convert fractional PDB confidence B-factors to the pLDDT 0-100 scale."""
    lines = path.read_text(errors="ignore").splitlines()
    ca_values: list[float] = []
    for line in lines:
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
            continue
        try:
            ca_values.append(float(line[60:66]))
        except ValueError:
            continue
    if not ca_values or min(ca_values) < 0.0 or max(ca_values) > 1.5:
        return False
    scaled: list[str] = []
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 66:
            try:
                confidence = min(100.0, max(0.0, float(line[60:66]) * 100.0))
                line = f"{line[:60]}{confidence:6.2f}{line[66:]}"
            except ValueError:
                pass
        scaled.append(line)
    path.write_text("\n".join(scaled) + "\n")
    return True


def _normalize_structure_for_group_metrics(
    source_path: Path,
    target_path: Path,
    *,
    binder_chains: list[str],
    target_chains: list[str],
) -> bool:
    structure_chains = refolding_workflow._structure_chains(source_path)
    if not binder_chains:
        target_group = [chain for chain in structure_chains if not target_chains or chain in set(target_chains)] or structure_chains
        if not target_group:
            return False
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if source_path.suffix.lower() == ".pdb":
            shutil.copy2(source_path, target_path)
        else:
            refolding_workflow._cif_to_pdb(source_path, target_path)
        target_rows = [
            {"original_chain": original_chain, "engine_chain": engine_chain, "role": "target"}
            for original_chain, engine_chain in zip(target_chains or target_group, target_group)
        ]
        write_json(
            target_path.with_suffix(".chain_map.json"),
            {
                "source_structure": str(source_path),
                "evaluation_contract": {
                    "target_chains": target_group,
                    "target_only": True,
                },
                "binder_source_chains": [],
                "target_source_chains": target_chains,
                "target_observed_chains": target_group,
                "target_unrepresented_source_chains": [
                    chain for chain in target_chains if chain not in {row["original_chain"] for row in target_rows}
                ],
                "targets": target_rows,
                "scope": "temporary target-only evaluation copy",
            },
        )
        return True
    binder_group = [chain for chain in binder_chains if chain in structure_chains]
    target_group = [chain for chain in target_chains if chain in structure_chains and chain not in binder_group]
    if not binder_group or not target_group:
        if "A" in structure_chains and "B" in structure_chains:
            binder_group = ["A"]
            target_group = [chain for chain in structure_chains if chain != "A"]
        else:
            return False
    binder_lines, next_atom = refolding_workflow._renumber_structure_chain(
        source_path,
        "A",
        1,
        set(binder_group),
    )
    target_lines, _ = refolding_workflow._renumber_structure_chain(
        source_path,
        "B",
        next_atom,
        set(target_group),
    )
    if not binder_lines or not target_lines:
        return False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text("\n".join(binder_lines + ["TER"] + target_lines + ["TER", "END", ""]))
    write_json(
        target_path.with_suffix(".chain_map.json"),
        {
            "source_structure": str(source_path),
            "evaluation_contract": {
                "binder_chain": "A",
                "target_chain": "B",
            },
            "binder_source_chains": binder_group,
            "target_source_chains": target_group,
            "scope": "temporary PyRosetta/PyMOL evaluation copy",
        },
    )
    return True


def _benchmark_group_roles(run_csv: Path) -> dict[str, tuple[list[str], list[str]]]:
    if not run_csv.exists():
        return {}
    roles: dict[str, tuple[list[str], list[str]]] = {}
    for _, row in pd.read_csv(run_csv).iterrows():
        binder_id = str(row.get("binder_id") or "").strip()
        if not binder_id:
            continue
        capacity_target_only = str(row.get("capacity_target_only") or "").strip().lower() in {"1", "true", "yes", "y"}
        if capacity_target_only:
            binder_chains = []
            target_chains = (
                _split_list(row.get("refolding_target_source_chains"))
                or _split_list(row.get("target_only_chains"))
                or _split_list(row.get("target_chains"))
                or _split_list(row.get("binder_chains"))
            )
        else:
            binder_chains = _split_list(row.get("binder_chains")) or _split_list(row.get("binder_chain")) or ["A"]
            target_chains = _split_list(row.get("target_chains"))
        roles[binder_id] = (binder_chains, target_chains)
        roles[_safe_id(binder_id)] = (binder_chains, target_chains)
    return roles


def _roles_for_model_name(
    model_name: str,
    roles: dict[str, tuple[list[str], list[str]]],
) -> tuple[list[str], list[str]]:
    matches = [key for key in roles if model_name == key or model_name.startswith(f"{key}_")]
    if matches:
        return roles[max(matches, key=len)]
    return ["A"], ["B"]


def _strip_pdb_suffixes(name: str) -> str:
    clean = str(name).strip()
    while clean.lower().endswith(".pdb"):
        clean = clean[:-4]
    return clean


def _is_staged_input_structure_path(path: Path | str) -> bool:
    """Return True for synthetic/input structures that must not count as predictions."""
    text = str(path).replace("\\", "/")
    staged_markers = (
        "/input_pdbs/",
        "/queued_inputs/",
        "/de_novo_binder_scoring/output/input_pdbs/",
    )
    return any(marker in text for marker in staged_markers)


def _stage_child_candidate_pdbs(
    child_runs: list[str],
    stage_dir: Path,
    *,
    source_label: str,
) -> tuple[Path | None, int]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for raw_child in child_runs:
        child_dir = Path(str(raw_child))
        if not (child_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl").exists():
            continue
        for candidate in read_candidates(child_dir):
            parents = candidate.get("parents") or []
            binder_id = str(parents[0]) if parents else str(candidate.get("candidate_id") or "")
            raw_source = (candidate.get("raw_metadata") or {}).get("source_candidate") or {}
            raw_record = (candidate.get("raw_metadata") or {}).get("benchmark_record") or {}
            if isinstance(raw_source, dict) and raw_source.get("candidate_id"):
                binder_id = str(raw_source.get("candidate_id"))
            if isinstance(raw_record, dict) and raw_record.get("candidate_id"):
                binder_id = str(raw_record.get("candidate_id"))
            complex_rel = candidate.get("complex_pdb")
            if not binder_id or not complex_rel:
                continue
            safe_id = _safe_id(binder_id)
            viewer_prediction_path = _child_viewer_prediction_path(child_dir, source_label, safe_id)
            if viewer_prediction_path is None and _is_staged_input_structure_path(complex_rel):
                continue
            complex_path = viewer_prediction_path or child_dir / str(complex_rel)
            if not complex_path.exists():
                continue
            target = stage_dir / f"{safe_id}.pdb"
            if target.exists() or target.is_symlink():
                target.unlink()
            binder_chains = [str(chain) for chain in candidate.get("binder_chains") or [] if str(chain)]
            target_chains = [str(chain) for chain in candidate.get("target_chains") or [] if str(chain)]
            if complex_path.suffix.lower() == ".pdb":
                if not _normalize_structure_for_group_metrics(
                    complex_path,
                    target,
                    binder_chains=binder_chains,
                    target_chains=target_chains,
                ):
                    continue
            elif complex_path.suffix.lower() == ".cif" or complex_path.name.endswith(".cif.gz"):
                tmp = stage_dir / f"{_safe_id(binder_id)}.{source_label}.raw.pdb"
                refolding_workflow._cif_to_pdb(complex_path, tmp)
                if _normalize_structure_for_group_metrics(
                    tmp,
                    target,
                    binder_chains=binder_chains,
                    target_chains=target_chains,
                ):
                    tmp.unlink(missing_ok=True)
                else:
                    tmp.unlink(missing_ok=True)
                    continue
            else:
                continue
            count += 1
    return (stage_dir if count else None), count


def _stage_engine_artifact_candidate_pdbs(
    parent_run_dir: Path,
    engine_dir: Path,
    stage_dir: Path,
    *,
    source_label: str,
) -> tuple[Path | None, int]:
    candidates_path = engine_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl"
    if not candidates_path.exists():
        return None, 0
    stage_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for line in candidates_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        parents = candidate.get("parents") or []
        binder_id = str(parents[0]) if parents else str(candidate.get("candidate_id") or "")
        raw_source = (candidate.get("raw_metadata") or {}).get("source_candidate") or {}
        raw_record = (candidate.get("raw_metadata") or {}).get("benchmark_record") or {}
        if isinstance(raw_source, dict) and raw_source.get("candidate_id"):
            binder_id = str(raw_source.get("candidate_id"))
        if isinstance(raw_record, dict) and raw_record.get("candidate_id"):
            binder_id = str(raw_record.get("candidate_id"))
        complex_rel = candidate.get("complex_pdb")
        if not binder_id or not complex_rel:
            continue
        safe_id = _safe_id(binder_id)
        complex_path = Path(str(complex_rel))
        if not complex_path.is_absolute():
            complex_path = parent_run_dir / complex_path
        if not complex_path.exists():
            fallback = _child_viewer_prediction_path(parent_run_dir, source_label, safe_id)
            if fallback is not None:
                complex_path = fallback
        if not complex_path.exists():
            continue
        target = stage_dir / f"{safe_id}.pdb"
        if target.exists() or target.is_symlink():
            target.unlink()
        binder_chains = [str(chain) for chain in candidate.get("binder_chains") or [] if str(chain)]
        target_chains = [str(chain) for chain in candidate.get("target_chains") or [] if str(chain)]
        if complex_path.suffix.lower() == ".pdb":
            if not _normalize_structure_for_group_metrics(
                complex_path,
                target,
                binder_chains=binder_chains,
                target_chains=target_chains,
            ):
                continue
        elif complex_path.suffix.lower() == ".cif" or complex_path.name.endswith(".cif.gz"):
            tmp = stage_dir / f"{safe_id}.{source_label}.raw.pdb"
            refolding_workflow._cif_to_pdb(complex_path, tmp)
            if _normalize_structure_for_group_metrics(
                tmp,
                target,
                binder_chains=binder_chains,
                target_chains=target_chains,
            ):
                tmp.unlink(missing_ok=True)
            else:
                tmp.unlink(missing_ok=True)
                continue
        else:
            continue
        count += 1
    return (stage_dir if count else None), count


def _child_viewer_prediction_path(child_dir: Path, source_label: str, safe_id: str) -> Path | None:
    """Return an engine-native prediction structure when the candidate points to staged input."""
    search_roots: list[Path] = []
    if source_label in {"af3", "alphafast_af3"}:
        search_roots.extend(
            [
                child_dir
                / "artifacts"
                / "engines"
                / "alphafast_af3"
                / "alphafast_output"
                / safe_id,
                child_dir / "artifacts" / "raw" / "alphafast_af3" / "alphafast_output" / safe_id,
            ]
        )
    elif source_label == "af2":
        search_roots.extend(
            [
                child_dir
                / "artifacts"
                / "engines"
                / "af2_initial_guess"
                / "artifacts"
                / "raw"
                / "af2_initial_guess"
                / "output"
                / "af2_initial_guess",
                child_dir / "artifacts" / "raw" / "af2_initial_guess" / "output" / "af2_initial_guess",
            ]
        )
    elif source_label in {"colab", "colabfold"}:
        search_roots.extend(
            [
                child_dir / "artifacts" / "engines" / "colabfold" / "ptm_output",
                child_dir / "artifacts" / "engines" / "colabfold" / "pdbs",
                child_dir / "artifacts" / "raw" / "colabfold" / "ptm_output",
                child_dir / "artifacts" / "raw" / "colabfold" / "pdbs",
            ]
        )
    elif source_label == "boltz2":
        search_roots.extend(
            [
                child_dir
                / "artifacts"
                / "engines"
                / "boltz2"
                / "artifacts"
                / "raw"
                / "boltz2_initial_guess"
                / "output"
                / "predictions"
                / safe_id,
                child_dir
                / "artifacts"
                / "raw"
                / "boltz2_initial_guess"
                / "output"
                / "predictions"
                / safe_id,
            ]
        )
    elif source_label == "boltzgen_fold":
        search_roots.extend(
            [
                child_dir
                / "artifacts"
                / "engines"
                / "boltzgen_fold"
                / "artifacts"
                / "raw"
                / "boltzgen_fold"
                / "output"
                / safe_id,
                child_dir / "artifacts" / "raw" / "boltzgen_fold" / "output" / safe_id,
            ]
        )
    elif source_label == "protenix":
        search_roots.extend(
            [
                child_dir
                / "artifacts"
                / "engines"
                / "protenix"
                / "artifacts"
                / "raw"
                / "protenix"
                / "output"
                / safe_id,
                child_dir / "artifacts" / "raw" / "protenix" / "output" / safe_id,
            ]
        )
    for root in search_roots:
        if not root.exists():
            continue
        def _native(paths: Iterable[Path]) -> list[Path]:
            return [path for path in sorted(paths) if "_targetgroup_" not in path.name]

        matches = _native(root.glob("*model_0.cif"))
        if not matches and source_label in {"af3", "alphafast_af3"}:
            matches = _native(root.glob("*model.cif"))
        if not matches and source_label == "af2":
            matches = (
                _native(root.glob(f"{safe_id}_af2_initial_guess.pdb"))
                or _native(root.glob(f"{safe_id}_af2_initial_guess_model*.pdb"))
                or _native(root.glob(f"{safe_id}_af2ig_*.pdb"))
                or _native(root.glob(f"{safe_id}_af2_initial_guess_binder_model*.pdb"))
            )
        if not matches and source_label in {"colab", "colabfold"}:
            matches = _native(root.glob(f"{safe_id}*rank_001*.pdb")) or _native(root.glob(f"{safe_id}*.pdb"))
        if not matches:
            matches = _native(root.glob("*sample_0.cif"))
        if not matches and source_label == "boltzgen_fold":
            matches = _native(root.rglob("*.cif"))
        if not matches and source_label == "protenix":
            matches = _native(root.rglob("*sample_0.cif"))
        if not matches:
            matches = _native(root.glob("*.cif")) + _native(root.glob("*.pdb"))
        if matches:
            return matches[0]
    return None


def _stage_child_viewer_structures(
    child_runs: list[str],
    stage_dir: Path,
    *,
    source_label: str,
) -> tuple[Path | None, int]:
    """Stage prediction structures for display without collapsing target chains."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for raw_child in child_runs:
        child_dir = Path(str(raw_child))
        if not (child_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl").exists():
            continue
        for candidate in read_candidates(child_dir):
            parents = candidate.get("parents") or []
            binder_id = str(parents[0]) if parents else str(candidate.get("candidate_id") or "")
            raw_source = (candidate.get("raw_metadata") or {}).get("source_candidate") or {}
            raw_record = (candidate.get("raw_metadata") or {}).get("benchmark_record") or {}
            if isinstance(raw_source, dict) and raw_source.get("candidate_id"):
                binder_id = str(raw_source.get("candidate_id"))
            if isinstance(raw_record, dict) and raw_record.get("candidate_id"):
                binder_id = str(raw_record.get("candidate_id"))
            complex_rel = candidate.get("complex_pdb")
            if not binder_id or not complex_rel:
                continue
            safe_id = _safe_id(binder_id)
            viewer_prediction_path = _child_viewer_prediction_path(child_dir, source_label, safe_id)
            if viewer_prediction_path is None and _is_staged_input_structure_path(complex_rel):
                # Some normalized candidates point at the staged synthetic input complex.
                # For capacity views, that would display the reference as a successful
                # prediction. Only stage these rows when an engine-native file was found.
                continue
            complex_path = viewer_prediction_path or child_dir / str(complex_rel)
            if not complex_path.exists():
                continue
            target = stage_dir / f"{safe_id}.pdb"
            if target.exists() or target.is_symlink():
                target.unlink()
            if complex_path.suffix.lower() == ".pdb":
                shutil.copy2(complex_path, target)
            elif complex_path.suffix.lower() == ".cif" or complex_path.name.endswith(".cif.gz"):
                refolding_workflow._cif_to_pdb(complex_path, target)
            else:
                continue
            chain_map_path = child_dir / str(complex_rel).replace(complex_path.name, f"{complex_path.stem}.chain_map.json")
            if chain_map_path.exists():
                chain_map_target = stage_dir / f"{safe_id}.chain_map.json"
                if chain_map_target.exists() or chain_map_target.is_symlink():
                    chain_map_target.unlink()
                _link_or_copy(chain_map_path, chain_map_target)
            count += 1
    return (stage_dir if count else None), count


def _stage_alphafast_pdbs(
    alphafast_output_dir: Path,
    stage_dir: Path,
    roles: dict[str, tuple[list[str], list[str]]],
) -> tuple[Path | None, int]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    if not alphafast_output_dir.exists():
        return None, 0
    for folder in sorted(path for path in alphafast_output_dir.iterdir() if path.is_dir()):
        binder_id = folder.name
        cif_candidates = [path for path in sorted(folder.glob("*.cif")) if "_targetgroup_" not in path.name]
        if not cif_candidates:
            continue
        target = stage_dir / f"{_safe_id(binder_id)}.pdb"
        if not target.exists():
            tmp = stage_dir / f"{_safe_id(binder_id)}.af3.raw.pdb"
            refolding_workflow._cif_to_pdb(cif_candidates[0], tmp)
            binder_chains, target_chains = _roles_for_model_name(binder_id, roles)
            if _normalize_structure_for_group_metrics(
                tmp,
                target,
                binder_chains=binder_chains,
                target_chains=target_chains,
            ):
                tmp.unlink(missing_ok=True)
            else:
                tmp.unlink(missing_ok=True)
                continue
        count += 1
    return (stage_dir if count else None), count


def _stage_alphafast_viewer_structures(
    alphafast_output_dir: Path,
    stage_dir: Path,
) -> tuple[Path | None, int]:
    """Stage AF3 native prediction structures without metric chain merging."""
    if not alphafast_output_dir.exists():
        return None, 0
    stage_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for folder in sorted(path for path in alphafast_output_dir.iterdir() if path.is_dir()):
        binder_id = folder.name
        cif_candidates = [path for path in sorted(folder.glob("*model.cif")) if "_targetgroup_" not in path.name]
        if not cif_candidates:
            cif_candidates = [path for path in sorted(folder.glob("*.cif")) if "_targetgroup_" not in path.name]
        if not cif_candidates:
            continue
        target = stage_dir / f"{_safe_id(binder_id)}.pdb"
        if target.exists() or target.is_symlink():
            target.unlink()
        refolding_workflow._cif_to_pdb(cif_candidates[0], target)
        count += 1
    return (stage_dir if count else None), count


def _stage_colabfold_pdbs(
    output_dir: Path,
    stage_dir: Path,
    roles: dict[str, tuple[list[str], list[str]]],
) -> tuple[Path | None, int]:
    pdb_dir = output_dir / "ColabFold" / "pdbs"
    if not (pdb_dir.exists() and any(pdb_dir.glob("*.pdb"))):
        pdb_dir = output_dir / "ColabFold" / "ptm_output"
    if not (pdb_dir.exists() and any(pdb_dir.glob("*.pdb"))):
        return None, 0
    stage_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    def _native_pdbs(paths: Iterable[Path]) -> list[Path]:
        return [path for path in sorted(paths) if "_targetgroup_" not in path.name]

    pdb_files = _native_pdbs(pdb_dir.glob("*_unrelaxed_rank_001*.pdb"))
    if not pdb_files:
        pdb_files = _native_pdbs(pdb_dir.glob("*_relaxed_rank_001*.pdb"))
    if not pdb_files:
        pdb_files = _native_pdbs(pdb_dir.glob("*.pdb"))
    for source in pdb_files:
        binder_id = source.name.split("_unrelaxed_rank_", 1)[0].split("_relaxed_rank_", 1)[0]
        binder_id = _strip_pdb_suffixes(binder_id)
        target = stage_dir / f"{_safe_id(binder_id)}.pdb"
        if target.exists():
            continue
        binder_chains, target_chains = _roles_for_model_name(binder_id, roles)
        if not _normalize_structure_for_group_metrics(
            source,
            target,
            binder_chains=binder_chains,
            target_chains=target_chains,
        ):
            continue
        count += 1
    return (stage_dir if count else None), count


def _stage_colabfold_viewer_structures(
    output_dir: Path,
    stage_dir: Path,
) -> tuple[Path | None, int]:
    """Stage ColabFold native prediction PDBs without metric chain merging."""
    search_roots = [
        output_dir / "ColabFold" / "ptm_output",
        output_dir / "ColabFold" / "pdbs",
    ]
    pdb_files: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        pdb_files = [path for path in sorted(root.glob("*_unrelaxed_rank_001*.pdb")) if "_targetgroup_" not in path.name]
        if not pdb_files:
            pdb_files = [path for path in sorted(root.glob("*_relaxed_rank_001*.pdb")) if "_targetgroup_" not in path.name]
        if not pdb_files:
            pdb_files = [path for path in sorted(root.glob("*.pdb")) if "_targetgroup_" not in path.name]
        if pdb_files:
            break
    if not pdb_files:
        return None, 0
    stage_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for source in pdb_files:
        binder_id = source.name.split("_unrelaxed_rank_", 1)[0].split("_relaxed_rank_", 1)[0]
        binder_id = _strip_pdb_suffixes(binder_id)
        target = stage_dir / f"{_safe_id(binder_id)}.pdb"
        if target.exists() or target.is_symlink():
            target.unlink()
        shutil.copy2(source, target)
        count += 1
    return (stage_dir if count else None), count


def _stage_input_pdbs_for_group_metrics(
    input_pdb_dir: Path,
    stage_dir: Path,
    roles: dict[str, tuple[list[str], list[str]]],
) -> tuple[Path | None, int]:
    if not input_pdb_dir.exists():
        return None, 0
    stage_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for source in sorted(input_pdb_dir.glob("*.pdb")):
        binder_id = source.stem
        target = stage_dir / f"{_safe_id(binder_id)}.pdb"
        binder_chains, target_chains = _roles_for_model_name(binder_id, roles)
        if not _normalize_structure_for_group_metrics(
            source,
            target,
            binder_chains=binder_chains,
            target_chains=target_chains,
        ):
            continue
        count += 1
    return (stage_dir if count else None), count


def _stage_af2_outputs_for_common_metrics(child_runs: list[str], adapter_dir: Path) -> tuple[Path | None, int]:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for raw_child in child_runs:
        child_dir = Path(str(raw_child))
        candidates_path = child_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl"
        if not candidates_path.exists():
            continue
        for candidate in read_candidates(child_dir):
            raw_source = (candidate.get("raw_metadata") or {}).get("source_candidate") or {}
            parents = candidate.get("parents") or []
            binder_id = str(parents[0]) if parents else str(candidate.get("candidate_id") or "")
            if isinstance(raw_source, dict) and raw_source.get("candidate_id"):
                binder_id = str(raw_source.get("candidate_id"))
            if not binder_id:
                continue
            complex_rel = candidate.get("complex_pdb")
            pae_rel = (candidate.get("raw_metadata") or {}).get("pae_path")
            if not complex_rel or not pae_rel:
                continue
            pdb_path = child_dir / str(complex_rel)
            pae_path = child_dir / str(pae_rel)
            if not pdb_path.exists() or not pae_path.exists():
                continue
            safe_id = _safe_id(binder_id)
            pdb_target = adapter_dir / f"{safe_id}_unrelaxed_rank_001_af2_initial_guess.pdb"
            json_target = adapter_dir / f"{safe_id}_scores_rank_001_af2_initial_guess.json"
            _link_or_copy(pdb_path, pdb_target)
            if not json_target.exists() and not _write_af2_adapter_json(pae_path, pdb_path, json_target):
                shutil.copy2(pae_path, json_target)
            count += 1
    return (adapter_dir if count else None), count


def _stage_esmfold2_outputs_for_common_metrics(child_runs: list[str], adapter_dir: Path) -> tuple[Path | None, int]:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for raw_child in child_runs:
        child_dir = Path(str(raw_child))
        candidates_path = child_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl"
        if not candidates_path.exists():
            continue
        for candidate in read_candidates(child_dir):
            parents = candidate.get("parents") or []
            binder_id = str(parents[0]) if parents else str(candidate.get("candidate_id") or "")
            raw_record = ((candidate.get("raw_metadata") or {}).get("benchmark_record") or {})
            if isinstance(raw_record, dict) and raw_record.get("candidate_id"):
                binder_id = str(raw_record.get("candidate_id"))
            if not binder_id:
                continue
            complex_rel = candidate.get("complex_pdb")
            pae_rel = (candidate.get("raw_metadata") or {}).get("pae_path")
            if not complex_rel or not pae_rel:
                continue
            cif_path = child_dir / str(complex_rel)
            pae_path = child_dir / str(pae_rel)
            if not cif_path.exists() or not pae_path.exists():
                continue
            safe_id = _safe_id(binder_id)
            pdb_target = adapter_dir / f"{safe_id}_unrelaxed_rank_001_esmfold2.pdb"
            json_target = adapter_dir / f"{safe_id}_scores_rank_001_esmfold2.json"
            tmp_pdb = adapter_dir / f"{safe_id}_unrelaxed_rank_001_esmfold2.raw.pdb"
            refolding_workflow._cif_to_pdb(cif_path, tmp_pdb)
            chain_map = _esmfold2_chain_role_map(pae_path)
            if chain_map:
                _write_pdb_with_chain_map(tmp_pdb, pdb_target, chain_map)
                tmp_pdb.unlink(missing_ok=True)
            else:
                tmp_pdb.replace(pdb_target)
            if not json_target.exists() and not _write_af2_adapter_json(pae_path, pdb_target, json_target):
                shutil.copy2(pae_path, json_target)
            count += 1
    return (adapter_dir if count else None), count


def _stage_child_outputs_for_common_metrics(
    child_runs: list[str],
    adapter_dir: Path,
    *,
    source_label: str,
) -> tuple[Path | None, int]:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for raw_child in child_runs:
        child_dir = Path(str(raw_child))
        candidates_path = child_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl"
        if not candidates_path.exists():
            continue
        for candidate in read_candidates(child_dir):
            parents = candidate.get("parents") or []
            binder_id = str(parents[0]) if parents else str(candidate.get("candidate_id") or "")
            raw_source = (candidate.get("raw_metadata") or {}).get("source_candidate") or {}
            raw_record = (candidate.get("raw_metadata") or {}).get("benchmark_record") or {}
            if isinstance(raw_source, dict) and raw_source.get("candidate_id"):
                binder_id = str(raw_source.get("candidate_id"))
            if isinstance(raw_record, dict) and raw_record.get("candidate_id"):
                binder_id = str(raw_record.get("candidate_id"))
            complex_rel = candidate.get("complex_pdb")
            pae_rel = (candidate.get("raw_metadata") or {}).get("pae_path")
            if not binder_id or not complex_rel or not pae_rel:
                continue
            complex_path = child_dir / str(complex_rel)
            pae_path = child_dir / str(pae_rel)
            if not complex_path.exists() or not pae_path.exists() or pae_path.suffix.lower() != ".json":
                continue
            safe_id = _safe_id(binder_id)
            pdb_target = adapter_dir / f"{safe_id}_unrelaxed_rank_001_{source_label}.pdb"
            json_target = adapter_dir / f"{safe_id}_scores_rank_001_{source_label}.json"
            tmp_pdb = adapter_dir / f"{safe_id}_unrelaxed_rank_001_{source_label}.raw.pdb"
            binder_chains = [str(chain) for chain in candidate.get("binder_chains") or [] if str(chain)]
            target_chains = [str(chain) for chain in candidate.get("target_chains") or [] if str(chain)]
            if not pdb_target.exists():
                if complex_path.suffix.lower() == ".pdb":
                    if not _normalize_structure_for_group_metrics(
                        complex_path,
                        pdb_target,
                        binder_chains=binder_chains,
                        target_chains=target_chains,
                    ):
                        _link_or_copy(complex_path, pdb_target)
                elif complex_path.suffix.lower() == ".cif" or complex_path.name.endswith(".cif.gz"):
                    refolding_workflow._cif_to_pdb(complex_path, tmp_pdb)
                    if _normalize_structure_for_group_metrics(
                        tmp_pdb,
                        pdb_target,
                        binder_chains=binder_chains,
                        target_chains=target_chains,
                    ):
                        tmp_pdb.unlink(missing_ok=True)
                    else:
                        tmp_pdb.replace(pdb_target)
                else:
                    continue
            if source_label == "rf3":
                _scale_fractional_pdb_confidence(pdb_target)
            if not json_target.exists() and not _write_af2_adapter_json(pae_path, pdb_target, json_target):
                shutil.copy2(pae_path, json_target)
            count += 1
    return (adapter_dir if count else None), count


def _merge_metric_tables(
    *,
    parent_run_dir: Path,
    run_csv: Path,
    metric_csvs: list[Path],
    output_name: str = "merged_benchmark_metrics.csv",
) -> Path | None:
    existing = []
    for path in metric_csvs:
        if not path.exists():
            continue
        try:
            columns = pd.read_csv(path, nrows=0).columns
        except Exception:
            continue
        if "binder_id" in columns:
            existing.append(path)
    if not existing:
        return None
    merged = pd.read_csv(run_csv)
    stale_engine_columns = [
        col
        for col in merged.columns
        if _benchmark_engine_for_column(str(col)) is not None
    ]
    if stale_engine_columns:
        merged = merged.drop(columns=stale_engine_columns)
    for csv_path in existing:
        df = pd.read_csv(csv_path)
        duplicate_cols = [col for col in df.columns if col != "binder_id" and col in merged.columns]
        if duplicate_cols:
            df = df.drop(columns=duplicate_cols)
        merged = merged.merge(df, on="binder_id", how="left")
    out = parent_run_dir / "artifacts" / "benchmark" / output_name
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    return out


def _write_metric_table_feature_ranking(
    *,
    run_csv: Path,
    metric_csv: Path,
) -> tuple[Path | None, Path | None]:
    if not metric_csv.exists():
        return None, None
    try:
        run_df = pd.read_csv(run_csv)
        metric_df = pd.read_csv(metric_csv)
    except Exception:
        return None, None
    if "binder_id" not in run_df.columns or "binder_id" not in metric_df.columns:
        return None, None
    label_cols = [col for col in ["binder", "label"] if col in run_df.columns]
    if not label_cols:
        return None, None
    label_col = label_cols[0]
    labels = run_df[["binder_id", label_col]].drop_duplicates("binder_id")
    duplicate_cols = [col for col in metric_df.columns if col != "binder_id" and col in labels.columns]
    if duplicate_cols:
        metric_df = metric_df.drop(columns=duplicate_cols)
    df = labels.merge(metric_df, on="binder_id", how="inner")
    if df.empty:
        return None, None
    feature_rows, summary = _metric_column_summary(df, label_col, max_columns=1000)
    stem = metric_csv.stem
    if stem.endswith("_metrics"):
        prefix = stem[: -len("_metrics")]
    else:
        prefix = stem
    out_dir = metric_csv.parent
    feature_table = out_dir / f"{prefix}_feature_benchmark.csv"
    summary_path = out_dir / f"{prefix}_feature_summary.json"
    _write_feature_ranking(feature_table, feature_rows)
    write_json(summary_path, summary)
    return feature_table, summary_path


def _run_benchmark_metric_postprocessing(
    *,
    job_run_dir: Path,
    run_csv: Path,
    output_dir: Path,
    metric_csvs: list[Path],
    esmfold2_child_runs: list[str],
    af2_child_runs: list[str],
    boltz2_child_runs: list[str],
    extra_child_runs: dict[str, list[str]],
    run_common_interface_metrics: bool,
    run_predicted_rosetta_metrics: bool,
    run_pymol_metrics: bool,
    pyrosetta_nprocs: int,
    docker_image: str,
    resume: bool = False,
) -> tuple[dict[str, Any], list[list[str]]]:
    metrics: dict[str, Any] = {}
    commands: list[list[str]] = []
    benchmark_dir = job_run_dir / "artifacts" / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    expected_rows = _csv_data_row_count(run_csv)

    if run_common_interface_metrics and run_csv.exists():
        ipsae_csv = benchmark_dir / "common_interface_metrics.csv"
        base_ipsae_cmd = [
            "docker",
            "run",
            "--rm",
            *_repo_and_runs_mounts(),
            "-w",
            str(DE_NOVO_BINDER_SCORING_DIR),
            docker_image,
            "python",
            "./scripts/run_ipsae_batch.py",
            "--run-csv",
            str(run_csv),
            "--ipsae-script-path",
            "./scripts/ipsae_w_ipae.py",
            "--max-workers",
            "1",
        ]
        af3_dir = output_dir / "AF3" / "alphafast_output"
        colab_dir = output_dir / "ColabFold" / "ptm_output"
        af2_adapter_dir, af2_adapter_count = _stage_af2_outputs_for_common_metrics(
            af2_child_runs,
            benchmark_dir / "af2_common_metric_inputs",
        )
        esmfold2_adapter_dir, esmfold2_adapter_count = _stage_esmfold2_outputs_for_common_metrics(
            esmfold2_child_runs,
            benchmark_dir / "esmfold2_common_metric_inputs",
        )
        extra_common_adapters: dict[str, tuple[Path | None, int]] = {}
        for engine_key in ("rf3", "openfold3", "protenix", "protenix_v1", "protenix_v2", "boltzgen_fold"):
            extra_common_adapters[engine_key] = _stage_child_outputs_for_common_metrics(
                extra_child_runs.get(engine_key, []),
                benchmark_dir / f"{engine_key}_common_metric_inputs",
                source_label=engine_key,
            )
        boltz2_dir: Path | None = None
        for raw_child in boltz2_child_runs:
            child_dir = Path(str(raw_child))
            candidate_dir = child_dir / "artifacts" / "raw" / "boltz2_initial_guess" / "output"
            if candidate_dir.exists():
                boltz2_dir = candidate_dir
                break
        sources = []
        ipsae_cmd = [*base_ipsae_cmd, "--out-csv", str(ipsae_csv)]
        if af3_dir.exists():
            ipsae_cmd.extend(["--af3-dir", str(af3_dir)])
            sources.append("af3")
        if colab_dir.exists():
            ipsae_cmd.extend(["--colab-dir", str(colab_dir)])
            sources.append("colab")
        if boltz2_dir is not None:
            ipsae_cmd.extend(["--boltz-dir", str(boltz2_dir)])
            sources.append("boltz")
        if sources:
            if resume and _csv_data_row_count(ipsae_csv) == expected_rows:
                metric_csvs.append(ipsae_csv)
                metrics["common_interface_metrics_table"] = str(ipsae_csv.relative_to(job_run_dir))
                metrics["common_interface_metrics_resumed_from_artifacts"] = True
            else:
                commands.append(ipsae_cmd)
                rc = _run_docker_command(job_run_dir, ipsae_cmd)
                metrics["common_interface_metrics_return_code"] = rc
                if rc == 0 and ipsae_csv.exists():
                    _rename_metric_prefix(ipsae_csv, "boltz1_", "boltz2_")
                    metric_csvs.append(ipsae_csv)
                    metrics["common_interface_metrics_table"] = str(ipsae_csv.relative_to(job_run_dir))
        else:
            metrics["common_interface_metrics_skipped"] = "no supported AF3, ColabFold, or Boltz2 output folders found"
        if af2_adapter_dir is not None:
            af2_ipsae_csv = benchmark_dir / "af2_common_interface_metrics.csv"
            af2_ipsae_cmd = [
                *base_ipsae_cmd,
                "--out-csv",
                str(af2_ipsae_csv),
                "--colab-dir",
                str(af2_adapter_dir),
            ]
            commands.append(af2_ipsae_cmd)
            rc = _run_docker_command(job_run_dir, af2_ipsae_cmd)
            metrics["af2_common_interface_metrics_return_code"] = rc
            metrics["af2_common_metric_adapter_count"] = af2_adapter_count
            if rc == 0 and af2_ipsae_csv.exists():
                _rename_metric_prefix(af2_ipsae_csv, "colab_", "af2_")
                metric_csvs.append(af2_ipsae_csv)
                metrics["af2_common_interface_metrics_table"] = str(af2_ipsae_csv.relative_to(job_run_dir))
        if esmfold2_adapter_dir is not None:
            esmfold2_ipsae_csv = benchmark_dir / "esmfold2_common_interface_metrics.csv"
            esmfold2_ipsae_cmd = [
                *base_ipsae_cmd,
                "--out-csv",
                str(esmfold2_ipsae_csv),
                "--colab-dir",
                str(esmfold2_adapter_dir),
            ]
            commands.append(esmfold2_ipsae_cmd)
            rc = _run_docker_command(job_run_dir, esmfold2_ipsae_cmd)
            metrics["esmfold2_common_interface_metrics_return_code"] = rc
            metrics["esmfold2_common_metric_adapter_count"] = esmfold2_adapter_count
            if rc == 0 and esmfold2_ipsae_csv.exists():
                _rename_metric_prefix(esmfold2_ipsae_csv, "colab_", "esmfold2_")
                metric_csvs.append(esmfold2_ipsae_csv)
                metrics["esmfold2_common_interface_metrics_table"] = str(esmfold2_ipsae_csv.relative_to(job_run_dir))
        for engine_key, (adapter_dir, adapter_count) in extra_common_adapters.items():
            if adapter_dir is None:
                metrics[f"{engine_key}_common_interface_metrics_skipped"] = "no full PAE matrix output found"
                metrics[f"{engine_key}_common_metric_adapter_count"] = 0
                continue
            engine_ipsae_csv = benchmark_dir / f"{engine_key}_common_interface_metrics.csv"
            engine_ipsae_cmd = [
                *base_ipsae_cmd,
                "--out-csv",
                str(engine_ipsae_csv),
                "--colab-dir",
                str(adapter_dir),
            ]
            commands.append(engine_ipsae_cmd)
            rc = _run_docker_command(job_run_dir, engine_ipsae_cmd)
            metrics[f"{engine_key}_common_interface_metrics_return_code"] = rc
            metrics[f"{engine_key}_common_metric_adapter_count"] = adapter_count
            if rc == 0 and engine_ipsae_csv.exists():
                _rename_metric_prefix(engine_ipsae_csv, "colab_", f"{engine_key}_")
                metric_csvs.append(engine_ipsae_csv)
                metrics[f"{engine_key}_common_interface_metrics_table"] = str(engine_ipsae_csv.relative_to(job_run_dir))
    predicted_pdb_dirs: list[tuple[str, Path]] = []
    group_roles = _benchmark_group_roles(run_csv)
    unique_group_roles = {
        key: value
        for key, value in group_roles.items()
        if key == _safe_id(key)
    }
    metrics["structure_metric_chain_contract"] = {
        "binder_group": "normalized to chain A",
        "target_group": "all declared target chains normalized together to chain B",
        "scope": "Rosetta and PyMOL structure-based evaluation only",
    }
    metrics["multi_chain_target_record_count"] = sum(
        1 for _binder_chains, target_chains in unique_group_roles.values() if len(target_chains) > 1
    )
    viewer_structure_counts: dict[str, int] = {}
    af3_pdb_dir, af3_pdb_count = _stage_alphafast_pdbs(
        output_dir / "AF3" / "alphafast_output",
        benchmark_dir / "predicted_metric_pdbs" / "af3",
        group_roles,
    )
    if af3_pdb_dir is not None:
        predicted_pdb_dirs.append(("af3", af3_pdb_dir))
        metrics["af3_predicted_metric_pdb_count"] = af3_pdb_count
    af3_viewer_dir, af3_viewer_count = _stage_alphafast_viewer_structures(
        output_dir / "AF3" / "alphafast_output",
        benchmark_dir / "predicted_viewer_structures" / "af3",
    )
    if af3_viewer_dir is not None:
        viewer_structure_counts["af3"] = af3_viewer_count
    colab_pdb_dir, colab_pdb_count = _stage_colabfold_pdbs(
        output_dir,
        benchmark_dir / "predicted_metric_pdbs" / "colab",
        group_roles,
    )
    if colab_pdb_dir is not None:
        predicted_pdb_dirs.append(("colab", colab_pdb_dir))
        metrics["colab_predicted_metric_pdb_count"] = colab_pdb_count
    colab_viewer_dir, colab_viewer_count = _stage_colabfold_viewer_structures(
        output_dir,
        benchmark_dir / "predicted_viewer_structures" / "colab",
    )
    if colab_viewer_dir is not None:
        viewer_structure_counts["colab"] = colab_viewer_count
    af2_pdb_dir, af2_pdb_count = _stage_child_candidate_pdbs(
        af2_child_runs,
        benchmark_dir / "predicted_metric_pdbs" / "af2",
        source_label="af2",
    )
    if af2_pdb_dir is not None:
        predicted_pdb_dirs.append(("af2", af2_pdb_dir))
        metrics["af2_predicted_metric_pdb_count"] = af2_pdb_count
    af2_viewer_dir, af2_viewer_count = _stage_child_viewer_structures(
        af2_child_runs,
        benchmark_dir / "predicted_viewer_structures" / "af2",
        source_label="af2",
    )
    if af2_viewer_dir is not None:
        viewer_structure_counts["af2"] = af2_viewer_count
    boltz2_pdb_dir, boltz2_pdb_count = _stage_child_candidate_pdbs(
        boltz2_child_runs,
        benchmark_dir / "predicted_metric_pdbs" / "boltz2",
        source_label="boltz2",
    )
    if boltz2_pdb_dir is not None:
        predicted_pdb_dirs.append(("boltz2", boltz2_pdb_dir))
        metrics["boltz2_predicted_metric_pdb_count"] = boltz2_pdb_count
    boltz2_viewer_dir, boltz2_viewer_count = _stage_child_viewer_structures(
        boltz2_child_runs,
        benchmark_dir / "predicted_viewer_structures" / "boltz2",
        source_label="boltz2",
    )
    if boltz2_viewer_dir is not None:
        viewer_structure_counts["boltz2"] = boltz2_viewer_count
    esmfold2_pdb_dir, esmfold2_pdb_count = _stage_child_candidate_pdbs(
        esmfold2_child_runs,
        benchmark_dir / "predicted_metric_pdbs" / "esmfold2",
        source_label="esmfold2",
    )
    if esmfold2_pdb_dir is not None:
        predicted_pdb_dirs.append(("esmfold2", esmfold2_pdb_dir))
        metrics["esmfold2_predicted_metric_pdb_count"] = esmfold2_pdb_count
    esmfold2_viewer_dir, esmfold2_viewer_count = _stage_child_viewer_structures(
        esmfold2_child_runs,
        benchmark_dir / "predicted_viewer_structures" / "esmfold2",
        source_label="esmfold2",
    )
    if esmfold2_viewer_dir is not None:
        viewer_structure_counts["esmfold2"] = esmfold2_viewer_count
    for engine, child_runs in sorted((extra_child_runs or {}).items()):
        engine_pdb_dir, engine_pdb_count = _stage_child_candidate_pdbs(
            child_runs,
            benchmark_dir / "predicted_metric_pdbs" / engine,
            source_label=engine,
        )
        if engine_pdb_dir is not None:
            predicted_pdb_dirs.append((engine, engine_pdb_dir))
            metrics[f"{engine}_predicted_metric_pdb_count"] = engine_pdb_count
        engine_viewer_dir, engine_viewer_count = _stage_child_viewer_structures(
            child_runs,
            benchmark_dir / "predicted_viewer_structures" / engine,
            source_label=engine,
        )
        if engine_viewer_dir is not None:
            viewer_structure_counts[engine] = engine_viewer_count
    if viewer_structure_counts:
        metrics["predicted_viewer_structure_counts"] = viewer_structure_counts

    pose_rmsd_output = benchmark_dir / "target_aligned_binder_rmsd_metrics.csv"
    if resume and _csv_data_row_count(pose_rmsd_output) == expected_rows:
        pose_rmsd_csv = pose_rmsd_output
        pose_rmsd_summary = {"target_aligned_binder_rmsd_resumed_from_artifacts": True}
    else:
        pose_rmsd_csv, pose_rmsd_summary = _write_target_aligned_binder_rmsd_table(
            job_run_dir=job_run_dir,
            run_csv=run_csv,
            predicted_pdb_dirs=predicted_pdb_dirs,
            output_path=pose_rmsd_output,
        )
    metrics.update(pose_rmsd_summary)
    if pose_rmsd_csv is not None:
        metric_csvs.append(pose_rmsd_csv)
        metrics["target_aligned_binder_rmsd_metrics_table"] = str(pose_rmsd_csv.relative_to(job_run_dir))

    if run_predicted_rosetta_metrics and run_csv.exists():
        rosetta_csv = benchmark_dir / "predicted_rosetta_metrics.csv"
        if resume and _csv_data_row_count(rosetta_csv) == expected_rows:
            metric_csvs.append(rosetta_csv)
            metrics["predicted_rosetta_metrics_table"] = str(rosetta_csv.relative_to(job_run_dir))
            metrics["predicted_rosetta_metrics_resumed_from_artifacts"] = True
        elif predicted_pdb_dirs:
            rosetta_cmd = [
                "docker",
                "run",
                "--rm",
                *_repo_and_runs_mounts(),
                "-w",
                str(DE_NOVO_BINDER_SCORING_DIR),
                PYROSETTA_METRICS_IMAGE,
                "python",
                "./scripts/compute_rosetta_metrics.py",
                "--run-csv",
                str(run_csv),
                "--out-csv",
                str(rosetta_csv),
                "--nprocs",
                str(max(1, int(pyrosetta_nprocs))),
                "--dalphaball-path",
                "./functions/DAlphaBall.gcc",
            ]
            for prefix, folder in predicted_pdb_dirs:
                rosetta_cmd.extend(["--folder", f"{prefix}:{folder}"])
            commands.append(rosetta_cmd)
            rc = _run_docker_command(job_run_dir, rosetta_cmd)
            metrics["predicted_rosetta_metrics_return_code"] = rc
            if rc == 0 and rosetta_csv.exists():
                metric_csvs.append(rosetta_csv)
                metrics["predicted_rosetta_metrics_table"] = str(rosetta_csv.relative_to(job_run_dir))
        else:
            metrics["predicted_rosetta_metrics_skipped"] = "no predicted PDB folders found"

    if run_pymol_metrics:
        pymol_dirs: dict[str, str] = {}
        input_pdb_dir = output_dir / "input_pdbs"
        staged_input_pdb_dir, staged_input_pdb_count = _stage_input_pdbs_for_group_metrics(
            input_pdb_dir,
            benchmark_dir / "predicted_metric_pdbs" / "input",
            group_roles,
        )
        if staged_input_pdb_dir is not None:
            pymol_dirs["input"] = str(staged_input_pdb_dir)
            metrics["input_predicted_metric_pdb_count"] = staged_input_pdb_count
        for prefix, folder in predicted_pdb_dirs:
            pymol_dirs[prefix] = str(folder)
        if pymol_dirs:
            pymol_work_dir = benchmark_dir
            pymol_files_dir = pymol_work_dir / "pymol_files"
            pymol_files_dir.mkdir(parents=True, exist_ok=True)
            write_json(pymol_files_dir / "pdb_dirs.json", pymol_dirs)
            pymol_cmd = [
                str(PYMOL_PYTHON),
                "-m",
                "pymol",
                "-c",
                "-d",
                f"run {DE_NOVO_BINDER_SCORING_DIR / 'scripts' / 'pymol_metrics.py'}",
            ]
            commands.append(pymol_cmd)
            rc = _run_local_command(job_run_dir, pymol_cmd, cwd=pymol_work_dir)
            metrics["pymol_metrics_return_code"] = rc
            if rc == 0:
                pymol_tables = sorted(pymol_files_dir.glob("pymol_metrics_*.csv"))
                for pymol_table in pymol_tables:
                    metric_csvs.append(pymol_table)
                metrics["pymol_metrics_tables"] = [
                    str(path.relative_to(job_run_dir)) for path in pymol_tables
                ]
        else:
            metrics["pymol_metrics_skipped"] = "no input or predicted PDB folders found"

    for metric_csv in list(metric_csvs):
        feature_table, summary_path = _write_metric_table_feature_ranking(
            run_csv=run_csv,
            metric_csv=metric_csv,
        )
        if feature_table is not None:
            metrics[f"{metric_csv.stem}_feature_ranking"] = str(feature_table.relative_to(job_run_dir))
        if summary_path is not None:
            metrics[f"{metric_csv.stem}_feature_summary"] = str(summary_path.relative_to(job_run_dir))

    merged = _merge_metric_tables(parent_run_dir=job_run_dir, run_csv=run_csv, metric_csvs=metric_csvs)
    if merged is not None:
        metrics["merged_benchmark_metrics"] = str(merged.relative_to(job_run_dir))
        try:
            merged_df = pd.read_csv(merged)
            feature_rows, summary = _metric_column_summary(merged_df, "binder", max_columns=1000)
            summary_path = benchmark_dir / "merged_benchmark_feature_summary.json"
            feature_table = benchmark_dir / "merged_benchmark_feature_ranking.csv"
            write_json(summary_path, summary)
            _write_feature_ranking(feature_table, feature_rows)
            metrics["merged_benchmark_feature_summary"] = str(summary_path.relative_to(job_run_dir))
            metrics["merged_benchmark_feature_ranking"] = str(feature_table.relative_to(job_run_dir))
            metrics["merged_benchmark_top_feature"] = feature_rows[0]["feature"] if feature_rows else None
            metrics["merged_benchmark_top_feature_average_precision"] = (
                feature_rows[0]["best_average_precision"] if feature_rows else None
            )
            metrics["merged_benchmark_top_feature_auroc"] = feature_rows[0]["best_auroc"] if feature_rows else None
        except Exception as exc:
            metrics["merged_benchmark_feature_error"] = str(exc)
    return metrics, commands


def _run_alphafast_af3_refolding(
    *,
    job_run_dir: Path,
    run_csv: Path | None,
    af3_input_dir: Path,
    output_dir: Path,
    db_dir: Path,
    weights_dir: Path,
    batch_size: int = 0,
    num_recycles: int = 10,
    gpu_device: object = 0,
    max_records: int = 0,
    image: str = ALPHAFAST_IMAGE,
    run_data_pipeline: bool = True,
    use_target_templates: bool = False,
    query_only_msa: bool = False,
) -> tuple[int, list[list[str]], dict[str, Any]]:
    commands: list[list[str]] = []
    template_metrics: dict[str, Any] = {}
    if run_data_pipeline:
        rc, commands, template_metrics = _run_alphafast_data_pipeline(
            job_run_dir=job_run_dir,
            run_csv=run_csv,
            af3_input_dir=af3_input_dir,
            output_dir=output_dir,
            db_dir=db_dir,
            batch_size=batch_size,
            gpu_device=gpu_device,
            max_records=max_records,
            image=image,
            use_target_templates=use_target_templates,
            query_only_msa=query_only_msa,
        )
        if rc != 0:
            return rc, commands, template_metrics
    if not weights_dir.exists():
        raise ValueError(f"AlphaFast weights directory does not exist: {weights_dir}")
    if not any(weights_dir.glob("af3*.bin.zst")):
        raise ValueError(f"AlphaFast weights directory must contain the AF3 weights file, e.g. af3.bin.zst: {weights_dir}")

    common_mounts = _alphafast_common_mounts(db_dir=db_dir, weights_dir=weights_dir, gpu_device=gpu_device, image=image)
    inference_cmd = [
        *common_mounts,
        "python",
        "/app/alphafold/run_alphafold.py",
        f"--input_dir={output_dir}",
        "--model_dir=/data/models",
        "--norun_data_pipeline",
        f"--output_dir={output_dir}",
        "--force_output_dir",
        f"--num_recycles={int(num_recycles)}",
    ]
    commands.append(inference_cmd)
    rc = _run_docker_command(job_run_dir, inference_cmd)
    return rc, commands, template_metrics


def _alphafast_common_mounts(*, db_dir: Path, weights_dir: Path | None, gpu_device: object, image: str) -> list[str]:
    mounts = [
        "docker",
        "run",
        "--rm",
        *docker_gpu_args(gpu_device),
        *_repo_and_runs_mounts(),
        "-v",
        f"{db_dir}:/data/public_databases:ro",
        "-v",
        f"{db_dir / 'mmseqs'}:/data/mmseqs_databases:ro",
    ]
    if weights_dir is not None:
        mounts.extend(["-v", f"{weights_dir}:/data/models:ro"])
    mounts.extend(["-w", "/app/alphafold", image])
    return mounts


def _a3m_non_insert_length(sequence: str) -> int:
    return sum(1 for char in str(sequence or "") if char.strip() and not char.islower())


def _parse_a3m_records(text: str) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence_lines: list[str] = []
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(sequence_lines)))
            header = line
            sequence_lines = []
        else:
            sequence_lines.append(line)
    if header is not None:
        records.append((header, "".join(sequence_lines)))
    return records


def _sanitize_af3_a3m(msa_text: str, *, target_sequence: str) -> tuple[str, dict[str, int]]:
    """AF3 rejects A3M rows whose non-insert length differs from the target."""
    target_sequence = str(target_sequence or "").strip()
    target_len = _a3m_non_insert_length(target_sequence)
    records = _parse_a3m_records(msa_text)
    if target_len <= 0:
        return msa_text, {"sanitized": 0, "invalid_rows": 0, "query_fallback": 0}
    valid_records: list[tuple[str, str]] = []
    invalid_rows = 0
    for header, sequence in records:
        if _a3m_non_insert_length(sequence) == target_len:
            valid_records.append((header, sequence))
        else:
            invalid_rows += 1
    query_fallback = 0
    if not valid_records:
        valid_records = [(">query", target_sequence)]
        query_fallback = 1
    elif _a3m_non_insert_length(valid_records[0][1]) != target_len:
        valid_records.insert(0, (">query", target_sequence))
    sanitized = int(invalid_rows > 0 or query_fallback > 0)
    sanitized_text = "\n".join(f"{header}\n{sequence}" for header, sequence in valid_records) + "\n"
    return sanitized_text, {
        "sanitized": sanitized,
        "invalid_rows": invalid_rows,
        "query_fallback": query_fallback,
    }


def _stage_alphafast_inputs(
    af3_input_dir: Path,
    output_dir: Path,
    max_records: int,
    *,
    run_csv: Path | None = None,
    use_target_templates: bool = False,
    query_only_msa: bool = False,
) -> tuple[Path, int, dict[str, Any]]:
    if not af3_input_dir.exists() or not any(af3_input_dir.glob("*.json")):
        raise ValueError("AlphaFast AF3 benchmark needs generated AF3/input_folder/*.json inputs.")
    selected_inputs = sorted(af3_input_dir.glob("*.json"))
    if max_records > 0:
        selected_inputs = selected_inputs[: int(max_records)]
    staged_input_dir = output_dir.parent / "alphafast_input"
    staged_input_dir.mkdir(parents=True, exist_ok=True)
    for existing in staged_input_dir.glob("*.json"):
        existing.unlink()
    template_dir = staged_input_dir / "target_templates"
    if template_dir.exists():
        shutil.rmtree(template_dir)
    template_rows = _run_csv_rows_by_binder_id(run_csv, max_records=max_records) if use_target_templates and run_csv else {}
    input_pdb_dir = run_csv.parent / "input_pdbs" if use_target_templates and run_csv else Path()
    template_metrics: dict[str, Any] = {
        "alphafast_use_target_templates": bool(use_target_templates),
        "alphafast_target_template_input_count": 0,
        "alphafast_target_template_protein_count": 0,
        "alphafast_target_template_templated_protein_count": 0,
        "alphafast_target_template_count": 0,
        "alphafast_target_template_missing_source_pdb_count": 0,
        "alphafast_target_template_missing_target_chain_count": 0,
        "alphafast_target_template_build_failed_count": 0,
        "alphafast_msa_sanitized_count": 0,
        "alphafast_msa_invalid_row_count": 0,
        "alphafast_msa_query_fallback_count": 0,
    }
    for source in selected_inputs:
        target = staged_input_dir / source.name
        payload = json.loads(source.read_text())
        if use_target_templates and template_rows and input_pdb_dir.exists():
            row = template_rows.get(source.stem) or template_rows.get(_safe_id(source.stem))
            staged = _stage_af3_target_templates(
                payload=payload,
                source_name=source.stem,
                row=row,
                input_pdb_dir=input_pdb_dir,
                template_dir=template_dir,
            )
            template_metrics["alphafast_target_template_input_count"] += 1
            template_metrics["alphafast_target_template_protein_count"] += staged["protein_count"]
            template_metrics["alphafast_target_template_templated_protein_count"] += staged["templated_protein_count"]
            template_metrics["alphafast_target_template_count"] += staged["template_count"]
            template_metrics["alphafast_target_template_missing_source_pdb_count"] += staged["missing_source_pdb_count"]
            template_metrics["alphafast_target_template_missing_target_chain_count"] += staged["missing_target_chain_count"]
            template_metrics["alphafast_target_template_build_failed_count"] += staged["template_build_failed_count"]
            expected_target_templates = len(_split_list(_row_value(row or {}, "target_chains", "target_chain")))
            if staged["templated_protein_count"] < expected_target_templates:
                raise ValueError(
                    f"AlphaFast AF3 target-template staging for {source.stem} produced "
                    f"{staged['templated_protein_count']} of {expected_target_templates} declared target templates."
                )
        for sequence in payload.get("sequences", []):
            protein = sequence.get("protein") if isinstance(sequence, dict) else None
            if not isinstance(protein, dict):
                continue
            msa_path = protein.get("unpairedMsaPath")
            if msa_path:
                host_msa_path = Path(str(msa_path))
                if host_msa_path.exists():
                    sanitized_msa, msa_stats = _sanitize_af3_a3m(
                        host_msa_path.read_text(errors="replace"),
                        target_sequence=str(protein.get("sequence") or ""),
                    )
                    protein["unpairedMsa"] = sanitized_msa
                    protein.pop("unpairedMsaPath", None)
                    template_metrics["alphafast_msa_sanitized_count"] += int(msa_stats["sanitized"])
                    template_metrics["alphafast_msa_invalid_row_count"] += int(msa_stats["invalid_rows"])
                    template_metrics["alphafast_msa_query_fallback_count"] += int(msa_stats["query_fallback"])
                else:
                    protein.pop("unpairedMsaPath", None)
                    if query_only_msa:
                        protein["unpairedMsa"] = ""
                    else:
                        protein.pop("unpairedMsa", None)
                    protein.pop("pairedMsaPath", None)
                    if query_only_msa:
                        protein["pairedMsa"] = ""
                    else:
                        protein.pop("pairedMsa", None)
                    if query_only_msa:
                        protein["templates"] = protein.get("templates") or []
                    elif not (use_target_templates and protein.get("templates")):
                        protein.pop("templates", None)
            elif query_only_msa:
                protein["unpairedMsa"] = ""
                protein["pairedMsa"] = ""
                if not (use_target_templates and protein.get("templates")):
                    protein["templates"] = []
            paired_path = protein.get("pairedMsaPath")
            if paired_path:
                host_paired_path = Path(str(paired_path))
                if host_paired_path.exists():
                    protein["pairedMsa"] = host_paired_path.read_text(errors="replace")
                    protein.pop("pairedMsaPath", None)
                else:
                    protein.pop("pairedMsaPath", None)
                    protein.pop("pairedMsa", None)
                    if query_only_msa:
                        protein["templates"] = protein.get("templates") or []
                    elif not (use_target_templates and protein.get("templates")):
                        protein.pop("templates", None)
        target.write_text(json.dumps(payload, indent=2))
    return staged_input_dir, len(selected_inputs), template_metrics


def _run_alphafast_data_pipeline(
    *,
    job_run_dir: Path,
    run_csv: Path | None = None,
    af3_input_dir: Path,
    output_dir: Path,
    db_dir: Path,
    batch_size: int = 0,
    gpu_device: object = 0,
    max_records: int = 0,
    image: str = ALPHAFAST_IMAGE,
    use_target_templates: bool = False,
    query_only_msa: bool = False,
) -> tuple[int, list[list[str]], dict[str, Any]]:
    if not db_dir.exists():
        raise ValueError(f"AlphaFast database directory does not exist: {db_dir}")
    if not (db_dir / "mmseqs").exists():
        raise ValueError(f"AlphaFast database directory must contain an mmseqs subdirectory: {db_dir / 'mmseqs'}")
    staged_input_dir, input_count, template_metrics = _stage_alphafast_inputs(
        af3_input_dir,
        output_dir,
        max_records,
        run_csv=run_csv,
        use_target_templates=use_target_templates,
        query_only_msa=query_only_msa,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    if query_only_msa:
        for source in staged_input_dir.glob("*.json"):
            shutil.copy2(source, output_dir / source.name)
        return 0, [], template_metrics
    effective_batch = int(batch_size) if batch_size and int(batch_size) > 0 else max(1, input_count)
    common_mounts = _alphafast_common_mounts(db_dir=db_dir, weights_dir=None, gpu_device=gpu_device, image=image)
    pipeline_cmd = [
        *common_mounts,
        "python",
        "/app/alphafold/run_data_pipeline.py",
        f"--input_dir={staged_input_dir}",
        f"--output_dir={output_dir}",
        "--db_dir=/data/public_databases",
        "--mmseqs_db_dir=/data/mmseqs_databases",
        "--use_mmseqs_gpu",
        f"--batch_size={effective_batch}",
    ]
    rc = _run_docker_command(job_run_dir, pipeline_cmd)
    return rc, [pipeline_cmd], template_metrics


def _write_alphafast_msas_to_run_csv(
    *,
    run_csv: Path,
    alphafast_output_dir: Path,
    output_dir: Path,
    msa_repository_dir: Path = MSA_REPOSITORY_DIR,
) -> dict[str, Any]:
    df = pd.read_csv(run_csv)
    msa_dir = output_dir / "unique_msa" / "msa"
    msa_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    repository_written = 0
    repository_existing = 0
    repository_invalid = 0
    data_files = {path.parent.name: path for path in alphafast_output_dir.glob("*/*_data.json")}
    for index, row in df.iterrows():
        binder_id = str(row.get("binder_id") or "").strip()
        data_path = data_files.get(binder_id)
        if not data_path:
            continue
        payload = json.loads(data_path.read_text())
        binder_chain = str(row.get("binder_chain") or "A").strip() or "A"
        target_only = _run_csv_is_capacity_target_only(row)
        for sequence in payload.get("sequences", []):
            protein = sequence.get("protein") if isinstance(sequence, dict) else None
            if not isinstance(protein, dict):
                continue
            chain_id = str(protein.get("id") or "").strip()
            if not chain_id or (chain_id == binder_chain and not target_only):
                continue
            current_msa = str(row.get(f"msa_path_{chain_id}") or "").strip()
            if (
                current_msa
                and current_msa.lower() != "no_msa"
                and Path(current_msa).exists()
                and _a3m_sequence_count(Path(current_msa)) > 1
            ):
                continue
            msa_text = str(protein.get("unpairedMsa") or "").strip()
            if not msa_text:
                continue
            msa_path = msa_dir / f"{_safe_id(binder_id)}_chain_{_safe_id(chain_id)}_alphafast.a3m"
            msa_path.write_text(msa_text + "\n")
            raw_sequence = row.get(f"target_subchain_{chain_id}_seq")
            if raw_sequence is None or pd.isna(raw_sequence) or not str(raw_sequence).strip():
                raw_sequence = row.get(f"{chain_id}_seq")
            target_sequence = "" if raw_sequence is None or pd.isna(raw_sequence) else str(raw_sequence).strip()
            if target_sequence:
                repository_path, _container_path = target_msa_workflow.boltz_msa_paths(
                    target_sequence,
                    msa_repository_dir=Path(msa_repository_dir),
                )
                repository_valid, _reason = target_msa_workflow.validate_a3m_file(repository_path)
                if repository_valid:
                    repository_existing += 1
                else:
                    repository_path.parent.mkdir(parents=True, exist_ok=True)
                    repository_path.write_text(msa_text + "\n")
                    repository_valid, _reason = target_msa_workflow.validate_a3m_file(repository_path)
                    if repository_valid:
                        repository_written += 1
                    else:
                        repository_invalid += 1
                        repository_path.unlink(missing_ok=True)
            df.loc[index, f"msa_path_{chain_id}"] = str(msa_path)
            written += 1
    df.to_csv(run_csv, index=False)
    return {
        "alphafast_msa_source": str(alphafast_output_dir),
        "alphafast_msa_written_count": written,
        "alphafast_msa_dir": str(msa_dir),
        "alphafast_msa_repository_dir": str(msa_repository_dir),
        "alphafast_msa_repository_written_count": repository_written,
        "alphafast_msa_repository_existing_count": repository_existing,
        "alphafast_msa_repository_invalid_count": repository_invalid,
    }


def _run_alphafast_serial_chain_msa_pipeline(
    *,
    job_run_dir: Path,
    run_csv: Path,
    af3_input_dir: Path,
    output_dir: Path,
    db_dir: Path,
    batch_size: int,
    gpu_device: object,
    max_records: int,
    image: str,
    msa_repository_dir: Path = MSA_REPOSITORY_DIR,
) -> tuple[dict[str, Any], list[list[str]]]:
    df = pd.read_csv(run_csv)
    limit = int(max_records) if max_records and int(max_records) > 0 else len(df)
    wanted: dict[str, set[str]] = {}
    for _, row in df.head(limit).iterrows():
        binder_id = str(row.get("binder_id") or "").strip()
        if not binder_id:
            continue
        binder_chain = str(row.get("binder_chain") or "A").strip() or "A"
        for chain in _run_csv_target_msa_chains(row, binder_chain):
            msa_value = str(row.get(f"msa_path_{chain}") or "").strip()
            if (
                msa_value
                and msa_value.lower() != "no_msa"
                and Path(msa_value).exists()
                and _a3m_sequence_count(Path(msa_value)) > 1
            ):
                continue
            wanted.setdefault(binder_id, set()).add(chain)
    if not wanted:
        return {"alphafast_serial_msa_needed_count": 0}, []

    serial_root = output_dir / "AF3" / "serial_chain_msa"
    serial_root.mkdir(parents=True, exist_ok=True)
    msa_dir = output_dir / "unique_msa" / "msa"
    msa_dir.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []
    attempted = 0
    succeeded = 0
    failed = 0
    written = 0
    repository_written = 0
    repository_existing = 0
    repository_invalid = 0
    failure_rows: list[dict[str, Any]] = []

    input_files = sorted(af3_input_dir.glob("*.json"))
    if max_records and int(max_records) > 0:
        input_files = input_files[: int(max_records)]
    for source_json in input_files:
        payload = json.loads(source_json.read_text())
        binder_id = str(payload.get("name") or source_json.stem)
        wanted_chains = wanted.get(binder_id) or wanted.get(source_json.stem) or set()
        if not wanted_chains:
            continue
        for sequence_entry in payload.get("sequences", []):
            protein = sequence_entry.get("protein") if isinstance(sequence_entry, dict) else None
            if not isinstance(protein, dict):
                continue
            chain_id = str(protein.get("id") or "").strip()
            if chain_id not in wanted_chains:
                continue
            attempted += 1
            serial_name = f"{binder_id}_chain_{_safe_id(chain_id)}"
            serial_input_dir = serial_root / serial_name / "input"
            serial_output_dir = serial_root / serial_name / "output"
            if serial_input_dir.exists():
                shutil.rmtree(serial_input_dir)
            if serial_output_dir.exists():
                shutil.rmtree(serial_output_dir)
            serial_input_dir.mkdir(parents=True, exist_ok=True)
            serial_payload = copy.deepcopy(payload)
            serial_payload["name"] = serial_name
            serial_payload["sequences"] = [copy.deepcopy(sequence_entry)]
            (serial_input_dir / f"{serial_name}.json").write_text(json.dumps(serial_payload, indent=2))
            rc, serial_commands, _serial_template_metrics = _run_alphafast_data_pipeline(
                job_run_dir=job_run_dir,
                af3_input_dir=serial_input_dir,
                output_dir=serial_output_dir,
                db_dir=db_dir,
                batch_size=max(1, int(batch_size) if batch_size and int(batch_size) > 0 else 1),
                gpu_device=gpu_device,
                max_records=1,
                image=image,
            )
            commands.extend(serial_commands)
            if rc != 0:
                failed += 1
                failure_rows.append({"binder_id": binder_id, "chain": chain_id, "return_code": rc})
                continue
            data_files = sorted(serial_output_dir.glob("*/*_data.json"))
            if not data_files:
                failed += 1
                failure_rows.append({"binder_id": binder_id, "chain": chain_id, "reason": "missing_data_json"})
                continue
            data_payload = json.loads(data_files[0].read_text())
            msa_text = ""
            for serial_sequence in data_payload.get("sequences", []):
                serial_protein = serial_sequence.get("protein") if isinstance(serial_sequence, dict) else None
                if isinstance(serial_protein, dict):
                    msa_text = str(serial_protein.get("unpairedMsa") or "").strip()
                    if msa_text:
                        break
            if not msa_text or len(msa_text.splitlines()) <= 1:
                failed += 1
                failure_rows.append({"binder_id": binder_id, "chain": chain_id, "reason": "nonreal_or_empty_msa"})
                continue
            msa_path = msa_dir / f"{_safe_id(binder_id)}_chain_{_safe_id(chain_id)}_alphafast_serial.a3m"
            msa_path.write_text(msa_text + "\n")
            row_mask = df["binder_id"].astype(str) == binder_id
            df.loc[row_mask, f"msa_path_{chain_id}"] = str(msa_path)
            raw_sequence = ""
            matching_rows = df[row_mask]
            if not matching_rows.empty:
                value = matching_rows.iloc[0].get(f"target_subchain_{chain_id}_seq")
                if value is None or pd.isna(value) or not str(value).strip():
                    value = matching_rows.iloc[0].get(f"{chain_id}_seq")
                raw_sequence = "" if value is None or pd.isna(value) else str(value).strip()
            if raw_sequence:
                repository_path, _container_path = target_msa_workflow.boltz_msa_paths(
                    raw_sequence,
                    msa_repository_dir=Path(msa_repository_dir),
                )
                repository_valid, _reason = target_msa_workflow.validate_a3m_file(repository_path)
                if repository_valid:
                    repository_existing += 1
                else:
                    repository_path.parent.mkdir(parents=True, exist_ok=True)
                    repository_path.write_text(msa_text + "\n")
                    repository_valid, _reason = target_msa_workflow.validate_a3m_file(repository_path)
                    if repository_valid:
                        repository_written += 1
                    else:
                        repository_invalid += 1
                        repository_path.unlink(missing_ok=True)
            written += 1
            succeeded += 1
    df.to_csv(run_csv, index=False)
    write_json(serial_root / "serial_msa_summary.json", {"failures": failure_rows})
    return (
        {
            "alphafast_serial_msa_needed_count": sum(len(chains) for chains in wanted.values()),
            "alphafast_serial_msa_attempted_count": attempted,
            "alphafast_serial_msa_success_count": succeeded,
            "alphafast_serial_msa_failed_count": failed,
            "alphafast_serial_msa_written_count": written,
            "alphafast_serial_msa_dir": str(serial_root),
            "alphafast_serial_msa_repository_written_count": repository_written,
            "alphafast_serial_msa_repository_existing_count": repository_existing,
            "alphafast_serial_msa_repository_invalid_count": repository_invalid,
        },
        commands,
    )


def _apply_msa_repository_to_run_csv(
    *,
    run_csv: Path,
    msa_repository_dir: Path = MSA_REPOSITORY_DIR,
    max_records: int = 0,
) -> dict[str, Any]:
    df = pd.read_csv(run_csv)
    repository_hits = 0
    repository_misses = 0
    invalid = 0
    limit = int(max_records) if max_records and int(max_records) > 0 else len(df)
    for index, row in df.head(limit).iterrows():
        binder_chain = str(row.get("binder_chain") or "A").strip() or "A"
        chains = _run_csv_target_msa_chains(row, binder_chain)
        for chain in chains:
            if not chain:
                continue
            raw_sequence = row.get(f"target_subchain_{chain}_seq")
            if raw_sequence is None or pd.isna(raw_sequence) or not str(raw_sequence).strip():
                raw_sequence = row.get(f"{chain}_seq")
            sequence = "" if raw_sequence is None or pd.isna(raw_sequence) else str(raw_sequence).strip()
            if not sequence:
                continue
            host_path, source = target_msa_workflow.find_cached_msa_for_sequence(
                sequence,
                msa_repository_dir=msa_repository_dir,
            )
            if host_path is not None:
                df.loc[index, f"msa_path_{chain}"] = str(host_path)
                repository_hits += 1
                if source.startswith("sequence_scan"):
                    preexisting_source_col = f"msa_path_{chain}_cache_source"
                    df.loc[index, preexisting_source_col] = source
            else:
                expected_path, _container_path = target_msa_workflow.boltz_msa_paths(
                    sequence,
                    msa_repository_dir=msa_repository_dir,
                )
                if expected_path.exists():
                    invalid += 1
                repository_misses += 1
    df.to_csv(run_csv, index=False)
    return {
        "msa_repository_dir": str(msa_repository_dir),
        "msa_repository_hit_count": repository_hits,
        "msa_repository_miss_count": repository_misses,
        "msa_repository_invalid_count": invalid,
    }


def _run_csv_missing_target_msas(run_csv: Path, max_records: int = 0) -> int:
    df = pd.read_csv(run_csv)
    limit = int(max_records) if max_records and int(max_records) > 0 else len(df)
    missing = 0
    for _, row in df.head(limit).iterrows():
        binder_chain = str(row.get("binder_chain") or "A").strip() or "A"
        for chain in _run_csv_target_msa_chains(row, binder_chain):
            if not chain:
                continue
            msa_value = str(row.get(f"msa_path_{chain}") or "").strip()
            if not msa_value or msa_value.lower() == "no_msa" or not Path(msa_value).exists():
                missing += 1
    return missing


def _a3m_sequence_count(path: Path | None) -> int:
    if path is None or not Path(path).exists():
        return 0
    count = 0
    seen_sequence = False
    try:
        for raw_line in Path(path).read_text(errors="ignore").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seen_sequence:
                    count += 1
                seen_sequence = False
            else:
                seen_sequence = True
    except OSError:
        return 0
    if seen_sequence:
        count += 1
    return count


def _a3m_match_state_sequence(sequence: str) -> str:
    return "".join(char.upper() for char in str(sequence or "") if char == "-" or not char.islower())


def _build_colabfold_multichain_a3m_for_row(row: pd.Series) -> tuple[str | None, dict[str, Any]]:
    binder_id = str(row.get("binder_id") or "").strip()
    binder_chain = str(row.get("binder_chain") or "A").strip() or "A"
    target_chains = _run_csv_target_msa_chains(row, binder_chain)
    if _run_csv_is_capacity_target_only(row):
        chains = target_chains
    else:
        chains = [binder_chain, *[chain for chain in target_chains if chain != binder_chain]]
    sequences = [_run_csv_chain_sequence(row, chain, binder_chain) for chain in chains]
    if not binder_id or not chains or not all(sequences):
        return None, {"status": "missing_sequence", "chain_count": len(chains)}

    chain_lengths = [len(sequence) for sequence in sequences]
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for length in chain_lengths:
        offsets.append((cursor, cursor + length))
        cursor += length

    lines = [
        f"#{','.join(str(length) for length in chain_lengths)}\t{','.join('1' for _chain in chains)}",
        ">" + "\t".join(str(101 + index) for index, _chain in enumerate(chains)),
        "".join(sequences),
    ]
    appended = 0
    source_counts: dict[str, int] = {}
    invalid_counts: dict[str, int] = {}
    for chain, sequence, (start, end) in zip(chains, sequences, offsets):
        if chain == binder_chain and not _run_csv_is_capacity_target_only(row):
            msa_value = ""
        else:
            msa_value = str(row.get(f"msa_path_{chain}") or "").strip()
        msa_path = Path(msa_value).expanduser() if msa_value and msa_value.lower() != "no_msa" else None
        records: list[tuple[str, str]] = []
        if msa_path is not None and msa_path.exists():
            records = _parse_a3m_records(msa_path.read_text(errors="ignore"))
        if not records:
            records = [(f">{chain}_query", sequence)]
        source_counts[chain] = len(records)
        invalid = 0
        for record_index, (header, record_sequence) in enumerate(records, start=1):
            match_sequence = _a3m_match_state_sequence(record_sequence)
            if _a3m_non_insert_length(match_sequence) != len(sequence):
                invalid += 1
                continue
            padded = "-" * start + match_sequence + "-" * (cursor - end)
            label = str(header or f">{chain}_{record_index}").lstrip(">").split()[0] or f"{chain}_{record_index}"
            lines.append(f">{chain}_{record_index}_{label}")
            lines.append(padded)
            appended += 1
        invalid_counts[chain] = invalid
    return "\n".join(lines + [""]), {
        "status": "real_multichain" if appended > len(chains) else "query_only",
        "chain_count": len(chains),
        "sequence_count": appended + 1,
        "source_counts": source_counts,
        "invalid_counts": invalid_counts,
    }


def _run_csv_nonreal_target_msas(run_csv: Path, max_records: int = 0) -> int:
    df = pd.read_csv(run_csv)
    limit = int(max_records) if max_records and int(max_records) > 0 else len(df)
    nonreal = 0
    for _, row in df.head(limit).iterrows():
        binder_chain = str(row.get("binder_chain") or "A").strip() or "A"
        for chain in _run_csv_target_msa_chains(row, binder_chain):
            if not chain:
                continue
            target_length = _safe_int(row.get(f"target_subchain_{chain}_len") or row.get(f"{chain}_length"))
            if target_length and target_length < TARGET_MSA_REQUIRED_MIN_LENGTH:
                continue
            msa_value = str(row.get(f"msa_path_{chain}") or "").strip()
            if not msa_value or msa_value.lower() == "no_msa":
                nonreal += 1
                continue
            msa_path = Path(msa_value)
            if not msa_path.exists() or _a3m_sequence_count(msa_path) <= 1:
                nonreal += 1
    return nonreal


def _repair_colabfold_target_msas(
    *,
    run_csv: Path,
    output_dir: Path,
    max_records: int = 0,
) -> dict[str, Any]:
    """Make ColabFold inputs use the prepared target-chain MSAs.

    The shared input generator is binder/design oriented and can emit weak or
    query-only multichain A3Ms. Rebuild those A3Ms from the per-chain target
    MSAs, padding each chain's rows across the other chains in ColabFold
    multimer A3M format. Binder chains remain query-only unless the row is a
    capacity target-only row.
    """
    input_dir = output_dir / "ColabFold" / "input_folder"
    if not run_csv.exists() or not input_dir.exists():
        return {}
    df = pd.read_csv(run_csv)
    limit = int(max_records) if max_records and int(max_records) > 0 else len(df)
    repaired = 0
    already_real = 0
    missing = 0
    query_only_before = 0
    source_nonreal = 0
    max_sequence_count = 0
    repaired_records: list[dict[str, Any]] = []
    for _, row in df.head(limit).iterrows():
        binder_id = str(row.get("binder_id") or "").strip()
        if not binder_id:
            continue
        staged = input_dir / f"{_safe_id(binder_id)}.a3m"
        if not staged.exists():
            missing += 1
            continue
        staged_count = _a3m_sequence_count(staged)
        if staged_count <= 1:
            query_only_before += 1
        rebuilt, build_stats = _build_colabfold_multichain_a3m_for_row(row)
        if rebuilt is None:
            missing += 1
            continue
        sequence_count = int(build_stats.get("sequence_count") or 0)
        max_sequence_count = max(max_sequence_count, sequence_count)
        if str(build_stats.get("status") or "") != "real_multichain":
            source_nonreal += 1
        if staged_count == sequence_count and staged.read_text(errors="ignore") == rebuilt:
            already_real += 1
            continue
        staged.write_text(rebuilt)
        repaired_records.append({"binder_id": binder_id, **build_stats})
        repaired += 1
    if not (repaired or already_real or missing or query_only_before or source_nonreal):
        return {}
    return {
        "colabfold_capacity_target_msa_repaired_count": repaired,
        "colabfold_capacity_target_msa_already_real_count": already_real,
        "colabfold_capacity_target_msa_missing_count": missing,
        "colabfold_capacity_target_msa_query_only_before_count": query_only_before,
        "colabfold_capacity_target_msa_source_nonreal_count": source_nonreal,
        "colabfold_capacity_target_msa_max_sequence_count": max_sequence_count,
        "colabfold_capacity_target_msa_records": repaired_records[:20],
    }


def _write_chain_msa_map(run_csv: Path, output_path: Path, max_records: int = 0) -> dict[str, Any]:
    df = pd.read_csv(run_csv)
    limit = int(max_records) if max_records and int(max_records) > 0 else len(df)
    records: list[dict[str, Any]] = []
    msa_present = 0
    msa_missing = 0
    msa_real = 0
    msa_query_only = 0
    msa_empty = 0
    target_chain_count = 0
    target_msa_blocking_nonreal = 0
    target_msa_short_nonreal = 0
    for _, row in df.head(limit).iterrows():
        binder_id = str(row.get("binder_id") or "").strip()
        if not binder_id:
            continue
        binder_chain = str(row.get("binder_chain") or "A").strip() or "A"
        target_chains = _run_csv_target_msa_chains(row, binder_chain)
        target_entries: list[dict[str, Any]] = []
        for chain in target_chains:
            msa_path = str(row.get(f"msa_path_{chain}") or "").strip()
            has_msa = bool(msa_path and msa_path.lower() != "no_msa" and Path(msa_path).exists())
            msa_sequence_count = _a3m_sequence_count(Path(msa_path)) if has_msa else 0
            if not has_msa:
                msa_status = "missing"
            elif msa_sequence_count <= 0:
                msa_status = "empty"
                msa_empty += 1
            elif msa_sequence_count == 1:
                msa_status = "query_only"
                msa_query_only += 1
            else:
                msa_status = "real_msa"
                msa_real += 1
            target_length = _safe_int(row.get(f"target_subchain_{chain}_len") or row.get(f"{chain}_length"))
            real_msa_required = not (target_length and target_length < TARGET_MSA_REQUIRED_MIN_LENGTH)
            if msa_status != "real_msa":
                if real_msa_required:
                    target_msa_blocking_nonreal += 1
                else:
                    target_msa_short_nonreal += 1
            if has_msa:
                msa_present += 1
            else:
                msa_missing += 1
            target_chain_count += 1
            target_entries.append(
                {
                    "chain": chain,
                    "sequence_column": f"target_subchain_{chain}_seq"
                    if f"target_subchain_{chain}_seq" in df.columns
                    else f"{chain}_seq",
                    "length": target_length,
                    "msa_path": msa_path or None,
                    "msa_available": has_msa,
                    "msa_status": msa_status,
                    "msa_sequence_count": msa_sequence_count,
                    "real_msa_required": real_msa_required,
                }
            )
        records.append(
            {
                "binder_id": binder_id,
                "binder": {
                    "chain": binder_chain,
                    "sequence_column": f"{binder_chain}_seq" if f"{binder_chain}_seq" in df.columns else "A_seq",
                    "msa_path": None,
                    "msa_available": False,
                },
                "targets": target_entries,
            }
        )
    payload = {
        "records": records,
        "summary": {
            "record_count": len(records),
            "target_chain_count": target_chain_count,
            "target_msa_available_count": msa_present,
            "target_msa_missing_count": msa_missing,
            "target_msa_real_count": msa_real,
            "target_msa_query_only_count": msa_query_only,
            "target_msa_empty_count": msa_empty,
            "target_msa_blocking_nonreal_count": target_msa_blocking_nonreal,
            "target_msa_short_nonreal_count": target_msa_short_nonreal,
            "target_msa_required_min_length": TARGET_MSA_REQUIRED_MIN_LENGTH,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, payload)
    return payload["summary"]


def _run_csv_runtime_size(run_csv: Path, max_records: int = 0) -> dict[str, int]:
    if not run_csv.exists():
        return {"candidate_count": 0}
    df = pd.read_csv(run_csv)
    if max_records and int(max_records) > 0:
        df = df.head(int(max_records))
    stats = {"candidate_count": int(len(df))}
    total = 0
    for _, row in df.iterrows():
        binder_chain = str(row.get("binder_chain") or "A").strip() or "A"
        binder_len = row.get(f"{binder_chain}_length")
        if pd.isna(binder_len):
            binder_len = row.get("A_length")
        total += int(pd.to_numeric(pd.Series([binder_len]), errors="coerce").fillna(0).iloc[0])
        for chain in _split_list(row.get("target_chains")):
            if not chain or chain == binder_chain:
                continue
            target_len = row.get(f"{chain}_length")
            if pd.isna(target_len):
                target_len = row.get(f"target_subchain_{chain}_len")
            if pd.isna(target_len):
                sequence = str(row.get(f"target_subchain_{chain}_seq") or "")
                target_len = len(sequence) if sequence and sequence.lower() != "nan" else 0
            total += int(pd.to_numeric(pd.Series([target_len]), errors="coerce").fillna(0).iloc[0])
    if total:
        stats["total_residues"] = total
    return stats


def _record_runtime_timing(
    metrics: dict[str, Any],
    engine: str,
    started_at: float,
    *,
    candidate_count: int | None = None,
    total_residues: int | None = None,
) -> None:
    timings = metrics.setdefault("runtime_engine_timings", {})
    if not isinstance(timings, dict):
        timings = {}
        metrics["runtime_engine_timings"] = timings
    payload: dict[str, Any] = {"seconds": max(0.0, time.monotonic() - started_at)}
    if candidate_count is not None:
        payload["candidate_count"] = int(candidate_count)
    if total_residues is not None:
        payload["total_residues"] = int(total_residues)
    timings[engine] = payload


def _adopt_engine_child_run(parent_run_dir: Path, child_run_dir: Path, engine: str) -> Path:
    """Move a completed internal engine run into the canonical benchmark run."""
    engine_dir = parent_run_dir / "artifacts" / "engines" / engine
    engine_dir.parent.mkdir(parents=True, exist_ok=True)
    if engine_dir.exists():
        shutil.rmtree(engine_dir)
    shutil.move(str(child_run_dir), str(engine_dir))

    metadata_path = engine_dir / "metadata.json"
    if metadata_path.exists():
        metadata = read_json(metadata_path)
        for key in ("hidden", "parent_task_group", "parent_run_id", "parent_run_dir", "parent_role"):
            metadata.pop(key, None)
        metadata["embedded_in_benchmark"] = True
        metadata["benchmark_engine"] = engine
        write_json(metadata_path, metadata)
    return engine_dir


def _adopt_parent_engine_output(parent_run_dir: Path, source_dir: Path, engine: str) -> Path | None:
    """Move a parent-local engine output tree into artifacts/engines."""
    if not source_dir.exists():
        return None
    engine_dir = parent_run_dir / "artifacts" / "engines" / engine
    engine_dir.parent.mkdir(parents=True, exist_ok=True)
    if engine_dir.exists():
        shutil.rmtree(engine_dir)
    shutil.move(str(source_dir), str(engine_dir))
    return engine_dir


def _adopt_parent_engine_inputs(parent_run_dir: Path, source_dir: Path, engine: str) -> Path | None:
    """Attach generated model inputs to an already adopted engine run."""
    if not source_dir.exists():
        return None
    engine_dir = parent_run_dir / "artifacts" / "engines" / engine
    engine_dir.mkdir(parents=True, exist_ok=True)
    target_dir = engine_dir / "model_inputs"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.move(str(source_dir), str(target_dir))
    return engine_dir


def _rewrite_canonical_engine_references(
    parent_run_dir: Path,
    child_sources: dict[str, Path] | None = None,
) -> None:
    """Rewrite artifact references after engine outputs move into the parent run."""
    relative_replacements = {
        "artifacts/raw/esmfold2_benchmark/": "artifacts/engines/esmfold2/artifacts/raw/esmfold2_benchmark/",
        "artifacts/raw/af2_initial_guess/": "artifacts/engines/af2_initial_guess/artifacts/raw/af2_initial_guess/",
        "artifacts/raw/boltz2_initial_guess/": "artifacts/engines/boltz2/artifacts/raw/boltz2_initial_guess/",
        "artifacts/raw/de_novo_binder_scoring/output/AF3/": "artifacts/engines/alphafast_af3/",
        "artifacts/raw/de_novo_binder_scoring/output/ColabFold/": "artifacts/engines/colabfold/",
        "artifacts/raw/de_novo_binder_scoring/output/Boltz/": "artifacts/engines/boltz2/model_inputs/",
    }
    child_relative_roots = {
        "esmfold2": (
            "artifacts/raw/esmfold2_benchmark/",
            "artifacts/raw/inputs/",
            "artifacts/normalized_candidates/",
            "artifacts/benchmark/",
        ),
        "af2_initial_guess": (
            "artifacts/raw/af2_initial_guess/",
            "artifacts/normalized_candidates/",
        ),
        "boltz2": (
            "artifacts/raw/boltz2_initial_guess/",
            "artifacts/normalized_candidates/",
        ),
    }
    absolute_replacements: dict[str, str] = {}
    for engine, source in (child_sources or {}).items():
        engine_dir = parent_run_dir / "artifacts" / "engines" / engine
        for relative_root in child_relative_roots.get(engine, ()):
            absolute_replacements[f"{source}/{relative_root}"] = f"{engine_dir}/{relative_root}"
    text_suffixes = {".csv", ".json", ".jsonl", ".pml", ".txt", ".yaml", ".yml"}
    roots = (
        (parent_run_dir / "artifacts" / "benchmark", {}),
        (
            parent_run_dir / "artifacts" / "engines" / "esmfold2",
            {
                "artifacts/benchmark/": "artifacts/engines/esmfold2/artifacts/benchmark/",
                "artifacts/normalized_candidates/": "artifacts/engines/esmfold2/artifacts/normalized_candidates/",
            },
        ),
        (
            parent_run_dir / "artifacts" / "engines" / "af2_initial_guess",
            {
                "artifacts/normalized_candidates/": "artifacts/engines/af2_initial_guess/artifacts/normalized_candidates/",
            },
        ),
        (
            parent_run_dir / "artifacts" / "engines" / "boltz2",
            {
                "artifacts/normalized_candidates/": "artifacts/engines/boltz2/artifacts/normalized_candidates/",
            },
        ),
        (parent_run_dir / "artifacts" / "engines" / "colabfold", {}),
        (parent_run_dir / "artifacts" / "engines" / "alphafast_af3", {}),
    )
    for root, local_replacements in roots:
        if not root.exists():
            continue
        replacements = {**absolute_replacements, **relative_replacements, **local_replacements}
        replacement_pattern = re.compile(
            "|".join(re.escape(old) for old in sorted(replacements, key=len, reverse=True))
        )
        for path in root.rglob("*"):
            if (
                not path.is_file()
                or path.name == "command.json"
                or path.suffix.lower() not in text_suffixes
            ):
                continue
            try:
                original = path.read_text()
            except (OSError, UnicodeDecodeError):
                continue

            def replace_reference(match: re.Match[str]) -> str:
                old = match.group(0)
                new = replacements[old]
                canonical_prefix = new[: -len(old)] if new.endswith(old) else ""
                if canonical_prefix and original[max(0, match.start() - len(canonical_prefix)) : match.start()] == canonical_prefix:
                    return old
                return new

            rewritten = replacement_pattern.sub(replace_reference, original)
            if rewritten != original:
                path.write_text(rewritten)


def _canonicalize_benchmark_engine_artifacts(
    *,
    parent_run_dir: Path,
    output_dir: Path,
    esmfold2_child_runs: list[str],
    af2_child_runs: list[str],
    boltz2_child_runs: list[str],
    extra_child_runs: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    engine_dirs: dict[str, Path] = {}
    child_sources: dict[str, Path] = {}
    child_groups = {
        "esmfold2": esmfold2_child_runs,
        "af2_initial_guess": af2_child_runs,
        "boltz2": boltz2_child_runs,
        **(extra_child_runs or {}),
    }
    for engine, child_runs in child_groups.items():
        if not child_runs:
            continue
        child_dir = Path(child_runs[-1])
        if child_dir.exists():
            child_sources[engine] = child_dir
            engine_dirs[engine] = _adopt_engine_child_run(parent_run_dir, child_dir, engine)

    parent_outputs = {
        "colabfold": output_dir / "ColabFold",
        "alphafast_af3": output_dir / "AF3",
    }
    for engine, source_dir in parent_outputs.items():
        adopted = _adopt_parent_engine_output(parent_run_dir, source_dir, engine)
        if adopted is not None:
            engine_dirs[engine] = adopted

    boltz2_dir = _adopt_parent_engine_inputs(parent_run_dir, output_dir / "Boltz", "boltz2")
    if boltz2_dir is not None:
        engine_dirs["boltz2"] = boltz2_dir

    _rewrite_canonical_engine_references(parent_run_dir, child_sources)
    return {
        engine: str(engine_dir.relative_to(parent_run_dir))
        for engine, engine_dir in sorted(engine_dirs.items())
    }


def _emit_benchmark_progress(
    run_dir: Path,
    *,
    phase: str,
    engine: str = "",
    step: int | None = None,
    total_steps: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    label = phase if not engine else f"{phase}: {engine}"
    payload: dict[str, Any] = {
        "current_phase": phase,
        "current_engine": engine,
        "progress_label": label,
    }
    if step is not None:
        payload["progress_step"] = int(step)
    if total_steps is not None:
        payload["progress_total"] = int(total_steps)
    update_status(run_dir, "running", **payload)
    if progress_callback is not None:
        progress_callback(dict(payload))


def _generate_model_inputs_command(*, output_dir: Path, docker_image: str, models: list[str] | tuple[str, ...] | None = None) -> list[str]:
    command = [
        "docker",
        "run",
        "--rm",
        *_repo_and_runs_mounts(),
        "-w",
        str(DE_NOVO_BINDER_SCORING_DIR),
        docker_image,
        "python",
        "./scripts/generate_model_inputs.py",
        "--run-csv",
        str(output_dir / "run.csv"),
        "--out-dir",
        str(output_dir),
    ]
    if models:
        command.extend(["--models", *models])
    return command


def _model_inputs_ready(output_dir: Path, models: list[str] | tuple[str, ...]) -> bool:
    checks = {
        "af3": output_dir / "AF3" / "input_folder",
        "boltz": output_dir / "Boltz" / "input_folder",
        "colabfold": output_dir / "ColabFold" / "input_folder",
    }
    for model in models:
        folder = checks.get(str(model))
        if folder is not None and not (folder.exists() and any(folder.iterdir())):
            return False
    return True


def _generate_colabfold_inputs_command(*, output_dir: Path, docker_image: str) -> list[str]:
    return _generate_model_inputs_command(output_dir=output_dir, docker_image=docker_image, models=["colabfold"])


def _summarize_alphafast_af3_outputs(
    *,
    parent_run_dir: Path,
    run_csv: Path,
    alphafast_output_dir: Path,
) -> tuple[Path | None, Path | None, dict[str, Any]]:
    if not alphafast_output_dir.exists():
        return None, None, {"record_count": 0, "scored_feature_count": 0}
    labels: dict[str, Any] = {}
    if run_csv.exists():
        run_df = pd.read_csv(run_csv)
        for _, row in run_df.iterrows():
            binder_id = str(row.get("binder_id") or "").strip()
            if binder_id:
                labels[binder_id] = _truthy_label(row.get("binder") if "binder" in run_df.columns else row.get("label"))
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(alphafast_output_dir.glob("*/*_summary_confidences.json")):
        binder_id = summary_path.parent.name
        try:
            summary = json.loads(summary_path.read_text())
        except json.JSONDecodeError:
            continue
        row: dict[str, Any] = {
            "binder_id": binder_id,
            "candidate_id": binder_id,
            "label": labels.get(binder_id),
            "summary_confidences": str(summary_path.relative_to(parent_run_dir)),
        }
        for key in ("ranking_score", "iptm", "ptm", "fraction_disordered", "has_clash"):
            if key in summary:
                row[f"alphafast_af3_{key}"] = summary.get(key)
        chain_iptm = summary.get("chain_iptm")
        if isinstance(chain_iptm, list):
            for index, value in enumerate(chain_iptm, start=1):
                row[f"alphafast_af3_chain_{index}_iptm"] = value
        chain_ptm = summary.get("chain_ptm")
        if isinstance(chain_ptm, list):
            for index, value in enumerate(chain_ptm, start=1):
                row[f"alphafast_af3_chain_{index}_ptm"] = value
        chain_pair_iptm = summary.get("chain_pair_iptm")
        if isinstance(chain_pair_iptm, list):
            for row_index, values in enumerate(chain_pair_iptm, start=1):
                if isinstance(values, list):
                    for col_index, value in enumerate(values, start=1):
                        row[f"alphafast_af3_chain_pair_{row_index}_{col_index}_iptm"] = value
        chain_pair_pae_min = summary.get("chain_pair_pae_min")
        if isinstance(chain_pair_pae_min, list):
            for row_index, values in enumerate(chain_pair_pae_min, start=1):
                if isinstance(values, list):
                    for col_index, value in enumerate(values, start=1):
                        row[f"alphafast_af3_chain_pair_{row_index}_{col_index}_pae_min"] = value
        ranking_csv = summary_path.parent / f"{binder_id}_ranking_scores.csv"
        if ranking_csv.exists():
            scores = pd.read_csv(ranking_csv)
            numeric_scores = pd.to_numeric(scores.get("ranking_score"), errors="coerce") if "ranking_score" in scores else pd.Series(dtype=float)
            if numeric_scores.notna().any():
                row["alphafast_af3_best_sample_ranking_score"] = float(numeric_scores.max())
                row["alphafast_af3_mean_sample_ranking_score"] = float(numeric_scores.mean())
        rows.append(row)
    if not rows:
        return None, None, {"record_count": 0, "scored_feature_count": 0}
    out_dir = parent_run_dir / "artifacts" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = out_dir / "alphafast_af3_metrics.csv"
    summary_path = out_dir / "alphafast_af3_feature_summary.json"
    df = pd.DataFrame(rows)
    df.to_csv(table, index=False)
    feature_rows, summary = _metric_column_summary(df, "label", max_columns=500)
    summary.update(
        {
            "record_count": len(rows),
            "scored_feature_count": len(feature_rows),
            "top_feature": feature_rows[0]["feature"] if feature_rows else None,
            "top_feature_average_precision": feature_rows[0]["best_average_precision"] if feature_rows else None,
            "top_feature_auroc": feature_rows[0]["best_auroc"] if feature_rows else None,
        }
    )
    write_json(summary_path, summary)
    _write_feature_ranking(out_dir / "alphafast_af3_feature_benchmark.csv", feature_rows)
    return table, summary_path, summary


def _run_colabfold_prediction(
    *,
    job_run_dir: Path,
    run_csv: Path,
    output_dir: Path,
    cache_dir: Path,
    num_recycles: int,
    num_models: int,
    gpu_device: object,
    max_records: int,
    use_target_templates: bool,
    max_template_hits: int,
    image: str,
) -> tuple[int, int, int, list[list[str]]]:
    input_dir = output_dir / "ColabFold" / "input_folder"
    ptm_output_dir = output_dir / "ColabFold" / "ptm_output"
    if not input_dir.exists():
        raise ValueError(f"ColabFold input folder is missing: {input_dir}")
    _prepare_colabfold_params_cache(cache_dir)
    ptm_output_dir.mkdir(parents=True, exist_ok=True)

    selected_input_dir = input_dir
    selected_count = len(list(input_dir.glob("*.a3m")))
    if max_records > 0 and run_csv.exists():
        selected_input_dir = output_dir / "ColabFold" / "input_subset"
        if selected_input_dir.exists():
            shutil.rmtree(selected_input_dir)
        selected_input_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(run_csv).head(int(max_records))
        selected_count = 0
        for binder_id in df.get("binder_id", []):
            source = input_dir / f"{_safe_id(binder_id)}.a3m"
            if source.exists():
                shutil.copy2(source, selected_input_dir / source.name)
                selected_count += 1

    template_count = 0
    template_dir: Path | None = None
    if use_target_templates and run_csv.exists():
        template_dir, template_count = _stage_colabfold_target_templates(run_csv, output_dir, max_records=max_records)

    command = [
        "docker",
        "run",
        "--rm",
        *docker_gpu_args(gpu_device),
        "--shm-size=32G",
        *_repo_and_runs_mounts(),
        "-v",
        f"{cache_dir}:/cache/params:rw",
        "-w",
        str(DE_NOVO_BINDER_SCORING_DIR),
        image,
        "colabfold_batch",
        str(selected_input_dir),
        str(ptm_output_dir),
        "--data",
        "/cache",
        "--calc-extra-ptm",
        "--num-recycle",
        str(int(num_recycles)),
        "--num-models",
        str(int(num_models)),
    ]
    if template_dir is not None and template_count:
        template_cache_dir = output_dir / "ColabFold" / "target_template_cache"
        template_cache_dir.mkdir(parents=True, exist_ok=True)
        command.extend(
            [
                "--templates",
                "--custom-template-path",
                str(template_dir),
                "--custom-template-cache-path",
                str(template_cache_dir),
                "--max-template-hits",
                str(int(max_template_hits)),
            ]
        )
    rc = _run_docker_command(job_run_dir, command)
    return rc, selected_count, template_count, [command]


def _prepare_colabfold_params_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    multimer_files = [cache_dir / f"params_model_{index}_multimer_v3.npz" for index in range(1, 6)]
    ptm_files = [cache_dir / f"params_model_{index}_ptm.npz" for index in range(1, 6)]
    monomer_files = [cache_dir / f"params_model_{index}.npz" for index in range(1, 6)]
    if all(path.exists() for path in multimer_files):
        (cache_dir / "download_complexes_multimer_v3_finished.txt").touch()
    if all(path.exists() for path in ptm_files) or all(path.exists() for path in monomer_files):
        (cache_dir / "download_finished.txt").touch()


def _extract_and_summarize_colabfold_outputs(
    *,
    parent_run_dir: Path,
    run_csv: Path,
    output_dir: Path,
    docker_image: str,
) -> tuple[Path | None, Path | None, dict[str, Any], list[list[str]]]:
    command = [
        "docker",
        "run",
        "--rm",
        *_repo_and_runs_mounts(),
        "-w",
        str(DE_NOVO_BINDER_SCORING_DIR),
        docker_image,
        "python",
        "./scripts/extract_confidence_metrics.py",
        "--run-csv",
        str(run_csv),
        "--out-dir",
        str(output_dir),
        "--models",
        "colab",
    ]
    rc = _run_docker_command(parent_run_dir, command)
    if rc != 0:
        return None, None, {"extract_colabfold_return_code": rc}, [command]

    metrics_csv = output_dir / "ColabFold" / "colab_metrics.csv"
    if not metrics_csv.exists():
        return None, None, {"record_count": 0, "scored_feature_count": 0}, [command]

    run_df = pd.read_csv(run_csv)
    metrics_df = pd.read_csv(metrics_csv)
    colab_metric_cols = [column for column in metrics_df.columns if str(column).startswith("colab_")]
    if colab_metric_cols:
        metrics_df = metrics_df.dropna(subset=colab_metric_cols, how="all")
    if "binder_id" in run_df.columns and "binder_id" in metrics_df.columns:
        labels = run_df[["binder_id"]].copy()
        labels["label"] = run_df.get("binder", run_df.get("label"))
        metrics_df = labels.merge(metrics_df, on="binder_id", how="right")
    elif "label" not in metrics_df.columns:
        metrics_df["label"] = None

    out_dir = parent_run_dir / "artifacts" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = out_dir / "colabfold_metrics.csv"
    summary_path = out_dir / "colabfold_feature_summary.json"
    metrics_df.to_csv(table, index=False)
    feature_rows, summary = _metric_column_summary(metrics_df, "label", max_columns=500)
    summary.update(
        {
            "metrics_csv": str(metrics_csv.relative_to(parent_run_dir)),
            "scored_feature_count": len(feature_rows),
            "top_feature": feature_rows[0]["feature"] if feature_rows else None,
            "top_feature_average_precision": feature_rows[0]["best_average_precision"] if feature_rows else None,
            "top_feature_auroc": feature_rows[0]["best_auroc"] if feature_rows else None,
        }
    )
    write_json(summary_path, summary)
    _write_feature_ranking(out_dir / "colabfold_feature_benchmark.csv", feature_rows)
    return table, summary_path, summary, [command]


def run_de_novo_binder_scoring_dataset(
    *,
    input_zip: Path | None = None,
    input_zip_bytes: bytes | None = None,
    input_csv: Path | None = None,
    input_csv_text: str | None = None,
    input_pdb_dir: Path | None = None,
    source_run_dir: Path | None = None,
    candidates_jsonl: Path | None = None,
    selected_candidate_ids: list[str] | None = None,
    target_override_pdb: Path | None = None,
    target_override_chains: list[str] | None = None,
    mode: str = "pdb_only",
    generate_inputs: bool = True,
    models: list[str] | None = None,
    run_pyrosetta_input_metrics: bool = False,
    pyrosetta_nprocs: int = 1,
    run_common_interface_metrics: bool = True,
    run_predicted_rosetta_metrics: bool = False,
    run_pymol_metrics: bool = False,
    run_esmfold2: bool = False,
    esmfold2_modes: list[str] | None = None,
    esmfold2_use_target_msa: bool = False,
    run_af2_initial_guess: bool = False,
    af2_num_recycles: int = 3,
    af2_multimer: bool = True,
    af2_use_initial_guess: bool = False,
    af2_use_binder_template: bool = False,
    af2_use_interface_template: bool = False,
    run_boltz2_initial_guess: bool = False,
    boltz2_use_target_template: bool = True,
    boltz2_use_target_msa: bool = True,
    boltz2_recycling_steps: int = 10,
    boltz2_sampling_steps: int = 200,
    boltz2_diffusion_samples: int = 3,
    boltz2_write_full_pae: bool = True,
    run_rf3: bool = False,
    rf3_checkpoint_path: Path = refolding_workflow.RF3_CHECKPOINT,
    rf3_use_target_msa: bool = True,
    rf3_use_target_template: bool = True,
    rf3_recycles: int = 10,
    rf3_num_steps: int = 50,
    rf3_diffusion_batch_size: int = 5,
    rf3_seed: int = 0,
    run_openfold3: bool = False,
    openfold3_checkpoint_path: Path = refolding_workflow.OPENFOLD3_CHECKPOINT,
    openfold3_use_target_msa: bool = True,
    openfold3_num_diffusion_samples: int = 5,
    openfold3_num_model_seeds: int = 1,
    openfold3_num_recycles: int = 3,
    openfold3_use_msa_server: bool = False,
    run_protenix: bool = False,
    protenix_use_msa: bool = True,
    protenix_cycle: int = 3,
    protenix_diffusion_steps: int = 50,
    protenix_samples: int = 5,
    run_protenix_v1: bool = False,
    protenix_v1_model_name: str = refolding_workflow.PROTENIX_V1_MODEL,
    protenix_v1_use_msa: bool = True,
    protenix_v1_use_template: bool = False,
    protenix_v1_use_default_params: bool = True,
    protenix_v1_cycle: int = 10,
    protenix_v1_diffusion_steps: int = 200,
    protenix_v1_samples: int = 5,
    run_protenix_v2: bool = False,
    protenix_v2_model_name: str = refolding_workflow.PROTENIX_V2_MODEL,
    protenix_v2_use_msa: bool = True,
    protenix_v2_use_template: bool = False,
    protenix_v2_use_default_params: bool = True,
    protenix_v2_cycle: int = 10,
    protenix_v2_diffusion_steps: int = 200,
    protenix_v2_samples: int = 5,
    run_boltzgen_fold: bool = False,
    boltzgen_recycling_steps: int = 3,
    boltzgen_sampling_steps: int = 200,
    boltzgen_diffusion_samples: int = 5,
    run_colabfold: bool = False,
    colabfold_cache_dir: Path = COLABFOLD_CACHE_DIR,
    colabfold_msa_source: str = "msa_repository_then_alphafast_mmseqs_gpu",
    msa_repository_dir: Path = MSA_REPOSITORY_DIR,
    require_real_target_msa: bool = True,
    colabfold_num_recycles: int = 3,
    colabfold_num_models: int = 3,
    colabfold_use_target_templates: bool = True,
    colabfold_use_target_msa: bool = True,
    colabfold_max_template_hits: int = 4,
    colabfold_gpu_device: object = 0,
    run_alphafast_af3: bool = False,
    alphafast_db_dir: Path = ALPHAFAST_DB_DIR,
    alphafast_weights_dir: Path = ALPHAFAST_WEIGHTS_DIR,
    alphafast_batch_size: int = 0,
    alphafast_num_recycles: int = 10,
    alphafast_use_target_templates: bool = True,
    alphafast_query_only_msa: bool = False,
    alphafast_gpu_device: object = 0,
    gpu_device: object | None = None,
    max_records: int = 0,
    num_loops: int = 3,
    num_sampling_steps: int = 32,
    seed: int = 0,
    device: str = "cuda",
    docker_image: str = SCORING_SCRIPTS_IMAGE,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    existing_job: JobPaths | None = None,
    resume: bool = False,
    job_type: str = "de_novo_binder_scoring_dataset",
    tool_name: str = "de_novo_binder_scoring_scripts",
    task_group: str = BENCHMARK_GROUP,
    target_refolding: bool = False,
    internal_parent_run_dir: Path | None = None,
    internal_parent_task_group: str = "",
    internal_parent_role: str = "",
    internal_parent_engine: str | None = None,
) -> Path:
    selected_gpu_device = normalize_gpu_device(gpu_device if gpu_device is not None else alphafast_gpu_device)
    if gpu_device is not None:
        colabfold_gpu_device = gpu_device
        alphafast_gpu_device = gpu_device
    colabfold_uses_target_msa = bool(colabfold_use_target_msa)
    msa_source = str(colabfold_msa_source or "msa_repository_then_alphafast_mmseqs_gpu")
    if not colabfold_uses_target_msa and msa_source != "repo_run_csv":
        msa_source = "repo_run_csv"
    msa_consuming_engines = bool(
        (run_colabfold and colabfold_uses_target_msa)
        or (run_alphafast_af3 and not alphafast_query_only_msa)
        or (run_boltz2_initial_guess and boltz2_use_target_msa)
        or (run_esmfold2 and esmfold2_use_target_msa)
        or (run_rf3 and rf3_use_target_msa)
        or (run_openfold3 and openfold3_use_target_msa)
        or (run_protenix and protenix_use_msa)
        or (run_protenix_v1 and protenix_v1_use_msa)
        or (run_protenix_v2 and protenix_v2_use_msa)
    )
    selected_models = [model for model in (models or ["af3", "boltz", "colabfold"]) if model in {"af3", "boltz", "colabfold"}]
    if run_colabfold and "colabfold" not in selected_models:
        selected_models.append("colabfold")
    if run_boltz2_initial_guess and "boltz" not in selected_models:
        selected_models.append("boltz")
    if msa_consuming_engines and msa_source in {"alphafast_mmseqs_gpu", "msa_repository_then_alphafast_mmseqs_gpu"} and "af3" not in selected_models:
        selected_models.append("af3")
    if run_alphafast_af3 and "af3" not in selected_models:
        selected_models.append("af3")
    input_payload = {
        "input_zip": str(input_zip) if input_zip else None,
        "input_zip_uploaded": input_zip_bytes is not None,
        "input_csv": str(input_csv) if input_csv else None,
        "input_pdb_dir": str(input_pdb_dir) if input_pdb_dir else None,
        "source_run_dir": str(source_run_dir) if source_run_dir else None,
        "candidates_jsonl": str(candidates_jsonl) if candidates_jsonl else None,
        "selected_candidate_ids": list(selected_candidate_ids or []),
        "target_override_pdb": str(target_override_pdb) if target_override_pdb else None,
        "target_override_chains": list(target_override_chains or []),
        "target_refolding": bool(target_refolding),
    }
    evaluation_mode = "refolding_validation" if candidates_jsonl else "binder_benchmark"
    prediction_phase = "Predicting target folds" if target_refolding else "Predicting complexes"
    metrics_phase = (
        "Calculating target-refolding metrics"
        if target_refolding
        else "Calculating evaluation metrics"
        if evaluation_mode == "refolding_validation"
        else "Calculating benchmark metrics"
    )
    params_payload = {
        "evaluation_mode": evaluation_mode,
        "target_refolding": bool(target_refolding),
        "mode": mode,
        "generate_inputs": generate_inputs,
        "models": selected_models,
        "run_pyrosetta_input_metrics": run_pyrosetta_input_metrics,
        "pyrosetta_nprocs": pyrosetta_nprocs,
        "run_common_interface_metrics": run_common_interface_metrics,
        "run_predicted_rosetta_metrics": run_predicted_rosetta_metrics,
        "run_pymol_metrics": run_pymol_metrics,
        "run_esmfold2": run_esmfold2,
        "esmfold2_modes": esmfold2_modes or ["initial_guess"],
        "esmfold2_use_target_msa": esmfold2_use_target_msa,
        "run_af2_initial_guess": run_af2_initial_guess,
        "af2_num_recycles": af2_num_recycles,
        "af2_multimer": af2_multimer,
        "af2_use_initial_guess": af2_use_initial_guess,
        "af2_use_binder_template": af2_use_binder_template,
        "af2_use_interface_template": af2_use_interface_template,
        "run_boltz2_initial_guess": run_boltz2_initial_guess,
        "boltz2_use_target_template": boltz2_use_target_template,
        "boltz2_use_target_msa": boltz2_use_target_msa,
        "boltz2_recycling_steps": boltz2_recycling_steps,
        "boltz2_sampling_steps": boltz2_sampling_steps,
        "boltz2_diffusion_samples": boltz2_diffusion_samples,
        "boltz2_write_full_pae": boltz2_write_full_pae,
        "boltz2_image": BOLTZ2_IMAGE,
        "run_rf3": run_rf3,
        "rf3_checkpoint_path": str(rf3_checkpoint_path),
        "rf3_use_target_msa": rf3_use_target_msa,
        "rf3_use_target_template": rf3_use_target_template,
        "rf3_recycles": rf3_recycles,
        "rf3_num_steps": rf3_num_steps,
        "rf3_diffusion_batch_size": rf3_diffusion_batch_size,
        "rf3_seed": rf3_seed,
        "run_openfold3": run_openfold3,
        "openfold3_checkpoint_path": str(openfold3_checkpoint_path),
        "openfold3_use_target_msa": openfold3_use_target_msa,
        "openfold3_num_diffusion_samples": openfold3_num_diffusion_samples,
        "openfold3_num_model_seeds": openfold3_num_model_seeds,
        "openfold3_num_recycles": openfold3_num_recycles,
        "openfold3_use_msa_server": openfold3_use_msa_server,
        "run_protenix": run_protenix,
        "protenix_use_msa": protenix_use_msa,
        "protenix_cycle": protenix_cycle,
        "protenix_diffusion_steps": protenix_diffusion_steps,
        "protenix_samples": protenix_samples,
        "run_protenix_v1": run_protenix_v1,
        "protenix_v1_model_name": protenix_v1_model_name,
        "protenix_v1_use_msa": protenix_v1_use_msa,
        "protenix_v1_use_template": protenix_v1_use_template,
        "protenix_v1_use_default_params": protenix_v1_use_default_params,
        "protenix_v1_cycle": protenix_v1_cycle,
        "protenix_v1_diffusion_steps": protenix_v1_diffusion_steps,
        "protenix_v1_samples": protenix_v1_samples,
        "run_protenix_v2": run_protenix_v2,
        "protenix_v2_model_name": protenix_v2_model_name,
        "protenix_v2_use_msa": protenix_v2_use_msa,
        "protenix_v2_use_template": protenix_v2_use_template,
        "protenix_v2_use_default_params": protenix_v2_use_default_params,
        "protenix_v2_cycle": protenix_v2_cycle,
        "protenix_v2_diffusion_steps": protenix_v2_diffusion_steps,
        "protenix_v2_samples": protenix_v2_samples,
        "run_boltzgen_fold": run_boltzgen_fold,
        "boltzgen_recycling_steps": boltzgen_recycling_steps,
        "boltzgen_sampling_steps": boltzgen_sampling_steps,
        "boltzgen_diffusion_samples": boltzgen_diffusion_samples,
        "run_colabfold": run_colabfold,
        "colabfold_image": COLABFOLD_IMAGE,
        "colabfold_cache_dir": str(colabfold_cache_dir),
        "colabfold_msa_source": msa_source,
        "shared_msa_source": msa_source,
        "msa_repository_dir": str(msa_repository_dir),
        "require_real_target_msa": bool(require_real_target_msa),
        "gpu_device": selected_gpu_device,
        "colabfold_num_recycles": colabfold_num_recycles,
        "colabfold_num_models": colabfold_num_models,
        "colabfold_use_target_templates": colabfold_use_target_templates,
        "colabfold_use_target_msa": colabfold_uses_target_msa,
        "colabfold_max_template_hits": colabfold_max_template_hits,
        "colabfold_gpu_device": colabfold_gpu_device,
        "run_alphafast_af3": run_alphafast_af3,
        "alphafast_image": ALPHAFAST_IMAGE,
        "alphafast_db_dir": str(alphafast_db_dir),
        "alphafast_weights_dir": str(alphafast_weights_dir),
        "alphafast_batch_size": alphafast_batch_size,
        "alphafast_num_recycles": alphafast_num_recycles,
        "alphafast_use_target_templates": bool(alphafast_use_target_templates),
        "alphafast_query_only_msa": bool(alphafast_query_only_msa),
        "alphafast_gpu_device": alphafast_gpu_device,
        "max_records": max_records,
        "selected_candidate_count": len(selected_candidate_ids or []),
        "docker_image": docker_image,
        "resume": bool(resume),
    }
    if existing_job is None:
        job = create_job(
            task_group,
            job_type,
            tool_name,
            input_payload,
            params_payload,
        )
    else:
        job = existing_job
        write_json(
            job.run_dir / "input.json",
            {
                "job_type": job_type,
                "tool": tool_name,
                "inputs": input_payload,
                "params": params_payload,
            },
        )
    if internal_parent_run_dir is not None and internal_parent_task_group and internal_parent_role:
        mark_internal_job(
            job.run_dir,
            parent_run_dir=Path(internal_parent_run_dir),
            parent_task_group=str(internal_parent_task_group),
            role=str(internal_parent_role),
            engine=internal_parent_engine,
        )
    update_status(job.run_dir, "running")
    raw_dir = job.run_dir / "artifacts" / "raw" / "de_novo_binder_scoring"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir = raw_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_run_csv = output_dir / "run.csv"
    resume_prepared_inputs = bool(resume and prepared_run_csv.exists() and _csv_data_row_count(prepared_run_csv) > 0)
    if candidates_jsonl is not None and not resume_prepared_inputs:
        try:
            if source_run_dir is None:
                raise ValueError("source_run_dir is required when candidates_jsonl is provided.")
            candidate_stage_dir = job.run_dir / "artifacts" / "queued_inputs" / "candidate_repo_dataset"
            input_csv, input_pdb_dir, candidate_stage_metrics = _stage_candidate_set_as_repo_dataset(
                source_run_dir=Path(source_run_dir),
                candidates_jsonl=Path(candidates_jsonl),
                staged_dir=candidate_stage_dir,
                max_candidates=int(max_records or 0),
                selected_candidate_ids=list(selected_candidate_ids or []),
                target_override_pdb=Path(target_override_pdb) if target_override_pdb else None,
                target_override_chains=list(target_override_chains or []),
            )
        except Exception as exc:
            finish_job(
                job.run_dir,
                False,
                {
                    "metrics": {
                        "error": f"{type(exc).__name__}: {exc}",
                        "phase": "stage_candidate_set",
                    }
                },
            )
            raise
        input_payload["input_csv"] = str(input_csv)
        input_payload["input_pdb_dir"] = str(input_pdb_dir)
        params_payload.update(candidate_stage_metrics)
        write_json(
            job.run_dir / "input.json",
            {
                "job_type": job_type,
                "tool": tool_name,
                "inputs": input_payload,
                "params": params_payload,
            },
        )
    if resume_prepared_inputs:
        staged_csv = raw_dir / "dataset" / "input.csv"
        staged_pdb_dir = raw_dir / "dataset" / "input_pdbs"
        if not staged_csv.exists():
            staged_csv = None
        if not staged_pdb_dir.exists():
            staged_pdb_dir = output_dir / "input_pdbs"
        if not staged_pdb_dir.exists():
            staged_pdb_dir = None
    else:
        staged_csv, staged_pdb_dir = _stage_repo_format_inputs(
            raw_dir=raw_dir,
            input_zip=input_zip,
            input_zip_bytes=input_zip_bytes,
            input_csv=input_csv,
            input_csv_text=input_csv_text,
            input_pdb_dir=input_pdb_dir,
        )
    if resume_prepared_inputs and not _prepared_run_csv_matches_staged_input(prepared_run_csv, staged_csv):
        resume_prepared_inputs = False
        params_payload["prepared_inputs_resume_invalidated"] = True
        params_payload["prepared_inputs_resume_invalidated_reason"] = (
            "Existing output/run.csv binder IDs do not match staged dataset/input.csv binder IDs."
        )
        write_json(
            job.run_dir / "input.json",
            {
                "job_type": job_type,
                "tool": tool_name,
                "inputs": input_payload,
                "params": params_payload,
            },
        )
    progress_total = sum(
        1
        for enabled in [
            True,
            bool(msa_consuming_engines and msa_source != "repo_run_csv"),
            bool(generate_inputs),
            bool(run_pyrosetta_input_metrics),
            bool(run_esmfold2),
            bool(run_af2_initial_guess),
            bool(run_boltz2_initial_guess),
            bool(run_rf3),
            bool(run_openfold3),
            bool(run_protenix),
            bool(run_protenix_v1),
            bool(run_protenix_v2),
            bool(run_boltzgen_fold),
            bool(run_colabfold),
            bool(run_alphafast_af3),
            True,
        ]
        if enabled
    )
    progress_step = 1
    _emit_benchmark_progress(
        job.run_dir,
        phase="Preparing benchmark inputs",
        engine="input",
        step=progress_step,
        total_steps=progress_total,
        progress_callback=progress_callback,
    )
    process_cmd = [
        "docker",
        "run",
        "--rm",
        *_repo_and_runs_mounts(),
        "-w",
        str(DE_NOVO_BINDER_SCORING_DIR),
        docker_image,
        "python",
        "./scripts/process_inputs.py",
        "--mode",
        mode,
        "--output_dir",
        str(output_dir),
    ]
    if mode in {"hybrid", "pdb_only"}:
        if staged_pdb_dir is None:
            finish_job(job.run_dir, False, {"metrics": {"error": "input_pdbs missing"}})
            raise ValueError("Repo-format benchmark needs input_pdbs for pdb_only or hybrid mode.")
        process_cmd.extend(["--input_pdbs", str(staged_pdb_dir)])
    if mode in {"hybrid", "seq_only_csv"}:
        if staged_csv is None:
            finish_job(job.run_dir, False, {"metrics": {"error": "input_csv missing"}})
            raise ValueError("Repo-format benchmark needs input.csv for hybrid or seq_only_csv mode.")
        process_cmd.extend(["--input_csv", str(staged_csv)])
    if resume_prepared_inputs:
        existing_commands = read_json(job.run_dir / "command.json").get("commands")
        commands = list(existing_commands) if isinstance(existing_commands, list) else []
    else:
        write_json(job.run_dir / "command.json", {"mode": "docker", "commands": [process_cmd]})
        return_code = _run_docker_command(job.run_dir, process_cmd)
        if return_code != 0:
            finish_job(job.run_dir, False, {"metrics": {"process_inputs_return_code": return_code}})
            raise RuntimeError(f"de_novo_binder_scoring process_inputs failed with return code {return_code}.")
        commands = [process_cmd]
    run_csv = output_dir / "run.csv"
    if not resume_prepared_inputs and staged_pdb_dir is not None and staged_pdb_dir.exists():
        canonical_input_dir = output_dir / "input_pdbs"
        canonical_input_dir.mkdir(parents=True, exist_ok=True)
        for source in staged_pdb_dir.glob("*.pdb"):
            shutil.copy2(source, canonical_input_dir / source.name)
            chain_map = source.with_suffix(".chain_map.json")
            if chain_map.exists():
                shutil.copy2(chain_map, canonical_input_dir / chain_map.name)
    pre_input_metrics: dict[str, Any] = {}
    pre_input_metrics.update(_apply_staged_csv_metadata_to_run_csv(run_csv, staged_csv))
    pre_input_metrics.update(_validate_run_csv_chain_roles(run_csv, max_records=int(max_records)))
    runtime_size = _run_csv_runtime_size(run_csv, max_records=int(max_records)) if run_csv.exists() else {"candidate_count": 0}
    alphafast_data_pipeline_done = False
    if not resume_prepared_inputs and msa_consuming_engines and msa_source != "repo_run_csv" and run_csv.exists():
        progress_step += 1
        _emit_benchmark_progress(
            job.run_dir,
            phase="Preparing target MSAs",
            engine="shared MSA resolver",
            step=progress_step,
            total_steps=progress_total,
            progress_callback=progress_callback,
        )
        pre_input_metrics["shared_msa_source"] = msa_source
        if msa_source in {"msa_repository", "msa_repository_then_alphafast_mmseqs_gpu"}:
            pre_input_metrics.update(
                _apply_msa_repository_to_run_csv(
                    run_csv=run_csv,
                    msa_repository_dir=Path(msa_repository_dir).expanduser(),
                    max_records=int(max_records),
                )
            )
        pre_input_metrics["msa_repository_missing_after_lookup"] = _run_csv_missing_target_msas(run_csv, max_records=int(max_records))
        pre_input_metrics["msa_repository_nonreal_after_lookup"] = _run_csv_nonreal_target_msas(run_csv, max_records=int(max_records))
        repository_block_count = (
            pre_input_metrics["msa_repository_nonreal_after_lookup"]
            if require_real_target_msa
            else pre_input_metrics["msa_repository_missing_after_lookup"]
        )
        if msa_source == "msa_repository" and repository_block_count:
            finish_job(job.run_dir, False, {"metrics": pre_input_metrics})
            raise RuntimeError(
                "MSA repository mode was selected, but some selected target-chain MSAs are missing or query-only. "
                "Use repository-then-AlphaFast/MMseqs to fill misses with the local MMseqs GPU pipeline."
            )
        use_alphafast_msa = msa_source == "alphafast_mmseqs_gpu" or (
            msa_source == "msa_repository_then_alphafast_mmseqs_gpu"
            and (
                pre_input_metrics["msa_repository_missing_after_lookup"] > 0
                or pre_input_metrics["msa_repository_nonreal_after_lookup"] > 0
            )
        )
        if use_alphafast_msa:
            af3_input_dir = output_dir / "AF3" / "input_folder"
            if not (af3_input_dir.exists() and any(af3_input_dir.glob("*.json"))):
                af3_input_cmd = _generate_model_inputs_command(
                    output_dir=output_dir,
                    docker_image=docker_image,
                    models=["af3"],
                )
                commands.append(af3_input_cmd)
                write_json(job.run_dir / "command.json", {"mode": "docker", "commands": commands})
                return_code = _run_docker_command(job.run_dir, af3_input_cmd)
                pre_input_metrics["shared_msa_af3_input_generation_return_code"] = return_code
                if return_code != 0:
                    finish_job(job.run_dir, False, {"metrics": pre_input_metrics})
                    raise RuntimeError(f"AF3 input generation for shared MSA search failed with return code {return_code}.")
            msa_started_at = time.monotonic()
            af3_output_dir = output_dir / "AF3" / "alphafast_output"
            if target_refolding:
                serial_metrics, serial_commands = _run_alphafast_serial_chain_msa_pipeline(
                    job_run_dir=job.run_dir,
                    run_csv=run_csv,
                    af3_input_dir=af3_input_dir,
                    output_dir=output_dir,
                    db_dir=Path(alphafast_db_dir).expanduser(),
                    batch_size=int(alphafast_batch_size),
                    gpu_device=alphafast_gpu_device,
                    max_records=int(max_records),
                    image=ALPHAFAST_IMAGE,
                    msa_repository_dir=Path(msa_repository_dir).expanduser(),
                )
                commands.extend(serial_commands)
                write_json(job.run_dir / "command.json", {"mode": "docker", "commands": commands})
                pre_input_metrics.update(serial_metrics)
                rc = 0 if int(serial_metrics.get("alphafast_serial_msa_failed_count") or 0) == 0 else 1
                pre_input_metrics["alphafast_msa_pipeline_return_code"] = rc
                pre_input_metrics["alphafast_msa_pipeline_mode"] = "serial_chain_msa"
                remaining_nonreal = _run_csv_nonreal_target_msas(run_csv, max_records=int(max_records))
                pre_input_metrics["alphafast_serial_msa_nonreal_after_generation"] = remaining_nonreal
                if remaining_nonreal > 0:
                    warning = (
                        "MSA fallback used: target refolding generated missing target MSAs per chain. "
                        "Some chains still use query-only/non-real target MSAs."
                    )
                    pre_input_metrics["target_refolding_msa_warning"] = warning
                    pre_input_metrics["require_real_target_msa_disabled_after_msa_failure"] = True
                    require_real_target_msa = False
                    alphafast_query_only_msa = True
                    update_status(job.run_dir, "running", warning=warning, warning_level="degraded")
            else:
                rc, data_pipeline_commands, af3_template_metrics = _run_alphafast_data_pipeline(
                    job_run_dir=job.run_dir,
                    run_csv=run_csv,
                    af3_input_dir=af3_input_dir,
                    output_dir=af3_output_dir,
                    db_dir=Path(alphafast_db_dir).expanduser(),
                    batch_size=int(alphafast_batch_size),
                    gpu_device=alphafast_gpu_device,
                    max_records=int(max_records),
                    image=ALPHAFAST_IMAGE,
                    use_target_templates=bool(run_alphafast_af3 and alphafast_use_target_templates),
                )
                commands.extend(data_pipeline_commands)
                write_json(job.run_dir / "command.json", {"mode": "docker", "commands": commands})
                pre_input_metrics.update(af3_template_metrics)
                pre_input_metrics["alphafast_msa_pipeline_return_code"] = rc
                if rc != 0:
                    finish_job(job.run_dir, False, {"metrics": pre_input_metrics})
                    raise RuntimeError(f"AlphaFast/MMseqs shared MSA generation failed with return code {rc}.")
            alphafast_data_pipeline_done = (not target_refolding) and rc == 0
            _record_runtime_timing(pre_input_metrics, "alphafast_msa", msa_started_at, **runtime_size)
            if rc == 0 and not target_refolding:
                pre_input_metrics.update(
                    _write_alphafast_msas_to_run_csv(
                        run_csv=run_csv,
                        alphafast_output_dir=af3_output_dir,
                        output_dir=output_dir,
                        msa_repository_dir=Path(msa_repository_dir).expanduser(),
                    )
                )
            pre_input_metrics["shared_msa_missing_after_generation"] = _run_csv_missing_target_msas(run_csv, max_records=int(max_records))
            pre_input_metrics["shared_msa_nonreal_after_generation"] = _run_csv_nonreal_target_msas(run_csv, max_records=int(max_records))
    if run_csv.exists():
        chain_msa_path = job.run_dir / "artifacts" / "benchmark" / "chain_msa_map.json"
        chain_msa_summary = _write_chain_msa_map(
            run_csv,
            chain_msa_path,
            max_records=int(max_records),
        )
        shutil.copy2(chain_msa_path, job.run_dir / "artifacts" / "benchmark" / "target_msa_manifest.json")
        pre_input_metrics.update({f"chain_msa_{key}": value for key, value in chain_msa_summary.items()})
        if msa_consuming_engines and require_real_target_msa:
            nonreal_count = int(chain_msa_summary.get("target_msa_blocking_nonreal_count") or 0)
            pre_input_metrics["shared_msa_nonreal_final_count"] = nonreal_count
            if nonreal_count > 0:
                finish_job(
                    job.run_dir,
                    False,
                    {
                        "outputs": {
                            "run_csv": str(run_csv.relative_to(job.run_dir)) if run_csv.exists() else None,
                            "chain_msa_map": "artifacts/benchmark/chain_msa_map.json",
                            "target_msa_manifest": "artifacts/benchmark/target_msa_manifest.json",
                        },
                        "metrics": pre_input_metrics,
                    },
                )
                raise RuntimeError(
                    f"{nonreal_count} selected protein target-chain MSA(s) are missing, empty, or query-only after shared MSA preparation. "
                    f"Short target chains under {TARGET_MSA_REQUIRED_MIN_LENGTH} residues are allowed without a real MSA. "
                    "Prediction was stopped before engine input generation because real target MSAs are required for protein targets."
                )
    needs_model_inputs = bool(
        generate_inputs
        and run_csv.exists()
        and (not resume_prepared_inputs or not _model_inputs_ready(output_dir, selected_models))
    )
    if needs_model_inputs:
        progress_step += 1
        _emit_benchmark_progress(
            job.run_dir,
            phase="Generating model inputs",
            engine="input adapters",
            step=progress_step,
            total_steps=progress_total,
            progress_callback=progress_callback,
        )
        model_cmd = _generate_model_inputs_command(
            output_dir=output_dir,
            docker_image=docker_image,
            models=selected_models,
        )
        commands.append(model_cmd)
        write_json(job.run_dir / "command.json", {"mode": "docker", "commands": commands})
        return_code = _run_docker_command(job.run_dir, model_cmd)
        if return_code != 0:
            finish_job(job.run_dir, False, {"metrics": {"generate_model_inputs_return_code": return_code}})
            raise RuntimeError(f"de_novo_binder_scoring generate_model_inputs failed with return code {return_code}.")
        if run_colabfold and colabfold_uses_target_msa and run_csv.exists():
            pre_input_metrics.update(
                _repair_colabfold_target_msas(
                    run_csv=run_csv,
                    output_dir=output_dir,
                    max_records=int(max_records),
                )
            )
        if resume_prepared_inputs:
            pre_input_metrics["model_inputs_regenerated_on_resume"] = True
    metric_csvs: list[Path] = []
    existing_input_rosetta = output_dir / "input_rosetta_metrics.csv"
    if existing_input_rosetta.exists():
        metric_csvs.append(existing_input_rosetta)
        pre_input_metrics["input_rosetta_resumed_from_artifacts"] = True
    if run_pyrosetta_input_metrics:
        if not existing_input_rosetta.exists():
            pre_input_metrics.update(
                _reuse_cached_input_rosetta_metrics(
                    run_csv=run_csv,
                    out_csv=existing_input_rosetta,
                    current_run_dir=job.run_dir,
                )
            )
        progress_step += 1
        _emit_benchmark_progress(
            job.run_dir,
            phase="Scoring input structures",
            engine="PyRosetta",
            step=progress_step,
            total_steps=progress_total,
            progress_callback=progress_callback,
        )
        if existing_input_rosetta.exists():
            pre_input_metrics["input_rosetta_resumed_from_artifacts"] = True
            if existing_input_rosetta not in metric_csvs:
                metric_csvs.append(existing_input_rosetta)
        else:
            input_group_roles = _benchmark_group_roles(run_csv)
            input_metric_dir, input_metric_count = _stage_input_pdbs_for_group_metrics(
                output_dir / "input_pdbs",
                job.run_dir / "artifacts" / "benchmark" / "predicted_metric_pdbs" / "input",
                input_group_roles,
            )
            pre_input_metrics["input_rosetta_evaluation_contract"] = {
                "binder_group": "declared binder chains normalized to A",
                "target_group": "all declared target chains normalized together to B",
                "source_structures_modified": False,
            }
            pre_input_metrics["input_rosetta_evaluation_pdb_count"] = input_metric_count
            if input_metric_dir is None:
                finish_job(
                    job.run_dir,
                    False,
                    {
                        "metrics": {
                            **pre_input_metrics,
                            "error": "No input structures could be normalized for PyRosetta evaluation.",
                        }
                    },
                )
                raise RuntimeError(
                    "No input structures could be normalized to binder A / combined target B "
                    "for PyRosetta evaluation."
                )
            rosetta_cmd = [
            "docker",
            "run",
            "--rm",
            *_repo_and_runs_mounts(),
            "-w",
            str(DE_NOVO_BINDER_SCORING_DIR),
            PYROSETTA_METRICS_IMAGE,
            "python",
            "./scripts/compute_rosetta_metrics.py",
            "--run-csv",
            str(output_dir / "run.csv"),
            "--out-csv",
            str(output_dir / "input_rosetta_metrics.csv"),
            "--folder",
            f"input:{input_metric_dir}",
            "--nprocs",
            str(max(1, int(pyrosetta_nprocs))),
            "--dalphaball-path",
            "./functions/DAlphaBall.gcc",
            ]
            commands.append(rosetta_cmd)
            write_json(job.run_dir / "command.json", {"mode": "docker", "commands": commands})
            return_code = _run_docker_command(job.run_dir, rosetta_cmd)
            if return_code != 0:
                finish_job(job.run_dir, False, {"metrics": {"pyrosetta_input_metrics_return_code": return_code}})
                raise RuntimeError(f"de_novo_binder_scoring PyRosetta input metrics failed with return code {return_code}.")
            metric_csvs.append(output_dir / "input_rosetta_metrics.csv")
    metrics: dict[str, Any] = {
        "process_inputs_return_code": 0,
        "input_generation_done": bool(generate_inputs),
        "pyrosetta_input_metrics_done": bool(run_pyrosetta_input_metrics),
        "available_script_image": docker_image,
        "pyrosetta_metrics_image": PYROSETTA_METRICS_IMAGE,
        "af2_initial_guess_candidate_image": AF2_INITIAL_GUESS_IMAGE,
        "boltz2_initial_guess_candidate_image": BOLTZ2_IMAGE,
        "colabfold_image": COLABFOLD_IMAGE,
        "alphafast_af3_image": ALPHAFAST_IMAGE,
    }
    metrics.update(pre_input_metrics)
    if target_refolding and run_boltzgen_fold:
        run_boltzgen_fold = False
        metrics["boltzgen_fold_skipped"] = True
        metrics["boltzgen_fold_skip_reason"] = (
            "BoltzGen Fold target-template folding requires at least one designed residue. "
            "Target-only refolding has no binder/design chain, so BoltzGen Fold was not launched."
        )
    if run_csv.exists():
        run_df = pd.read_csv(run_csv)
        metrics.update(
            {
                "record_count": int(len(run_df)),
                "positive_count": int(sum(1 for value in run_df.get("binder", []) if _truthy_label(value) == 1)),
                "negative_count": int(sum(1 for value in run_df.get("binder", []) if _truthy_label(value) == 0)),
            }
        )
    esmfold2_child_runs: list[str] = []
    af2_initial_guess_child_runs: list[str] = []
    boltz2_initial_guess_child_runs: list[str] = []
    rf3_child_runs: list[str] = []
    openfold3_child_runs: list[str] = []
    protenix_child_runs: list[str] = []
    protenix_v1_child_runs: list[str] = []
    protenix_v2_child_runs: list[str] = []
    boltzgen_fold_child_runs: list[str] = []
    esmfold2_summary: dict[str, Any] = {}
    af2_summary: dict[str, Any] = {}
    boltz2_summary: dict[str, Any] = {}
    rf3_summary: dict[str, Any] = {}
    openfold3_summary: dict[str, Any] = {}
    protenix_summary: dict[str, Any] = {}
    protenix_v1_summary: dict[str, Any] = {}
    protenix_v2_summary: dict[str, Any] = {}
    boltzgen_fold_summary: dict[str, Any] = {}
    colabfold_summary: dict[str, Any] = {}
    if run_esmfold2 and run_csv.exists():
        progress_step += 1
        _emit_benchmark_progress(
            job.run_dir,
            phase=prediction_phase,
            engine="ESMFold2",
            step=progress_step,
            total_steps=progress_total,
            progress_callback=progress_callback,
        )
        existing_table = job.run_dir / "artifacts" / "benchmark" / "esmfold2_metrics.csv"
        if resume and _csv_data_row_count(existing_table) == int(metrics.get("record_count") or 0):
            metric_csvs.append(existing_table)
            esmfold2_child_runs.extend(_completed_benchmark_child_runs(job.run_dir, "esmfold2"))
            esmfold2_summary = read_json(job.run_dir / "artifacts" / "benchmark" / "esmfold2_feature_summary.json")
            metrics["esmfold2_metrics_table"] = str(existing_table.relative_to(job.run_dir))
            metrics["esmfold2_resumed_from_artifacts"] = True
            started_at = None
        else:
            started_at = time.monotonic()
        if started_at is not None:
            esm_csv_text = _repo_run_csv_to_esmfold2_csv(job.run_dir, run_csv)
            if esm_csv_text.strip() and len(esm_csv_text.splitlines()) > 1:
                child_run = run_esmfold2_binder_benchmark(
                    input_csv_text=esm_csv_text,
                    modes=esmfold2_modes or ["initial_guess"],
                    max_records=max_records,
                    num_loops=num_loops,
                    num_sampling_steps=num_sampling_steps,
                    seed=seed,
                    device=device,
                    gpu_device=selected_gpu_device,
                    use_target_msa=bool(esmfold2_use_target_msa),
                    use_docker=True,
                    internal_parent_run_dir=job.run_dir,
                )
                esmfold2_child_runs.append(str(child_run))
                esmfold2_table, esmfold2_summary_path, esmfold2_summary = _summarize_child_candidate_metrics(
                    parent_run_dir=job.run_dir,
                    child_run_dir=child_run,
                    output_prefix="esmfold2",
                )
                if esmfold2_table is not None:
                    metrics["esmfold2_metrics_table"] = str(esmfold2_table.relative_to(job.run_dir))
                    metric_csvs.append(esmfold2_table)
                if esmfold2_summary_path is not None:
                    metrics["esmfold2_summary"] = str(esmfold2_summary_path.relative_to(job.run_dir))
            _record_runtime_timing(metrics, "esmfold2", started_at, **runtime_size)
    if run_af2_initial_guess and run_csv.exists():
        progress_step += 1
        _emit_benchmark_progress(
            job.run_dir,
            phase=prediction_phase,
            engine="AF2 initial guess",
            step=progress_step,
            total_steps=progress_total,
            progress_callback=progress_callback,
        )
        existing_table = job.run_dir / "artifacts" / "benchmark" / "af2_initial_guess_metrics.csv"
        af2_fragment_inputs_complete = _benchmark_fragment_child_inputs_complete(
            job.run_dir,
            "af2_initial_guess",
            run_csv,
        )
        if resume and _csv_data_row_count(existing_table) == int(metrics.get("record_count") or 0) and af2_fragment_inputs_complete:
            metric_csvs.append(existing_table)
            af2_initial_guess_child_runs.extend(_completed_benchmark_child_runs(job.run_dir, "af2_initial_guess"))
            af2_summary = read_json(job.run_dir / "artifacts" / "benchmark" / "af2_initial_guess_feature_summary.json")
            metrics["af2_initial_guess_metrics_table"] = str(existing_table.relative_to(job.run_dir))
            metrics["af2_initial_guess_resumed_from_artifacts"] = True
            started_at = None
        else:
            if resume and not af2_fragment_inputs_complete:
                metrics["af2_initial_guess_resume_invalidated"] = True
                metrics["af2_initial_guess_resume_invalidated_reason"] = "Existing child inputs are missing declared target fragments."
            started_at = time.monotonic()
        if started_at is not None:
            benchmark_candidates = _repo_run_csv_to_candidates(job.run_dir, run_csv, max_records=max_records)
            child_run = refolding_workflow.run_af2_initial_guess_complex_refolding(
                source_run_dir=job.run_dir,
                candidates_jsonl=benchmark_candidates,
                require_monomer_success=False,
                num_recycles=int(af2_num_recycles),
                multimer=bool(af2_multimer),
                use_initial_guess=bool(af2_use_initial_guess),
                use_binder_template=bool(af2_use_binder_template),
                use_interface_template=bool(af2_use_interface_template),
                docker_image=AF2_INITIAL_GUESS_IMAGE,
                internal_parent_run_dir=job.run_dir,
                gpu_device=selected_gpu_device,
            )
            af2_initial_guess_child_runs.append(str(child_run))
            af2_table, af2_summary_path, af2_summary = _summarize_child_candidate_metrics(
                parent_run_dir=job.run_dir,
                child_run_dir=child_run,
                output_prefix="af2_initial_guess",
            )
            if af2_table is not None:
                metrics["af2_initial_guess_metrics_table"] = str(af2_table.relative_to(job.run_dir))
                metric_csvs.append(af2_table)
            if af2_summary_path is not None:
                metrics["af2_initial_guess_summary"] = str(af2_summary_path.relative_to(job.run_dir))
            _record_runtime_timing(metrics, "af2_initial_guess", started_at, **runtime_size)
    if run_boltz2_initial_guess and run_csv.exists():
        progress_step += 1
        _emit_benchmark_progress(
            job.run_dir,
            phase=prediction_phase,
            engine="Boltz-2",
            step=progress_step,
            total_steps=progress_total,
            progress_callback=progress_callback,
        )
        existing_table = job.run_dir / "artifacts" / "benchmark" / "boltz2_initial_guess_metrics.csv"
        boltz2_fragment_inputs_complete = _benchmark_fragment_child_inputs_complete(
            job.run_dir,
            "boltz2",
            run_csv,
        )
        if resume and _csv_data_row_count(existing_table) == int(metrics.get("record_count") or 0) and boltz2_fragment_inputs_complete:
            metric_csvs.append(existing_table)
            boltz2_initial_guess_child_runs.extend(_completed_benchmark_child_runs(job.run_dir, "boltz2"))
            boltz2_summary = read_json(job.run_dir / "artifacts" / "benchmark" / "boltz2_initial_guess_feature_summary.json")
            metrics["boltz2_initial_guess_metrics_table"] = str(existing_table.relative_to(job.run_dir))
            metrics["boltz2_initial_guess_resumed_from_artifacts"] = True
            started_at = None
        else:
            if resume and not boltz2_fragment_inputs_complete:
                metrics["boltz2_initial_guess_resume_invalidated"] = True
                metrics["boltz2_initial_guess_resume_invalidated_reason"] = "Existing child inputs are missing declared target fragments."
            started_at = time.monotonic()
        if started_at is not None:
            benchmark_candidates = _repo_run_csv_to_candidates(job.run_dir, run_csv, max_records=max_records)
            child_run = refolding_workflow.run_boltz2_complex_refolding(
                source_run_dir=job.run_dir,
                candidates_jsonl=benchmark_candidates,
                require_monomer_success=False,
                use_target_template=bool(boltz2_use_target_template),
                recycling_steps=int(boltz2_recycling_steps),
                sampling_steps=int(boltz2_sampling_steps),
                diffusion_samples=int(boltz2_diffusion_samples),
                write_full_pae=bool(boltz2_write_full_pae),
                internal_parent_run_dir=job.run_dir,
                benchmark_run_csv=run_csv if boltz2_use_target_msa else None,
                gpu_device=selected_gpu_device,
            )
            boltz2_initial_guess_child_runs.append(str(child_run))
            boltz2_table, boltz2_summary_path, boltz2_summary = _summarize_child_candidate_metrics(
                parent_run_dir=job.run_dir,
                child_run_dir=child_run,
                output_prefix="boltz2_initial_guess",
            )
            if boltz2_table is not None:
                metrics["boltz2_initial_guess_metrics_table"] = str(boltz2_table.relative_to(job.run_dir))
                metric_csvs.append(boltz2_table)
            if boltz2_summary_path is not None:
                metrics["boltz2_initial_guess_summary"] = str(boltz2_summary_path.relative_to(job.run_dir))
            _record_runtime_timing(metrics, "boltz2_initial_guess", started_at, **runtime_size)
    extra_engine_specs = [
        (
            bool(run_rf3),
            "RF3",
            "rf3",
            rf3_child_runs,
            rf3_summary,
            lambda candidates: refolding_workflow.run_rf3_complex_refolding(
                source_run_dir=job.run_dir,
                candidates_jsonl=candidates,
                require_monomer_success=False,
                checkpoint_path=Path(rf3_checkpoint_path),
                use_target_msa=bool(rf3_use_target_msa),
                use_target_template=bool(rf3_use_target_template),
                n_recycles=int(rf3_recycles),
                num_steps=int(rf3_num_steps),
                diffusion_batch_size=int(rf3_diffusion_batch_size),
                seed=int(rf3_seed),
                benchmark_run_csv=run_csv if rf3_use_target_msa else None,
                internal_parent_run_dir=job.run_dir,
                gpu_device=selected_gpu_device,
            ),
        ),
        (
            bool(run_openfold3),
            "OpenFold-3",
            "openfold3",
            openfold3_child_runs,
            openfold3_summary,
            lambda candidates: refolding_workflow.run_openfold3_complex_refolding(
                source_run_dir=job.run_dir,
                candidates_jsonl=candidates,
                require_monomer_success=False,
                use_target_msa=bool(openfold3_use_target_msa),
                checkpoint_path=Path(openfold3_checkpoint_path) if openfold3_checkpoint_path else None,
                num_diffusion_samples=int(openfold3_num_diffusion_samples),
                num_model_seeds=int(openfold3_num_model_seeds),
                num_recycles=int(openfold3_num_recycles),
                use_msa_server=bool(openfold3_use_msa_server),
                benchmark_run_csv=run_csv if openfold3_use_target_msa else None,
                internal_parent_run_dir=job.run_dir,
                gpu_device=selected_gpu_device,
            ),
        ),
        (
            bool(run_protenix),
            "Protenix v0.5",
            "protenix",
            protenix_child_runs,
            protenix_summary,
            lambda candidates: refolding_workflow.run_protenix_complex_refolding(
                source_run_dir=job.run_dir,
                candidates_jsonl=candidates,
                require_monomer_success=False,
                use_msa=bool(protenix_use_msa),
                benchmark_run_csv=run_csv if protenix_use_msa else None,
                cycle=int(protenix_cycle),
                diffusion_steps=int(protenix_diffusion_steps),
                samples=int(protenix_samples),
                internal_parent_run_dir=job.run_dir,
                existing_job=_recoverable_benchmark_child_job(job.run_dir, "protenix") if resume else None,
                gpu_device=selected_gpu_device,
            ),
        ),
        (
            bool(run_protenix_v1),
            "Protenix v1",
            "protenix_v1",
            protenix_v1_child_runs,
            protenix_v1_summary,
            lambda candidates: refolding_workflow.run_protenix_cli_complex_refolding(
                source_run_dir=job.run_dir,
                candidates_jsonl=candidates,
                require_monomer_success=False,
                use_msa=bool(protenix_v1_use_msa),
                use_template=bool(protenix_v1_use_template),
                use_default_params=bool(protenix_v1_use_default_params),
                benchmark_run_csv=run_csv if protenix_v1_use_msa else None,
                cycle=int(protenix_v1_cycle),
                diffusion_steps=int(protenix_v1_diffusion_steps),
                samples=int(protenix_v1_samples),
                model_name=str(protenix_v1_model_name or refolding_workflow.PROTENIX_V1_MODEL),
                tool="protenix_v1",
                internal_parent_run_dir=job.run_dir,
                existing_job=_recoverable_benchmark_child_job(job.run_dir, "protenix_v1") if resume else None,
                gpu_device=selected_gpu_device,
            ),
        ),
        (
            bool(run_protenix_v2),
            "Protenix v2",
            "protenix_v2",
            protenix_v2_child_runs,
            protenix_v2_summary,
            lambda candidates: refolding_workflow.run_protenix_cli_complex_refolding(
                source_run_dir=job.run_dir,
                candidates_jsonl=candidates,
                require_monomer_success=False,
                use_msa=bool(protenix_v2_use_msa),
                use_template=bool(protenix_v2_use_template),
                use_default_params=bool(protenix_v2_use_default_params),
                benchmark_run_csv=run_csv if protenix_v2_use_msa else None,
                cycle=int(protenix_v2_cycle),
                diffusion_steps=int(protenix_v2_diffusion_steps),
                samples=int(protenix_v2_samples),
                model_name=str(protenix_v2_model_name or refolding_workflow.PROTENIX_V2_MODEL),
                tool="protenix_v2",
                internal_parent_run_dir=job.run_dir,
                existing_job=_recoverable_benchmark_child_job(job.run_dir, "protenix_v2") if resume else None,
                gpu_device=selected_gpu_device,
            ),
        ),
        (
            bool(run_boltzgen_fold),
            "BoltzGen target-template fold",
            "boltzgen_fold",
            boltzgen_fold_child_runs,
            boltzgen_fold_summary,
            lambda candidates: refolding_workflow.run_boltzgen_fold_complex_refolding(
                source_run_dir=job.run_dir,
                candidates_jsonl=candidates,
                require_monomer_success=False,
                recycling_steps=int(boltzgen_recycling_steps),
                sampling_steps=int(boltzgen_sampling_steps),
                diffusion_samples=int(boltzgen_diffusion_samples),
                internal_parent_run_dir=job.run_dir,
                gpu_device=selected_gpu_device,
            ),
        ),
    ]
    for enabled, display_name, engine_key, child_runs, summary_target, runner in extra_engine_specs:
        if not enabled or not run_csv.exists():
            continue
        progress_step += 1
        _emit_benchmark_progress(
            job.run_dir,
            phase=prediction_phase,
            engine=display_name,
            step=progress_step,
            total_steps=progress_total,
            progress_callback=progress_callback,
        )
        existing_table = job.run_dir / "artifacts" / "benchmark" / f"{engine_key}_metrics.csv"
        fragment_inputs_complete = _benchmark_fragment_resume_complete(job.run_dir, engine_key, run_csv)
        if resume and _csv_data_row_count(existing_table) == int(metrics.get("record_count") or 0) and fragment_inputs_complete:
            metric_csvs.append(existing_table)
            child_runs.extend(_completed_benchmark_child_runs(job.run_dir, engine_key))
            summary_target.update(read_json(job.run_dir / "artifacts" / "benchmark" / f"{engine_key}_feature_summary.json"))
            metrics[f"{engine_key}_metrics_table"] = str(existing_table.relative_to(job.run_dir))
            metrics[f"{engine_key}_resumed_from_artifacts"] = True
            continue
        if resume and not fragment_inputs_complete:
            metrics[f"{engine_key}_resume_invalidated"] = True
            metrics[f"{engine_key}_resume_invalidated_reason"] = "Existing child inputs are missing declared target fragments."
        started_at = time.monotonic()
        benchmark_candidates = _repo_run_csv_to_candidates(job.run_dir, run_csv, max_records=max_records)
        child_run = runner(benchmark_candidates)
        child_runs.append(str(child_run))
        child_result = read_json(Path(child_run) / "result.json")
        child_metrics = (
            child_result.get("metrics")
            if isinstance(child_result.get("metrics"), dict)
            else {}
        )
        child_candidate_count = int(child_metrics.get("candidate_count") or 0)
        if child_result.get("success") is not True or child_candidate_count <= 0:
            return_code = child_metrics.get("return_code")
            failure_message = (
                f"{display_name} produced no predicted structures"
                if child_candidate_count <= 0
                else f"{display_name} subrun failed"
            )
            if return_code not in (None, ""):
                failure_message += f" with return code {return_code}"
            metrics[f"{engine_key}_return_code"] = return_code
            metrics[f"{engine_key}_candidate_count"] = child_candidate_count
            metrics[f"{engine_key}_child_run"] = str(child_run)
            metrics["worker_error"] = failure_message
            finish_job(job.run_dir, False, {"metrics": metrics})
            raise RuntimeError(failure_message)
        table, summary_path, summary = _summarize_child_candidate_metrics(
            parent_run_dir=job.run_dir,
            child_run_dir=child_run,
            output_prefix=engine_key,
        )
        summary_target.update(summary)
        if table is not None:
            metrics[f"{engine_key}_metrics_table"] = str(table.relative_to(job.run_dir))
            metric_csvs.append(table)
        if summary_path is not None:
            metrics[f"{engine_key}_summary"] = str(summary_path.relative_to(job.run_dir))
        _record_runtime_timing(metrics, engine_key, started_at, **runtime_size)
    if run_colabfold and run_csv.exists():
        progress_step += 1
        _emit_benchmark_progress(
            job.run_dir,
            phase=prediction_phase,
            engine="ColabFold",
            step=progress_step,
            total_steps=progress_total,
            progress_callback=progress_callback,
        )
        existing_table = job.run_dir / "artifacts" / "benchmark" / "colabfold_metrics.csv"
        if resume and _csv_data_row_count(existing_table) == int(metrics.get("record_count") or 0):
            metric_csvs.append(existing_table)
            colabfold_summary = read_json(job.run_dir / "artifacts" / "benchmark" / "colabfold_feature_summary.json")
            metrics["colabfold_metrics_table"] = str(existing_table.relative_to(job.run_dir))
            metrics["colabfold_resumed_from_artifacts"] = True
            colabfold_started_at = None
        else:
            colabfold_started_at = time.monotonic()
        if colabfold_started_at is not None:
            metrics["colabfold_msa_source"] = msa_source
            metrics["colabfold_use_target_msa"] = bool(colabfold_uses_target_msa)
            metrics["colabfold_use_target_templates"] = bool(colabfold_use_target_templates)
            metrics["colabfold_max_template_hits"] = int(colabfold_max_template_hits)
            rc, selected_count, template_count, colabfold_commands = _run_colabfold_prediction(
                job_run_dir=job.run_dir,
                run_csv=run_csv,
                output_dir=output_dir,
                cache_dir=Path(colabfold_cache_dir).expanduser(),
                num_recycles=int(colabfold_num_recycles),
                num_models=int(colabfold_num_models),
                gpu_device=colabfold_gpu_device,
                max_records=int(max_records),
                use_target_templates=bool(colabfold_use_target_templates),
                max_template_hits=int(colabfold_max_template_hits),
                image=COLABFOLD_IMAGE,
            )
            commands.extend(colabfold_commands)
            write_json(job.run_dir / "command.json", {"mode": "docker", "commands": commands})
            metrics["colabfold_return_code"] = rc
            metrics["colabfold_selected_count"] = selected_count
            metrics["colabfold_target_template_count"] = template_count
            metrics["colabfold_output_dir"] = str((output_dir / "ColabFold" / "ptm_output").relative_to(job.run_dir))
            if rc != 0:
                finish_job(job.run_dir, False, {"metrics": metrics})
                raise RuntimeError(f"ColabFold benchmark failed with return code {rc}.")
            prediction_dir = output_dir / "ColabFold" / "ptm_output"
            prediction_files = [
                path
                for pattern in ("**/*.pdb", "**/*.cif", "**/*.mmcif")
                for path in prediction_dir.glob(pattern)
                if path.is_file()
            ]
            metrics["colabfold_prediction_count"] = len(prediction_files)
            if not prediction_files:
                message = (
                    "ColabFold completed without producing a predicted PDB/mmCIF structure. "
                    "Inspect the ColabFold log for input-feature or template errors."
                )
                metrics["worker_error"] = message
                finish_job(job.run_dir, False, {"metrics": metrics})
                raise RuntimeError(message)
            colab_table, colab_summary_path, colabfold_summary, extract_commands = _extract_and_summarize_colabfold_outputs(
                parent_run_dir=job.run_dir,
                run_csv=run_csv,
                output_dir=output_dir,
                docker_image=docker_image,
            )
            commands.extend(extract_commands)
            write_json(job.run_dir / "command.json", {"mode": "docker", "commands": commands})
            if colab_table is not None:
                metrics["colabfold_metrics_table"] = str(colab_table.relative_to(job.run_dir))
                metric_csvs.append(colab_table)
            if colab_summary_path is not None:
                metrics["colabfold_summary"] = str(colab_summary_path.relative_to(job.run_dir))
            _record_runtime_timing(metrics, "colabfold", colabfold_started_at, **runtime_size)
    alphafast_commands: list[list[str]] = []
    if run_alphafast_af3:
        progress_step += 1
        _emit_benchmark_progress(
            job.run_dir,
            phase=prediction_phase,
            engine="AlphaFast AF3",
            step=progress_step,
            total_steps=progress_total,
            progress_callback=progress_callback,
        )
        existing_table = job.run_dir / "artifacts" / "benchmark" / "alphafast_af3_metrics.csv"
        if resume and _csv_data_row_count(existing_table) == int(metrics.get("record_count") or 0):
            metric_csvs.append(existing_table)
            metrics["alphafast_af3_metrics_table"] = str(existing_table.relative_to(job.run_dir))
            metrics["alphafast_af3_resumed_from_artifacts"] = True
            started_at = None
        else:
            started_at = time.monotonic()
        if started_at is not None:
            af3_input_dir = output_dir / "AF3" / "input_folder"
            af3_output_dir = output_dir / "AF3" / "alphafast_output"
            rc, alphafast_commands, af3_template_metrics = _run_alphafast_af3_refolding(
                job_run_dir=job.run_dir,
                run_csv=run_csv,
                af3_input_dir=af3_input_dir,
                output_dir=af3_output_dir,
                db_dir=Path(alphafast_db_dir).expanduser(),
                weights_dir=Path(alphafast_weights_dir).expanduser(),
                batch_size=int(alphafast_batch_size),
                num_recycles=int(alphafast_num_recycles),
                gpu_device=alphafast_gpu_device,
                max_records=int(max_records),
                image=ALPHAFAST_IMAGE,
                run_data_pipeline=not alphafast_data_pipeline_done,
                use_target_templates=bool(alphafast_use_target_templates),
                query_only_msa=bool(alphafast_query_only_msa),
            )
            commands.extend(alphafast_commands)
            write_json(job.run_dir / "command.json", {"mode": "docker", "commands": commands})
            metrics["alphafast_af3_return_code"] = rc
            metrics["alphafast_use_target_templates"] = bool(alphafast_use_target_templates)
            metrics.update(af3_template_metrics)
            metrics["alphafast_af3_output_dir"] = str(af3_output_dir.relative_to(job.run_dir))
            if rc != 0:
                finish_job(job.run_dir, False, {"metrics": metrics})
                raise RuntimeError(f"AlphaFast AF3 benchmark failed with return code {rc}.")
            af3_table, af3_summary_path, af3_summary = _summarize_alphafast_af3_outputs(
                parent_run_dir=job.run_dir,
                run_csv=run_csv,
                alphafast_output_dir=af3_output_dir,
            )
            if af3_table is None:
                message = (
                    "AlphaFast AF3 completed without producing summary confidence files. "
                    "Inspect stdout.log/stderr.log for the internal AF3 error."
                )
                metrics["alphafast_af3_missing_prediction_count"] = int(metrics.get("record_count") or 0)
                metrics["worker_error"] = message
                finish_job(job.run_dir, False, {"metrics": metrics})
                raise RuntimeError(message)
            if af3_table is not None:
                metrics["alphafast_af3_metrics_table"] = str(af3_table.relative_to(job.run_dir))
                metric_csvs.append(af3_table)
            if af3_summary_path is not None:
                metrics["alphafast_af3_summary"] = str(af3_summary_path.relative_to(job.run_dir))
            _record_runtime_timing(metrics, "alphafast_af3", started_at, **runtime_size)
    progress_step += 1
    _emit_benchmark_progress(
        job.run_dir,
        phase=metrics_phase,
        engine="Rosetta / PyMOL / interface metrics",
        step=progress_step,
        total_steps=progress_total,
        progress_callback=progress_callback,
    )
    started_at = time.monotonic()
    try:
        postprocess_metrics, postprocess_commands = _run_benchmark_metric_postprocessing(
            job_run_dir=job.run_dir,
            run_csv=run_csv,
            output_dir=output_dir,
            metric_csvs=metric_csvs,
            esmfold2_child_runs=esmfold2_child_runs,
            af2_child_runs=af2_initial_guess_child_runs,
            boltz2_child_runs=boltz2_initial_guess_child_runs,
            extra_child_runs={
                "rf3": rf3_child_runs,
                "openfold3": openfold3_child_runs,
                "protenix": protenix_child_runs,
                "protenix_v1": protenix_v1_child_runs,
                "protenix_v2": protenix_v2_child_runs,
                "boltzgen_fold": boltzgen_fold_child_runs,
            },
            run_common_interface_metrics=bool(run_common_interface_metrics),
            run_predicted_rosetta_metrics=bool(run_predicted_rosetta_metrics),
            run_pymol_metrics=bool(run_pymol_metrics),
            pyrosetta_nprocs=int(pyrosetta_nprocs),
            docker_image=docker_image,
            resume=bool(resume),
        )
    except Exception as exc:
        metrics["benchmark_postprocessing_error"] = str(exc)
        metrics["benchmark_postprocessing_exception_type"] = exc.__class__.__name__
        metrics["benchmark_postprocessing_phase"] = "Rosetta / PyMOL / interface metrics"
        finish_job(
            job.run_dir,
            False,
            {
                "outputs": {
                    "run_csv": str(run_csv.relative_to(job.run_dir)) if run_csv.exists() else None,
                    "output_dir": str(output_dir.relative_to(job.run_dir)),
                },
                "metrics": metrics,
            },
        )
        raise
    metrics.update(postprocess_metrics)
    _record_runtime_timing(metrics, "postprocessing", started_at, **runtime_size)
    if postprocess_commands:
        commands.extend(postprocess_commands)
        write_json(job.run_dir / "command.json", {"mode": "docker", "commands": commands})
    missing_images = {
        "af2_initial_guess": f"Available through the app refolding adapter using {AF2_INITIAL_GUESS_IMAGE}; benchmark wiring reuses run_af2_initial_guess_complex_refolding.",
        "colabfold": f"Available after building containers/colabfold/Dockerfile as {COLABFOLD_IMAGE}; it runs generated ColabFold/input_folder A3M files directly and uses {COLABFOLD_CACHE_DIR} for weights.",
        "boltz2_initial_guess": f"Available through the app refolding adapter using {BOLTZ2_IMAGE}; benchmark wiring reuses run_boltz2_complex_refolding.",
        "af3": f"AlphaFast AF3 benchmark is wired through {ALPHAFAST_IMAGE}. It requires generated AF3/input_folder JSONs, an AlphaFast database directory, and AF3 weights.",
        "pyrosetta_rosetta_metrics": "Available now via ovo-bindcraft:latest for compute_rosetta_metrics.py. RMSD should also be possible because PyRosetta imports, but still needs a workflow toggle/output merge.",
        "pymol_metrics": "Available through pymol-open-source in the mn-protein-design conda environment; benchmark wiring stages input/predicted PDB folders and runs pymol_metrics.py.",
    }
    write_json(job.run_dir / "artifacts" / "benchmark" / "missing_external_tool_images.json", missing_images)
    engine_artifacts = _canonicalize_benchmark_engine_artifacts(
        parent_run_dir=job.run_dir,
        output_dir=output_dir,
        esmfold2_child_runs=esmfold2_child_runs,
        af2_child_runs=af2_initial_guess_child_runs,
        boltz2_child_runs=boltz2_initial_guess_child_runs,
        extra_child_runs={
            "rf3": rf3_child_runs,
            "openfold3": openfold3_child_runs,
            "protenix": protenix_child_runs,
            "protenix_v1": protenix_v1_child_runs,
            "protenix_v2": protenix_v2_child_runs,
            "boltzgen_fold": boltzgen_fold_child_runs,
        },
    )
    metrics["artifact_layout_version"] = "benchmark.single_run.v1"
    metrics["engine_artifacts"] = engine_artifacts
    if "alphafast_af3" in engine_artifacts:
        metrics["alphafast_af3_output_dir"] = f"{engine_artifacts['alphafast_af3']}/alphafast_output"
    if "colabfold" in engine_artifacts:
        metrics["colabfold_output_dir"] = f"{engine_artifacts['colabfold']}/ptm_output"
    if colabfold_summary and metrics.get("colabfold_metrics_table"):
        colabfold_summary["metrics_csv"] = metrics["colabfold_metrics_table"]
    outputs = {
        "run_csv": str(run_csv.relative_to(job.run_dir)) if run_csv.exists() else None,
        "output_dir": str(output_dir.relative_to(job.run_dir)),
        "missing_external_tool_images": "artifacts/benchmark/missing_external_tool_images.json",
        "chain_msa_map": "artifacts/benchmark/chain_msa_map.json"
        if (job.run_dir / "artifacts" / "benchmark" / "chain_msa_map.json").exists()
        else None,
        "target_msa_manifest": "artifacts/benchmark/target_msa_manifest.json"
        if (job.run_dir / "artifacts" / "benchmark" / "target_msa_manifest.json").exists()
        else None,
        "artifact_layout_version": "benchmark.single_run.v1",
        "engine_artifacts": engine_artifacts,
    }
    if af2_summary:
        outputs["af2_initial_guess_summary_metrics"] = af2_summary
    if esmfold2_summary:
        outputs["esmfold2_summary_metrics"] = esmfold2_summary
    if boltz2_summary:
        outputs["boltz2_initial_guess_summary_metrics"] = boltz2_summary
    if rf3_summary:
        outputs["rf3_summary_metrics"] = rf3_summary
    if protenix_summary:
        outputs["protenix_summary_metrics"] = protenix_summary
    if protenix_v1_summary:
        outputs["protenix_v1_summary_metrics"] = protenix_v1_summary
    if protenix_v2_summary:
        outputs["protenix_v2_summary_metrics"] = protenix_v2_summary
    if boltzgen_fold_summary:
        outputs["boltzgen_fold_summary_metrics"] = boltzgen_fold_summary
    if colabfold_summary:
        outputs["colabfold_summary_metrics"] = colabfold_summary
    if (output_dir / "Binder_seq.fasta").exists():
        outputs["binder_fasta"] = str((output_dir / "Binder_seq.fasta").relative_to(job.run_dir))
    if (output_dir / "input_rosetta_metrics.csv").exists():
        outputs["input_rosetta_metrics"] = str((output_dir / "input_rosetta_metrics.csv").relative_to(job.run_dir))
    if metrics.get("predicted_rosetta_metrics_table"):
        outputs["predicted_rosetta_metrics"] = metrics.get("predicted_rosetta_metrics_table")
    if metrics.get("pymol_metrics_tables"):
        outputs["pymol_metrics"] = metrics.get("pymol_metrics_tables")
    if metrics.get("common_interface_metrics_table"):
        outputs["common_interface_metrics"] = metrics.get("common_interface_metrics_table")
    if metrics.get("esmfold2_common_interface_metrics_table"):
        outputs["esmfold2_common_interface_metrics"] = metrics.get("esmfold2_common_interface_metrics_table")
    if metrics.get("af2_common_interface_metrics_table"):
        outputs["af2_common_interface_metrics"] = metrics.get("af2_common_interface_metrics_table")
    if metrics.get("rf3_common_interface_metrics_table"):
        outputs["rf3_common_interface_metrics"] = metrics.get("rf3_common_interface_metrics_table")
    if metrics.get("protenix_common_interface_metrics_table"):
        outputs["protenix_common_interface_metrics"] = metrics.get("protenix_common_interface_metrics_table")
    if metrics.get("protenix_v1_common_interface_metrics_table"):
        outputs["protenix_v1_common_interface_metrics"] = metrics.get("protenix_v1_common_interface_metrics_table")
    if metrics.get("protenix_v2_common_interface_metrics_table"):
        outputs["protenix_v2_common_interface_metrics"] = metrics.get("protenix_v2_common_interface_metrics_table")
    if metrics.get("boltzgen_fold_common_interface_metrics_table"):
        outputs["boltzgen_fold_common_interface_metrics"] = metrics.get("boltzgen_fold_common_interface_metrics_table")
    if metrics.get("merged_benchmark_metrics"):
        outputs["merged_benchmark_metrics"] = metrics.get("merged_benchmark_metrics")
    if metrics.get("merged_benchmark_feature_ranking"):
        outputs["merged_benchmark_feature_ranking"] = metrics.get("merged_benchmark_feature_ranking")
    if "alphafast_af3" in engine_artifacts:
        outputs["alphafast_af3_output"] = f"{engine_artifacts['alphafast_af3']}/alphafast_output"
    if "colabfold" in engine_artifacts:
        outputs["colabfold_output"] = f"{engine_artifacts['colabfold']}/ptm_output"
    finish_job(
        job.run_dir,
        True,
        {
            "outputs": outputs,
            "metrics": metrics,
            "downstream_artifacts": {
                "run_csv": outputs.get("run_csv"),
                "output_dir": outputs.get("output_dir"),
                "chain_msa_map": outputs.get("chain_msa_map"),
                "engine_artifacts": engine_artifacts,
                "colabfold_output": outputs.get("colabfold_output"),
                "alphafast_af3_output": outputs.get("alphafast_af3_output"),
                "merged_benchmark_metrics": outputs.get("merged_benchmark_metrics"),
                "merged_benchmark_feature_ranking": outputs.get("merged_benchmark_feature_ranking"),
            },
        },
    )
    return job.run_dir


def _run_esmfold2_benchmark_worker(
    *,
    run_dir: Path,
    records_json: Path,
    modes: list[str],
    num_loops: int,
    num_sampling_steps: int,
    seed: int,
    device: str,
    contact_cutoff: float,
    use_target_msa: bool,
) -> None:
    records = json.loads(records_json.read_text())
    esm_binder_workflow._ensure_esm_import_path()
    from esm.models.esmfold2 import (
        DistogramConditioning,
        ESMFold2InputBuilder,
        ProteinInput,
        StructurePredictionInput,
    )
    from esm.utils.msa import MSA

    model = esm_binder_workflow._load_esmfold2_model(device)
    builder = ESMFold2InputBuilder(ccd_cache=esm_binder_workflow.ESMFOLD2_MODEL_DIR)
    raw_root = run_dir / "artifacts" / "raw" / "esmfold2_benchmark"
    raw_root.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    fold_index = 0
    with (run_dir / "stdout.log").open("a") as stdout:
        for record in records:
            source_candidate = {
                "candidate_id": record["candidate_id"],
                "target_pdb": record.get("target_pdb"),
                "complex_pdb": record.get("complex_pdb"),
                "binder_pdb": record.get("binder_pdb"),
                "binder_sequence": record.get("binder_sequence"),
                "binder_chains": record.get("binder_chains") or ["A"],
                "target_chains": record.get("target_chains") or [],
                "hotspots": record.get("hotspots") or [],
                "stage": STAGE_BENCHMARK,
            }
            target_pdb = Path(str(record.get("target_pdb") or ""))
            if not target_pdb.exists():
                raise ValueError(f"Benchmark record {record['candidate_id']} has no readable target_pdb.")
            source_run_dir = run_dir
            binder_sequence = str(record.get("binder_sequence") or "").strip()
            if not binder_sequence:
                binder_sequence = refolding_workflow._candidate_binder_sequence(source_run_dir, source_candidate)
            declared_target_chains = list(record.get("target_chains") or [])
            target_only = bool(record.get("capacity_target_only")) or str(record.get("target_source") or "").lower() == "capacity_target_only"
            if target_only:
                target_only_chains = (
                    _split_list(record.get("target_chains"))
                    or _split_list(record.get("target_only_chains"))
                    or _split_list(record.get("legacy_capacity_binder_chains"))
                    or _split_list(record.get("binder_chains"))
                    or ["A"]
                )
                pdb_sequences = refolding_workflow._sequences_by_chain(target_pdb)
                target_sequences = {
                    chain: str(pdb_sequences.get(chain) or "").strip()
                    for chain in target_only_chains
                    if str(pdb_sequences.get(chain) or "").strip()
                }
                residue_maps = {}
            else:
                raw_input = record.get("raw_input") if isinstance(record.get("raw_input"), dict) else {}
                declared_target_sequences = {
                    chain: str(raw_input.get(f"target_subchain_{chain}_seq") or "").strip()
                    for chain in declared_target_chains
                }
                declared_target_sequences = {
                    chain: sequence
                    for chain, sequence in declared_target_sequences.items()
                    if chain and sequence
                }
                if declared_target_sequences and all(chain in declared_target_sequences for chain in declared_target_chains):
                    target_sequences = {chain: declared_target_sequences[chain] for chain in declared_target_chains}
                    residue_maps = {}
                    fragment_specs = refolding_workflow._target_fragment_specs_from_declared_sequences(
                        target_pdb,
                        declared_target_sequences,
                        declared_target_chains,
                    )
                    if fragment_specs:
                        staged_target_dir = raw_root / "staged_targets"
                        staged_target_dir.mkdir(parents=True, exist_ok=True)
                        staged_target_pdb = staged_target_dir / f"{_safe_id(record['candidate_id'])}_target_fragments.pdb"
                        target_lines: list[str] = []
                        next_atom = 1
                        for spec in fragment_specs:
                            residue_range = (
                                (int(spec["start"]), int(spec["end"]))
                                if spec.get("is_fragment")
                                else None
                            )
                            chain_lines, next_atom = refolding_workflow._renumber_structure_chain(
                                target_pdb,
                                str(spec["engine_chain"]),
                                next_atom,
                                {str(spec["source_chain"])},
                                residue_range,
                            )
                            if chain_lines:
                                target_lines.extend(chain_lines + ["TER"])
                        if target_lines:
                            staged_target_pdb.write_text("\n".join(target_lines + ["END", ""]))
                            target_pdb = staged_target_pdb
                else:
                    target_sequences, residue_maps = esm_binder_workflow._target_sequences(target_pdb, declared_target_chains)
            target_chains = list(target_sequences)
            binder_chain = (
                _split_list(record.get("binder_chain"))
                or _split_list(record.get("binder_chains"))
                or ["Z"]
            )[0]
            mapped_hotspots = esm_binder_workflow._mapped_hotspots(record.get("hotspots") or [], residue_maps)
            record_msa_paths = record.get("msa_paths") if use_target_msa and isinstance(record.get("msa_paths"), dict) else {}
            target_msas: dict[str, Any] = {}
            msa_notes: list[str] = []
            for target_chain, target_sequence in target_sequences.items():
                msa, note = _load_esmfold2_msa(MSA, record_msa_paths.get(target_chain), target_sequence)
                if msa is not None:
                    target_msas[target_chain] = msa
                if note:
                    msa_notes.append(f"{target_chain}:{note}")

            for mode in modes:
                use_initial_guess = mode == "initial_guess"
                fold_index += 1
                mode_label = "ig" if use_initial_guess else "seq"
                candidate_id = f"{record['candidate_id']}_esmfold2bm_{mode_label}"
                mode_dir = raw_root / mode_label
                mode_dir.mkdir(parents=True, exist_ok=True)
                distogram_conditioning = None
                initial_guess_note = None
                if use_initial_guess:
                    distogram_conditioning = []
                    for target_chain, target_sequence in target_sequences.items():
                        distogram, note = refolding_workflow._chain_initial_guess_distogram(
                            target_pdb,
                            target_chain,
                            len(target_sequence),
                        )
                        if distogram is not None:
                            distogram_conditioning.append(DistogramConditioning(chain_id=target_chain, distogram=distogram))
                        elif note and initial_guess_note is None:
                            initial_guess_note = note
                    initial_guess_note = "target distogram conditioning applied" if distogram_conditioning else initial_guess_note
                stdout.write(f"Folding benchmark {candidate_id} mode={mode}\n")
                stdout.flush()
                spi = StructurePredictionInput(
                    sequences=[
                        *[
                            ProteinInput(id=chain, sequence=sequence, msa=target_msas.get(chain))
                            for chain, sequence in target_sequences.items()
                        ],
                        *([] if target_only else [ProteinInput(id=binder_chain, sequence=binder_sequence, msa=None)]),
                    ],
                    distogram_conditioning=distogram_conditioning,
                )
                result = builder.fold(
                    model,
                    spi,
                    num_loops=int(num_loops),
                    num_sampling_steps=int(num_sampling_steps),
                    num_diffusion_samples=1,
                    seed=int(seed) + fold_index - 1,
                    complex_id=candidate_id,
                )
                complex_path = mode_dir / f"{candidate_id}.cif"
                complex_path.write_text(result.complex.to_mmcif())
                confidence_metrics, confidence_analysis = refolding_workflow._esmfold2_confidence_analysis(
                    result,
                    binder_chain=binder_chain,
                    target_chains=target_chains,
                    contact_cutoff=float(contact_cutoff),
                    output_prefix=mode_dir / candidate_id,
                )
                hotspot_metrics = esm_binder_workflow._hotspot_metrics_from_complex(
                    result.complex,
                    binder_chain=binder_chain,
                    target_chains=target_chains,
                    mapped_hotspots=mapped_hotspots,
                    contact_cutoff=float(contact_cutoff),
                )
                metrics = {
                    "benchmark_backend": "esmfold2",
                    "benchmark_mode": mode,
                    "complex_refolding_backend": f"esmfold2_benchmark_{mode}",
                    "label": record.get("label"),
                    "iptm": float(result.iptm) if result.iptm is not None else None,
                    "ptm": float(result.ptm) if result.ptm is not None else None,
                    "plddt_mean": esm_binder_workflow._mean_plddt(result),
                    "binder_length": 0 if target_only else len(binder_sequence),
                    "target_length": sum(len(sequence) for sequence in target_sequences.values()) if target_only else None,
                    "initial_guess_used": bool(distogram_conditioning),
                    "initial_guess_note": initial_guess_note,
                    "target_msa_enabled": bool(use_target_msa),
                    "target_msa_count": len(target_msas),
                    "target_msa_notes": ";".join(msa_notes) if msa_notes else None,
                    **confidence_metrics,
                    **hotspot_metrics,
                }
                metrics["esmfold2_benchmark_score"] = _score_record_metrics(metrics)
                candidate = {
                    "candidate_id": candidate_id,
                    "stage": STAGE_BENCHMARK,
                    "source_tool": "esmfold2_benchmark",
                    "tool": "esmfold2_benchmark",
                    "target_pdb": str(target_pdb),
                    "complex_pdb": str(complex_path.relative_to(run_dir)),
                    "binder_sequence": binder_sequence,
                    "target_chains": target_chains,
                    "binder_chains": [] if target_only else [binder_chain],
                    "legacy_capacity_binder_chains": target_chains if target_only else [],
                    "target_only": bool(target_only),
                    "hotspots": list(record.get("hotspots") or []),
                    "binder_length": "0" if target_only else str(len(binder_sequence)),
                    "metrics": metrics,
                    "parents": [str(record["candidate_id"])],
                    "raw_metadata": {
                        "benchmark_record": record,
                        "confidence": confidence_analysis,
                        "confidence_json": confidence_metrics.get("esmfold2_confidence_json"),
                        "confidence_arrays": confidence_metrics.get("esmfold2_confidence_arrays"),
                        "capacity_target_only": bool(target_only),
                        "staged_target_chains": target_chains if target_only else [],
                        "biological_target_chains": target_chains if target_only else [],
                        "legacy_capacity_binder_chains": target_chains if target_only else [],
                        "pae_path": str((mode_dir / str(confidence_metrics.get("esmfold2_pae_json") or "")).relative_to(run_dir))
                        if confidence_metrics.get("esmfold2_pae_json")
                        else None,
                    },
                }
                candidates.append(candidate)
                table_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "parent_id": record["candidate_id"],
                        "target_id": record.get("target_id"),
                        "source": record.get("source"),
                        "label": record.get("label"),
                        "label_raw": record.get("label_raw"),
                        "mode": mode,
                        **{key: value for key, value in metrics.items() if isinstance(value, (str, int, float, bool)) or value is None},
                    }
                )

    candidates.sort(key=lambda candidate: candidate["metrics"].get("esmfold2_benchmark_score") or 0.0, reverse=True)
    for rank, candidate in enumerate(candidates, start=1):
        candidate["metrics"]["esmfold2_benchmark_rank"] = rank
    normalized = write_candidates(run_dir, "esmfold2_benchmark", candidates)
    table = pd.DataFrame(table_rows).sort_values("esmfold2_benchmark_score", ascending=False)
    analysis_dir = run_dir / "artifacts" / "benchmark"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    table_path = analysis_dir / "benchmark_table.csv"
    summary_path = analysis_dir / "benchmark_summary.json"
    table.to_csv(table_path, index=False)
    summary = _benchmark_summary(table.to_dict(orient="records"))
    write_json(summary_path, summary)
    finish_job(
        run_dir,
        bool(normalized),
        {
            "outputs": {
                "benchmark_table": str(table_path.relative_to(run_dir)),
                "benchmark_summary": str(summary_path.relative_to(run_dir)),
                "candidates": normalized,
            },
            "metrics": summary,
            "downstream_artifacts": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
                "benchmark_table": str(table_path.relative_to(run_dir)),
            },
        },
    )


def run_esmfold2_binder_benchmark(
    *,
    input_csv: Path | None = None,
    input_csv_text: str | None = None,
    modes: list[str] | None = None,
    max_records: int = 0,
    num_loops: int = 3,
    num_sampling_steps: int = 32,
    seed: int = 0,
    device: str = "cuda",
    contact_cutoff: float = 8.0,
    use_target_msa: bool = False,
    use_docker: bool = True,
    internal_parent_run_dir: Path | None = None,
    gpu_device: object = "0",
) -> Path:
    selected_modes = [mode for mode in (modes or ["sequence", "initial_guess"]) if mode in {"sequence", "initial_guess"}]
    if not selected_modes:
        raise ValueError("Select at least one benchmark mode.")
    job = create_job(
        BENCHMARK_GROUP,
        "binder_benchmark",
        "esmfold2_benchmark",
        {"input_csv": str(input_csv) if input_csv else None},
        {
            "modes": selected_modes,
            "max_records": max_records,
            "num_loops": num_loops,
            "num_sampling_steps": num_sampling_steps,
            "seed": seed,
            "device": device,
            "gpu_device": normalize_gpu_device(gpu_device),
            "contact_cutoff": contact_cutoff,
            "use_target_msa": use_target_msa,
            "use_docker": use_docker,
            "image": ESMFOLD2_IMAGE,
        },
    )
    if internal_parent_run_dir is not None:
        mark_internal_job(
            job.run_dir,
            parent_run_dir=Path(internal_parent_run_dir),
            parent_task_group=BENCHMARK_GROUP,
            role="benchmark_engine_subrun",
            engine="esmfold2",
        )
    raw_dir = job.run_dir / "artifacts" / "raw" / "inputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    staged_csv = raw_dir / "input.csv"
    csv_base_dir = raw_dir
    if input_csv_text is not None:
        staged_csv.write_text(input_csv_text)
    elif input_csv is not None:
        source_csv = Path(input_csv).expanduser().resolve()
        csv_base_dir = source_csv.parent
        shutil.copy2(source_csv, staged_csv)
    else:
        raise ValueError("Provide input_csv or input_csv_text.")
    records = _prepare_benchmark_records(staged_csv, raw_dir, max_records=max_records, base_dir=csv_base_dir)
    records_json = raw_dir / "benchmark_records.json"
    records_json.write_text(json.dumps(records, indent=2) + "\n")
    update_status(job.run_dir, "running")

    if use_docker:
        command = [
            "docker",
            "run",
            "--rm",
            *docker_gpu_args(gpu_device),
            "--shm-size=64G",
            *_repo_and_runs_mounts(),
            "-v",
            f"{BIOHUB_ESM_ROOT}:{BIOHUB_ESM_ROOT}:ro",
            "-v",
            "/mnt/db/reference_files:/mnt/db/reference_files:ro",
            "-w",
            str(REPO_ROOT),
            "-e",
            f"PYTHONPATH={REPO_ROOT}:{REPO_ROOT / 'tools_to_implement' / 'esm'}",
            ESMFOLD2_IMAGE,
            "python",
            "-m",
            "mn_protein_design.workflows.benchmark",
            "worker",
            "--run-dir",
            str(job.run_dir),
            "--records-json",
            str(records_json),
            "--modes",
            ",".join(selected_modes),
            "--num-loops",
            str(num_loops),
            "--num-sampling-steps",
            str(num_sampling_steps),
            "--seed",
            str(seed),
            "--device",
            device,
            "--contact-cutoff",
            str(contact_cutoff),
            "--use-target-msa",
            "1" if use_target_msa else "0",
        ]
        write_json(job.run_dir / "command.json", {"mode": "docker", "command": command})
        with (job.run_dir / "stdout.log").open("a") as stdout, (job.run_dir / "stderr.log").open("a") as stderr:
            stdout.write(f"$ {' '.join(command)}\n")
            stdout.flush()
            proc = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        if proc.returncode != 0:
            finish_job(job.run_dir, False, {"metrics": {"return_code": int(proc.returncode)}})
            raise RuntimeError(f"ESMFold2 benchmark Docker job failed with return code {proc.returncode}.")
    else:
        _run_esmfold2_benchmark_worker(
            run_dir=job.run_dir,
            records_json=records_json,
            modes=selected_modes,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            seed=seed,
            device=device,
            contact_cutoff=contact_cutoff,
            use_target_msa=use_target_msa,
        )
    return job.run_dir


def _worker_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("worker")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--records-json", required=True)
    parser.add_argument("--modes", default="sequence,initial_guess")
    parser.add_argument("--num-loops", type=int, default=3)
    parser.add_argument("--num-sampling-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--contact-cutoff", type=float, default=8.0)
    parser.add_argument("--use-target-msa", default="0")
    args = parser.parse_args(argv)
    _run_esmfold2_benchmark_worker(
        run_dir=Path(args.run_dir),
        records_json=Path(args.records_json),
        modes=[mode for mode in args.modes.split(",") if mode],
        num_loops=args.num_loops,
        num_sampling_steps=args.num_sampling_steps,
        seed=args.seed,
        device=args.device,
        contact_cutoff=args.contact_cutoff,
        use_target_msa=str(args.use_target_msa).lower() in {"1", "true", "yes", "on"},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker_cli(sys.argv[1:]))
