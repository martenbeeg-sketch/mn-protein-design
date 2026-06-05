from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from mn_protein_design.core.candidates import STAGE_COMPLEX_REFOLDING, STAGE_BENCHMARK, read_candidates, write_candidates
from mn_protein_design.core.jobs import JobPaths, create_job, finish_job, mark_internal_job, read_json, update_status, write_json
from mn_protein_design.core.structures import filter_pdb_text
from mn_protein_design.runtime import runs_root
from mn_protein_design.workflows import esm_binder as esm_binder_workflow
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
    "Protenix": ("protenix_",),
    "RF3": ("rf3_",),
    "Input": ("input_",),
}

BENCHMARK_ENGINE_ARTIFACTS: dict[str, tuple[str, str, str]] = {
    "AF3": ("alphafast_af3", "alphafast_af3_metrics.csv", "af3"),
    "AF2-IG": ("af2_initial_guess", "af2_initial_guess_metrics.csv", "af2"),
    "Boltz-2": ("boltz2", "boltz2_initial_guess_metrics.csv", "boltz2"),
    "BoltzGen Fold": ("boltzgen_fold", "boltzgen_fold_metrics.csv", "boltzgen_fold"),
    "ColabFold": ("colabfold", "colabfold_metrics.csv", "colab"),
    "ESMFold2": ("esmfold2", "esmfold2_metrics.csv", "esmfold2"),
    "Protenix": ("protenix", "protenix_metrics.csv", "protenix"),
    "RF3": ("rf3", "rf3_metrics.csv", "rf3"),
}

BENCHMARK_DATASET_PATH_KWARGS = {
    "input_zip",
    "input_csv",
    "input_pdb_dir",
    "candidates_jsonl",
    "source_run_dir",
    "rf3_checkpoint_path",
    "colabfold_cache_dir",
    "msa_repository_dir",
    "alphafast_db_dir",
    "alphafast_weights_dir",
}


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
) -> Path:
    job = create_job(BENCHMARK_GROUP, job_type, tool_name, input_payload, params_payload)
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
            "queue_resource": "gpu",
            "queued_worker": "local",
            "mode": kwargs.get("mode", "pdb_only"),
            "models": list(kwargs.get("models") or []),
            "max_records": int(kwargs.get("max_records") or 0),
            "run_alphafast_af3": bool(kwargs.get("run_alphafast_af3")),
            "run_colabfold": bool(kwargs.get("run_colabfold")),
            "run_af2_initial_guess": bool(kwargs.get("run_af2_initial_guess")),
            "run_boltz2_initial_guess": bool(kwargs.get("run_boltz2_initial_guess")),
            "run_esmfold2": bool(kwargs.get("run_esmfold2")),
            "run_rf3": bool(kwargs.get("run_rf3")),
            "run_protenix": bool(kwargs.get("run_protenix")),
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


def _stage_candidate_set_as_repo_dataset(
    *,
    source_run_dir: Path,
    candidates_jsonl: Path,
    staged_dir: Path,
    max_candidates: int = 0,
) -> tuple[Path, Path, dict[str, Any]]:
    source_run_dir = Path(source_run_dir).expanduser().resolve()
    candidates = read_candidates(Path(candidates_jsonl).expanduser())
    if max_candidates and int(max_candidates) > 0:
        candidates = candidates[: int(max_candidates)]
    if not candidates:
        raise ValueError("No candidates are available for refolding evaluation.")

    input_pdb_dir = staged_dir / "input_pdbs"
    input_pdb_dir.mkdir(parents=True, exist_ok=True)
    csv_path = staged_dir / "input.csv"
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_id = str(candidate.get("candidate_id") or f"candidate_{index}").strip()
        binder_id = _safe_id(candidate_id, fallback=f"candidate_{index}")
        complex_pdb = _relative_candidate_path(source_run_dir, candidate.get("complex_pdb") or candidate.get("binder_pdb"))
        if complex_pdb is None:
            skipped.append({"candidate_id": candidate_id, "reason": "missing_complex_pdb"})
            continue
        staged_pdb = input_pdb_dir / f"{binder_id}.pdb"
        shutil.copy2(complex_pdb, staged_pdb)
        target_chains = _split_list(candidate.get("target_chains"))
        binder_chains = _split_list(candidate.get("binder_chains"))
        metrics = dict(candidate.get("metrics") or {})
        raw_metadata = dict(candidate.get("raw_metadata") or {})
        rows.append(
            {
                "binder_id": binder_id,
                "original_binder_id": candidate_id,
                "target_id": str(raw_metadata.get("target_id") or metrics.get("target_id") or Path(str(candidate.get("target_pdb") or "target")).stem),
                "binder": "",
                "label": "",
                "source": str(candidate.get("source_tool") or raw_metadata.get("import_name") or "candidate_set"),
                "binder_chain": ",".join(binder_chains),
                "target_chains": json.dumps(target_chains),
                "complex_pdb": str(staged_pdb),
                "target_pdb": str(_relative_candidate_path(source_run_dir, candidate.get("target_pdb")) or ""),
                "binder_sequence": str(candidate.get("binder_sequence") or ""),
            }
        )
    if not rows:
        raise ValueError("None of the selected candidates had an existing complex PDB.")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        staged_dir / "candidate_staging_summary.json",
        {
            "source_run_dir": str(source_run_dir),
            "candidates_jsonl": str(candidates_jsonl),
            "candidate_count": len(candidates),
            "staged_count": len(rows),
            "skipped_count": len(skipped),
            "skipped": skipped[:100],
        },
    )
    return csv_path, input_pdb_dir, {"candidate_count": len(candidates), "staged_count": len(rows), "skipped_count": len(skipped)}


def enqueue_candidate_refolding_evaluation(
    *,
    source_run_dir: Path,
    candidates_jsonl: Path,
    max_candidates: int = 0,
    evaluation_name: str = "Refolding evaluation",
    **kwargs: Any,
) -> Path:
    """Queue a benchmark-style engine evaluation for an unlabeled candidate set."""

    input_payload = {
        "queued_worker_request": True,
        "source_run_dir": str(source_run_dir),
        "candidates_jsonl": str(candidates_jsonl),
    }
    params_payload = {
        "queue_resource": "gpu",
        "queued_worker": "local",
        "evaluation_name": evaluation_name,
        "evaluation_mode": "refolding_validation",
        "max_candidates": int(max_candidates or 0),
        "models": list(kwargs.get("models") or []),
        "run_alphafast_af3": bool(kwargs.get("run_alphafast_af3")),
        "run_colabfold": bool(kwargs.get("run_colabfold")),
        "run_af2_initial_guess": bool(kwargs.get("run_af2_initial_guess")),
        "run_boltz2_initial_guess": bool(kwargs.get("run_boltz2_initial_guess")),
        "run_esmfold2": bool(kwargs.get("run_esmfold2")),
        "run_rf3": bool(kwargs.get("run_rf3")),
        "run_protenix": bool(kwargs.get("run_protenix")),
        "run_boltzgen_fold": bool(kwargs.get("run_boltzgen_fold")),
    }
    worker_kwargs = {
        **kwargs,
        "source_run_dir": Path(source_run_dir),
        "candidates_jsonl": Path(candidates_jsonl),
        "max_records": int(max_candidates or 0),
        "mode": "pdb_only",
        "generate_inputs": True,
        "job_type": "refolding_evaluation",
        "tool_name": "refolding_evaluation_engines",
    }
    return _enqueue_benchmark_worker_job(
        job_type="refolding_evaluation",
        tool_name="refolding_evaluation_engines",
        input_payload=input_payload,
        params_payload=params_payload,
        kwargs=worker_kwargs,
        kind="candidate_refolding_evaluation",
    )


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
        return None, f"invalid:{path.name}:{exc}"
    query = "".join(str(getattr(msa, "query", "") or "").upper().split())
    expected = "".join(str(expected_sequence or "").upper().split())
    if query and expected and query != expected:
        return None, f"query_mismatch:{path.name}"
    return msa, None


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
    labels_raw = [_truthy_label(value) for value in df[label_col].tolist()]
    numeric_cols: list[str] = []
    label_like_cols = {label_col, "label", "binder", "is_binder", "binds"}
    for column in df.columns:
        if str(column) in label_like_cols:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().sum() >= 2:
            numeric_cols.append(column)
    rows: list[dict[str, Any]] = []
    for column in numeric_cols[:max_columns]:
        values = pd.to_numeric(df[column], errors="coerce")
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
        "target_id",
        "binder",
        "label",
        "source",
        "binder_chain",
        "target_chains",
        "A_seq",
        "A_length",
        "B_length",
        "target_chain_range",
        "segment_ids",
    }
    return [col for col in df.columns if col in metadata_names]


def _benchmark_engine_for_column(column: str) -> str | None:
    for engine, prefixes in BENCHMARK_ENGINE_PREFIXES.items():
        if any(column.startswith(prefix) for prefix in prefixes):
            return engine
    return None


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


def _stage_collection_engine_artifacts(
    *,
    source_run_dir: Path,
    collection_run_dir: Path,
    benchmark_dir: Path,
    engines: list[str],
) -> None:
    source_benchmark = source_run_dir / "artifacts" / "benchmark"
    for engine in engines:
        artifact_info = BENCHMARK_ENGINE_ARTIFACTS.get(engine)
        if not artifact_info:
            continue
        engine_key, metrics_csv, metric_pdb_key = artifact_info
        source_metrics = source_benchmark / metrics_csv
        if source_metrics.exists():
            _replace_path_with_link_or_copy(source_metrics, benchmark_dir / metrics_csv)
        source_engine_dir = source_run_dir / "artifacts" / "engines" / engine_key
        if source_engine_dir.exists():
            _replace_path_with_link_or_copy(
                source_engine_dir,
                collection_run_dir / "artifacts" / "engines" / engine_key,
            )
        source_metric_pdbs = source_benchmark / "predicted_metric_pdbs" / metric_pdb_key
        if source_metric_pdbs.exists():
            _replace_path_with_link_or_copy(
                source_metric_pdbs,
                benchmark_dir / "predicted_metric_pdbs" / metric_pdb_key,
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
        engines = [str(engine) for engine in (selection.get("engines") or []) if str(engine).strip()]
        metrics_path = _benchmark_run_metrics_path(run_id)
        if metrics_path is None:
            source_rows.append({"order": order, "run_id": run_id, "status": "missing_metrics", "engines": ",".join(engines)})
            continue
        source_run_dir = metrics_path.parents[2]
        metadata = read_json(source_run_dir / "metadata.json")
        result = read_json(source_run_dir / "result.json")
        try:
            df = pd.read_csv(metrics_path)
        except Exception as exc:
            source_rows.append({"order": order, "run_id": run_id, "status": f"read_error:{exc}", "engines": ",".join(engines)})
            continue
        if selected_targets and "target_id" in df.columns:
            df = df[df["target_id"].astype(str).isin(selected_targets)].copy()
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
                "keys": _collection_merge_keys(df),
            }
        )

    if loaded_sources:
        base_source = max(loaded_sources, key=lambda item: int(len(item["df"])))
        keys = list(base_source["keys"])
        metadata_cols = _collection_metadata_columns(base_source["df"])
        if not set(keys).issubset(metadata_cols):
            metadata_cols = [col for col in keys if col in base_source["df"].columns] + metadata_cols
            metadata_cols = list(dict.fromkeys(metadata_cols))
        base = base_source["df"][metadata_cols].drop_duplicates(keys).copy()
        _stage_collection_reference_artifacts(base_source["source_run_dir"], job.run_dir)

    for source in loaded_sources:
        order = int(source["order"])
        run_id = str(source["run_id"])
        engines = list(source["engines"])
        df: pd.DataFrame = source["df"]
        metadata: dict[str, Any] = source["metadata"]
        result: dict[str, Any] = source["result"]
        current_keys = list(source["keys"])
        if current_keys != keys:
            source_rows.append({"order": order, "run_id": run_id, "status": "incompatible_merge_keys", "engines": ",".join(engines)})
            continue

        selected_columns: list[str] = []
        if include_input_columns and "Input" in engines:
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
        incoming = df[keys + selected_columns].drop_duplicates(keys).copy()
        source_key_index = set(tuple(row) for row in incoming[keys].itertuples(index=False, name=None))
        base_key_index = set(tuple(row) for row in base[keys].itertuples(index=False, name=None))
        records_not_in_base = len(source_key_index - base_key_index)
        matched_records = len(source_key_index & base_key_index)
        coverage_records = int(incoming[selected_columns].notna().any(axis=1).sum())
        replaced = [col for col in selected_columns if col in base.columns]
        if replaced:
            base = base.drop(columns=replaced)
            for column in replaced:
                replacement_rows.append({"column": column, "replaced_by_run_id": run_id, "order": order})
        base = base.merge(incoming, on=keys, how="left")
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
                }
            )
        source_rows.append(
            {
                "order": order,
                "run_id": run_id,
                "status": "included",
                "engines": ",".join(engines),
                "selected_feature_count": len(selected_columns),
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

    metadata_cols = [
        col
        for col in ["binder", "label", "source", "target_id", "binder_chain", "target_chains", "target_chain_range"]
        if col in staged_df.columns
    ]
    if not metadata_cols:
        return {"staged_metadata_rows": int(len(staged_df)), "staged_metadata_matched_rows": 0}

    staged = staged_df.copy()
    staged["_safe_binder_id"] = staged["binder_id"].map(lambda value: _safe_id(value).lower())
    run_df["_safe_binder_id"] = run_df["binder_id"].map(lambda value: _safe_id(value).lower())
    meta = staged.drop_duplicates("_safe_binder_id").set_index("_safe_binder_id")
    matched = run_df["_safe_binder_id"].isin(meta.index)
    for col in metadata_cols:
        mapped = run_df["_safe_binder_id"].map(meta[col])
        if col in {"target_id", "binder_chain", "target_chains", "target_chain_range"} and col in run_df.columns:
            run_df[col] = mapped.where(mapped.notna(), run_df[col])
        else:
            run_df[col] = mapped
    run_df["original_binder_id"] = run_df["_safe_binder_id"].map(meta["binder_id"])
    run_df = run_df.drop(columns=["_safe_binder_id"])
    run_df.to_csv(run_csv, index=False)
    return {
        "staged_metadata_rows": int(len(staged_df)),
        "staged_metadata_matched_rows": int(matched.sum()),
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
        if not target_chains:
            try:
                parsed = json.loads(str(row.get("target_chains") or "[]"))
                target_chains = [str(item) for item in parsed]
            except Exception:
                target_chains = ["B"]
        binder_chain = str(row.get("binder_chain") or "A").strip() or "A"
        binder_sequence = str(row.get(f"{binder_chain}_seq") or row.get("A_seq") or "").strip()
        msa_columns = {
            str(col): str(row.get(col) or "").strip()
            for col in df.columns
            if str(col).startswith("msa_path_") and str(row.get(col) or "").strip()
        }
        rows.append(
            {
                "candidate_id": binder_id,
                "target_id": row.get("target_id") or "",
                "source": row.get("source") or "",
                "label": row.get("binder") if "binder" in df.columns else row.get("label"),
                "target_pdb": str(complex_pdb.resolve()),
                "complex_pdb": str(complex_pdb.resolve()),
                "binder_sequence": binder_sequence,
                "binder_chains": binder_chain,
                "target_chains": ",".join(target_chains),
                **msa_columns,
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
        binder_chain = str(row.get("binder_chain") or "A").strip() or "A"
        target_chains = _split_list(row.get("target_chains"))
        if not target_chains:
            try:
                parsed = json.loads(str(row.get("target_chains") or "[]"))
                target_chains = [str(item) for item in parsed if str(item)]
            except Exception:
                target_chains = ["B"]
        label = _truthy_label(row.get("binder") if "binder" in df.columns else row.get("label"))
        rows.append(
            {
                "candidate_id": binder_id,
                "stage": STAGE_COMPLEX_REFOLDING,
                "source_tool": "de_novo_binder_scoring_dataset",
                "target_pdb": str(complex_pdb.relative_to(run_dir)),
                "complex_pdb": str(complex_pdb.relative_to(run_dir)),
                "binder_sequence": str(row.get(f"{binder_chain}_seq") or row.get("A_seq") or "").strip(),
                "binder_chains": [binder_chain],
                "target_chains": target_chains,
                "metrics": {
                    "label": label,
                    "binder": label,
                    "target_id": row.get("target_id") or "",
                    "source": row.get("source") or "",
                },
                "raw_metadata": {"repo_run_csv_row": {str(key): value for key, value in row.to_dict().items()}},
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
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists() or target_path.is_symlink():
        return
    try:
        target_path.symlink_to(source_path)
    except OSError:
        shutil.copy2(source_path, target_path)


def _normalize_structure_for_group_metrics(
    source_path: Path,
    target_path: Path,
    *,
    binder_chains: list[str],
    target_chains: list[str],
) -> bool:
    structure_chains = refolding_workflow._structure_chains(source_path)
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
    return True


def _benchmark_group_roles(run_csv: Path) -> dict[str, tuple[list[str], list[str]]]:
    if not run_csv.exists():
        return {}
    roles: dict[str, tuple[list[str], list[str]]] = {}
    for _, row in pd.read_csv(run_csv).iterrows():
        binder_id = str(row.get("binder_id") or "").strip()
        if not binder_id:
            continue
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
            complex_path = child_dir / str(complex_rel)
            if not complex_path.exists():
                continue
            target = stage_dir / f"{_safe_id(binder_id)}.pdb"
            if target.exists():
                count += 1
                continue
            binder_chains = [str(chain) for chain in candidate.get("binder_chains") or [] if str(chain)]
            target_chains = [str(chain) for chain in candidate.get("target_chains") or [] if str(chain)]
            if complex_path.suffix.lower() == ".pdb":
                if not _normalize_structure_for_group_metrics(
                    complex_path,
                    target,
                    binder_chains=binder_chains,
                    target_chains=target_chains,
                ):
                    _link_or_copy(complex_path, target)
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
                    tmp.replace(target)
            else:
                continue
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
        cif_candidates = sorted(folder.glob("*.cif"))
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
                tmp.replace(target)
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
    pdb_files = sorted(pdb_dir.glob("*_unrelaxed_rank_001*.pdb"))
    if not pdb_files:
        pdb_files = sorted(pdb_dir.glob("*_relaxed_rank_001*.pdb"))
    if not pdb_files:
        pdb_files = sorted(pdb_dir.glob("*.pdb"))
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
            _link_or_copy(source, target)
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
            _link_or_copy(source, target)
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
            if not pdb_target.exists():
                try:
                    pdb_target.symlink_to(pdb_path)
                except OSError:
                    shutil.copy2(pdb_path, pdb_target)
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
) -> tuple[dict[str, Any], list[list[str]]]:
    metrics: dict[str, Any] = {}
    commands: list[list[str]] = []
    benchmark_dir = job_run_dir / "artifacts" / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    if run_common_interface_metrics and run_csv.exists():
        ipsae_csv = benchmark_dir / "common_interface_metrics.csv"
        base_ipsae_cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{REPO_ROOT}:{REPO_ROOT}",
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
        for engine_key in ("rf3", "protenix", "boltzgen_fold"):
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
    af3_pdb_dir, af3_pdb_count = _stage_alphafast_pdbs(
        output_dir / "AF3" / "alphafast_output",
        benchmark_dir / "predicted_metric_pdbs" / "af3",
        group_roles,
    )
    if af3_pdb_dir is not None:
        predicted_pdb_dirs.append(("af3", af3_pdb_dir))
        metrics["af3_predicted_metric_pdb_count"] = af3_pdb_count
    colab_pdb_dir, colab_pdb_count = _stage_colabfold_pdbs(
        output_dir,
        benchmark_dir / "predicted_metric_pdbs" / "colab",
        group_roles,
    )
    if colab_pdb_dir is not None:
        predicted_pdb_dirs.append(("colab", colab_pdb_dir))
        metrics["colab_predicted_metric_pdb_count"] = colab_pdb_count
    af2_pdb_dir, af2_pdb_count = _stage_child_candidate_pdbs(
        af2_child_runs,
        benchmark_dir / "predicted_metric_pdbs" / "af2",
        source_label="af2",
    )
    if af2_pdb_dir is not None:
        predicted_pdb_dirs.append(("af2", af2_pdb_dir))
        metrics["af2_predicted_metric_pdb_count"] = af2_pdb_count
    boltz2_pdb_dir, boltz2_pdb_count = _stage_child_candidate_pdbs(
        boltz2_child_runs,
        benchmark_dir / "predicted_metric_pdbs" / "boltz2",
        source_label="boltz2",
    )
    if boltz2_pdb_dir is not None:
        predicted_pdb_dirs.append(("boltz2", boltz2_pdb_dir))
        metrics["boltz2_predicted_metric_pdb_count"] = boltz2_pdb_count
    esmfold2_pdb_dir, esmfold2_pdb_count = _stage_child_candidate_pdbs(
        esmfold2_child_runs,
        benchmark_dir / "predicted_metric_pdbs" / "esmfold2",
        source_label="esmfold2",
    )
    if esmfold2_pdb_dir is not None:
        predicted_pdb_dirs.append(("esmfold2", esmfold2_pdb_dir))
        metrics["esmfold2_predicted_metric_pdb_count"] = esmfold2_pdb_count
    for engine, child_runs in sorted((extra_child_runs or {}).items()):
        engine_pdb_dir, engine_pdb_count = _stage_child_candidate_pdbs(
            child_runs,
            benchmark_dir / "predicted_metric_pdbs" / engine,
            source_label=engine,
        )
        if engine_pdb_dir is not None:
            predicted_pdb_dirs.append((engine, engine_pdb_dir))
            metrics[f"{engine}_predicted_metric_pdb_count"] = engine_pdb_count

    if run_predicted_rosetta_metrics and run_csv.exists():
        rosetta_csv = benchmark_dir / "predicted_rosetta_metrics.csv"
        if predicted_pdb_dirs:
            rosetta_cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{REPO_ROOT}:{REPO_ROOT}",
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
    af3_input_dir: Path,
    output_dir: Path,
    db_dir: Path,
    weights_dir: Path,
    batch_size: int = 0,
    num_recycles: int = 10,
    gpu_device: int = 0,
    max_records: int = 0,
    image: str = ALPHAFAST_IMAGE,
    run_data_pipeline: bool = True,
) -> tuple[int, list[list[str]]]:
    commands: list[list[str]] = []
    if run_data_pipeline:
        rc, commands = _run_alphafast_data_pipeline(
            job_run_dir=job_run_dir,
            af3_input_dir=af3_input_dir,
            output_dir=output_dir,
            db_dir=db_dir,
            batch_size=batch_size,
            gpu_device=gpu_device,
            max_records=max_records,
            image=image,
        )
        if rc != 0:
            return rc, commands
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
    return rc, commands


def _alphafast_common_mounts(*, db_dir: Path, weights_dir: Path | None, gpu_device: int, image: str) -> list[str]:
    mounts = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        f"device={int(gpu_device)}",
        "-v",
        f"{REPO_ROOT}:{REPO_ROOT}",
        "-v",
        f"{db_dir}:/data/public_databases:ro",
        "-v",
        f"{db_dir / 'mmseqs'}:/data/mmseqs_databases:ro",
    ]
    if weights_dir is not None:
        mounts.extend(["-v", f"{weights_dir}:/data/models:ro"])
    mounts.extend(["-w", "/app/alphafold", image])
    return mounts


def _stage_alphafast_inputs(af3_input_dir: Path, output_dir: Path, max_records: int) -> tuple[Path, int]:
    if not af3_input_dir.exists() or not any(af3_input_dir.glob("*.json")):
        raise ValueError("AlphaFast AF3 benchmark needs generated AF3/input_folder/*.json inputs.")
    selected_inputs = sorted(af3_input_dir.glob("*.json"))
    if max_records > 0:
        selected_inputs = selected_inputs[: int(max_records)]
    staged_input_dir = output_dir.parent / "alphafast_input"
    staged_input_dir.mkdir(parents=True, exist_ok=True)
    for existing in staged_input_dir.glob("*.json"):
        existing.unlink()
    for source in selected_inputs:
        target = staged_input_dir / source.name
        payload = json.loads(source.read_text())
        for sequence in payload.get("sequences", []):
            protein = sequence.get("protein") if isinstance(sequence, dict) else None
            if not isinstance(protein, dict):
                continue
            msa_path = protein.get("unpairedMsaPath")
            if msa_path:
                host_msa_path = Path(str(msa_path))
                if host_msa_path.exists():
                    protein["unpairedMsa"] = host_msa_path.read_text(errors="replace")
                    protein.pop("unpairedMsaPath", None)
                else:
                    protein.pop("unpairedMsaPath", None)
                    protein.pop("unpairedMsa", None)
                    protein.pop("pairedMsaPath", None)
                    protein.pop("pairedMsa", None)
                    protein.pop("templates", None)
            paired_path = protein.get("pairedMsaPath")
            if paired_path:
                host_paired_path = Path(str(paired_path))
                if host_paired_path.exists():
                    protein["pairedMsa"] = host_paired_path.read_text(errors="replace")
                    protein.pop("pairedMsaPath", None)
                else:
                    protein.pop("pairedMsaPath", None)
                    protein.pop("pairedMsa", None)
                    protein.pop("templates", None)
        target.write_text(json.dumps(payload, indent=2))
    return staged_input_dir, len(selected_inputs)


def _run_alphafast_data_pipeline(
    *,
    job_run_dir: Path,
    af3_input_dir: Path,
    output_dir: Path,
    db_dir: Path,
    batch_size: int = 0,
    gpu_device: int = 0,
    max_records: int = 0,
    image: str = ALPHAFAST_IMAGE,
) -> tuple[int, list[list[str]]]:
    if not db_dir.exists():
        raise ValueError(f"AlphaFast database directory does not exist: {db_dir}")
    if not (db_dir / "mmseqs").exists():
        raise ValueError(f"AlphaFast database directory must contain an mmseqs subdirectory: {db_dir / 'mmseqs'}")
    staged_input_dir, input_count = _stage_alphafast_inputs(af3_input_dir, output_dir, max_records)
    output_dir.mkdir(parents=True, exist_ok=True)
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
    return rc, [pipeline_cmd]


def _write_alphafast_msas_to_run_csv(*, run_csv: Path, alphafast_output_dir: Path, output_dir: Path) -> dict[str, Any]:
    df = pd.read_csv(run_csv)
    msa_dir = output_dir / "unique_msa" / "msa"
    msa_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    data_files = {path.parent.name: path for path in alphafast_output_dir.glob("*/*_data.json")}
    for index, row in df.iterrows():
        binder_id = str(row.get("binder_id") or "").strip()
        data_path = data_files.get(binder_id)
        if not data_path:
            continue
        payload = json.loads(data_path.read_text())
        binder_chain = str(row.get("binder_chain") or "A").strip() or "A"
        for sequence in payload.get("sequences", []):
            protein = sequence.get("protein") if isinstance(sequence, dict) else None
            if not isinstance(protein, dict):
                continue
            chain_id = str(protein.get("id") or "").strip()
            if not chain_id or chain_id == binder_chain:
                continue
            current_msa = str(row.get(f"msa_path_{chain_id}") or "").strip()
            if current_msa and current_msa.lower() != "no_msa" and Path(current_msa).exists():
                continue
            msa_text = str(protein.get("unpairedMsa") or "").strip()
            if not msa_text:
                continue
            msa_path = msa_dir / f"{_safe_id(binder_id)}_chain_{_safe_id(chain_id)}_alphafast.a3m"
            msa_path.write_text(msa_text + "\n")
            df.loc[index, f"msa_path_{chain_id}"] = str(msa_path)
            written += 1
    df.to_csv(run_csv, index=False)
    return {
        "alphafast_msa_source": str(alphafast_output_dir),
        "alphafast_msa_written_count": written,
        "alphafast_msa_dir": str(msa_dir),
    }


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
        chains = _split_list(row.get("target_chains"))
        for chain in chains:
            if not chain or chain == binder_chain:
                continue
            raw_sequence = row.get(f"target_subchain_{chain}_seq")
            if raw_sequence is None or pd.isna(raw_sequence) or not str(raw_sequence).strip():
                raw_sequence = row.get(f"{chain}_seq")
            sequence = "" if raw_sequence is None or pd.isna(raw_sequence) else str(raw_sequence).strip()
            if not sequence:
                continue
            host_path, _container_path = target_msa_workflow.boltz_msa_paths(sequence, msa_repository_dir=msa_repository_dir)
            valid, _reason = target_msa_workflow.validate_a3m_file(host_path)
            if valid:
                df.loc[index, f"msa_path_{chain}"] = str(host_path)
                repository_hits += 1
            elif host_path.exists():
                invalid += 1
                repository_misses += 1
            else:
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
        for chain in _split_list(row.get("target_chains")):
            if not chain or chain == binder_chain:
                continue
            msa_value = str(row.get(f"msa_path_{chain}") or "").strip()
            if not msa_value or msa_value.lower() == "no_msa" or not Path(msa_value).exists():
                missing += 1
    return missing


def _write_chain_msa_map(run_csv: Path, output_path: Path, max_records: int = 0) -> dict[str, Any]:
    df = pd.read_csv(run_csv)
    limit = int(max_records) if max_records and int(max_records) > 0 else len(df)
    records: list[dict[str, Any]] = []
    msa_present = 0
    msa_missing = 0
    target_chain_count = 0
    for _, row in df.head(limit).iterrows():
        binder_id = str(row.get("binder_id") or "").strip()
        if not binder_id:
            continue
        binder_chain = str(row.get("binder_chain") or "A").strip() or "A"
        target_chains = [chain for chain in _split_list(row.get("target_chains")) if chain and chain != binder_chain]
        target_entries: list[dict[str, Any]] = []
        for chain in target_chains:
            msa_path = str(row.get(f"msa_path_{chain}") or "").strip()
            has_msa = bool(msa_path and msa_path.lower() != "no_msa" and Path(msa_path).exists())
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
                    "length": _safe_int(row.get(f"target_subchain_{chain}_len") or row.get(f"{chain}_length")),
                    "msa_path": msa_path or None,
                    "msa_available": has_msa,
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
        "-v",
        f"{REPO_ROOT}:{REPO_ROOT}",
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
    gpu_device: int,
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
        "--gpus",
        f"device={int(gpu_device)}",
        "--shm-size=32G",
        "-v",
        f"{REPO_ROOT}:{REPO_ROOT}",
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
        "-v",
        f"{REPO_ROOT}:{REPO_ROOT}",
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
    rf3_recycles: int = 10,
    rf3_num_steps: int = 50,
    rf3_diffusion_batch_size: int = 5,
    rf3_seed: int = 0,
    run_protenix: bool = False,
    protenix_use_msa: bool = True,
    protenix_cycle: int = 3,
    protenix_diffusion_steps: int = 50,
    protenix_samples: int = 5,
    run_boltzgen_fold: bool = False,
    boltzgen_recycling_steps: int = 3,
    boltzgen_sampling_steps: int = 200,
    boltzgen_diffusion_samples: int = 5,
    run_colabfold: bool = False,
    colabfold_cache_dir: Path = COLABFOLD_CACHE_DIR,
    colabfold_msa_source: str = "msa_repository_then_alphafast_mmseqs_gpu",
    msa_repository_dir: Path = MSA_REPOSITORY_DIR,
    colabfold_num_recycles: int = 3,
    colabfold_num_models: int = 3,
    colabfold_use_target_templates: bool = True,
    colabfold_max_template_hits: int = 4,
    colabfold_gpu_device: int = 0,
    run_alphafast_af3: bool = False,
    alphafast_db_dir: Path = ALPHAFAST_DB_DIR,
    alphafast_weights_dir: Path = ALPHAFAST_WEIGHTS_DIR,
    alphafast_batch_size: int = 0,
    alphafast_num_recycles: int = 10,
    alphafast_gpu_device: int = 0,
    max_records: int = 0,
    num_loops: int = 3,
    num_sampling_steps: int = 32,
    seed: int = 0,
    device: str = "cuda",
    docker_image: str = SCORING_SCRIPTS_IMAGE,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    existing_job: JobPaths | None = None,
    job_type: str = "de_novo_binder_scoring_dataset",
    tool_name: str = "de_novo_binder_scoring_scripts",
) -> Path:
    msa_source = str(colabfold_msa_source or "msa_repository_then_alphafast_mmseqs_gpu")
    msa_consuming_engines = bool(
        run_colabfold
        or run_alphafast_af3
        or (run_boltz2_initial_guess and boltz2_use_target_msa)
        or (run_esmfold2 and esmfold2_use_target_msa)
        or (run_rf3 and rf3_use_target_msa)
        or (run_protenix and protenix_use_msa)
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
    }
    params_payload = {
        "evaluation_mode": "refolding_validation" if candidates_jsonl else "binder_benchmark",
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
        "rf3_recycles": rf3_recycles,
        "rf3_num_steps": rf3_num_steps,
        "rf3_diffusion_batch_size": rf3_diffusion_batch_size,
        "rf3_seed": rf3_seed,
        "run_protenix": run_protenix,
        "protenix_use_msa": protenix_use_msa,
        "protenix_cycle": protenix_cycle,
        "protenix_diffusion_steps": protenix_diffusion_steps,
        "protenix_samples": protenix_samples,
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
        "colabfold_num_recycles": colabfold_num_recycles,
        "colabfold_num_models": colabfold_num_models,
        "colabfold_use_target_templates": colabfold_use_target_templates,
        "colabfold_max_template_hits": colabfold_max_template_hits,
        "colabfold_gpu_device": colabfold_gpu_device,
        "run_alphafast_af3": run_alphafast_af3,
        "alphafast_image": ALPHAFAST_IMAGE,
        "alphafast_db_dir": str(alphafast_db_dir),
        "alphafast_weights_dir": str(alphafast_weights_dir),
        "alphafast_batch_size": alphafast_batch_size,
        "alphafast_num_recycles": alphafast_num_recycles,
        "alphafast_gpu_device": alphafast_gpu_device,
        "max_records": max_records,
        "docker_image": docker_image,
    }
    if existing_job is None:
        job = create_job(
            BENCHMARK_GROUP,
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
    update_status(job.run_dir, "running")
    raw_dir = job.run_dir / "artifacts" / "raw" / "de_novo_binder_scoring"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if candidates_jsonl is not None:
        if source_run_dir is None:
            raise ValueError("source_run_dir is required when candidates_jsonl is provided.")
        candidate_stage_dir = job.run_dir / "artifacts" / "queued_inputs" / "candidate_repo_dataset"
        input_csv, input_pdb_dir, candidate_stage_metrics = _stage_candidate_set_as_repo_dataset(
            source_run_dir=Path(source_run_dir),
            candidates_jsonl=Path(candidates_jsonl),
            staged_dir=candidate_stage_dir,
            max_candidates=int(max_records or 0),
        )
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
    staged_csv, staged_pdb_dir = _stage_repo_format_inputs(
        raw_dir=raw_dir,
        input_zip=input_zip,
        input_zip_bytes=input_zip_bytes,
        input_csv=input_csv,
        input_csv_text=input_csv_text,
        input_pdb_dir=input_pdb_dir,
    )
    output_dir = raw_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
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
            bool(run_protenix),
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
        "-v",
        f"{REPO_ROOT}:{REPO_ROOT}",
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
    write_json(job.run_dir / "command.json", {"mode": "docker", "commands": [process_cmd]})
    return_code = _run_docker_command(job.run_dir, process_cmd)
    if return_code != 0:
        finish_job(job.run_dir, False, {"metrics": {"process_inputs_return_code": return_code}})
        raise RuntimeError(f"de_novo_binder_scoring process_inputs failed with return code {return_code}.")
    commands = [process_cmd]
    run_csv = output_dir / "run.csv"
    pre_input_metrics: dict[str, Any] = {}
    pre_input_metrics.update(_apply_staged_csv_metadata_to_run_csv(run_csv, staged_csv))
    runtime_size = _run_csv_runtime_size(run_csv, max_records=int(max_records)) if run_csv.exists() else {"candidate_count": 0}
    alphafast_data_pipeline_done = False
    if msa_consuming_engines and msa_source != "repo_run_csv" and run_csv.exists():
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
        if msa_source == "msa_repository" and pre_input_metrics["msa_repository_missing_after_lookup"]:
            finish_job(job.run_dir, False, {"metrics": pre_input_metrics})
            raise RuntimeError(
                "MSA repository mode was selected, but some selected target-chain MSAs are missing. "
                "Use repository-then-AlphaFast/MMseqs to fill misses with the local MMseqs GPU pipeline."
            )
        use_alphafast_msa = msa_source == "alphafast_mmseqs_gpu" or (
            msa_source == "msa_repository_then_alphafast_mmseqs_gpu"
            and pre_input_metrics["msa_repository_missing_after_lookup"] > 0
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
            rc, data_pipeline_commands = _run_alphafast_data_pipeline(
                job_run_dir=job.run_dir,
                af3_input_dir=af3_input_dir,
                output_dir=af3_output_dir,
                db_dir=Path(alphafast_db_dir).expanduser(),
                batch_size=int(alphafast_batch_size),
                gpu_device=int(alphafast_gpu_device),
                max_records=int(max_records),
                image=ALPHAFAST_IMAGE,
            )
            commands.extend(data_pipeline_commands)
            write_json(job.run_dir / "command.json", {"mode": "docker", "commands": commands})
            pre_input_metrics["alphafast_msa_pipeline_return_code"] = rc
            if rc != 0:
                finish_job(job.run_dir, False, {"metrics": pre_input_metrics})
                raise RuntimeError(f"AlphaFast/MMseqs shared MSA generation failed with return code {rc}.")
            alphafast_data_pipeline_done = True
            _record_runtime_timing(pre_input_metrics, "alphafast_msa", msa_started_at, **runtime_size)
            pre_input_metrics.update(
                _write_alphafast_msas_to_run_csv(
                    run_csv=run_csv,
                    alphafast_output_dir=af3_output_dir,
                    output_dir=output_dir,
                )
            )
            pre_input_metrics["shared_msa_missing_after_generation"] = _run_csv_missing_target_msas(run_csv, max_records=int(max_records))
    if run_csv.exists():
        chain_msa_summary = _write_chain_msa_map(
            run_csv,
            job.run_dir / "artifacts" / "benchmark" / "chain_msa_map.json",
            max_records=int(max_records),
        )
        pre_input_metrics.update({f"chain_msa_{key}": value for key, value in chain_msa_summary.items()})
    if generate_inputs:
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
    metric_csvs: list[Path] = []
    if run_pyrosetta_input_metrics:
        progress_step += 1
        _emit_benchmark_progress(
            job.run_dir,
            phase="Scoring input structures",
            engine="PyRosetta",
            step=progress_step,
            total_steps=progress_total,
            progress_callback=progress_callback,
        )
        rosetta_cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{REPO_ROOT}:{REPO_ROOT}",
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
            f"input:{output_dir / 'input_pdbs'}",
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
    protenix_child_runs: list[str] = []
    boltzgen_fold_child_runs: list[str] = []
    esmfold2_summary: dict[str, Any] = {}
    af2_summary: dict[str, Any] = {}
    boltz2_summary: dict[str, Any] = {}
    rf3_summary: dict[str, Any] = {}
    protenix_summary: dict[str, Any] = {}
    boltzgen_fold_summary: dict[str, Any] = {}
    colabfold_summary: dict[str, Any] = {}
    if run_esmfold2 and run_csv.exists():
        progress_step += 1
        _emit_benchmark_progress(
            job.run_dir,
            phase="Predicting complexes",
            engine="ESMFold2",
            step=progress_step,
            total_steps=progress_total,
            progress_callback=progress_callback,
        )
        started_at = time.monotonic()
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
            phase="Predicting complexes",
            engine="AF2 initial guess",
            step=progress_step,
            total_steps=progress_total,
            progress_callback=progress_callback,
        )
        started_at = time.monotonic()
        benchmark_candidates = _repo_run_csv_to_candidates(job.run_dir, run_csv, max_records=max_records)
        child_run = refolding_workflow.run_af2_initial_guess_complex_refolding(
            source_run_dir=job.run_dir,
            candidates_jsonl=benchmark_candidates,
            require_monomer_success=False,
            num_recycles=int(af2_num_recycles),
            multimer=bool(af2_multimer),
            use_binder_template=bool(af2_use_binder_template),
            use_interface_template=bool(af2_use_interface_template),
            docker_image=AF2_INITIAL_GUESS_IMAGE,
            internal_parent_run_dir=job.run_dir,
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
            phase="Predicting complexes",
            engine="Boltz-2",
            step=progress_step,
            total_steps=progress_total,
            progress_callback=progress_callback,
        )
        started_at = time.monotonic()
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
                n_recycles=int(rf3_recycles),
                num_steps=int(rf3_num_steps),
                diffusion_batch_size=int(rf3_diffusion_batch_size),
                seed=int(rf3_seed),
                benchmark_run_csv=run_csv if rf3_use_target_msa else None,
                internal_parent_run_dir=job.run_dir,
            ),
        ),
        (
            bool(run_protenix),
            "Protenix",
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
            ),
        ),
    ]
    for enabled, display_name, engine_key, child_runs, summary_target, runner in extra_engine_specs:
        if not enabled or not run_csv.exists():
            continue
        progress_step += 1
        _emit_benchmark_progress(
            job.run_dir,
            phase="Predicting complexes",
            engine=display_name,
            step=progress_step,
            total_steps=progress_total,
            progress_callback=progress_callback,
        )
        started_at = time.monotonic()
        benchmark_candidates = _repo_run_csv_to_candidates(job.run_dir, run_csv, max_records=max_records)
        child_run = runner(benchmark_candidates)
        child_runs.append(str(child_run))
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
            phase="Predicting complexes",
            engine="ColabFold",
            step=progress_step,
            total_steps=progress_total,
            progress_callback=progress_callback,
        )
        colabfold_started_at = time.monotonic()
        metrics["colabfold_msa_source"] = msa_source
        metrics["colabfold_use_target_templates"] = bool(colabfold_use_target_templates)
        metrics["colabfold_max_template_hits"] = int(colabfold_max_template_hits)
        rc, selected_count, template_count, colabfold_commands = _run_colabfold_prediction(
            job_run_dir=job.run_dir,
            run_csv=run_csv,
            output_dir=output_dir,
            cache_dir=Path(colabfold_cache_dir).expanduser(),
            num_recycles=int(colabfold_num_recycles),
            num_models=int(colabfold_num_models),
            gpu_device=int(colabfold_gpu_device),
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
            phase="Predicting complexes",
            engine="AlphaFast AF3",
            step=progress_step,
            total_steps=progress_total,
            progress_callback=progress_callback,
        )
        started_at = time.monotonic()
        af3_input_dir = output_dir / "AF3" / "input_folder"
        af3_output_dir = output_dir / "AF3" / "alphafast_output"
        rc, alphafast_commands = _run_alphafast_af3_refolding(
            job_run_dir=job.run_dir,
            af3_input_dir=af3_input_dir,
            output_dir=af3_output_dir,
            db_dir=Path(alphafast_db_dir).expanduser(),
            weights_dir=Path(alphafast_weights_dir).expanduser(),
            batch_size=int(alphafast_batch_size),
            num_recycles=int(alphafast_num_recycles),
            gpu_device=int(alphafast_gpu_device),
            max_records=int(max_records),
            image=ALPHAFAST_IMAGE,
            run_data_pipeline=not alphafast_data_pipeline_done,
        )
        commands.extend(alphafast_commands)
        write_json(job.run_dir / "command.json", {"mode": "docker", "commands": commands})
        metrics["alphafast_af3_return_code"] = rc
        metrics["alphafast_af3_output_dir"] = str(af3_output_dir.relative_to(job.run_dir))
        if rc != 0:
            finish_job(job.run_dir, False, {"metrics": metrics})
            raise RuntimeError(f"AlphaFast AF3 benchmark failed with return code {rc}.")
        af3_table, af3_summary_path, af3_summary = _summarize_alphafast_af3_outputs(
            parent_run_dir=job.run_dir,
            run_csv=run_csv,
            alphafast_output_dir=af3_output_dir,
        )
        if af3_table is not None:
            metrics["alphafast_af3_metrics_table"] = str(af3_table.relative_to(job.run_dir))
            metric_csvs.append(af3_table)
        if af3_summary_path is not None:
            metrics["alphafast_af3_summary"] = str(af3_summary_path.relative_to(job.run_dir))
        _record_runtime_timing(metrics, "alphafast_af3", started_at, **runtime_size)
    progress_step += 1
    _emit_benchmark_progress(
        job.run_dir,
        phase="Calculating benchmark metrics",
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
                "protenix": protenix_child_runs,
                "boltzgen_fold": boltzgen_fold_child_runs,
            },
            run_common_interface_metrics=bool(run_common_interface_metrics),
            run_predicted_rosetta_metrics=bool(run_predicted_rosetta_metrics),
            run_pymol_metrics=bool(run_pymol_metrics),
            pyrosetta_nprocs=int(pyrosetta_nprocs),
            docker_image=docker_image,
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
            "protenix": protenix_child_runs,
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
            source_complex = Path(str(record.get("complex_pdb") or "")) if record.get("complex_pdb") else None
            binder_sequence = str(record.get("binder_sequence") or "").strip()
            if not binder_sequence:
                binder_sequence = refolding_workflow._candidate_binder_sequence(source_run_dir, source_candidate)
            binder_chains, inferred_target_chains = refolding_workflow._infer_chain_roles(source_run_dir, source_candidate, source_complex)
            target_sequences, residue_maps = esm_binder_workflow._target_sequences(target_pdb, inferred_target_chains)
            target_chains = list(target_sequences)
            binder_chain = esm_binder_workflow._choose_binder_chain(target_chains)
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
                        ProteinInput(id=binder_chain, sequence=binder_sequence),
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
                    "binder_length": len(binder_sequence),
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
                    "binder_chains": [binder_chain],
                    "hotspots": list(record.get("hotspots") or []),
                    "binder_length": str(len(binder_sequence)),
                    "metrics": metrics,
                    "parents": [str(record["candidate_id"])],
                    "raw_metadata": {
                        "benchmark_record": record,
                        "confidence": confidence_analysis,
                        "confidence_json": confidence_metrics.get("esmfold2_confidence_json"),
                        "confidence_arrays": confidence_metrics.get("esmfold2_confidence_arrays"),
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
            "--gpus",
            "all",
            "--shm-size=64G",
            "-v",
            f"{REPO_ROOT}:{REPO_ROOT}",
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
