from __future__ import annotations

import argparse
import csv
import json
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
from mn_protein_design.core.jobs import create_job, finish_job, read_json, update_status, write_json
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


def _safe_id(value: object, fallback: str = "benchmark") -> str:
    text = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value or "").strip())
    return text.strip("_") or fallback


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
                "hotspots": _split_list(_row_value(row, "hotspots", "target_hotspots")),
                "raw_input": row,
            }
        )
    if not records:
        raise ValueError("Benchmark CSV did not contain any rows.")
    return records


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
    summary = {
        "record_count": int(len(df)),
        "label_column": label_col,
        "labeled_count": sum(1 for label in labels_raw if label in {0, 1}),
        "positive_count": sum(1 for label in labels_raw if label == 1),
        "negative_count": sum(1 for label in labels_raw if label == 0),
        "numeric_feature_count": len(numeric_cols),
        "scored_feature_count": len(rows),
        "top_feature": rows[0]["feature"] if rows else None,
        "top_feature_average_precision": rows[0]["best_average_precision"] if rows else None,
        "top_feature_auroc": rows[0]["best_auroc"] if rows else None,
        "top_feature_direction": rows[0]["direction"] if rows else None,
        "ranking_metric": "average_precision",
    }
    return rows, summary


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
    pd.DataFrame(feature_rows).to_csv(feature_table, index=False)
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
    df.to_csv(table, index=False)
    feature_rows, summary = _metric_column_summary(df, "label", max_columns=500)
    summary.update(
        {
            "child_run_dir": str(child_run_dir),
            "scored_feature_count": len(feature_rows),
            "top_feature": feature_rows[0]["feature"] if feature_rows else None,
            "top_feature_average_precision": feature_rows[0]["best_average_precision"] if feature_rows else None,
            "top_feature_auroc": feature_rows[0]["best_auroc"] if feature_rows else None,
        }
    )
    write_json(summary_path, summary)
    pd.DataFrame(feature_rows).to_csv(out_dir / f"{output_prefix}_feature_benchmark.csv", index=False)
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
    if target_chains:
        chain_map[target_chains[0]] = "A"
    if binder_chain:
        chain_map[binder_chain] = "B"
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


def _esmfold2_rosetta_chain_map(pae_path: Path) -> dict[str, str]:
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
    if target_chains:
        chain_map[target_chains[0]] = "B"
    return chain_map


def _stage_child_candidate_pdbs(
    child_runs: list[str],
    stage_dir: Path,
    *,
    source_label: str,
    esmfold2_chain_map: bool = False,
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
            if complex_path.suffix.lower() == ".pdb":
                if esmfold2_chain_map:
                    pae_rel = (candidate.get("raw_metadata") or {}).get("pae_path")
                    pae_path = child_dir / str(pae_rel) if pae_rel else None
                    chain_map = _esmfold2_rosetta_chain_map(pae_path) if pae_path and pae_path.exists() else {}
                    if chain_map:
                        _write_pdb_with_chain_map(complex_path, target, chain_map)
                    else:
                        _link_or_copy(complex_path, target)
                else:
                    _link_or_copy(complex_path, target)
            elif complex_path.suffix.lower() == ".cif" or complex_path.name.endswith(".cif.gz"):
                tmp = stage_dir / f"{_safe_id(binder_id)}.{source_label}.raw.pdb"
                refolding_workflow._cif_to_pdb(complex_path, tmp)
                if esmfold2_chain_map:
                    pae_rel = (candidate.get("raw_metadata") or {}).get("pae_path")
                    pae_path = child_dir / str(pae_rel) if pae_rel else None
                    chain_map = _esmfold2_rosetta_chain_map(pae_path) if pae_path and pae_path.exists() else {}
                    if chain_map:
                        _write_pdb_with_chain_map(tmp, target, chain_map)
                        tmp.unlink(missing_ok=True)
                    else:
                        tmp.replace(target)
                else:
                    tmp.replace(target)
            else:
                continue
            count += 1
    return (stage_dir if count else None), count


def _stage_alphafast_pdbs(alphafast_output_dir: Path, stage_dir: Path) -> tuple[Path | None, int]:
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
            refolding_workflow._cif_to_pdb(cif_candidates[0], target)
        count += 1
    return (stage_dir if count else None), count


def _stage_colabfold_pdbs(output_dir: Path) -> Path | None:
    pdb_dir = output_dir / "ColabFold" / "pdbs"
    if pdb_dir.exists() and any(pdb_dir.glob("*.pdb")):
        return pdb_dir
    ptm_dir = output_dir / "ColabFold" / "ptm_output"
    if ptm_dir.exists() and any(ptm_dir.glob("*.pdb")):
        return ptm_dir
    return None


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


def _run_benchmark_metric_postprocessing(
    *,
    job_run_dir: Path,
    run_csv: Path,
    output_dir: Path,
    metric_csvs: list[Path],
    esmfold2_child_runs: list[str],
    af2_child_runs: list[str],
    boltz2_child_runs: list[str],
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

    predicted_pdb_dirs: list[tuple[str, Path]] = []
    af3_pdb_dir, af3_pdb_count = _stage_alphafast_pdbs(
        output_dir / "AF3" / "alphafast_output",
        benchmark_dir / "predicted_metric_pdbs" / "af3",
    )
    if af3_pdb_dir is not None:
        predicted_pdb_dirs.append(("af3", af3_pdb_dir))
        metrics["af3_predicted_metric_pdb_count"] = af3_pdb_count
    colab_pdb_dir = _stage_colabfold_pdbs(output_dir)
    if colab_pdb_dir is not None:
        predicted_pdb_dirs.append(("colab", colab_pdb_dir))
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
        esmfold2_chain_map=True,
    )
    if esmfold2_pdb_dir is not None:
        predicted_pdb_dirs.append(("esmfold2", esmfold2_pdb_dir))
        metrics["esmfold2_predicted_metric_pdb_count"] = esmfold2_pdb_count

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
        if input_pdb_dir.exists() and any(input_pdb_dir.glob("*.pdb")):
            pymol_dirs["input"] = str(input_pdb_dir)
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

    merged = _merge_metric_tables(parent_run_dir=job_run_dir, run_csv=run_csv, metric_csvs=metric_csvs)
    if merged is not None:
        metrics["merged_benchmark_metrics"] = str(merged.relative_to(job_run_dir))
        try:
            merged_df = pd.read_csv(merged)
            feature_rows, summary = _metric_column_summary(merged_df, "binder", max_columns=1000)
            summary_path = benchmark_dir / "merged_benchmark_feature_summary.json"
            feature_table = benchmark_dir / "merged_benchmark_feature_ranking.csv"
            write_json(summary_path, summary)
            pd.DataFrame(feature_rows).to_csv(feature_table, index=False)
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


def _write_alphafast_msas_for_colabfold(*, run_csv: Path, alphafast_output_dir: Path, output_dir: Path) -> dict[str, Any]:
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


def _hide_child_run_for_parent(child_run_dir: Path, parent_run_dir: Path, engine: str) -> None:
    metadata_path = child_run_dir / "metadata.json"
    if not metadata_path.exists():
        return
    metadata = read_json(metadata_path)
    metadata["hidden"] = True
    metadata["parent_task_group"] = BENCHMARK_GROUP
    metadata["parent_run_id"] = parent_run_dir.name
    metadata["parent_run_dir"] = str(parent_run_dir)
    metadata["parent_role"] = "benchmark_engine_subrun"
    metadata["benchmark_engine"] = engine
    write_json(metadata_path, metadata)


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


def _generate_colabfold_inputs_command(*, output_dir: Path, docker_image: str) -> list[str]:
    return [
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
        "--models",
        "colabfold",
    ]


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
    pd.DataFrame(feature_rows).to_csv(out_dir / "alphafast_af3_feature_benchmark.csv", index=False)
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
    image: str,
) -> tuple[int, int, list[list[str]]]:
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
    rc = _run_docker_command(job_run_dir, command)
    return rc, selected_count, [command]


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
    pd.DataFrame(feature_rows).to_csv(out_dir / "colabfold_feature_benchmark.csv", index=False)
    return table, summary_path, summary, [command]


def run_de_novo_binder_scoring_dataset(
    *,
    input_zip: Path | None = None,
    input_zip_bytes: bytes | None = None,
    input_csv: Path | None = None,
    input_csv_text: str | None = None,
    input_pdb_dir: Path | None = None,
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
    run_af2_initial_guess: bool = False,
    af2_num_recycles: int = 3,
    af2_multimer: bool = True,
    af2_use_binder_template: bool = False,
    af2_use_interface_template: bool = False,
    run_boltz2_initial_guess: bool = False,
    boltz2_use_target_template: bool = True,
    boltz2_recycling_steps: int = 10,
    boltz2_sampling_steps: int = 200,
    boltz2_diffusion_samples: int = 3,
    boltz2_write_full_pae: bool = True,
    run_colabfold: bool = False,
    colabfold_cache_dir: Path = COLABFOLD_CACHE_DIR,
    colabfold_msa_source: str = "msa_repository_then_alphafast_mmseqs_gpu",
    msa_repository_dir: Path = MSA_REPOSITORY_DIR,
    colabfold_num_recycles: int = 3,
    colabfold_num_models: int = 3,
    colabfold_gpu_device: int = 0,
    run_alphafast_af3: bool = False,
    alphafast_db_dir: Path = ALPHAFAST_DB_DIR,
    alphafast_weights_dir: Path = ALPHAFAST_WEIGHTS_DIR,
    alphafast_batch_size: int = 0,
    alphafast_num_recycles: int = 10,
    alphafast_gpu_device: int = 0,
    max_records: int = 10,
    num_loops: int = 3,
    num_sampling_steps: int = 32,
    seed: int = 0,
    device: str = "cuda",
    docker_image: str = SCORING_SCRIPTS_IMAGE,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    selected_models = [model for model in (models or ["af3", "boltz", "colabfold"]) if model in {"af3", "boltz", "colabfold"}]
    if run_colabfold and "colabfold" not in selected_models:
        selected_models.append("colabfold")
    if run_colabfold and colabfold_msa_source in {"alphafast_mmseqs_gpu", "msa_repository_then_alphafast_mmseqs_gpu"} and "af3" not in selected_models:
        selected_models.append("af3")
    if run_alphafast_af3 and "af3" not in selected_models:
        selected_models.append("af3")
    job = create_job(
        BENCHMARK_GROUP,
        "de_novo_binder_scoring_dataset",
        "de_novo_binder_scoring_scripts",
        {
            "input_zip": str(input_zip) if input_zip else None,
            "input_zip_uploaded": input_zip_bytes is not None,
            "input_csv": str(input_csv) if input_csv else None,
            "input_pdb_dir": str(input_pdb_dir) if input_pdb_dir else None,
        },
        {
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
            "run_af2_initial_guess": run_af2_initial_guess,
            "af2_num_recycles": af2_num_recycles,
            "af2_multimer": af2_multimer,
            "af2_use_binder_template": af2_use_binder_template,
            "af2_use_interface_template": af2_use_interface_template,
            "run_boltz2_initial_guess": run_boltz2_initial_guess,
            "boltz2_use_target_template": boltz2_use_target_template,
            "boltz2_recycling_steps": boltz2_recycling_steps,
            "boltz2_sampling_steps": boltz2_sampling_steps,
            "boltz2_diffusion_samples": boltz2_diffusion_samples,
            "boltz2_write_full_pae": boltz2_write_full_pae,
            "boltz2_image": BOLTZ2_IMAGE,
            "run_colabfold": run_colabfold,
            "colabfold_image": COLABFOLD_IMAGE,
            "colabfold_cache_dir": str(colabfold_cache_dir),
            "colabfold_msa_source": colabfold_msa_source,
            "msa_repository_dir": str(msa_repository_dir),
            "colabfold_num_recycles": colabfold_num_recycles,
            "colabfold_num_models": colabfold_num_models,
            "colabfold_gpu_device": colabfold_gpu_device,
            "run_alphafast_af3": run_alphafast_af3,
            "alphafast_image": ALPHAFAST_IMAGE,
            "alphafast_db_dir": str(alphafast_db_dir),
            "alphafast_weights_dir": str(alphafast_weights_dir),
            "alphafast_batch_size": alphafast_batch_size,
            "alphafast_num_recycles": alphafast_num_recycles,
            "alphafast_gpu_device": alphafast_gpu_device,
            "docker_image": docker_image,
        },
    )
    raw_dir = job.run_dir / "artifacts" / "raw" / "de_novo_binder_scoring"
    raw_dir.mkdir(parents=True, exist_ok=True)
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
            bool(generate_inputs),
            bool(run_pyrosetta_input_metrics),
            bool(run_esmfold2),
            bool(run_af2_initial_guess),
            bool(run_boltz2_initial_guess),
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
    if run_colabfold and str(colabfold_msa_source) in {"msa_repository", "msa_repository_then_alphafast_mmseqs_gpu"} and run_csv.exists():
        pre_input_metrics.update(
            _apply_msa_repository_to_run_csv(
                run_csv=run_csv,
                msa_repository_dir=Path(msa_repository_dir).expanduser(),
                max_records=int(max_records),
            )
        )
        pre_input_metrics["msa_repository_missing_after_lookup"] = _run_csv_missing_target_msas(run_csv, max_records=int(max_records))
        if str(colabfold_msa_source) == "msa_repository" and pre_input_metrics["msa_repository_missing_after_lookup"]:
            finish_job(job.run_dir, False, {"metrics": pre_input_metrics})
            raise RuntimeError(
                "MSA repository mode was selected, but some selected target-chain MSAs are missing. "
                "Use msa_repository_then_alphafast_mmseqs_gpu to fill misses with AlphaFast/MMseqs GPU."
            )
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
        model_cmd = [
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
        if selected_models:
            model_cmd.extend(["--models", *selected_models])
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
    runtime_size = _run_csv_runtime_size(run_csv, max_records=int(max_records)) if run_csv.exists() else {"candidate_count": 0}
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
    af2_summary: dict[str, Any] = {}
    boltz2_summary: dict[str, Any] = {}
    colabfold_summary: dict[str, Any] = {}
    alphafast_data_pipeline_done = False
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
                use_docker=True,
            )
            esmfold2_child_runs.append(str(child_run))
            _hide_child_run_for_parent(child_run, job.run_dir, "esmfold2")
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
        )
        af2_initial_guess_child_runs.append(str(child_run))
        _hide_child_run_for_parent(child_run, job.run_dir, "af2_initial_guess")
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
        )
        boltz2_initial_guess_child_runs.append(str(child_run))
        _hide_child_run_for_parent(child_run, job.run_dir, "boltz2_initial_guess")
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
        use_alphafast_msa = str(colabfold_msa_source) == "alphafast_mmseqs_gpu" or (
            str(colabfold_msa_source) == "msa_repository_then_alphafast_mmseqs_gpu"
            and _run_csv_missing_target_msas(run_csv, max_records=int(max_records)) > 0
        )
        if use_alphafast_msa:
            _emit_benchmark_progress(
                job.run_dir,
                phase="Preparing target MSAs",
                engine="AlphaFast/MMseqs for ColabFold",
                step=progress_step,
                total_steps=progress_total,
                progress_callback=progress_callback,
            )
            msa_started_at = time.monotonic()
            af3_input_dir = output_dir / "AF3" / "input_folder"
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
            metrics["colabfold_msa_source"] = str(colabfold_msa_source)
            metrics["alphafast_msa_pipeline_return_code"] = rc
            if rc != 0:
                finish_job(job.run_dir, False, {"metrics": metrics})
                raise RuntimeError(f"AlphaFast MMseqs GPU MSA generation failed with return code {rc}.")
            alphafast_data_pipeline_done = True
            _record_runtime_timing(metrics, "alphafast_msa", msa_started_at, **runtime_size)
            msa_metrics = _write_alphafast_msas_for_colabfold(
                run_csv=run_csv,
                alphafast_output_dir=af3_output_dir,
                output_dir=output_dir,
            )
            metrics.update(msa_metrics)
            colab_input_cmd = _generate_colabfold_inputs_command(output_dir=output_dir, docker_image=docker_image)
            commands.append(colab_input_cmd)
            write_json(job.run_dir / "command.json", {"mode": "docker", "commands": commands})
            return_code = _run_docker_command(job.run_dir, colab_input_cmd)
            metrics["regenerate_colabfold_inputs_return_code"] = return_code
            if return_code != 0:
                finish_job(job.run_dir, False, {"metrics": metrics})
                raise RuntimeError(f"ColabFold input regeneration from AlphaFast MSAs failed with return code {return_code}.")
        else:
            metrics["colabfold_msa_source"] = str(colabfold_msa_source)
        rc, selected_count, colabfold_commands = _run_colabfold_prediction(
            job_run_dir=job.run_dir,
            run_csv=run_csv,
            output_dir=output_dir,
            cache_dir=Path(colabfold_cache_dir).expanduser(),
            num_recycles=int(colabfold_num_recycles),
            num_models=int(colabfold_num_models),
            gpu_device=int(colabfold_gpu_device),
            max_records=int(max_records),
            image=COLABFOLD_IMAGE,
        )
        commands.extend(colabfold_commands)
        write_json(job.run_dir / "command.json", {"mode": "docker", "commands": commands})
        metrics["colabfold_return_code"] = rc
        metrics["colabfold_selected_count"] = selected_count
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
    postprocess_metrics, postprocess_commands = _run_benchmark_metric_postprocessing(
        job_run_dir=job.run_dir,
        run_csv=run_csv,
        output_dir=output_dir,
        metric_csvs=metric_csvs,
        esmfold2_child_runs=esmfold2_child_runs,
        af2_child_runs=af2_initial_guess_child_runs,
        boltz2_child_runs=boltz2_initial_guess_child_runs,
        run_common_interface_metrics=bool(run_common_interface_metrics),
        run_predicted_rosetta_metrics=bool(run_predicted_rosetta_metrics),
        run_pymol_metrics=bool(run_pymol_metrics),
        pyrosetta_nprocs=int(pyrosetta_nprocs),
        docker_image=docker_image,
    )
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
    outputs = {
        "run_csv": str(run_csv.relative_to(job.run_dir)) if run_csv.exists() else None,
        "output_dir": str(output_dir.relative_to(job.run_dir)),
        "missing_external_tool_images": "artifacts/benchmark/missing_external_tool_images.json",
        "esmfold2_child_runs": esmfold2_child_runs,
        "af2_initial_guess_child_runs": af2_initial_guess_child_runs,
        "boltz2_initial_guess_child_runs": boltz2_initial_guess_child_runs,
    }
    if af2_summary:
        outputs["af2_initial_guess_summary_metrics"] = af2_summary
    if boltz2_summary:
        outputs["boltz2_initial_guess_summary_metrics"] = boltz2_summary
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
    if metrics.get("merged_benchmark_metrics"):
        outputs["merged_benchmark_metrics"] = metrics.get("merged_benchmark_metrics")
    if metrics.get("merged_benchmark_feature_ranking"):
        outputs["merged_benchmark_feature_ranking"] = metrics.get("merged_benchmark_feature_ranking")
    if (output_dir / "AF3" / "alphafast_output").exists():
        outputs["alphafast_af3_output"] = str((output_dir / "AF3" / "alphafast_output").relative_to(job.run_dir))
    if (output_dir / "ColabFold" / "ptm_output").exists():
        outputs["colabfold_output"] = str((output_dir / "ColabFold" / "ptm_output").relative_to(job.run_dir))
    finish_job(
        job.run_dir,
        True,
        {
            "outputs": outputs,
            "metrics": metrics,
            "downstream_artifacts": {
                "run_csv": outputs.get("run_csv"),
                "output_dir": outputs.get("output_dir"),
                "esmfold2_child_runs": esmfold2_child_runs,
                "af2_initial_guess_child_runs": af2_initial_guess_child_runs,
                "boltz2_initial_guess_child_runs": boltz2_initial_guess_child_runs,
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
) -> None:
    records = json.loads(records_json.read_text())
    esm_binder_workflow._ensure_esm_import_path()
    from esm.models.esmfold2 import (
        DistogramConditioning,
        ESMFold2InputBuilder,
        ProteinInput,
        StructurePredictionInput,
    )

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
                        *[ProteinInput(id=chain, sequence=sequence) for chain, sequence in target_sequences.items()],
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
    max_records: int = 10,
    num_loops: int = 3,
    num_sampling_steps: int = 32,
    seed: int = 0,
    device: str = "cuda",
    contact_cutoff: float = 8.0,
    use_docker: bool = True,
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
            "use_docker": use_docker,
            "image": ESMFOLD2_IMAGE,
        },
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker_cli(sys.argv[1:]))
