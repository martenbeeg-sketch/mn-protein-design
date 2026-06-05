from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from mn_protein_design.core.candidates import STAGE_GENERATION_BACKBONE_SEQUENCE, write_candidates
from mn_protein_design.core.jobs import create_job, finish_job, update_status, write_json
from mn_protein_design.workflows import refolding as refolding_workflow


CANDIDATE_IMPORT_GROUP = "candidate-import"


def _split_chains(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).replace(";", ",").split(",") if part.strip()]


def _rel_path(run_dir: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def _safe_name(value: object) -> str:
    text = str(value or "candidate").strip() or "candidate"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _copy_or_link(source: Path, target: Path, *, copy_files: bool) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return target
    if copy_files:
        shutil.copy2(source, target)
    else:
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)
    return target


def _candidate_id_from_path(path: Path) -> str:
    stem = path.stem
    for suffix in ["_unrelaxed_rank_001", "_relaxed_rank_001", "_rank_001", "_model", "_complex"]:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _find_structure_files(input_dir: Path, pattern: str) -> list[Path]:
    files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    return [
        path
        for path in files
        if path.suffix.lower() in {".pdb", ".cif", ".mmcif"}
        and not path.name.startswith(".")
    ]


def _load_metric_rows(input_dir: Path) -> dict[str, dict[str, Any]]:
    rows_by_id: dict[str, dict[str, Any]] = {}
    id_columns = [
        "binder_id",
        "candidate_id",
        "design",
        "design_id",
        "name",
        "model",
        "description",
        "pdb",
    ]
    for csv_path in sorted(input_dir.glob("**/*.csv")):
        try:
            with csv_path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    row = dict(row)
                    lower_to_original = {str(key).lower(): key for key in row}
                    identifiers: list[str] = []
                    for column in id_columns:
                        row_key = lower_to_original.get(column.lower(), column)
                        value = str(row.get(row_key) or "").strip()
                        if value:
                            identifiers.append(Path(value).stem)
                            identifiers.append(value)
                    if not identifiers:
                        continue
                    id_key_set = set(lower_to_original.get(column.lower(), column) for column in id_columns)
                    metrics = {
                        f"bindcraft_{key}": value
                        for key, value in row.items()
                        if key and key not in id_key_set and str(value).strip() != ""
                    }
                    metrics["bindcraft_metrics_csv"] = str(csv_path)
                    for identifier in identifiers:
                        rows_by_id.setdefault(identifier, {}).update(metrics)
        except Exception:
            continue
    return rows_by_id


def _metrics_for_candidate(candidate_id: str, path: Path, rows_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if candidate_id in rows_by_id:
        return dict(rows_by_id[candidate_id])
    if path.stem in rows_by_id:
        return dict(rows_by_id[path.stem])
    for key, metrics in rows_by_id.items():
        if key and (key in candidate_id or candidate_id in key or key in path.stem or path.stem in key):
            return dict(metrics)
    return {}


def _target_chain_list(
    *,
    complex_chains: list[str],
    binder_chains: list[str],
    explicit_target_chains: list[str],
) -> list[str]:
    if explicit_target_chains:
        return [chain for chain in explicit_target_chains if not complex_chains or chain in complex_chains]
    binder_set = set(binder_chains)
    inferred = [chain for chain in complex_chains if chain not in binder_set]
    return inferred or ["B"]


def run_bindcraft_import(
    *,
    input_dir: Path,
    target_pdb: Path | None = None,
    binder_chains: list[str] | None = None,
    target_chains: list[str] | None = None,
    file_pattern: str = "**/*.pdb",
    max_candidates: int = 0,
    copy_files: bool = True,
    import_name: str = "BindCraft import",
) -> Path:
    source_dir = Path(input_dir).expanduser().resolve()
    if not source_dir.exists() or not source_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {source_dir}")
    target_path = Path(target_pdb).expanduser().resolve() if target_pdb else None
    if target_path is not None and not target_path.exists():
        raise FileNotFoundError(f"Target PDB not found: {target_path}")
    selected_binder_chains = binder_chains or ["A"]
    explicit_target_chains = target_chains or []
    job = create_job(
        CANDIDATE_IMPORT_GROUP,
        "candidate_import",
        "bindcraft_import",
        {
            "input_dir": str(source_dir),
            "target_pdb": str(target_path) if target_path else None,
        },
        {
            "import_name": import_name,
            "binder_chains": selected_binder_chains,
            "target_chains": explicit_target_chains,
            "file_pattern": file_pattern,
            "max_candidates": max_candidates,
            "copy_files": copy_files,
        },
    )
    update_status(job.run_dir, "running")
    raw_dir = job.run_dir / "artifacts" / "raw" / "bindcraft_import"
    structure_dir = raw_dir / "structures"
    target_staged = None
    if target_path:
        target_staged = _copy_or_link(target_path, raw_dir / "target" / target_path.name, copy_files=copy_files)
    structure_files = _find_structure_files(source_dir, file_pattern)
    if target_path:
        structure_files = [path for path in structure_files if path.resolve() != target_path]
    if max_candidates and int(max_candidates) > 0:
        structure_files = structure_files[: int(max_candidates)]
    rows_by_id = _load_metric_rows(source_dir)
    candidates: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for index, source_pdb in enumerate(structure_files, start=1):
        candidate_id = _candidate_id_from_path(source_pdb)
        safe_id = _safe_name(candidate_id)
        staged_structure = _copy_or_link(
            source_pdb,
            structure_dir / f"{safe_id}{source_pdb.suffix.lower()}",
            copy_files=copy_files,
        )
        complex_chains = refolding_workflow._structure_chains(staged_structure)
        binder_for_candidate = [chain for chain in selected_binder_chains if not complex_chains or chain in complex_chains]
        if not binder_for_candidate:
            binder_for_candidate = [complex_chains[0]] if complex_chains else selected_binder_chains
        target_for_candidate = _target_chain_list(
            complex_chains=complex_chains,
            binder_chains=binder_for_candidate,
            explicit_target_chains=explicit_target_chains,
        )
        sequences = refolding_workflow._sequences_by_chain(staged_structure)
        binder_sequence = "".join(sequences.get(chain, "") for chain in binder_for_candidate) or None
        metrics = _metrics_for_candidate(candidate_id, source_pdb, rows_by_id)
        metrics.update(
            {
                "import_index": index,
                "import_source_path": str(source_pdb),
                "import_chain_count": len(complex_chains),
            }
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "stage": STAGE_GENERATION_BACKBONE_SEQUENCE,
                "source_tool": "bindcraft",
                "target_pdb": _rel_path(job.run_dir, target_staged),
                "complex_pdb": _rel_path(job.run_dir, staged_structure),
                "binder_pdb": _rel_path(job.run_dir, staged_structure),
                "binder_sequence": binder_sequence,
                "binder_chains": binder_for_candidate,
                "target_chains": target_for_candidate,
                "metrics": metrics,
                "raw_metadata": {
                    "import_name": import_name,
                    "original_path": str(source_pdb),
                    "structure_chains": complex_chains,
                },
            }
        )
        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "complex_pdb": _rel_path(job.run_dir, staged_structure),
                "binder_sequence": binder_sequence,
                "binder_chains": ",".join(binder_for_candidate),
                "target_chains": ",".join(target_for_candidate),
                "metric_count": len(metrics),
            }
        )
    if not candidates:
        finish_job(
            job.run_dir,
            False,
            {
                "metrics": {"error": f"No structure files matched {file_pattern} in {source_dir}."},
                "outputs": {},
            },
        )
        raise ValueError(f"No structure files matched {file_pattern} in {source_dir}.")
    normalized = write_candidates(job.run_dir, "bindcraft", candidates)
    summary_path = job.run_dir / "artifacts" / "normalized_candidates" / "import_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    write_json(
        raw_dir / "import_source.json",
        {
            "input_dir": str(source_dir),
            "target_pdb": str(target_path) if target_path else None,
            "file_pattern": file_pattern,
            "copy_files": copy_files,
            "candidate_count": len(normalized),
        },
    )
    finish_job(
        job.run_dir,
        True,
        {
            "outputs": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
                "import_summary": str(summary_path.relative_to(job.run_dir)),
                "source_manifest": str((raw_dir / "import_source.json").relative_to(job.run_dir)),
            },
            "metrics": {
                "candidate_count": len(normalized),
                "input_structure_count": len(structure_files),
                "target_pdb_provided": bool(target_staged),
                "copy_files": bool(copy_files),
            },
            "downstream_artifacts": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "import_summary": str(summary_path.relative_to(job.run_dir)),
            },
        },
    )
    return job.run_dir


def split_chains(value: str | None) -> list[str]:
    return _split_chains(value)
