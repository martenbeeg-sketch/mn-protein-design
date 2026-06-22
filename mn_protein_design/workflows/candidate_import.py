from __future__ import annotations

import csv
import re
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd

from mn_protein_design.core.candidates import STAGE_GENERATION_BACKBONE_SEQUENCE, write_candidates
from mn_protein_design.core.jobs import create_job, finish_job, update_status, write_json
from mn_protein_design.workflows import refolding as refolding_workflow


CANDIDATE_IMPORT_GROUP = "candidate-import"
_RANKED_PDB_RE = re.compile(r"^(?P<rank>\d+)_(?P<candidate>.+)$")
_STRUCTURE_SUFFIXES = {".pdb", ".cif", ".mmcif"}
_TABLE_SUFFIXES = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}


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


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _clean_sequence(value: object) -> str:
    return "".join(ch for ch in str(value or "").upper() if ch.isalpha())


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
    ranked = _ranked_candidate_from_path(path)
    if ranked is not None:
        _, stem = ranked
    for suffix in ["_unrelaxed_rank_001", "_relaxed_rank_001", "_rank_001", "_model", "_complex"]:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _ranked_candidate_from_path(path: Path) -> tuple[int, str] | None:
    if path.parent.name.lower() != "ranked":
        return None
    match = _RANKED_PDB_RE.match(path.stem)
    if not match:
        return None
    return int(match.group("rank")), match.group("candidate")


def _structure_preference(path: Path) -> tuple[int, str]:
    parts = {part.lower() for part in path.parts}
    if "ranked" in parts:
        return (0, str(path))
    if "accepted" in parts:
        return (1, str(path))
    return (2, str(path))


def _find_structure_files(input_dir: Path, pattern: str) -> list[Path]:
    files = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    return [
        path
        for path in files
        if path.suffix.lower() in _STRUCTURE_SUFFIXES
        and not path.name.startswith(".")
    ]


def _strip_bindcraft_model_suffix(value: object) -> str:
    text = str(value or "").strip()
    return re.sub(r"_model\d+$", "", Path(text).stem)


def _find_bindcraft_final_stats(input_dir: Path) -> Path | None:
    direct = input_dir / "final_design_stats.csv"
    if direct.exists():
        return direct
    matches = sorted(input_dir.glob("**/final_design_stats.csv"))
    return matches[0] if matches else None


def _bindcraft_metric_row(row: dict[str, Any], source_csv: Path) -> dict[str, Any]:
    metrics = {
        f"bindcraft_{key}": _json_safe(value)
        for key, value in row.items()
        if key and str(value).strip() != ""
    }
    rank_value = row.get("Rank") or row.get("rank")
    design_value = row.get("Design") or row.get("design")
    sequence_value = row.get("Sequence") or row.get("sequence")
    if rank_value not in {None, ""}:
        metrics["bindcraft_final_rank"] = _json_safe(rank_value)
        metrics["bindcraft_original_rank"] = _json_safe(rank_value)
    if design_value not in {None, ""}:
        metrics["bindcraft_original_design"] = _json_safe(design_value)
    if sequence_value not in {None, ""}:
        metrics["bindcraft_original_sequence"] = _clean_sequence(sequence_value)
    metrics["bindcraft_metrics_csv"] = str(source_csv)
    metrics["bindcraft_metric_sources"] = str(source_csv)
    metrics["sequence_source"] = "final_design_stats.csv" if sequence_value not in {None, ""} else ""
    return metrics


def _bindcraft_structure_index(input_dir: Path, structure_files: list[Path]) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in structure_files:
        candidate_id = _candidate_id_from_path(path)
        keys = {
            candidate_id,
            path.stem,
            _strip_bindcraft_model_suffix(candidate_id),
            _strip_bindcraft_model_suffix(path.stem),
        }
        ranked = _ranked_candidate_from_path(path)
        if ranked is not None:
            _, ranked_candidate = ranked
            keys.add(ranked_candidate)
            keys.add(_strip_bindcraft_model_suffix(ranked_candidate))
        try:
            relative = path.relative_to(input_dir)
            keys.add(relative.stem)
            keys.add(_strip_bindcraft_model_suffix(relative.stem))
        except ValueError:
            pass
        for key in keys:
            if key:
                index.setdefault(key, []).append(path)
    for key, paths in list(index.items()):
        index[key] = sorted(paths, key=_structure_preference)
    return index


def _canonical_bindcraft_entries(
    input_dir: Path,
    structure_files: list[Path],
    *,
    max_candidates: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    final_stats = _find_bindcraft_final_stats(input_dir)
    if final_stats is None:
        return [], {"mode": "structure_discovery", "final_design_stats": None}
    try:
        table = pd.read_csv(final_stats)
    except Exception as exc:
        return [], {
            "mode": "structure_discovery",
            "final_design_stats": str(final_stats),
            "final_design_stats_error": str(exc),
        }
    if table.empty or "Design" not in table.columns:
        return [], {
            "mode": "structure_discovery",
            "final_design_stats": str(final_stats),
            "final_design_stats_rows": int(len(table)),
            "final_design_stats_error": "Missing Design column.",
        }
    structure_index = _bindcraft_structure_index(input_dir, structure_files)
    entries: list[dict[str, Any]] = []
    missing_structures: list[str] = []
    seen_candidate_ids: set[str] = set()
    table = table.copy()
    if "Rank" in table.columns:
        table["_bindcraft_rank_sort"] = pd.to_numeric(table["Rank"], errors="coerce")
        table = table.sort_values(["_bindcraft_rank_sort", "Design"], na_position="last")
    for _, row_series in table.iterrows():
        row = {str(key): _json_safe(value) for key, value in row_series.items() if not str(key).startswith("_")}
        design = str(row.get("Design") or "").strip()
        if not design:
            continue
        rank_value = row.get("Rank")
        preferred: list[Path] = []
        if rank_value not in {None, ""}:
            try:
                rank_text = str(int(float(str(rank_value))))
            except (TypeError, ValueError):
                rank_text = str(rank_value).strip()
            ranked_dir = input_dir / "Accepted" / "Ranked"
            preferred.extend(sorted(ranked_dir.glob(f"{rank_text}_{design}*.pdb")))
            preferred.extend(sorted(ranked_dir.glob(f"{rank_text}_{design}*.cif")))
        accepted_dir = input_dir / "Accepted"
        preferred.extend(sorted(accepted_dir.glob(f"{design}*.pdb")))
        preferred.extend(sorted(accepted_dir.glob(f"{design}*.cif")))
        preferred.extend(structure_index.get(design, []))
        preferred.extend(structure_index.get(_strip_bindcraft_model_suffix(design), []))
        selected_path = next((path for path in preferred if path.exists()), None)
        if selected_path is None:
            missing_structures.append(design)
            continue
        candidate_id = _candidate_id_from_path(selected_path)
        if candidate_id in seen_candidate_ids:
            continue
        seen_candidate_ids.add(candidate_id)
        metrics = _bindcraft_metric_row(row, final_stats)
        ranked_path = _ranked_candidate_from_path(selected_path)
        if ranked_path is not None:
            metrics["bindcraft_ranked_pdb_rank"] = ranked_path[0]
            metrics["bindcraft_rank_consistent"] = str(metrics.get("bindcraft_original_rank")) == str(ranked_path[0])
        metrics.update(
            {
                "bindcraft_source_folder": str(input_dir),
                "bindcraft_final_stats_csv": str(final_stats),
                "bindcraft_structure_selection": "final_design_stats",
            }
        )
        entries.append(
            {
                "candidate_id": candidate_id,
                "source_pdb": selected_path,
                "metrics": metrics,
                "binder_sequence": _clean_sequence(row.get("Sequence")),
                "original_design": design,
            }
        )
        if max_candidates and int(max_candidates) > 0 and len(entries) >= int(max_candidates):
            break
    return entries, {
        "mode": "final_design_stats",
        "final_design_stats": str(final_stats),
        "final_design_stats_rows": int(len(table)),
        "missing_structure_count": len(missing_structures),
        "missing_structure_examples": missing_structures[:20],
    }


def _read_table(path: Path | None = None, *, data: bytes | None = None, filename: str = "") -> pd.DataFrame:
    name = str(filename or path or "").lower()
    if name.endswith((".xlsx", ".xls")):
        source = BytesIO(data) if data is not None else path
        try:
            return pd.read_excel(source)
        except ImportError as exc:
            raise ImportError("Excel import requires pandas Excel support such as openpyxl.") from exc
    source = BytesIO(data) if data is not None else path
    sep = "\t" if name.endswith((".tsv", ".txt")) else ","
    return pd.read_csv(source, sep=sep)


def _resolve_optional_path(value: object, base_dir: Path | None = None) -> Path | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve() if path.exists() else path


def _stage_target_structure(target_path: Path | None, raw_dir: Path, *, copy_files: bool) -> Path | None:
    if target_path is None:
        return None
    target_path = Path(target_path).expanduser().resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"Target structure not found: {target_path}")
    staged_original = _copy_or_link(target_path, raw_dir / "target" / target_path.name, copy_files=copy_files)
    if staged_original.suffix.lower() == ".pdb":
        return staged_original
    if staged_original.suffix.lower() in {".cif", ".mmcif"} or staged_original.name.endswith(".cif.gz"):
        staged_pdb = raw_dir / "target" / f"{target_path.stem}.pdb"
        return refolding_workflow._cif_to_pdb(staged_original, staged_pdb)
    return staged_original


def _stage_structure(
    source_path: Path | None,
    structure_dir: Path,
    candidate_id: str,
    *,
    role: str,
    copy_files: bool,
) -> Path | None:
    if source_path is None:
        return None
    source_path = Path(source_path).expanduser()
    if not source_path.exists() or not source_path.is_file():
        return None
    suffix = source_path.suffix.lower()
    if suffix not in _STRUCTURE_SUFFIXES and not source_path.name.endswith(".cif.gz"):
        return None
    safe_id = _safe_name(candidate_id)
    target_name = f"{safe_id}_{role}{suffix if suffix else '.pdb'}"
    return _copy_or_link(source_path.resolve(), structure_dir / target_name, copy_files=copy_files)


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
                    rank_key = lower_to_original.get("rank")
                    if rank_key and str(row.get(rank_key) or "").strip():
                        metrics["bindcraft_final_rank"] = row[rank_key]
                    metrics["bindcraft_metrics_csv"] = str(csv_path)
                    for identifier in identifiers:
                        existing = rows_by_id.setdefault(identifier, {})
                        sources = {
                            source
                            for source in str(existing.get("bindcraft_metric_sources") or "").split(";")
                            if source
                        }
                        sources.add(str(csv_path))
                        existing.update(metrics)
                        existing["bindcraft_metric_sources"] = ";".join(sorted(sources))
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


def run_generic_table_import(
    *,
    table_path: Path | None = None,
    table_bytes: bytes | None = None,
    table_filename: str = "",
    base_dir: Path | None = None,
    id_column: str = "",
    sequence_column: str = "",
    complex_path_column: str = "",
    binder_path_column: str = "",
    binder_chains_column: str = "",
    target_chains_column: str = "",
    target_structure: Path | None = None,
    default_binder_chains: list[str] | None = None,
    default_target_chains: list[str] | None = None,
    max_candidates: int = 0,
    copy_files: bool = True,
    import_name: str = "Generic candidate import",
    source_tool: str = "generic_import",
) -> Path:
    if table_path is None and table_bytes is None:
        raise ValueError("Provide a CSV/Excel table path or uploaded table bytes.")
    source_table_path = Path(table_path).expanduser().resolve() if table_path else None
    if source_table_path is not None and not source_table_path.exists():
        raise FileNotFoundError(f"Candidate table not found: {source_table_path}")
    filename = table_filename or (source_table_path.name if source_table_path else "candidate_table.csv")
    if not any(filename.lower().endswith(suffix) for suffix in _TABLE_SUFFIXES):
        raise ValueError("Candidate table must be CSV, TSV, TXT, XLSX, or XLS.")
    source_base_dir = Path(base_dir).expanduser().resolve() if base_dir else (
        source_table_path.parent if source_table_path is not None else None
    )
    table = _read_table(source_table_path, data=table_bytes, filename=filename)
    if table.empty:
        raise ValueError("Candidate table has no rows.")
    if not sequence_column and not binder_path_column and not complex_path_column:
        raise ValueError("Select a sequence column or a structure path column.")

    target_path = Path(target_structure).expanduser().resolve() if target_structure else None
    if target_path is not None and not target_path.exists():
        raise FileNotFoundError(f"Target structure not found: {target_path}")
    selected_binder_chains = default_binder_chains or ["A"]
    selected_target_chains = default_target_chains or []
    job = create_job(
        CANDIDATE_IMPORT_GROUP,
        "candidate_import",
        "generic_table_import",
        {
            "table_path": str(source_table_path) if source_table_path else None,
            "table_uploaded": table_bytes is not None,
            "base_dir": str(source_base_dir) if source_base_dir else None,
            "target_structure": str(target_path) if target_path else None,
        },
        {
            "import_name": import_name,
            "source_tool": source_tool,
            "id_column": id_column,
            "sequence_column": sequence_column,
            "complex_path_column": complex_path_column,
            "binder_path_column": binder_path_column,
            "binder_chains_column": binder_chains_column,
            "target_chains_column": target_chains_column,
            "default_binder_chains": selected_binder_chains,
            "default_target_chains": selected_target_chains,
            "max_candidates": max_candidates,
            "copy_files": copy_files,
        },
    )
    update_status(job.run_dir, "running")
    raw_dir = job.run_dir / "artifacts" / "raw" / "generic_import"
    table_dir = raw_dir / "tables"
    structure_dir = raw_dir / "structures"
    table_dir.mkdir(parents=True, exist_ok=True)
    if table_bytes is not None:
        (table_dir / filename).write_bytes(table_bytes)
    elif source_table_path is not None:
        _copy_or_link(source_table_path, table_dir / source_table_path.name, copy_files=copy_files)
    table_snapshot_path = table_dir / "candidate_table_snapshot.csv"
    table.to_csv(table_snapshot_path, index=False)
    target_staged = _stage_target_structure(target_path, raw_dir, copy_files=copy_files)

    candidates: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    seen_ids: dict[str, int] = {}
    rows = table.to_dict(orient="records")
    if max_candidates and int(max_candidates) > 0:
        rows = rows[: int(max_candidates)]
    for index, row in enumerate(rows, start=1):
        raw_id = row.get(id_column) if id_column else ""
        candidate_id = _safe_name(raw_id or f"candidate_{index:05d}")
        seen_ids[candidate_id] = seen_ids.get(candidate_id, 0) + 1
        if seen_ids[candidate_id] > 1:
            candidate_id = f"{candidate_id}_{seen_ids[candidate_id]:03d}"

        binder_chains = _split_chains(str(row.get(binder_chains_column) or "")) if binder_chains_column else []
        target_chains = _split_chains(str(row.get(target_chains_column) or "")) if target_chains_column else []
        binder_chains = binder_chains or selected_binder_chains
        target_chains = target_chains or selected_target_chains

        complex_source = _resolve_optional_path(row.get(complex_path_column), source_base_dir) if complex_path_column else None
        binder_source = _resolve_optional_path(row.get(binder_path_column), source_base_dir) if binder_path_column else None
        staged_complex = _stage_structure(
            complex_source,
            structure_dir,
            candidate_id,
            role="complex",
            copy_files=copy_files,
        )
        staged_binder = _stage_structure(
            binder_source,
            structure_dir,
            candidate_id,
            role="binder",
            copy_files=copy_files,
        )

        structure_for_sequence = staged_binder or staged_complex
        structure_chains = refolding_workflow._structure_chains(structure_for_sequence) if structure_for_sequence else []
        binder_for_candidate = [chain for chain in binder_chains if not structure_chains or chain in structure_chains]
        if not binder_for_candidate and structure_chains:
            binder_for_candidate = [structure_chains[0]]
        if not binder_for_candidate:
            binder_for_candidate = binder_chains
        target_for_candidate = _target_chain_list(
            complex_chains=structure_chains,
            binder_chains=binder_for_candidate,
            explicit_target_chains=target_chains,
        ) if staged_complex else target_chains

        binder_sequence = _clean_sequence(row.get(sequence_column)) if sequence_column else ""
        if not binder_sequence and structure_for_sequence:
            sequences = refolding_workflow._sequences_by_chain(structure_for_sequence)
            binder_sequence = "".join(sequences.get(chain, "") for chain in binder_for_candidate)
        if not binder_sequence:
            summary_rows.append(
                {
                    "candidate_id": candidate_id,
                    "status": "skipped",
                    "reason": "missing_binder_sequence",
                }
            )
            continue

        row_payload = {str(key): _json_safe(value) for key, value in row.items()}
        mapped_columns = {
            column
            for column in [
                id_column,
                sequence_column,
                complex_path_column,
                binder_path_column,
                binder_chains_column,
                target_chains_column,
            ]
            if column
        }
        metrics = {
            f"import_{key}": value
            for key, value in row_payload.items()
            if key not in mapped_columns and value not in {None, ""}
        }
        metrics.update(
            {
                "import_index": index,
                "import_table_row": index,
                "import_source_table": str(source_table_path or filename),
                "import_has_complex_structure": bool(staged_complex),
                "import_has_binder_structure": bool(staged_binder),
            }
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "stage": STAGE_GENERATION_BACKBONE_SEQUENCE,
                "source_tool": source_tool or "generic_import",
                "target_pdb": _rel_path(job.run_dir, target_staged),
                "complex_pdb": _rel_path(job.run_dir, staged_complex),
                "binder_pdb": _rel_path(job.run_dir, staged_binder),
                "binder_sequence": binder_sequence,
                "binder_chains": binder_for_candidate,
                "target_chains": target_for_candidate,
                "binder_length": str(len(binder_sequence)),
                "metrics": metrics,
                "raw_metadata": {
                    "import_name": import_name,
                    "source_row": row_payload,
                    "source_table": str(source_table_path) if source_table_path else filename,
                    "original_complex_path": str(complex_source) if complex_source else None,
                    "original_binder_path": str(binder_source) if binder_source else None,
                    "structure_chains": structure_chains,
                },
            }
        )
        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "status": "imported",
                "binder_sequence_len": len(binder_sequence),
                "complex_pdb": _rel_path(job.run_dir, staged_complex),
                "binder_pdb": _rel_path(job.run_dir, staged_binder),
                "target_pdb": _rel_path(job.run_dir, target_staged),
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
                "metrics": {"error": "No importable candidates with binder sequences were found."},
                "outputs": {},
            },
        )
        raise ValueError("No importable candidates with binder sequences were found.")
    normalized = write_candidates(job.run_dir, source_tool or "generic_import", candidates)
    normalized_dir = job.run_dir / "artifacts" / "normalized_candidates"
    summary_path = normalized_dir / "import_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    metrics_path = normalized_dir / "imported_candidate_metrics.csv"
    pd.DataFrame(
        [{"candidate_id": candidate.get("candidate_id"), **(candidate.get("metrics") or {})} for candidate in candidates]
    ).to_csv(metrics_path, index=False)
    write_json(
        raw_dir / "import_source.json",
        {
            "import_type": "generic_table",
            "table_path": str(source_table_path) if source_table_path else None,
            "table_filename": filename,
            "base_dir": str(source_base_dir) if source_base_dir else None,
            "target_structure": str(target_path) if target_path else None,
            "candidate_count": len(normalized),
            "columns": list(table.columns),
            "mapping": {
                "id_column": id_column,
                "sequence_column": sequence_column,
                "complex_path_column": complex_path_column,
                "binder_path_column": binder_path_column,
                "binder_chains_column": binder_chains_column,
                "target_chains_column": target_chains_column,
            },
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
                "imported_candidate_metrics": str(metrics_path.relative_to(job.run_dir)),
                "original_table": str((table_dir / filename).relative_to(job.run_dir)),
                "table_snapshot": str(table_snapshot_path.relative_to(job.run_dir)),
                "source_manifest": str((raw_dir / "import_source.json").relative_to(job.run_dir)),
            },
            "metrics": {
                "candidate_count": len(normalized),
                "input_table_rows": int(len(table)),
                "target_pdb_provided": bool(target_staged),
                "copy_files": bool(copy_files),
                "complex_structure_count": sum(1 for candidate in candidates if candidate.get("complex_pdb")),
                "binder_structure_count": sum(1 for candidate in candidates if candidate.get("binder_pdb")),
            },
            "downstream_artifacts": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "import_summary": str(summary_path.relative_to(job.run_dir)),
                "imported_candidate_metrics": str(metrics_path.relative_to(job.run_dir)),
            },
        },
    )
    return job.run_dir


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
    rows_by_id = _load_metric_rows(source_dir)
    canonical_entries, import_manifest = _canonical_bindcraft_entries(
        source_dir,
        structure_files,
        max_candidates=int(max_candidates or 0),
    )
    if canonical_entries:
        selected_entries = canonical_entries
    else:
        deduplicated: dict[str, Path] = {}
        for path in structure_files:
            candidate_id = _candidate_id_from_path(path)
            previous = deduplicated.get(candidate_id)
            if previous is None or _structure_preference(path) < _structure_preference(previous):
                deduplicated[candidate_id] = path
        selected_structure_files = sorted(deduplicated.values(), key=_structure_preference)
        if max_candidates and int(max_candidates) > 0:
            selected_structure_files = selected_structure_files[: int(max_candidates)]
        selected_entries = [
            {
                "candidate_id": _candidate_id_from_path(source_pdb),
                "source_pdb": source_pdb,
                "metrics": {},
                "binder_sequence": None,
                "original_design": _strip_bindcraft_model_suffix(_candidate_id_from_path(source_pdb)),
            }
            for source_pdb in selected_structure_files
        ]
    candidates: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for index, entry in enumerate(selected_entries, start=1):
        source_pdb = Path(entry["source_pdb"])
        candidate_id = str(entry.get("candidate_id") or _candidate_id_from_path(source_pdb))
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
        binder_sequence = (
            _clean_sequence(entry.get("binder_sequence"))
            or "".join(sequences.get(chain, "") for chain in binder_for_candidate)
            or None
        )
        metrics = _metrics_for_candidate(candidate_id, source_pdb, rows_by_id)
        design_key = str(entry.get("original_design") or "")
        if design_key:
            metrics.update(_metrics_for_candidate(design_key, source_pdb, rows_by_id))
        canonical_metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        secondary_sources = {
            source
            for source in str(metrics.get("bindcraft_metric_sources") or "").split(";")
            if source
        }
        metrics.update(canonical_metrics)
        canonical_sources = {
            source
            for source in str(canonical_metrics.get("bindcraft_metric_sources") or "").split(";")
            if source
        }
        all_sources = sorted(secondary_sources | canonical_sources)
        if all_sources:
            metrics["bindcraft_metric_sources"] = ";".join(all_sources)
        ranked_path = _ranked_candidate_from_path(source_pdb)
        if ranked_path is not None:
            metrics["bindcraft_ranked_pdb_rank"] = ranked_path[0]
        if "bindcraft_final_rank" not in metrics and metrics.get("bindcraft_Rank") not in {None, ""}:
            metrics["bindcraft_final_rank"] = metrics["bindcraft_Rank"]
        if "bindcraft_original_rank" not in metrics and metrics.get("bindcraft_final_rank") not in {None, ""}:
            metrics["bindcraft_original_rank"] = metrics["bindcraft_final_rank"]
        if "bindcraft_original_design" not in metrics and design_key:
            metrics["bindcraft_original_design"] = design_key
        if metrics.get("bindcraft_ranked_pdb_rank") not in {None, ""} and metrics.get("bindcraft_original_rank") not in {None, ""}:
            metrics["bindcraft_rank_consistent"] = str(metrics["bindcraft_ranked_pdb_rank"]) == str(
                metrics["bindcraft_original_rank"]
            )
        metrics["bindcraft_import_mode"] = import_manifest.get("mode", "structure_discovery")
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
                "binder_pdb": None,
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
                "bindcraft_final_rank": metrics.get("bindcraft_final_rank"),
                "bindcraft_original_rank": metrics.get("bindcraft_original_rank"),
                "bindcraft_original_design": metrics.get("bindcraft_original_design"),
                "bindcraft_ranked_pdb_rank": metrics.get("bindcraft_ranked_pdb_rank"),
                "bindcraft_rank_consistent": metrics.get("bindcraft_rank_consistent"),
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
    metric_rows = [
        {"candidate_id": candidate.get("candidate_id"), **(candidate.get("metrics") or {})}
        for candidate in candidates
    ]
    metrics_path = job.run_dir / "artifacts" / "normalized_candidates" / "imported_candidate_metrics.csv"
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    write_json(
        raw_dir / "import_source.json",
        {
            "input_dir": str(source_dir),
            "target_pdb": str(target_path) if target_path else None,
            "file_pattern": file_pattern,
            "copy_files": copy_files,
            "candidate_count": len(normalized),
            "bindcraft_import_manifest": import_manifest,
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
                "imported_candidate_metrics": str(metrics_path.relative_to(job.run_dir)),
                "source_manifest": str((raw_dir / "import_source.json").relative_to(job.run_dir)),
            },
            "metrics": {
                "candidate_count": len(normalized),
                "input_structure_count": len(structure_files),
                "canonical_input_rows": import_manifest.get("final_design_stats_rows"),
                "canonical_missing_structures": import_manifest.get("missing_structure_count"),
                "bindcraft_import_mode": import_manifest.get("mode"),
                "bindcraft_ranking_imported": any(
                    candidate.get("metrics", {}).get("bindcraft_final_rank") not in {None, ""}
                    for candidate in candidates
                ),
                "target_pdb_provided": bool(target_staged),
                "copy_files": bool(copy_files),
            },
            "downstream_artifacts": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "import_summary": str(summary_path.relative_to(job.run_dir)),
                "imported_candidate_metrics": str(metrics_path.relative_to(job.run_dir)),
            },
        },
    )
    return job.run_dir


def split_chains(value: str | None) -> list[str]:
    return _split_chains(value)
