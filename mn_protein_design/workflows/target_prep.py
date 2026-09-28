from __future__ import annotations

import math
from pathlib import Path
import shutil
from typing import Any

from mn_protein_design.core.artifacts import Artifact, artifact_path
from mn_protein_design.core.candidates import STAGE_COMPLEX_REFOLDING, write_candidates
from mn_protein_design.core.jobs import JobPaths, create_job, finish_job, mark_internal_job, read_json, update_status, write_json
from mn_protein_design.core.residue_selection import parse_selection
from mn_protein_design.core.structures import (
    clean_pdb,
    filter_pdb_text,
    map_modified_residues_to_standard,
    pdb_chains,
    pdb_summary,
    replace_nonstandard_residues_with_pdbfixer,
)
from mn_protein_design.workflows import chain_roles
from mn_protein_design.workflows import benchmark as benchmark_workflow
from mn_protein_design.workflows.refolding import AA3_TO_1, _renumber_structure_chain, _sequences_by_chain
from mn_protein_design.workflows.target_msa import ensure_boltz_msas_for_target


TASK_GROUP = "target-prep"
TARGET_REFOLD_FRAGMENT_CHAINS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
DEFAULT_CA_BREAK_DISTANCE = 4.5
DEFAULT_PEPTIDE_BREAK_DISTANCE = 2.0


def _safe_fragment_chain_id(index: int) -> str:
    if index < len(TARGET_REFOLD_FRAGMENT_CHAINS):
        return TARGET_REFOLD_FRAGMENT_CHAINS[index]
    return TARGET_REFOLD_FRAGMENT_CHAINS[-1]


def _chain_id_from_summary_row(row: object) -> str:
    if isinstance(row, dict):
        return str(row.get("chain_id") or "").strip()
    return str(row or "").strip()


def _normalized_chain_summary_rows(rows: object) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not isinstance(rows, list):
        return normalized
    for row in rows:
        if isinstance(row, dict):
            chain_id = _chain_id_from_summary_row(row)
            if not chain_id:
                continue
            normalized.append({**row, "chain_id": chain_id})
            continue
        chain_id = _chain_id_from_summary_row(row)
        if chain_id:
            normalized.append({"chain_id": chain_id})
    return normalized


def _atom_xyz(line: str) -> tuple[float, float, float] | None:
    try:
        return (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    except ValueError:
        return None


def _distance(first: tuple[float, float, float] | None, second: tuple[float, float, float] | None) -> float | None:
    if first is None or second is None:
        return None
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _pdb_chain_residue_fragments(path: Path, chain: str) -> list[dict[str, Any]]:
    residues: list[dict[str, Any]] = []
    residue_lookup: dict[tuple[int, str], dict[str, Any]] = {}
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM  ") or len(line) < 27:
            continue
        line_chain = line[21].strip() or "_"
        if line_chain != chain:
            continue
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        insertion = line[26].strip()
        key = (resseq, insertion)
        residue = residue_lookup.get(key)
        if residue is None:
            residue = {
                "resseq": resseq,
                "insertion": insertion,
                "resname": line[17:20].strip().upper(),
                "lines": [],
                "ca": None,
                "c": None,
                "n": None,
            }
            residue_lookup[key] = residue
            residues.append(residue)
        residue["lines"].append(line)
        atom_name = line[12:16].strip().upper()
        if atom_name in {"CA", "C", "N"}:
            residue[atom_name.lower()] = _atom_xyz(line)
    if not residues:
        return []
    residues.sort(key=lambda item: (int(item["resseq"]), str(item["insertion"])))
    fragments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    breaks: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for residue in residues:
        resseq = int(residue["resseq"])
        missing_count = 0
        ca_distance = None
        peptide_distance = None
        break_reason = ""
        if previous is not None:
            previous_resseq = int(previous["resseq"])
            missing_count = max(0, resseq - previous_resseq - 1)
            ca_distance = _distance(previous.get("ca"), residue.get("ca"))
            peptide_distance = _distance(previous.get("c"), residue.get("n"))
            if missing_count:
                break_reason = "residue_number_gap"
            elif peptide_distance is not None and peptide_distance > DEFAULT_PEPTIDE_BREAK_DISTANCE:
                break_reason = "peptide_bond_gap"
            elif peptide_distance is None and ca_distance is not None and ca_distance > DEFAULT_CA_BREAK_DISTANCE:
                break_reason = "ca_distance_gap"
        if current and break_reason:
            breaks.append(
                {
                    "source_chain": chain,
                    "before_residue": int(previous["resseq"]) if previous is not None else None,
                    "after_residue": resseq,
                    "missing_residues": missing_count,
                    "ca_distance": ca_distance,
                    "peptide_distance": peptide_distance,
                    "reason": break_reason,
                }
            )
            fragments.append(current)
            current = []
        current.append(residue)
        previous = residue
    if current:
        fragments.append(current)
    parsed = [
        {
            "source_chain": chain,
            "source_start": int(fragment[0]["resseq"]),
            "source_end": int(fragment[-1]["resseq"]),
            "missing_before": None if index == 0 else int(fragment[0]["resseq"]) - int(fragments[index - 1][-1]["resseq"]) - 1,
            "break_before": None if index == 0 else breaks[index - 1],
            "sequence": "".join(AA3_TO_1.get(str(residue["resname"]), "X") for residue in fragment).replace("X", ""),
            "residues": fragment,
        }
        for index, fragment in enumerate(fragments)
    ]
    for index, fragment in enumerate(parsed):
        fragment["fragment_index"] = index + 1
    return parsed


def _renumber_fragment_lines(fragment: dict[str, Any], chain_id: str, start_atom: int) -> tuple[list[str], int]:
    lines: list[str] = []
    atom_serial = start_atom
    residue_map: dict[tuple[int, str], int] = {}
    next_resseq = 1
    for residue in fragment.get("residues") or []:
        key = (int(residue["resseq"]), str(residue.get("insertion") or ""))
        if key not in residue_map:
            residue_map[key] = next_resseq
            next_resseq += 1
        for line in residue.get("lines") or []:
            lines.append(f"{line[:6]}{atom_serial:5d}{line[11:21]}{chain_id}{residue_map[key]:4d} {line[27:]}")
            atom_serial += 1
    return lines, atom_serial


def target_chain_break_summary(target_pdb: Path, chain: str) -> dict[str, Any]:
    fragments = _pdb_chain_residue_fragments(Path(target_pdb), str(chain))
    if not fragments:
        sequence = str(_sequences_by_chain(Path(target_pdb)).get(str(chain)) or "").replace("X", "")
        return {"fragment_count": 1 if sequence else 0, "break_count": 0, "fragments": [], "sequence_length": len(sequence)}
    break_rows = [
        fragment.get("break_before")
        for fragment in fragments[1:]
        if isinstance(fragment.get("break_before"), dict)
    ]
    return {
        "fragment_count": len(fragments),
        "break_count": max(0, len(fragments) - 1),
        "residue_number_break_count": sum(1 for row in break_rows if row.get("reason") == "residue_number_gap"),
        "coordinate_break_count": sum(1 for row in break_rows if row.get("reason") in {"ca_distance_gap", "peptide_bond_gap"}),
        "sequence_length": sum(len(str(fragment.get("sequence") or "")) for fragment in fragments),
        "breaks": break_rows,
        "fragments": [
            {
                "source_chain": fragment["source_chain"],
                "fragment_index": fragment.get("fragment_index"),
                "source_start": fragment["source_start"],
                "source_end": fragment["source_end"],
                "sequence_length": len(str(fragment.get("sequence") or "")),
                "missing_before": fragment.get("missing_before"),
                "break_before": fragment.get("break_before"),
                "sequence": fragment.get("sequence"),
            }
            for fragment in fragments
        ],
    }


def analyze_target_chain(target_pdb: Path, chain: str) -> dict[str, Any]:
    summary = target_chain_break_summary(target_pdb, chain)
    warnings: list[str] = []
    breaks = [row for row in summary.get("breaks") or [] if isinstance(row, dict)]
    likely_concatenation = any(
        int(row.get("missing_residues") or 0) == 0
        and float(row.get("peptide_distance") or row.get("ca_distance") or 0.0) > 10.0
        for row in breaks
    )
    if likely_concatenation:
        warnings.append("This looks like separate structural chains or domains were concatenated into one PDB chain.")
    if int(summary.get("coordinate_break_count") or 0):
        warnings.append("Backbone coordinates are discontinuous although residue numbering may be continuous.")
    if int(summary.get("residue_number_break_count") or 0):
        warnings.append("Residue numbering has explicit gaps.")
    if int(summary.get("fragment_count") or 0) > 1:
        warnings.append("Downstream folding should treat disconnected fragments explicitly or use a repaired target.")
    return {**summary, "warnings": warnings}


def split_target_chain_fragments(
    source_pdb: Path,
    chain: str,
    *,
    target_name: str | None = None,
    source_label: str | None = None,
) -> Path:
    source_pdb = Path(source_pdb).expanduser().resolve()
    chain = str(chain or "").strip()
    if not source_pdb.exists():
        raise FileNotFoundError(f"Target PDB does not exist: {source_pdb}")
    if not chain:
        raise ValueError("Select a source chain to split.")
    fragments = _pdb_chain_residue_fragments(source_pdb, chain)
    if len(fragments) < 2:
        raise ValueError(f"Chain {chain} does not contain multiple detected fragments.")
    if len(fragments) > len(TARGET_REFOLD_FRAGMENT_CHAINS):
        raise ValueError(f"Too many fragments to assign one-character PDB chain IDs: {len(fragments)}")

    default_name = f"{source_pdb.stem} chain {chain} split fragments"
    target_name = str(target_name or default_name).strip() or default_name
    job = create_job(
        TASK_GROUP,
        job_type="target_preparation",
        tool="target_fragment_splitter",
        inputs={"source_pdb": str(source_pdb), "target_name": target_name, "source_chain": chain, "source_label": source_label or ""},
        params={
            "mode": "split_detected_chain_fragments",
            "ca_distance_angstrom": DEFAULT_CA_BREAK_DISTANCE,
            "peptide_distance_angstrom": DEFAULT_PEPTIDE_BREAK_DISTANCE,
        },
    )
    update_status(job.run_dir, "running")
    write_json(
        job.run_dir / "command.json",
        {"mode": "internal", "command": ["target_fragment_splitter"], "source_pdb": str(source_pdb), "source_chain": chain},
    )

    try:
        raw_target = artifact_path(job.run_dir, "target_input.pdb")
        split_target = artifact_path(job.run_dir, "target_split_fragments.pdb")
        cleaned_target = artifact_path(job.run_dir, "target_clean.pdb")
        trimmed_target = artifact_path(job.run_dir, "target_trimmed.pdb")
        target_json = artifact_path(job.run_dir, "target.json")
        shutil.copyfile(source_pdb, raw_target)

        next_atom = 1
        split_lines: list[str] = []
        fragment_rows: list[dict[str, Any]] = []
        for index, fragment in enumerate(fragments):
            new_chain = _safe_fragment_chain_id(index)
            fragment_lines, next_atom = _renumber_fragment_lines(fragment, new_chain, next_atom)
            if not fragment_lines:
                continue
            split_lines.extend(fragment_lines + ["TER"])
            fragment_rows.append(
                {
                    "source_chain": chain,
                    "source_start": fragment.get("source_start"),
                    "source_end": fragment.get("source_end"),
                    "new_chain": new_chain,
                    "sequence_length": len(str(fragment.get("sequence") or "")),
                    "sequence": fragment.get("sequence"),
                    "break_before": fragment.get("break_before"),
                }
            )
        if not split_lines:
            raise ValueError(f"Chain {chain} in {source_pdb} did not produce split ATOM records.")
        split_target.write_text("\n".join(split_lines + ["END", ""]))
        shutil.copyfile(split_target, cleaned_target)
        shutil.copyfile(split_target, trimmed_target)

        split_summary = pdb_summary(trimmed_target.read_text(errors="ignore"))
        chains = split_summary.get("chains") or []
        residue_count = sum(int(row.get("residue_count") or 0) for row in chains if isinstance(row, dict))
        target_payload = {
            "target_name": target_name,
            "source_pdb": str(source_pdb),
            "source_chain": chain,
            "chain_role_schema": chain_roles.CHAIN_ROLE_SCHEMA_EXPLICIT_V2,
            "chain_roles": {
                "schema": chain_roles.CHAIN_ROLE_SCHEMA_EXPLICIT_V2,
                "target_chains": [str(row.get("new_chain") or "") for row in fragment_rows if str(row.get("new_chain") or "")],
                "binder_chains": [],
                "roles": {
                    str(row.get("new_chain") or ""): "target"
                    for row in fragment_rows
                    if str(row.get("new_chain") or "")
                },
                "chain_map": {
                    "binder": [],
                    "targets": [
                        {
                            "original_chain": chain,
                            "engine_chain": str(row.get("new_chain") or ""),
                            "role": "target",
                            "source_start": row.get("source_start"),
                            "source_end": row.get("source_end"),
                        }
                        for row in fragment_rows
                        if str(row.get("new_chain") or "")
                    ],
                },
            },
            "chains": chains,
            "residue_count": residue_count,
            "prepared_kind": "split_fragments",
            "source_label": source_label or "Target fragment split",
            "target_entities": [
                {
                    "entity_id": "target",
                    "role": "target",
                    "source_chain": chain,
                    "chains": [str(row.get("new_chain") or "") for row in fragment_rows if str(row.get("new_chain") or "")],
                    "fragmented_from_single_chain": True,
                }
            ],
            "fragment_split": {
                "source_pdb": str(source_pdb),
                "source_chain": chain,
                "biological_role": "target",
                "fragments": fragment_rows,
            },
            "artifacts": {
                "target_input": "artifacts/target_input.pdb",
                "target_split_fragments": "artifacts/target_split_fragments.pdb",
                "target_clean": "artifacts/target_clean.pdb",
                "target_trimmed": "artifacts/target_trimmed.pdb",
            },
        }
        write_json(target_json, target_payload)
        artifacts = [
            Artifact("target_input", raw_target, "pdb", "Original source target PDB").to_json(job.run_dir),
            Artifact("target_split_fragments", split_target, "pdb", "Detected fragments split into separate chains").to_json(job.run_dir),
            Artifact("target_clean", cleaned_target, "pdb", "Split target PDB").to_json(job.run_dir),
            Artifact("target_trimmed", trimmed_target, "pdb", "Split target PDB for downstream tools").to_json(job.run_dir),
            Artifact("target_json", target_json, "target_manifest", "Split-fragment target manifest").to_json(job.run_dir),
        ]
        finish_job(
            job.run_dir,
            True,
            {
                "outputs": {"artifacts": artifacts, "target": target_payload},
                "metrics": {
                    "residue_count": residue_count,
                    "source_fragment_count": len(fragments),
                    "split_chain_count": len(fragment_rows),
                    "source_break_count": max(0, len(fragments) - 1),
                },
                "downstream_artifacts": {
                    "target_clean_pdb": "artifacts/target_clean.pdb",
                    "target_trimmed_pdb": "artifacts/target_trimmed.pdb",
                    "target_json": "artifacts/target.json",
                },
            },
        )
    except Exception as exc:
        (job.run_dir / "stderr.log").write_text(f"{type(exc).__name__}: {exc}\n")
        finish_job(job.run_dir, False, {"outputs": {}, "metrics": {}, "error": str(exc)})
        raise
    return job.run_dir


def prepare_target(
    source_pdb: Path,
    target_name: str,
    keep_chains: list[str] | None = None,
    residue_range_text: str = "",
    remove_waters: bool = True,
    remove_hetero: bool = False,
    map_known_modified_residues: bool = False,
    replace_nonstandard_residues: bool = False,
    prepare_msa: bool = True,
    existing_job: JobPaths | None = None,
) -> Path:
    source_pdb = source_pdb.expanduser().resolve()
    if not source_pdb.exists():
        raise FileNotFoundError(f"Input PDB does not exist: {source_pdb}")
    if source_pdb.suffix.lower() not in {".pdb", ".ent"}:
        raise ValueError("Target preparation currently expects a PDB file.")

    keep_chain_set = {c.strip() for c in keep_chains or [] if c.strip()} or None
    residue_selections = parse_selection(residue_range_text)
    inputs = {
        "source_pdb": str(source_pdb),
        "target_name": target_name,
        "keep_chains": sorted(keep_chain_set or []),
        "residue_selections": [s.to_json() for s in residue_selections],
    }
    params = {
        "remove_waters": remove_waters,
        "remove_hetero": remove_hetero,
        "map_known_modified_residues": map_known_modified_residues,
        "replace_nonstandard_residues": replace_nonstandard_residues,
        "prepare_msa": prepare_msa,
    }
    job = existing_job or create_job(TASK_GROUP, job_type="target_preparation", tool="internal_pdb_cleaner", inputs=inputs, params=params)
    update_status(job.run_dir, "running")
    write_json(
        job.run_dir / "command.json",
        {"mode": "internal", "command": ["internal_pdb_cleaner"], "source_pdb": str(source_pdb)},
    )

    try:
        raw_target = artifact_path(job.run_dir, "target_input.pdb")
        mapped_target = artifact_path(job.run_dir, "target_mapped.pdb")
        repaired_target = artifact_path(job.run_dir, "target_repaired.pdb")
        cleaned_target = artifact_path(job.run_dir, "target_clean.pdb")
        trimmed_target = artifact_path(job.run_dir, "target_trimmed.pdb")
        target_json = artifact_path(job.run_dir, "target.json")
        shutil.copyfile(source_pdb, raw_target)
        mapping_metrics = {}
        repair_metrics = {}
        clean_input = raw_target
        current_text = raw_target.read_text(errors="ignore")
        if map_known_modified_residues:
            mapped_text, mapping_metrics = map_modified_residues_to_standard(current_text)
            mapped_target.write_text(mapped_text)
            clean_input = mapped_target
            current_text = mapped_text
        if replace_nonstandard_residues:
            repaired_text, repair_metrics = replace_nonstandard_residues_with_pdbfixer(current_text)
            repaired_target.write_text(repaired_text)
            clean_input = repaired_target
        stats = clean_pdb(
            clean_input,
            cleaned_target,
            keep_chains=keep_chain_set,
            remove_waters=remove_waters,
            remove_hetero=remove_hetero,
            residue_selections=residue_selections,
        )
        shutil.copyfile(cleaned_target, trimmed_target)
        cleaned_summary = pdb_summary(cleaned_target.read_text(errors="ignore"))
        chain_rows = _normalized_chain_summary_rows(cleaned_summary.get("chains") or stats.get("chains") or [])
        target_chains = [_chain_id_from_summary_row(row) for row in chain_rows if _chain_id_from_summary_row(row)]
        target_payload = {
            "target_name": target_name,
            "source_pdb": str(source_pdb),
            "chains": chain_rows,
            "chain_role_schema": chain_roles.CHAIN_ROLE_SCHEMA_EXPLICIT_V2,
            "chain_roles": {
                "schema": chain_roles.CHAIN_ROLE_SCHEMA_EXPLICIT_V2,
                "target_chains": target_chains,
                "binder_chains": [],
                "roles": {chain: "target" for chain in target_chains},
                "chain_map": {
                    "binder": [],
                    "targets": [
                        {
                            "original_chain": chain,
                            "engine_chain": chain,
                            "role": "target",
                        }
                        for chain in target_chains
                    ],
                },
            },
            "residue_count": stats["residue_count"],
            "artifacts": {
                "target_clean": "artifacts/target_clean.pdb",
                "target_trimmed": "artifacts/target_trimmed.pdb",
            },
        }
        msa_by_chain = {}
        source_msa_by_chain = {}
        msa_error = ""
        if prepare_msa:
            try:
                source_msa_by_chain = ensure_boltz_msas_for_target(
                    job.run_dir,
                    raw_target,
                    sorted(keep_chain_set or []),
                    raw_subdir="source_target_msa",
                )
                msa_by_chain = ensure_boltz_msas_for_target(
                    job.run_dir,
                    trimmed_target,
                    sorted(keep_chain_set or []),
                    raw_subdir="target_msa",
                )
                target_payload["boltz_msa_by_chain"] = msa_by_chain
                target_payload["source_boltz_msa_by_chain"] = source_msa_by_chain
            except Exception as exc:
                msa_error = str(exc)
                with (job.run_dir / "stderr.log").open("a") as stderr:
                    stderr.write(f"Target MSA preparation failed: {type(exc).__name__}: {exc}\n")
        if map_known_modified_residues:
            target_payload["artifacts"]["target_mapped"] = "artifacts/target_mapped.pdb"
        if replace_nonstandard_residues:
            target_payload["artifacts"]["target_repaired"] = "artifacts/target_repaired.pdb"
        write_json(target_json, target_payload)
        artifacts = [
            Artifact("target_input", raw_target, "pdb", "Original uploaded PDB").to_json(job.run_dir),
            *(
                [
                    Artifact(
                        "target_mapped",
                        mapped_target,
                        "pdb",
                        "Known modified residues mapped to canonical amino acids",
                    ).to_json(job.run_dir)
                ]
                if map_known_modified_residues
                else []
            ),
            *(
                [Artifact("target_repaired", repaired_target, "pdb", "PDBFixer output with nonstandard residues replaced").to_json(job.run_dir)]
                if replace_nonstandard_residues
                else []
            ),
            Artifact("target_clean", cleaned_target, "pdb", "Cleaned target PDB").to_json(job.run_dir),
            Artifact("target_trimmed", trimmed_target, "pdb", "Chain-trimmed target PDB").to_json(job.run_dir),
            Artifact("target_json", target_json, "target_manifest", "Prepared target manifest").to_json(job.run_dir),
        ]
        finish_job(
            job.run_dir,
            True,
            {
                "outputs": {"artifacts": artifacts, "target": target_payload},
                "metrics": {**stats, **mapping_metrics, **repair_metrics},
                "target_msa_by_chain": msa_by_chain,
                "source_target_msa_by_chain": source_msa_by_chain,
                "target_msa_error": msa_error,
                "downstream_artifacts": {
                    "target_clean_pdb": "artifacts/target_clean.pdb",
                    "target_trimmed_pdb": "artifacts/target_trimmed.pdb",
                    "target_msa_by_chain": msa_by_chain,
                    "source_target_msa_by_chain": source_msa_by_chain,
                    **({"target_mapped_pdb": "artifacts/target_mapped.pdb"} if map_known_modified_residues else {}),
                    **({"target_repaired_pdb": "artifacts/target_repaired.pdb"} if replace_nonstandard_residues else {}),
                    "target_json": "artifacts/target.json",
                },
            },
        )
    except Exception as exc:
        (job.run_dir / "stderr.log").write_text(f"{type(exc).__name__}: {exc}\n")
        finish_job(job.run_dir, False, {"outputs": {}, "metrics": {}, "error": str(exc)})
        raise
    return job.run_dir


def enqueue_target_preparation(
    source_pdb: Path,
    target_name: str,
    keep_chains: list[str] | None = None,
    residue_range_text: str = "",
    remove_waters: bool = True,
    remove_hetero: bool = False,
    map_known_modified_residues: bool = False,
    replace_nonstandard_residues: bool = False,
    prepare_msa: bool = True,
) -> Path:
    source_pdb = source_pdb.expanduser().resolve()
    keep_chain_set = {c.strip() for c in keep_chains or [] if c.strip()} or None
    residue_selections = parse_selection(residue_range_text)
    inputs = {
        "source_pdb": str(source_pdb),
        "target_name": target_name,
        "keep_chains": sorted(keep_chain_set or []),
        "residue_selections": [s.to_json() for s in residue_selections],
    }
    params = {
        "remove_waters": remove_waters,
        "remove_hetero": remove_hetero,
        "map_known_modified_residues": map_known_modified_residues,
        "replace_nonstandard_residues": replace_nonstandard_residues,
        "prepare_msa": prepare_msa,
    }
    job = create_job(TASK_GROUP, job_type="target_preparation", tool="internal_pdb_cleaner", inputs=inputs, params=params)
    write_json(
        job.run_dir / "worker_request.json",
        {
            "kind": "target_preparation",
            "kwargs": {
                "source_pdb": str(source_pdb),
                "target_name": target_name,
                "keep_chains": sorted(keep_chain_set or []),
                "residue_range_text": residue_range_text,
                **params,
            },
            "path_kwargs": ["source_pdb"],
        },
    )
    update_status(job.run_dir, "queued", current_phase="Waiting for target-preparation worker")
    return job.run_dir


def run_queued_target_preparation(run_dir: Path, **kwargs: Any) -> None:
    metadata = read_json(run_dir / "metadata.json")
    job = JobPaths(
        task_group=str(metadata.get("task_group") or TASK_GROUP),
        run_id=run_dir.name,
        run_dir=run_dir,
    )
    prepare_target(existing_job=job, **kwargs)


def inspect_pdb(path: Path) -> dict:
    return {"chains": pdb_chains(path)}


def create_target_refolding_candidate_set(
    target_entries: list[dict[str, Any]],
    *,
    name: str = "Target refolding set",
    split_chain_breaks: bool = False,
    chain_break_mode: str | None = None,
) -> Path:
    """Create a normalized target-only candidate set for all-engine folding."""
    if not target_entries:
        raise ValueError("Select at least one target chain for refolding.")
    chain_break_mode = str(
        chain_break_mode
        or ("split_fragments" if split_chain_breaks else "preserve_original_chain")
    )

    job = create_job(
        "target-prep",
        job_type="target_refolding_candidate_set",
        tool="target_refolding_input_builder",
        inputs={"targets": target_entries, "name": name},
        params={
            "mode": "target_chain_as_foldable_sequence",
            "target_refolding": True,
            "campaign_name": name,
            "split_chain_breaks": bool(split_chain_breaks),
            "chain_break_mode": chain_break_mode,
        },
    )
    update_status(job.run_dir, "running")
    write_json(job.run_dir / "command.json", {"mode": "internal", "command": ["target_refolding_input_builder"]})

    try:
        systems_dir = job.run_dir / "artifacts" / "target_refolding_inputs"
        systems_dir.mkdir(parents=True, exist_ok=True)
        candidates: list[dict[str, Any]] = []
        manifest_rows: list[dict[str, Any]] = []
        for index, entry in enumerate(target_entries, start=1):
            target_pdb = Path(str(entry.get("target_pdb") or "")).expanduser()
            if not target_pdb.exists():
                raise FileNotFoundError(f"Target PDB does not exist: {target_pdb}")
            chain = str(entry.get("chain") or "").strip()
            if not chain:
                raise ValueError("Each target entry needs a chain.")
            target_entity_chains = [str(value) for value in (entry.get("target_entity_chains") or []) if str(value)]
            entity_chain_mode = len(target_entity_chains) > 1 and chain in set(target_entity_chains)
            parsed_fragments = [] if entity_chain_mode else (_pdb_chain_residue_fragments(target_pdb, chain) if split_chain_breaks else [])
            if not parsed_fragments:
                sequences_by_chain = _sequences_by_chain(target_pdb)
                source_chains = target_entity_chains or [chain]
                parsed_fragments = [
                    {
                        "source_chain": source_chain,
                        "source_start": None,
                        "source_end": None,
                        "missing_before": None,
                        "sequence": str(sequences_by_chain.get(source_chain) or "").replace("X", "").strip(),
                        "residues": [],
                    }
                    for source_chain in source_chains
                ]
            parsed_fragments = [fragment for fragment in parsed_fragments if str(fragment.get("sequence") or "").strip()]
            if not parsed_fragments:
                raise ValueError(f"Could not extract a protein sequence for chain {chain} in {target_pdb}.")
            split_fragment_count = len(parsed_fragments) if split_chain_breaks or entity_chain_mode else 1
            target_name = str(entry.get("target_name") or target_pdb.stem).strip() or target_pdb.stem
            source_category = str(entry.get("source_category") or "target").strip() or "target"
            safe_target = "".join(ch.lower() if ch.isalnum() else "_" for ch in target_name)
            safe_target = "_".join(part for part in safe_target.split("_") if part) or f"target_{index}"
            safe_chain = "".join(ch.lower() if ch.isalnum() else "_" for ch in chain) or "chain"
            candidate_id = f"target_refold_{safe_target}_{safe_chain}_{index}"
            staged_chain_ids = [_safe_fragment_chain_id(fragment_index) for fragment_index in range(len(parsed_fragments))]
            complex_pdb = systems_dir / f"{candidate_id}.pdb"
            split_into_fragments = bool((split_chain_breaks or entity_chain_mode) and split_fragment_count > 1)
            if split_into_fragments:
                next_atom = 1
                fragment_lines: list[str] = []
                for fragment, engine_chain in zip(parsed_fragments, staged_chain_ids):
                    if fragment.get("residues"):
                        chain_lines, next_atom = _renumber_fragment_lines(fragment, engine_chain, next_atom)
                    else:
                        chain_lines, next_atom = _renumber_structure_chain(
                            target_pdb,
                            engine_chain,
                            next_atom,
                            {str(fragment.get("source_chain") or "")},
                        )
                    if chain_lines:
                        fragment_lines.extend(chain_lines + ["TER"])
                if not fragment_lines:
                    raise ValueError(f"Chain {chain} in {target_pdb} did not produce protein ATOM records.")
                complex_pdb.write_text("\n".join(fragment_lines + ["END", ""]))
            else:
                target_text = filter_pdb_text(
                    target_pdb.read_text(errors="ignore"),
                    keep_chains={chain},
                    remove_waters=True,
                    remove_hetero=True,
                ).rstrip()
                if "ATOM" not in target_text:
                    raise ValueError(f"Chain {chain} in {target_pdb} did not produce protein ATOM records.")
                complex_pdb.write_text(target_text + "\nEND\n")
                staged_chain_ids = [chain]
            sequence = "".join(str(fragment.get("sequence") or "") for fragment in parsed_fragments)
            target_sequence = sequence
            target_chains = list(staged_chain_ids)
            legacy_binder_chains = list(staged_chain_ids)
            biological_target_chains = list(staged_chain_ids)
            role_map = chain_roles.build_explicit_role_map(
                binder_source_chains=[],
                binder_engine_chains=[],
                target_source_chains=[str(fragment.get("source_chain") or "") for fragment in parsed_fragments],
                target_engine_chains=target_chains,
                target_fragments=[
                    {
                        "source_chain": fragment.get("source_chain"),
                        "engine_chain": engine_chain,
                        "source_start": fragment.get("source_start"),
                        "source_end": fragment.get("source_end"),
                        "sequence_length": len(str(fragment.get("sequence") or "")),
                        "role": "target",
                        "missing_before": fragment.get("missing_before"),
                    }
                    for fragment, engine_chain in zip(parsed_fragments, staged_chain_ids)
                ],
            )
            manifest_row = {
                "candidate_id": candidate_id,
                "target_name": target_name,
                "target_pdb": str(target_pdb),
                "chain": chain,
                "target_entity_chains": target_entity_chains or [chain],
                "biological_target_chains": biological_target_chains,
                "staged_target_chains": target_chains,
                "sequence_length": len(sequence),
                "fragment_count": len(parsed_fragments),
                "chain_break_count": max(0, len(parsed_fragments) - 1),
                "split_chain_breaks": bool(split_into_fragments),
                "chain_break_mode": chain_break_mode,
                "source_category": source_category,
                "source_label": entry.get("source_label") or "",
            }
            manifest_rows.append(manifest_row)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "stage": STAGE_COMPLEX_REFOLDING,
                    "source_tool": "target_refolding_input_builder",
                    "target_pdb": str(complex_pdb.relative_to(job.run_dir)) if split_into_fragments else str(target_pdb),
                    "complex_pdb": str(complex_pdb.relative_to(job.run_dir)),
                    # Legacy compatibility: existing benchmark/refolding adapters
                    # still read a foldable sequence from binder_sequence for
                    # target-only rows. The explicit role metadata below is the
                    # canonical target-only contract.
                    "binder_sequence": target_sequence,
                    "target_sequence": target_sequence,
                    "binder_chains": [],
                    "legacy_capacity_binder_chains": legacy_binder_chains,
                    "target_chains": target_chains,
                    "target_only": True,
                    "chain_role_schema": role_map.schema,
                    "chain_roles": role_map.to_dict(),
                    "binder_length": "0",
                    "metrics": {
                        "target_id": target_name,
                        "source_chain": chain,
                        "sequence_length": len(sequence),
                        "target_length": len(sequence),
                        "binder_length": 0,
                        "total_length": len(sequence),
                        "fragment_count": len(parsed_fragments),
                        "chain_break_count": max(0, len(parsed_fragments) - 1),
                    },
                    "raw_metadata": {
                        "target_refolding": True,
                        "capacity_target_only": True,
                        "biological_role": "target",
                        "chain_role_schema": role_map.schema,
                        "chain_roles": role_map.to_dict(),
                        "target_entity_chains": target_entity_chains or [chain],
                        "biological_target_chains": biological_target_chains,
                        "staged_target_chains": target_chains,
                        "legacy_capacity_chain_field": "legacy_capacity_binder_chains",
                        "legacy_capacity_binder_chains": legacy_binder_chains,
                        "target_source_pdb": str(target_pdb),
                        "source_chain": chain,
                        "split_chain_breaks": bool(split_into_fragments),
                        "chain_break_mode": chain_break_mode,
                        "fragment_count": len(parsed_fragments),
                        "chain_break_count": max(0, len(parsed_fragments) - 1),
                        "target_refolding_fragments": [
                            {
                                "source_chain": fragment.get("source_chain"),
                                "engine_chain": engine_chain,
                                "source_start": fragment.get("source_start"),
                                "source_end": fragment.get("source_end"),
                                "sequence_length": len(str(fragment.get("sequence") or "")),
                                "role": "target",
                                "missing_before": fragment.get("missing_before"),
                            }
                            for fragment, engine_chain in zip(parsed_fragments, staged_chain_ids)
                        ],
                        "source_category": source_category,
                        "source_label": entry.get("source_label") or "",
                    },
                }
            )
        normalized = write_candidates(job.run_dir, "target_refolding_input_builder", candidates)
        write_json(job.run_dir / "artifacts" / "target_refolding_manifest.json", {"targets": manifest_rows})
        finish_job(
            job.run_dir,
            True,
            {
                "outputs": {
                    "target_refolding_manifest": "artifacts/target_refolding_manifest.json",
                    "candidate_count": len(normalized),
                },
                "metrics": {
                    "target_structure_count": len(normalized),
                    "target_chain_count": sum(len(row.get("staged_target_chains") or []) for row in manifest_rows),
                    "target_residue_count": sum(int(row.get("sequence_length") or 0) for row in manifest_rows),
                },
                "target_refolding": {
                    "split_chain_breaks": bool(split_chain_breaks),
                    "chain_break_mode": chain_break_mode,
                    "split_fragment_candidates": sum(
                        1
                        for row in manifest_rows
                        if int(row.get("fragment_count") or 0) > 1 and bool(row.get("split_chain_breaks"))
                    ),
                },
                "downstream_artifacts": {
                    "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                    "target_refolding_manifest": "artifacts/target_refolding_manifest.json",
                },
            },
        )
    except Exception as exc:
        (job.run_dir / "stderr.log").write_text(f"{type(exc).__name__}: {exc}\n")
        finish_job(job.run_dir, False, {"outputs": {}, "metrics": {}, "error": str(exc)})
        raise
    return job.run_dir


def enqueue_target_refolding_evaluation(
    target_entries: list[dict[str, Any]],
    *,
    evaluation_name: str = "Target refolding evaluation",
    **kwargs: Any,
) -> tuple[Path, Path]:
    """Build target-only candidates, then queue benchmark-style all-engine refolding."""
    split_chain_breaks = bool(kwargs.pop("split_chain_breaks", False))
    chain_break_mode = str(
        kwargs.pop(
            "chain_break_mode",
            "split_fragments" if split_chain_breaks else "preserve_original_chain",
        )
    )
    kwargs.setdefault("af2_multimer", True)
    kwargs.setdefault("af2_use_initial_guess", True)
    kwargs.setdefault("af2_use_binder_template", True)
    kwargs.setdefault("af2_use_interface_template", True)
    kwargs.setdefault("boltz2_use_target_template", True)
    source_run_dir = create_target_refolding_candidate_set(
        target_entries,
        name=evaluation_name,
        split_chain_breaks=split_chain_breaks,
        chain_break_mode=chain_break_mode,
    )
    run_dir = benchmark_workflow.enqueue_candidate_refolding_evaluation(
        source_run_dir=source_run_dir,
        candidates_jsonl=source_run_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl",
        selected_candidate_ids=[],
        max_candidates=0,
        evaluation_name=evaluation_name,
        target_refolding=True,
        task_group="target-refolding",
        tool_name="target_refolding_evaluation_engines",
        **kwargs,
    )
    mark_internal_job(
        source_run_dir,
        parent_run_dir=run_dir,
        parent_task_group="target-refolding",
        role="target_refolding_input_builder",
    )
    return source_run_dir, run_dir
