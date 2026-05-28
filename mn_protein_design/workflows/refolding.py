from __future__ import annotations

import gzip
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from mn_protein_design.core.candidates import (
    STAGE_COMPLEX_REFOLDING,
    STAGE_GENERATION_BACKBONE_SEQUENCE,
    STAGE_MONOMER_REFOLDING,
    STAGE_SEQUENCE_DESIGN,
    read_candidates,
    write_candidates,
)
from mn_protein_design.core.jobs import create_job, finish_job, read_json, update_status, write_json


REFOLDING_GROUP = "refolding-validation"
BOLTZ_MODELS_DIR = Path("/mnt/db/reference_files/boltz_models")
ALPHAFOLD_MODELS_DIR = Path("/mnt/db/reference_files/alphafold_models")
AF2_BINDER_EVAL = Path(__file__).resolve().parents[1] / "tools" / "af2_initial_guess_binder_eval.py"
BOLTZ_PREPARE_INPUTS = Path("/home/user/programs/ovo/original/src/ovo/pipelines/boltz-refolding/bin/prepare_inputs.py")

MONOMER_SUFFIXES = (
    "_boltz2_monomer",
    "_esmfold_monomer",
    "_af2_monomer",
    "_monomer_boltz2",
    "_monomer_esmfold",
    "_monomer_af2",
)

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def _candidate_artifacts(run_dir: Path) -> list[dict[str, str]]:
    artifacts = []
    for path in sorted((run_dir / "artifacts" / "normalized_candidates").glob("*")):
        if path.is_file():
            artifacts.append(
                {
                    "name": path.stem,
                    "path": str(path.relative_to(run_dir)),
                    "type": "normalized_candidates" if path.suffix == ".jsonl" else "campaign_result",
                }
            )
    return artifacts


def _source_candidates(candidates_jsonl: Path, allowed_stages: set[str]) -> list[dict[str, Any]]:
    candidates = [
        candidate
        for candidate in read_candidates(Path(candidates_jsonl))
        if candidate.get("stage") in allowed_stages
    ]
    if not candidates:
        stages = ", ".join(sorted(allowed_stages))
        raise ValueError(f"No candidates with stage {stages} were found in the selected candidate set.")
    return candidates


def _rel_path(run_dir: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def _safe_id(value: object) -> str:
    return str(value or "candidate").replace("/", "_").replace(" ", "_")


def _candidate_chains(candidate: dict[str, Any], key: str, default: list[str]) -> list[str]:
    chains = candidate.get(key)
    if isinstance(chains, str):
        return [chains]
    if isinstance(chains, list):
        values = [str(chain) for chain in chains if str(chain)]
        if values:
            return values
    return default


def _strip_monomer_suffix(candidate_id: object) -> str:
    text = str(candidate_id or "candidate")
    for suffix in MONOMER_SUFFIXES:
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _monomer_candidate_id(source: dict[str, Any], tool: str) -> str:
    label = {
        "boltz2_monomer": "monomer_boltz2",
        "esmfold": "monomer_esmfold",
        "af2_monomer": "monomer_af2",
    }.get(tool, f"monomer_{tool}")
    return f"{_strip_monomer_suffix(source.get('candidate_id'))}_{label}"


def _complex_candidate_id(source: dict[str, Any], tool: str, template_mode: str = "target_template", multimer: bool = True) -> str:
    base = _strip_monomer_suffix(source.get("candidate_id"))
    if tool == "af2_initial_guess":
        model = "mt" if multimer else "ptm"
        template = {
            "target_template": "tt",
            "target_binder_template": "tbt",
            "complex_template": "ct",
        }.get(template_mode, template_mode.replace("_", "-"))
        return f"{base}_af2ig_{model}_{template}"
    if tool == "boltz2_initial_guess":
        template = "tt" if template_mode == "target_template" else "nt"
        return f"{base}_boltz2ig_{template}"
    return f"{base}_complex_{tool}"


def _resolve_candidate_path(source_run_dir: Path, path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else source_run_dir / path


def _sequence_from_pdb(path: Path, chains: list[str] | None = None) -> str:
    keep_chains = set(chains or [])
    residues: list[tuple[str, int, str]] = []
    seen: set[tuple[str, int, str]] = set()
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM  "):
            continue
        chain = line[21].strip() or "_"
        if keep_chains and chain not in keep_chains:
            continue
        resseq = int(line[22:26])
        icode = line[26].strip()
        resname = line[17:20].strip().upper()
        key = (chain, resseq, icode)
        if key in seen:
            continue
        seen.add(key)
        residues.append((chain, resseq, AA3_TO_1.get(resname, "X")))
    return "".join(residue[2] for residue in residues)


def _cif_atom_rows(path: Path) -> list[dict[str, str]]:
    text = gzip.open(path, "rt", errors="ignore").read() if path.name.endswith(".gz") else path.read_text(errors="ignore")
    rows: list[dict[str, str]] = []
    atom_headers: list[str] = []
    in_atom_loop = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "loop_":
            atom_headers = []
            in_atom_loop = False
            continue
        if line.startswith("_atom_site."):
            atom_headers.append(line.split(".", 1)[1])
            in_atom_loop = True
            continue
        if not in_atom_loop or not line.startswith(("ATOM ", "HETATM ")):
            continue
        parts = line.split()
        if len(parts) >= len(atom_headers):
            rows.append(dict(zip(atom_headers, parts)))
    return rows


def _sequence_from_cif(path: Path, chains: list[str] | None = None) -> str:
    keep_chains = set(chains or [])
    residues: list[tuple[str, int, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in _cif_atom_rows(path):
        if (row.get("label_atom_id") or row.get("auth_atom_id")) != "CA":
            continue
        chain = row.get("auth_asym_id") or row.get("label_asym_id") or "_"
        if keep_chains and chain not in keep_chains:
            continue
        residue = row.get("auth_seq_id") or row.get("label_seq_id") or "0"
        insertion = row.get("pdbx_PDB_ins_code") or ""
        key = (chain, residue, insertion)
        if key in seen:
            continue
        seen.add(key)
        try:
            residue_number = int(residue)
        except ValueError:
            residue_number = 0
        residues.append((chain, residue_number, insertion, AA3_TO_1.get((row.get("label_comp_id") or row.get("auth_comp_id") or "").upper(), "X")))
    residues.sort(key=lambda item: (item[0], item[1], item[2]))
    return "".join(row[3] for row in residues)


def _pdb_sequences_by_chain(path: Path) -> dict[str, str]:
    residues: dict[str, list[tuple[int, str, str]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM  "):
            continue
        chain = line[21].strip() or "_"
        key = (chain, line[22:26].strip(), line[26].strip())
        if key in seen:
            continue
        seen.add(key)
        try:
            residue_number = int(line[22:26])
        except ValueError:
            residue_number = 0
        residues.setdefault(chain, []).append(
            (residue_number, line[26].strip(), AA3_TO_1.get(line[17:20].strip().upper(), "X"))
        )
    return {
        chain: "".join(item[2] for item in sorted(chain_residues, key=lambda item: (item[0], item[1])))
        for chain, chain_residues in residues.items()
    }


def _cif_sequences_by_chain(path: Path) -> dict[str, str]:
    residues: dict[str, list[tuple[int, str, str]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for row in _cif_atom_rows(path):
        if (row.get("label_atom_id") or row.get("auth_atom_id")) != "CA":
            continue
        chain = row.get("auth_asym_id") or row.get("label_asym_id") or "_"
        residue = row.get("auth_seq_id") or row.get("label_seq_id") or "0"
        insertion = row.get("pdbx_PDB_ins_code") or ""
        key = (chain, residue, insertion)
        if key in seen:
            continue
        seen.add(key)
        try:
            residue_number = int(residue)
        except ValueError:
            residue_number = 0
        residues.setdefault(chain, []).append(
            (
                residue_number,
                insertion,
                AA3_TO_1.get((row.get("label_comp_id") or row.get("auth_comp_id") or "").upper(), "X"),
            )
        )
    return {
        chain: "".join(item[2] for item in sorted(chain_residues, key=lambda item: (item[0], item[1])))
        for chain, chain_residues in residues.items()
    }


def _sequences_by_chain(path: Path) -> dict[str, str]:
    if path.suffix.lower() == ".cif" or path.name.endswith(".cif.gz"):
        return _cif_sequences_by_chain(path)
    return _pdb_sequences_by_chain(path)


def _pdb_chains(path: Path) -> list[str]:
    chains: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        chain = line[21].strip() or "_"
        if chain not in seen:
            seen.add(chain)
            chains.append(chain)
    return chains


def _cif_chains(path: Path) -> list[str]:
    chains: list[str] = []
    seen: set[str] = set()
    for row in _cif_atom_rows(path):
        chain = row.get("auth_asym_id") or row.get("label_asym_id") or "_"
        if chain not in seen:
            seen.add(chain)
            chains.append(chain)
    return chains


def _structure_chains(path: Path) -> list[str]:
    if path.suffix.lower() == ".cif" or path.name.endswith(".cif.gz"):
        return _cif_chains(path)
    return _pdb_chains(path)


def _renumber_pdb_chain(
    path: Path,
    chain_id: str,
    start_atom: int = 1,
    source_chains: set[str] | None = None,
) -> tuple[list[str], int]:
    lines: list[str] = []
    atom_serial = start_atom
    residue_map: dict[tuple[str, str, str], int] = {}
    next_resseq = 1
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        original_chain = line[21].strip() or "_"
        if source_chains and original_chain not in source_chains:
            continue
        original_key = (line[21], line[22:26], line[26])
        if original_key not in residue_map:
            residue_map[original_key] = next_resseq
            next_resseq += 1
        lines.append(f"{line[:6]}{atom_serial:5d}{line[11:21]}{chain_id}{residue_map[original_key]:4d} {line[27:]}")
        atom_serial += 1
    return lines, atom_serial


def _renumber_cif_chain(
    path: Path,
    chain_id: str,
    start_atom: int = 1,
    source_chains: set[str] | None = None,
) -> tuple[list[str], int]:
    lines: list[str] = []
    atom_serial = start_atom
    residue_map: dict[tuple[str, str, str], int] = {}
    next_resseq = 1
    for row in _cif_atom_rows(path):
        group = row.get("group_PDB") or "ATOM"
        if group not in {"ATOM", "HETATM"}:
            continue
        atom_name = (row.get("auth_atom_id") or row.get("label_atom_id") or "X").strip("'\"")
        resname = (row.get("auth_comp_id") or row.get("label_comp_id") or "UNK").strip("'\"")[:3]
        original_chain = row.get("auth_asym_id") or row.get("label_asym_id") or "_"
        if source_chains and original_chain not in source_chains:
            continue
        original_residue = row.get("auth_seq_id") or row.get("label_seq_id") or "1"
        insertion = row.get("pdbx_PDB_ins_code") or ""
        key = (original_chain, original_residue, insertion)
        if key not in residue_map:
            residue_map[key] = next_resseq
            next_resseq += 1
        try:
            x = float(row["Cartn_x"])
            y = float(row["Cartn_y"])
            z = float(row["Cartn_z"])
        except (KeyError, ValueError):
            continue
        try:
            occupancy = float(row.get("occupancy") or 1.0)
        except ValueError:
            occupancy = 1.0
        try:
            bfactor = float(row.get("B_iso_or_equiv") or 0.0)
        except ValueError:
            bfactor = 0.0
        element = (row.get("type_symbol") or atom_name[:1]).strip("'\"")[:2].rjust(2)
        record = "HETATM" if group == "HETATM" else "ATOM  "
        lines.append(
            f"{record}{atom_serial:5d} {atom_name[:4]:>4s} {resname:>3s} {chain_id[:1]}{residue_map[key]:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{bfactor:6.2f}          {element}"
        )
        atom_serial += 1
    return lines, atom_serial


def _cif_to_pdb(path: Path, output: Path) -> Path:
    lines: list[str] = []
    atom_serial = 1
    residue_map: dict[tuple[str, str, str], int] = {}
    next_resseq_by_chain: dict[str, int] = {}
    for row in _cif_atom_rows(path):
        group = row.get("group_PDB") or "ATOM"
        if group not in {"ATOM", "HETATM"}:
            continue
        atom_name = (row.get("auth_atom_id") or row.get("label_atom_id") or "X").strip("'\"")
        resname = (row.get("auth_comp_id") or row.get("label_comp_id") or "UNK").strip("'\"")[:3]
        chain = (row.get("auth_asym_id") or row.get("label_asym_id") or "_").strip("'\"")[:1]
        original_residue = row.get("auth_seq_id") or row.get("label_seq_id") or "1"
        insertion = row.get("pdbx_PDB_ins_code") or ""
        key = (chain, original_residue, insertion)
        if key not in residue_map:
            next_resseq_by_chain[chain] = next_resseq_by_chain.get(chain, 1)
            residue_map[key] = next_resseq_by_chain[chain]
            next_resseq_by_chain[chain] += 1
        try:
            x = float(row["Cartn_x"])
            y = float(row["Cartn_y"])
            z = float(row["Cartn_z"])
        except (KeyError, ValueError):
            continue
        try:
            occupancy = float(row.get("occupancy") or 1.0)
        except ValueError:
            occupancy = 1.0
        try:
            bfactor = float(row.get("B_iso_or_equiv") or 0.0)
        except ValueError:
            bfactor = 0.0
        element = (row.get("type_symbol") or atom_name[:1]).strip("'\"")[:2].rjust(2)
        record = "HETATM" if group == "HETATM" else "ATOM  "
        lines.append(
            f"{record}{atom_serial:5d} {atom_name[:4]:>4s} {resname:>3s} {chain}{residue_map[key]:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{bfactor:6.2f}          {element}"
        )
        atom_serial += 1
    output.write_text("\n".join(lines + ["TER", "END", ""]))
    return output


def _renumber_structure_chain(
    path: Path,
    chain_id: str,
    start_atom: int = 1,
    source_chains: set[str] | None = None,
) -> tuple[list[str], int]:
    if path.suffix.lower() == ".cif" or path.name.endswith(".cif.gz"):
        return _renumber_cif_chain(path, chain_id, start_atom, source_chains)
    return _renumber_pdb_chain(path, chain_id, start_atom, source_chains)


def _target_pdb_for_candidate(source_run_dir: Path, candidate: dict[str, Any]) -> Path | None:
    checked: set[tuple[str, str]] = set()

    def candidate_targets(base_dir: Path, payload: dict[str, Any] | None) -> Path | None:
        if not isinstance(payload, dict):
            return None
        identity = (str(base_dir), str(payload.get("candidate_id") or id(payload)))
        if identity in checked:
            return None
        checked.add(identity)
        target = _resolve_candidate_path(base_dir, payload.get("target_pdb"))
        if target and target.exists() and target.suffix.lower() == ".pdb":
            return target

        raw = payload.get("raw_metadata") or {}
        upstream_dir_text = raw.get("upstream_source_run_dir")
        if upstream_dir_text:
            upstream_dir = Path(str(upstream_dir_text))
            target = candidate_targets(upstream_dir, raw.get("source_candidate"))
            if target:
                return target
            upstream_input = read_json(upstream_dir / "input.json")
            parent_dir_text = (upstream_input.get("inputs") or {}).get("source_run_dir")
            if parent_dir_text:
                target = candidate_targets(Path(str(parent_dir_text)), raw.get("source_candidate"))
                if target:
                    return target

        source_candidate = raw.get("source_candidate")
        if isinstance(source_candidate, dict):
            target = candidate_targets(base_dir, source_candidate)
            if target:
                return target
            parent_input = read_json(base_dir / "input.json")
            parent_dir_text = (parent_input.get("inputs") or {}).get("source_run_dir")
            if parent_dir_text:
                target = candidate_targets(Path(str(parent_dir_text)), source_candidate)
                if target:
                    return target
        return None

    target = candidate_targets(source_run_dir, candidate)
    if target:
        return target
    return None


def _infer_chain_roles(source_run_dir: Path, candidate: dict[str, Any], complex_path: Path | None = None) -> tuple[list[str], list[str]]:
    if complex_path is None:
        complex_path = _resolve_candidate_path(source_run_dir, candidate.get("complex_pdb") or candidate.get("binder_pdb"))
    complex_chains = _structure_chains(complex_path) if complex_path and complex_path.exists() else []
    explicit_binder = [chain for chain in _candidate_chains(candidate, "binder_chains", []) if not complex_chains or chain in complex_chains]
    explicit_target = [chain for chain in _candidate_chains(candidate, "target_chains", []) if not complex_chains or chain in complex_chains]
    if explicit_binder and explicit_target:
        return explicit_binder, explicit_target
    if explicit_target and complex_chains:
        binder = [chain for chain in complex_chains if chain not in set(explicit_target)]
        if binder:
            return binder, explicit_target
    if explicit_binder and complex_chains:
        target = [chain for chain in complex_chains if chain not in set(explicit_binder)]
        if target:
            return explicit_binder, target

    target_path = _target_pdb_for_candidate(source_run_dir, candidate)
    if complex_path and complex_path.exists() and target_path and target_path.exists():
        complex_sequences = _sequences_by_chain(complex_path)
        target_sequences = _sequences_by_chain(target_path)
        unmatched_complex = set(complex_sequences)
        inferred_target: list[str] = []
        used_target: set[str] = set()
        for complex_chain, complex_sequence in complex_sequences.items():
            if not complex_sequence:
                continue
            for target_chain, target_sequence in target_sequences.items():
                if target_chain in used_target or not target_sequence:
                    continue
                if complex_sequence == target_sequence:
                    inferred_target.append(complex_chain)
                    used_target.add(target_chain)
                    unmatched_complex.discard(complex_chain)
                    break
        inferred_binder = [chain for chain in complex_chains if chain in unmatched_complex]
        if inferred_binder and inferred_target:
            return inferred_binder, inferred_target

    if complex_chains:
        return [complex_chains[-1]], complex_chains[:-1] or ["B"]
    return ["A"], ["B"]


def _complex_pdb_for_candidate(source_run_dir: Path, candidate: dict[str, Any], input_dir: Path) -> Path:
    source_complex = _resolve_candidate_path(source_run_dir, candidate.get("complex_pdb"))
    binder = None

    def normalized_complex_from(path: Path, suffix: str = "") -> Path | None:
        safe_id = _safe_id(candidate.get("candidate_id"))
        output = input_dir / f"{safe_id}{suffix}.pdb"
        inferred_binder_chains, inferred_target_chains = _infer_chain_roles(source_run_dir, candidate, path)
        binder_lines, next_atom = _renumber_structure_chain(path, "A", 1, set(inferred_binder_chains))
        target_lines, _ = _renumber_structure_chain(path, "B", next_atom, set(inferred_target_chains))
        if not binder_lines or not target_lines:
            return None
        output.write_text("\n".join(binder_lines + ["TER"] + target_lines + ["TER", "END", ""]))
        return output

    if source_complex and source_complex.exists() and len(_structure_chains(source_complex)) >= 2:
        normalized = normalized_complex_from(source_complex)
        if normalized:
            return normalized
        if source_complex.suffix.lower() == ".pdb":
            return source_complex
        output = input_dir / f"{_safe_id(candidate.get('candidate_id'))}_from_cif.pdb"
        return _cif_to_pdb(source_complex, output)
    if source_complex and source_complex.exists() and len(_structure_chains(source_complex)) == 1:
        binder = source_complex

    raw = candidate.get("raw_metadata") or {}
    upstream_run_dir = raw.get("upstream_source_run_dir")
    input_complex = raw.get("input_complex")
    if upstream_run_dir and input_complex:
        upstream_complex = _resolve_candidate_path(Path(str(upstream_run_dir)), str(input_complex))
        if upstream_complex and upstream_complex.exists() and len(_structure_chains(upstream_complex)) >= 2:
            normalized = normalized_complex_from(upstream_complex, "_upstream")
            if normalized:
                return normalized
            if upstream_complex.suffix.lower() == ".pdb":
                return upstream_complex
            output = input_dir / f"{_safe_id(candidate.get('candidate_id'))}_upstream_from_cif.pdb"
            return _cif_to_pdb(upstream_complex, output)
        if upstream_complex and upstream_complex.exists() and len(_structure_chains(upstream_complex)) == 1 and binder is None:
            binder = upstream_complex

    if binder is None:
        binder = _resolve_candidate_path(source_run_dir, candidate.get("binder_pdb"))
    if binder is None or not binder.exists():
        source_candidate = (candidate.get("raw_metadata") or {}).get("source_candidate")
        if isinstance(source_candidate, dict):
            binder = _resolve_candidate_path(source_run_dir, source_candidate.get("binder_pdb") or source_candidate.get("complex_pdb"))
            if (binder is None or not binder.exists()) and upstream_run_dir:
                binder = _resolve_candidate_path(Path(str(upstream_run_dir)), source_candidate.get("binder_pdb") or source_candidate.get("complex_pdb"))
    target = _target_pdb_for_candidate(source_run_dir, candidate)
    if binder is None or not binder.exists() or target is None:
        raise ValueError(f"Candidate {candidate.get('candidate_id')} does not have enough PDB data to build a binder-target complex.")

    safe_id = _safe_id(candidate.get("candidate_id"))
    output = input_dir / f"{safe_id}.pdb"
    binder_chains, _target_chains = _infer_chain_roles(source_run_dir, candidate, binder)
    binder_lines, next_atom = _renumber_structure_chain(binder, "A", 1, binder_chains if len(_structure_chains(binder)) > 1 else None)
    target_lines, _ = _renumber_pdb_chain(target, "B", next_atom)
    output.write_text("\n".join(binder_lines + ["TER"] + target_lines + ["TER", "END", ""]))
    return output


def _stage_complex_inputs(source_run_dir: Path, source_candidates: list[dict[str, Any]], input_dir: Path) -> dict[str, Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    for candidate in source_candidates:
        safe_id = _safe_id(candidate.get("candidate_id"))
        complex_path = _complex_pdb_for_candidate(source_run_dir, candidate, input_dir)
        staged_path = input_dir / f"{safe_id}.pdb"
        if complex_path.resolve() != staged_path.resolve():
            shutil.copy2(complex_path, staged_path)
            if complex_path.parent.resolve() == input_dir.resolve():
                complex_path.unlink(missing_ok=True)
        staged[safe_id] = staged_path
    return staged


def _update_source_monomer_rmsd(source_run_dir: Path, source: dict[str, Any], metrics: dict[str, Any]) -> None:
    raw = source.get("raw_metadata") or {}
    source_candidate = raw.get("source_candidate") if isinstance(raw.get("source_candidate"), dict) else source
    upstream_dir_text = raw.get("upstream_source_run_dir")
    input_complex = raw.get("input_complex") or source_candidate.get("complex_pdb")
    reference_backbone = None
    if upstream_dir_text and input_complex:
        reference_backbone = _resolve_candidate_path(Path(str(upstream_dir_text)), str(input_complex))
    if reference_backbone is None or not reference_backbone.exists():
        reference_backbone = _resolve_candidate_path(source_run_dir, source_candidate.get("complex_pdb"))
    predicted_monomer = _resolve_candidate_path(source_run_dir, source.get("complex_pdb"))
    binder_chain = _candidate_chains(source_candidate, "binder_chains", _candidate_chains(source, "binder_chains", ["A"]))[0]
    metrics.update(_monomer_refolding_rmsd(reference_backbone, predicted_monomer, binder_chain=binder_chain))


def _candidate_sequence(source_run_dir: Path, candidate: dict[str, Any]) -> str:
    sequence = str(candidate.get("binder_sequence") or "").strip().replace(" ", "")
    if sequence:
        return sequence
    structure_path = _resolve_candidate_path(source_run_dir, candidate.get("binder_pdb") or candidate.get("complex_pdb"))
    if structure_path and structure_path.exists():
        if structure_path.suffix.lower() == ".pdb":
            sequence = _sequence_from_pdb(structure_path, candidate.get("binder_chains") or None)
        elif structure_path.suffix.lower() == ".cif" or structure_path.name.endswith(".cif.gz"):
            sequence = _sequence_from_cif(structure_path, candidate.get("binder_chains") or None)
    if not sequence:
        raise ValueError(f"Candidate {candidate.get('candidate_id')} has no binder sequence or readable binder structure.")
    return sequence


def _mean_pdb_bfactor(path: Path) -> float | None:
    values = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("ATOM  "):
            try:
                values.append(float(line[60:66]))
            except ValueError:
                continue
    if not values:
        return None
    return sum(values) / len(values)


def _pdb_ca_by_residue(path: Path, chains: set[str] | None = None) -> dict[tuple[str, str, str], tuple[float, float, float]]:
    coords: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or line[12:16].strip() != "CA":
            continue
        chain = line[21].strip() or "_"
        if chains and chain not in chains:
            continue
        key = (chain, line[22:26].strip(), line[26].strip())
        try:
            coords[key] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
    return coords


def _cif_ca_by_residue(path: Path, chains: set[str] | None = None) -> dict[tuple[str, str, str], tuple[float, float, float]]:
    coords: dict[tuple[str, str, str], tuple[float, float, float]] = {}
    atom_headers: list[str] = []
    in_atom_loop = False
    text = gzip.open(path, "rt", errors="ignore").read() if path.name.endswith(".gz") else path.read_text(errors="ignore")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "loop_":
            atom_headers = []
            in_atom_loop = False
            continue
        if line.startswith("_atom_site."):
            atom_headers.append(line.split(".", 1)[1])
            in_atom_loop = True
            continue
        if not in_atom_loop or not line.startswith(("ATOM ", "HETATM ")):
            continue
        parts = line.split()
        if len(parts) < len(atom_headers):
            continue
        row = dict(zip(atom_headers, parts))
        atom_name = row.get("label_atom_id") or row.get("auth_atom_id")
        if atom_name != "CA":
            continue
        chain = row.get("auth_asym_id") or row.get("label_asym_id") or "_"
        if chains and chain not in chains:
            continue
        residue = row.get("auth_seq_id") or row.get("label_seq_id")
        if not residue:
            continue
        key = (chain, residue, "")
        try:
            coords[key] = (float(row["Cartn_x"]), float(row["Cartn_y"]), float(row["Cartn_z"]))
        except (KeyError, ValueError):
            continue
    return coords


def _ca_by_residue(path: Path, chains: set[str] | None = None) -> dict[tuple[str, str, str], tuple[float, float, float]]:
    if path.suffix.lower() == ".cif" or path.name.endswith(".cif.gz"):
        return _cif_ca_by_residue(path, chains)
    return _pdb_ca_by_residue(path, chains)


def _ordered_ca_coords(path: Path, chains: set[str] | None = None) -> list[tuple[float, float, float]]:
    coords = _ca_by_residue(path, chains)
    return [coords[key] for key in sorted(coords, key=_residue_sort_key)]


def _residue_sort_key(key: tuple[str, str, str]) -> tuple[str, int, str]:
    chain, residue, insertion = key
    try:
        residue_number = int(residue)
    except ValueError:
        residue_number = 0
    return chain, residue_number, insertion


def _kabsch_rmsd(reference: np.ndarray, model: np.ndarray) -> float | None:
    if reference.shape != model.shape or reference.shape[0] < 3:
        return None
    ref_center = reference.mean(axis=0)
    model_center = model.mean(axis=0)
    ref = reference - ref_center
    mob = model - model_center
    covariance = mob.T @ ref
    u_mat, _, vt_mat = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = math.copysign(1.0, np.linalg.det(u_mat @ vt_mat))
    aligned = mob @ (u_mat @ correction @ vt_mat)
    return float(np.sqrt(np.mean(np.sum((aligned - ref) ** 2, axis=1))))


def _monomer_refolding_rmsd(reference_path: Path | None, model_path: Path | None, binder_chain: str = "A") -> dict[str, Any]:
    if not reference_path or not model_path or not reference_path.exists() or not model_path.exists():
        return {}
    reference_coords = _ordered_ca_coords(reference_path, {binder_chain})
    model_coords = _ordered_ca_coords(model_path, {binder_chain})
    if len(model_coords) < 3 and binder_chain != "A":
        model_coords = _ordered_ca_coords(model_path, {"A"})
    if len(reference_coords) < 3 or len(model_coords) < 3:
        return {}
    ca_count = min(len(reference_coords), len(model_coords))
    reference = np.array(reference_coords[:ca_count], dtype=float)
    model = np.array(model_coords[:ca_count], dtype=float)
    rmsd = _kabsch_rmsd(reference, model)
    if rmsd is None:
        return {}
    return {
        "monomer_refolding_rmsd": rmsd,
        "monomer_refolding_rmsd_ca_count": ca_count,
        "monomer_refolding_reference": str(reference_path),
    }


def _run_shell_steps(run_dir: Path, steps: list[dict[str, Any]]) -> int:
    write_json(run_dir / "command.json", {"mode": "docker", "steps": steps})
    update_status(run_dir, "running")
    with (run_dir / "stdout.log").open("w") as stdout, (run_dir / "stderr.log").open("w") as stderr:
        for step in steps:
            stdout.write(f"$ {' '.join(step['command'])}\n")
            stdout.flush()
            proc = subprocess.run(step["command"], stdout=stdout, stderr=stderr, check=False)
            if proc.returncode != 0:
                return int(proc.returncode)
    return 0


def _collect_refolding_artifacts(run_dir: Path) -> list[dict[str, str]]:
    artifacts = _candidate_artifacts(run_dir)
    for path in sorted((run_dir / "artifacts").glob("raw/**/*")):
        if not path.is_file():
            continue
        artifact_type = {
            ".pdb": "pdb",
            ".cif": "cif",
            ".json": "metrics",
            ".jsonl": "metrics",
            ".yaml": "input",
            ".yml": "input",
            ".fa": "fasta",
            ".fasta": "fasta",
        }.get(path.suffix.lower(), "artifact")
        artifacts.append({"name": path.stem, "path": str(path.relative_to(run_dir)), "type": artifact_type})
    return artifacts


def _finish_passthrough_job(
    run_dir: Path,
    tool: str,
    candidates: list[dict[str, Any]],
    backend_note: str,
) -> None:
    artifacts = _candidate_artifacts(run_dir)
    finish_job(
        run_dir,
        bool(candidates),
        {
            "outputs": {"artifacts": artifacts, "candidates": candidates},
            "metrics": {
                "candidate_count": len(candidates),
                "backend_status": "contract_only",
            },
            "downstream_artifacts": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
            },
            "backend_note": backend_note,
            "tool": tool,
        },
    )


def run_monomer_refolding_contract(
    source_run_dir: Path,
    candidates_jsonl: Path,
    tool: str = "af2_monomer",
    min_plddt: float = 70.0,
) -> Path:
    if tool == "esmfold":
        return run_esmfold_monomer_refolding(source_run_dir, candidates_jsonl, min_plddt=min_plddt)
    if tool == "boltz2_monomer":
        return run_boltz2_monomer_refolding(source_run_dir, candidates_jsonl, min_plddt=min_plddt)

    source_run_dir = Path(source_run_dir)
    candidates_jsonl = Path(candidates_jsonl)
    source_candidates = _source_candidates(
        candidates_jsonl,
        {STAGE_SEQUENCE_DESIGN, STAGE_GENERATION_BACKBONE_SEQUENCE, STAGE_COMPLEX_REFOLDING},
    )
    job = create_job(
        REFOLDING_GROUP,
        job_type="monomer_refolding",
        tool=tool,
        inputs={"source_run_dir": str(source_run_dir), "candidates_jsonl": str(candidates_jsonl)},
        params={"min_plddt": min_plddt, "backend": "contract_only"},
    )
    write_json(
        job.run_dir / "command.json",
        {
            "mode": "contract_only",
            "command": [],
            "note": "AF2/Boltz2 monomer container is not wired yet. This job preserves the normalized campaign contract.",
        },
    )
    update_status(job.run_dir, "running")
    candidates = []
    for index, source in enumerate(source_candidates, start=1):
        metrics = dict(source.get("metrics") or {})
        metrics.update({"monomer_refolding_backend": "contract_only", "min_plddt": min_plddt})
        candidates.append(
            {
                **source,
                "candidate_id": _monomer_candidate_id(source, tool),
                "stage": STAGE_MONOMER_REFOLDING,
                "source_tool": tool,
                "tool": tool,
                "metrics": metrics,
                "parents": [str(source.get("candidate_id") or "")],
                "raw_metadata": {
                    **dict(source.get("raw_metadata") or {}),
                    "source_candidate": source,
                    "backend_status": "contract_only",
                },
            }
        )
    normalized = write_candidates(job.run_dir, tool, candidates)
    _finish_passthrough_job(
        job.run_dir,
        tool,
        normalized,
        "Monomer refolding backend is not wired yet; candidates were advanced contract-only.",
    )
    return job.run_dir


def run_esmfold_monomer_refolding(
    source_run_dir: Path,
    candidates_jsonl: Path,
    min_plddt: float = 70.0,
    num_recycles: int = 4,
) -> Path:
    source_run_dir = Path(source_run_dir)
    candidates_jsonl = Path(candidates_jsonl)
    source_candidates = _source_candidates(
        candidates_jsonl,
        {STAGE_SEQUENCE_DESIGN, STAGE_GENERATION_BACKBONE_SEQUENCE, STAGE_COMPLEX_REFOLDING},
    )
    job = create_job(
        REFOLDING_GROUP,
        job_type="monomer_refolding",
        tool="esmfold",
        inputs={"source_run_dir": str(source_run_dir), "candidates_jsonl": str(candidates_jsonl)},
        params={"min_plddt": min_plddt, "num_recycles": num_recycles, "backend": "docker", "image": "ovo-esm:latest"},
    )
    raw_root = job.run_dir / "artifacts" / "raw" / "esmfold"
    input_dir = raw_root / "inputs"
    output_dir = raw_root / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = input_dir / "sequences.fa"
    with fasta_path.open("w") as handle:
        for candidate in source_candidates:
            sequence = _candidate_sequence(source_run_dir, candidate)
            safe_id = _safe_id(candidate.get("candidate_id"))
            handle.write(f">{safe_id}\n{sequence}\n")

    steps = [
        {
            "name": "esmfold",
            "command": [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "-v",
                f"{job.run_dir}:/work",
                "-w",
                "/work",
                "ovo-esm:latest",
                "esm-fold",
                "-i",
                "/work/artifacts/raw/esmfold/inputs/sequences.fa",
                "-o",
                "/work/artifacts/raw/esmfold/output",
                "--num-recycles",
                str(num_recycles),
            ],
        }
    ]
    rc = _run_shell_steps(job.run_dir, steps)
    candidates = []
    if rc == 0:
        for source in source_candidates:
            safe_id = _safe_id(source.get("candidate_id"))
            pdb_path = output_dir / f"{safe_id}.pdb"
            if not pdb_path.exists():
                matches = sorted(output_dir.glob(f"{safe_id}*.pdb"))
                pdb_path = matches[0] if matches else pdb_path
            plddt = _mean_pdb_bfactor(pdb_path) if pdb_path.exists() else None
            metrics = dict(source.get("metrics") or {})
            metrics.update({"monomer_refolding_backend": "esmfold", "min_plddt": min_plddt})
            if plddt is not None:
                metrics["plddt"] = plddt
                metrics["passes_monomer_plddt"] = plddt >= min_plddt
            candidates.append(
                {
                    **source,
                    "candidate_id": _monomer_candidate_id(source, "esmfold"),
                    "stage": STAGE_MONOMER_REFOLDING,
                    "source_tool": "esmfold",
                    "tool": "esmfold",
                    "binder_pdb": _rel_path(job.run_dir, pdb_path if pdb_path.exists() else None),
                    "metrics": metrics,
                    "parents": [str(source.get("candidate_id") or "")],
                    "raw_metadata": {
                        **dict(source.get("raw_metadata") or {}),
                        "source_candidate": source,
                        "backend_status": "docker",
                        "upstream_source_run_dir": str(source_run_dir),
                        "input_complex": source.get("complex_pdb"),
                    },
                }
            )
    normalized = write_candidates(job.run_dir, "esmfold", candidates) if candidates else []
    artifacts = _collect_refolding_artifacts(job.run_dir)
    finish_job(
        job.run_dir,
        rc == 0 and bool(normalized),
        {
            "outputs": {"artifacts": artifacts, "candidates": normalized},
            "metrics": {"return_code": rc, "candidate_count": len(normalized), "artifact_count": len(artifacts)},
            "downstream_artifacts": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
            },
        },
    )
    return job.run_dir


def run_boltz2_monomer_refolding(
    source_run_dir: Path,
    candidates_jsonl: Path,
    min_plddt: float = 0.7,
) -> Path:
    source_run_dir = Path(source_run_dir)
    candidates_jsonl = Path(candidates_jsonl)
    source_candidates = _source_candidates(
        candidates_jsonl,
        {STAGE_SEQUENCE_DESIGN, STAGE_GENERATION_BACKBONE_SEQUENCE, STAGE_COMPLEX_REFOLDING},
    )
    job = create_job(
        REFOLDING_GROUP,
        job_type="monomer_refolding",
        tool="boltz2_monomer",
        inputs={"source_run_dir": str(source_run_dir), "candidates_jsonl": str(candidates_jsonl)},
        params={
            "min_plddt": min_plddt,
            "backend": "docker",
            "image": "ovoex-boltz2:latest",
            "models_dir": str(BOLTZ_MODELS_DIR),
        },
    )
    raw_root = job.run_dir / "artifacts" / "raw" / "boltz2_monomer"
    input_dir = raw_root / "inputs"
    output_dir = raw_root / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    id_map: dict[str, dict[str, Any]] = {}
    for candidate in source_candidates:
        safe_id = _safe_id(candidate.get("candidate_id"))
        sequence = _candidate_sequence(source_run_dir, candidate)
        id_map[safe_id] = candidate
        (input_dir / f"{safe_id}.yaml").write_text(
            "\n".join(
                [
                    "version: 1",
                    "sequences:",
                    "  - protein:",
                    "      id: A",
                    f"      sequence: {sequence}",
                    "      msa: empty",
                    "",
                ]
            )
        )
    write_json(raw_root / "candidate_id_map.json", id_map)
    steps = [
        {
            "name": "boltz2-monomer",
            "command": [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "-v",
                f"{job.run_dir}:/work",
                "-v",
                f"{BOLTZ_MODELS_DIR}:/models",
                "-w",
                "/work",
                "ovoex-boltz2:latest",
                "predict",
                "/work/artifacts/raw/boltz2_monomer/inputs",
                "--cache",
                "/models",
                "--accelerator",
                "gpu",
                "--model",
                "boltz2",
            ],
        }
    ]
    rc = _run_shell_steps(job.run_dir, steps)
    candidates = []
    if rc == 0:
        predictions_root = job.run_dir / "boltz_results_inputs" / "predictions"
        if predictions_root.exists():
            target_root = output_dir / "predictions"
            target_root.parent.mkdir(parents=True, exist_ok=True)
            if target_root.exists():
                pass
            predictions_root.rename(target_root)
        else:
            target_root = output_dir / "predictions"
        for safe_id, source in id_map.items():
            prediction_dir = target_root / safe_id
            cif_matches = sorted(prediction_dir.glob("*.cif")) if prediction_dir.exists() else []
            confidence_matches = sorted(prediction_dir.glob("confidence*.json")) if prediction_dir.exists() else []
            metrics = dict(source.get("metrics") or {})
            metrics.update({"monomer_refolding_backend": "boltz2_monomer", "min_plddt": min_plddt})
            if confidence_matches:
                try:
                    confidence = json.loads(confidence_matches[0].read_text())
                    for key, value in confidence.items():
                        if isinstance(value, (int, float, str, bool)):
                            metrics[f"boltz2_{key}"] = value
                except json.JSONDecodeError:
                    pass
            predicted_monomer = cif_matches[0] if cif_matches else None
            reference_backbone = _resolve_candidate_path(source_run_dir, source.get("complex_pdb"))
            binder_chain = _candidate_chains(source, "binder_chains", ["A"])[0]
            metrics.update(_monomer_refolding_rmsd(reference_backbone, predicted_monomer, binder_chain=binder_chain))
            candidates.append(
                {
                    **source,
                    "candidate_id": _monomer_candidate_id(source, "boltz2_monomer"),
                    "stage": STAGE_MONOMER_REFOLDING,
                    "source_tool": "boltz2_monomer",
                    "tool": "boltz2_monomer",
                    "binder_pdb": None,
                    "complex_pdb": _rel_path(job.run_dir, predicted_monomer) if predicted_monomer else source.get("complex_pdb"),
                    "metrics": metrics,
                    "parents": [str(source.get("candidate_id") or "")],
                    "raw_metadata": {
                        **dict(source.get("raw_metadata") or {}),
                        "source_candidate": source,
                        "backend_status": "docker",
                        "prediction_dir": _rel_path(job.run_dir, prediction_dir),
                        "upstream_source_run_dir": str(source_run_dir),
                        "input_complex": source.get("complex_pdb"),
                    },
                }
            )
    normalized = write_candidates(job.run_dir, "boltz2_monomer", candidates) if candidates else []
    artifacts = _collect_refolding_artifacts(job.run_dir)
    finish_job(
        job.run_dir,
        rc == 0 and bool(normalized),
        {
            "outputs": {"artifacts": artifacts, "candidates": normalized},
            "metrics": {"return_code": rc, "candidate_count": len(normalized), "artifact_count": len(artifacts)},
            "downstream_artifacts": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
            },
        },
    )
    return job.run_dir


def run_complex_refolding_contract(
    source_run_dir: Path,
    candidates_jsonl: Path,
    tool: str = "af2_initial_guess",
    require_monomer_success: bool = True,
    template_mode: str = "target_template",
    num_recycles: int = 3,
    multimer: bool = True,
) -> Path:
    if tool == "af2_initial_guess":
        use_binder_template = template_mode in {"target_binder_template", "complex_template"}
        return run_af2_initial_guess_complex_refolding(
            source_run_dir,
            candidates_jsonl,
            require_monomer_success=require_monomer_success,
            num_recycles=num_recycles,
            multimer=multimer,
            use_binder_template=use_binder_template,
            use_interface_template=template_mode == "complex_template",
        )
    if tool == "boltz2_initial_guess":
        return run_boltz2_complex_refolding(
            source_run_dir,
            candidates_jsonl,
            require_monomer_success=require_monomer_success,
            use_target_template=template_mode == "target_template",
        )

    source_run_dir = Path(source_run_dir)
    candidates_jsonl = Path(candidates_jsonl)
    allowed = {STAGE_MONOMER_REFOLDING} if require_monomer_success else {
        STAGE_MONOMER_REFOLDING,
        STAGE_SEQUENCE_DESIGN,
        STAGE_GENERATION_BACKBONE_SEQUENCE,
        STAGE_COMPLEX_REFOLDING,
    }
    source_candidates = _source_candidates(candidates_jsonl, allowed)
    job = create_job(
        REFOLDING_GROUP,
        job_type="complex_refolding",
        tool=tool,
        inputs={"source_run_dir": str(source_run_dir), "candidates_jsonl": str(candidates_jsonl)},
        params={"require_monomer_success": require_monomer_success, "backend": "contract_only"},
    )
    write_json(
        job.run_dir / "command.json",
        {
            "mode": "contract_only",
            "command": [],
            "note": "AF2/Boltz2 complex initial-guess container is not wired yet. This job preserves the normalized campaign contract.",
        },
    )
    update_status(job.run_dir, "running")
    candidates = []
    for index, source in enumerate(source_candidates, start=1):
        metrics = dict(source.get("metrics") or {})
        metrics.update({"complex_refolding_backend": "contract_only"})
        candidates.append(
            {
                **source,
                "candidate_id": f"{source.get('candidate_id')}_{tool}_{index:03d}",
                "stage": STAGE_COMPLEX_REFOLDING,
                "source_tool": tool,
                "tool": tool,
                "metrics": metrics,
                "parents": [str(source.get("candidate_id") or "")],
                "raw_metadata": {
                    **dict(source.get("raw_metadata") or {}),
                    "source_candidate": source,
                    "backend_status": "contract_only",
                },
            }
        )
    normalized = write_candidates(job.run_dir, tool, candidates)
    _finish_passthrough_job(
        job.run_dir,
        tool,
        normalized,
        "Complex refolding backend is not wired yet; monomer-stage candidates were advanced contract-only.",
    )
    return job.run_dir


def run_af2_initial_guess_complex_refolding(
    source_run_dir: Path,
    candidates_jsonl: Path,
    require_monomer_success: bool = True,
    num_recycles: int = 3,
    multimer: bool = True,
    use_binder_template: bool = False,
    use_interface_template: bool = False,
) -> Path:
    source_run_dir = Path(source_run_dir)
    candidates_jsonl = Path(candidates_jsonl)
    allowed = {STAGE_MONOMER_REFOLDING} if require_monomer_success else {
        STAGE_MONOMER_REFOLDING,
        STAGE_SEQUENCE_DESIGN,
        STAGE_GENERATION_BACKBONE_SEQUENCE,
        STAGE_COMPLEX_REFOLDING,
    }
    source_candidates = _source_candidates(candidates_jsonl, allowed)
    job = create_job(
        REFOLDING_GROUP,
        job_type="complex_refolding",
        tool="af2_initial_guess",
        inputs={"source_run_dir": str(source_run_dir), "candidates_jsonl": str(candidates_jsonl)},
        params={
            "require_monomer_success": require_monomer_success,
            "num_recycles": num_recycles,
            "multimer": multimer,
            "use_binder_template": use_binder_template,
            "use_interface_template": use_interface_template,
            "backend": "docker",
            "image": "ovo-colabdesign:latest",
            "alphafold_models_dir": str(ALPHAFOLD_MODELS_DIR),
        },
    )
    raw_root = job.run_dir / "artifacts" / "raw" / "af2_initial_guess"
    input_dir = raw_root / "inputs"
    output_dir = raw_root / "output"
    staged = _stage_complex_inputs(source_run_dir, source_candidates, input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{job.run_dir}:/work",
        "-v",
        f"{AF2_BINDER_EVAL.parent}:/scripts:ro",
        "-v",
        f"{ALPHAFOLD_MODELS_DIR}:/models:ro",
        "-w",
        "/work",
        "ovo-colabdesign:latest",
        "python",
        f"/scripts/{AF2_BINDER_EVAL.name}",
        "/work/artifacts/raw/af2_initial_guess/inputs",
        "/work/artifacts/raw/af2_initial_guess/output/af2_initial_guess",
        "--params",
        "/models",
        "--num-recycles",
        str(num_recycles),
        "--designed_chains",
        "A",
    ]
    if multimer:
        command.append("--multimer")
    if use_binder_template:
        command.append("--use-binder-template")
    if use_interface_template:
        command.append("--use-interface-template")
    rc = _run_shell_steps(job.run_dir, [{"name": "af2-initial-guess", "command": command}])
    metrics_by_id: dict[str, dict[str, Any]] = {}
    metrics_path = output_dir / "af2_initial_guess.jsonl"
    if metrics_path.exists():
        for line in metrics_path.read_text(errors="ignore").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            metrics_by_id[str(row.get("id"))] = row

    candidates = []
    if rc == 0:
        for safe_id, staged_path in staged.items():
            source = next(candidate for candidate in source_candidates if safe_id == _safe_id(candidate.get("candidate_id")))
            pdb_matches = sorted((output_dir / "af2_initial_guess").glob(f"{safe_id}*.pdb"))
            predicted_pdb = pdb_matches[0] if pdb_matches else None
            pae_path = predicted_pdb.with_name(f"{predicted_pdb.stem}_pae.json") if predicted_pdb else None
            candidate_id = _complex_candidate_id(
                source,
                "af2_initial_guess",
                "complex_template" if use_interface_template else "target_binder_template" if use_binder_template else "target_template",
                multimer,
            )
            if predicted_pdb and predicted_pdb.exists():
                renamed_pdb = predicted_pdb.with_name(f"{_safe_id(candidate_id)}.pdb")
                if renamed_pdb != predicted_pdb:
                    predicted_pdb.rename(renamed_pdb)
                    predicted_pdb = renamed_pdb
                old_pae = pae_path
                pae_path = predicted_pdb.with_name(f"{predicted_pdb.stem}_pae.json")
                if old_pae and old_pae.exists() and old_pae != pae_path:
                    old_pae.rename(pae_path)
            metrics = dict(source.get("metrics") or {})
            _update_source_monomer_rmsd(source_run_dir, source, metrics)
            metrics.update(metrics_by_id.get(safe_id, {}))
            metrics["complex_refolding_backend"] = "af2_initial_guess"
            candidates.append(
                {
                    **source,
                    "candidate_id": candidate_id,
                    "stage": STAGE_COMPLEX_REFOLDING,
                    "source_tool": "af2_initial_guess",
                    "tool": "af2_initial_guess",
                    "binder_chains": ["A"],
                    "target_chains": ["B"],
                    "complex_pdb": _rel_path(job.run_dir, predicted_pdb) if predicted_pdb else _rel_path(job.run_dir, staged_path),
                    "metrics": metrics,
                    "parents": [str(source.get("candidate_id") or "")],
                    "raw_metadata": {
                        **dict(source.get("raw_metadata") or {}),
                        "source_candidate": source,
                        "input_binder_chains": source.get("binder_chains"),
                        "input_target_chains": source.get("target_chains"),
                        "backend_status": "docker",
                        "input_complex": _rel_path(job.run_dir, staged_path),
                        "prediction_dir": _rel_path(job.run_dir, output_dir / "af2_initial_guess"),
                        "pae_path": _rel_path(job.run_dir, pae_path) if pae_path and pae_path.exists() else None,
                    },
                }
            )
    normalized = write_candidates(job.run_dir, "af2_initial_guess", candidates) if candidates else []
    artifacts = _collect_refolding_artifacts(job.run_dir)
    finish_job(
        job.run_dir,
        rc == 0 and bool(normalized),
        {
            "outputs": {"artifacts": artifacts, "candidates": normalized},
            "metrics": {"return_code": rc, "candidate_count": len(normalized), "artifact_count": len(artifacts)},
            "downstream_artifacts": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
            },
        },
    )
    return job.run_dir


def run_boltz2_complex_refolding(
    source_run_dir: Path,
    candidates_jsonl: Path,
    require_monomer_success: bool = True,
    use_target_template: bool = True,
) -> Path:
    source_run_dir = Path(source_run_dir)
    candidates_jsonl = Path(candidates_jsonl)
    allowed = {STAGE_MONOMER_REFOLDING} if require_monomer_success else {
        STAGE_MONOMER_REFOLDING,
        STAGE_SEQUENCE_DESIGN,
        STAGE_GENERATION_BACKBONE_SEQUENCE,
        STAGE_COMPLEX_REFOLDING,
    }
    source_candidates = _source_candidates(candidates_jsonl, allowed)
    job = create_job(
        REFOLDING_GROUP,
        job_type="complex_refolding",
        tool="boltz2_initial_guess",
        inputs={"source_run_dir": str(source_run_dir), "candidates_jsonl": str(candidates_jsonl)},
        params={
            "require_monomer_success": require_monomer_success,
            "backend": "docker",
            "image": "ovoex-boltz2:latest",
            "models_dir": str(BOLTZ_MODELS_DIR),
            "template_mode": "target_template" if use_target_template else "no_template",
        },
    )
    raw_root = job.run_dir / "artifacts" / "raw" / "boltz2_initial_guess"
    input_pdb_dir = raw_root / "input_pdbs"
    yaml_dir = raw_root / "yaml_inputs"
    output_dir = raw_root / "output"
    staged = _stage_complex_inputs(source_run_dir, source_candidates, input_pdb_dir)
    yaml_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_by_id = {}
    target_template = None
    for safe_id, pdb_path in staged.items():
        source = next(candidate for candidate in source_candidates if safe_id == _safe_id(candidate.get("candidate_id")))
        source_by_id[safe_id] = source
        if target_template is None:
            target_template = _target_pdb_for_candidate(source_run_dir, source)
    write_json(raw_root / "candidate_id_map.json", source_by_id)
    staged_template_arg = "--no-template "
    if use_target_template and target_template and target_template.exists():
        staged_template = raw_root / "target_template.pdb"
        shutil.copy2(target_template, staged_template)
        staged_template_arg = "--template /work/artifacts/raw/boltz2_initial_guess/target_template.pdb "
    prep_command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{job.run_dir}:/work",
        "-v",
        f"{BOLTZ_PREPARE_INPUTS.parent}:/scripts:ro",
        "-w",
        "/work",
        "--entrypoint",
        "/bin/bash",
        "ovoex-boltz2:latest",
        "-lc",
        (
            "source /opt/conda/etc/profile.d/conda.sh && conda activate boltz2 && "
            "python /scripts/prepare_inputs.py "
            "--input_dir /work/artifacts/raw/boltz2_initial_guess/input_pdbs "
            "--output_dir /work/artifacts/raw/boltz2_initial_guess/yaml_inputs "
            "--design_type binder "
            "--designed_chains A "
            + staged_template_arg
        ),
    ]
    predict_command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{job.run_dir}:/work",
        "-v",
        f"{BOLTZ_MODELS_DIR}:/models",
        "-w",
        "/work",
        "ovoex-boltz2:latest",
        "predict",
        "/work/artifacts/raw/boltz2_initial_guess/yaml_inputs",
        "--cache",
        "/models",
        "--accelerator",
        "gpu",
        "--model",
        "boltz2",
    ]
    rc = _run_shell_steps(
        job.run_dir,
        [
            {"name": "boltz2-prepare-inputs", "command": prep_command},
            {"name": "boltz2-initial-guess", "command": predict_command},
        ],
    )
    predictions_root = job.run_dir / "boltz_results_yaml_inputs" / "predictions"
    if predictions_root.exists():
        target_root = output_dir / "predictions"
        target_root.parent.mkdir(parents=True, exist_ok=True)
        if not target_root.exists():
            predictions_root.rename(target_root)
    else:
        target_root = output_dir / "predictions"

    candidates = []
    if rc == 0:
        for safe_id, source in source_by_id.items():
            prediction_dir = target_root / safe_id
            cif_matches = sorted(prediction_dir.glob("*.cif")) if prediction_dir.exists() else []
            confidence_matches = sorted(prediction_dir.glob("confidence*.json")) if prediction_dir.exists() else []
            metrics = dict(source.get("metrics") or {})
            _update_source_monomer_rmsd(source_run_dir, source, metrics)
            metrics["complex_refolding_backend"] = "boltz2_initial_guess"
            if confidence_matches:
                try:
                    confidence = json.loads(confidence_matches[0].read_text())
                    for key, value in confidence.items():
                        if isinstance(value, (int, float, str, bool)):
                            metrics[f"boltz2_{key}"] = value
                except json.JSONDecodeError:
                    pass
            candidates.append(
                {
                    **source,
                    "candidate_id": _complex_candidate_id(
                        source,
                        "boltz2_initial_guess",
                        "target_template" if use_target_template else "no_template",
                    ),
                    "stage": STAGE_COMPLEX_REFOLDING,
                    "source_tool": "boltz2_initial_guess",
                    "tool": "boltz2_initial_guess",
                    "binder_chains": ["A"],
                    "target_chains": ["B"],
                    "complex_pdb": _rel_path(job.run_dir, cif_matches[0]) if cif_matches else source.get("complex_pdb"),
                    "metrics": metrics,
                    "parents": [str(source.get("candidate_id") or "")],
                    "raw_metadata": {
                        **dict(source.get("raw_metadata") or {}),
                        "source_candidate": source,
                        "input_binder_chains": source.get("binder_chains"),
                        "input_target_chains": source.get("target_chains"),
                        "backend_status": "docker",
                        "prediction_dir": _rel_path(job.run_dir, prediction_dir),
                    },
                }
            )
    normalized = write_candidates(job.run_dir, "boltz2_initial_guess", candidates) if candidates else []
    artifacts = _collect_refolding_artifacts(job.run_dir)
    finish_job(
        job.run_dir,
        rc == 0 and bool(normalized),
        {
            "outputs": {"artifacts": artifacts, "candidates": normalized},
            "metrics": {"return_code": rc, "candidate_count": len(normalized), "artifact_count": len(artifacts)},
            "downstream_artifacts": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
            },
        },
    )
    return job.run_dir
