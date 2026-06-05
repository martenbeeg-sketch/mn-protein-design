from __future__ import annotations

import gzip
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mn_protein_design.core.candidates import (
    STAGE_COMPLEX_REFOLDING,
    STAGE_GENERATION_BACKBONE_SEQUENCE,
    STAGE_MONOMER_REFOLDING,
    STAGE_SEQUENCE_DESIGN,
    read_candidates,
    write_candidates,
)
from mn_protein_design.core.jobs import create_job, finish_job, mark_internal_job, read_json, update_status, write_json
from mn_protein_design.workflows import esm_binder as esm_binder_workflow


REFOLDING_GROUP = "refolding-validation"
BOLTZ_MODELS_DIR = Path("/mnt/db/reference_files/boltz_models")
ALPHAFOLD_MODELS_DIR = Path("/mnt/db/reference_files/alphafold_models")
AF2_BINDER_EVAL = Path(__file__).resolve().parents[1] / "tools" / "af2_initial_guess_binder_eval.py"
BOLTZ_PREPARE_INPUTS = Path("/home/user/programs/ovo/original/src/ovo/pipelines/boltz-refolding/bin/prepare_inputs.py")
RF3_IMAGE = "ovoex-foundry-cu128:latest"
RF3_CHECKPOINT = Path("/mnt/db/reference_files/foundry/rf3_foundry_01_24_latest_remapped.ckpt")
PROTENIX_IMAGE = "mnprot-pxdesign-cu128:latest"
PROTENIX_REFERENCE_DIR = Path("/mnt/db/reference_files/pxdesign")
BOLTZGEN_IMAGE = "boltzgen:latest"
BOLTZGEN_LOCAL_SOURCE = Path(__file__).resolve().parents[2] / "tools_to_implement" / "boltzgen" / "src" / "boltzgen"
BOLTZGEN_BENCHMARK_PAE_CONFIG = Path(__file__).resolve().parents[2] / "tools_to_implement" / "boltzgen" / "config" / "fold_benchmark_pae.yaml"

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


def _clean_sequence(value: object) -> str:
    return "".join(ch for ch in str(value or "").upper() if ch.isalpha())


def _load_run_csv_msa_records(run_csv: Path | None) -> dict[str, dict[str, dict[str, str]]]:
    if run_csv is None or not Path(run_csv).exists():
        return {}
    records: dict[str, dict[str, dict[str, str]]] = {}
    with Path(run_csv).open(newline="") as handle:
        for row in csv.DictReader(handle):
            binder_id = str(row.get("binder_id") or "").strip()
            if not binder_id:
                continue
            chain_records: dict[str, dict[str, str]] = {}
            binder_chain = str(row.get("binder_chain") or "A").strip() or "A"
            for key, value in row.items():
                if not key.startswith("msa_path_"):
                    continue
                chain = key.removeprefix("msa_path_")
                if not chain or chain == binder_chain:
                    continue
                msa_path = str(value or "").strip()
                if not msa_path or msa_path.lower() == "no_msa":
                    continue
                seq = (
                    row.get(f"target_subchain_{chain}_seq")
                    or row.get(f"{chain}_seq")
                    or ""
                )
                chain_records[chain] = {"msa_path": msa_path, "sequence": _clean_sequence(seq)}
            records[binder_id] = chain_records
            records[_safe_id(binder_id)] = chain_records
    return records


def _inject_boltz_yaml_msas(
    *,
    yaml_dir: Path,
    raw_root: Path,
    benchmark_run_csv: Path | None,
) -> dict[str, int]:
    msa_records = _load_run_csv_msa_records(benchmark_run_csv)
    metrics = {
        "boltz2_msa_injected_count": 0,
        "boltz2_msa_missing_count": 0,
        "boltz2_msa_sequence_mismatch_count": 0,
    }
    if not msa_records or not yaml_dir.exists():
        return metrics
    staged_msa_dir = raw_root / "msas"
    staged_msa_dir.mkdir(parents=True, exist_ok=True)
    for yaml_path in sorted(yaml_dir.glob("*.yaml")):
        binder_id = yaml_path.stem
        chain_records = msa_records.get(binder_id) or msa_records.get(_safe_id(binder_id)) or {}
        if not chain_records:
            continue
        try:
            payload = yaml.safe_load(yaml_path.read_text()) or {}
        except yaml.YAMLError:
            continue
        changed = False
        for entry in payload.get("sequences") or []:
            protein = entry.get("protein") if isinstance(entry, dict) else None
            if not isinstance(protein, dict):
                continue
            chain_id = str(protein.get("id") or "").strip()
            if not chain_id:
                continue
            record = chain_records.get(chain_id)
            if record is None:
                continue
            yaml_seq = _clean_sequence(protein.get("sequence"))
            record_seq = _clean_sequence(record.get("sequence"))
            if record_seq and yaml_seq and record_seq != yaml_seq:
                metrics["boltz2_msa_sequence_mismatch_count"] += 1
                continue
            source_msa = Path(record.get("msa_path") or "")
            if not source_msa.exists():
                metrics["boltz2_msa_missing_count"] += 1
                continue
            staged_name = f"{_safe_id(binder_id)}_chain_{_safe_id(chain_id)}{source_msa.suffix or '.a3m'}"
            staged_msa = staged_msa_dir / staged_name
            shutil.copy2(source_msa, staged_msa)
            protein["msa"] = f"/work/artifacts/raw/boltz2_initial_guess/msas/{staged_name}"
            metrics["boltz2_msa_injected_count"] += 1
            changed = True
        if changed:
            yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return metrics


def _rf3_msa_records_for_candidate(
    candidate: dict[str, Any],
    benchmark_msa_records: dict[str, dict[str, dict[str, str]]],
) -> dict[str, dict[str, str]]:
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    records = benchmark_msa_records.get(candidate_id) or benchmark_msa_records.get(_safe_id(candidate_id))
    if records:
        return records
    raw = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
    row = raw.get("repo_run_csv_row") if isinstance(raw.get("repo_run_csv_row"), dict) else {}
    binder_chain = str(row.get("binder_chain") or "A").strip() or "A"
    chain_records: dict[str, dict[str, str]] = {}
    for key, value in row.items():
        if not str(key).startswith("msa_path_"):
            continue
        chain = str(key).removeprefix("msa_path_")
        msa_path = str(value or "").strip()
        if not chain or chain == binder_chain or not msa_path or msa_path.lower() == "no_msa":
            continue
        chain_records[chain] = {
            "msa_path": msa_path,
            "sequence": _clean_sequence(row.get(f"target_subchain_{chain}_seq") or row.get(f"{chain}_seq") or ""),
        }
    return chain_records


def _write_rf3_json_inputs(
    *,
    source_run_dir: Path,
    source_candidates: list[dict[str, Any]],
    staged: dict[str, Path],
    input_dir: Path,
    benchmark_run_csv: Path | None,
    use_target_msa: bool,
) -> tuple[dict[str, Path], dict[str, int]]:
    benchmark_msa_records = _load_run_csv_msa_records(benchmark_run_csv)
    rf3_inputs: dict[str, Path] = {}
    metrics = {
        "rf3_target_msa_injected_count": 0,
        "rf3_target_msa_missing_count": 0,
        "rf3_target_msa_sequence_mismatch_count": 0,
    }
    msa_dir = input_dir / "msas"
    msa_dir.mkdir(parents=True, exist_ok=True)
    for safe_id, staged_path in staged.items():
        source = next(item for item in source_candidates if safe_id == _safe_id(item.get("candidate_id")))
        binder_chains = _candidate_chains(source, "binder_chains", ["A"])
        target_chains = _candidate_chains(source, "target_chains", [])
        if not target_chains:
            _inferred_binder, target_chains = _infer_chain_roles(source_run_dir, source)
        target_set = set(target_chains)
        sequences = _pdb_sequences_by_chain(staged_path)
        raw = source.get("raw_metadata") if isinstance(source.get("raw_metadata"), dict) else {}
        row = raw.get("repo_run_csv_row") if isinstance(raw.get("repo_run_csv_row"), dict) else {}
        declared_sequences: dict[str, str] = {}
        for chain in binder_chains:
            sequence = _clean_sequence(row.get(f"{chain}_seq") or (row.get("A_seq") if chain == "A" else ""))
            if sequence:
                declared_sequences[chain] = sequence
        for chain in target_chains:
            sequence = _clean_sequence(row.get(f"target_subchain_{chain}_seq") or row.get(f"{chain}_seq") or "")
            if sequence:
                declared_sequences[chain] = sequence
        if declared_sequences and all(chain in declared_sequences for chain in [*binder_chains, *target_chains]):
            sequences = declared_sequences
        msa_records = _rf3_msa_records_for_candidate(source, benchmark_msa_records) if use_target_msa else {}
        components: list[dict[str, str]] = []
        used_msa_records: set[str] = set()
        for chain, sequence in sequences.items():
            component = {"seq": sequence, "chain_id": chain}
            if use_target_msa and chain in target_set:
                record_chain = chain if chain in msa_records else ""
                if not record_chain:
                    for candidate_chain, record in msa_records.items():
                        if candidate_chain in used_msa_records:
                            continue
                        if _clean_sequence(record.get("sequence")) == _clean_sequence(sequence):
                            record_chain = candidate_chain
                            break
                record = msa_records.get(record_chain) if record_chain else None
                if record is None:
                    metrics["rf3_target_msa_missing_count"] += 1
                elif _clean_sequence(record.get("sequence")) and _clean_sequence(record.get("sequence")) != _clean_sequence(sequence):
                    metrics["rf3_target_msa_sequence_mismatch_count"] += 1
                else:
                    source_msa = Path(str(record.get("msa_path") or ""))
                    if source_msa.exists():
                        staged_msa = msa_dir / f"{safe_id}_chain_{_safe_id(chain)}{source_msa.suffix or '.a3m'}"
                        shutil.copy2(source_msa, staged_msa)
                        component["msa_path"] = f"/work/artifacts/raw/rf3/inputs/msas/{staged_msa.name}"
                        metrics["rf3_target_msa_injected_count"] += 1
                        used_msa_records.add(record_chain)
                    else:
                        metrics["rf3_target_msa_missing_count"] += 1
            components.append(component)
        input_path = input_dir / f"{safe_id}.json"
        write_json(input_path, {"name": safe_id, "components": components})
        rf3_inputs[safe_id] = input_path
    return rf3_inputs, metrics


def _write_protenix_json_inputs(
    *,
    source_run_dir: Path,
    source_candidates: list[dict[str, Any]],
    staged: dict[str, Path],
    json_root: Path,
    msa_root: Path,
    benchmark_run_csv: Path | None,
    use_target_msa: bool,
) -> tuple[dict[str, Path], dict[str, int]]:
    benchmark_msa_records = _load_run_csv_msa_records(benchmark_run_csv)
    protenix_inputs: dict[str, Path] = {}
    metrics = {
        "protenix_target_msa_injected_count": 0,
        "protenix_target_msa_missing_count": 0,
        "protenix_target_msa_sequence_mismatch_count": 0,
        "protenix_binder_query_only_msa_count": 0,
        "protenix_target_query_only_msa_fallback_count": 0,
    }
    job_run_dir = json_root.parents[3]

    def attach_msa(protein: dict[str, Any], chain_dir: Path) -> None:
        protein["msa"] = {
            "precomputed_msa_dir": f"/work/{chain_dir.relative_to(job_run_dir)}",
            "pairing_db": "uniref100",
            "pairing_db_fpath": None,
            "non_pairing_db_fpath": None,
            "search_too": None,
            "msa_save_dir": None,
        }

    def write_query_only_msa(chain_dir: Path, safe_id: str, chain: str, sequence: str) -> None:
        chain_dir.mkdir(parents=True, exist_ok=True)
        content = f">{safe_id}_chain_{_safe_id(chain)}\n{sequence}\n"
        for filename in ["pairing.a3m", "non_pairing.a3m"]:
            (chain_dir / filename).write_text(content)

    for safe_id, staged_path in staged.items():
        source = next(item for item in source_candidates if safe_id == _safe_id(item.get("candidate_id")))
        binder_chains = _candidate_chains(source, "binder_chains", ["A"])
        target_chains = _candidate_chains(source, "target_chains", [])
        if not target_chains:
            _inferred_binder, target_chains = _infer_chain_roles(source_run_dir, source)
        target_set = set(target_chains)
        sequences = _pdb_sequences_by_chain(staged_path)
        raw = source.get("raw_metadata") if isinstance(source.get("raw_metadata"), dict) else {}
        row = raw.get("repo_run_csv_row") if isinstance(raw.get("repo_run_csv_row"), dict) else {}
        declared_sequences: dict[str, str] = {}
        for chain in binder_chains:
            sequence = _clean_sequence(row.get(f"{chain}_seq") or (row.get("A_seq") if chain == "A" else ""))
            if sequence:
                declared_sequences[chain] = sequence
        for chain in target_chains:
            sequence = _clean_sequence(row.get(f"target_subchain_{chain}_seq") or row.get(f"{chain}_seq") or "")
            if sequence:
                declared_sequences[chain] = sequence
        if declared_sequences and all(chain in declared_sequences for chain in [*binder_chains, *target_chains]):
            sequences = declared_sequences
        msa_records = _rf3_msa_records_for_candidate(source, benchmark_msa_records) if use_target_msa else {}
        used_msa_records: set[str] = set()
        sequence_entries: list[dict[str, Any]] = []
        for chain, sequence in sequences.items():
            protein: dict[str, Any] = {"sequence": sequence, "count": 1}
            chain_dir = msa_root / safe_id / f"chain_{_safe_id(chain)}"
            if use_target_msa and chain in target_set:
                record_chain = chain if chain in msa_records else ""
                if not record_chain:
                    for candidate_chain, record in msa_records.items():
                        if candidate_chain in used_msa_records:
                            continue
                        if _clean_sequence(record.get("sequence")) == _clean_sequence(sequence):
                            record_chain = candidate_chain
                            break
                record = msa_records.get(record_chain) if record_chain else None
                if record is None:
                    metrics["protenix_target_msa_missing_count"] += 1
                elif _clean_sequence(record.get("sequence")) and _clean_sequence(record.get("sequence")) != _clean_sequence(sequence):
                    metrics["protenix_target_msa_sequence_mismatch_count"] += 1
                else:
                    source_msa = Path(str(record.get("msa_path") or ""))
                    if source_msa.exists():
                        chain_dir.mkdir(parents=True, exist_ok=True)
                        for filename in ["pairing.a3m", "non_pairing.a3m"]:
                            shutil.copy2(source_msa, chain_dir / filename)
                        attach_msa(protein, chain_dir)
                        metrics["protenix_target_msa_injected_count"] += 1
                        used_msa_records.add(record_chain)
                    else:
                        metrics["protenix_target_msa_missing_count"] += 1
            if use_target_msa and "msa" not in protein:
                write_query_only_msa(chain_dir, safe_id, chain, sequence)
                attach_msa(protein, chain_dir)
                if chain in target_set:
                    metrics["protenix_target_query_only_msa_fallback_count"] += 1
                else:
                    metrics["protenix_binder_query_only_msa_count"] += 1
            sequence_entries.append({"proteinChain": protein})
        candidate_json_dir = json_root / safe_id
        candidate_json_dir.mkdir(parents=True, exist_ok=True)
        input_path = candidate_json_dir / f"{safe_id}.json"
        write_json(input_path, [{"sequences": sequence_entries, "name": safe_id}])
        protenix_inputs[safe_id] = candidate_json_dir
    return protenix_inputs, metrics


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


def _target_output_chain_ids(target_chains: list[str]) -> list[str]:
    reserved = {"A"}
    fallback = [chr(code) for code in range(ord("B"), ord("Z") + 1)]
    output_ids: list[str] = []
    for index, chain in enumerate(target_chains):
        candidate = str(chain or "").strip()
        if len(candidate) != 1 or candidate in reserved or candidate in output_ids:
            candidate = next((item for item in fallback if item not in reserved and item not in output_ids), "Z")
        output_ids.append(candidate)
    return output_ids


def _write_engine_chain_map(
    input_dir: Path,
    safe_id: str,
    *,
    binder_source_chains: list[str],
    target_source_chains: list[str],
    target_engine_chains: list[str],
) -> None:
    payload = {
        "candidate_id": safe_id,
        "binder": [
            {"original_chain": chain, "engine_chain": "A"}
            for chain in binder_source_chains
        ],
        "targets": [
            {"original_chain": original, "engine_chain": engine}
            for original, engine in zip(target_source_chains, target_engine_chains)
        ],
    }
    write_json(input_dir / f"{safe_id}.chain_map.json", payload)


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
        target_engine_chains = _target_output_chain_ids(inferred_target_chains)
        target_lines: list[str] = []
        for source_chain, output_chain in zip(inferred_target_chains, target_engine_chains):
            chain_lines, next_atom = _renumber_structure_chain(path, output_chain, next_atom, {source_chain})
            if chain_lines:
                target_lines.extend(chain_lines + ["TER"])
        if not binder_lines or not target_lines:
            return None
        output.write_text("\n".join(binder_lines + ["TER"] + target_lines + ["END", ""]))
        _write_engine_chain_map(
            input_dir,
            safe_id,
            binder_source_chains=inferred_binder_chains,
            target_source_chains=inferred_target_chains,
            target_engine_chains=target_engine_chains,
        )
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
    target_structure_chains = _structure_chains(target)
    requested_targets = _candidate_chains(candidate, "target_chains", target_structure_chains)
    target_source_chains = [chain for chain in requested_targets if chain in target_structure_chains] or target_structure_chains
    target_engine_chains = _target_output_chain_ids(target_source_chains)
    target_lines: list[str] = []
    for source_chain, output_chain in zip(target_source_chains, target_engine_chains):
        chain_lines, next_atom = _renumber_structure_chain(target, output_chain, next_atom, {source_chain})
        if chain_lines:
            target_lines.extend(chain_lines + ["TER"])
    output.write_text("\n".join(binder_lines + ["TER"] + target_lines + ["END", ""]))
    _write_engine_chain_map(
        input_dir,
        safe_id,
        binder_source_chains=binder_chains,
        target_source_chains=target_source_chains,
        target_engine_chains=target_engine_chains,
    )
    return output


def _stage_complex_inputs(source_run_dir: Path, source_candidates: list[dict[str, Any]], input_dir: Path) -> dict[str, Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    for candidate in source_candidates:
        safe_id = _safe_id(candidate.get("candidate_id"))
        complex_path = _complex_pdb_for_candidate(source_run_dir, candidate, input_dir)
        staged_path = input_dir / f"{safe_id}.pdb"
        if complex_path.resolve() != staged_path.resolve():
            source_map = complex_path.with_suffix(".chain_map.json")
            shutil.copy2(complex_path, staged_path)
            if source_map.exists():
                shutil.copy2(source_map, input_dir / f"{safe_id}.chain_map.json")
            if complex_path.parent.resolve() == input_dir.resolve():
                complex_path.unlink(missing_ok=True)
                source_map.unlink(missing_ok=True)
        elif not (input_dir / f"{safe_id}.chain_map.json").exists():
            binder_chains, target_chains = _infer_chain_roles(source_run_dir, candidate, staged_path)
            _write_engine_chain_map(
                input_dir,
                safe_id,
                binder_source_chains=binder_chains,
                target_source_chains=target_chains,
                target_engine_chains=target_chains,
            )
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


def _candidate_binder_sequence(source_run_dir: Path, candidate: dict[str, Any]) -> str:
    sequence = str(candidate.get("binder_sequence") or "").strip().replace(" ", "")
    if sequence:
        return sequence
    structure_path = _resolve_candidate_path(source_run_dir, candidate.get("binder_pdb") or candidate.get("complex_pdb"))
    if structure_path and structure_path.exists():
        binder_chains, _target_chains = _infer_chain_roles(source_run_dir, candidate, structure_path)
        if structure_path.suffix.lower() == ".pdb":
            sequence = _sequence_from_pdb(structure_path, binder_chains)
        elif structure_path.suffix.lower() == ".cif" or structure_path.name.endswith(".cif.gz"):
            sequence = _sequence_from_cif(structure_path, binder_chains)
    if not sequence:
        raise ValueError(f"Candidate {candidate.get('candidate_id')} has no binder sequence or readable binder structure.")
    return sequence


def _chain_initial_guess_distogram(
    path: Path | None,
    chain: str,
    expected_length: int,
) -> tuple[np.ndarray | None, str | None]:
    if path is None or not path.exists():
        return None, "no input structure available"
    coords = _ordered_ca_coords(path, {chain})
    if len(coords) != expected_length:
        return None, f"chain {chain} CA count {len(coords)} does not match sequence length {expected_length}"
    if len(coords) < 3:
        return None, f"chain {chain} is too short for distogram conditioning"
    array = np.asarray(coords, dtype=np.float32)
    deltas = array[:, None, :] - array[None, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=-1)).astype(np.float32), None


def _as_numpy_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        return np.asarray(value)
    except Exception:
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, np.generic):
            value = value.item()
        return float(value)
    except (TypeError, ValueError):
        return None


def _array_summary(values: np.ndarray | None) -> dict[str, float | None]:
    if values is None:
        return {"mean": None, "median": None, "min": None, "max": None}
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _result_chain_indices(complex_obj: Any) -> dict[str, list[int]]:
    chain_lookup = dict(getattr(complex_obj.metadata, "chain_lookup", {}) or {})
    chain_indices: dict[str, list[int]] = {}
    for index, chain_numeric in enumerate(complex_obj.chain_id):
        chain = str(chain_lookup.get(int(chain_numeric), chain_numeric))
        chain_indices.setdefault(chain, []).append(index)
    return chain_indices


def _matrix_block(matrix: np.ndarray | None, rows: list[int], cols: list[int]) -> np.ndarray | None:
    if matrix is None or not rows or not cols or matrix.ndim < 2:
        return None
    return np.asarray(matrix[np.ix_(rows, cols)], dtype=float)


def _esmfold2_confidence_analysis(
    result: Any,
    *,
    binder_chain: str,
    target_chains: list[str],
    contact_cutoff: float,
    output_prefix: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    chain_indices = _result_chain_indices(result.complex)
    binder_indices = chain_indices.get(binder_chain, [])
    target_indices = [
        index
        for chain in target_chains
        for index in chain_indices.get(chain, [])
    ]
    plddt = _as_numpy_array(getattr(result, "plddt", None))
    pae = _as_numpy_array(getattr(result, "pae", None))
    pair_chains_iptm = _as_numpy_array(getattr(result, "pair_chains_iptm", None))
    distogram = _as_numpy_array(getattr(result, "distogram", None))

    binder_plddt = plddt[binder_indices] if plddt is not None and binder_indices else None
    target_plddt = plddt[target_indices] if plddt is not None and target_indices else None
    interface_pae_blocks = [
        _matrix_block(pae, target_indices, binder_indices),
        _matrix_block(pae, binder_indices, target_indices),
    ]
    interface_pae = np.concatenate(
        [block.reshape(-1) for block in interface_pae_blocks if block is not None],
    ) if any(block is not None for block in interface_pae_blocks) else None
    contact_values: list[float] = []
    contact_pairs = 0
    if pae is not None and binder_indices and target_indices:
        residues = esm_binder_workflow._chain_residue_atoms(result.complex)
        binder_residues = residues.get(binder_chain, {})
        for target_chain in target_chains:
            for binder_residue, binder_atoms in binder_residues.items():
                if binder_residue - 1 >= len(binder_indices):
                    continue
                binder_token = binder_indices[binder_residue - 1]
                for target_residue, target_atoms in residues.get(target_chain, {}).items():
                    if target_residue - 1 >= len(chain_indices.get(target_chain, [])):
                        continue
                    distance = esm_binder_workflow._min_distance(binder_atoms, target_atoms)
                    if distance is None or distance > contact_cutoff:
                        continue
                    target_token = chain_indices[target_chain][target_residue - 1]
                    contact_pairs += 1
                    contact_values.append(float(pae[binder_token, target_token]))
                    contact_values.append(float(pae[target_token, binder_token]))
    contact_interface_pae = np.asarray(contact_values, dtype=float) if contact_values else None
    binder_pae = _matrix_block(pae, binder_indices, binder_indices)
    target_pae = _matrix_block(pae, target_indices, target_indices)

    plddt_summary = _array_summary(plddt)
    binder_plddt_summary = _array_summary(binder_plddt)
    target_plddt_summary = _array_summary(target_plddt)
    pae_summary = _array_summary(pae)
    interface_pae_summary = _array_summary(interface_pae)
    contact_interface_pae_summary = _array_summary(contact_interface_pae)
    binder_pae_summary = _array_summary(binder_pae)
    target_pae_summary = _array_summary(target_pae)
    pair_chains_summary = _array_summary(pair_chains_iptm)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    arrays_path = output_prefix.with_suffix(".confidence_arrays.npz")
    pae_json_path = output_prefix.with_name(f"{output_prefix.name}_pae.json")
    np.savez_compressed(
        arrays_path,
        plddt=plddt if plddt is not None else np.array([], dtype=np.float32),
        pae=pae if pae is not None else np.array([], dtype=np.float32),
        pair_chains_iptm=pair_chains_iptm if pair_chains_iptm is not None else np.array([], dtype=np.float32),
        distogram=distogram if distogram is not None else np.array([], dtype=np.float32),
        binder_indices=np.asarray(binder_indices, dtype=np.int32),
        target_indices=np.asarray(target_indices, dtype=np.int32),
    )
    if pae is not None:
        write_json(
            pae_json_path,
            {
                "predicted_aligned_error": np.asarray(pae, dtype=float).tolist(),
                "pae": np.asarray(pae, dtype=float).tolist(),
                "max_predicted_aligned_error": pae_summary["max"],
                "binder_chain": binder_chain,
                "target_chains": target_chains,
            },
        )
    analysis = {
        "chain_indices": chain_indices,
        "binder_chain": binder_chain,
        "target_chains": target_chains,
        "has_pae": pae is not None,
        "has_distogram": distogram is not None,
        "has_pair_chains_iptm": pair_chains_iptm is not None,
        "ptm": _float_or_none(getattr(result, "ptm", None)),
        "iptm": _float_or_none(getattr(result, "iptm", None)),
        "plddt": plddt_summary,
        "binder_plddt": binder_plddt_summary,
        "target_plddt": target_plddt_summary,
        "pae": pae_summary,
        "interface_pae": interface_pae_summary,
        "contact_interface_pae": contact_interface_pae_summary,
        "contact_interface_pair_count": contact_pairs,
        "binder_self_pae": binder_pae_summary,
        "target_self_pae": target_pae_summary,
        "pair_chains_iptm": pair_chains_summary,
        "array_artifact": arrays_path.name,
        "pae_json_artifact": pae_json_path.name if pae is not None else None,
        "array_shapes": {
            "plddt": list(plddt.shape) if plddt is not None else None,
            "pae": list(pae.shape) if pae is not None else None,
            "pair_chains_iptm": list(pair_chains_iptm.shape) if pair_chains_iptm is not None else None,
            "distogram": list(distogram.shape) if distogram is not None else None,
        },
    }
    analysis_path = output_prefix.with_suffix(".confidence.json")
    write_json(analysis_path, analysis)
    metrics = {
        "esmfold2_has_pae": pae is not None,
        "esmfold2_has_distogram": distogram is not None,
        "esmfold2_has_pair_chains_iptm": pair_chains_iptm is not None,
        "esmfold2_plddt_mean": plddt_summary["mean"],
        "esmfold2_plddt_min": plddt_summary["min"],
        "esmfold2_binder_plddt_mean": binder_plddt_summary["mean"],
        "esmfold2_target_plddt_mean": target_plddt_summary["mean"],
        "esmfold2_pae_mean": pae_summary["mean"],
        "esmfold2_pae_max": pae_summary["max"],
        "esmfold2_ipae_mean": interface_pae_summary["mean"],
        "esmfold2_ipae_min": interface_pae_summary["min"],
        "esmfold2_contact_ipae_mean": contact_interface_pae_summary["mean"],
        "esmfold2_contact_ipae_min": contact_interface_pae_summary["min"],
        "esmfold2_contact_ipae_pairs": contact_pairs,
        "esmfold2_binder_self_pae_mean": binder_pae_summary["mean"],
        "esmfold2_target_self_pae_mean": target_pae_summary["mean"],
        "esmfold2_pair_chains_iptm_mean": pair_chains_summary["mean"],
        "binder_plddt": binder_plddt_summary["mean"],
        "confidence": plddt_summary["mean"],
        "pae": pae_summary["mean"],
        "ipae": contact_interface_pae_summary["mean"] if contact_interface_pae_summary["mean"] is not None else interface_pae_summary["mean"],
        "ipae_binder_to_target": _array_summary(_matrix_block(pae, binder_indices, target_indices))["mean"],
        "ipae_target_to_binder": _array_summary(_matrix_block(pae, target_indices, binder_indices))["mean"],
        "interaction_pae": contact_interface_pae_summary["mean"] if contact_interface_pae_summary["mean"] is not None else interface_pae_summary["mean"],
        "min_interaction_pae": contact_interface_pae_summary["min"] if contact_interface_pae_summary["min"] is not None else interface_pae_summary["min"],
        "ipae_contact_pairs": contact_pairs,
        "ipae_dist_cutoff": contact_cutoff,
        "pair_chains_iptm": pair_chains_summary["mean"],
        "ipsae_ready": pae is not None,
        "esmfold2_confidence_json": str(analysis_path.name),
        "esmfold2_confidence_arrays": str(arrays_path.name),
        "esmfold2_pae_json": str(pae_json_path.name) if pae is not None else None,
    }
    return metrics, analysis


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
                "--shm-size=64G",
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
    max_candidates: int = 20,
    num_sampling_steps: int = 32,
    seed: int = 0,
    device: str = "auto",
    contact_cutoff: float = 8.0,
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
    if tool in {"esmfold2_complex_validation", "esmfold2_initial_guess_validation"}:
        return run_esmfold2_complex_validation(
            source_run_dir,
            candidates_jsonl,
            max_candidates=max_candidates,
            num_loops=num_recycles,
            num_sampling_steps=num_sampling_steps,
            seed=seed,
            device=device,
            contact_cutoff=contact_cutoff,
            use_initial_guess=tool == "esmfold2_initial_guess_validation",
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


def run_esmfold2_complex_validation(
    source_run_dir: Path,
    candidates_jsonl: Path,
    max_candidates: int = 20,
    num_loops: int = 3,
    num_sampling_steps: int = 32,
    seed: int = 0,
    device: str = "auto",
    contact_cutoff: float = 8.0,
    use_initial_guess: bool = False,
) -> Path:
    source_run_dir = Path(source_run_dir)
    candidates_jsonl = Path(candidates_jsonl)
    allowed = {
        STAGE_MONOMER_REFOLDING,
        STAGE_SEQUENCE_DESIGN,
        STAGE_GENERATION_BACKBONE_SEQUENCE,
        STAGE_COMPLEX_REFOLDING,
    }
    source_candidates = _source_candidates(candidates_jsonl, allowed)
    if max_candidates > 0:
        source_candidates = source_candidates[: int(max_candidates)]

    tool_name = "esmfold2_initial_guess_validation" if use_initial_guess else "esmfold2_complex_validation"
    backend_name = tool_name
    job = create_job(
        REFOLDING_GROUP,
        job_type="complex_refolding",
        tool=tool_name,
        inputs={"source_run_dir": str(source_run_dir), "candidates_jsonl": str(candidates_jsonl)},
        params={
            "max_candidates": max_candidates,
            "num_loops": num_loops,
            "num_sampling_steps": num_sampling_steps,
            "seed": seed,
            "device": device,
            "contact_cutoff": contact_cutoff,
            "use_initial_guess": use_initial_guess,
            "backend": "biohub_esmfold2_local",
            "model_dir": str(esm_binder_workflow.ESMFOLD2_MODEL_DIR),
            "esmc_model_dir": str(esm_binder_workflow.ESMC_MODEL_DIR),
        },
    )
    update_status(job.run_dir, "running")
    raw_dir = job.run_dir / "artifacts" / "raw" / tool_name
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        esm_binder_workflow._ensure_esm_import_path()
        try:
            from esm.models.esmfold2 import (
                DistogramConditioning,
                ESMFold2InputBuilder,
                ProteinInput,
                StructurePredictionInput,
            )
        except Exception as exc:  # pragma: no cover - dependency/runtime specific
            raise RuntimeError(
                "ESMFold2 runtime dependencies are not importable. "
                "Run this from the Biohub ESM environment/container before starting validation."
            ) from exc

        model = esm_binder_workflow._load_esmfold2_model(device)
        builder = ESMFold2InputBuilder(ccd_cache=esm_binder_workflow.ESMFOLD2_MODEL_DIR)
        candidates: list[dict[str, Any]] = []
        input_rows: list[dict[str, Any]] = []
        confidence_rows: list[dict[str, Any]] = []
        with (job.run_dir / "stdout.log").open("a") as stdout:
            stdout.write(f"{tool_name} of existing candidates only.\n")
            stdout.flush()
            for index, source in enumerate(source_candidates, start=1):
                parent_id = str(source.get("candidate_id") or f"candidate_{index:05d}")
                safe_parent_id = _safe_id(parent_id)
                candidate_id = f"{safe_parent_id}_{'esmfold2ig' if use_initial_guess else 'esmfold2cv'}"
                target_pdb = _target_pdb_for_candidate(source_run_dir, source)
                if target_pdb is None or not target_pdb.exists():
                    raise ValueError(f"Candidate {parent_id} does not have a readable target PDB.")
                source_complex = _resolve_candidate_path(source_run_dir, source.get("complex_pdb"))
                binder_sequence = _candidate_binder_sequence(source_run_dir, source)
                binder_chains, inferred_target_chains = _infer_chain_roles(source_run_dir, source, source_complex)
                target_sequences, residue_maps = esm_binder_workflow._target_sequences(target_pdb, inferred_target_chains)
                target_chains = list(target_sequences)
                binder_chain = esm_binder_workflow._choose_binder_chain(target_chains)
                hotspots = source.get("hotspots") or []
                mapped_hotspots = esm_binder_workflow._mapped_hotspots(hotspots, residue_maps)
                distogram_conditioning = None
                initial_guess_note = None
                if use_initial_guess:
                    distogram_conditioning = []
                    notes: list[str] = []
                    for target_chain, target_sequence in target_sequences.items():
                        distogram, note = _chain_initial_guess_distogram(
                            target_pdb,
                            target_chain,
                            len(target_sequence),
                        )
                        if distogram is not None:
                            distogram_conditioning.append(
                                DistogramConditioning(chain_id=target_chain, distogram=distogram)
                            )
                        elif note:
                            notes.append(note)
                    if distogram_conditioning:
                        initial_guess_note = "target distogram conditioning applied"
                    else:
                        initial_guess_note = "; ".join(notes) if notes else "target distogram conditioning unavailable"
                stdout.write(
                    f"Folding {candidate_id} target_chains={','.join(target_chains)} "
                    f"binder_chain={binder_chain} binder_length={len(binder_sequence)}"
                    f" initial_guess={bool(distogram_conditioning)}\n"
                )
                stdout.flush()
                spi = StructurePredictionInput(
                    sequences=[
                        *[
                            ProteinInput(id=chain, sequence=sequence)
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
                    seed=int(seed) + index - 1,
                    complex_id=candidate_id,
                )
                complex_path = raw_dir / f"{candidate_id}.cif"
                complex_path.write_text(result.complex.to_mmcif())
                confidence_metrics, confidence_analysis = _esmfold2_confidence_analysis(
                    result,
                    binder_chain=binder_chain,
                    target_chains=target_chains,
                    contact_cutoff=float(contact_cutoff),
                    output_prefix=raw_dir / candidate_id,
                )
                metrics = dict(source.get("metrics") or {})
                _update_source_monomer_rmsd(source_run_dir, source, metrics)
                metrics.update(
                    {
                        "complex_refolding_backend": backend_name,
                        "initial_guess_used": bool(distogram_conditioning),
                        "initial_guess_note": initial_guess_note,
                        "iptm": float(result.iptm) if result.iptm is not None else None,
                        "ptm": float(result.ptm) if result.ptm is not None else None,
                        "plddt_mean": esm_binder_workflow._mean_plddt(result),
                        "binder_length": len(binder_sequence),
                        **confidence_metrics,
                        **esm_binder_workflow._hotspot_metrics_from_complex(
                            result.complex,
                            binder_chain=binder_chain,
                            target_chains=target_chains,
                            mapped_hotspots=mapped_hotspots,
                            contact_cutoff=float(contact_cutoff),
                        ),
                    }
                )
                metrics["esmfold2_validation_score"] = esm_binder_workflow._ranking_score(metrics)
                candidates.append(
                    {
                        **source,
                        "candidate_id": candidate_id,
                        "stage": STAGE_COMPLEX_REFOLDING,
                        "source_tool": tool_name,
                        "tool": tool_name,
                        "target_pdb": str(target_pdb),
                        "complex_pdb": _rel_path(job.run_dir, complex_path),
                        "binder_sequence": binder_sequence,
                        "target_chains": target_chains,
                        "binder_chains": [binder_chain],
                        "binder_length": str(len(binder_sequence)),
                        "metrics": metrics,
                        "parents": [parent_id],
                        "raw_metadata": {
                            **dict(source.get("raw_metadata") or {}),
                            "source_candidate": source,
                            "backend_status": "biohub_esmfold2_local",
                            "initial_guess_used": bool(distogram_conditioning),
                            "initial_guess_note": initial_guess_note,
                            "confidence_json": confidence_metrics.get("esmfold2_confidence_json"),
                            "confidence_arrays": confidence_metrics.get("esmfold2_confidence_arrays"),
                            "pae_path": _rel_path(job.run_dir, raw_dir / str(confidence_metrics.get("esmfold2_pae_json") or "")) if confidence_metrics.get("esmfold2_pae_json") else None,
                            "prediction_dir": _rel_path(job.run_dir, raw_dir),
                            "input_target_chains": inferred_target_chains,
                            "mapped_hotspots": [f"{chain}{residue}" for chain, residue in mapped_hotspots],
                        },
                    }
                )
                confidence_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "parent_id": parent_id,
                        "confidence": confidence_analysis,
                    }
                )
                input_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "parent_id": parent_id,
                        "target_pdb": str(target_pdb),
                        "target_chains": target_chains,
                        "binder_chain": binder_chain,
                        "binder_length": len(binder_sequence),
                        "initial_guess_used": bool(distogram_conditioning),
                        "initial_guess_note": initial_guess_note,
                        "hotspots": hotspots,
                    }
                )

        (raw_dir / "binder_sequences.fasta").write_text(
            "".join(
                f">{candidate['candidate_id']}\n{candidate['binder_sequence']}\n"
                for candidate in candidates
            )
        )
        write_json(raw_dir / "input_mapping.json", input_rows)
        write_json(raw_dir / "confidence_summary.json", {"candidates": confidence_rows})
        candidates.sort(
            key=lambda candidate: candidate["metrics"].get("esmfold2_validation_score") or 0.0,
            reverse=True,
        )
        for rank, candidate in enumerate(candidates, start=1):
            candidate["metrics"]["esmfold2_validation_rank"] = rank
        normalized = write_candidates(job.run_dir, tool_name, candidates)
        artifacts = _collect_refolding_artifacts(job.run_dir)
        finish_job(
            job.run_dir,
            bool(normalized),
            {
                "outputs": {"artifacts": artifacts, "candidates": normalized},
                "metrics": {
                    "candidate_count": len(normalized),
                    "artifact_count": len(artifacts),
                    "best_score": normalized[0]["metrics"].get("esmfold2_validation_score") if normalized else None,
                },
                "downstream_artifacts": {
                    "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                    "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
                },
            },
        )
        return job.run_dir
    except Exception as exc:
        with (job.run_dir / "stderr.log").open("a") as stderr:
            stderr.write(f"{type(exc).__name__}: {exc}\n")
        finish_job(job.run_dir, False, {"metrics": {"error": str(exc)}})
        raise


def run_af2_initial_guess_complex_refolding(
    source_run_dir: Path,
    candidates_jsonl: Path,
    require_monomer_success: bool = True,
    num_recycles: int = 3,
    multimer: bool = True,
    use_binder_template: bool = False,
    use_interface_template: bool = False,
    docker_image: str = "ovo-colabdesign:latest",
    internal_parent_run_dir: Path | None = None,
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
            "image": docker_image,
            "alphafold_models_dir": str(ALPHAFOLD_MODELS_DIR),
        },
    )
    if internal_parent_run_dir is not None:
        mark_internal_job(
            job.run_dir,
            parent_run_dir=Path(internal_parent_run_dir),
            parent_task_group="benchmark",
            role="benchmark_engine_subrun",
            engine="af2_initial_guess",
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
        "--shm-size=64G",
        "-v",
        f"{job.run_dir}:/work",
        "-v",
        f"{AF2_BINDER_EVAL.parent}:/scripts:ro",
        "-v",
        f"{ALPHAFOLD_MODELS_DIR}:/models:ro",
        "-w",
        "/work",
        docker_image,
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
    recycling_steps: int = 10,
    sampling_steps: int = 200,
    diffusion_samples: int = 3,
    write_full_pae: bool = True,
    internal_parent_run_dir: Path | None = None,
    benchmark_run_csv: Path | None = None,
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
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
            "write_full_pae": write_full_pae,
            "benchmark_run_csv": str(benchmark_run_csv) if benchmark_run_csv else None,
        },
    )
    if internal_parent_run_dir is not None:
        mark_internal_job(
            job.run_dir,
            parent_run_dir=Path(internal_parent_run_dir),
            parent_task_group="benchmark",
            role="benchmark_engine_subrun",
            engine="boltz2",
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
        staged_template = yaml_dir / "target_template.pdb"
        shutil.copy2(target_template, staged_template)
        staged_template_arg = "--template /work/artifacts/raw/boltz2_initial_guess/yaml_inputs/target_template.pdb "
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
        "--shm-size=32G",
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
        "--recycling_steps",
        str(int(recycling_steps)),
        "--sampling_steps",
        str(int(sampling_steps)),
        "--diffusion_samples",
        str(int(diffusion_samples)),
    ]
    if write_full_pae:
        predict_command.append("--write_full_pae")
    cleanup_command = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{job.run_dir}:/work",
        "-w",
        "/work",
        "--entrypoint",
        "/bin/bash",
        "ovoex-boltz2:latest",
        "-lc",
        (
            "cp /work/artifacts/raw/boltz2_initial_guess/yaml_inputs/target_template.cif "
            "/work/artifacts/raw/boltz2_initial_guess/target_template.cif && "
            "rm -f /work/artifacts/raw/boltz2_initial_guess/yaml_inputs/target_template.pdb "
            "/work/artifacts/raw/boltz2_initial_guess/yaml_inputs/target_template.cif && "
            "sed -i 's#cif: target_template.cif#cif: /work/artifacts/raw/boltz2_initial_guess/target_template.cif#g' "
            "/work/artifacts/raw/boltz2_initial_guess/yaml_inputs/*.yaml"
        ),
    ]
    rc = _run_shell_steps(
        job.run_dir,
        [
            {"name": "boltz2-prepare-inputs", "command": prep_command},
        ],
    )
    msa_metrics = {}
    if rc == 0:
        msa_metrics = _inject_boltz_yaml_msas(
            yaml_dir=yaml_dir,
            raw_root=raw_root,
            benchmark_run_csv=benchmark_run_csv,
        )
        rc = _run_shell_steps(
            job.run_dir,
            [
                {"name": "boltz2-clean-template-pdb", "command": cleanup_command},
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
            "metrics": {
                "return_code": rc,
                "candidate_count": len(normalized),
                "artifact_count": len(artifacts),
                **msa_metrics,
            },
            "downstream_artifacts": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
            },
        },
    )
    return job.run_dir


def _first_prediction_file(output_dir: Path, safe_id: str) -> Path | None:
    for pattern in (
        f"**/{safe_id}*_model.cif",
        f"**/{safe_id}*.cif",
        f"**/{safe_id}*.pdb",
        "**/*_model.cif",
        "**/*.cif",
        "**/*.pdb",
    ):
        matches = sorted(output_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _prediction_summary_metrics(output_dir: Path, prefix: str) -> dict[str, Any]:
    summary_paths: list[Path] = []
    for pattern in ("**/*summary_confidence*.json", "**/*summary_confidences*.json", "**/confidence*.json"):
        summary_paths = sorted(output_dir.glob(pattern))
        if summary_paths:
            break
    if not summary_paths:
        npz_paths = sorted(output_dir.glob("**/*.npz"))
        if not npz_paths:
            return {}
        try:
            payload = np.load(npz_paths[0])
            return {
                f"{prefix}_{key}": float(np.asarray(payload[key]).mean())
                for key in payload.files
                if np.asarray(payload[key]).dtype.kind in "biuf" and np.asarray(payload[key]).size
            }
        except (OSError, ValueError):
            return {}
    try:
        payload = json.loads(summary_paths[0].read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        return {}
    return {
        f"{prefix}_{key}": value
        for key, value in payload.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    }


def _prediction_pae_file(output_dir: Path, prefix: str) -> Path | None:
    patterns = {
        "rf3": ("**/*_confidences.json", "**/*confidences.json"),
        "protenix": ("**/*_full_data_sample_*.json", "**/*full_data*.json"),
        "boltzgen_fold": ("**/*.npz",),
    }.get(prefix, ("**/*pae*.json", "**/*confidences.json", "**/*.npz"))
    for pattern in patterns:
        for path in sorted(output_dir.glob(pattern)):
            name = path.name.lower()
            if "summary" in name:
                continue
            if prefix == "boltzgen_fold":
                try:
                    payload = np.load(path)
                    if any(np.asarray(payload[key]).ndim >= 2 and "pae" in key.lower() for key in payload.files):
                        return path
                except (OSError, ValueError):
                    continue
                continue
            return path
    return None


def _standard_pae_json_path(pae_source: Path, output_dir: Path, safe_id: str, prefix: str) -> Path | None:
    if pae_source.suffix.lower() == ".npz":
        try:
            payload = np.load(pae_source)
            matrix_array = None
            for key in ("predicted_aligned_error", "pae", "token_pair_pae"):
                if key in payload.files:
                    matrix_array = np.asarray(payload[key])
                    break
            if matrix_array is None or matrix_array.ndim < 2:
                return None
            if matrix_array.ndim == 3:
                matrix_array = matrix_array[0]
            matrix = np.asarray(matrix_array, dtype=float).tolist()
            plddt = None
            for key in ("plddt", "atom_plddt", "token_plddt"):
                if key in payload.files:
                    plddt_array = np.asarray(payload[key])
                    if plddt_array.ndim > 1:
                        plddt_array = plddt_array[0]
                    plddt = np.asarray(plddt_array, dtype=float).tolist()
                    break
        except (OSError, ValueError, KeyError):
            return None
    else:
        if pae_source.suffix.lower() != ".json":
            return None
        try:
            payload = json.loads(pae_source.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        matrix = payload.get("predicted_aligned_error") or payload.get("pae") or payload.get("token_pair_pae")
        plddt = payload.get("plddt")
        if plddt is None:
            plddt = payload.get("atom_plddts") or payload.get("atom_plddt") or payload.get("token_plddt")
    if matrix is None:
        return None
    target = output_dir / f"{safe_id}_{prefix}_pae.json"
    target.write_text(
        json.dumps(
            {
                "predicted_aligned_error": matrix,
                "pae": matrix,
                "plddt": plddt,
                "source_pae_file": str(pae_source.name),
            }
        )
    )
    return target


def _finish_external_complex_refolding(
    *,
    job_run_dir: Path,
    source_run_dir: Path,
    source_candidates: list[dict[str, Any]],
    staged: dict[str, Path],
    output_root: Path,
    tool: str,
    rc: int,
    extra_metrics: dict[str, Any] | None = None,
) -> Path:
    candidates: list[dict[str, Any]] = []
    if rc == 0:
        for safe_id, staged_path in staged.items():
            source = next(item for item in source_candidates if safe_id == _safe_id(item.get("candidate_id")))
            candidate_output = output_root / safe_id
            prediction = _first_prediction_file(candidate_output, safe_id)
            pae_source = _prediction_pae_file(candidate_output, tool)
            pae_path = _standard_pae_json_path(pae_source, candidate_output, safe_id, tool) if pae_source else None
            binder_chains, target_chains = _infer_chain_roles(source_run_dir, source, prediction or staged_path)
            metrics = dict(source.get("metrics") or {})
            metrics.update(_prediction_summary_metrics(candidate_output, tool))
            metrics["complex_refolding_backend"] = tool
            metrics[f"{tool}_has_full_pae"] = pae_path is not None
            candidates.append(
                {
                    **source,
                    "candidate_id": _complex_candidate_id(
                        source,
                        tool,
                        "target_template" if tool == "boltzgen_fold" else "sequence",
                    ),
                    "stage": STAGE_COMPLEX_REFOLDING,
                    "source_tool": tool,
                    "tool": tool,
                    "binder_chains": binder_chains,
                    "target_chains": target_chains,
                    "complex_pdb": _rel_path(job_run_dir, prediction) if prediction else _rel_path(job_run_dir, staged_path),
                    "metrics": metrics,
                    "parents": [str(source.get("candidate_id") or "")],
                    "raw_metadata": {
                        **dict(source.get("raw_metadata") or {}),
                        "source_candidate": source,
                        "input_complex": _rel_path(job_run_dir, staged_path),
                        "prediction_dir": _rel_path(job_run_dir, candidate_output),
                        "pae_source_path": _rel_path(job_run_dir, pae_source) if pae_source else None,
                        "pae_path": _rel_path(job_run_dir, pae_path) if pae_path else None,
                    },
                }
            )
    normalized = write_candidates(job_run_dir, tool, candidates) if candidates else []
    artifacts = _collect_refolding_artifacts(job_run_dir)
    finish_job(
        job_run_dir,
        rc == 0 and bool(normalized),
        {
            "outputs": {"artifacts": artifacts, "candidates": normalized},
            "metrics": {
                "return_code": rc,
                "candidate_count": len(normalized),
                "artifact_count": len(artifacts),
                **(extra_metrics or {}),
            },
            "downstream_artifacts": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
            },
        },
    )
    return job_run_dir


def run_rf3_complex_refolding(
    source_run_dir: Path,
    candidates_jsonl: Path,
    require_monomer_success: bool = True,
    checkpoint_path: Path = RF3_CHECKPOINT,
    use_target_msa: bool = True,
    n_recycles: int = 10,
    num_steps: int = 50,
    diffusion_batch_size: int = 5,
    seed: int = 0,
    benchmark_run_csv: Path | None = None,
    internal_parent_run_dir: Path | None = None,
) -> Path:
    source_run_dir = Path(source_run_dir)
    allowed = {STAGE_MONOMER_REFOLDING} if require_monomer_success else {
        STAGE_MONOMER_REFOLDING,
        STAGE_SEQUENCE_DESIGN,
        STAGE_GENERATION_BACKBONE_SEQUENCE,
        STAGE_COMPLEX_REFOLDING,
    }
    source_candidates = _source_candidates(Path(candidates_jsonl), allowed)
    job = create_job(
        REFOLDING_GROUP,
        "complex_refolding",
        "rf3",
        {"source_run_dir": str(source_run_dir), "candidates_jsonl": str(candidates_jsonl)},
        {
            "image": RF3_IMAGE,
            "checkpoint_path": str(checkpoint_path),
            "initial_guess_supported": False,
            "target_msa_supported": True,
            "use_target_msa": use_target_msa,
            "n_recycles": n_recycles,
            "num_steps": num_steps,
            "diffusion_batch_size": diffusion_batch_size,
            "seed": seed,
            "benchmark_run_csv": str(benchmark_run_csv) if benchmark_run_csv else None,
        },
    )
    if internal_parent_run_dir is not None:
        mark_internal_job(
            job.run_dir,
            parent_run_dir=Path(internal_parent_run_dir),
            parent_task_group="benchmark",
            role="benchmark_engine_subrun",
            engine="rf3",
        )
    raw_root = job.run_dir / "artifacts" / "raw" / "rf3"
    input_dir = raw_root / "inputs"
    staged = _stage_complex_inputs(source_run_dir, source_candidates, input_dir)
    rf3_inputs, msa_metrics = _write_rf3_json_inputs(
        source_run_dir=source_run_dir,
        source_candidates=source_candidates,
        staged=staged,
        input_dir=input_dir,
        benchmark_run_csv=benchmark_run_csv,
        use_target_msa=use_target_msa,
    )
    output_root = raw_root / "output"
    steps: list[dict[str, Any]] = []
    for safe_id, rf3_input in rf3_inputs.items():
        (output_root / safe_id).mkdir(parents=True, exist_ok=True)
        steps.append(
            {
                "name": f"rf3-{safe_id}",
                "command": [
                    "docker",
                    "run",
                    "--rm",
                    "--gpus",
                    "all",
                    "--shm-size=32G",
                    "-v",
                    f"{job.run_dir}:/work",
                    "-v",
                    f"{checkpoint_path.parent}:/weights:ro",
                    "-w",
                    "/work",
                    RF3_IMAGE,
                    "rf3",
                    "fold",
                    f"inputs=/work/{rf3_input.relative_to(job.run_dir)}",
                    f"ckpt_path=/weights/{checkpoint_path.name}",
                    f"out_dir=/work/artifacts/raw/rf3/output/{safe_id}",
                    f"n_recycles={int(n_recycles)}",
                    f"num_steps={int(num_steps)}",
                    f"diffusion_batch_size={int(diffusion_batch_size)}",
                    f"seed={int(seed)}",
                    "raise_if_missing_msa_for_protein_of_length_n=10000",
                ],
            }
        )
    rc = _run_shell_steps(job.run_dir, steps)
    return _finish_external_complex_refolding(
        job_run_dir=job.run_dir,
        source_run_dir=source_run_dir,
        source_candidates=source_candidates,
        staged=staged,
        output_root=output_root,
        tool="rf3",
        rc=rc,
        extra_metrics={
            "initial_guess_supported": False,
            "target_msa_supported": True,
            "use_target_msa": use_target_msa,
            "n_recycles": n_recycles,
            "num_steps": num_steps,
            "diffusion_batch_size": diffusion_batch_size,
            "seed": seed,
            **msa_metrics,
        },
    )


def run_protenix_complex_refolding(
    source_run_dir: Path,
    candidates_jsonl: Path,
    require_monomer_success: bool = True,
    use_msa: bool = True,
    benchmark_run_csv: Path | None = None,
    cycle: int = 10,
    diffusion_steps: int = 200,
    samples: int = 5,
    internal_parent_run_dir: Path | None = None,
) -> Path:
    source_run_dir = Path(source_run_dir)
    allowed = {STAGE_MONOMER_REFOLDING} if require_monomer_success else {
        STAGE_MONOMER_REFOLDING,
        STAGE_SEQUENCE_DESIGN,
        STAGE_GENERATION_BACKBONE_SEQUENCE,
        STAGE_COMPLEX_REFOLDING,
    }
    source_candidates = _source_candidates(Path(candidates_jsonl), allowed)
    job = create_job(
        REFOLDING_GROUP,
        "complex_refolding",
        "protenix",
        {"source_run_dir": str(source_run_dir), "candidates_jsonl": str(candidates_jsonl)},
        {
            "image": PROTENIX_IMAGE,
            "reference_dir": str(PROTENIX_REFERENCE_DIR),
            "use_msa": use_msa,
            "benchmark_run_csv": str(benchmark_run_csv) if benchmark_run_csv else None,
            "cycle": cycle,
            "diffusion_steps": diffusion_steps,
            "samples": samples,
            "initial_guess_supported": False,
        },
    )
    if internal_parent_run_dir is not None:
        mark_internal_job(
            job.run_dir,
            parent_run_dir=Path(internal_parent_run_dir),
            parent_task_group="benchmark",
            role="benchmark_engine_subrun",
            engine="protenix",
        )
    raw_root = job.run_dir / "artifacts" / "raw" / "protenix"
    staged = _stage_complex_inputs(source_run_dir, source_candidates, raw_root / "inputs")
    protenix_inputs, msa_metrics = _write_protenix_json_inputs(
        source_run_dir=source_run_dir,
        source_candidates=source_candidates,
        staged=staged,
        json_root=raw_root / "json",
        msa_root=raw_root / "msas",
        benchmark_run_csv=benchmark_run_csv,
        use_target_msa=use_msa,
    )
    output_root = raw_root / "output"
    steps: list[dict[str, Any]] = []
    for safe_id, protenix_input_dir in protenix_inputs.items():
        (output_root / safe_id).mkdir(parents=True, exist_ok=True)
        protenix_code = "\n".join(
            [
                "import copy, os",
                "from pathlib import Path",
                "from protenix.utils.file_io import save_json",
                "from protenix.utils.torch_utils import round_values",
                "from runner.batch_inference import get_default_runner",
                "from runner.msa_search import update_infer_json",
                "from runner.inference import infer_predict",
                "from runner import dumper as pxd_dumper",
                f"input_dir = Path('/work/{protenix_input_dir.relative_to(job.run_dir)}')",
                f"out_dir = '/work/artifacts/raw/protenix/output/{safe_id}'",
                "files = sorted(str(path) for path in input_dir.rglob('*.json'))",
                "orig_save_confidence = pxd_dumper.DataDumper._save_confidence",
                "",
                "def save_full_confidence(self, data, prediction_save_dir, sample_name, **kwargs):",
                "    orig_save_confidence(self, data, prediction_save_dir, sample_name, **kwargs)",
                "    full = data.get('full_confidence') or data.get('full_data')",
                "    n = len(full) if full is not None else 0",
                "    for idx in range(n):",
                "        payload = {",
                "            key: value",
                "            for key, value in copy.deepcopy(full[idx]).items()",
                "            if key not in {'atom_coordinate', 'atom_is_polymer'}",
                "        }",
                "        save_json(",
                "            round_values(payload),",
                "            os.path.join(prediction_save_dir, f'{sample_name}_full_data_sample_{idx}.json'),",
                "            indent=4,",
                "        )",
                "",
                "pxd_dumper.DataDumper._save_confidence = save_full_confidence",
                "",
                "for path in files:",
                f"    updated = update_infer_json(path, out_dir=out_dir, use_msa={bool(use_msa)!r})",
                f"    runner = get_default_runner(seeds=(101,), n_cycle={int(cycle)}, n_step={int(diffusion_steps)}, n_sample={int(samples)}, model_name='protenix_base_default_v0.5.0', use_msa={bool(use_msa)!r})",
                "    runner.configs.dump_dir = out_dir",
                "    runner.configs.input_json_path = updated",
                "    runner.configs.need_atom_confidence = True",
                "    runner.dumper.need_atom_confidence = True",
                "    runner.dumper.base_dir = out_dir",
                "    infer_predict(runner, runner.configs)",
            ]
        )
        steps.append(
                {
                    "name": f"protenix-predict-{safe_id}",
                    "command": [
                        "docker",
                        "run",
                        "--rm",
                        "--gpus",
                        "all",
                        "--shm-size=32G",
                        "-v",
                        f"{job.run_dir}:/work",
                        "-v",
                        f"{PROTENIX_REFERENCE_DIR}:/ref/pxdesign:ro",
                        "-v",
                        f"{PROTENIX_REFERENCE_DIR / 'release_data'}:/opt/conda/lib/python3.11/site-packages/release_data:ro",
                        "-e",
                        "PROTENIX_DATA_ROOT_DIR=/ref/pxdesign/release_data/ccd_cache",
                        "-e",
                        "TOOL_WEIGHTS_ROOT=/ref/pxdesign/tool_weights",
                        PROTENIX_IMAGE,
                        "python",
                        "-c",
                        protenix_code,
                    ],
                },
        )
    rc = _run_shell_steps(job.run_dir, steps)
    return _finish_external_complex_refolding(
        job_run_dir=job.run_dir,
        source_run_dir=source_run_dir,
        source_candidates=source_candidates,
        staged=staged,
        output_root=output_root,
        tool="protenix",
        rc=rc,
        extra_metrics={
            "initial_guess_supported": False,
            "target_msa_supported": True,
            "use_msa": use_msa,
            **msa_metrics,
        },
    )


def run_boltzgen_fold_complex_refolding(
    source_run_dir: Path,
    candidates_jsonl: Path,
    require_monomer_success: bool = True,
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 5,
    internal_parent_run_dir: Path | None = None,
) -> Path:
    """Run BoltzGen's template-conditioned folding stage on existing complexes."""
    source_run_dir = Path(source_run_dir)
    allowed = {STAGE_MONOMER_REFOLDING} if require_monomer_success else {
        STAGE_MONOMER_REFOLDING,
        STAGE_SEQUENCE_DESIGN,
        STAGE_GENERATION_BACKBONE_SEQUENCE,
        STAGE_COMPLEX_REFOLDING,
    }
    source_candidates = _source_candidates(Path(candidates_jsonl), allowed)
    job = create_job(
        REFOLDING_GROUP,
        "complex_refolding",
        "boltzgen_fold",
        {"source_run_dir": str(source_run_dir), "candidates_jsonl": str(candidates_jsonl)},
        {
            "image": BOLTZGEN_IMAGE,
            "mode": "target_template_folding",
            "target_templates": True,
            "use_msa": False,
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
        },
    )
    if internal_parent_run_dir is not None:
        mark_internal_job(
            job.run_dir,
            parent_run_dir=Path(internal_parent_run_dir),
            parent_task_group="benchmark",
            role="benchmark_engine_subrun",
            engine="boltzgen_fold",
        )
    raw_root = job.run_dir / "artifacts" / "raw" / "boltzgen_fold"
    input_dir = raw_root / "inputs"
    staged = _stage_complex_inputs(source_run_dir, source_candidates, input_dir)
    for safe_id, staged_path in staged.items():
        source = next(item for item in source_candidates if safe_id == _safe_id(item.get("candidate_id")))
        binder_chains, _target_chains = _infer_chain_roles(source_run_dir, source, staged_path)
        sequences = _pdb_sequences_by_chain(staged_path)
        design_mask = np.asarray(
            [chain in set(binder_chains) for chain, sequence in sequences.items() for _residue in sequence],
            dtype=bool,
        )
        np.savez(input_dir / f"{safe_id}.npz", design_mask=design_mask)
    output_root = raw_root / "output"
    output_root.mkdir(parents=True, exist_ok=True)
    hot_patch_mounts: list[str] = []
    if BOLTZGEN_LOCAL_SOURCE.exists():
        hot_patch_mounts.extend(["-v", f"{BOLTZGEN_LOCAL_SOURCE}:/app/src/boltzgen:ro"])
    boltzgen_config_path = "/app/config/fold.yaml"
    if BOLTZGEN_BENCHMARK_PAE_CONFIG.exists():
        boltzgen_config_path = "/app/config/fold_benchmark_pae.yaml"
        hot_patch_mounts.extend(["-v", f"{BOLTZGEN_BENCHMARK_PAE_CONFIG}:{boltzgen_config_path}:ro"])
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--shm-size=32G",
        "-v",
        f"{job.run_dir}:/work",
        *hot_patch_mounts,
        "-w",
        "/work",
        "--entrypoint",
        "python",
        BOLTZGEN_IMAGE,
        "/app/src/boltzgen/resources/main.py",
        boltzgen_config_path,
        "data.design_dir=/work/artifacts/raw/boltzgen_fold/inputs",
        "data.cfg.suffix=.pdb",
        "data.cfg.num_workers=1",
        "data.cfg.moldir=/cache/datasets--boltzgen--inference-data/snapshots/c3d36fd276e9caf098c75d4113c6d5eb320b1a4c/mols.zip",
        "output=/work/artifacts/raw/boltzgen_fold/output",
        "checkpoint=/cache/models--boltzgen--boltzgen-1/snapshots/c1be29e1f82ffcc72264f64b993c43fb4e0d17f0/boltz2_conf_final.ckpt",
        f"recycling_steps={int(recycling_steps)}",
        f"sampling_steps={int(sampling_steps)}",
        f"diffusion_samples={int(diffusion_samples)}",
        "trainer.devices=1",
    ]
    rc = _run_shell_steps(job.run_dir, [{"name": "boltzgen-target-template-fold", "command": command}])
    # BoltzGen writes folded files beside the staged designs.
    for safe_id in staged:
        candidate_output = output_root / safe_id
        candidate_output.mkdir(parents=True, exist_ok=True)
        for path in sorted((input_dir / "refold_cif").glob(f"{safe_id}*.cif")):
            shutil.copy2(path, candidate_output / path.name)
        for path in sorted((input_dir / "fold_out_npz").glob(f"{safe_id}*.npz")):
            shutil.copy2(path, candidate_output / path.name)
    return _finish_external_complex_refolding(
        job_run_dir=job.run_dir,
        source_run_dir=source_run_dir,
        source_candidates=source_candidates,
        staged=staged,
        output_root=output_root,
        tool="boltzgen_fold",
        rc=rc,
        extra_metrics={
            "initial_guess_supported": True,
            "initial_guess_mode": "target template from non-designed chains",
            "target_msa_supported": False,
        },
    )
