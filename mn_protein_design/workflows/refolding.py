from __future__ import annotations

import gzip
import csv
import json
import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from mn_protein_design.core.gpu import docker_gpu_args, normalize_gpu_device
from mn_protein_design.core.candidates import (
    STAGE_COMPLEX_REFOLDING,
    STAGE_GENERATION_BACKBONE_SEQUENCE,
    STAGE_MONOMER_REFOLDING,
    STAGE_SEQUENCE_DESIGN,
    read_candidates,
    write_candidates,
)
from mn_protein_design.core.portable_paths import resolve_stored_path
from mn_protein_design.core.jobs import JobPaths, create_job, finish_job, mark_internal_job, read_json, update_status, utc_now, write_json
from mn_protein_design.core.scheduler import apply_docker_cpu_limits_to_steps
from mn_protein_design.workflows import esm_binder as esm_binder_workflow
from mn_protein_design.workflows import chain_roles
from mn_protein_design.workflows import target_msa as target_msa_workflow


REFOLDING_GROUP = "refolding-validation"
BOLTZ_MODELS_DIR = Path("/mnt/db/reference_files/boltz_models")
ALPHAFOLD_MODELS_DIR = Path("/mnt/db/reference_files/alphafold_models")
AF2_BINDER_EVAL = Path(__file__).resolve().parents[1] / "tools" / "af2_initial_guess_binder_eval.py"
BOLTZ_PREPARE_INPUTS = Path("/home/user/programs/ovo/original/src/ovo/pipelines/boltz-refolding/bin/prepare_inputs.py")
RF3_IMAGE = "mn-foundry:cu128"
RF3_CHECKPOINT = Path("/mnt/db/reference_files/foundry/rf3_foundry_01_24_latest_remapped.ckpt")
PROTENIX_IMAGE = "mn-pxdesign:cu128"
PROTENIX_REFERENCE_DIR = Path("/mnt/db/reference_files/pxdesign")
PROTENIX_CLI_IMAGE = "mn-protenix:cu128"
PROTENIX_CLI_REFERENCE_DIR = Path("/mnt/db/reference_files/protenix")
PROTENIX_V1_MODEL = "protenix_base_default_v1.0.0"
PROTENIX_V1_20250630_MODEL = "protenix_base_20250630_v1.0.0"
PROTENIX_V2_MODEL = "protenix-v2"
BOLTZGEN_IMAGE = "mn-boltzgen:latest"
OPENFOLD3_IMAGE = "mn-openfold3:cu13"
OPENFOLD3_CHECKPOINT = Path("/mnt/db/reference_files/openfold3/of3-p2-155k.pt")
OPENFOLD3_REFERENCE_MSA_PREFILL = Path("/mnt/db/reference_files/de_novo_binder_scoring_overath_2025/target_msa_prefill")
SYNTHETIC_TEMPLATE_RELEASE_DATE = "1970-01-01"
OPENFOLD3_RUNNER_YAML_TEMPLATE = """model_update:
  presets:
    - predict
  custom:
    architecture:
      shared:
        num_recycles: {num_recycles}
    settings:
      memory:
        eval:
          use_deepspeed_evo_attention: false
"""
BOLTZGEN_LOCAL_SOURCE = Path(__file__).resolve().parents[2] / "tools_to_implement" / "boltzgen" / "src" / "boltzgen"
BOLTZGEN_BENCHMARK_PAE_CONFIG = Path(__file__).resolve().parents[2] / "tools_to_implement" / "boltzgen" / "config" / "fold_benchmark_pae.yaml"

MONOMER_SUFFIXES = (
    "_boltz2_monomer",
    "_esmfold2_monomer",
    "_esmfold_monomer",
    "_af2_monomer",
    "_monomer_boltz2",
    "_monomer_esmfold2",
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


def _attach_chain_role_warnings(
    candidate: dict[str, Any],
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    if not findings:
        return candidate
    warnings = chain_roles.problem_findings(findings)
    out = dict(candidate)
    raw_metadata = dict(out.get("raw_metadata") or {})
    existing = raw_metadata.get("chain_role_findings")
    if isinstance(existing, list):
        raw_metadata["chain_role_findings"] = [*existing, *findings]
    else:
        raw_metadata["chain_role_findings"] = findings
    if warnings:
        raw_metadata["chain_role_warnings"] = warnings
    out["raw_metadata"] = raw_metadata
    metrics = dict(out.get("metrics") or {})
    metrics["chain_role_finding_count"] = int(metrics.get("chain_role_finding_count") or 0) + len(findings)
    metrics["chain_role_warning_count"] = int(metrics.get("chain_role_warning_count") or 0) + len(warnings)
    out["metrics"] = metrics
    return out


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
        text = chains.strip()
        if not text:
            return default
        if text.startswith("["):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                values = [str(chain) for chain in parsed if str(chain)]
                if values:
                    return values
        values = [part.strip().strip("'\"") for part in text.replace(";", ",").split(",") if part.strip().strip("'\"")]
        if values:
            return values
    if isinstance(chains, list):
        values = [str(chain) for chain in chains if str(chain)]
        if values:
            return values
    return default


def _chunk_items(items: list[tuple[str, Path]], chunk_size: int) -> list[list[tuple[str, Path]]]:
    size = max(1, int(chunk_size))
    return [items[index : index + size] for index in range(0, len(items), size)]


def _is_target_refolding_input_candidate(candidate: dict[str, Any]) -> bool:
    raw = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
    if bool(raw.get("capacity_target_only")):
        return True
    if str(candidate.get("source_tool") or "") == "target_refolding_input_builder":
        return True
    if str(candidate.get("source") or "") == "target_refolding_input_builder":
        return True
    row = raw.get("repo_run_csv_row") if isinstance(raw.get("repo_run_csv_row"), dict) else {}
    return str(row.get("source") or "") == "target_refolding_input_builder"


def _target_only_candidate_chains(candidate: dict[str, Any], default: list[str] | None = None) -> list[str]:
    """Return target chains for explicit target-only rows, with legacy fallbacks."""
    raw = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
    for payload, keys in (
        (candidate, ("target_chains", "staged_target_chains", "biological_target_chains")),
        (raw, ("target_chains", "staged_target_chains", "biological_target_chains")),
        (candidate, ("legacy_capacity_binder_chains", "binder_chains")),
        (raw, ("legacy_capacity_binder_chains", "binder_chains")),
    ):
        if not isinstance(payload, dict):
            continue
        for key in keys:
            chains = _candidate_chains(payload, key, [])
            if chains:
                return chains
    return list(default or [])


def _is_split_fragment_target_candidate(candidate: dict[str, Any]) -> bool:
    raw = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
    row = raw.get("repo_run_csv_row") if isinstance(raw.get("repo_run_csv_row"), dict) else {}
    text_fields = [
        candidate.get("source_category"),
        candidate.get("prepared_kind"),
        candidate.get("target_alignment"),
        raw.get("source_category"),
        raw.get("prepared_kind"),
        raw.get("target_alignment"),
        row.get("source_category"),
        row.get("prepared_kind"),
        row.get("target_alignment"),
    ]
    if any("fragment" in str(value or "").lower() for value in text_fields):
        return True
    for payload in (candidate, raw, row):
        if not isinstance(payload, dict):
            continue
        for key in ("fragment_count", "fragments", "chain_break_count", "breaks"):
            try:
                if int(payload.get(key) or 0) > 1:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def _structure_chain_retention_metrics(path: Path | None, expected_chains: list[str]) -> dict[str, Any]:
    expected = [str(chain) for chain in expected_chains if str(chain)]
    if not path or not expected or not Path(path).exists():
        return {}
    observed = _structure_chains(Path(path))
    observed_set = set(observed)
    missing = [chain for chain in expected if chain not in observed_set]
    return {
        "expected_chain_count": len(expected),
        "observed_chain_count": len(observed),
        "missing_chain_count": len(missing),
        "missing_chains": ",".join(missing),
        "chain_retention_warning": (
            f"Prediction contains {len(observed)} chain(s), missing expected chain(s): {', '.join(missing)}"
            if missing
            else ""
        ),
    }


def _strip_monomer_suffix(candidate_id: object) -> str:
    text = str(candidate_id or "candidate")
    for suffix in MONOMER_SUFFIXES:
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _clean_sequence(value: object) -> str:
    return "".join(ch for ch in str(value or "").upper() if ch.isalpha())


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


def _copy_a3m_match_columns_only(source: Path, destination: Path) -> None:
    records: list[tuple[str, str]] = []
    header: str | None = None
    chunks: list[str] = []
    for raw_line in Path(source).read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks)))
            header = line
            chunks = []
        else:
            chunks.append("".join(ch for ch in line if ch == "-" or ch.isupper()))
    if header is not None:
        records.append((header, "".join(chunks)))
    query_length = len(records[0][1]) if records else 0
    lines: list[str] = []
    for header, sequence in records:
        if query_length:
            sequence = sequence[:query_length].ljust(query_length, "-")
        lines.extend([header, sequence])
    destination.write_text("\n".join(lines) + "\n")


def _msa_status(path: Path | None, *, expected_sequence: object = "", record_sequence: object = "") -> tuple[str, int]:
    if path is None or not Path(path).exists():
        return "missing", 0
    expected = _clean_sequence(expected_sequence)
    recorded = _clean_sequence(record_sequence)
    if expected and recorded and expected != recorded:
        return "sequence_mismatch", _a3m_sequence_count(path)
    sequence_count = _a3m_sequence_count(path)
    if sequence_count <= 0:
        return "empty", sequence_count
    if sequence_count == 1:
        return "query_only", sequence_count
    return "real_msa", sequence_count


def _resolve_msa_record(
    *,
    chain: str,
    sequence: object,
    msa_records: dict[str, dict[str, str]],
    used_msa_records: set[str],
) -> tuple[str, dict[str, str] | None, str]:
    if chain in msa_records and chain not in used_msa_records:
        return chain, msa_records[chain], "chain"
    clean_sequence = _clean_sequence(sequence)
    if clean_sequence:
        for candidate_chain, record in msa_records.items():
            if candidate_chain in used_msa_records:
                continue
            if _clean_sequence(record.get("sequence")) == clean_sequence:
                return candidate_chain, record, "sequence"
    return "", None, "none"


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
            capacity_target_only = str(row.get("target_source") or "").strip().lower() == "capacity_target_only" or str(
                row.get("capacity_target_only") or ""
            ).strip().lower() in {"1", "true", "yes", "y"}
            for key, value in row.items():
                if not key.startswith("msa_path_"):
                    continue
                chain = key.removeprefix("msa_path_")
                if not chain or (chain == binder_chain and not capacity_target_only):
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
    capacity_target_only = bool(raw.get("capacity_target_only")) or str(row.get("target_source") or "").strip().lower() == "capacity_target_only" or str(
        row.get("capacity_target_only") or ""
    ).strip().lower() in {"1", "true", "yes", "y"}
    chain_records: dict[str, dict[str, str]] = {}
    for key, value in row.items():
        if not str(key).startswith("msa_path_"):
            continue
        chain = str(key).removeprefix("msa_path_")
        msa_path = str(value or "").strip()
        if not chain or (chain == binder_chain and not capacity_target_only) or not msa_path or msa_path.lower() == "no_msa":
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
    use_target_template: bool = False,
) -> tuple[dict[str, Path], dict[str, int]]:
    benchmark_msa_records = _load_run_csv_msa_records(benchmark_run_csv)
    rf3_inputs: dict[str, Path] = {}
    metrics = {
        "rf3_target_msa_injected_count": 0,
        "rf3_target_msa_real_count": 0,
        "rf3_target_msa_query_only_count": 0,
        "rf3_target_msa_missing_count": 0,
        "rf3_target_msa_sequence_mismatch_count": 0,
        "rf3_binder_no_msa_count": 0,
        "rf3_query_only_msa_fallback_count": 0,
        "rf3_nonreal_msa_omitted_count": 0,
        "rf3_target_template_injected_count": 0,
        "rf3_target_template_missing_count": 0,
    }
    msa_dir = input_dir / "msas"
    template_dir = input_dir / "templates"
    job_run_dir = input_dir.parents[3]
    msa_dir.mkdir(parents=True, exist_ok=True)
    template_dir.mkdir(parents=True, exist_ok=True)
    manifest_records: list[dict[str, Any]] = []

    def stage_target_template_chain(staged_path: Path, safe_id: str, chain: str) -> Path | None:
        if staged_path.suffix.lower() not in {".pdb", ".ent"}:
            return None
        template_path = template_dir / f"{safe_id}_chain_{_safe_id(chain)}_template.pdb"
        wrote = False
        with staged_path.open() as src, template_path.open("w") as dst:
            for line in src:
                record = line[:6].strip()
                if record in {"ATOM", "HETATM", "ANISOU"}:
                    if len(line) > 21 and line[21].strip() == chain:
                        dst.write(line)
                        wrote = True
                elif record == "TER":
                    if len(line) > 21 and line[21].strip() == chain:
                        dst.write(line)
                elif record == "END":
                    continue
            if wrote:
                dst.write("END\n")
        if not wrote:
            template_path.unlink(missing_ok=True)
            return None
        return template_path

    for safe_id, staged_path in staged.items():
        source = next(item for item in source_candidates if safe_id == _safe_id(item.get("candidate_id")))
        candidate_id = str(source.get("candidate_id") or safe_id)
        raw = source.get("raw_metadata") if isinstance(source.get("raw_metadata"), dict) else {}
        capacity_target_only = bool(raw.get("capacity_target_only"))
        if capacity_target_only:
            binder_chains = []
            target_chains = _target_only_candidate_chains(source)
        else:
            binder_chains = _candidate_chains(source, "binder_chains", ["A"])
            target_chains = _candidate_chains(source, "target_chains", [])
        if not capacity_target_only and not target_chains:
            _inferred_binder, target_chains = _infer_chain_roles(source_run_dir, source)
        target_set = set(target_chains)
        sequences = _pdb_sequences_by_chain(staged_path)
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
        msa_paths: dict[str, str] = {}
        template_selection: list[str] = []
        target_template_paths: dict[str, Path] = {}
        if use_target_template:
            for chain in target_chains:
                template_path = stage_target_template_chain(staged_path, safe_id, chain)
                if template_path is None:
                    metrics["rf3_target_template_missing_count"] += 1
                    continue
                target_template_paths[chain] = template_path
                template_selection.append(chain)
                metrics["rf3_target_template_injected_count"] += 1
        used_msa_records: set[str] = set()
        for chain, sequence in sequences.items():
            role = "target" if chain in target_set else "binder"
            template_path = target_template_paths.get(chain) if role == "target" else None
            if template_path is not None:
                component = {"path": f"/work/{template_path.relative_to(job_run_dir)}"}
            else:
                component = {"seq": sequence, "chain_id": chain}
            manifest_row: dict[str, Any] = {
                "candidate_id": candidate_id,
                "safe_id": safe_id,
                "engine_chain": chain,
                "role": role,
                "sequence_length": len(_clean_sequence(sequence)),
                "template_status": "disabled" if not use_target_template else "no_template_expected",
                "template_path": None,
                "msa_status": "disabled" if not use_target_msa else "no_msa_expected",
                "source_record_chain": None,
                "source_match": None,
                "source_msa_path": None,
                "staged_msa_path": None,
                "msa_attachment": "disabled" if not use_target_msa else "none",
                "msa_sequence_count": 0,
            }
            if role == "binder":
                metrics["rf3_binder_no_msa_count"] += 1
            if template_path is not None:
                manifest_row["template_status"] = "attached"
                manifest_row["template_path"] = str(template_path)
            if use_target_msa and chain in target_set:
                record_chain, record, match_mode = _resolve_msa_record(
                    chain=chain,
                    sequence=sequence,
                    msa_records=msa_records,
                    used_msa_records=used_msa_records,
                )
                manifest_row["source_record_chain"] = record_chain or None
                manifest_row["source_match"] = match_mode
                if record is None:
                    repository_msa, repository_source = target_msa_workflow.find_cached_msa_for_sequence(sequence)
                    if repository_msa:
                        record = {"msa_path": str(repository_msa), "sequence": sequence}
                        record_chain = chain
                        match_mode = f"sequence_hash_{repository_source}"
                        manifest_row["source_record_chain"] = record_chain
                        manifest_row["source_match"] = match_mode
                    else:
                        manifest_row["msa_status"] = "missing"
                        metrics["rf3_target_msa_missing_count"] += 1
                if record is not None:
                    source_msa = Path(str(record.get("msa_path") or ""))
                    status, sequence_count = _msa_status(
                        source_msa,
                        expected_sequence=sequence,
                        record_sequence=record.get("sequence"),
                    )
                    if status != "real_msa":
                        repository_msa, repository_source = target_msa_workflow.find_cached_msa_for_sequence(sequence)
                        if repository_msa and Path(repository_msa) != source_msa:
                            repository_status, repository_sequence_count = _msa_status(
                                Path(repository_msa),
                                expected_sequence=sequence,
                                record_sequence=sequence,
                            )
                            if repository_status == "real_msa":
                                source_msa = Path(repository_msa)
                                status = repository_status
                                sequence_count = repository_sequence_count
                                manifest_row["source_record_chain"] = chain
                                manifest_row["source_match"] = f"sequence_hash_{repository_source}"
                    manifest_row["msa_status"] = status
                    manifest_row["source_msa_path"] = str(source_msa)
                    manifest_row["msa_sequence_count"] = sequence_count
                    if status == "sequence_mismatch":
                        metrics["rf3_target_msa_sequence_mismatch_count"] += 1
                    elif status in {"missing", "empty"}:
                        metrics["rf3_target_msa_missing_count"] += 1
                    elif status == "real_msa":
                        staged_msa = msa_dir / f"{safe_id}_chain_{_safe_id(chain)}{source_msa.suffix or '.a3m'}"
                        _copy_a3m_match_columns_only(source_msa, staged_msa)
                        msa_path = f"/work/artifacts/raw/rf3/inputs/msas/{staged_msa.name}"
                        if template_path is None:
                            component["msa_path"] = msa_path
                            manifest_row["msa_attachment"] = "component"
                        else:
                            manifest_row["msa_attachment"] = "top_level_msa_paths"
                        msa_paths[chain] = msa_path
                        manifest_row["staged_msa_path"] = str(staged_msa)
                        metrics["rf3_target_msa_injected_count"] += 1
                        metrics["rf3_target_msa_real_count"] += 1
                        used_msa_records.add(record_chain)
                    elif status == "query_only":
                        metrics["rf3_target_msa_query_only_count"] += 1
                        metrics["rf3_nonreal_msa_omitted_count"] += 1
                        manifest_row["msa_status"] = "query_only_omitted"
                    else:
                        metrics["rf3_nonreal_msa_omitted_count"] += 1
                if "msa_path" not in component and use_target_msa and chain not in msa_paths:
                    manifest_row["staged_msa_path"] = None
            manifest_records.append(manifest_row)
            components.append(component)
        input_path = input_dir / f"{safe_id}.json"
        payload: dict[str, Any] = {"name": safe_id, "components": components}
        if msa_paths:
            payload["msa_paths"] = msa_paths
        if template_selection:
            payload["template_selection"] = template_selection
        write_json(input_path, payload)
        rf3_inputs[safe_id] = input_path
    manifest_path = input_dir / "msa_manifest.json"
    write_json(
        manifest_path,
        {
            "engine": "rf3",
            "use_target_msa": bool(use_target_msa),
            "use_target_template": bool(use_target_template),
            "summary": metrics,
            "records": manifest_records,
        },
    )
    metrics["rf3_msa_manifest"] = str(manifest_path)
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
    use_target_template: bool = False,
) -> tuple[dict[str, Path], dict[str, int]]:
    benchmark_msa_records = _load_run_csv_msa_records(benchmark_run_csv)
    protenix_inputs: dict[str, Path] = {}
    metrics = {
        "protenix_target_msa_injected_count": 0,
        "protenix_target_msa_real_count": 0,
        "protenix_target_msa_query_only_count": 0,
        "protenix_target_msa_missing_count": 0,
        "protenix_target_msa_sequence_mismatch_count": 0,
        "protenix_binder_query_only_msa_count": 0,
        "protenix_target_query_only_msa_fallback_count": 0,
        "protenix_target_template_injected_count": 0,
        "protenix_target_template_missing_count": 0,
        "protenix_target_template_sequence_mismatch_count": 0,
    }
    job_run_dir = json_root.parents[3]
    template_root = json_root.parent / "templates"
    manifest_records: list[dict[str, Any]] = []

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

    def should_attach_precomputed_msa(role: str, status: str, sequence: str) -> bool:
        if role != "target":
            return status in {"real_msa", "query_only"}
        # Protenix is stricter than the other engines about old-format A3M
        # directories. Short/query-only fragment placeholders can trigger
        # "Inconsistent row lengths in A3M"; let Protenix handle those chains
        # without a staged precomputed MSA instead.
        if len(_clean_sequence(sequence)) < 30:
            return False
        return status == "real_msa"

    for safe_id, staged_path in staged.items():
        source = next(item for item in source_candidates if safe_id == _safe_id(item.get("candidate_id")))
        candidate_id = str(source.get("candidate_id") or safe_id)
        raw = source.get("raw_metadata") if isinstance(source.get("raw_metadata"), dict) else {}
        capacity_target_only = bool(raw.get("capacity_target_only"))
        if capacity_target_only:
            binder_chains = []
            target_chains = _target_only_candidate_chains(source)
        else:
            binder_chains = _candidate_chains(source, "binder_chains", ["A"])
            target_chains = _candidate_chains(source, "target_chains", [])
        if not capacity_target_only and not target_chains:
            _inferred_binder, target_chains = _infer_chain_roles(source_run_dir, source)
        target_set = set(target_chains)
        sequences = _pdb_sequences_by_chain(staged_path)
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
        target_order = [chain for chain in target_chains if chain in sequences]
        # Keep split/fragment targets as separate Protenix chains. Collapsing them
        # into one query sequence can make downstream chain-role normalization and
        # target-template RMSD scoring treat only the first fragment as preserved.
        collapse_fragment_template = False
        chain_specs: list[dict[str, Any]] = []
        if collapse_fragment_template:
            combined_sequence = "".join(_clean_sequence(sequences[chain]) for chain in target_order)
            if combined_sequence:
                chain_specs.append(
                    {
                        "chain": target_order[0],
                        "sequence": combined_sequence,
                        "role": "target",
                        "template_chains": target_order,
                        "template_mode": "collapsed_fragments",
                    }
                )
            for chain, sequence in sequences.items():
                if chain not in target_set:
                    chain_specs.append(
                        {
                            "chain": chain,
                            "sequence": sequence,
                            "role": "binder",
                            "template_chains": [chain],
                            "template_mode": "per_chain",
                        }
                    )
        else:
            for chain, sequence in sequences.items():
                chain_specs.append(
                    {
                        "chain": chain,
                        "sequence": sequence,
                        "role": "target" if chain in target_set else "binder",
                        "template_chains": [chain],
                        "template_mode": "per_chain",
                    }
                )
        if collapse_fragment_template and chain_specs:
            metrics["protenix_target_template_collapsed_fragment_count"] = (
                int(metrics.get("protenix_target_template_collapsed_fragment_count") or 0) + 1
            )
        for spec in chain_specs:
            chain = str(spec.get("chain") or "A")
            sequence = str(spec.get("sequence") or "")
            role = str(spec.get("role") or ("target" if chain in target_set else "binder"))
            template_chains = [str(item) for item in spec.get("template_chains") or [chain] if str(item)]
            template_mode = str(spec.get("template_mode") or "per_chain")
            protein: dict[str, Any] = {"sequence": sequence, "count": 1}
            chain_dir = msa_root / safe_id / f"chain_{_safe_id(chain)}"
            manifest_row: dict[str, Any] = {
                "candidate_id": candidate_id,
                "safe_id": safe_id,
                "engine_chain": chain,
                "role": role,
                "sequence_length": len(_clean_sequence(sequence)),
                "source_template_chains": template_chains,
                "template_chain_mode": template_mode,
                "msa_status": "disabled" if not use_target_msa else "no_msa_expected",
                "source_record_chain": None,
                "source_match": None,
                "source_msa_path": None,
                "staged_msa_path": None,
                "msa_sequence_count": 0,
                "template_status": "disabled" if not use_target_template else "no_template_expected",
                "template_path": None,
                "template_match": None,
                "template_aligned_residues": 0,
            }
            if use_target_template and role == "target":
                template_label = "fragment_chains" if template_mode == "collapsed_fragments" else f"chain_{_safe_id(chain)}"
                template_path = template_root / safe_id / f"{template_label}_template.json"
                template_info = None
                if staged_path.suffix.lower() == ".pdb":
                    template_info = _pdb_chains_to_protenix_template_json(
                        pdb_path=staged_path,
                        chains=template_chains,
                        query_sequence=sequence,
                        output_path=template_path,
                        entry_id=f"{_safe_id(safe_id)}_{_safe_id(template_label)}",
                    )
                if template_info is None:
                    manifest_row["template_status"] = "missing"
                    metrics["protenix_target_template_missing_count"] += 1
                else:
                    protein["templatesPath"] = f"/work/{template_path.relative_to(job_run_dir)}"
                    manifest_row["template_status"] = "attached"
                    manifest_row["template_path"] = str(template_path)
                    manifest_row["template_match"] = template_info.get("match_mode")
                    manifest_row["template_aligned_residues"] = int(template_info.get("aligned_residues") or 0)
                    manifest_row["template_sequence_length"] = int(template_info.get("template_sequence_length") or 0)
                    metrics["protenix_target_template_injected_count"] += 1
                    if template_info.get("match_mode") != "exact":
                        metrics["protenix_target_template_sequence_mismatch_count"] += 1
            if use_target_msa and role == "target" and template_mode == "collapsed_fragments":
                manifest_row["msa_status"] = "disabled_for_collapsed_fragment_template"
            elif use_target_msa and chain in target_set:
                record_chain, record, match_mode = _resolve_msa_record(
                    chain=chain,
                    sequence=sequence,
                    msa_records=msa_records,
                    used_msa_records=used_msa_records,
                )
                manifest_row["source_record_chain"] = record_chain or None
                manifest_row["source_match"] = match_mode
                if record is None:
                    manifest_row["msa_status"] = "missing"
                    metrics["protenix_target_msa_missing_count"] += 1
                else:
                    source_msa = Path(str(record.get("msa_path") or ""))
                    status, sequence_count = _msa_status(
                        source_msa,
                        expected_sequence=sequence,
                        record_sequence=record.get("sequence"),
                    )
                    manifest_row["msa_status"] = status
                    manifest_row["source_msa_path"] = str(source_msa)
                    manifest_row["msa_sequence_count"] = sequence_count
                    if status == "sequence_mismatch":
                        metrics["protenix_target_msa_sequence_mismatch_count"] += 1
                    elif status in {"real_msa", "query_only"} and should_attach_precomputed_msa(role, status, sequence):
                        chain_dir.mkdir(parents=True, exist_ok=True)
                        for filename in ["pairing.a3m", "non_pairing.a3m"]:
                            _copy_a3m_match_columns_only(source_msa, chain_dir / filename)
                        attach_msa(protein, chain_dir)
                        manifest_row["staged_msa_path"] = str(chain_dir)
                        metrics["protenix_target_msa_injected_count"] += 1
                        if status == "real_msa":
                            metrics["protenix_target_msa_real_count"] += 1
                        elif status == "query_only":
                            metrics["protenix_target_msa_query_only_count"] += 1
                        used_msa_records.add(record_chain)
                    elif status in {"real_msa", "query_only"}:
                        manifest_row["msa_status"] = f"{status}_not_attached"
                        metrics["protenix_target_msa_missing_count"] += 1
                    else:
                        metrics["protenix_target_msa_missing_count"] += 1
            if use_target_msa and "msa" not in protein and role != "target":
                write_query_only_msa(chain_dir, safe_id, chain, sequence)
                attach_msa(protein, chain_dir)
                manifest_row["staged_msa_path"] = str(chain_dir)
                manifest_row["msa_sequence_count"] = 1
                manifest_row["msa_status"] = "query_only_placeholder"
                metrics["protenix_binder_query_only_msa_count"] += 1
            elif use_target_msa and "msa" not in protein and role == "target":
                if manifest_row["msa_status"] in {"missing", "no_msa_expected"}:
                    manifest_row["msa_status"] = "not_attached"
            manifest_records.append(manifest_row)
            sequence_entries.append({"proteinChain": protein})
        candidate_json_dir = json_root / safe_id
        candidate_json_dir.mkdir(parents=True, exist_ok=True)
        input_path = candidate_json_dir / f"{safe_id}.json"
        write_json(input_path, [{"sequences": sequence_entries, "name": safe_id}])
        protenix_inputs[safe_id] = candidate_json_dir
    manifest_path = msa_root / "msa_manifest.json"
    write_json(
        manifest_path,
        {
            "engine": "protenix",
            "use_target_msa": bool(use_target_msa),
            "use_target_template": bool(use_target_template),
            "summary": metrics,
            "records": manifest_records,
        },
    )
    metrics["protenix_msa_manifest"] = str(manifest_path)
    return protenix_inputs, metrics


def _monomer_candidate_id(source: dict[str, Any], tool: str) -> str:
    label = {
        "boltz2_monomer": "monomer_boltz2",
        "esmfold2_monomer": "monomer_esmfold2",
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
    return resolve_stored_path(path_text, run_dir=source_run_dir)


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


def _pdb_sequence_residues_by_chain(path: Path) -> dict[str, list[tuple[int, str, str]]]:
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
        chain: sorted(chain_residues, key=lambda item: (item[0], item[1]))
        for chain, chain_residues in residues.items()
    }


def _mmcif_quote(value: object) -> str:
    text = str(value)
    if not text:
        return "?"
    if any(char.isspace() for char in text) or text.startswith(("#", "_", ";")):
        return "'" + text.replace("'", "''") + "'"
    return text


def _pdb_chain_to_protenix_template_json(
    *,
    pdb_path: Path,
    chain: str,
    query_sequence: str,
    output_path: Path,
    entry_id: str,
) -> dict[str, Any] | None:
    return _pdb_chains_to_protenix_template_json(
        pdb_path=pdb_path,
        chains=[chain],
        query_sequence=query_sequence,
        output_path=output_path,
        entry_id=entry_id,
    )


def _pdb_chains_to_protenix_template_json(
    *,
    pdb_path: Path,
    chains: list[str],
    query_sequence: str,
    output_path: Path,
    entry_id: str,
) -> dict[str, Any] | None:
    """Write a Protenix JSON template for one staged target chain.

    Protenix accepts templatesPath values that point to a JSON list of
    templates. Each item embeds one mmCIF string plus the 0-based query and
    template residue indices. Normal refolding writes one Protenix chain per
    target fragment so coverage/RMSD scoring remains fragment-aware.
    """
    keep_chains = set(chains)
    residue_order: list[tuple[str, str, str]] = []
    residue_seen: set[tuple[str, str, str]] = set()
    atom_rows: list[dict[str, str]] = []
    atom_id = 1
    for line in pdb_path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        line_chain = line[21].strip() or "_"
        if keep_chains and line_chain not in keep_chains:
            continue
        residue_key = (line[22:26].strip(), line[26].strip())
        resname = line[17:20].strip().upper()
        full_residue_key = (line_chain, residue_key[0], residue_key[1])
        if full_residue_key not in residue_seen:
            residue_seen.add(full_residue_key)
            residue_order.append((residue_key[0], residue_key[1], resname))
        label_seq_id = len(residue_order)
        atom_name = line[12:16].strip()
        element = (line[76:78].strip() or atom_name.lstrip("0123456789")[:1] or "C").upper()
        atom_rows.append(
            {
                "group": "ATOM",
                "id": str(atom_id),
                "type_symbol": element,
                "atom_id": atom_name,
                "comp_id": resname,
                "label_seq_id": str(label_seq_id),
                "auth_seq_id": residue_key[0] or str(label_seq_id),
                "x": line[30:38].strip() or "0.0",
                "y": line[38:46].strip() or "0.0",
                "z": line[46:54].strip() or "0.0",
                "occupancy": line[54:60].strip() or "1.00",
                "b_iso": line[60:66].strip() or "0.00",
                "auth_atom_id": atom_name,
            }
        )
        atom_id += 1
    if not residue_order or not atom_rows:
        return None

    template_sequence = "".join(AA3_TO_1.get(resname, "X") for _resseq, _icode, resname in residue_order)
    clean_query = _clean_sequence(query_sequence)
    if clean_query == template_sequence:
        query_indices = list(range(len(clean_query)))
        template_indices = list(range(len(template_sequence)))
        match_mode = "exact"
    elif template_sequence and template_sequence in clean_query:
        offset = clean_query.index(template_sequence)
        query_indices = list(range(offset, offset + len(template_sequence)))
        template_indices = list(range(len(template_sequence)))
        match_mode = "template_in_query"
    elif clean_query and clean_query in template_sequence:
        offset = template_sequence.index(clean_query)
        query_indices = list(range(len(clean_query)))
        template_indices = list(range(offset, offset + len(clean_query)))
        match_mode = "query_in_template"
    else:
        n = min(len(clean_query), len(template_sequence))
        query_indices = list(range(n))
        template_indices = list(range(n))
        match_mode = "prefix_fallback"

    cif_lines = [
        f"data_{entry_id}",
        "#",
        f"_entry.id {entry_id}",
        "#",
        "loop_",
        "_pdbx_audit_revision_history.ordinal",
        "_pdbx_audit_revision_history.data_content_type",
        "_pdbx_audit_revision_history.major_revision",
        "_pdbx_audit_revision_history.minor_revision",
        "_pdbx_audit_revision_history.revision_date",
        f"1 {_mmcif_quote('Structure model')} 1 0 {SYNTHETIC_TEMPLATE_RELEASE_DATE}",
        "#",
        "loop_",
        "_entity_poly_seq.entity_id",
        "_entity_poly_seq.num",
        "_entity_poly_seq.mon_id",
        "_entity_poly_seq.hetero",
    ]
    for idx, (_resseq, _icode, resname) in enumerate(residue_order, start=1):
        cif_lines.append(f"1 {idx} {_mmcif_quote(resname)} n")
    cif_lines.extend(
        [
            "#",
            "loop_",
            "_struct_asym.id",
            "_struct_asym.entity_id",
            "A 1",
            "#",
            "loop_",
            "_atom_site.group_PDB",
            "_atom_site.id",
            "_atom_site.type_symbol",
            "_atom_site.label_atom_id",
            "_atom_site.label_alt_id",
            "_atom_site.label_comp_id",
            "_atom_site.label_asym_id",
            "_atom_site.label_entity_id",
            "_atom_site.label_seq_id",
            "_atom_site.pdbx_PDB_ins_code",
            "_atom_site.Cartn_x",
            "_atom_site.Cartn_y",
            "_atom_site.Cartn_z",
            "_atom_site.occupancy",
            "_atom_site.B_iso_or_equiv",
            "_atom_site.pdbx_formal_charge",
            "_atom_site.auth_seq_id",
            "_atom_site.auth_comp_id",
            "_atom_site.auth_asym_id",
            "_atom_site.auth_atom_id",
            "_atom_site.pdbx_PDB_model_num",
        ]
    )
    for row in atom_rows:
        cif_lines.append(
            " ".join(
                [
                    row["group"],
                    row["id"],
                    _mmcif_quote(row["type_symbol"]),
                    _mmcif_quote(row["atom_id"]),
                    ".",
                    _mmcif_quote(row["comp_id"]),
                    "A",
                    "1",
                    row["label_seq_id"],
                    "?",
                    row["x"],
                    row["y"],
                    row["z"],
                    row["occupancy"],
                    row["b_iso"],
                    "?",
                    row["auth_seq_id"],
                    _mmcif_quote(row["comp_id"]),
                    "A",
                    _mmcif_quote(row["auth_atom_id"]),
                    "1",
                ]
            )
        )
    cif_lines.append("#")
    template_payload = [
        {
            "mmcif": "\n".join(cif_lines) + "\n",
            "queryIndices": query_indices,
            "templateIndices": template_indices,
        }
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, template_payload)
    return {
        "template_sequence_length": len(template_sequence),
        "query_sequence_length": len(clean_query),
        "aligned_residues": len(query_indices),
        "match_mode": match_mode,
    }


def _pdb_chain_ca_segments(path: Path, chain: str, distance_threshold: float = 4.5) -> list[tuple[int, int]]:
    residues: list[tuple[int, tuple[float, float, float]]] = []
    seen: set[tuple[str, str]] = set()
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
            continue
        line_chain = line[21].strip() or "_"
        if line_chain != chain:
            continue
        key = (line[22:26], line[26])
        if key in seen:
            continue
        seen.add(key)
        try:
            residue_number = int(line[22:26])
            coord = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        residues.append((residue_number, coord))
    residues = sorted(residues, key=lambda item: item[0])
    if not residues:
        return []
    segments: list[tuple[int, int]] = []
    start = previous = residues[0][0]
    previous_coord = residues[0][1]
    for residue, coord in residues[1:]:
        ca_distance = (
            (coord[0] - previous_coord[0]) ** 2
            + (coord[1] - previous_coord[1]) ** 2
            + (coord[2] - previous_coord[2]) ** 2
        ) ** 0.5
        if residue == previous + 1 and ca_distance <= distance_threshold:
            previous = residue
            previous_coord = coord
            continue
        segments.append((start, previous))
        start = previous = residue
        previous_coord = coord
    segments.append((start, previous))
    return segments


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
    source_residue_range: tuple[int, int] | None = None,
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
        try:
            original_resseq = int(line[22:26])
        except ValueError:
            continue
        if source_residue_range is not None:
            start_residue, end_residue = source_residue_range
            if original_resseq < start_residue or original_resseq > end_residue:
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
    source_residue_range: tuple[int, int] | None = None,
) -> tuple[list[str], int]:
    if path.suffix.lower() == ".cif" or path.name.endswith(".cif.gz"):
        return _renumber_cif_chain(path, chain_id, start_atom, source_chains)
    return _renumber_pdb_chain(path, chain_id, start_atom, source_chains, source_residue_range)


def _target_output_chain_ids(target_chains: list[str]) -> list[str]:
    return chain_roles.assign_target_engine_chains(target_chains)


def _declared_target_subchain_sequences(candidate: dict[str, Any], chains: list[str]) -> dict[str, str]:
    raw = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
    row = raw.get("repo_run_csv_row") if isinstance(raw.get("repo_run_csv_row"), dict) else {}
    sequences: dict[str, str] = {}
    for chain in chains:
        sequence = _clean_sequence(row.get(f"target_subchain_{chain}_seq"))
        if sequence:
            sequences[str(chain)] = sequence
    return sequences


def _target_fragment_specs_from_declared_sequences(
    target_path: Path,
    declared_sequences: dict[str, str],
    declared_chains: list[str],
) -> list[dict[str, Any]]:
    if target_path.suffix.lower() != ".pdb" or not declared_sequences:
        return []
    residues_by_chain = _pdb_sequence_residues_by_chain(target_path)
    chain_sequences = {
        chain: "".join(residue[2] for residue in residues)
        for chain, residues in residues_by_chain.items()
    }
    cursors = {chain: 0 for chain in chain_sequences}
    engine_chains = chain_roles.assign_target_engine_chains(declared_chains)
    specs: list[dict[str, Any]] = []
    for declared_chain, engine_chain in zip(declared_chains, engine_chains):
        fragment_sequence = declared_sequences.get(str(declared_chain), "")
        if not fragment_sequence:
            return []
        match: tuple[str, int] | None = None
        for source_chain, source_sequence in chain_sequences.items():
            start = source_sequence.find(fragment_sequence, cursors.get(source_chain, 0))
            if start >= 0:
                match = (source_chain, start)
                break
        if match is None:
            for source_chain, source_sequence in chain_sequences.items():
                start = source_sequence.find(fragment_sequence)
                if start >= 0:
                    match = (source_chain, start)
                    break
        if match is None:
            return []
        source_chain, start_index = match
        end_index = start_index + len(fragment_sequence) - 1
        residues = residues_by_chain[source_chain]
        if end_index >= len(residues):
            return []
        cursors[source_chain] = end_index + 1
        specs.append(
            {
                "source_chain": source_chain,
                "declared_chain": str(declared_chain),
                "engine_chain": str(engine_chain),
                "start": int(residues[start_index][0]),
                "end": int(residues[end_index][0]),
                "is_fragment": True,
            }
        )
    return specs


def _target_fragment_specs(target_path: Path, source_chains: list[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    target_segments: list[tuple[str, int, int, bool]] = []
    for source_chain in source_chains:
        segments = (
            _pdb_chain_ca_segments(target_path, source_chain)
            if target_path.suffix.lower() == ".pdb"
            else []
        )
        if not segments:
            segments = [(0, 0)]
        for start, end in segments:
            target_segments.append((source_chain, start, end, bool(start and end)))
    engine_ids = chain_roles.assign_target_engine_chains([source for source, _start, _end, _fragment in target_segments])
    for (source_chain, start, end, is_fragment), engine_chain in zip(target_segments, engine_ids):
        specs.append(
            {
                "source_chain": source_chain,
                "engine_chain": engine_chain,
                "start": start,
                "end": end,
                "is_fragment": is_fragment,
            }
        )
    return specs


def _append_role_chains(
    path: Path,
    *,
    source_chains: list[str],
    engine_chains: list[str],
    next_atom: int,
) -> tuple[list[str], int]:
    lines: list[str] = []
    for source_chain, engine_chain in zip(source_chains, engine_chains):
        chain_lines, next_atom = _renumber_structure_chain(
            path,
            engine_chain,
            next_atom,
            {source_chain},
        )
        if chain_lines:
            lines.extend(chain_lines + ["TER"])
    return lines, next_atom


def _write_engine_chain_map(
    input_dir: Path,
    safe_id: str,
    *,
    binder_source_chains: list[str],
    target_source_chains: list[str],
    target_engine_chains: list[str],
    binder_engine_chains: list[str] | None = None,
    target_fragments: list[dict[str, Any]] | None = None,
) -> None:
    binder_engine_chains = binder_engine_chains or chain_roles.assign_binder_engine_chains(
        binder_source_chains,
        reserved=target_engine_chains,
    )
    role_map = chain_roles.build_explicit_role_map(
        binder_source_chains=binder_source_chains,
        binder_engine_chains=binder_engine_chains,
        target_source_chains=target_source_chains,
        target_engine_chains=target_engine_chains,
        target_fragments=target_fragments,
    )
    payload = {
        "candidate_id": safe_id,
        "chain_role_schema": role_map.schema,
        "chain_roles": role_map.to_dict(),
        "binder": role_map.chain_map["binder"],
        "targets": role_map.chain_map["targets"],
    }
    if target_fragments:
        payload["target_fragments"] = target_fragments
    write_json(input_dir / f"{safe_id}.chain_map.json", payload)


def normalize_candidate_structure_roles(
    *,
    structure_path: Path,
    output_dir: Path,
    safe_id: str,
    binder_source_chains: list[str],
    target_source_chains: list[str],
    target_only: bool = False,
) -> tuple[Path | None, list[str], list[str], list[dict[str, Any]]]:
    """Write an app-normalized PDB while preserving model residue/chain order.

    Engines may emit native layouts such as binder A / target B. This boundary
    helper rewrites chain identifiers to the app contract without reordering
    chain blocks, so confidence arrays that follow structure residue order remain
    aligned with the normalized structure.
    """
    structure_path = Path(structure_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    available_chains = _structure_chains(structure_path)
    expected_target_source_chains = list(target_source_chains)
    if target_only:
        binder_source_chains = []
        target_source_chains = [chain for chain in target_source_chains if chain in available_chains] or available_chains
    else:
        binder_source_chains = [chain for chain in binder_source_chains if chain in available_chains]
        target_source_chains = [
            chain
            for chain in target_source_chains
            if chain in available_chains and chain not in set(binder_source_chains)
        ]

    target_engine_chains = chain_roles.assign_target_engine_chains(target_source_chains)
    binder_engine_chains = (
        []
        if target_only
        else chain_roles.assign_binder_engine_chains(
            binder_source_chains,
            reserved=target_engine_chains,
        )
    )
    findings = chain_roles.validate_chain_roles(
        candidate_id=safe_id,
        binder_chains=binder_engine_chains,
        target_chains=chain_roles.assign_target_engine_chains(expected_target_source_chains or target_source_chains),
        structure_chains=[*binder_engine_chains, *target_engine_chains],
        schema=chain_roles.CHAIN_ROLE_SCHEMA_EXPLICIT_V2,
        target_only=target_only,
    )
    if (not target_only and (not binder_engine_chains or not target_engine_chains)) or (target_only and not target_engine_chains):
        return None, binder_engine_chains, target_engine_chains, findings

    source_to_engine = {
        **dict(zip(target_source_chains, target_engine_chains)),
        **dict(zip(binder_source_chains, binder_engine_chains)),
    }
    lines: list[str] = []
    next_atom = 1
    for source_chain in available_chains:
        engine_chain = source_to_engine.get(source_chain)
        if not engine_chain:
            continue
        chain_lines, next_atom = _renumber_structure_chain(
            structure_path,
            engine_chain,
            next_atom,
            {source_chain},
        )
        if chain_lines:
            lines.extend(chain_lines + ["TER"])
    if not lines:
        return None, binder_engine_chains, target_engine_chains, findings

    normalized_path = output_dir / f"{safe_id}_role_normalized.pdb"
    normalized_path.write_text("\n".join(lines + ["END", ""]))
    _write_engine_chain_map(
        output_dir,
        f"{safe_id}_role_normalized",
        binder_source_chains=binder_source_chains,
        binder_engine_chains=binder_engine_chains,
        target_source_chains=target_source_chains,
        target_engine_chains=target_engine_chains,
    )
    write_json(
        normalized_path.with_suffix(".role_normalization.json"),
        {
            "source_structure": str(structure_path),
            "normalized_structure": str(normalized_path),
            "chain_role_schema": chain_roles.CHAIN_ROLE_SCHEMA_EXPLICIT_V2,
            "binder_source_chains": binder_source_chains,
            "binder_chains": binder_engine_chains,
            "target_source_chains": target_source_chains,
            "target_chains": target_engine_chains,
            "target_only": bool(target_only),
            "findings": findings,
        },
    )
    return normalized_path, binder_engine_chains, target_engine_chains, findings


def _engine_chains_from_map(input_dir: Path, safe_id: str, section: str) -> list[str]:
    payload = read_json(input_dir / f"{safe_id}.chain_map.json")
    rows = payload.get(section)
    chains: list[str] = []
    if isinstance(rows, list):
        candidate_rows = list(rows)
    else:
        candidate_rows = []
    if section == "targets":
        fragments = payload.get("target_fragments")
        if isinstance(fragments, list):
            candidate_rows.extend(fragments)
    for row in candidate_rows:
        if not isinstance(row, dict):
            continue
        chain = str(row.get("engine_chain") or "").strip()
        if chain and chain not in chains:
            chains.append(chain)
    return chains


def _read_staged_chain_sidecar(input_dir: Path, safe_id: str, suffix: str) -> list[str]:
    path = input_dir / f"{safe_id}.{suffix}"
    if not path.exists():
        return []
    return [
        part.strip()[:1]
        for part in path.read_text(encoding="utf-8", errors="replace").replace("\n", ",").split(",")
        if part.strip()
    ]


def _af2_designed_chains_arg(
    input_dir: Path,
    staged: dict[str, Path],
    source_candidates: list[dict[str, Any]],
    *,
    capacity_target_only_run: bool,
) -> str:
    if capacity_target_only_run and staged:
        first_staged = next(iter(staged.values()))
        staged_chains = _structure_chains(first_staged)
        if staged_chains:
            return ",".join(staged_chains)

    source_by_safe_id = {
        _safe_id(candidate.get("candidate_id")): candidate
        for candidate in source_candidates
    }
    selected: list[str] = []
    for safe_id, staged_path in staged.items():
        available = set(_structure_chains(staged_path))
        chains = [
            chain
            for chain in _read_staged_chain_sidecar(input_dir, safe_id, "binder_source_chains.txt")
            if not available or chain in available
        ]
        if not chains:
            chains = [chain for chain in _engine_chains_from_map(input_dir, safe_id, "binder") if not available or chain in available]
        if not chains:
            source = source_by_safe_id.get(safe_id) or {}
            chains = [chain for chain in _candidate_chains(source, "binder_chains", []) if not available or chain in available]
        for chain in chains:
            if chain and chain not in selected:
                selected.append(chain)
    return ",".join(selected or ["A"])


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
    raw = candidate.get("raw_metadata") or {}
    capacity_target_only = _is_target_refolding_input_candidate(candidate)

    def append_target_fragments(
        target_path: Path,
        source_chains: list[str],
        next_atom: int,
        fragment_specs: list[dict[str, Any]] | None = None,
    ) -> tuple[list[str], int, list[dict[str, Any]]]:
        if fragment_specs is None:
            fragment_specs = _target_fragment_specs(target_path, source_chains)
        target_lines: list[str] = []
        for spec in fragment_specs:
            residue_range = (
                (int(spec["start"]), int(spec["end"]))
                if spec.get("is_fragment")
                else None
            )
            chain_lines, next_atom = _renumber_structure_chain(
                target_path,
                str(spec["engine_chain"]),
                next_atom,
                {str(spec["source_chain"])},
                residue_range,
            )
            if chain_lines:
                target_lines.extend(chain_lines + ["TER"])
        return target_lines, next_atom, fragment_specs

    def target_fragment_plan(
        target_path: Path,
        fallback_chains: list[str],
    ) -> tuple[list[str], list[dict[str, Any]]]:
        target_structure_chains = _structure_chains(target_path)
        requested_targets = _candidate_chains(candidate, "target_chains", target_structure_chains)
        declared_sequences = _declared_target_subchain_sequences(candidate, requested_targets)
        declared_target_chains = [chain for chain in requested_targets if chain in declared_sequences]
        if declared_target_chains:
            declared_specs = _target_fragment_specs_from_declared_sequences(
                target_path,
                declared_sequences,
                declared_target_chains,
            )
            if declared_specs:
                return declared_target_chains, declared_specs
        target_source_chains = [chain for chain in requested_targets if chain in target_structure_chains] or fallback_chains
        return target_source_chains, _target_fragment_specs(target_path, target_source_chains)

    def normalized_complex_from(path: Path, suffix: str = "") -> Path | None:
        safe_id = _safe_id(candidate.get("candidate_id"))
        output = input_dir / f"{safe_id}{suffix}.pdb"
        inferred_binder_chains, inferred_target_chains = _infer_chain_roles(source_run_dir, candidate, path)
        target_path = _target_pdb_for_candidate(source_run_dir, candidate)
        if target_path and target_path.exists():
            target_source_chains, fragment_specs = target_fragment_plan(
                target_path,
                _structure_chains(target_path),
            )
        else:
            target_source_chains = inferred_target_chains
            fragment_specs = [
                {"source_chain": source, "engine_chain": engine, "start": 0, "end": 0, "is_fragment": False}
                for source, engine in zip(inferred_target_chains, _target_output_chain_ids(inferred_target_chains))
            ]
        target_engine_chains = [str(spec["engine_chain"]) for spec in fragment_specs]
        binder_engine_chains = chain_roles.assign_binder_engine_chains(
            inferred_binder_chains,
            reserved=target_engine_chains,
        )
        binder_lines, next_atom = _append_role_chains(
            path,
            source_chains=inferred_binder_chains,
            engine_chains=binder_engine_chains,
            next_atom=1,
        )
        if target_path and target_path.exists():
            target_lines, next_atom, fragment_specs = append_target_fragments(
                target_path,
                target_source_chains,
                next_atom,
                fragment_specs,
            )
        else:
            target_lines, next_atom = _append_role_chains(
                path,
                source_chains=[str(spec["source_chain"]) for spec in fragment_specs],
                engine_chains=target_engine_chains,
                next_atom=next_atom,
            )
        if not binder_lines or not target_lines:
            return None
        output.write_text("\n".join(binder_lines + target_lines + ["END", ""]))
        _write_engine_chain_map(
            input_dir,
            safe_id,
            binder_source_chains=inferred_binder_chains,
            target_source_chains=[str(spec["source_chain"]) for spec in fragment_specs],
            target_engine_chains=target_engine_chains,
            binder_engine_chains=binder_engine_chains,
            target_fragments=fragment_specs,
        )
        return output

    if capacity_target_only and source_complex and source_complex.exists():
        safe_id = _safe_id(candidate.get("candidate_id"))
        output = input_dir / f"{safe_id}.pdb"
        source_chains = _target_only_candidate_chains(candidate)
        available_chains = _structure_chains(source_complex)
        source_chains = [chain for chain in source_chains if chain in available_chains] or available_chains
        next_atom = 1
        target_lines, next_atom, fragment_specs = append_target_fragments(source_complex, source_chains, next_atom)
        if not target_lines:
            raise ValueError(f"Candidate {candidate.get('candidate_id')} does not contain usable target chains.")
        output.write_text("\n".join(target_lines + ["END", ""]))
        _write_engine_chain_map(
            input_dir,
            safe_id,
            binder_source_chains=[],
            target_source_chains=[str(spec["source_chain"]) for spec in fragment_specs],
            target_engine_chains=[str(spec["engine_chain"]) for spec in fragment_specs],
            target_fragments=fragment_specs,
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
        if capacity_target_only:
            safe_id = _safe_id(candidate.get("candidate_id"))
            output = input_dir / f"{safe_id}.pdb"
            source_chain = _structure_chains(source_complex)[0]
            chain_lines, _next_atom = _renumber_structure_chain(source_complex, "A", 1, {source_chain})
            if not chain_lines:
                raise ValueError(f"Candidate {candidate.get('candidate_id')} does not contain a usable target-only chain.")
            output.write_text("\n".join(chain_lines + ["TER", "END", ""]))
            _write_engine_chain_map(
                input_dir,
                safe_id,
                binder_source_chains=[],
                target_source_chains=["A"],
                target_engine_chains=["A"],
            )
            return output
        binder = source_complex

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
    target_source_chains, fragment_specs = target_fragment_plan(
        target,
        _structure_chains(target),
    )
    target_engine_chains = [str(spec["engine_chain"]) for spec in fragment_specs]
    binder_engine_chains = chain_roles.assign_binder_engine_chains(
        binder_chains,
        reserved=target_engine_chains,
    )
    binder_lines, next_atom = _append_role_chains(
        binder,
        source_chains=binder_chains if len(_structure_chains(binder)) > 1 else [_structure_chains(binder)[0]],
        engine_chains=binder_engine_chains,
        next_atom=1,
    )
    target_lines, next_atom, fragment_specs = append_target_fragments(
        target,
        target_source_chains,
        next_atom,
        fragment_specs,
    )
    output.write_text("\n".join(binder_lines + target_lines + ["END", ""]))
    _write_engine_chain_map(
        input_dir,
        safe_id,
        binder_source_chains=binder_chains,
        target_source_chains=[str(spec["source_chain"]) for spec in fragment_specs],
        target_engine_chains=[str(spec["engine_chain"]) for spec in fragment_specs],
        binder_engine_chains=binder_engine_chains,
        target_fragments=fragment_specs,
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
        available_chains = _structure_chains(staged_path)
        capacity_target_only = _is_target_refolding_input_candidate(candidate)
        if capacity_target_only:
            binder_chains = []
            target_chains = list(available_chains)
        else:
            mapped_binders = _engine_chains_from_map(input_dir, safe_id, "binder")
            mapped_targets = _engine_chains_from_map(input_dir, safe_id, "targets")
            binder_chains = [chain for chain in mapped_binders if chain in available_chains]
            if not binder_chains:
                binder_chains = [chain for chain in _candidate_chains(candidate, "binder_chains", ["A"]) if chain in available_chains] or ["A"]
            target_chains = [chain for chain in mapped_targets if chain in available_chains]
            if not target_chains:
                target_chains = [chain for chain in _candidate_chains(candidate, "target_chains", []) if chain in available_chains]
        if not target_chains:
            _inferred_binder, inferred_target = _infer_chain_roles(source_run_dir, candidate, staged_path)
            target_chains = [chain for chain in inferred_target if chain in available_chains]
            binder_chains = [chain for chain in _inferred_binder if chain in available_chains] or binder_chains
        chain_map_payload = read_json(input_dir / f"{safe_id}.chain_map.json")
        findings = chain_roles.validate_chain_roles(
            candidate_id=candidate.get("candidate_id") or safe_id,
            binder_chains=binder_chains,
            target_chains=target_chains,
            structure_chains=available_chains,
            schema=chain_map_payload.get("chain_role_schema"),
            target_only=capacity_target_only,
        )
        if findings:
            write_json(input_dir / f"{safe_id}.chain_role_findings.json", findings)
            warnings = chain_roles.problem_findings(findings)
            if warnings:
                write_json(input_dir / f"{safe_id}.chain_role_warnings.json", warnings)
        try:
            binder_sequence = (
                _candidate_target_refolding_sequence(source_run_dir, candidate, target_chains, staged_path)
                if capacity_target_only
                else _candidate_binder_sequence(source_run_dir, candidate)
            )
        except ValueError:
            binder_sequence = ""
        if binder_sequence:
            sidecar = "target_sequence.txt" if capacity_target_only else "binder_sequence.txt"
            (input_dir / f"{safe_id}.{sidecar}").write_text(f"{binder_sequence}\n", encoding="utf-8")
        (input_dir / f"{safe_id}.target_chains.txt").write_text(f"{','.join(target_chains)}\n", encoding="utf-8")
        if capacity_target_only:
            (input_dir / f"{safe_id}.target_source_chains.txt").write_text(f"{','.join(target_chains)}\n", encoding="utf-8")
        else:
            (input_dir / f"{safe_id}.binder_source_chains.txt").write_text(f"{','.join(binder_chains)}\n", encoding="utf-8")
        staged[safe_id] = staged_path
    return staged


def _target_only_staged_fragment_count(input_dir: Path, safe_id: str, staged_path: Path) -> int:
    chain_map = read_json(input_dir / f"{safe_id}.chain_map.json")
    fragments = chain_map.get("target_fragments")
    if isinstance(fragments, list) and fragments:
        engine_chains = {
            str(row.get("engine_chain") or "")
            for row in fragments
            if isinstance(row, dict) and str(row.get("engine_chain") or "")
        }
        return len(engine_chains) or len(fragments)
    return len(_structure_chains(staged_path))


def _write_openfold3_query_inputs(
    *,
    job_run_dir: Path,
    source_run_dir: Path,
    source_candidates: list[dict[str, Any]],
    staged: dict[str, Path],
    query_root: Path,
    benchmark_run_csv: Path | None,
    use_target_msa: bool,
) -> tuple[dict[str, Path], dict[str, int]]:
    query_root.mkdir(parents=True, exist_ok=True)
    msa_root = query_root.parent / "msas"
    msa_root.mkdir(parents=True, exist_ok=True)
    run_msa_records = _load_run_csv_msa_records(benchmark_run_csv)
    reference_msa_records = _load_reference_target_msa_records()
    metrics = {
        "openfold3_msa_attached_count": 0,
        "openfold3_reference_msa_attached_count": 0,
        "openfold3_msa_missing_count": 0,
        "openfold3_msa_sequence_mismatch_count": 0,
        "openfold3_short_target_msa_skipped_count": 0,
        "openfold3_msa_alignment_rows_retained_count": 0,
        "openfold3_msa_alignment_rows_dropped_count": 0,
    }
    outputs: dict[str, Path] = {}
    for source in source_candidates:
        safe_id = _safe_id(source.get("candidate_id"))
        staged_path = staged[safe_id]
        binder_chains, target_chains = _infer_chain_roles(source_run_dir, source, staged_path)
        sequences = _pdb_sequences_by_chain(staged_path)
        raw = source.get("raw_metadata") if isinstance(source.get("raw_metadata"), dict) else {}
        repo_row = raw.get("repo_run_csv_row") if isinstance(raw.get("repo_run_csv_row"), dict) else {}
        declared_targets = _candidate_chains(source, "target_chains", [])
        declared_target_sequences: dict[str, str] = {}
        for chain_id in declared_targets:
            for column in (f"target_subchain_{chain_id}_seq", f"{chain_id}_seq"):
                value = repo_row.get(column)
                if str(value or "").strip().lower() in {"", "nan", "none"}:
                    continue
                sequence = _clean_sequence(value)
                if sequence:
                    declared_target_sequences[chain_id] = sequence
                    break
        if declared_target_sequences:
            target_chains = [chain for chain in declared_targets if chain in declared_target_sequences]
            sequences.update(declared_target_sequences)
            _write_engine_chain_map(
                staged_path.parent,
                safe_id,
                binder_source_chains=[] if _is_target_refolding_input_candidate(source) else binder_chains,
                target_source_chains=target_chains,
                target_engine_chains=target_chains,
            )
        capacity_target_only = bool(raw.get("capacity_target_only"))
        if capacity_target_only:
            binder_chains = []
            target_chains = _target_only_candidate_chains(source, target_chains) or target_chains
        target_set = set(target_chains)
        chains: list[dict[str, Any]] = []
        used_msa_records: set[str] = set()
        has_msa = False
        msa_records = run_msa_records.get(str(source.get("candidate_id") or "")) or run_msa_records.get(safe_id) or {}
        query_chains = list(target_chains) if capacity_target_only else [*binder_chains, *target_chains]
        for chain_id in query_chains:
            sequence = _clean_sequence(sequences.get(chain_id) or "")
            if not sequence:
                continue
            entry: dict[str, Any] = {
                "molecule_type": "protein",
                "chain_ids": chain_id,
                "sequence": sequence,
            }
            is_target = chain_id in target_set
            if use_target_msa and is_target:
                record_chain, record, _source_kind = _resolve_msa_record(
                    chain=chain_id,
                    sequence=sequence,
                    msa_records=msa_records,
                    used_msa_records=used_msa_records,
                )
                if record is None:
                    clean_sequence = _clean_sequence(sequence)
                    record = reference_msa_records.get(clean_sequence)
                    record_chain = "reference_sequence" if record else record_chain
                msa_path = Path(str(record.get("msa_path"))) if record else None
                status, _count = _msa_status(
                    msa_path,
                    expected_sequence=sequence,
                    record_sequence=record.get("sequence") if record else "",
                )
                if status == "query_only" and len(sequence) < 30:
                    metrics["openfold3_short_target_msa_skipped_count"] += 1
                elif record and msa_path and status in {"real_msa", "query_only"}:
                    staged_msa = msa_root / safe_id / f"chain_{_safe_id(chain_id)}" / "colabfold_main.a3m"
                    staged_msa.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(msa_path, staged_msa)
                    retained, dropped = _sanitize_openfold3_msa(staged_msa, expected_sequence=sequence)
                    metrics["openfold3_msa_alignment_rows_retained_count"] += retained
                    metrics["openfold3_msa_alignment_rows_dropped_count"] += dropped
                    entry["main_msa_file_paths"] = f"/work/{staged_msa.parent.relative_to(job_run_dir)}"
                    has_msa = True
                    if record_chain:
                        used_msa_records.add(record_chain)
                    metrics["openfold3_msa_attached_count"] += 1
                    if record_chain == "reference_sequence":
                        metrics["openfold3_reference_msa_attached_count"] += 1
                elif status == "sequence_mismatch":
                    metrics["openfold3_msa_sequence_mismatch_count"] += 1
                else:
                    metrics["openfold3_msa_missing_count"] += 1
            chains.append(entry)
        if not chains:
            raise ValueError(f"Candidate {source.get('candidate_id')} has no protein chains for OpenFold-3 input.")
        query_path = query_root / f"{safe_id}.json"
        query_path.write_text(
            json.dumps(
                {
                    "queries": {
                        safe_id: {
                            "chains": chains,
                            "use_msas": bool(has_msa),
                            "use_main_msas": bool(has_msa),
                            "use_paired_msas": False,
                        }
                    }
                },
                indent=2,
            )
        )
        outputs[safe_id] = query_path
    return outputs, metrics


def _sanitize_openfold3_msa(msa_path: Path, *, expected_sequence: str) -> tuple[int, int]:
    """Remove malformed A3M rows that OpenFold cannot align to the target sequence."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence_lines: list[str] = []
    for raw_line in msa_path.read_text().splitlines():
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
    if not records:
        raise ValueError(f"OpenFold-3 MSA is empty: {msa_path}")

    expected_length = len(_clean_sequence(expected_sequence))
    retained: list[tuple[str, str]] = []
    dropped = 0
    for index, (record_header, sequence) in enumerate(records):
        # OpenFold strips lowercase A3M insertions before constructing its MSA array.
        aligned_length = sum(1 for residue in sequence if not residue.islower())
        if aligned_length == expected_length:
            retained.append((record_header, sequence))
        elif index == 0:
            raise ValueError(
                f"OpenFold-3 MSA query length {aligned_length} does not match target length {expected_length}: {msa_path}"
            )
        else:
            dropped += 1
    if not retained:
        raise ValueError(f"OpenFold-3 MSA has no usable aligned rows: {msa_path}")
    msa_path.write_text("".join(f"{record_header}\n{sequence}\n" for record_header, sequence in retained))
    return len(retained), dropped


def _load_reference_target_msa_records(
    prefill_root: Path = OPENFOLD3_REFERENCE_MSA_PREFILL,
) -> dict[str, dict[str, str]]:
    manifest_path = prefill_root / "manifest.csv"
    report_path = prefill_root / "repository_write_report.csv"
    if not manifest_path.exists() or not report_path.exists():
        return {}
    repo_paths: dict[str, str] = {}
    try:
        with report_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("status") or "").lower() != "ok":
                    continue
                sha = str(row.get("sha256") or "").strip()
                repo_path = str(row.get("repo_path") or "").strip()
                if sha and repo_path:
                    repo_paths[sha] = repo_path
    except OSError:
        return {}
    records: dict[str, dict[str, str]] = {}
    try:
        with manifest_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                sequence = _clean_sequence(row.get("sequence"))
                sha = str(row.get("sha256") or "").strip()
                msa_path = repo_paths.get(sha)
                if sequence and msa_path and Path(msa_path).exists():
                    records[sequence] = {
                        "msa_path": msa_path,
                        "sequence": sequence,
                        "source": str(row.get("name") or sha),
                    }
    except OSError:
        return {}
    return records


def _write_openfold3_runner_yaml(raw_root: Path, *, num_recycles: int = 3) -> Path:
    runner_path = raw_root / "runner.yaml"
    runner_path.parent.mkdir(parents=True, exist_ok=True)
    runner_path.write_text(OPENFOLD3_RUNNER_YAML_TEMPLATE.format(num_recycles=max(1, int(num_recycles))))
    return runner_path


def _stage_target_template_inputs(
    source_run_dir: Path,
    source_candidates: list[dict[str, Any]],
    input_dir: Path,
) -> dict[str, Path]:
    """Stage target coordinates and binder sequences without binder coordinates."""
    input_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = input_dir / "references"
    reference_dir.mkdir(parents=True, exist_ok=True)
    staged: dict[str, Path] = {}
    for candidate in source_candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        safe_id = _safe_id(candidate_id)
        raw = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
        capacity_target_only = _is_target_refolding_input_candidate(candidate)
        target_path = (
            _resolve_candidate_path(source_run_dir, candidate.get("complex_pdb"))
            if capacity_target_only
            else None
        )
        if target_path is None or not target_path.exists():
            target_path = _target_pdb_for_candidate(source_run_dir, candidate)
        if target_path is None or not target_path.exists():
            raise ValueError(f"Candidate {candidate_id} does not have a readable target PDB.")

        available_chains = _structure_chains(target_path)
        if capacity_target_only:
            requested_chains = _target_only_candidate_chains(candidate)
            if not requested_chains:
                requested_chains = available_chains
        else:
            requested_chains = _candidate_chains(candidate, "target_chains", available_chains)
        declared_sequences = _declared_target_subchain_sequences(candidate, requested_chains)
        declared_chains = [chain for chain in requested_chains if chain in declared_sequences]
        declared_fragment_specs = (
            _target_fragment_specs_from_declared_sequences(
                target_path,
                declared_sequences,
                declared_chains,
            )
            if declared_chains
            else []
        )
        if declared_fragment_specs:
            source_chains = declared_chains
            fragment_specs = declared_fragment_specs
        else:
            source_chains = [chain for chain in requested_chains if chain in available_chains] or available_chains
            fragment_specs = _target_fragment_specs(target_path, source_chains)
        if not source_chains:
            raise ValueError(f"Candidate {candidate_id} target PDB has no readable protein chains.")

        if capacity_target_only and not any(spec.get("is_fragment") for spec in fragment_specs):
            for spec, source_chain in zip(fragment_specs, source_chains):
                spec["engine_chain"] = source_chain
        engine_chains = [str(spec["engine_chain"]) for spec in fragment_specs]
        target_lines: list[str] = []
        next_atom = 1
        for spec in fragment_specs:
            residue_range = (
                (int(spec["start"]), int(spec["end"]))
                if spec.get("is_fragment")
                else None
            )
            chain_lines, next_atom = _renumber_structure_chain(
                target_path,
                str(spec["engine_chain"]),
                next_atom,
                {str(spec["source_chain"])},
                residue_range,
            )
            if chain_lines:
                target_lines.extend(chain_lines + ["TER"])
        if not target_lines:
            raise ValueError(
                f"Candidate {candidate_id} target chains could not be staged."
            )

        staged_path = input_dir / f"{safe_id}.pdb"
        staged_path.write_text("\n".join(target_lines + ["END", ""]))
        design_sequence = (
            _candidate_target_refolding_sequence(source_run_dir, candidate, source_chains, target_path)
            if capacity_target_only
            else _candidate_binder_sequence(source_run_dir, candidate)
        )
        sequence_sidecar = "target_sequence.txt" if capacity_target_only else "binder_sequence.txt"
        (input_dir / f"{safe_id}.{sequence_sidecar}").write_text(f"{design_sequence}\n", encoding="utf-8")
        (input_dir / f"{safe_id}.target_chains.txt").write_text(
            f"{','.join(engine_chains)}\n",
            encoding="utf-8",
        )
        binder_source_chains = [] if capacity_target_only else _candidate_chains(candidate, "binder_chains", ["A"])
        reference_complex = _resolve_candidate_path(
            source_run_dir,
            candidate.get("complex_pdb") or candidate.get("binder_pdb"),
        )
        if reference_complex and reference_complex.exists() and reference_complex.suffix.lower() == ".pdb":
            shutil.copy2(reference_complex, reference_dir / f"{safe_id}.pdb")
            (input_dir / f"{safe_id}.target_source_chains.txt").write_text(
                f"{','.join(source_chains)}\n",
                encoding="utf-8",
            )
        _write_engine_chain_map(
            input_dir,
            safe_id,
            binder_source_chains=binder_source_chains,
            target_source_chains=[str(spec["source_chain"]) for spec in fragment_specs],
            target_engine_chains=engine_chains,
            target_fragments=fragment_specs,
        )
        chain_map_payload = read_json(input_dir / f"{safe_id}.chain_map.json")
        findings = chain_roles.validate_chain_roles(
            candidate_id=candidate_id or safe_id,
            binder_chains=[] if capacity_target_only else _engine_chains_from_map(input_dir, safe_id, "binder"),
            target_chains=engine_chains,
            structure_chains=_structure_chains(staged_path),
            schema=chain_map_payload.get("chain_role_schema"),
            target_only=capacity_target_only,
        )
        if findings:
            write_json(input_dir / f"{safe_id}.chain_role_findings.json", findings)
            warnings = chain_roles.problem_findings(findings)
            if warnings:
                write_json(input_dir / f"{safe_id}.chain_role_warnings.json", warnings)
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


def _candidate_target_refolding_sequence(
    source_run_dir: Path,
    candidate: dict[str, Any],
    chains: list[str],
    target_path: Path | None = None,
) -> str:
    raw = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
    row = raw.get("repo_run_csv_row") if isinstance(raw.get("repo_run_csv_row"), dict) else {}
    parts: list[str] = []
    for chain in chains:
        sequence = _clean_sequence(
            row.get(f"target_subchain_{chain}_seq")
            or row.get(f"{chain}_seq")
            or (row.get("A_seq") if chain == "A" else "")
        )
        if sequence:
            parts.append(sequence)
    if parts and len(parts) == len(chains):
        return "".join(parts)
    if target_path is None:
        target_path = _target_pdb_for_candidate(source_run_dir, candidate)
    if target_path is not None and target_path.exists():
        sequences = _sequences_by_chain(target_path)
        parts = [sequences.get(chain, "") for chain in chains]
        if all(parts):
            return "".join(parts)
    sequence = _clean_sequence(candidate.get("binder_sequence"))
    if sequence:
        return sequence
    raise ValueError(f"Candidate {candidate.get('candidate_id')} has no readable target-refolding sequence.")


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


def _run_shell_steps(run_dir: Path, steps: list[dict[str, Any]], verify_step: Any | None = None) -> int:
    steps = apply_docker_cpu_limits_to_steps(run_dir, steps)
    write_json(run_dir / "command.json", {"mode": "docker", "steps": steps})
    update_status(run_dir, "running")
    timing_path = run_dir / "artifacts" / "runtime_step_timings.json"
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    timings: list[dict[str, Any]] = []
    with (run_dir / "stdout.log").open("w") as stdout, (run_dir / "stderr.log").open("w") as stderr:
        for step_index, step in enumerate(steps, start=1):
            started_at = utc_now()
            started_perf = time.perf_counter()
            update_status(
                run_dir,
                "running",
                current_phase="Running prediction step",
                progress_label=str(step.get("name") or f"step {step_index}"),
                current_step_name=str(step.get("name") or ""),
                current_step_index=int(step_index),
                current_step_total=int(len(steps)),
                current_candidate_ids=[str(value) for value in step.get("candidate_ids") or []],
                current_target_ids=[str(value) for value in step.get("target_ids") or []],
            )
            stdout.write(f"$ {' '.join(step['command'])}\n")
            stdout.flush()
            proc = subprocess.run(step["command"], stdout=stdout, stderr=stderr, check=False)
            elapsed_seconds = max(0.0, time.perf_counter() - started_perf)
            timing_row = {
                "step_index": int(step_index),
                "step_total": int(len(steps)),
                "step_name": str(step.get("name") or ""),
                "candidate_ids": [str(value) for value in step.get("candidate_ids") or []],
                "candidate_count": len(step.get("candidate_ids") or []),
                "target_ids": [str(value) for value in step.get("target_ids") or []],
                "total_residues": int(step.get("total_residues") or 0) or None,
                "started_at": started_at,
                "completed_at": utc_now(),
                "seconds": elapsed_seconds,
                "return_code": int(proc.returncode),
            }
            timings.append(timing_row)
            write_json(
                timing_path,
                {
                    "step_count": len(steps),
                    "completed_step_count": len(timings),
                    "timings": timings,
                },
            )
            update_status(
                run_dir,
                "running",
                current_step_seconds=elapsed_seconds,
                completed_step_count=len(timings),
            )
            if proc.returncode != 0:
                return int(proc.returncode)
            if verify_step is not None:
                verify_error = verify_step(step, step_index)
                if verify_error:
                    stderr.write(str(verify_error) + "\n")
                    stderr.flush()
                    return 1
    return 0


def _safe_positive_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        parsed = int(float(str(value)))
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _candidate_residue_count(candidate: dict[str, Any], staged_path: Path | None = None) -> int | None:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    raw_metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
    row = raw_metadata.get("repo_run_csv_row") if isinstance(raw_metadata.get("repo_run_csv_row"), dict) else {}
    for key in ("total_residues", "total_length", "capacity_total_length", "sequence_length"):
        value = _safe_positive_int(metrics.get(key) if key in metrics else candidate.get(key))
        if value:
            return value
    lengths = [
        _safe_positive_int(metrics.get("binder_length") or candidate.get("binder_length") or row.get("A_length")),
        _safe_positive_int(metrics.get("target_length") or candidate.get("target_length") or row.get("B_length")),
    ]
    total = sum(length for length in lengths if length)
    if total:
        return total
    if row:
        row_total = 0
        for key, value in row.items():
            key_text = str(key)
            if key_text.endswith("_length") or key_text.endswith("_len"):
                row_total += int(_safe_positive_int(value) or 0)
        if row_total:
            return row_total
    if staged_path is not None and staged_path.exists():
        sequences = _pdb_sequences_by_chain(staged_path)
        pdb_total = sum(len(sequence) for sequence in sequences.values())
        if pdb_total:
            return int(pdb_total)
    sequence = str(candidate.get("binder_sequence") or "")
    return len(sequence) if sequence else None


def _candidate_residue_counts(
    candidates: list[dict[str, Any]],
    staged: dict[str, Path] | None = None,
) -> dict[str, int]:
    staged = staged or {}
    counts: dict[str, int] = {}
    for candidate in candidates:
        safe_id = _safe_id(candidate.get("candidate_id"))
        count = _candidate_residue_count(candidate, staged.get(safe_id))
        if count:
            counts[safe_id] = int(count)
    return counts


def _annotate_step_residue_totals(steps: list[dict[str, Any]], residue_counts: dict[str, int]) -> None:
    if not residue_counts:
        return
    lower_counts = {key.lower(): value for key, value in residue_counts.items()}
    for step in steps:
        candidate_ids = [str(value) for value in step.get("candidate_ids") or [] if str(value)]
        if not candidate_ids:
            continue
        total = 0
        for candidate_id in candidate_ids:
            total += int(residue_counts.get(candidate_id) or lower_counts.get(candidate_id.lower()) or 0)
        if total:
            step["total_residues"] = int(total)


def _candidate_target_ids(candidates: list[dict[str, Any]]) -> dict[str, str]:
    targets: dict[str, str] = {}
    for candidate in candidates:
        metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
        raw_metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
        raw_row = raw_metadata.get("repo_run_csv_row") if isinstance(raw_metadata.get("repo_run_csv_row"), dict) else {}
        target_id = (
            candidate.get("target_id")
            or metrics.get("target_id")
            or raw_row.get("target_id")
        )
        safe_id = _safe_id(candidate.get("candidate_id"))
        if safe_id and target_id not in (None, ""):
            targets[safe_id] = str(target_id)
    return targets


def _annotate_step_target_ids(steps: list[dict[str, Any]], target_ids: dict[str, str]) -> None:
    if not target_ids:
        return
    lower_targets = {key.lower(): value for key, value in target_ids.items()}
    for step in steps:
        values: list[str] = []
        for candidate_id in [str(value) for value in step.get("candidate_ids") or [] if str(value)]:
            target_id = target_ids.get(candidate_id) or lower_targets.get(candidate_id.lower())
            if target_id and target_id not in values:
                values.append(str(target_id))
        if values:
            step["target_ids"] = values


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
    gpu_device: object = "0",
    existing_job: JobPaths | None = None,
) -> Path:
    if tool == "esmfold2_monomer":
        return run_esmfold2_monomer_refolding(
            source_run_dir,
            candidates_jsonl,
            min_plddt=min_plddt,
            gpu_device=gpu_device,
            existing_job=existing_job,
        )
    if tool == "esmfold":
        return run_esmfold_monomer_refolding(
            source_run_dir,
            candidates_jsonl,
            min_plddt=min_plddt,
            gpu_device=gpu_device,
            existing_job=existing_job,
        )
    if tool == "boltz2_monomer":
        return run_boltz2_monomer_refolding(
            source_run_dir,
            candidates_jsonl,
            min_plddt=min_plddt,
            gpu_device=gpu_device,
            existing_job=existing_job,
        )

    source_run_dir = Path(source_run_dir)
    candidates_jsonl = Path(candidates_jsonl)
    source_candidates = _source_candidates(
        candidates_jsonl,
        {STAGE_SEQUENCE_DESIGN, STAGE_GENERATION_BACKBONE_SEQUENCE, STAGE_COMPLEX_REFOLDING},
    )
    job = existing_job or create_job(
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


def run_esmfold2_monomer_refolding(
    source_run_dir: Path,
    candidates_jsonl: Path,
    min_plddt: float = 70.0,
    num_loops: int = 3,
    num_sampling_steps: int = 32,
    seed: int = 0,
    device: str = "auto",
    gpu_device: object = "0",
    existing_job: JobPaths | None = None,
) -> Path:
    source_run_dir = Path(source_run_dir)
    candidates_jsonl = Path(candidates_jsonl)
    source_candidates = _source_candidates(
        candidates_jsonl,
        {STAGE_SEQUENCE_DESIGN, STAGE_GENERATION_BACKBONE_SEQUENCE, STAGE_COMPLEX_REFOLDING},
    )
    job = existing_job or create_job(
        REFOLDING_GROUP,
        job_type="monomer_refolding",
        tool="esmfold2_monomer",
        inputs={"source_run_dir": str(source_run_dir), "candidates_jsonl": str(candidates_jsonl)},
        params={
            "min_plddt": min_plddt,
            "num_loops": num_loops,
            "num_sampling_steps": num_sampling_steps,
            "seed": seed,
            "device": device,
            "gpu_device": normalize_gpu_device(gpu_device),
            "backend": "biohub_esmfold2_local",
            "model_dir": str(esm_binder_workflow.ESMFOLD2_MODEL_DIR),
            "esmc_model_dir": str(esm_binder_workflow.ESMC_MODEL_DIR),
        },
    )
    write_json(
        job.run_dir / "command.json",
        {
            "mode": "python",
            "tool": "esmfold2_monomer",
            "backend": "biohub_esmfold2_local",
            "source_run_dir": str(source_run_dir),
            "candidates_jsonl": str(candidates_jsonl),
            "device": device,
            "gpu_device": normalize_gpu_device(gpu_device),
            "num_loops": num_loops,
            "num_sampling_steps": num_sampling_steps,
            "seed": seed,
        },
    )
    update_status(job.run_dir, "running")
    raw_dir = job.run_dir / "artifacts" / "raw" / "esmfold2_monomer"
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        esm_binder_workflow._ensure_esm_import_path()
        try:
            from esm.models.esmfold2 import ESMFold2InputBuilder, ProteinInput, StructurePredictionInput
        except Exception as exc:  # pragma: no cover - dependency/runtime specific
            raise RuntimeError(
                "ESMFold2 runtime dependencies are not importable. "
                "Run this from the Biohub ESM environment/container before starting monomer refolding."
            ) from exc

        model = esm_binder_workflow._load_esmfold2_model(device)
        builder = ESMFold2InputBuilder(ccd_cache=esm_binder_workflow.ESMFOLD2_MODEL_DIR)
        candidates: list[dict[str, Any]] = []
        input_rows: list[dict[str, Any]] = []
        with (job.run_dir / "stdout.log").open("a") as stdout:
            stdout.write("ESMFold2 binder-only monomer refolding.\n")
            stdout.flush()
            for index, source in enumerate(source_candidates, start=1):
                parent_id = str(source.get("candidate_id") or f"candidate_{index:05d}")
                candidate_id = _monomer_candidate_id(source, "esmfold2_monomer")
                binder_sequence = _candidate_sequence(source_run_dir, source)
                if not binder_sequence:
                    raise ValueError(f"Candidate {parent_id} does not have a readable binder sequence.")
                stdout.write(
                    f"Folding {candidate_id} binder_length={len(binder_sequence)} "
                    f"seed={int(seed) + index - 1}\n"
                )
                stdout.flush()
                spi = StructurePredictionInput(
                    sequences=[ProteinInput(id="A", sequence=binder_sequence)],
                    distogram_conditioning=None,
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
                metrics = dict(source.get("metrics") or {})
                plddt_mean = esm_binder_workflow._mean_plddt(result)
                metrics.update(
                    {
                        "monomer_refolding_backend": "esmfold2_monomer",
                        "min_plddt": min_plddt,
                        "iptm": float(result.iptm) if result.iptm is not None else None,
                        "ptm": float(result.ptm) if result.ptm is not None else None,
                        "plddt_mean": plddt_mean,
                        "plddt": plddt_mean,
                        "esmfold2_monomer_iptm": float(result.iptm) if result.iptm is not None else None,
                        "esmfold2_monomer_ptm": float(result.ptm) if result.ptm is not None else None,
                        "esmfold2_monomer_plddt_mean": plddt_mean,
                        "passes_monomer_plddt": plddt_mean >= float(min_plddt) if plddt_mean is not None else None,
                        "binder_length": len(binder_sequence),
                    }
                )
                reference_backbone = _resolve_candidate_path(source_run_dir, source.get("complex_pdb"))
                binder_chain = _candidate_chains(source, "binder_chains", ["A"])[0]
                metrics.update(_monomer_refolding_rmsd(reference_backbone, complex_path, binder_chain=binder_chain))
                candidates.append(
                    {
                        **source,
                        "candidate_id": candidate_id,
                        "stage": STAGE_MONOMER_REFOLDING,
                        "source_tool": "esmfold2_monomer",
                        "tool": "esmfold2_monomer",
                        "binder_pdb": None,
                        "complex_pdb": _rel_path(job.run_dir, complex_path),
                        "binder_sequence": binder_sequence,
                        "binder_chains": ["A"],
                        "target_chains": [],
                        "binder_length": str(len(binder_sequence)),
                        "metrics": metrics,
                        "parents": [parent_id],
                        "raw_metadata": {
                            **dict(source.get("raw_metadata") or {}),
                            "source_candidate": source,
                            "backend_status": "biohub_esmfold2_local",
                            "prediction_dir": _rel_path(job.run_dir, raw_dir),
                            "upstream_source_run_dir": str(source_run_dir),
                            "input_complex": source.get("complex_pdb"),
                        },
                    }
                )
                input_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "parent_id": parent_id,
                        "binder_length": len(binder_sequence),
                        "source_complex": source.get("complex_pdb"),
                    }
                )

        (raw_dir / "binder_sequences.fasta").write_text(
            "".join(f">{candidate['candidate_id']}\n{candidate['binder_sequence']}\n" for candidate in candidates)
        )
        write_json(raw_dir / "input_mapping.json", input_rows)
        normalized = write_candidates(job.run_dir, "esmfold2_monomer", candidates)
        artifacts = _collect_refolding_artifacts(job.run_dir)
        finish_job(
            job.run_dir,
            bool(normalized),
            {
                "outputs": {"artifacts": artifacts, "candidates": normalized},
                "metrics": {
                    "candidate_count": len(normalized),
                    "artifact_count": len(artifacts),
                    "backend": "biohub_esmfold2_local",
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


def run_esmfold_monomer_refolding(
    source_run_dir: Path,
    candidates_jsonl: Path,
    min_plddt: float = 70.0,
    num_recycles: int = 4,
    gpu_device: object = "0",
    existing_job: JobPaths | None = None,
) -> Path:
    source_run_dir = Path(source_run_dir)
    candidates_jsonl = Path(candidates_jsonl)
    source_candidates = _source_candidates(
        candidates_jsonl,
        {STAGE_SEQUENCE_DESIGN, STAGE_GENERATION_BACKBONE_SEQUENCE, STAGE_COMPLEX_REFOLDING},
    )
    job = existing_job or create_job(
        REFOLDING_GROUP,
        job_type="monomer_refolding",
        tool="esmfold",
        inputs={"source_run_dir": str(source_run_dir), "candidates_jsonl": str(candidates_jsonl)},
        params={
            "min_plddt": min_plddt,
            "num_recycles": num_recycles,
            "backend": "docker",
            "image": "ovo-esm:latest",
            "gpu_device": normalize_gpu_device(gpu_device),
        },
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
                *docker_gpu_args(gpu_device),
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
            reference_backbone = _resolve_candidate_path(source_run_dir, source.get("complex_pdb"))
            binder_chain = _candidate_chains(source, "binder_chains", ["A"])[0]
            metrics.update(_monomer_refolding_rmsd(reference_backbone, pdb_path if pdb_path.exists() else None, binder_chain=binder_chain))
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
    gpu_device: object = "0",
    existing_job: JobPaths | None = None,
) -> Path:
    source_run_dir = Path(source_run_dir)
    candidates_jsonl = Path(candidates_jsonl)
    source_candidates = _source_candidates(
        candidates_jsonl,
        {STAGE_SEQUENCE_DESIGN, STAGE_GENERATION_BACKBONE_SEQUENCE, STAGE_COMPLEX_REFOLDING},
    )
    job = existing_job or create_job(
        REFOLDING_GROUP,
        job_type="monomer_refolding",
        tool="boltz2_monomer",
        inputs={"source_run_dir": str(source_run_dir), "candidates_jsonl": str(candidates_jsonl)},
        params={
            "min_plddt": min_plddt,
            "backend": "docker",
            "image": "mn-boltz2:cu128",
            "models_dir": str(BOLTZ_MODELS_DIR),
            "gpu_device": normalize_gpu_device(gpu_device),
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
                *docker_gpu_args(gpu_device),
                "-v",
                f"{job.run_dir}:/work",
                "-v",
                f"{BOLTZ_MODELS_DIR}:/models",
                "-w",
                "/work",
                "mn-boltz2:cu128",
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
    use_target_msa: bool = True,
    num_recycles: int = 3,
    multimer: bool = True,
    max_candidates: int = 20,
    num_sampling_steps: int = 32,
    seed: int = 0,
    device: str = "auto",
    contact_cutoff: float = 8.0,
    gpu_device: object = "0",
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
            gpu_device=gpu_device,
        )
    if tool == "boltz2_initial_guess":
        return run_boltz2_complex_refolding(
            source_run_dir,
            candidates_jsonl,
            require_monomer_success=require_monomer_success,
            use_target_template=template_mode == "target_template",
            gpu_device=gpu_device,
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
            use_target_msa=use_target_msa,
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
    use_target_msa: bool = True,
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
            "use_target_msa": use_target_msa,
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
            from esm.utils.msa import MSA
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
                target_msas: dict[str, Any] = {}
                msa_notes: list[str] = []
                if use_target_msa:
                    for target_chain, target_sequence in target_sequences.items():
                        msa_path, msa_source = target_msa_workflow.find_cached_msa_for_sequence(target_sequence)
                        if msa_path is None:
                            msa_notes.append(f"{target_chain}:missing:{msa_source}")
                            continue
                        try:
                            msa = MSA.from_a3m(msa_path)
                        except Exception as exc:
                            try:
                                with tempfile.NamedTemporaryFile("w", suffix=".a3m", delete=True) as handle:
                                    _copy_a3m_match_columns_only(Path(msa_path), Path(handle.name))
                                    handle.flush()
                                    msa = MSA.from_a3m(Path(handle.name))
                            except Exception as sanitized_exc:
                                msa_notes.append(
                                    f"{target_chain}:invalid:{Path(msa_path).name}:{exc}; sanitized_invalid:{sanitized_exc}"
                                )
                                continue
                            msa_notes.append(f"{target_chain}:sanitized:{Path(msa_path).name}")
                        query = "".join(str(getattr(msa, "query", "") or "").upper().split())
                        expected = "".join(str(target_sequence or "").upper().split())
                        if query and expected and query != expected:
                            msa_notes.append(f"{target_chain}:query_mismatch:{Path(msa_path).name}")
                            continue
                        target_msas[target_chain] = msa
                        msa_notes.append(f"{target_chain}:loaded:{msa_source}:{msa_path}")
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
                    f" initial_guess={bool(distogram_conditioning)} target_msa_count={len(target_msas)}\n"
                )
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
                        "target_msa_enabled": bool(use_target_msa),
                        "target_msa_count": len(target_msas),
                        "target_msa_notes": ";".join(msa_notes) if msa_notes else None,
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
                            "target_msa_enabled": bool(use_target_msa),
                            "target_msa_count": len(target_msas),
                            "target_msa_notes": ";".join(msa_notes) if msa_notes else None,
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
                        "target_msa_enabled": bool(use_target_msa),
                        "target_msa_count": len(target_msas),
                        "target_msa_notes": ";".join(msa_notes) if msa_notes else None,
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


def _rank_af2_initial_guess_outputs(
    output_dir: Path,
    safe_id: str,
    *,
    binder_chains: list[str],
    target_chains: list[str],
) -> list[Path]:
    """Prefer complex AF2-IG outputs over binder-only helper folds."""
    if not output_dir.exists():
        return []
    patterns = [
        f"{safe_id}_af2_initial_guess.pdb",
        f"{safe_id}_af2_initial_guess_model*.pdb",
        f"{safe_id}_af2ig_*.pdb",
        f"{safe_id}_af2_initial_guess_binder_model*.pdb",
    ]
    ranked: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(output_dir.glob(pattern)):
            if path in seen or not path.exists():
                continue
            seen.add(path)
            ranked.append(path)
    if not target_chains:
        return ranked

    complex_outputs: list[Path] = []
    fallback_outputs: list[Path] = []
    required = [chain for chain in [*binder_chains, *target_chains] if chain]
    for path in ranked:
        chains = set(_structure_chains(path))
        if all(chain in chains for chain in required) or {"A", "B"}.issubset(chains):
            complex_outputs.append(path)
        else:
            fallback_outputs.append(path)
    return complex_outputs + fallback_outputs


def run_af2_initial_guess_complex_refolding(
    source_run_dir: Path,
    candidates_jsonl: Path,
    require_monomer_success: bool = True,
    num_recycles: int = 3,
    model_count: int = 1,
    multimer: bool = True,
    binder_multimer: bool | None = None,
    use_initial_guess: bool = False,
    use_binder_template: bool = False,
    use_interface_template: bool = False,
    docker_image: str = "mn-colabdesign:latest",
    internal_parent_run_dir: Path | None = None,
    gpu_device: object = "0",
) -> Path:
    if binder_multimer is None:
        binder_multimer = multimer
    if not use_initial_guess and (use_binder_template or use_interface_template):
        raise ValueError(
            "Binder/interface templates require legacy whole-complex initial-guess mode. "
            "Target-template mode uses only the selected target structure plus binder sequence."
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
    capacity_target_only_run = bool(source_candidates) and all(
        _is_target_refolding_input_candidate(candidate)
        for candidate in source_candidates
    )
    job = create_job(
        REFOLDING_GROUP,
        job_type="complex_refolding",
        tool="af2_initial_guess",
        inputs={"source_run_dir": str(source_run_dir), "candidates_jsonl": str(candidates_jsonl)},
        params={
            "require_monomer_success": require_monomer_success,
            "num_recycles": num_recycles,
            "model_count": model_count,
            "multimer": multimer,
            "binder_multimer": binder_multimer,
            "prediction_input_mode": (
                "single_chain_template_fold"
                if capacity_target_only_run
                else
                "legacy_whole_complex_initial_guess"
                if use_initial_guess
                else "target_only_initial_guess"
            ),
            "use_initial_guess": use_initial_guess,
            "use_binder_template": use_binder_template,
            "use_interface_template": use_interface_template,
            "backend": "docker",
            "image": docker_image,
            "alphafold_models_dir": str(ALPHAFOLD_MODELS_DIR),
            "gpu_device": normalize_gpu_device(gpu_device),
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
    staged = (
        _stage_complex_inputs(source_run_dir, source_candidates, input_dir)
        if use_initial_guess
        else _stage_target_template_inputs(source_run_dir, source_candidates, input_dir)
    )
    target_fragment_template_run = bool(
        capacity_target_only_run
        and any(
            _target_only_staged_fragment_count(input_dir, safe_id, staged_path) > 1
            for safe_id, staged_path in staged.items()
        )
    )
    target_only_prediction_mode = (
        "target_fragment_template_fold"
        if target_fragment_template_run
        else "single_chain_template_fold"
    )
    if capacity_target_only_run:
        input_payload = read_json(job.run_dir / "input.json")
        params_payload = dict(input_payload.get("params") or {})
        params_payload["prediction_input_mode"] = target_only_prediction_mode
        params_payload["target_fragment_template_forced"] = bool(target_fragment_template_run)
        write_json(job.run_dir / "input.json", {**input_payload, "params": params_payload})
    output_dir.mkdir(parents=True, exist_ok=True)
    designed_chains_arg = _af2_designed_chains_arg(
        input_dir,
        staged,
        source_candidates,
        capacity_target_only_run=capacity_target_only_run,
    )
    command = [
        "docker",
        "run",
        "--rm",
        *docker_gpu_args(gpu_device),
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
        "--model-count",
        str(max(1, min(5, int(model_count)))),
        "--designed_chains",
        designed_chains_arg,
    ]
    if multimer and not capacity_target_only_run:
        command.append("--multimer")
    if binder_multimer and not capacity_target_only_run:
        command.append("--binder-multimer")
    if target_fragment_template_run:
        command.append("--target-fragment-template-only")
    elif capacity_target_only_run:
        command.append("--single-chain-template-only")
    if not use_initial_guess:
        command.append("--target-template-only")
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
            output_target_chains = (
                _structure_chains(staged_path)
                if capacity_target_only_run
                else _engine_chains_from_map(input_dir, safe_id, "targets")
                or _candidate_chains(source, "target_chains", ["B"])
            )
            output_binder_chains = (
                []
                if capacity_target_only_run
                else _engine_chains_from_map(input_dir, safe_id, "binder")
                or _candidate_chains(source, "binder_chains", ["A"])
            )
            pdb_matches = _rank_af2_initial_guess_outputs(
                output_dir / "af2_initial_guess",
                safe_id,
                binder_chains=output_binder_chains,
                target_chains=output_target_chains,
            )
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
            metrics["prediction_input_mode"] = (
                target_only_prediction_mode
                if capacity_target_only_run
                else
                "legacy_whole_complex_initial_guess"
                if use_initial_guess
                else "target_only_initial_guess"
            )
            metrics["target_fragment_template_forced"] = bool(target_fragment_template_run)
            metrics["initial_guess_used"] = True
            metrics["initial_guess_scope"] = (
                "target_fragment_template"
                if target_fragment_template_run
                else "single_chain_template"
                if capacity_target_only_run
                else "whole_complex" if use_initial_guess else "target_only"
            )
            metrics["whole_complex_initial_guess_used"] = bool(use_initial_guess)
            metrics.update(
                {
                    f"af2_{key}": value
                    for key, value in _structure_chain_retention_metrics(
                        predicted_pdb,
                        output_target_chains
                        if capacity_target_only_run
                        else [*output_binder_chains, *output_target_chains],
                    ).items()
                }
            )
            candidates.append(
                {
                    **source,
                    "candidate_id": candidate_id,
                    "stage": STAGE_COMPLEX_REFOLDING,
                    "source_tool": "af2_initial_guess",
                    "tool": "af2_initial_guess",
                    "binder_chains": output_binder_chains,
                    "target_chains": output_target_chains,
                    "complex_pdb": _rel_path(job.run_dir, predicted_pdb) if predicted_pdb else _rel_path(job.run_dir, staged_path),
                    "metrics": metrics,
                    "parents": [str(source.get("candidate_id") or "")],
                    "raw_metadata": {
                        **dict(source.get("raw_metadata") or {}),
                        "source_candidate": source,
                        "input_binder_chains": source.get("binder_chains"),
                        "input_target_chains": source.get("target_chains"),
                        "backend_status": "docker",
                        "prediction_input_mode": (
                            target_only_prediction_mode
                            if capacity_target_only_run
                            else
                            "legacy_whole_complex_initial_guess"
                            if use_initial_guess
                            else "target_only_initial_guess"
                        ),
                        "target_fragment_template_forced": bool(target_fragment_template_run),
                        "initial_guess_used": True,
                        "initial_guess_scope": (
                            "target_fragment_template"
                            if target_fragment_template_run
                            else "single_chain_template"
                            if capacity_target_only_run
                            else "whole_complex" if use_initial_guess else "target_only"
                        ),
                        "whole_complex_initial_guess_used": bool(use_initial_guess),
                        "input_complex_role": (
                            "coordinate_initial_guess" if use_initial_guess else "post_hoc_reference_only"
                        ),
                        "prediction_input_pdb": _rel_path(job.run_dir, staged_path),
                        "input_complex": _rel_path(job.run_dir, staged_path) if use_initial_guess else None,
                        "target_template_pdb": _rel_path(job.run_dir, staged_path) if not use_initial_guess else None,
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
    gpu_device: object = "0",
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
            "image": "mn-boltz2:cu128",
            "models_dir": str(BOLTZ_MODELS_DIR),
            "template_mode": "target_template" if use_target_template else "no_template",
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": diffusion_samples,
            "write_full_pae": write_full_pae,
            "benchmark_run_csv": str(benchmark_run_csv) if benchmark_run_csv else None,
            "gpu_device": normalize_gpu_device(gpu_device),
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
    target_only_by_id = {
        _safe_id(candidate.get("candidate_id")): _is_target_refolding_input_candidate(candidate)
        for candidate in source_candidates
    }
    all_target_only = bool(target_only_by_id) and all(target_only_by_id.values())
    effective_use_target_template = bool(use_target_template)
    staged = _stage_complex_inputs(source_run_dir, source_candidates, input_pdb_dir)
    yaml_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_by_id = {}
    target_template = None
    designed_chains_arg = "A"
    for safe_id, pdb_path in staged.items():
        source = next(candidate for candidate in source_candidates if safe_id == _safe_id(candidate.get("candidate_id")))
        source_by_id[safe_id] = source
        if all_target_only:
            chains = _structure_chains(pdb_path)
            if chains:
                designed_chains_arg = ",".join(chains)
        if target_template is None:
            target_template = (
                _resolve_candidate_path(source_run_dir, source.get("complex_pdb"))
                if target_only_by_id.get(safe_id)
                else None
            )
            if target_template is None or not target_template.exists():
                target_template = _target_pdb_for_candidate(source_run_dir, source)
    write_json(raw_root / "candidate_id_map.json", source_by_id)
    staged_template_arg = "--no-template "
    if effective_use_target_template and target_template and target_template.exists():
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
        "mn-boltz2:cu128",
        "-lc",
        (
            "source /opt/conda/etc/profile.d/conda.sh && conda activate boltz2 && "
            "python /scripts/prepare_inputs.py "
            "--input_dir /work/artifacts/raw/boltz2_initial_guess/input_pdbs "
            "--output_dir /work/artifacts/raw/boltz2_initial_guess/yaml_inputs "
            f"--design_type {'scaffold' if all_target_only else 'binder'} "
            f"--designed_chains {designed_chains_arg} "
            + staged_template_arg
        ),
    ]
    predict_command = [
        "docker",
        "run",
        "--rm",
        *docker_gpu_args(gpu_device),
        "--shm-size=32G",
        "-v",
        f"{job.run_dir}:/work",
        "-v",
        f"{BOLTZ_MODELS_DIR}:/models",
        "-w",
        "/work",
        "mn-boltz2:cu128",
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
    cleanup_command = None
    if effective_use_target_template and target_template and target_template.exists():
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
            "mn-boltz2:cu128",
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
        predict_steps = []
        if cleanup_command is not None:
            predict_steps.append({"name": "boltz2-clean-template-pdb", "command": cleanup_command})
        predict_steps.append({"name": "boltz2-initial-guess", "command": predict_command})
        rc = _run_shell_steps(job.run_dir, predict_steps)
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
            target_only = bool(target_only_by_id.get(safe_id))
            output_target_chains = (
                _structure_chains(staged[safe_id])
                if target_only
                else _engine_chains_from_map(input_pdb_dir, safe_id, "targets")
                or _candidate_chains(source, "target_chains", ["B"])
            )
            output_binder_chains = (
                []
                if target_only
                else _engine_chains_from_map(input_pdb_dir, safe_id, "binder")
                or _candidate_chains(source, "binder_chains", ["A"])
            )
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
            predicted_cif = cif_matches[0] if cif_matches else None
            metrics.update(
                {
                    f"boltz2_{key}": value
                    for key, value in _structure_chain_retention_metrics(
                        predicted_cif,
                        output_target_chains
                        if target_only
                        else [*output_binder_chains, *output_target_chains],
                    ).items()
                }
            )
            candidates.append(
                {
                    **source,
                    "candidate_id": _complex_candidate_id(
                        source,
                        "boltz2_initial_guess",
                        "target_template" if effective_use_target_template else "no_template",
                    ),
                    "stage": STAGE_COMPLEX_REFOLDING,
                    "source_tool": "boltz2_initial_guess",
                    "tool": "boltz2_initial_guess",
                    "binder_chains": output_binder_chains,
                    "target_chains": output_target_chains,
                    "complex_pdb": _rel_path(job.run_dir, predicted_cif) if predicted_cif else source.get("complex_pdb"),
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
    for pattern in (
        "**/*summary_confidence*.json",
        "**/*summary_confidences*.json",
        "**/*confidences_aggregated.json",
        "**/confidence*.json",
        "**/*_confidences.json",
        "**/*confidences.json",
    ):
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
    metrics: dict[str, Any] = {}
    for key, value in payload.items():
        metric_key = f"{prefix}_{key}"
        if isinstance(value, (str, int, float, bool)) or value is None:
            metrics[metric_key] = value
            continue
        try:
            array = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            continue
        if not array.size:
            continue
        finite = array[np.isfinite(array)]
        if not finite.size:
            continue
        metrics[f"{metric_key}_mean"] = float(np.mean(finite))
        metrics[f"{metric_key}_min"] = float(np.min(finite))
        metrics[f"{metric_key}_max"] = float(np.max(finite))
    return metrics


def _prediction_pae_file(output_dir: Path, prefix: str) -> Path | None:
    if prefix.startswith("protenix"):
        patterns = ("**/*_full_data_sample_*.json", "**/*full_data*.json")
    else:
        patterns = {
            "rf3": ("**/*_confidences.json", "**/*confidences.json"),
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
    chain_role_finding_count = 0
    chain_role_warning_count = 0
    if rc == 0:
        for safe_id, staged_path in staged.items():
            source = next(item for item in source_candidates if safe_id == _safe_id(item.get("candidate_id")))
            candidate_output = output_root / safe_id
            prediction = _first_prediction_file(candidate_output, safe_id)
            pae_source = _prediction_pae_file(candidate_output, tool)
            pae_path = _standard_pae_json_path(pae_source, candidate_output, safe_id, tool) if pae_source else None
            raw_metadata = source.get("raw_metadata") if isinstance(source.get("raw_metadata"), dict) else {}
            if prediction is None:
                continue
            if _is_target_refolding_input_candidate(source):
                target_only = True
                target_source_chains = _target_only_candidate_chains(source)
                if not target_source_chains:
                    target_source_chains = _structure_chains(prediction)
                binder_source_chains = []
            else:
                target_only = False
                binder_source_chains, target_source_chains = _infer_chain_roles(source_run_dir, source, prediction or staged_path)
            normalized_prediction, binder_chains, target_chains, normalization_findings = normalize_candidate_structure_roles(
                structure_path=prediction,
                output_dir=candidate_output / "role_normalized",
                safe_id=safe_id,
                binder_source_chains=binder_source_chains,
                target_source_chains=target_source_chains,
                target_only=target_only,
            )
            prediction_for_candidate = normalized_prediction or prediction
            if normalized_prediction is None:
                binder_chains = [] if target_only else binder_source_chains
                target_chains = target_source_chains
            metrics = dict(source.get("metrics") or {})
            metrics.update(_prediction_summary_metrics(candidate_output, tool))
            metrics["complex_refolding_backend"] = tool
            metrics[f"{tool}_has_full_pae"] = pae_path is not None
            metrics["chain_role_normalized"] = normalized_prediction is not None
            findings = read_json(staged_path.with_name(f"{safe_id}.chain_role_findings.json"))
            if not isinstance(findings, list):
                findings = read_json(staged_path.with_name(f"{safe_id}.chain_role_warnings.json"))
            if not isinstance(findings, list):
                findings = []
            findings = [*findings, *normalization_findings]
            warnings = chain_roles.problem_findings(findings)
            chain_role_finding_count += len(findings)
            chain_role_warning_count += len(warnings)
            candidates.append(
                _attach_chain_role_warnings(
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
                        "complex_pdb": _rel_path(job_run_dir, prediction_for_candidate) if prediction_for_candidate else _rel_path(job_run_dir, staged_path),
                        "metrics": metrics,
                        "parents": [str(source.get("candidate_id") or "")],
                        "raw_metadata": {
                            **dict(raw_metadata),
                            "source_candidate": source,
                            "input_complex": _rel_path(job_run_dir, staged_path),
                            "original_prediction": _rel_path(job_run_dir, prediction),
                            "role_normalized_prediction": _rel_path(job_run_dir, normalized_prediction) if normalized_prediction else None,
                            "prediction_dir": _rel_path(job_run_dir, candidate_output),
                            "pae_source_path": _rel_path(job_run_dir, pae_source) if pae_source else None,
                            "pae_path": _rel_path(job_run_dir, pae_path) if pae_path else None,
                        },
                    },
                    findings,
                )
            )
    normalized = write_candidates(job_run_dir, tool, candidates) if candidates else []
    artifacts = _collect_refolding_artifacts(job_run_dir)
    residue_counts = _candidate_residue_counts(source_candidates, staged)
    total_residues = sum(residue_counts.values())
    finish_job(
        job_run_dir,
        rc == 0 and bool(normalized),
        {
            "outputs": {"artifacts": artifacts, "candidates": normalized},
            "metrics": {
                "return_code": rc,
                "candidate_count": len(normalized),
                "total_residues": int(total_residues) if total_residues else None,
                "max_system_residues": max(residue_counts.values()) if residue_counts else None,
                "artifact_count": len(artifacts),
                "chain_role_finding_count": chain_role_finding_count,
                "chain_role_warning_count": chain_role_warning_count,
                **(extra_metrics or {}),
            },
            "downstream_artifacts": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
            },
        },
    )
    return job_run_dir


def run_openfold3_complex_refolding(
    source_run_dir: Path,
    candidates_jsonl: Path,
    require_monomer_success: bool = True,
    use_target_msa: bool = True,
    checkpoint_path: Path | None = OPENFOLD3_CHECKPOINT,
    num_diffusion_samples: int = 5,
    num_model_seeds: int = 1,
    num_recycles: int = 3,
    use_msa_server: bool = False,
    benchmark_run_csv: Path | None = None,
    internal_parent_run_dir: Path | None = None,
    gpu_device: object = "0",
    docker_image: str = OPENFOLD3_IMAGE,
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
        "openfold3",
        {"source_run_dir": str(source_run_dir), "candidates_jsonl": str(candidates_jsonl)},
        {
            "image": docker_image,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "target_msa_supported": True,
            "use_target_msa": use_target_msa,
            "use_msa_server": use_msa_server,
            "num_diffusion_samples": num_diffusion_samples,
            "num_model_seeds": num_model_seeds,
            "num_recycles": num_recycles,
            "benchmark_run_csv": str(benchmark_run_csv) if benchmark_run_csv else None,
            "gpu_device": normalize_gpu_device(gpu_device),
        },
    )
    if internal_parent_run_dir is not None:
        mark_internal_job(
            job.run_dir,
            parent_run_dir=Path(internal_parent_run_dir),
            parent_task_group="benchmark",
            role="benchmark_engine_subrun",
            engine="openfold3",
        )
    raw_root = job.run_dir / "artifacts" / "raw" / "openfold3"
    staged = _stage_complex_inputs(source_run_dir, source_candidates, raw_root / "inputs")
    query_inputs, msa_metrics = _write_openfold3_query_inputs(
        job_run_dir=job.run_dir,
        source_run_dir=source_run_dir,
        source_candidates=source_candidates,
        staged=staged,
        query_root=raw_root / "queries",
        benchmark_run_csv=benchmark_run_csv,
        use_target_msa=use_target_msa,
    )
    output_root = raw_root / "output"
    runner_yaml = _write_openfold3_runner_yaml(raw_root, num_recycles=num_recycles)
    checkpoint_mount: list[str] = []
    checkpoint_args: list[str] = []
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_mount = ["-v", f"{checkpoint_path.parent}:/ref/openfold3:ro"]
        checkpoint_args = ["--inference-ckpt-path", f"/ref/openfold3/{checkpoint_path.name}"]
    steps: list[dict[str, Any]] = []
    for safe_id, query_path in query_inputs.items():
        candidate_output_dir = output_root / safe_id
        candidate_output_dir.mkdir(parents=True, exist_ok=True)
        steps.append(
            {
                "name": f"openfold3-{safe_id}",
                "candidate_ids": [safe_id],
                "command": [
                    "docker",
                    "run",
                    "--rm",
                    *docker_gpu_args(gpu_device),
                    "--shm-size=32G",
                    "-v",
                    f"{job.run_dir}:/work",
                    *checkpoint_mount,
                    "-w",
                    "/work",
                    docker_image,
                    "run_openfold",
                    "predict",
                    "--runner-yaml",
                    f"/work/{runner_yaml.relative_to(job.run_dir)}",
                    "--query-json",
                    f"/work/{query_path.relative_to(job.run_dir)}",
                    "--output-dir",
                    f"/work/{candidate_output_dir.relative_to(job.run_dir)}",
                    "--use-msa-server",
                    str(bool(use_msa_server)),
                    "--num-diffusion-samples",
                    str(max(1, int(num_diffusion_samples))),
                    "--num-model-seeds",
                    str(max(1, int(num_model_seeds))),
                    *checkpoint_args,
                ],
            }
        )
    _annotate_step_residue_totals(steps, _candidate_residue_counts(source_candidates, staged))
    rc = _run_shell_steps(job.run_dir, steps)
    missing_prediction_ids = [
        safe_id
        for safe_id in query_inputs
        if _first_prediction_file(output_root / safe_id, safe_id) is None
    ]
    effective_rc = int(rc) if rc else (1 if missing_prediction_ids else 0)
    return _finish_external_complex_refolding(
        job_run_dir=job.run_dir,
        source_run_dir=source_run_dir,
        source_candidates=source_candidates,
        staged=staged,
        output_root=output_root,
        tool="openfold3",
        rc=effective_rc,
        extra_metrics={
            "initial_guess_supported": False,
            "target_msa_supported": True,
            "use_target_msa": use_target_msa,
            "use_msa_server": use_msa_server,
            "num_diffusion_samples": num_diffusion_samples,
            "num_model_seeds": num_model_seeds,
            "num_recycles": num_recycles,
            "openfold3_prediction_count": len(query_inputs) - len(missing_prediction_ids),
            "openfold3_missing_prediction_count": len(missing_prediction_ids),
            "openfold3_missing_prediction_ids": missing_prediction_ids,
            **msa_metrics,
        },
    )


def run_rf3_complex_refolding(
    source_run_dir: Path,
    candidates_jsonl: Path,
    require_monomer_success: bool = True,
    checkpoint_path: Path = RF3_CHECKPOINT,
    use_target_msa: bool = True,
    use_target_template: bool = True,
    n_recycles: int = 10,
    num_steps: int = 50,
    diffusion_batch_size: int = 5,
    seed: int = 0,
    benchmark_run_csv: Path | None = None,
    internal_parent_run_dir: Path | None = None,
    gpu_device: object = "0",
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
            "target_template_supported": True,
            "use_target_template": use_target_template,
            "n_recycles": n_recycles,
            "num_steps": num_steps,
            "diffusion_batch_size": diffusion_batch_size,
            "seed": seed,
            "benchmark_run_csv": str(benchmark_run_csv) if benchmark_run_csv else None,
            "gpu_device": normalize_gpu_device(gpu_device),
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
        use_target_template=use_target_template,
    )
    output_root = raw_root / "output"
    steps: list[dict[str, Any]] = []
    for safe_id, rf3_input in rf3_inputs.items():
        (output_root / safe_id).mkdir(parents=True, exist_ok=True)
        steps.append(
            {
                "name": f"rf3-{safe_id}",
                "candidate_ids": [safe_id],
                "command": [
                    "docker",
                    "run",
                    "--rm",
                    *docker_gpu_args(gpu_device),
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
    _annotate_step_residue_totals(steps, _candidate_residue_counts(source_candidates, staged))
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
            "target_template_supported": True,
            "use_target_template": use_target_template,
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
    existing_job: JobPaths | None = None,
    gpu_device: object = "0",
) -> Path:
    source_run_dir = Path(source_run_dir)
    allowed = {STAGE_MONOMER_REFOLDING} if require_monomer_success else {
        STAGE_MONOMER_REFOLDING,
        STAGE_SEQUENCE_DESIGN,
        STAGE_GENERATION_BACKBONE_SEQUENCE,
        STAGE_COMPLEX_REFOLDING,
    }
    source_candidates = _source_candidates(Path(candidates_jsonl), allowed)
    job = existing_job or create_job(
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
            "gpu_device": normalize_gpu_device(gpu_device),
        },
    )
    if existing_job is not None:
        update_status(job.run_dir, "running", recovery_resumed=True)
    elif internal_parent_run_dir is not None:
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
        use_target_template=False,
    )
    output_root = raw_root / "output"
    steps: list[dict[str, Any]] = []
    for safe_id, protenix_input_dir in protenix_inputs.items():
        candidate_output_dir = output_root / safe_id
        candidate_output_dir.mkdir(parents=True, exist_ok=True)
        completed_structures = [
            path
            for path in candidate_output_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".cif", ".pdb"}
        ]
        if existing_job is not None and len(completed_structures) >= max(1, int(samples)):
            continue
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
                    "candidate_ids": [safe_id],
                    "command": [
                        "docker",
                        "run",
                        "--rm",
                        *docker_gpu_args(gpu_device),
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
    _annotate_step_residue_totals(steps, _candidate_residue_counts(source_candidates, staged))
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


def run_protenix_cli_complex_refolding(
    source_run_dir: Path,
    candidates_jsonl: Path,
    require_monomer_success: bool = True,
    use_msa: bool = True,
    use_template: bool = False,
    benchmark_run_csv: Path | None = None,
    cycle: int = 10,
    diffusion_steps: int = 200,
    samples: int = 5,
    seeds: str = "101",
    model_name: str = PROTENIX_V1_MODEL,
    tool: str = "protenix_v1",
    dtype: str = "bf16",
    use_default_params: bool = True,
    batch_mode: bool = True,
    batch_size: int = 32,
    internal_parent_run_dir: Path | None = None,
    existing_job: JobPaths | None = None,
    gpu_device: object = "0",
    docker_image: str = PROTENIX_CLI_IMAGE,
    reference_dir: Path = PROTENIX_CLI_REFERENCE_DIR,
) -> Path:
    """Run the standalone Protenix CLI adapter.

    This is intentionally separate from the PXDesign-backed v0.5.0 adapter above.
    """
    source_run_dir = Path(source_run_dir)
    reference_dir = Path(reference_dir)
    allowed = {STAGE_MONOMER_REFOLDING} if require_monomer_success else {
        STAGE_MONOMER_REFOLDING,
        STAGE_SEQUENCE_DESIGN,
        STAGE_GENERATION_BACKBONE_SEQUENCE,
        STAGE_COMPLEX_REFOLDING,
    }
    source_candidates = _source_candidates(Path(candidates_jsonl), allowed)
    job = existing_job or create_job(
        REFOLDING_GROUP,
        "complex_refolding",
        tool,
        {"source_run_dir": str(source_run_dir), "candidates_jsonl": str(candidates_jsonl)},
        {
            "image": docker_image,
            "reference_dir": str(reference_dir),
            "model_name": model_name,
            "use_msa": use_msa,
            "use_template": use_template,
            "benchmark_run_csv": str(benchmark_run_csv) if benchmark_run_csv else None,
            "cycle": cycle,
            "diffusion_steps": diffusion_steps,
            "samples": samples,
            "seeds": seeds,
            "dtype": dtype,
            "use_default_params": bool(use_default_params),
            "batch_mode": bool(batch_mode),
            "batch_size": max(1, int(batch_size)),
            "initial_guess_supported": False,
            "gpu_device": normalize_gpu_device(gpu_device),
        },
    )
    if existing_job is not None:
        update_status(job.run_dir, "running", recovery_resumed=True)
    elif internal_parent_run_dir is not None:
        mark_internal_job(
            job.run_dir,
            parent_run_dir=Path(internal_parent_run_dir),
            parent_task_group="benchmark",
            role="benchmark_engine_subrun",
            engine=tool,
        )
    raw_root = job.run_dir / "artifacts" / "raw" / tool
    staged = _stage_complex_inputs(source_run_dir, source_candidates, raw_root / "inputs")
    protenix_inputs, msa_metrics = _write_protenix_json_inputs(
        source_run_dir=source_run_dir,
        source_candidates=source_candidates,
        staged=staged,
        json_root=raw_root / "json",
        msa_root=raw_root / "msas",
        benchmark_run_csv=benchmark_run_csv,
        use_target_msa=use_msa,
        use_target_template=use_template,
    )
    output_root = raw_root / "output"
    steps: list[dict[str, Any]] = []
    pending_inputs: dict[str, Path] = {}
    for safe_id, protenix_input_dir in protenix_inputs.items():
        candidate_output_dir = output_root / safe_id
        candidate_output_dir.mkdir(parents=True, exist_ok=True)
        completed_structures = [
            path
            for path in candidate_output_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".cif", ".pdb"}
        ]
        if existing_job is not None and len(completed_structures) >= max(1, int(samples)):
            continue
        pending_inputs[safe_id] = protenix_input_dir

    common_command_prefix = [
        "docker",
        "run",
        "--rm",
        *docker_gpu_args(gpu_device),
        "--shm-size=32G",
        "-v",
        f"{job.run_dir}:/work",
        "-v",
        f"{reference_dir}:/ref/protenix:rw",
        "-w",
        "/work",
        "-e",
        "PROTENIX_ROOT_DIR=/ref/protenix",
        "-e",
        "CUTLASS_PATH=/opt/cutlass",
        docker_image,
        "protenix",
        "pred",
    ]
    common_predict_args = [
        "-s",
        str(seeds or "101"),
        "-n",
        str(model_name),
        "-c",
        str(max(1, int(cycle))),
        "-p",
        str(max(1, int(diffusion_steps))),
        "-e",
        str(max(1, int(samples))),
        "-d",
        str(dtype or "bf16"),
        "--use_msa",
        str(bool(use_msa)),
        "--use_template",
        str(bool(use_template)),
        "--use_default_params",
        str(bool(use_default_params)),
        "--need_atom_confidence",
        "True",
    ]
    pending_items = list(pending_inputs.items())
    target_ids_by_candidate = _candidate_target_ids(source_candidates)
    if batch_mode and pending_items:
        batch_input_root = raw_root / "batch_json"
        if batch_input_root.exists():
            shutil.rmtree(batch_input_root)
        batch_input_root.mkdir(parents=True, exist_ok=True)
        if target_ids_by_candidate:
            grouped_pending: dict[str, list[tuple[str, Path]]] = {}
            for safe_id, input_dir in pending_items:
                grouped_pending.setdefault(target_ids_by_candidate.get(safe_id) or "", []).append((safe_id, input_dir))
            chunks = [
                chunk
                for _target_id, target_items in grouped_pending.items()
                for chunk in _chunk_items(target_items, max(1, int(batch_size)))
            ]
        else:
            chunks = _chunk_items(pending_items, max(1, int(batch_size)))
        for chunk_index, chunk in enumerate(chunks, start=1):
            chunk_input_root = batch_input_root / f"chunk_{chunk_index:04d}"
            chunk_input_root.mkdir(parents=True, exist_ok=True)
            for safe_id, protenix_input_dir in chunk:
                batch_candidate_dir = chunk_input_root / safe_id
                batch_candidate_dir.mkdir(parents=True, exist_ok=True)
                original_input_json = protenix_input_dir / f"{safe_id}.json"
                input_jsons = [original_input_json] if original_input_json.exists() else [
                    path for path in sorted(protenix_input_dir.glob("*.json")) if not path.name.endswith("-update-msa.json")
                ]
                for input_json in input_jsons:
                    shutil.copy2(input_json, batch_candidate_dir / input_json.name)
            steps.append(
                {
                    "name": f"{tool}-predict-batch-{chunk_index:04d}-of-{len(chunks):04d}-{len(chunk)}",
                    "candidate_ids": [safe_id for safe_id, _input_dir in chunk],
                    "target_ids": sorted(
                        {
                            str(target_ids_by_candidate.get(safe_id) or "")
                            for safe_id, _input_dir in chunk
                            if target_ids_by_candidate.get(safe_id)
                        }
                    ),
                    "command": [
                        *common_command_prefix,
                        "-i",
                        f"/work/{chunk_input_root.relative_to(job.run_dir)}",
                        "-o",
                        f"/work/{output_root.relative_to(job.run_dir)}",
                        *common_predict_args,
                    ],
                }
            )
    else:
        for safe_id, protenix_input_dir in pending_items:
            candidate_output_dir = output_root / safe_id
            steps.append(
                {
                    "name": f"{tool}-predict-{safe_id}",
                    "candidate_ids": [safe_id],
                    "target_ids": [target_ids_by_candidate[safe_id]] if target_ids_by_candidate.get(safe_id) else [],
                    "command": [
                        *common_command_prefix,
                        "-i",
                        f"/work/{protenix_input_dir.relative_to(job.run_dir)}",
                        "-o",
                        f"/work/{candidate_output_dir.relative_to(job.run_dir)}",
                        *common_predict_args,
                    ],
                }
            )

    def verify_prediction_step(step: dict[str, Any], _step_index: int) -> str:
        missing = [
            safe_id
            for safe_id in step.get("candidate_ids", [])
            if len(
                [
                    path
                    for path in (output_root / str(safe_id)).rglob("*")
                    if path.is_file() and path.suffix.lower() in {".cif", ".pdb"}
                ]
            )
            < max(1, int(samples))
        ]
        if not missing:
            return ""
        write_json(
            job.run_dir / "artifacts" / "raw" / tool / "incomplete_predictions.json",
            {
                "tool": tool,
                "batch_mode": bool(batch_mode),
                "batch_size": max(1, int(batch_size)),
                "expected_samples": max(1, int(samples)),
                "failed_step": str(step.get("name") or ""),
                "incomplete_candidate_ids": missing,
            },
        )
        return f"{tool} step {step.get('name')} completed but missing predictions for: {', '.join(missing)}"

    _annotate_step_residue_totals(steps, _candidate_residue_counts(source_candidates, staged))
    _annotate_step_target_ids(steps, target_ids_by_candidate)
    rc = _run_shell_steps(job.run_dir, steps, verify_step=verify_prediction_step)
    if rc == 0:
        incomplete = [
            safe_id
            for safe_id in pending_inputs
            if len(
                [
                    path
                    for path in (output_root / safe_id).rglob("*")
                    if path.is_file() and path.suffix.lower() in {".cif", ".pdb"}
                ]
            )
            < max(1, int(samples))
        ]
        if incomplete:
            write_json(
                job.run_dir / "artifacts" / "raw" / tool / "incomplete_predictions.json",
                {
                    "tool": tool,
                    "batch_mode": bool(batch_mode),
                    "batch_size": max(1, int(batch_size)),
                    "expected_samples": max(1, int(samples)),
                    "incomplete_candidate_ids": incomplete,
                },
            )
            rc = 1
    return _finish_external_complex_refolding(
        job_run_dir=job.run_dir,
        source_run_dir=source_run_dir,
        source_candidates=source_candidates,
        staged=staged,
        output_root=output_root,
        tool=tool,
        rc=rc,
        extra_metrics={
            "initial_guess_supported": False,
            "target_msa_supported": True,
            "use_msa": use_msa,
            "use_template": use_template,
            "model_name": model_name,
            "reference_dir": str(reference_dir),
            "docker_image": docker_image,
            "batch_mode": bool(batch_mode),
            "batch_size": max(1, int(batch_size)),
            "batch_pending_count": len(pending_inputs),
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
    gpu_device: object = "0",
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
            "gpu_device": normalize_gpu_device(gpu_device),
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
        sequences = _pdb_sequences_by_chain(staged_path)
        binder_chains = [] if _is_target_refolding_input_candidate(source) else _infer_chain_roles(source_run_dir, source, staged_path)[0]
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
        *docker_gpu_args(gpu_device),
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
        "data.cfg.target_id_regex=^(.+)$",
        "data.cfg.num_workers=1",
        "data.cfg.moldir=/cache/datasets--boltzgen--inference-data/snapshots/c3d36fd276e9caf098c75d4113c6d5eb320b1a4c/mols.zip",
        "output=/work/artifacts/raw/boltzgen_fold/output",
        "checkpoint=/cache/models--boltzgen--boltzgen-1/snapshots/c1be29e1f82ffcc72264f64b993c43fb4e0d17f0/boltz2_conf_final.ckpt",
        f"recycling_steps={int(recycling_steps)}",
        f"sampling_steps={int(sampling_steps)}",
        f"diffusion_samples={int(diffusion_samples)}",
        "trainer.devices=1",
    ]
    boltzgen_steps = [
        {
            "name": "boltzgen-target-template-fold",
            "candidate_ids": list(staged),
            "command": command,
        }
    ]
    _annotate_step_residue_totals(boltzgen_steps, _candidate_residue_counts(source_candidates, staged))
    rc = _run_shell_steps(job.run_dir, boltzgen_steps)
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
