from __future__ import annotations

import csv
import json
import math
import random
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from mn_protein_design.core.candidates import STAGE_COMPLEX_REFOLDING, read_candidates, write_candidates
from mn_protein_design.core.jobs import (
    create_job,
    finish_job,
    mark_internal_job,
    read_json,
    update_status,
    write_json,
)
from mn_protein_design.core.portable_paths import resolve_stored_path, store_managed_paths
from mn_protein_design.core.scheduler import apply_docker_cpu_limit
from mn_protein_design.core.local_worker import spawn_worker_for_run
from mn_protein_design.runtime import runs_root
from mn_protein_design.workflows.analysis import _run_ipsae
from mn_protein_design.workflows.design import (
    default_target_contig,
    run_bindcraft,
    run_boltzgen,
    run_genie3,
    run_proteina_complexa,
    run_protpardelle_1c,
    run_pxdesign,
    run_rfdiffusion3_foundry,
    run_rfdiffusion_classic,
)
from mn_protein_design.workflows.esm_binder import run_esmfold2_native_binder_design
from mn_protein_design.workflows import chain_roles
from mn_protein_design.workflows import refolding as refolding_workflow
from mn_protein_design.workflows.refolding import run_af2_initial_guess_complex_refolding
from mn_protein_design.workflows.sequence_design import (
    _chain_ids_from_pdb,
    _remap_fixed_residues_to_engine_chains,
    _source_role_chains,
    _stage_ligandmpnn_role_normalized_input,
    run_ligandmpnn_sequence_design,
)


DESIGN_CAMPAIGN_GROUP = "design-campaign"
ENGINE_ORDER = (
    "template_redesign",
    "rfdiffusion_classic",
    "bindcraft",
    "bindcraft2",
    "rfdiffusion3_foundry",
    "boltzgen",
    "pxdesign",
    "genie3",
    "esmfold2_binder_design",
    "protpardelle_1c",
    "proteina_complexa",
)


def _sample_fixed_binder_lengths(binder_length: str, *, seed: int, engine: str, attempts: int) -> list[int]:
    tokens = [token for token in re.split(r"[-,\s]+", str(binder_length).strip()) if token]
    if not tokens:
        raise ValueError("Binder length is required.")
    lengths = [int(token) for token in tokens]
    if len(lengths) > 2:
        raise ValueError("Binder length should be a single value or a min-max range.")
    attempts = max(1, int(attempts))
    if len(lengths) == 1 or lengths[0] == lengths[-1]:
        return [int(lengths[0])] * attempts
    low, high = int(lengths[0]), int(lengths[-1])
    if high < low:
        raise ValueError("Binder length max must be greater than or equal to min.")
    rng = random.Random(f"{int(seed)}:{engine}:{low}-{high}")
    return [rng.randint(low, high) for _ in range(attempts)]


def _record_sampled_binder_length(
    run_dir: Path,
    *,
    requested_length: str,
    sampled_length: int,
    attempt_index: int,
    total_attempts: int,
    output_index: int | None = None,
    outputs_per_attempt: int | None = None,
) -> None:
    input_path = run_dir / "input.json"
    payload = read_json(input_path)
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    length_metadata = {
        "requested_binder_length": requested_length,
        "sampled_binder_length": int(sampled_length),
        "sampled_length_attempt_index": int(attempt_index),
        "sampled_length_total_attempts": int(total_attempts),
    }
    if output_index is not None:
        length_metadata["sampled_length_output_index"] = int(output_index)
    if outputs_per_attempt is not None:
        length_metadata["sampled_length_outputs_per_attempt"] = int(outputs_per_attempt)
    params.update(length_metadata)
    payload["params"] = params
    write_json(input_path, payload)
    metadata_path = run_dir / "metadata.json"
    metadata = read_json(metadata_path)
    metadata.update(length_metadata)
    write_json(metadata_path, metadata)


ENGINE_LABELS = {
    "template_redesign": "Template redesign",
    "rfdiffusion_classic": "RFdiffusion classic",
    "bindcraft": "BindCraft",
    "bindcraft2": "BindCraft 2",
    "rfdiffusion3_foundry": "RFdiffusion3 / Foundry",
    "boltzgen": "BoltzGen",
    "pxdesign": "PXDesign",
    "genie3": "Genie3",
    "esmfold2_binder_design": "ESMFold2 binder design",
    "protpardelle_1c": "Protpardelle-1c",
    "proteina_complexa": "Proteina-Complexa",
}
REFOLDING_ENGINE_OPTIONS = (
    "AlphaFast AF3",
    "ColabFold",
    "AF2-IG",
    "ESMFold2",
    "Boltz-2",
    "RF3",
    "OpenFold-3",
    "Protenix",
    "Protenix v1",
    "Protenix v2",
    "BoltzGen Fold",
)
REFOLDER_FLAG_BY_LABEL = {
    "AlphaFast AF3": "run_alphafast_af3",
    "AF3": "run_alphafast_af3",
    "ColabFold": "run_colabfold",
    "AF2-IG": "run_af2_initial_guess",
    "ESMFold2": "run_esmfold2",
    "Boltz-2": "run_boltz2_initial_guess",
    "RF3": "run_rf3",
    "OpenFold-3": "run_openfold3",
    "Protenix": "run_protenix",
    "Protenix v1": "run_protenix_v1",
    "Protenix v2": "run_protenix_v2",
    "BoltzGen Fold": "run_boltzgen_fold",
}


def _refolder_default_use_target_msa(refolder: str) -> bool:
    return str(refolder or "") != "AF2-IG"


GENERATOR_ONLY_RECIPES = {"staged", "engine_scout"}
VANILLA_ONLY_ENGINES = {"bindcraft2"}
NATIVE_SEQUENCE_GENERATOR_ENGINES = {"esmfold2_binder_design", "proteina_complexa"}
LEVEL1_SEQUENCE_REQUIRED_ENGINES = (
    set(ENGINE_ORDER)
    - NATIVE_SEQUENCE_GENERATOR_ENGINES
    - VANILLA_ONLY_ENGINES
    - {"template_redesign"}
)


def _staged_native_outputs_per_attempt(refinement_config: dict[str, Any] | None) -> int:
    refinement_config = refinement_config if isinstance(refinement_config, dict) else {}
    two_step = (
        refinement_config.get("staged_two_step")
        if isinstance(refinement_config.get("staged_two_step"), dict)
        else {}
    )
    if not bool(two_step.get("enabled")):
        return 1
    return max(1, int(two_step.get("first_keep_per_attempt", 2)))


def _normalize_staged_sequence_refinement_config(
    config: dict[str, Any] | None,
    selected_engines: list[str],
    *,
    sequences_per_backbone: int,
) -> dict[str, Any]:
    normalized = dict(config or {"mode": "redesign_all"})
    normalized.setdefault("mode", "redesign_all")
    normalized.setdefault("engine", "ProteinMPNN")
    normalized.setdefault("sequences_per_structure", max(1, int(sequences_per_backbone)))

    sequence_by_engine = (
        dict(normalized.get("sequence_by_engine"))
        if isinstance(normalized.get("sequence_by_engine"), dict)
        else {}
    )
    for engine in selected_engines:
        row = dict(sequence_by_engine.get(engine) or {})
        row.setdefault("engine", normalized.get("engine") or "ProteinMPNN")
        if engine in LEVEL1_SEQUENCE_REQUIRED_ENGINES:
            row["enabled"] = True
        else:
            row.setdefault("enabled", False)
        sequence_by_engine[engine] = row
    normalized["sequence_by_engine"] = sequence_by_engine

    two_step = (
        dict(normalized.get("staged_two_step"))
        if isinstance(normalized.get("staged_two_step"), dict)
        else {}
    )
    if two_step:
        second_sequence_by_engine = (
            dict(two_step.get("second_sequence_by_engine"))
            if isinstance(two_step.get("second_sequence_by_engine"), dict)
            else {}
        )
        for engine in selected_engines:
            row = dict(second_sequence_by_engine.get(engine) or {})
            row.setdefault("engine", two_step.get("second_engine") or "Soluble ProteinMPNN")
            row.setdefault("enabled", True)
            second_sequence_by_engine[engine] = row
        two_step["second_sequence_by_engine"] = second_sequence_by_engine
        two_step.setdefault("second_sequences_per_structure", max(1, int(sequences_per_backbone)))
        normalized["staged_two_step"] = two_step
    return normalized


def _sequence_by_engine_with_level_settings(
    sequence_by_engine: dict[str, Any] | None,
    *,
    sequences_per_structure: int,
    sampling_temp: float,
    omit_aas: str,
) -> dict[str, dict[str, Any]]:
    normalized: dict[str, dict[str, Any]] = {}
    for engine, row in (sequence_by_engine or {}).items():
        if not isinstance(row, dict):
            continue
        level_row = dict(row)
        level_row["sequences_per_structure"] = max(1, int(sequences_per_structure))
        level_row["sampling_temp"] = float(sampling_temp)
        level_row["omit_aas"] = _normalize_omit_aas(omit_aas)
        normalized[str(engine)] = level_row
    return normalized


def _normalize_omit_aas(value: object, default: str = "CX") -> str:
    text = str(value or default).upper()
    letters = "".join(char for char in text if "A" <= char <= "Z")
    return letters or default


def _generator_only_engine_config(
    engine: str,
    config: dict[str, Any] | None,
    *,
    binder_length: str,
) -> dict[str, Any]:
    """Normalize staged/scout generator configs to the capacity-runner contract."""
    normalized = {
        "campaign_workflow_recipe": "staged_backbone_sequence_refold",
        "binder_length": str(binder_length),
        **dict(config or {}),
    }
    if engine == "genie3":
        normalized.setdefault("cond_strategy", "extended")
        normalized.setdefault("direction_scale", 0.0)
    if engine == "proteina_complexa":
        normalized["replicas"] = 1
        normalized["batch_size"] = 1
    return normalized


def _target_has_residue_gaps(path: Path, chains: list[str]) -> bool:
    selected = set(chains or [])
    residues_by_chain: dict[str, list[int]] = {}
    last_seen: dict[str, tuple[int, str] | None] = {}
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
            continue
        chain = line[21].strip() or "_"
        if selected and chain not in selected:
            continue
        try:
            residue = int(line[22:26])
        except ValueError:
            continue
        residue_key = (residue, line[26].strip())
        if last_seen.get(chain) == residue_key:
            continue
        residues_by_chain.setdefault(chain, []).append(residue)
        last_seen[chain] = residue_key
    return any(
        any(next_residue > residue + 1 for residue, next_residue in zip(residues, residues[1:]))
        for residues in residues_by_chain.values()
    )


def _restrict_fragmented_target_refolders(config: dict[str, Any], *, target_pdb: Path, target_chains: list[str]) -> dict[str, Any]:
    if not _target_has_residue_gaps(target_pdb, target_chains):
        return config
    allowed = {"AF2-IG", "Boltz-2"}
    config = dict(config)
    staged = config.get("staged_two_step") if isinstance(config.get("staged_two_step"), dict) else None
    if staged:
        staged = dict(staged)
        for key in ("screen_refolder", "second_screen_refolder"):
            if str(staged.get(key) or "AF2-IG") not in allowed:
                staged[key] = "AF2-IG"
        config["staged_two_step"] = staged
    return config


def _restrict_fragmented_target_evaluation(config: dict[str, Any], *, target_pdb: Path, target_chains: list[str]) -> dict[str, Any]:
    if not _target_has_residue_gaps(target_pdb, target_chains):
        return config
    allowed = {"AF2-IG", "Boltz-2"}
    config = dict(config)
    if str(config.get("refolder") or "AF2-IG") not in allowed:
        config["refolder"] = "AF2-IG"
    refolders = [str(refolder) for refolder in config.get("refolders") or [] if str(refolder) in allowed]
    config["refolders"] = refolders or [str(config.get("refolder") or "AF2-IG")]
    return config
REPO_ROOT = Path(__file__).resolve().parents[2]
PYROSETTA_METRICS_SCRIPT = Path("/opt/de_novo_binder_scoring/scripts/compute_rosetta_metrics.py")
PYROSETTA_METRICS_WORKDIR = Path("/opt/de_novo_binder_scoring")
PYROSETTA_METRICS_IMAGE = "mn-protein-scoring-pyrosetta:latest"
COMMON_AF2_THRESHOLDS = {
    "model_1_binder_plddt": (80.0, "higher"),
    "model_2_binder_plddt": (80.0, "higher"),
    "model_1_ptm": (0.55, "higher"),
    "model_2_ptm": (0.55, "higher"),
    "model_1_iptm": (0.50, "higher"),
    "model_2_iptm": (0.50, "higher"),
    "model_1_ipae": (0.35 * 31.0, "lower"),
    "model_2_ipae": (0.35 * 31.0, "lower"),
    "model_1_binder_only_plddt": (80.0, "higher"),
    "model_2_binder_only_plddt": (80.0, "higher"),
    "model_1_binder_rmsd": (3.5, "lower"),
    "model_2_binder_rmsd": (3.5, "lower"),
}
COMMON_ROSETTA_THRESHOLDS = {
    "rosetta_binder_score": (0.0, "lower"),
    "rosetta_surface_hydrophobicity": (0.35, "lower"),
    "rosetta_interface_sc": (0.55, "higher"),
    "rosetta_interface_dG": (0.0, "lower"),
    "rosetta_interface_dSASA": (1.0, "higher"),
    "rosetta_interface_nres_binder": (7.0, "higher"),
    "rosetta_interface_interface_hbonds": (3.0, "higher"),
    "rosetta_interface_unsat_hbonds": (4.0, "lower"),
    "rosetta_interface_aa_LYS": (3.0, "lower"),
    "rosetta_interface_aa_MET": (3.0, "lower"),
}
COMMON_STRUCTURE_THRESHOLDS = {
    "average_binder_loop_percent": (90.0, "lower"),
    "model_1_binder_loop_percent": (90.0, "lower"),
    "model_2_binder_loop_percent": (90.0, "lower"),
    "average_hotspot_rmsd": (6.0, "lower"),
    "model_1_hotspot_rmsd": (6.0, "lower"),
    "model_2_hotspot_rmsd": (6.0, "lower"),
}
CANDIDATE_POOL_COVERAGE = {
    "template_redesign": "uploaded target-binder complex before sequence refinement",
    "rfdiffusion_classic": "normalized outputs retained by the app pipeline",
    "bindcraft": "emitted designs with native filters disabled by default",
    "rfdiffusion3_foundry": "RF3 native survivors plus RFdiffusion3/Foundry-MPNN prefilter structures",
    "boltzgen": "final ranked designs plus intermediate generated BoltzGen structures",
    "pxdesign": "PXDesign summary rows plus raw inference prediction CIFs",
    "genie3": "native evaluation rows plus generated Genie3 PDBs",
    "esmfold2_binder_design": "all completed optimization starts",
    "protpardelle_1c": "native ESMFold/MPNN rows plus raw scaffold samples",
    "proteina_complexa": "ranked evaluation rows plus generated inference complexes",
}
REFINEMENT_MODEL_TYPES = {
    "ProteinMPNN": "protein_mpnn",
    "LigandMPNN": "ligand_mpnn",
    "Soluble ProteinMPNN": "soluble_mpnn",
    "protein_mpnn": "protein_mpnn",
    "ligand_mpnn": "ligand_mpnn",
    "soluble_mpnn": "soluble_mpnn",
}
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
    "MSE": "M",
}


def _number(value: object, default: float = math.inf) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _first_metric(metrics: dict[str, Any], names: tuple[str, ...], default: float = math.inf) -> float:
    for name in names:
        if name in metrics and metrics[name] not in {None, ""}:
            return _number(metrics[name], default)
    return default


def _native_pass(candidate: dict[str, Any]) -> bool | None:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    for name in ("native_pass_filters", "pass_filters", "passes_filters", "passed_filters"):
        value = metrics.get(name)
        if isinstance(value, bool):
            return value
        if str(value).strip().lower() in {"true", "pass", "passed", "1", "yes"}:
            return True
        if str(value).strip().lower() in {"false", "fail", "failed", "0", "no"}:
            return False
    return None


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    passed = _native_pass(candidate)
    native_rank = _first_metric(
        metrics,
        (
            "native_final_rank",
            "native_design_rank",
            "esmfold2_screening_rank",
            "design_rank",
            "rank",
            "native_design_index",
        ),
    )
    confidence = _first_metric(
        metrics,
        ("analysis_score", "iptm", "ipTM", "confidence", "binder_plddt", "plddt", "ipsae", "ipSAE"),
        default=-math.inf,
    )
    interface_error = _first_metric(
        metrics,
        ("ipae", "iPAE", "interface_pae", "binder_rmsd"),
    )
    return (
        0.0 if passed is True else 1.0 if passed is None else 2.0,
        native_rank,
        -confidence,
        interface_error,
    )


def _absolute_candidate_paths(candidate: dict[str, Any], child_run: Path) -> dict[str, Any]:
    normalized = dict(candidate)
    for key in ("target_pdb", "complex_pdb", "binder_pdb"):
        resolved = resolve_stored_path(normalized.get(key), run_dir=child_run, must_exist=True)
        if resolved is not None:
            normalized[key] = str(resolved)
    raw_metadata = dict(normalized.get("raw_metadata") or {})
    for key, value in list(raw_metadata.items()):
        if not key.endswith(("_path", "_pdb", "_json", "_npz")):
            continue
        resolved = resolve_stored_path(value, run_dir=child_run, must_exist=True)
        if resolved is not None:
            raw_metadata[key] = str(resolved)
    normalized["raw_metadata"] = raw_metadata
    return normalized


def _bindcraft_design_parameter_family(child_run: Path) -> str:
    settings_path = child_run / "artifacts" / "raw" / "bindcraft" / "settings_advanced.json"
    settings = read_json(settings_path) if settings_path.exists() else {}
    if "use_multimer_design" in settings:
        return "multimer_v3" if bool(settings.get("use_multimer_design")) else "ptm"
    payload = read_json(child_run / "input.json")
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    settings_name = str(params.get("advanced_settings_file") or "").lower()
    if settings_name:
        return "multimer_v3" if "multimer" in settings_name else "ptm"
    return "multimer_v3"


def _bindcraft_generator_settings_file(settings_file: str) -> str:
    text = str(settings_file or "default_4stage_multimer.json")
    if "_mpnn" not in text:
        return text
    return text.replace("_mpnn", "")


def _candidate_design_parameter_family(
    candidate: dict[str, Any],
    *,
    engine: str,
    child_run: Path,
) -> str:
    metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    explicit = str(
        metadata.get("design_af2_parameter_family")
        or metrics.get("design_af2_parameter_family")
        or ""
    ).strip().lower()
    if explicit in {"multimer", "multimer_v3", "af2_multimer", "af2_multimer_v3"}:
        return "multimer_v3"
    if explicit in {"ptm", "af2_ptm", "monomer_ptm"}:
        return "ptm"
    if engine == "bindcraft":
        return _bindcraft_design_parameter_family(child_run)
    return "none"


def _validation_parameter_family(design_family: str) -> str:
    _ = design_family
    return "multimer_v3"


def _safe_candidate_token(value: object) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "")).strip("_")
    return token or "candidate"


def _campaign_candidate_id(
    *,
    engine: str,
    source_candidate_id: str,
    child_run: Path,
    local_index: int,
) -> str:
    source_token = _safe_candidate_token(source_candidate_id)
    run_token = _safe_candidate_token(child_run.name)
    return f"{_safe_candidate_token(engine)}__{source_token}__{run_token}_{int(local_index):03d}"


def _all_child_candidates(child_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for child in child_runs:
        engine = str(child["engine"])
        child_run = Path(str(child["run_dir"]))
        for index, candidate in enumerate(read_candidates(child_run), start=1):
            candidate = _absolute_candidate_paths(candidate, child_run)
            source_id = str(candidate.get("candidate_id") or f"candidate-{index}")
            metadata = dict(candidate.get("raw_metadata") or {})
            design_family = _candidate_design_parameter_family(
                candidate,
                engine=engine,
                child_run=child_run,
            )
            metadata.update(
                {
                    "design_campaign_engine": engine,
                    "design_af2_parameter_family": design_family,
                    "common_validation_af2_parameter_family": _validation_parameter_family(
                        design_family
                    ),
                    "native_source_run_dir": str(child_run),
                    "native_source_run_id": child_run.name,
                    "native_source_candidate_id": source_id,
                    "campaign_source_candidate_id": source_id,
                    "native_pass": _native_pass(candidate),
                    "candidate_pool_coverage": CANDIDATE_POOL_COVERAGE.get(engine, "native emitted candidates"),
                }
            )
            if not metadata.get("design_reference_pdb") and candidate.get("complex_pdb"):
                metadata["design_reference_pdb"] = candidate.get("complex_pdb")
                metadata["design_reference_kind"] = "native_emitted_complex"
            candidate.update(
                {
                    "candidate_id": _campaign_candidate_id(
                        engine=engine,
                        source_candidate_id=source_id,
                        child_run=child_run,
                        local_index=index,
                    ),
                    "raw_metadata": metadata,
                }
            )
            candidates.append(candidate)
    return candidates


def _write_candidate_jsonl(path: Path, candidates: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(store_managed_paths(candidate), sort_keys=True) + "\n" for candidate in candidates)
    )


def _threshold_failures(
    metrics: dict[str, Any],
    thresholds: dict[str, tuple[float, str]],
    *,
    missing_is_failure: bool = True,
) -> list[str]:
    failures: list[str] = []
    for metric, (threshold, direction) in thresholds.items():
        value = metrics.get(metric)
        if value in {None, ""}:
            if missing_is_failure:
                failures.append(f"{metric} missing")
            continue
        number = _number(value)
        if not math.isfinite(number):
            failures.append(f"{metric} invalid")
        elif direction == "higher" and number < threshold:
            failures.append(f"{metric} < {threshold:g}")
        elif direction == "lower" and number > threshold:
            failures.append(f"{metric} > {threshold:g}")
    return failures


def _docker_mounts_for_path(path: Path) -> list[str]:
    mounts = ["-v", f"{REPO_ROOT}:{REPO_ROOT}"]
    try:
        path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError:
        mounts.extend(["-v", f"{runs_root().resolve()}:{runs_root().resolve()}"])
    return mounts


def _pdb_chain_sequences(path: Path, chains: list[str]) -> dict[str, str]:
    keep = set(chains)
    residues: dict[str, list[tuple[tuple[int, str], str]]] = {}
    seen: set[tuple[str, int, str]] = set()
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
            continue
        chain = line[21].strip() or "_"
        if keep and chain not in keep:
            continue
        try:
            residue_number = int(line[22:26])
        except ValueError:
            continue
        insertion_code = line[26].strip()
        key = (chain, residue_number, insertion_code)
        if key in seen:
            continue
        seen.add(key)
        aa = AA3_TO_1.get(line[17:20].strip().upper(), "X")
        residues.setdefault(chain, []).append(((residue_number, insertion_code), aa))
    return {
        chain: "".join(aa for _, aa in sorted(values, key=lambda item: item[0]))
        for chain, values in residues.items()
    }


def _pdb_residue_tags(path: Path, chains: list[str]) -> list[str]:
    keep = set(chains)
    tags: list[str] = []
    seen: set[tuple[str, int, str]] = set()
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
            continue
        chain = line[21].strip() or "_"
        if keep and chain not in keep:
            continue
        try:
            residue_number = int(line[22:26])
        except ValueError:
            continue
        insertion_code = line[26].strip()
        key = (chain, residue_number, insertion_code)
        if key in seen:
            continue
        seen.add(key)
        tags.append(f"{chain}{residue_number}{insertion_code}")
    return tags


def _parse_residue_tags(text: object, *, default_chain: str = "") -> list[str]:
    tags: list[str] = []
    seen: set[str] = set()
    for token in str(text or "").replace(";", ",").replace(" ", ",").split(","):
        token = token.strip().replace(":", "")
        if not token:
            continue
        chain = token[0].upper() if token[0].isalpha() else default_chain
        residue = token[1:] if token and token[0].isalpha() else token
        residue = residue.strip()
        if not chain or not residue:
            continue
        tag = f"{chain}{residue}"
        if tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


def _create_template_redesign_source(
    run_dir: Path,
    *,
    target_pdb: Path,
    template_config: dict[str, Any],
) -> dict[str, Any] | None:
    if not bool(template_config.get("enabled")):
        return None
    source_pdb = Path(str(template_config.get("complex_pdb") or "")).expanduser()
    if not source_pdb.exists():
        raise FileNotFoundError(f"Template complex PDB does not exist: {source_pdb}")
    target_chains = [str(chain) for chain in template_config.get("target_chains") or []]
    binder_chains = [str(chain) for chain in template_config.get("binder_chains") or []]
    if not target_chains:
        raise ValueError("Template redesign requires at least one template target chain.")
    if not binder_chains:
        raise ValueError("Template redesign requires at least one template binder chain.")

    source_dir = run_dir / "artifacts" / "design_campaign" / "template_redesign" / "source_run"
    complex_dir = source_dir / "artifacts" / "template_redesign"
    complex_dir.mkdir(parents=True, exist_ok=True)
    copied_complex = complex_dir / "template_complex.pdb"
    role_metadata = _stage_ligandmpnn_role_normalized_input(
        source_pdb=source_pdb,
        staged_pdb=copied_complex,
        candidate={
            "candidate_id": "template_redesign_00001",
            "binder_chains": binder_chains,
            "target_chains": target_chains,
        },
        requested_design_chains="",
    )
    source_binder_chains = binder_chains
    source_target_chains = target_chains
    binder_chains = list(role_metadata.get("binder_chains") or binder_chains)
    target_chains = list(role_metadata.get("target_chains") or target_chains)
    sequences = _pdb_chain_sequences(copied_complex, binder_chains)
    binder_sequence = ":".join(sequences.get(chain, "") for chain in binder_chains)
    residue_tags = _pdb_residue_tags(copied_complex, binder_chains)
    locked_residues = _remap_fixed_residues_to_engine_chains(
        _parse_residue_tags(template_config.get("locked_residues"), default_chain=source_binder_chains[0] if source_binder_chains else ""),
        role_metadata,
    )
    unlocked_residues = _remap_fixed_residues_to_engine_chains(
        _parse_residue_tags(template_config.get("unlocked_residues"), default_chain=source_binder_chains[0] if source_binder_chains else ""),
        role_metadata,
    )
    candidate = {
        "candidate_id": "template_redesign_00001",
        "stage": STAGE_COMPLEX_REFOLDING,
        "source_tool": "template_redesign",
        "target_pdb": str(target_pdb),
        "complex_pdb": str(copied_complex.relative_to(source_dir)),
        "binder_sequence": binder_sequence,
        "target_chains": target_chains,
        "binder_chains": binder_chains,
        "binder_length": str(sum(len(sequence) for sequence in sequences.values())),
        "metrics": {
            "native_final_rank": 1,
            "native_design_rank": 1,
            "native_pass_filters": True,
            "pass_filters": True,
            "candidate_pool_level": "template_input",
            "template_binder_residue_count": len(residue_tags),
        },
        "raw_metadata": {
            "result_kind": "template_redesign",
            "template_source_pdb": str(source_pdb),
            "template_mode": str(template_config.get("mode") or "interface"),
            "template_locked_residues": ",".join(locked_residues),
            "template_unlocked_residues": ",".join(unlocked_residues),
            "candidate_pool_level": "template_input",
            "design_reference_pdb": str(copied_complex),
            "design_reference_kind": "template_complex",
            "input_binder_chains": binder_chains,
            "input_target_chains": target_chains,
            "source_binder_chains": source_binder_chains,
            "source_target_chains": source_target_chains,
            "chain_role_schema": role_metadata.get("chain_role_schema"),
            "chain_roles": role_metadata.get("chain_roles"),
            "template_binder_residues": residue_tags,
        },
    }
    write_candidates(source_dir, "template_redesign", [candidate])
    write_json(
        source_dir / "result.json",
        {
            "success": True,
            "metrics": {"candidate_count": 1, "engine": "template_redesign"},
            "outputs": {"candidates": "artifacts/normalized_candidates/candidates.jsonl"},
        },
    )
    return {
        "engine": "template_redesign",
        "run_dir": str(source_dir),
        "run_id": source_dir.name,
        "status": "completed",
        "error": "",
    }


def _run_common_pyrosetta(
    run_dir: Path,
    candidates: list[dict[str, Any]],
    *,
    nprocs: int,
) -> tuple[dict[str, dict[str, Any]], Path, int]:
    scoring_dir = run_dir / "artifacts" / "design_campaign" / "common_validation" / "pyrosetta_inputs"
    relaxed_dir = run_dir / "artifacts" / "design_campaign" / "common_validation" / "pyrosetta_relaxed"
    output_csv = run_dir / "artifacts" / "design_campaign" / "common_validation" / "pyrosetta_metrics.csv"
    scoring_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        structure = Path(str(candidate.get("complex_pdb") or ""))
        if not structure.exists() or structure.suffix.lower() != ".pdb":
            continue
        safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(candidate["candidate_id"]))
        staged_pdb = scoring_dir / f"{safe_id}.pdb"
        role_metadata = _stage_ligandmpnn_role_normalized_input(
            source_pdb=structure,
            staged_pdb=staged_pdb,
            candidate=candidate,
            requested_design_chains="",
        )
        metadata = candidate.setdefault("raw_metadata", {})
        metadata.setdefault("common_pyrosetta_chain_role_schema", role_metadata.get("chain_role_schema"))
        metadata.setdefault("common_pyrosetta_chain_roles", role_metadata.get("chain_roles"))
    if not list(scoring_dir.glob("*.pdb")):
        return {}, output_csv, 0
    command = [
        "docker",
        "run",
        "--rm",
        *_docker_mounts_for_path(run_dir),
        "-w",
        str(PYROSETTA_METRICS_WORKDIR),
        PYROSETTA_METRICS_IMAGE,
        "python",
        str(PYROSETTA_METRICS_SCRIPT),
        "--folder",
        f"common:{scoring_dir}",
        "--out-csv",
        str(output_csv),
        "--relaxed-dir",
        str(relaxed_dir),
        "--binder-chain",
        "Z",
        "--target-chain",
        "A",
        "--nprocs",
        str(max(1, int(nprocs))),
        "--dalphaball-path",
        str(PYROSETTA_METRICS_WORKDIR / "functions" / "DAlphaBall.gcc"),
    ]
    with (run_dir / "stdout.log").open("a") as stdout, (run_dir / "stderr.log").open("a") as stderr:
        stdout.write(f"$ {' '.join(command)}\n")
        stdout.flush()
        command = apply_docker_cpu_limit(command, run_dir)
        rc = subprocess.run(command, stdout=stdout, stderr=stderr, check=False).returncode
    rows: dict[str, dict[str, Any]] = {}
    if output_csv.exists():
        with output_csv.open(newline="") as handle:
            for row in csv.DictReader(handle):
                binder_id = str(row.get("binder_id") or "")
                metrics = {
                    key.removeprefix("common_"): value
                    for key, value in row.items()
                    if key != "binder_id" and value not in {None, ""}
                }
                rows[binder_id] = metrics
    return rows, output_csv, int(rc)


def _secondary_structure_metrics(
    path: Path,
    *,
    binder_chain: str = "A",
    target_chains: list[str] | None = None,
) -> dict[str, float]:
    from Bio.PDB import DSSP, PDBParser

    lines = [
        "HEADER    MN PROTEIN DESIGN DSSP INPUT",
        "CRYST1  500.000  500.000  500.000  90.00  90.00  90.00 P 1           1",
    ]
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith(("ATOM  ", "HETATM", "TER   ")):
            lines.append(line)
    lines.append("END")
    with tempfile.NamedTemporaryFile("w", suffix=".pdb") as handle:
        handle.write("\n".join(lines) + "\n")
        handle.flush()
        structure = PDBParser(QUIET=True).get_structure(
            "common_validation", handle.name
        )
        model = next(structure.get_models())
        dssp = DSSP(model, handle.name, dssp="mkdssp")
    targets = target_chains or [chain.id for chain in model if chain.id != binder_chain]
    interface = set(
        _binder_interface_residues(path, [binder_chain], targets, cutoff=4.0)
    )
    binder_counts = {"helix": 0, "sheet": 0, "loop": 0}
    interface_counts = {"helix": 0, "sheet": 0, "loop": 0}
    for key in dssp.keys():
        chain, residue_id = key
        if chain != binder_chain:
            continue
        code = str(dssp[key][2])
        kind = "helix" if code in {"H", "G", "I"} else "sheet" if code == "E" else "loop"
        binder_counts[kind] += 1
        residue_tag = f"{chain}{residue_id[1]}{str(residue_id[2]).strip()}"
        if residue_tag in interface:
            interface_counts[kind] += 1

    def percentages(counts: dict[str, int]) -> dict[str, float]:
        total = sum(counts.values())
        return {
            kind: round(100.0 * count / total, 2) if total else 0.0
            for kind, count in counts.items()
        }

    binder = percentages(binder_counts)
    interface_ss = percentages(interface_counts)
    return {
        "binder_helix_percent": binder["helix"],
        "binder_betasheet_percent": binder["sheet"],
        "binder_loop_percent": binder["loop"],
        "interface_helix_percent": interface_ss["helix"],
        "interface_betasheet_percent": interface_ss["sheet"],
        "interface_loop_percent": interface_ss["loop"],
    }


def _chain_ca_coordinates(path: Path, chains: list[str]) -> list[tuple[float, float, float]]:
    keep = set(chains)
    coordinates: list[tuple[float, float, float]] = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
            continue
        if (line[21].strip() or "_") not in keep:
            continue
        altloc = line[16].strip()
        if altloc not in {"", "A"}:
            continue
        try:
            coordinates.append(
                (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            )
        except ValueError:
            continue
    return coordinates


def _unaligned_ca_rmsd(
    reference: Path,
    prediction: Path,
    *,
    reference_chains: list[str],
    prediction_chains: list[str],
) -> float | None:
    reference_coords = _chain_ca_coordinates(reference, reference_chains)
    prediction_coords = _chain_ca_coordinates(prediction, prediction_chains)
    count = min(len(reference_coords), len(prediction_coords))
    if not count:
        return None
    squared = sum(
        sum(
            (reference_coords[index][axis] - prediction_coords[index][axis]) ** 2
            for axis in range(3)
        )
        for index in range(count)
    )
    return math.sqrt(squared / count)


def _common_prediction_model_paths(candidate: dict[str, Any]) -> list[Path]:
    raw = candidate.get("raw_metadata") or {}
    prediction_dir = Path(str(raw.get("prediction_dir") or ""))
    if not prediction_dir.exists():
        complex_path = Path(str(candidate.get("complex_pdb") or ""))
        prediction_dir = complex_path.parent
    safe_id = str((candidate.get("metrics") or {}).get("id") or "")
    paths: list[Path] = []
    for model_number in (1, 2):
        matches = sorted(
            path
            for path in prediction_dir.glob(f"{safe_id}_*_model{model_number}.pdb")
            if "_binder_model" not in path.name
        )
        if not matches:
            matches = sorted(
                path
                for path in prediction_dir.glob(f"*model{model_number}.pdb")
                if safe_id in path.name and "_binder_model" not in path.name
            )
        if matches:
            paths.append(matches[0])
    return paths


def _candidate_role_chains_for_structure(
    candidate: dict[str, Any],
    structure: Path,
) -> tuple[list[str], list[str]]:
    """Resolve prediction-chain roles without assuming the historical A/B layout."""
    raw = (
        candidate.get("raw_metadata")
        if isinstance(candidate.get("raw_metadata"), dict)
        else {}
    )
    available = (
        _chain_ids_from_pdb(structure)
        if structure.exists() and structure.suffix.lower() == ".pdb"
        else []
    )
    available_set = set(available)

    role_payload = candidate.get("chain_roles")
    if not isinstance(role_payload, dict):
        role_payload = raw.get("chain_roles")
    role_payload = role_payload if isinstance(role_payload, dict) else {}
    roles = role_payload.get("roles")
    roles = roles if isinstance(roles, dict) else {}
    binder_chains = [str(chain)[:1] for chain, role in roles.items() if role == "binder"]
    target_chains = [str(chain)[:1] for chain, role in roles.items() if role == "target"]

    if not binder_chains:
        binder_chains = [
            str(chain)[:1]
            for chain in (
                candidate.get("binder_chains")
                or raw.get("input_binder_chains")
                or raw.get("binder_source_chains")
                or []
            )
            if str(chain).strip()
        ]
    if not target_chains:
        target_chains = [
            str(chain)[:1]
            for chain in (
                candidate.get("target_chains")
                or raw.get("input_target_chains")
                or raw.get("target_source_chains")
                or []
            )
            if str(chain).strip()
        ]

    if available_set:
        binder_chains = [chain for chain in binder_chains if chain in available_set]
        target_chains = [chain for chain in target_chains if chain in available_set]
    if not binder_chains and available:
        binder_chains = [available[0]]
    if not target_chains and available:
        binder_set = set(binder_chains)
        target_chains = [chain for chain in available if chain not in binder_set]
    return binder_chains, target_chains


def _add_bindcraft_structure_metrics(candidate: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(candidate.get("metrics") or {})
    raw = candidate.get("raw_metadata") or {}
    reference_text = str(raw.get("design_reference_pdb") or "").strip()
    reference_path = Path(reference_text) if reference_text else None
    reference_chains = [
        str(chain)
        for chain in raw.get("input_binder_chains") or candidate.get("binder_chains") or ["A"]
    ]
    model_paths = _common_prediction_model_paths(candidate)
    structure_errors: list[str] = []
    for model_number, model_path in enumerate(model_paths, start=1):
        prediction_binder_chains, prediction_target_chains = (
            _candidate_role_chains_for_structure(candidate, model_path)
        )
        prediction_binder_chain = (
            prediction_binder_chains[0] if prediction_binder_chains else "A"
        )
        try:
            ss_metrics = _secondary_structure_metrics(
                model_path,
                binder_chain=prediction_binder_chain,
                target_chains=prediction_target_chains,
            )
            metrics.update(
                {
                    f"model_{model_number}_{key}": value
                    for key, value in ss_metrics.items()
                }
            )
        except Exception as exc:
            structure_errors.append(
                f"model {model_number} DSSP: {type(exc).__name__}: {exc}"
            )
        if reference_path is not None and reference_path.exists():
            rmsd = _unaligned_ca_rmsd(
                reference_path,
                model_path,
                reference_chains=reference_chains,
                prediction_chains=prediction_binder_chains
                or [prediction_binder_chain],
            )
            metrics[f"model_{model_number}_hotspot_rmsd"] = rmsd

    averaged_names = (
        "binder_helix_percent",
        "binder_betasheet_percent",
        "binder_loop_percent",
        "interface_helix_percent",
        "interface_betasheet_percent",
        "interface_loop_percent",
        "hotspot_rmsd",
    )
    for metric_name in averaged_names:
        values = [
            metrics.get(f"model_{model_number}_{metric_name}")
            for model_number in range(1, len(model_paths) + 1)
        ]
        numeric = [float(value) for value in values if value is not None]
        metrics[f"average_{metric_name}"] = (
            sum(numeric) / len(numeric) if numeric else None
        )
    metrics.update(
        {
            "bindcraft_structure_metric_model_count": len(model_paths),
            "hotspot_rmsd_reference_available": bool(
                reference_path is not None and reference_path.exists()
            ),
            "hotspot_rmsd_reference_kind": str(raw.get("design_reference_kind") or ""),
            "bindcraft_structure_metric_errors": "; ".join(structure_errors),
        }
    )
    candidate["metrics"] = metrics
    return candidate


def run_common_bindcraft_validation(
    run_dir: Path,
    child_runs: list[dict[str, Any]],
    *,
    gpu_device: str,
    num_recycles: int,
    min_ipsae: float,
    pyrosetta_nprocs: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    native_candidates = _all_child_candidates(child_runs)
    pool_level_counts: dict[str, int] = {}
    for candidate in native_candidates:
        pool_level = str(
            (candidate.get("metrics") or {}).get("candidate_pool_level")
            or (candidate.get("raw_metadata") or {}).get("candidate_pool_level")
            or "unspecified"
        )
        pool_level_counts[pool_level] = pool_level_counts.get(pool_level, 0) + 1
    validation_dir = run_dir / "artifacts" / "design_campaign" / "common_validation"
    source_jsonl = validation_dir / "native_candidates.jsonl"
    _write_candidate_jsonl(source_jsonl, native_candidates)
    if not native_candidates:
        return [], {"native_candidate_count": 0}

    validation_groups = {"multimer_v3": native_candidates}
    scored: list[dict[str, Any]] = []
    af2_child_runs: dict[str, str] = {}
    for family, family_candidates in validation_groups.items():
        if not family_candidates:
            continue
        family_source_jsonl = validation_dir / f"native_candidates_{family}.jsonl"
        _write_candidate_jsonl(family_source_jsonl, family_candidates)
        update_status(
            run_dir,
            "running",
            current_phase="Common validation",
            current_engine=f"AF2 {family} target-template",
            progress_label=(
                f"Running harmonized {family} complex validation for "
                f"{len(family_candidates)} candidate(s)"
            ),
        )
        af2_run = run_af2_initial_guess_complex_refolding(
            source_run_dir=run_dir,
            candidates_jsonl=family_source_jsonl,
            require_monomer_success=False,
            num_recycles=max(1, int(num_recycles)),
            model_count=2,
            multimer=True,
            binder_multimer=False,
            use_initial_guess=False,
            use_binder_template=False,
            use_interface_template=False,
            gpu_device=gpu_device,
        )
        af2_child_runs[family] = str(af2_run)
        mark_internal_job(
            af2_run,
            parent_run_dir=run_dir,
            parent_task_group=DESIGN_CAMPAIGN_GROUP,
            role="common_af2_target_template_validation",
            engine=f"af2_{family}_target_template",
        )
        af2_candidates = [
            _absolute_candidate_paths(candidate, af2_run)
            for candidate in read_candidates(af2_run)
        ]
        update_status(
            run_dir,
            "running",
            current_phase="Common validation",
            current_engine=f"ipSAE ({family})",
            progress_label=f"Calculating ipSAE for {family} predictions",
        )
        for candidate in _run_ipsae(af2_run, run_dir, af2_candidates):
            metrics = dict(candidate.get("metrics") or {})
            metadata = (
                candidate.get("raw_metadata")
                if isinstance(candidate.get("raw_metadata"), dict)
                else {}
            )
            metrics.update(
                {
                    "design_af2_parameter_family": metadata.get(
                        "design_af2_parameter_family",
                        "none",
                    ),
                    "af2_validation_parameter_family": family,
                    "complex_af2_parameter_family": "multimer_v3",
                    "binder_fold_af2_parameter_family": "ptm",
                    "af2_cross_validation_rule": (
                        "harmonized_multimer_complex_plus_ptm_binder_fold"
                    ),
                }
            )
            candidate["metrics"] = metrics
            scored.append(_add_bindcraft_structure_metrics(candidate))
    confidence_survivors: list[dict[str, Any]] = []
    for candidate in scored:
        metrics = dict(candidate.get("metrics") or {})
        failures = _threshold_failures(metrics, COMMON_AF2_THRESHOLDS)
        ipsae = metrics.get("ipsae")
        if min_ipsae > 0:
            if ipsae in {None, ""}:
                failures.append("ipsae missing")
            elif _number(ipsae) < min_ipsae:
                failures.append(f"ipsae < {min_ipsae:g}")
        metrics.update(
            {
                "common_confidence_pass": not failures,
                "common_confidence_failures": "; ".join(failures),
                "common_validation_profile": "harmonized_multimer_complex_ptm_binder",
            }
        )
        candidate["metrics"] = metrics
        if not failures:
            confidence_survivors.append(candidate)

    update_status(
        run_dir,
        "running",
        current_phase="Common validation",
        current_engine="PyRosetta",
        progress_label=f"Relaxing and scoring {len(confidence_survivors)} confidence survivors",
    )
    rosetta_rows, rosetta_csv, rosetta_rc = _run_common_pyrosetta(
        run_dir,
        confidence_survivors,
        nprocs=pyrosetta_nprocs,
    )
    validated: list[dict[str, Any]] = []
    for candidate in scored:
        metrics = dict(candidate.get("metrics") or {})
        safe_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(candidate["candidate_id"]))
        rosetta = rosetta_rows.get(safe_id, {})
        metrics.update({key: _number(value, default=value) for key, value in rosetta.items()})
        confidence_pass = metrics.get("common_confidence_pass") is True
        structure_failures = (
            _threshold_failures(metrics, COMMON_STRUCTURE_THRESHOLDS)
            if confidence_pass
            else ["skipped after common confidence gate"]
        )
        rosetta_failures = (
            _threshold_failures(metrics, COMMON_ROSETTA_THRESHOLDS)
            if confidence_pass
            else ["skipped after common confidence gate"]
        )
        common_pass = confidence_pass and not structure_failures and not rosetta_failures
        metrics.update(
            {
                "common_rosetta_pass": confidence_pass and not rosetta_failures,
                "common_rosetta_failures": "; ".join(rosetta_failures),
                "common_structure_pass": confidence_pass and not structure_failures,
                "common_structure_failures": "; ".join(structure_failures),
                "common_bindcraft_pass": common_pass,
                "common_filter_failures": "; ".join(
                    item
                    for item in [
                        str(metrics.get("common_confidence_failures") or ""),
                        "; ".join(structure_failures),
                        "; ".join(rosetta_failures),
                    ]
                    if item
                ),
                "common_bindcraft_parity_checks": (
                    "AF2-multimer-v3 target-binder complex confidence; "
                    "independent AF2-PTM binder-alone fold prediction; binder fold RMSD; "
                    "binder/interface secondary structure; target-aligned complex pose RMSD; "
                    "PyRosetta"
                ),
            }
        )
        candidate["metrics"] = metrics
        validated.append(candidate)
    _write_candidate_jsonl(validation_dir / "validated_candidates.jsonl", validated)
    write_json(
        validation_dir / "validation_summary.json",
        {
            "profile": "harmonized_multimer_complex_ptm_binder",
            "prediction_input_mode": "target_template_plus_binder_sequence",
            "native_candidate_count": len(native_candidates),
            "candidate_pool_level_counts": pool_level_counts,
            "af2_candidate_count": len(scored),
            "confidence_passing_count": len(confidence_survivors),
            "common_bindcraft_passing_count": sum(
                (candidate.get("metrics") or {}).get("common_bindcraft_pass") is True
                for candidate in validated
            ),
            "af2_thresholds": COMMON_AF2_THRESHOLDS,
            "structure_thresholds": COMMON_STRUCTURE_THRESHOLDS,
            "rosetta_thresholds": COMMON_ROSETTA_THRESHOLDS,
            "min_ipsae": min_ipsae,
            "pyrosetta_return_code": rosetta_rc,
            "pyrosetta_metrics_csv": str(rosetta_csv.relative_to(run_dir)) if rosetta_csv.exists() else "",
            "secondary_structure_scored_count": sum(
                (candidate.get("metrics") or {}).get("average_binder_loop_percent") is not None
                for candidate in validated
            ),
            "hotspot_rmsd_scored_count": sum(
                (candidate.get("metrics") or {}).get("average_hotspot_rmsd") is not None
                for candidate in validated
            ),
            "af2_child_runs": af2_child_runs,
            "validation_family_candidate_counts": {
                family: len(candidates)
                for family, candidates in validation_groups.items()
            },
        },
    )
    return validated, {
        "native_candidate_count": len(native_candidates),
        "candidate_pool_level_counts": pool_level_counts,
        "af2_candidate_count": len(scored),
        "confidence_passing_count": len(confidence_survivors),
        "common_bindcraft_passing_count": sum(
            (candidate.get("metrics") or {}).get("common_bindcraft_pass") is True
            for candidate in validated
        ),
        "secondary_structure_scored_count": sum(
            (candidate.get("metrics") or {}).get("average_binder_loop_percent") is not None
            for candidate in validated
        ),
        "hotspot_rmsd_scored_count": sum(
            (candidate.get("metrics") or {}).get("average_hotspot_rmsd") is not None
            for candidate in validated
        ),
        "af2_child_runs": af2_child_runs,
        "validation_family_candidate_counts": {
            family: len(candidates)
            for family, candidates in validation_groups.items()
        },
        "pyrosetta_return_code": rosetta_rc,
    }


def harmonize_child_candidates(
    child_runs: list[dict[str, Any]],
    *,
    survivors_per_engine: int,
    passing_only: bool,
    keep_best_failed: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    harmonized: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    survivor_limit = max(1, int(survivors_per_engine))
    by_engine: dict[str, list[tuple[dict[str, Any], dict[str, Any], Path]]] = {}
    children_by_engine: dict[str, list[dict[str, Any]]] = {}
    for child in child_runs:
        engine = str(child.get("engine") or "unknown")
        child_run = Path(str(child["run_dir"]))
        candidates = [_absolute_candidate_paths(row, child_run) for row in read_candidates(child_run)]
        children_by_engine.setdefault(engine, []).append(child)
        by_engine.setdefault(engine, []).extend((candidate, child, child_run) for candidate in candidates)

    for engine in [engine for engine in ENGINE_ORDER if engine in by_engine]:
        engine_records = by_engine.get(engine, [])
        engine_candidates = [record[0] for record in engine_records]
        passing = [row for row in engine_candidates if _native_pass(row) is True]
        if passing_only:
            eligible_records = [record for record in engine_records if _native_pass(record[0]) is True]
            selection_mode = "native_pass"
            if not eligible_records and keep_best_failed:
                eligible_records = engine_records
                selection_mode = "best_available_fallback"
        else:
            eligible_records = engine_records
            selection_mode = "all"
        selected_records = sorted(eligible_records, key=lambda record: _candidate_sort_key(record[0]))[:survivor_limit]

        for engine_rank, (candidate, child, child_run) in enumerate(selected_records, start=1):
            source_candidate_id = str(candidate.get("candidate_id") or f"candidate-{engine_rank}")
            metrics = dict(candidate.get("metrics") or {})
            metrics.update(
                {
                    "design_campaign_engine_rank": engine_rank,
                    "harmonized_native_pass": _native_pass(candidate),
                    "harmonized_selection_mode": selection_mode,
                }
            )
            raw_metadata = dict(candidate.get("raw_metadata") or {})
            raw_metadata.update(
                {
                    "design_campaign_engine": engine,
                    "source_run_dir": str(child_run),
                    "source_run_id": child_run.name,
                    "source_candidate_id": source_candidate_id,
                    "campaign_source_candidate_id": source_candidate_id,
                }
            )
            candidate.update(
                {
                    "candidate_id": _campaign_candidate_id(
                        engine=engine,
                        source_candidate_id=source_candidate_id,
                        child_run=child_run,
                        local_index=engine_rank,
                    ),
                    "metrics": metrics,
                    "raw_metadata": raw_metadata,
                    "parents": [*list(candidate.get("parents") or []), source_candidate_id],
                }
            )
            harmonized.append(candidate)

        engine_children = children_by_engine.get(engine, [])
        child_statuses = [str(child.get("status") or "") for child in engine_children]
        completed_children = sum(1 for status in child_statuses if status == "completed")
        if engine_children and completed_children == len(engine_children):
            status = "completed"
        elif completed_children:
            status = "partial"
        else:
            status = child_statuses[0] if child_statuses else ""
        summaries.append(
            {
                "engine": engine,
                "label": ENGINE_LABELS.get(engine, engine),
                "run_dir": str(Path(str(engine_children[0].get("run_dir")))) if engine_children else "",
                "run_id": str(engine_children[0].get("run_id") or "") if engine_children else "",
                "child_runs": len(engine_children),
                "child_run_ids": [str(child.get("run_id") or "") for child in engine_children if child.get("run_id")],
                "status": status,
                "candidate_count": len(engine_candidates),
                "native_passing_count": len(passing),
                "selected_count": len(selected_records),
                "selection_mode": selection_mode,
                "error": "; ".join(str(child.get("error") or "") for child in engine_children if child.get("error")),
            }
        )

    harmonized.sort(key=lambda row: (ENGINE_ORDER.index(str((row.get("raw_metadata") or {}).get("design_campaign_engine"))), _candidate_sort_key(row)))
    for campaign_rank, candidate in enumerate(harmonized, start=1):
        candidate.setdefault("metrics", {})["design_campaign_rank"] = campaign_rank
    return harmonized, summaries


def harmonize_validated_candidates(
    candidates: list[dict[str, Any]],
    *,
    survivors_per_engine: int,
    keep_best_failed: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    harmonized: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    survivor_limit = max(1, int(survivors_per_engine))
    by_engine: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        engine = str((candidate.get("raw_metadata") or {}).get("design_campaign_engine") or "unknown")
        by_engine.setdefault(engine, []).append(candidate)

    def common_key(candidate: dict[str, Any]) -> tuple[float, ...]:
        metrics = candidate.get("metrics") or {}
        return (
            0.0 if metrics.get("common_bindcraft_pass") is True else 1.0,
            -_first_metric(metrics, ("ipsae", "ipSAE", "iptm", "i_ptm"), default=-math.inf),
            _first_metric(metrics, ("ipae", "i_pae")),
            -_first_metric(metrics, ("binder_plddt", "plddt"), default=-math.inf),
            -_first_metric(metrics, ("rosetta_interface_sc",), default=-math.inf),
            _first_metric(metrics, ("rosetta_interface_dG",)),
        )

    for engine in [engine for engine in ENGINE_ORDER if engine in by_engine]:
        engine_candidates = by_engine.get(engine, [])
        passing = [
            candidate
            for candidate in engine_candidates
            if (candidate.get("metrics") or {}).get("common_bindcraft_pass") is True
        ]
        eligible = passing
        selection_mode = "common_bindcraft_pass"
        if not eligible and keep_best_failed:
            eligible = engine_candidates
            selection_mode = "best_common_validation_fallback"
        selected = sorted(eligible, key=common_key)[:survivor_limit]
        for engine_rank, candidate in enumerate(selected, start=1):
            metrics = dict(candidate.get("metrics") or {})
            metrics.update(
                {
                    "design_campaign_engine_rank": engine_rank,
                    "harmonized_selection_mode": selection_mode,
                }
            )
            candidate["metrics"] = metrics
            harmonized.append(candidate)
        summaries.append(
            {
                "engine": engine,
                "label": ENGINE_LABELS.get(engine, engine),
                "candidate_count": len(engine_candidates),
                "common_passing_count": len(passing),
                "selected_count": len(selected),
                "selection_mode": selection_mode,
                "candidate_pool_coverage": CANDIDATE_POOL_COVERAGE.get(engine, "native emitted candidates"),
            }
        )
    harmonized.sort(
        key=lambda row: (
            ENGINE_ORDER.index(str((row.get("raw_metadata") or {}).get("design_campaign_engine"))),
            common_key(row),
        )
    )
    for rank, candidate in enumerate(harmonized, start=1):
        candidate.setdefault("metrics", {})["design_campaign_rank"] = rank
    return harmonized, summaries


def _candidate_structure_path(run_dir: Path, candidate: dict[str, Any]) -> Path | None:
    value = candidate.get("complex_pdb") or candidate.get("binder_pdb")
    return resolve_stored_path(value, run_dir=run_dir)


def _candidate_refinement_pdb(
    run_dir: Path,
    candidate: dict[str, Any],
    refinement_dir: Path,
) -> Path | None:
    structure = _candidate_structure_path(run_dir, candidate)
    if structure is None or not structure.exists():
        return None
    if structure.suffix.lower() == ".pdb":
        return structure
    name = structure.name.lower()
    if structure.suffix.lower() not in {".cif", ".mmcif"} and not name.endswith((".cif.gz", ".mmcif.gz")):
        return None
    source_id = _safe_candidate_token(candidate.get("candidate_id") or structure.stem or "candidate")
    staged_dir = refinement_dir / "input_structures"
    staged_dir.mkdir(parents=True, exist_ok=True)
    staged_pdb = staged_dir / f"{source_id}.pdb"
    refolding_workflow._cif_to_pdb(structure, staged_pdb)
    return staged_pdb


def _candidate_with_refinement_structure(
    run_dir: Path,
    candidate: dict[str, Any],
    structure: Path,
) -> dict[str, Any]:
    try:
        relative = str(structure.relative_to(run_dir))
    except ValueError:
        relative = str(structure)
    if str(candidate.get("complex_pdb") or "") == relative:
        return candidate
    staged = dict(candidate)
    raw = dict(staged.get("raw_metadata") or {})
    raw.setdefault("pre_refinement_complex_pdb", candidate.get("complex_pdb"))
    staged["raw_metadata"] = raw
    staged["complex_pdb"] = relative
    return staged


def _annotate_campaign_chain_roles(run_dir: Path, candidate: dict[str, Any]) -> dict[str, Any]:
    if str(candidate.get("chain_role_schema") or "") == chain_roles.CHAIN_ROLE_SCHEMA_EXPLICIT_V2:
        return candidate
    structure = _candidate_structure_path(run_dir, candidate)
    binder_chains = [str(chain)[:1] for chain in candidate.get("binder_chains") or [] if str(chain)]
    target_chains = [str(chain)[:1] for chain in candidate.get("target_chains") or [] if str(chain)]
    if structure is not None and structure.exists() and structure.suffix.lower() == ".pdb":
        try:
            binder_chains, target_chains = _source_role_chains(candidate, structure)
        except Exception:
            pass
    if not binder_chains and not target_chains:
        return candidate
    role_map = chain_roles.build_explicit_role_map(
        binder_source_chains=binder_chains,
        binder_engine_chains=binder_chains,
        target_source_chains=target_chains,
        target_engine_chains=target_chains,
    )
    candidate["chain_role_schema"] = chain_roles.CHAIN_ROLE_SCHEMA_EXPLICIT_V2
    candidate["chain_roles"] = role_map.to_dict()
    raw = candidate.setdefault("raw_metadata", {})
    if isinstance(raw, dict):
        raw.setdefault("chain_role_schema", chain_roles.CHAIN_ROLE_SCHEMA_EXPLICIT_V2)
        raw.setdefault("chain_roles", role_map.to_dict())
        raw.setdefault("binder_source_chains", binder_chains)
        raw.setdefault("target_source_chains", target_chains)
    return candidate


def _binder_interface_residues(
    structure: Path,
    binder_chains: list[str],
    target_chains: list[str],
    *,
    cutoff: float = 5.0,
) -> list[str]:
    binder_set = set(binder_chains)
    target_set = set(target_chains)
    binder_atoms: list[tuple[str, tuple[float, float, float]]] = []
    target_atoms: list[tuple[float, float, float]] = []
    for line in structure.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        altloc = line[16].strip()
        if altloc not in {"", "A"}:
            continue
        element = line[76:78].strip().upper() if len(line) >= 78 else ""
        atom_name = line[12:16].strip().upper()
        if element == "H" or (not element and atom_name.startswith("H")):
            continue
        chain = line[21].strip() or "_"
        try:
            coords = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        if chain in binder_set:
            residue_number = line[22:26].strip()
            insertion_code = line[26].strip()
            if residue_number:
                binder_atoms.append((f"{chain}{residue_number}{insertion_code}", coords))
        elif chain in target_set:
            target_atoms.append(coords)
    if not binder_atoms or not target_atoms:
        return []

    cell_size = max(0.1, float(cutoff))
    target_grid: dict[tuple[int, int, int], list[tuple[float, float, float]]] = {}
    for coords in target_atoms:
        cell = tuple(math.floor(value / cell_size) for value in coords)
        target_grid.setdefault(cell, []).append(coords)
    cutoff_sq = cutoff * cutoff
    contacts: set[str] = set()
    for residue, coords in binder_atoms:
        cell = tuple(math.floor(value / cell_size) for value in coords)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for target in target_grid.get(
                        (cell[0] + dx, cell[1] + dy, cell[2] + dz), []
                    ):
                        if sum((coords[index] - target[index]) ** 2 for index in range(3)) <= cutoff_sq:
                            contacts.add(residue)
                            break
                    if residue in contacts:
                        break
                if residue in contacts:
                    break
            if residue in contacts:
                break
    return sorted(
        contacts,
        key=lambda tag: (
            tag[0],
            int("".join(ch for ch in tag[1:] if ch.isdigit()) or 0),
            tag,
        ),
    )


def _candidate_design_campaign_engine(candidate: dict[str, Any]) -> str:
    current: dict[str, Any] | None = candidate
    seen: set[int] = set()
    while isinstance(current, dict):
        marker = id(current)
        if marker in seen:
            break
        seen.add(marker)
        raw = current.get("raw_metadata") if isinstance(current.get("raw_metadata"), dict) else {}
        engine = raw.get("design_campaign_engine")
        if engine:
            return str(engine)
        for key in ("source_candidate", "pre_refinement_candidate"):
            source = raw.get(key)
            if isinstance(source, dict):
                current = source
                break
        else:
            break
    candidate_id = str(candidate.get("candidate_id") or "")
    for engine in sorted(ENGINE_ORDER, key=len, reverse=True):
        if candidate_id == engine or candidate_id.startswith(f"{engine}__") or candidate_id.startswith(f"{engine}_"):
            return engine
    return ""


def run_campaign_sequence_refinement(
    run_dir: Path,
    candidates: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    gpu_device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mode = str(config.get("mode") or "none")
    if mode == "none":
        return candidates, {"mode": "none", "status": "skipped", "candidate_count": len(candidates)}
    if mode not in {
        "redesign_non_interface",
        "redesign_all",
        "redesign_selected_unlocked",
        "redesign_selected_locked",
    }:
        raise ValueError(f"Unsupported sequence refinement mode: {mode}")

    refinement_dir = run_dir / "artifacts" / "design_campaign" / "sequence_refinement"
    artifact_label = _safe_candidate_token(config.get("artifact_label_prefix")) if config.get("artifact_label_prefix") else ""
    if artifact_label:
        refinement_dir = refinement_dir / artifact_label
    sequence_by_engine = (
        config.get("sequence_by_engine")
        if isinstance(config.get("sequence_by_engine"), dict)
        else {}
    )
    passthrough_disabled_sequence_engines = bool(config.get("passthrough_disabled_sequence_engines", False))
    enabled_sequence_engines = {
        str(engine)
        for engine, row in sequence_by_engine.items()
        if isinstance(row, dict) and bool(row.get("enabled"))
    }
    has_sequence_engine_filter = bool(sequence_by_engine)
    default_model_type = REFINEMENT_MODEL_TYPES.get(
        str(config.get("engine") or "protein_mpnn"), "protein_mpnn"
    )
    sequence_config_by_candidate: dict[str, dict[str, Any]] = {}
    fixed_residues: dict[str, list[str]] = {}
    eligible: list[dict[str, Any]] = []
    passthrough: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        candidate_engine = _candidate_design_campaign_engine(candidate)
        if has_sequence_engine_filter and candidate_engine not in enabled_sequence_engines:
            if passthrough_disabled_sequence_engines:
                passthrough.append(candidate)
                continue
            skipped.append({"candidate_id": candidate_id, "reason": "sequence generation disabled for design engine"})
            continue
        engine_sequence_config = (
            sequence_by_engine.get(candidate_engine)
            if isinstance(sequence_by_engine.get(candidate_engine), dict)
            else {}
        )
        sequence_config_by_candidate[candidate_id] = {
            "model_type": REFINEMENT_MODEL_TYPES.get(
                str(engine_sequence_config.get("engine") or config.get("engine") or "protein_mpnn"),
                default_model_type,
            ),
            "sequences_per_structure": max(
                1,
                int(engine_sequence_config.get("sequences_per_structure") or config.get("sequences_per_structure", 2)),
            ),
            "sampling_temp": float(engine_sequence_config.get("sampling_temp") or config.get("sampling_temp", 0.1)),
            "omit_aas": _normalize_omit_aas(engine_sequence_config.get("omit_aas") or config.get("omit_aas")),
        }
        structure = _candidate_refinement_pdb(run_dir, candidate, refinement_dir)
        if not candidate_id or structure is None or not structure.exists() or structure.suffix.lower() != ".pdb":
            skipped.append({"candidate_id": candidate_id, "reason": "missing PDB complex"})
            continue
        candidate = _candidate_with_refinement_structure(run_dir, candidate, structure)
        binder_chains, target_chains = _source_role_chains(candidate, structure)
        if not binder_chains:
            skipped.append({"candidate_id": candidate_id, "reason": "missing binder/design chains"})
            continue
        if not target_chains:
            skipped.append({"candidate_id": candidate_id, "reason": "missing target/reference chains"})
            continue
        if mode == "redesign_non_interface":
            contacts = _binder_interface_residues(structure, binder_chains, target_chains)
            if not contacts:
                skipped.append({"candidate_id": candidate_id, "reason": "no binder interface residues found"})
                continue
            fixed_residues[candidate_id] = contacts
        elif mode == "redesign_selected_unlocked":
            candidate_metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
            unlocked_text = candidate_metadata.get("template_unlocked_residues") or config.get("unlocked_residues")
            unlocked = set(
                _parse_residue_tags(
                    unlocked_text,
                    default_chain=binder_chains[0] if binder_chains else "",
                )
            )
            if not unlocked:
                skipped.append({"candidate_id": candidate_id, "reason": "no residues selected for redesign"})
                continue
            binder_residues = _pdb_residue_tags(structure, binder_chains)
            fixed_residues[candidate_id] = [tag for tag in binder_residues if tag not in unlocked]
        elif mode == "redesign_selected_locked":
            candidate_metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
            locked_text = candidate_metadata.get("template_locked_residues") or config.get("locked_residues")
            locked = _parse_residue_tags(
                locked_text,
                default_chain=binder_chains[0] if binder_chains else "",
            )
            if not locked:
                skipped.append({"candidate_id": candidate_id, "reason": "no residues selected to lock"})
                continue
            fixed_residues[candidate_id] = locked
        eligible.append(candidate)

    write_json(refinement_dir / "skipped_candidates.json", {"candidates": skipped})
    if not eligible:
        if passthrough:
            return passthrough, {
                "mode": mode,
                "status": "completed",
                "input_candidate_count": len(candidates),
                "eligible_candidate_count": 0,
                "passthrough_candidate_count": len(passthrough),
                "skipped_candidate_count": len(skipped),
                "refined_candidate_count": 0,
            }
        return candidates, {
            "mode": mode,
            "status": "failed",
            "error": "No candidates were eligible for sequence refinement.",
            "input_candidate_count": len(candidates),
            "skipped_candidate_count": len(skipped),
        }

    update_status(
        run_dir,
        "running",
        current_phase="Sequence refinement",
        current_engine=", ".join(sorted({row["model_type"] for row in sequence_config_by_candidate.values()})),
        progress_label=f"Refining {len(eligible)} harmonized candidates",
    )
    refined: list[dict[str, Any]] = []
    child_runs: list[dict[str, Any]] = []
    sequence_groups: dict[tuple[str, int, float, str], list[dict[str, Any]]] = {}
    for candidate in eligible:
        candidate_id = str(candidate.get("candidate_id") or "")
        row_config = sequence_config_by_candidate.get(candidate_id) or {
            "model_type": default_model_type,
            "sequences_per_structure": max(1, int(config.get("sequences_per_structure", 2))),
            "sampling_temp": float(config.get("sampling_temp", 0.1)),
            "omit_aas": _normalize_omit_aas(config.get("omit_aas")),
        }
        key = (
            str(row_config["model_type"]),
            max(1, int(row_config["sequences_per_structure"])),
            float(row_config["sampling_temp"]),
            _normalize_omit_aas(row_config["omit_aas"]),
        )
        sequence_groups.setdefault(key, []).append(candidate)
    effective_sequence_groups: list[dict[str, Any]] = []
    for group_index, ((model_type, num_seq_per_target, sampling_temp, omit_aas), model_candidates) in enumerate(
        sorted(sequence_groups.items(), key=lambda item: item[0]),
        start=1,
    ):
        if not model_candidates:
            continue
        group_suffix = f"{model_type}_{group_index:02d}"
        source_jsonl = refinement_dir / f"source_candidates_{group_suffix}.jsonl"
        _write_candidate_jsonl(source_jsonl, model_candidates)
        effective_sequence_groups.append(
            {
                "model_type": model_type,
                "candidate_count": len(model_candidates),
                "sequences_per_structure": num_seq_per_target,
                "sampling_temp": sampling_temp,
                "omit_aas": omit_aas,
            }
        )
        child_run = run_ligandmpnn_sequence_design(
            source_run_dir=run_dir,
            candidates_jsonl=source_jsonl,
            model_type=model_type,
            num_seq_per_target=num_seq_per_target,
            sampling_temp=sampling_temp,
            omit_aas=omit_aas,
            seed=int(config.get("seed", 0)),
            accepted_stages=sorted({str(candidate.get("stage") or "") for candidate in model_candidates}),
            fixed_residues_by_candidate={
                str(candidate.get("candidate_id") or ""): fixed_residues.get(str(candidate.get("candidate_id") or ""), [])
                for candidate in model_candidates
            },
            gpu_device=gpu_device,
        )
        mark_internal_job(
            child_run,
            parent_run_dir=run_dir,
            parent_task_group=DESIGN_CAMPAIGN_GROUP,
            role="sequence_refinement",
            engine=model_type,
        )
        child_result = read_json(child_run / "result.json")
        child_runs.append(
            {
                "model_type": model_type,
                "run_dir": str(child_run),
                "run_id": child_run.name,
                "status": "completed" if child_result.get("success") is True else "failed",
            }
        )
        parent_by_id = {str(candidate["candidate_id"]): candidate for candidate in model_candidates}
        for row in read_candidates(child_run):
            row = _absolute_candidate_paths(row, child_run)
            parent_id = str((row.get("parents") or [""])[0])
            parent = parent_by_id.get(parent_id)
            if parent is None:
                continue
            sequence_metrics = dict(row.get("metrics") or {})
            metrics = dict(parent.get("metrics") or {})
            metrics.update({f"refinement_{key}": value for key, value in sequence_metrics.items()})
            metrics.update(
                {
                    "sequence_refinement_mode": mode,
                    "sequence_refinement_engine": model_type,
                    "sequence_refinement_fixed_residue_count": len(fixed_residues.get(parent_id, [])),
                    "post_refinement_validation_required": True,
                }
            )
            metadata = dict(parent.get("raw_metadata") or {})
            metadata.update(
                {
                    "sequence_refinement_run_dir": str(child_run),
                    "sequence_refinement_run_id": child_run.name,
                    "sequence_refinement_parent_id": parent_id,
                    "sequence_refinement_fixed_residues": fixed_residues.get(parent_id, []),
                    "pre_refinement_candidate": parent,
                }
            )
            row.update(
                {
                    "metrics": metrics,
                    "raw_metadata": metadata,
                    "parents": [parent_id, *list(parent.get("parents") or [])],
                }
            )
            structure = _candidate_structure_path(child_run, row)
            if structure is not None and structure.exists() and structure.suffix.lower() == ".pdb":
                try:
                    binder_chains, target_chains = _source_role_chains(row, structure)
                    binder_sequences = _pdb_chain_sequences(structure, binder_chains)
                    binder_sequence = "".join(binder_sequences.get(chain, "") for chain in binder_chains)
                    if binder_sequence:
                        row["binder_sequence"] = binder_sequence
                        row["binder_length"] = str(len(binder_sequence))
                    row["binder_chains"] = binder_chains
                    row["target_chains"] = target_chains
                    row["chain_role_schema"] = chain_roles.CHAIN_ROLE_SCHEMA_EXPLICIT_V2
                    role_map = chain_roles.build_explicit_role_map(
                        binder_source_chains=binder_chains,
                        binder_engine_chains=binder_chains,
                        target_source_chains=target_chains,
                        target_engine_chains=target_chains,
                    )
                    row["chain_roles"] = role_map.to_dict()
                    metrics["binder_chains"] = ",".join(binder_chains)
                    metrics["target_chains"] = ",".join(target_chains)
                    metadata["chain_role_schema"] = chain_roles.CHAIN_ROLE_SCHEMA_EXPLICIT_V2
                    metadata["chain_roles"] = role_map.to_dict()
                    metadata["binder_source_chains"] = binder_chains
                    metadata["target_source_chains"] = target_chains
                except Exception:
                    pass
            refined.append(row)

    output_candidates = [*passthrough, *refined]
    success = bool(output_candidates)
    summary = {
        "mode": mode,
        "status": "completed" if success else "failed",
        "engine": ",".join(sorted({str(row.get("model_type") or "") for row in effective_sequence_groups if row.get("model_type")})),
        "sequence_by_engine": sequence_by_engine,
        "input_candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "passthrough_candidate_count": len(passthrough),
        "skipped_candidate_count": len(skipped),
        "refined_candidate_count": len(refined),
        "sequences_per_structure": max(1, int(config.get("sequences_per_structure", 2))),
        "sampling_temp": float(config.get("sampling_temp", 0.1)),
        "omit_aas": _normalize_omit_aas(config.get("omit_aas")),
        "effective_sequence_groups": effective_sequence_groups,
        "fixed_interface_candidate_count": len(fixed_residues),
        "child_runs": child_runs,
        "child_run": child_runs[0]["run_dir"] if child_runs else "",
        "error": "" if success else "Refinement failed",
    }
    write_json(refinement_dir / "refinement_summary.json", summary)
    return (output_candidates if success else candidates), summary


def _screen_candidates_with_refolder(
    run_dir: Path,
    candidates: list[dict[str, Any]],
    *,
    label: str,
    gpu_device: str,
    refolder: str,
    settings: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = dict(settings or {})
    refolder = str(refolder or "AF2-IG")
    if refolder not in REFOLDER_FLAG_BY_LABEL:
        raise ValueError(f"Unsupported staged sequence screening refolder: {refolder}")
    screen_dir = run_dir / "artifacts" / "design_campaign" / "staged_two_step" / label
    source_jsonl = screen_dir / "source_candidates.jsonl"
    _write_candidate_jsonl(source_jsonl, candidates)
    if not candidates:
        return [], {"status": "skipped", "candidate_count": 0}
    from mn_protein_design.workflows import benchmark as benchmark_workflow

    num_recycles = max(1, int(settings.get("num_recycles", 3)))
    num_sampling_steps = max(1, int(settings.get("num_sampling_steps", 68)))
    num_samples = max(1, int(settings.get("num_samples", 3)))
    use_target_msa = bool(settings.get("use_target_msa", _refolder_default_use_target_msa(refolder)))
    refolder_flags = {flag: False for flag in set(REFOLDER_FLAG_BY_LABEL.values())}
    refolder_flags[REFOLDER_FLAG_BY_LABEL[refolder]] = True
    safe_refolder = refolder.lower().replace(" ", "_").replace("-", "")
    update_status(
        run_dir,
        "running",
        current_phase="Two-step sequence screen",
        current_engine=refolder,
        progress_label=f"Screening {len(candidates)} candidates after {label}",
    )
    child_run = benchmark_workflow.run_de_novo_binder_scoring_dataset(
        source_run_dir=run_dir,
        candidates_jsonl=source_jsonl,
        selected_candidate_ids=[str(candidate.get("candidate_id") or "") for candidate in candidates],
        mode="seq_only_csv",
        generate_inputs=True,
        models=[],
        run_common_interface_metrics=False,
        run_pyrosetta_input_metrics=False,
        run_predicted_rosetta_metrics=False,
        run_pymol_metrics=False,
        esmfold2_modes=["sequence"],
        esmfold2_use_target_msa=use_target_msa,
        num_loops=num_recycles,
        num_sampling_steps=num_sampling_steps,
        seed=max(0, int(settings.get("seed", 0))),
        af2_num_recycles=num_recycles,
        af2_multimer=bool(settings.get("af2_multimer", True)),
        af2_use_initial_guess=bool(settings.get("af2_use_initial_guess", True)),
        af2_use_binder_template=bool(settings.get("af2_use_binder_template", False)),
        af2_use_interface_template=bool(settings.get("af2_use_interface_template", False)),
        colabfold_num_recycles=num_recycles,
        colabfold_num_models=num_samples,
        colabfold_use_target_templates=bool(settings.get("colabfold_use_target_templates", True)),
        colabfold_max_template_hits=max(1, int(settings.get("colabfold_max_template_hits", 4))),
        boltz2_use_target_template=bool(settings.get("boltz2_use_target_template", True)),
        boltz2_use_target_msa=use_target_msa,
        boltz2_recycling_steps=num_recycles,
        boltz2_sampling_steps=num_sampling_steps,
        boltz2_diffusion_samples=num_samples,
        boltz2_write_full_pae=bool(settings.get("boltz2_write_full_pae", True)),
        rf3_use_target_msa=use_target_msa,
        rf3_use_target_template=bool(settings.get("rf3_use_target_template", True)),
        rf3_recycles=max(2, num_recycles),
        rf3_num_steps=num_sampling_steps,
        rf3_diffusion_batch_size=num_samples,
        rf3_seed=max(0, int(settings.get("seed", 0))),
        openfold3_use_target_msa=use_target_msa,
        openfold3_num_diffusion_samples=num_samples,
        openfold3_num_model_seeds=max(1, int(settings.get("openfold3_num_model_seeds", 1))),
        openfold3_num_recycles=num_recycles,
        openfold3_use_msa_server=bool(settings.get("openfold3_use_msa_server", False)),
        protenix_use_msa=use_target_msa,
        protenix_cycle=num_recycles,
        protenix_diffusion_steps=num_sampling_steps,
        protenix_samples=num_samples,
        protenix_v1_model_name=str(settings.get("protenix_v1_model_name") or refolding_workflow.PROTENIX_V1_MODEL),
        protenix_v1_use_msa=use_target_msa,
        protenix_v1_use_template=bool(settings.get("protenix_v1_use_template", True)),
        protenix_v1_cycle=num_recycles,
        protenix_v1_diffusion_steps=num_sampling_steps,
        protenix_v1_samples=num_samples,
        protenix_v2_model_name=str(settings.get("protenix_v2_model_name") or refolding_workflow.PROTENIX_V2_MODEL),
        protenix_v2_use_msa=use_target_msa,
        protenix_v2_use_template=bool(settings.get("protenix_v2_use_template", True)),
        protenix_v2_cycle=num_recycles,
        protenix_v2_diffusion_steps=num_sampling_steps,
        protenix_v2_samples=num_samples,
        boltzgen_recycling_steps=num_recycles,
        boltzgen_sampling_steps=num_sampling_steps,
        boltzgen_diffusion_samples=num_samples,
        alphafast_num_recycles=num_recycles,
        alphafast_use_target_templates=bool(settings.get("alphafast_use_target_templates", True)),
        alphafast_query_only_msa=not bool(settings.get("alphafast_use_target_msa", settings.get("use_target_msa", True))),
        alphafast_gpu_device=gpu_device,
        gpu_device=gpu_device,
        max_records=0,
        job_type="design_campaign_sequence_screen",
        tool_name="design_campaign_sequence_screen",
        internal_parent_run_dir=run_dir,
        internal_parent_task_group=DESIGN_CAMPAIGN_GROUP,
        internal_parent_role=f"staged_two_step_{label}_{safe_refolder}_screen",
        internal_parent_engine=refolder,
        **refolder_flags,
    )
    mark_internal_job(
        child_run,
        parent_run_dir=run_dir,
        parent_task_group=DESIGN_CAMPAIGN_GROUP,
        role=f"staged_two_step_{label}_{safe_refolder}_screen",
        engine=refolder,
    )
    result = read_json(child_run / "result.json")
    if result.get("success") is not True:
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        return [], {
            "status": "failed",
            "candidate_count": len(candidates),
            "screened_candidate_count": 0,
            "refolder": refolder,
            "settings": settings,
            "child_run": str(child_run),
            "child_run_id": child_run.name,
            "error": str(metrics.get("error") or "Refolding screen failed"),
        }
    screened = [_absolute_candidate_paths(candidate, child_run) for candidate in read_candidates(child_run)]
    if not screened:
        screened = [
            _candidate_with_screen_complex(candidate, child_run, refolder)
            for candidate in candidates
            if _screen_prediction_complex_path(child_run, str(candidate.get("candidate_id") or ""), refolder) is not None
        ]
    screened = _merge_screen_metric_table(screened, child_run)
    screened = [_add_target_aligned_screen_rmsd(candidate) for candidate in screened]
    screened = [_add_bindcraft_structure_metrics(candidate) for candidate in screened]
    _write_candidate_jsonl(screen_dir / "screened_candidates.jsonl", screened)
    return screened, {
        "status": "completed" if screened else "failed",
        "candidate_count": len(candidates),
        "screened_candidate_count": len(screened),
        "refolder": refolder,
        "settings": settings,
        "child_run": str(child_run),
        "child_run_id": child_run.name,
    }


def _coerce_screen_metric_value(value: object) -> object:
    if value in {None, ""}:
        return None
    text = str(value)
    if text.lower() == "nan":
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return value


def _screen_metric_is_missing(value: object) -> bool:
    if value in {None, ""}:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).lower() == "nan"


def _merge_screen_metric_table(candidates: list[dict[str, Any]], child_run: Path) -> list[dict[str, Any]]:
    metrics_path = child_run / "artifacts" / "benchmark" / "merged_benchmark_metrics.csv"
    if not metrics_path.exists():
        return candidates
    rows_by_id: dict[str, dict[str, object]] = {}
    try:
        with metrics_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                for key in ("candidate_id", "binder_id", "original_binder_id"):
                    value = str(row.get(key) or "").strip()
                    if value:
                        rows_by_id.setdefault(value, row)
    except Exception:
        return candidates
    merged: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate.get("candidate_id") or "")
        row = rows_by_id.get(candidate_id)
        if row is None:
            merged.append(candidate)
            continue
        metrics = candidate.setdefault("metrics", {})
        for key, value in row.items():
            if key in {"complex_pdb", "target_pdb", "binder_sequence", "A_seq"}:
                continue
            coerced = _coerce_screen_metric_value(value)
            if key not in metrics or _screen_metric_is_missing(metrics.get(key)):
                metrics[key] = coerced
        _promote_target_aligned_rmsd(metrics)
        merged.append(candidate)
    return merged


def _screen_prediction_complex_path(child_run: Path, candidate_id: str, refolder: str) -> Path | None:
    engine_dirs = {
        "AF2-IG": ["af2"],
        "ColabFold": ["colab", "colabfold"],
        "Boltz-2": ["boltz2"],
        "AlphaFast AF3": ["af3"],
        "ESMFold2": ["esmfold2"],
        "RF3": ["rf3"],
        "OpenFold-3": ["openfold3"],
        "Protenix": ["protenix"],
        "Protenix v1": ["protenix_v1"],
        "Protenix v2": ["protenix_v2"],
        "BoltzGen Fold": ["boltzgen_fold"],
    }.get(refolder, [])
    if not engine_dirs:
        return None
    candidate_ids = [candidate_id]
    safe_candidate_id = _safe_candidate_token(candidate_id)
    if safe_candidate_id and safe_candidate_id not in candidate_ids:
        candidate_ids.append(safe_candidate_id)
    filesystem_candidate_id = re.sub(r"[^A-Za-z0-9_]+", "_", candidate_id).strip("_")
    if filesystem_candidate_id and filesystem_candidate_id not in candidate_ids:
        candidate_ids.append(filesystem_candidate_id)
    for root in ("predicted_viewer_structures", "predicted_metric_pdbs"):
        for engine_dir in engine_dirs:
            for candidate_name in candidate_ids:
                for suffix in ("", "_af2ig_mt_tt", "_af2ig_mt_ct"):
                    path = child_run / "artifacts" / "benchmark" / root / engine_dir / f"{candidate_name}{suffix}.pdb"
                    if path.exists():
                        return path
    return None


def _candidate_with_screen_complex(candidate: dict[str, Any], child_run: Path, refolder: str) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "")
    prediction = _screen_prediction_complex_path(child_run, candidate_id, refolder)
    if prediction is None:
        return candidate
    updated = dict(candidate)
    raw = dict(updated.get("raw_metadata") if isinstance(updated.get("raw_metadata"), dict) else {})
    raw.setdefault("pre_level2_complex_pdb", updated.get("complex_pdb"))
    raw["level1_screen_complex_pdb"] = str(prediction)
    raw["level1_screen_refolder"] = refolder
    updated["raw_metadata"] = raw
    updated["complex_pdb"] = str(prediction)
    return updated


def _promote_target_aligned_rmsd(metrics: dict[str, Any]) -> None:
    if not _screen_metric_is_missing(metrics.get("target_aligned_binder_rmsd")):
        return
    for key in (
        "af2_target_aligned_binder_rmsd",
        "af2_average_target_aligned_binder_rmsd",
        "af2_model_1_target_aligned_binder_rmsd",
        "first_refolding_screen_screen_rmsd",
        "second_refolding_screen_screen_rmsd",
    ):
        value = metrics.get(key)
        if _screen_metric_is_missing(value):
            continue
        metrics["target_aligned_binder_rmsd"] = value
        metrics["target_aligned_binder_rmsd_source"] = key
        if metrics.get("target_aligned_binder_rmsd_status") in {None, "", "missing reference or prediction complex"}:
            metrics["target_aligned_binder_rmsd_status"] = f"from {key}"
        return


def _path_from_candidate(value: object, *, source_run_dir: object = "") -> Path | None:
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path if path.exists() else None
    if source_run_dir:
        base = Path(str(source_run_dir)).expanduser()
        candidate = base / path
        if candidate.exists():
            return candidate
    return path if path.exists() else None


def _add_target_aligned_screen_rmsd(candidate: dict[str, Any]) -> dict[str, Any]:
    raw = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
    source = raw.get("source_candidate") if isinstance(raw.get("source_candidate"), dict) else {}
    if not source:
        source = raw.get("pre_refinement_candidate") if isinstance(raw.get("pre_refinement_candidate"), dict) else {}
    source_raw = source.get("raw_metadata") if isinstance(source.get("raw_metadata"), dict) else {}
    source_run_dir = raw.get("source_run_dir") or source_raw.get("source_run_dir")
    reference_pdb = _path_from_candidate(source.get("complex_pdb"), source_run_dir=source_run_dir)
    prediction_pdb = _path_from_candidate(candidate.get("complex_pdb"), source_run_dir="")
    metrics = candidate.setdefault("metrics", {})
    if reference_pdb is None or prediction_pdb is None:
        metrics["target_aligned_binder_rmsd"] = None
        metrics["target_aligned_binder_rmsd_status"] = "missing reference or prediction complex"
        return candidate
    reference_binder_chains = [str(chain) for chain in source.get("binder_chains") or candidate.get("binder_chains") or []]
    reference_target_chains = [str(chain) for chain in source.get("target_chains") or candidate.get("target_chains") or []]
    prediction_binder_chains = [str(chain) for chain in candidate.get("binder_chains") or raw.get("input_binder_chains") or []]
    prediction_target_chains = [str(chain) for chain in candidate.get("target_chains") or raw.get("input_target_chains") or []]
    if not reference_binder_chains or not reference_target_chains or not prediction_binder_chains or not prediction_target_chains:
        metrics["target_aligned_binder_rmsd"] = None
        metrics["target_aligned_binder_rmsd_status"] = "missing binder/target chain metadata"
        return candidate
    try:
        from mn_protein_design.workflows.benchmark import _target_aligned_binder_pose_metrics

        pose_metrics = _target_aligned_binder_pose_metrics(
            reference_pdb,
            prediction_pdb,
            reference_binder_chains=reference_binder_chains,
            reference_target_chains=reference_target_chains,
            prediction_binder_chains=prediction_binder_chains,
            prediction_target_chains=prediction_target_chains,
        )
    except Exception as exc:
        metrics["target_aligned_binder_rmsd"] = None
        metrics["target_aligned_binder_rmsd_status"] = f"calculation failed: {type(exc).__name__}: {exc}"
        return candidate
    metrics.update({f"screen_{key}": value for key, value in pose_metrics.items()})
    metrics["target_aligned_binder_rmsd"] = pose_metrics.get("target_aligned_binder_rmsd")
    metrics["target_aligned_binder_rmsd_status"] = pose_metrics.get("pose_rmsd_status", "")
    _promote_target_aligned_rmsd(metrics)
    return candidate


def _iptm_rank_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    metrics = candidate.get("metrics") or {}
    return (
        -_first_metric(metrics, ("af2_iptm", "af2_average_iptm", "af2_model_1_iptm", "model_1_iptm", "model_2_iptm", "iptm", "i_ptm", "ipTM"), default=-math.inf),
        _first_metric(metrics, ("af2_ipae", "af2_average_ipae", "af2_model_1_ipae", "model_1_ipae", "model_2_ipae", "ipae", "i_pae")),
        -_first_metric(metrics, ("af2_binder_plddt", "af2_average_binder_plddt", "af2_model_1_binder_plddt", "model_1_binder_plddt", "model_2_binder_plddt", "binder_plddt", "plddt"), default=-math.inf),
    )


def _screen_rank_key(candidate: dict[str, Any], metric: str) -> tuple[float, ...]:
    metrics = candidate.get("metrics") or {}
    metric = str(metric or "iptm")
    if metric == "ranking_score":
        primary = -_first_metric(
            metrics,
            (
                "ranking_score",
                "confidence_score",
                "aggregate_score",
                "overall_confidence",
                "confidence",
                "mean_confidence",
            ),
            default=-math.inf,
        )
    elif metric == "ipae":
        primary = _first_metric(metrics, ("af2_ipae", "af2_average_ipae", "af2_model_1_ipae", "model_1_ipae", "model_2_ipae", "ipae", "i_pae"))
    elif metric == "binder_plddt":
        primary = -_first_metric(metrics, ("af2_binder_plddt", "af2_average_binder_plddt", "af2_model_1_binder_plddt", "model_1_binder_plddt", "model_2_binder_plddt", "binder_plddt", "plddt"), default=-math.inf)
    elif metric == "ptm":
        primary = -_first_metric(metrics, ("af2_ptm", "af2_average_ptm", "af2_model_1_ptm", "model_1_ptm", "model_2_ptm", "ptm", "pTM"), default=-math.inf)
    elif metric == "target_aligned_binder_rmsd":
        value = _screen_rmsd_value(candidate)
        primary = value if value is not None else math.inf
    else:
        primary = -_first_metric(metrics, ("af2_iptm", "af2_average_iptm", "af2_model_1_iptm", "model_1_iptm", "model_2_iptm", "iptm", "i_ptm", "ipTM"), default=-math.inf)
    return (
        primary,
        *_iptm_rank_key(candidate),
    )


def _screen_rmsd_value(candidate: dict[str, Any]) -> float | None:
    metrics = candidate.get("metrics") or {}
    for key in (
        "target_aligned_binder_rmsd",
        "average_target_aligned_binder_rmsd",
        "model_1_target_aligned_binder_rmsd",
        "model_2_target_aligned_binder_rmsd",
        "af2_target_aligned_binder_rmsd",
        "af2_average_target_aligned_binder_rmsd",
        "af2_model_1_target_aligned_binder_rmsd",
    ):
        value = metrics.get(key)
        if value in {None, ""}:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _filter_by_screen_rmsd(
    candidates: list[dict[str, Any]],
    *,
    max_rmsd: float,
    stage: str,
    disabled: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if disabled:
        for candidate in candidates:
            metrics = candidate.setdefault("metrics", {})
            metrics[f"{stage}_screen_rmsd"] = _screen_rmsd_value(candidate)
            metrics[f"{stage}_screen_max_rmsd"] = float(max_rmsd)
            metrics[f"{stage}_screen_pass"] = True
            metrics[f"{stage}_screen_failure_reason"] = ""
        return list(candidates), {
            "input_count": len(candidates),
            "passing_count": len(candidates),
            "failed_count": 0,
            "missing_rmsd_count": sum(1 for candidate in candidates if _screen_rmsd_value(candidate) is None),
            "max_rmsd": float(max_rmsd),
            "disabled": True,
        }
    passing: list[dict[str, Any]] = []
    failed = 0
    missing = 0
    for candidate in candidates:
        rmsd = _screen_rmsd_value(candidate)
        metrics = candidate.setdefault("metrics", {})
        metrics[f"{stage}_screen_rmsd"] = rmsd
        metrics[f"{stage}_screen_max_rmsd"] = float(max_rmsd)
        if rmsd is None:
            missing += 1
            metrics[f"{stage}_screen_pass"] = False
            metrics[f"{stage}_screen_failure_reason"] = "missing_rmsd"
            continue
        if rmsd <= max_rmsd:
            metrics[f"{stage}_screen_pass"] = True
            metrics[f"{stage}_screen_failure_reason"] = ""
            passing.append(candidate)
        else:
            failed += 1
            metrics[f"{stage}_screen_pass"] = False
            metrics[f"{stage}_screen_failure_reason"] = f"rmsd>{max_rmsd:g}"
    return passing, {
        "input_count": len(candidates),
        "passing_count": len(passing),
        "failed_count": failed,
        "missing_rmsd_count": missing,
        "max_rmsd": float(max_rmsd),
    }


def _staged_attempt_key(candidate: dict[str, Any]) -> str:
    def normalize(value: object) -> str:
        text = str(value or "")
        patterns = (
            r"_af2ig(?:_[a-z0-9]+)*$",
            r"_colabfold(?:_[a-z0-9]+)*$",
            r"_boltz2ig(?:_[a-z0-9]+)*$",
            r"_complex_boltzgen_fold$",
            r"_monomer_boltz2$",
            r"_(?:soluble_)?protein_mpnn_\d+$",
            r"_(?:soluble_)?mpnn_\d+$",
            r"_ligandmpnn_\d+$",
        )
        previous = None
        while text and text != previous:
            previous = text
            for pattern in patterns:
                text = re.sub(pattern, "", text)
        return text

    raw = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
    source = raw.get("source_candidate") if isinstance(raw.get("source_candidate"), dict) else {}
    nested_source = source
    while isinstance(nested_source, dict) and nested_source.get("raw_metadata"):
        nested_raw = nested_source.get("raw_metadata")
        if not isinstance(nested_raw, dict):
            break
        next_source = nested_raw.get("source_candidate")
        if not isinstance(next_source, dict):
            break
        nested_source = next_source
    if isinstance(nested_source, dict) and nested_source.get("candidate_id"):
        return normalize(nested_source["candidate_id"])

    parents = [str(parent) for parent in candidate.get("parents") or [] if str(parent)]
    if parents:
        return normalize(parents[-1])
    for key in ("source_candidate_id", "sequence_refinement_parent_id"):
        value = raw.get(key)
        if value:
            return normalize(value)
    if source.get("candidate_id"):
        return normalize(source["candidate_id"])
    return normalize(candidate.get("candidate_id") or "")


def _select_best_screened_per_attempt(
    candidates: list[dict[str, Any]],
    *,
    rank_metric: str,
    keep_per_attempt: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    keep_per_attempt = max(1, int(keep_per_attempt))
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(_staged_attempt_key(candidate), []).append(candidate)
    selected: list[dict[str, Any]] = []
    for attempt_key, rows in grouped.items():
        ranked = sorted(rows, key=lambda row: _screen_rank_key(row, rank_metric))
        for rank, candidate in enumerate(ranked[:keep_per_attempt], start=1):
            metrics = candidate.setdefault("metrics", {})
            metrics["staged_attempt_key"] = attempt_key
            metrics["staged_attempt_level1_passing_sequence_count"] = len(rows)
            metrics["staged_attempt_selected_for_level2"] = True
            metrics["staged_attempt_selection_metric"] = rank_metric
            metrics["staged_attempt_selection_rank"] = rank
            selected.append(candidate)
    return sorted(selected, key=lambda row: _screen_rank_key(row, rank_metric)), {
        "attempt_count": len(grouped),
        "selected_count": len(selected),
        "passing_sequence_count": len(candidates),
        "rank_metric": rank_metric,
        "keep_per_attempt": keep_per_attempt,
    }


def _staged_structure_element_metric(candidate: dict[str, Any]) -> tuple[str, int | None, str]:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    engine = _candidate_design_campaign_engine(candidate)
    if engine == "genie3":
        value = metrics.get("binder_ca_trace_elements")
        source = "CA-trace geometry elements"
    else:
        value = metrics.get("binder_secondary_structure_elements")
        source = "PyDSSP total elements"
    try:
        return engine, int(value), source
    except (TypeError, ValueError):
        return engine, None, source


def _filter_by_staged_structure_elements(
    candidates: list[dict[str, Any]],
    *,
    two_step: dict[str, Any],
    stage: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    enabled = bool(two_step.get("structure_element_filter_enabled", True))
    minimum = max(0, int(two_step.get("min_structure_elements", 3)))
    metric_prefix = f"staged_{stage}_structure_element_filter"
    if not enabled:
        return candidates, {
            "status": "disabled",
            "enabled": False,
            "input_candidate_count": len(candidates),
            "output_candidate_count": len(candidates),
            "min_structure_elements": minimum,
        }

    passing: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    rejected_by_engine: dict[str, int] = {}
    for candidate in candidates:
        engine, element_count, source = _staged_structure_element_metric(candidate)
        candidate = {**candidate, "metrics": dict(candidate.get("metrics") or {})}
        metrics = candidate["metrics"]
        passed = element_count is not None and element_count >= minimum
        metrics[f"{metric_prefix}_enabled"] = True
        metrics[f"{metric_prefix}_source"] = source
        metrics[f"{metric_prefix}_min_elements"] = minimum
        metrics[f"{metric_prefix}_element_count"] = element_count
        metrics[f"{metric_prefix}_pass"] = bool(passed)
        if passed:
            passing.append(candidate)
        else:
            reason = "missing_structure_element_metric" if element_count is None else f"structure_elements<{minimum}"
            metrics[f"{metric_prefix}_failure_reason"] = reason
            rejected.append(candidate)
            rejected_by_engine[engine or "unknown"] = rejected_by_engine.get(engine or "unknown", 0) + 1

    return passing, {
        "status": "completed",
        "enabled": True,
        "stage": stage,
        "min_structure_elements": minimum,
        "input_candidate_count": len(candidates),
        "output_candidate_count": len(passing),
        "rejected_candidate_count": len(rejected),
        "rejected_by_engine": rejected_by_engine,
        "metric_by_engine": {
            "genie3": "binder_ca_trace_elements",
            "default": "binder_secondary_structure_elements",
        },
    }


def run_staged_two_step_sequence_refinement(
    run_dir: Path,
    candidates: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    gpu_device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    two_step = config.get("staged_two_step") if isinstance(config.get("staged_two_step"), dict) else {}
    if not bool(two_step.get("enabled")):
        return run_campaign_sequence_refinement(run_dir, candidates, config=config, gpu_device=gpu_device)
    split_by_start = bool(two_step.get("split_by_generator_start", True))
    if split_by_start and len(candidates) > 1:
        attempts_dir = run_dir / "artifacts" / "design_campaign" / "staged_two_step" / "generator_starts"
        attempts_dir.mkdir(parents=True, exist_ok=True)
        final_candidates: list[dict[str, Any]] = []
        attempt_summaries: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates, start=1):
            attempt_key = _staged_attempt_key(candidate)
            source_id = str(candidate.get("candidate_id") or f"candidate_{index}")
            artifact_label = f"start_{index:04d}_{_safe_candidate_token(attempt_key or source_id)}"
            child_config = dict(config)
            child_two_step = dict(two_step)
            child_two_step["split_by_generator_start"] = False
            child_two_step["artifact_label_prefix"] = artifact_label
            child_config["staged_two_step"] = child_two_step
            child_config["artifact_label_prefix"] = artifact_label
            update_status(
                run_dir,
                "running",
                current_phase="Two-step sequence screen",
                current_engine=_candidate_design_campaign_engine(candidate),
                progress_label=f"Processing staged start {index}/{len(candidates)}",
            )
            try:
                start_candidates, start_summary = run_staged_two_step_sequence_refinement(
                    run_dir,
                    [candidate],
                    config=child_config,
                    gpu_device=gpu_device,
                )
                start_status = str(start_summary.get("status") or "completed")
            except Exception as exc:
                start_candidates = []
                start_summary = {
                    "mode": "staged_two_step",
                    "status": "failed",
                    "result": "start_failed",
                    "reason": str(exc),
                    "error": str(exc),
                }
                start_status = "failed"
            for row in start_candidates:
                metrics = row.setdefault("metrics", {})
                metrics.setdefault("staged_attempt_key", attempt_key)
                metrics["staged_generator_start_index"] = index
                metrics["staged_generator_start_artifact_label"] = artifact_label
            start_summary.update(
                {
                    "generator_start_index": index,
                    "generator_start_count": len(candidates),
                    "artifact_label": artifact_label,
                    "attempt_key": attempt_key,
                    "source_candidate_id": source_id,
                    "engine": _candidate_design_campaign_engine(candidate),
                    "output_candidate_count": len(start_candidates),
                }
            )
            attempt_summaries.append(start_summary)
            final_candidates.extend(start_candidates)
            write_json(attempts_dir / f"{artifact_label}.json", start_summary)
            _write_candidate_jsonl(attempts_dir / f"{artifact_label}.jsonl", start_candidates)
            update_status(
                run_dir,
                "running",
                current_phase="Two-step sequence screen",
                current_engine=_candidate_design_campaign_engine(candidate),
                progress_label=f"Finished staged start {index}/{len(candidates)} ({start_status})",
            )
        completed = sum(1 for row in attempt_summaries if str(row.get("status") or "") == "completed")
        failed = sum(1 for row in attempt_summaries if str(row.get("status") or "") == "failed")
        summary = {
            "mode": "staged_two_step_split_by_generator_start",
            "status": "completed" if final_candidates or completed else "failed",
            "result": "survivors_found" if final_candidates else "zero_survivors",
            "reason": "" if final_candidates else "No staged generator starts produced surviving candidates.",
            "input_candidate_count": len(candidates),
            "generator_start_count": len(candidates),
            "completed_generator_starts": completed,
            "failed_generator_starts": failed,
            "output_candidate_count": len(final_candidates),
            "attempt_summaries": attempt_summaries,
        }
        summary_dir = run_dir / "artifacts" / "design_campaign" / "staged_two_step"
        write_json(summary_dir / "two_step_summary.json", summary)
        _write_candidate_jsonl(summary_dir / "two_step_candidates.jsonl", final_candidates)
        return final_candidates, summary
    screen_refolder = str(two_step.get("screen_refolder") or "AF2-IG")
    if screen_refolder not in REFOLDER_FLAG_BY_LABEL:
        raise ValueError(f"Unsupported staged sequence screening refolder: {screen_refolder}")
    second_screen_refolder = str(two_step.get("second_screen_refolder") or screen_refolder)
    if second_screen_refolder not in REFOLDER_FLAG_BY_LABEL:
        raise ValueError(f"Unsupported staged second-level screening refolder: {second_screen_refolder}")
    rank_metric = str(two_step.get("rank_metric") or "iptm")
    second_rank_metric = str(two_step.get("second_rank_metric") or rank_metric)

    first_config = dict(config)
    first_config["mode"] = "redesign_all"
    first_config["omit_aas"] = _normalize_omit_aas(first_config.get("omit_aas"))
    first_config["passthrough_disabled_sequence_engines"] = True
    first_config["sequence_by_engine"] = _sequence_by_engine_with_level_settings(
        first_config.get("sequence_by_engine") if isinstance(first_config.get("sequence_by_engine"), dict) else {},
        sequences_per_structure=max(1, int(first_config.get("sequences_per_structure", 2))),
        sampling_temp=float(first_config.get("sampling_temp", 0.1)),
        omit_aas=_normalize_omit_aas(first_config.get("omit_aas")),
    )
    artifact_label = str(two_step.get("artifact_label_prefix") or "").strip()
    first_label = f"{artifact_label}_first_pass" if artifact_label else "first_pass"
    second_label = f"{artifact_label}_second_pass" if artifact_label else "second_pass"
    first_config["artifact_label_prefix"] = first_label

    first_structure_candidates, first_structure_filter_summary = _filter_by_staged_structure_elements(
        candidates,
        two_step=two_step,
        stage="level_1",
    )
    if not first_structure_candidates:
        summary = {
            "mode": "staged_two_step",
            "status": "completed",
            "result": "zero_survivors",
            "reason": "No staged candidates passed the Level 1 structure-element filter.",
            "input_candidate_count": len(candidates),
            "level_1_structure_filter": first_structure_filter_summary,
            "first_pass": {"status": "skipped", "reason": "no_level_1_structure_filter_survivors"},
            "first_screen": {"status": "skipped", "reason": "no_level_1_structure_filter_survivors"},
            "first_rmsd_filter": {"status": "skipped", "reason": "no_level_1_structure_filter_survivors"},
            "first_attempt_selection": {"status": "skipped", "reason": "no_level_1_structure_filter_survivors"},
            "level_2_structure_filter": {"status": "skipped", "reason": "no_level_1_survivors"},
            "second_pass": {"status": "skipped", "reason": "no_level_1_survivors"},
            "second_screen": {"status": "skipped", "reason": "no_level_1_survivors"},
            "second_rmsd_filter": {"status": "skipped", "reason": "no_level_1_survivors"},
            "output_candidate_count": 0,
        }
        summary_dir = run_dir / "artifacts" / "design_campaign" / "staged_two_step"
        if artifact_label:
            summary_dir = summary_dir / _safe_candidate_token(artifact_label)
        write_json(summary_dir / "two_step_summary.json", summary)
        _write_candidate_jsonl(summary_dir / "two_step_candidates.jsonl", [])
        return [], summary

    first_candidates, first_summary = run_campaign_sequence_refinement(
        run_dir,
        first_structure_candidates,
        config=first_config,
        gpu_device=gpu_device,
    )
    first_screen_settings = dict(two_step.get("screen_settings") or {})
    if "num_recycles" not in first_screen_settings:
        first_screen_settings["num_recycles"] = max(1, int(two_step.get("screen_num_recycles", 3)))
    screened, screen_summary = _screen_candidates_with_refolder(
        run_dir,
        first_candidates,
        label=first_label,
        gpu_device=gpu_device,
        refolder=screen_refolder,
        settings=first_screen_settings,
    )
    rmsd_cutoff = float(two_step.get("max_target_aligned_binder_rmsd", 3.5))
    second_rmsd_cutoff = float(two_step.get("second_max_target_aligned_binder_rmsd", rmsd_cutoff))
    disable_first_rmsd_filter = bool(two_step.get("disable_rmsd_filter"))
    disable_second_rmsd_filter = bool(two_step.get("second_disable_rmsd_filter", disable_first_rmsd_filter))
    first_passing, first_rmsd_summary = _filter_by_screen_rmsd(
        screened,
        max_rmsd=rmsd_cutoff,
        stage="first_refolding_screen",
        disabled=disable_first_rmsd_filter,
    )
    first_keep_per_attempt = max(1, int(two_step.get("first_keep_per_attempt", 2)))
    selected, first_attempt_summary = _select_best_screened_per_attempt(
        first_passing,
        rank_metric=rank_metric,
        keep_per_attempt=first_keep_per_attempt,
    )
    for rank, candidate in enumerate(selected, start=1):
        metrics = candidate.setdefault("metrics", {})
        metrics["staged_sequence_level"] = 1
        metrics["staged_output_role"] = "level_1_winner"
        metrics["staged_final_output"] = True
        metrics["staged_two_step_first_screen_rank"] = rank
        metrics["staged_two_step_first_screen_selection_metric"] = rank_metric

    run_second_level = bool(two_step.get("run_second_level", True))
    second_candidates: list[dict[str, Any]] = []
    second_passing: list[dict[str, Any]] = []
    second_keep_total = max(1, int(two_step.get("second_keep_total", 2)))
    second_summary: dict[str, Any] = {"mode": "skipped", "status": "skipped", "reason": "level_2_disabled"}
    second_screen_summary: dict[str, Any] = {"status": "skipped", "reason": "level_2_disabled"}
    second_rmsd_summary: dict[str, Any] = {"status": "skipped", "reason": "level_2_disabled"}
    if run_second_level:
        first_screen_child = Path(str(screen_summary.get("child_run") or "")) if screen_summary.get("child_run") else None
        level2_inputs = (
            [
                _candidate_with_screen_complex(candidate, first_screen_child, screen_refolder)
                for candidate in selected
            ]
            if first_screen_child is not None
            else selected
        )
        level2_inputs, second_structure_filter_summary = _filter_by_staged_structure_elements(
            level2_inputs,
            two_step=two_step,
            stage="level_2",
        )
        if not level2_inputs:
            second_summary = {
                "mode": "skipped",
                "status": "skipped",
                "reason": "no_level_2_structure_filter_survivors",
            }
            second_screen_summary = {
                "status": "skipped",
                "reason": "no_level_2_structure_filter_survivors",
                "candidate_count": 0,
            }
            second_rmsd_summary = {
                "status": "skipped",
                "reason": "no_level_2_structure_filter_survivors",
            }
            run_second_level = False
        else:
            second_structure_filter_summary = second_structure_filter_summary
    else:
        second_structure_filter_summary = {"status": "skipped", "reason": "level_2_disabled"}
    if run_second_level:
        second_sequences_per_structure = max(1, int(two_step.get("second_sequences_per_structure", 2)))
        second_sampling_temp = float(two_step.get("second_sampling_temp", 0.1))
        second_omit_aas = _normalize_omit_aas(two_step.get("second_omit_aas"))
        second_sequence_by_engine = _sequence_by_engine_with_level_settings(
            two_step.get("second_sequence_by_engine")
            if isinstance(two_step.get("second_sequence_by_engine"), dict)
            else {},
            sequences_per_structure=second_sequences_per_structure,
            sampling_temp=second_sampling_temp,
            omit_aas=second_omit_aas,
        )
        second_config = {
            "mode": "redesign_non_interface",
            "engine": two_step.get("second_engine") or "Soluble ProteinMPNN",
            "sequences_per_structure": second_sequences_per_structure,
            "sampling_temp": second_sampling_temp,
            "omit_aas": second_omit_aas,
            "sequence_by_engine": second_sequence_by_engine,
            "passthrough_disabled_sequence_engines": True,
            "artifact_label_prefix": second_label,
            "seed": config.get("seed", 0),
        }
        second_candidates, second_summary = run_campaign_sequence_refinement(
            run_dir,
            level2_inputs,
            config=second_config,
            gpu_device=gpu_device,
        )
        if int(second_summary.get("refined_candidate_count") or 0) <= 0:
            second_candidates = []
            second_screen_summary = {
                "status": "skipped",
                "reason": "no_level_2_redesigned_candidates",
                "candidate_count": 0,
            }
            second_rmsd_summary = {
                "status": "skipped",
                "reason": "no_level_2_redesigned_candidates",
            }
            run_second_level = False
        if run_second_level:
            second_screen_settings = dict(two_step.get("second_screen_settings") or first_screen_settings)
            if "num_recycles" not in second_screen_settings:
                second_screen_settings["num_recycles"] = max(1, int(two_step.get("screen_num_recycles", 3)))
            second_screened, second_screen_summary = _screen_candidates_with_refolder(
                run_dir,
                second_candidates,
                label=second_label,
                gpu_device=gpu_device,
                refolder=second_screen_refolder,
                settings=second_screen_settings,
            )
            second_passing, second_rmsd_summary = _filter_by_screen_rmsd(
                second_screened,
                max_rmsd=second_rmsd_cutoff,
                stage="second_refolding_screen",
                disabled=disable_second_rmsd_filter,
            )
            second_passing = sorted(second_passing, key=lambda row: _screen_rank_key(row, second_rank_metric))
            second_passing = second_passing[:second_keep_total]
            for rank, candidate in enumerate(second_passing, start=1):
                metrics = candidate.setdefault("metrics", {})
                metrics["staged_sequence_level"] = 2
                metrics["staged_output_role"] = "level_2_winner"
                metrics["staged_final_output"] = True
                metrics["staged_two_step_second_screen_rank"] = rank
                metrics["staged_two_step_second_screen_selection_metric"] = second_rank_metric
    else:
        second_keep_total = 0
    include_first_stage = bool(two_step.get("include_first_stage_outputs", True))
    final_candidates = [*(selected if include_first_stage or not run_second_level else []), *second_passing]
    summary = {
        "mode": "staged_two_step",
        "status": "completed",
        "result": "survivors_found" if final_candidates else "zero_survivors",
        "reason": "" if final_candidates else "No staged candidates survived the configured sequence/refolding screens.",
        "screen_refolder": screen_refolder,
        "second_screen_refolder": second_screen_refolder,
        "rank_metric": rank_metric,
        "second_rank_metric": second_rank_metric,
        "max_target_aligned_binder_rmsd": rmsd_cutoff,
        "second_max_target_aligned_binder_rmsd": second_rmsd_cutoff,
        "disable_rmsd_filter": disable_first_rmsd_filter,
        "second_disable_rmsd_filter": disable_second_rmsd_filter,
        "first_keep_per_attempt": first_keep_per_attempt,
        "second_keep_total": second_keep_total,
        "run_second_level": run_second_level,
        "first_sequence_config": {
            "mode": first_config.get("mode"),
            "engine": first_config.get("engine"),
            "sequences_per_structure": first_config.get("sequences_per_structure"),
            "sampling_temp": first_config.get("sampling_temp"),
            "omit_aas": first_config.get("omit_aas"),
            "sequence_by_engine": first_config.get("sequence_by_engine"),
        },
        "second_sequence_config": (
            {
                "mode": "redesign_non_interface",
                "engine": two_step.get("second_engine") or "Soluble ProteinMPNN",
                "sequences_per_structure": max(1, int(two_step.get("second_sequences_per_structure", 2))),
                "sampling_temp": float(two_step.get("second_sampling_temp", 0.1)),
                "omit_aas": _normalize_omit_aas(two_step.get("second_omit_aas")),
                "sequence_by_engine": _sequence_by_engine_with_level_settings(
                    two_step.get("second_sequence_by_engine")
                    if isinstance(two_step.get("second_sequence_by_engine"), dict)
                    else {},
                    sequences_per_structure=max(1, int(two_step.get("second_sequences_per_structure", 2))),
                    sampling_temp=float(two_step.get("second_sampling_temp", 0.1)),
                    omit_aas=_normalize_omit_aas(two_step.get("second_omit_aas")),
                ),
            }
            if bool(two_step.get("run_second_level", True))
            else {"status": "skipped", "reason": "level_2_disabled"}
        ),
        "input_candidate_count": len(candidates),
        "level_1_structure_filter": first_structure_filter_summary,
        "first_pass": first_summary,
        "first_screen": screen_summary,
        "first_rmsd_filter": first_rmsd_summary,
        "first_attempt_selection": first_attempt_summary,
        "first_screen_passing_sequence_count": len(first_passing),
        "first_screen_selected_count": len(selected),
        "level_2_structure_filter": second_structure_filter_summary,
        "second_pass": second_summary,
        "second_screen": second_screen_summary,
        "second_rmsd_filter": second_rmsd_summary,
        "refined_candidate_count": len(second_candidates),
        "second_screen_passing_count": len(second_passing),
        "native_or_first_stage_output_count": len(selected),
        "include_first_stage_outputs": include_first_stage,
        "output_candidate_count": len(final_candidates),
    }
    summary_dir = run_dir / "artifacts" / "design_campaign" / "staged_two_step"
    if artifact_label:
        summary_dir = summary_dir / _safe_candidate_token(artifact_label)
    write_json(summary_dir / "two_step_summary.json", summary)
    _write_candidate_jsonl(summary_dir / "two_step_candidates.jsonl", final_candidates)
    return final_candidates, summary


def run_campaign_evaluation(
    run_dir: Path,
    candidates: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    gpu_device: str,
) -> dict[str, Any]:
    mode = str(config.get("mode") or "none")
    if mode == "none":
        return {"mode": "none", "status": "skipped", "candidate_count": len(candidates)}
    if mode not in {"metrics_only", "refold_and_metrics"}:
        raise ValueError(f"Unsupported campaign evaluation mode: {mode}")

    run_ipsae = bool(config.get("ipsae", True))
    run_rosetta = bool(config.get("rosetta", False))
    run_pymol = bool(config.get("pymol", False))
    refolders = [
        str(refolder)
        for refolder in (config.get("refolders") or [])
        if str(refolder).strip()
    ]
    if mode == "refold_and_metrics" and len(refolders) > 1:
        summaries = []
        for refolder in refolders:
            child_config = dict(config)
            child_config["refolder"] = refolder
            child_config["refolders"] = []
            if config.get("artifact_label_prefix"):
                child_config["artifact_label_prefix"] = (
                    f"{_safe_candidate_token(config.get('artifact_label_prefix'))}_"
                    f"{_safe_candidate_token(refolder)}"
                )
            summaries.append(
                run_campaign_evaluation(
                    run_dir,
                    candidates,
                    config=child_config,
                    gpu_device=gpu_device,
                )
            )
        completed = [row for row in summaries if row.get("status") == "completed"]
        evaluation_dir = run_dir / "artifacts" / "design_campaign" / "evaluation"
        if config.get("artifact_label_prefix"):
            evaluation_dir = evaluation_dir / _safe_candidate_token(config.get("artifact_label_prefix"))
        summary = {
            "mode": mode,
            "status": "completed" if completed else "failed",
            "candidate_count": len(candidates),
            "refolder": ",".join(refolders),
            "refolders": refolders,
            "ipsae": run_ipsae,
            "rosetta": run_rosetta,
            "pymol": run_pymol,
            "child_runs": [
                {
                    "refolder": row.get("refolder", ""),
                    "child_run": row.get("child_run", ""),
                    "child_run_id": row.get("child_run_id", ""),
                    "status": row.get("status", ""),
                    "error": row.get("error", ""),
                }
                for row in summaries
            ],
            "error": "" if completed else "; ".join(str(row.get("error") or "") for row in summaries),
        }
        write_json(evaluation_dir / "evaluation_summary.json", summary)
        return summary
    if mode == "metrics_only" and not any((run_ipsae, run_rosetta, run_pymol)):
        return {
            "mode": mode,
            "status": "skipped",
            "candidate_count": len(candidates),
            "reason": "No evaluation metrics were selected.",
        }
    if not candidates:
        evaluation_dir = run_dir / "artifacts" / "design_campaign" / "evaluation"
        if config.get("artifact_label_prefix"):
            evaluation_dir = evaluation_dir / _safe_candidate_token(config.get("artifact_label_prefix"))
        summary = {
            "mode": mode,
            "status": "completed",
            "result": "zero_survivors",
            "candidate_count": 0,
            "reason": "No candidates survived the configured campaign screens; evaluation skipped.",
        }
        write_json(evaluation_dir / "evaluation_summary.json", summary)
        return summary

    from mn_protein_design.workflows import benchmark as benchmark_workflow

    evaluation_dir = run_dir / "artifacts" / "design_campaign" / "evaluation"
    if config.get("artifact_label_prefix"):
        evaluation_dir = evaluation_dir / _safe_candidate_token(config.get("artifact_label_prefix"))
    source_jsonl = evaluation_dir / "source_candidates.jsonl"
    _write_candidate_jsonl(source_jsonl, candidates)
    refolder = str(config.get("refolder") or (refolders[0] if refolders else "AF3"))
    run_refolding = mode == "refold_and_metrics"
    if run_refolding and refolder not in REFOLDER_FLAG_BY_LABEL:
        raise ValueError(f"Unsupported campaign evaluation refolder: {refolder}")
    refolder_flags = {flag: False for flag in set(REFOLDER_FLAG_BY_LABEL.values())}
    if run_refolding:
        refolder_flags[REFOLDER_FLAG_BY_LABEL[refolder]] = True
    update_status(
        run_dir,
        "running",
        current_phase="Evaluation",
        current_engine=refolder if run_refolding else "Metrics",
        progress_label=(
            f"Refolding and evaluating {len(candidates)} candidates with {refolder}"
            if run_refolding
            else f"Evaluating {len(candidates)} existing candidate structures"
        ),
    )
    child_run = benchmark_workflow.run_de_novo_binder_scoring_dataset(
        source_run_dir=run_dir,
        candidates_jsonl=source_jsonl,
        selected_candidate_ids=[str(candidate.get("candidate_id") or "") for candidate in candidates],
        mode="seq_only_csv",
        generate_inputs=True,
        models=[],
        run_common_interface_metrics=run_ipsae,
        run_pyrosetta_input_metrics=run_rosetta,
        run_predicted_rosetta_metrics=run_rosetta and run_refolding,
        run_pymol_metrics=run_pymol,
        pyrosetta_nprocs=max(1, int(config.get("pyrosetta_nprocs", 4))),
        num_loops=max(1, int(config.get("num_recycles", 10))),
        num_sampling_steps=max(1, int(config.get("num_sampling_steps", 68))),
        seed=max(0, int(config.get("seed", 0))),
        device="cuda",
        esmfold2_modes=["sequence"],
        esmfold2_use_target_msa=bool(config.get("use_target_msa", _refolder_default_use_target_msa(refolder))),
        af2_num_recycles=max(1, int(config.get("num_recycles", 3))),
        af2_multimer=bool(config.get("af2_multimer", True)),
        af2_use_initial_guess=bool(config.get("af2_use_initial_guess", True)),
        af2_use_binder_template=bool(config.get("af2_use_binder_template", False)),
        af2_use_interface_template=bool(config.get("af2_use_interface_template", False)),
        colabfold_num_recycles=max(1, int(config.get("num_recycles", 3))),
        colabfold_num_models=max(1, int(config.get("num_samples", 3))),
        colabfold_use_target_templates=bool(config.get("colabfold_use_target_templates", True)),
        colabfold_max_template_hits=max(1, int(config.get("colabfold_max_template_hits", 4))),
        boltz2_use_target_template=bool(config.get("boltz2_use_target_template", True)),
        boltz2_use_target_msa=bool(config.get("use_target_msa", _refolder_default_use_target_msa(refolder))),
        boltz2_recycling_steps=max(1, int(config.get("num_recycles", 10))),
        boltz2_sampling_steps=max(1, int(config.get("num_sampling_steps", 200))),
        boltz2_diffusion_samples=max(1, int(config.get("num_samples", 3))),
        boltz2_write_full_pae=bool(config.get("boltz2_write_full_pae", True)),
        rf3_use_target_msa=bool(config.get("use_target_msa", _refolder_default_use_target_msa(refolder))),
        rf3_use_target_template=bool(config.get("rf3_use_target_template", True)),
        rf3_recycles=max(2, int(config.get("num_recycles", 10))),
        rf3_num_steps=max(1, int(config.get("num_sampling_steps", 50))),
        rf3_diffusion_batch_size=max(1, int(config.get("num_samples", 5))),
        rf3_seed=max(0, int(config.get("seed", 0))),
        openfold3_use_target_msa=bool(config.get("use_target_msa", _refolder_default_use_target_msa(refolder))),
        openfold3_num_diffusion_samples=max(1, int(config.get("num_samples", 5))),
        openfold3_num_model_seeds=max(1, int(config.get("openfold3_num_model_seeds", 1))),
        openfold3_num_recycles=max(1, int(config.get("num_recycles", 3))),
        openfold3_use_msa_server=bool(config.get("openfold3_use_msa_server", False)),
        protenix_use_msa=bool(config.get("use_target_msa", _refolder_default_use_target_msa(refolder))),
        protenix_cycle=max(1, int(config.get("num_recycles", 3))),
        protenix_diffusion_steps=max(1, int(config.get("num_sampling_steps", 50))),
        protenix_samples=max(1, int(config.get("num_samples", 5))),
        protenix_v1_model_name=str(config.get("protenix_v1_model_name") or refolding_workflow.PROTENIX_V1_MODEL),
        protenix_v1_use_msa=bool(config.get("use_target_msa", _refolder_default_use_target_msa(refolder))),
        protenix_v1_use_template=bool(config.get("protenix_v1_use_template", True)),
        protenix_v1_cycle=max(1, int(config.get("num_recycles", 10))),
        protenix_v1_diffusion_steps=max(1, int(config.get("num_sampling_steps", 200))),
        protenix_v1_samples=max(1, int(config.get("num_samples", 5))),
        protenix_v2_model_name=str(config.get("protenix_v2_model_name") or refolding_workflow.PROTENIX_V2_MODEL),
        protenix_v2_use_msa=bool(config.get("use_target_msa", _refolder_default_use_target_msa(refolder))),
        protenix_v2_use_template=bool(config.get("protenix_v2_use_template", True)),
        protenix_v2_cycle=max(1, int(config.get("num_recycles", 10))),
        protenix_v2_diffusion_steps=max(1, int(config.get("num_sampling_steps", 200))),
        protenix_v2_samples=max(1, int(config.get("num_samples", 5))),
        boltzgen_recycling_steps=max(1, int(config.get("num_recycles", 3))),
        boltzgen_sampling_steps=max(1, int(config.get("num_sampling_steps", 200))),
        boltzgen_diffusion_samples=max(1, int(config.get("num_samples", 5))),
        alphafast_num_recycles=max(1, int(config.get("num_recycles", 10))),
        alphafast_use_target_templates=bool(config.get("alphafast_use_target_templates", True)),
        alphafast_query_only_msa=not bool(config.get("alphafast_use_target_msa", config.get("use_target_msa", True))),
        alphafast_gpu_device=gpu_device,
        gpu_device=gpu_device,
        max_records=0,
        job_type="design_campaign_evaluation",
        tool_name="design_campaign_evaluation",
        internal_parent_run_dir=run_dir,
        internal_parent_task_group=DESIGN_CAMPAIGN_GROUP,
        internal_parent_role="campaign_evaluation",
        internal_parent_engine=refolder if run_refolding else "metrics_only",
        **refolder_flags,
    )
    mark_internal_job(
        child_run,
        parent_run_dir=run_dir,
        parent_task_group=DESIGN_CAMPAIGN_GROUP,
        role="campaign_evaluation",
        engine=refolder if run_refolding else "metrics_only",
    )
    result = read_json(child_run / "result.json")
    outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    copied_outputs: dict[str, str] = {}
    for name in (
        "merged_benchmark_metrics",
        "common_interface_metrics",
        "input_rosetta_metrics",
        "predicted_rosetta_metrics",
    ):
        value = outputs.get(name)
        if not isinstance(value, str) or not value:
            continue
        source = child_run / value
        if source.exists() and source.is_file():
            destination = evaluation_dir / source.name
            shutil.copy2(source, destination)
            copied_outputs[name] = str(destination.relative_to(run_dir))
    summary = {
        "mode": mode,
        "status": "completed" if result.get("success") is True else "failed",
        "candidate_count": len(candidates),
        "refolder": refolder if run_refolding else "",
        "ipsae": run_ipsae,
        "rosetta": run_rosetta,
        "pymol": run_pymol,
        "child_run": str(child_run),
        "child_run_id": child_run.name,
        "copied_outputs": copied_outputs,
        "engine_artifacts": metrics.get("engine_artifacts", {}),
        "error": "" if result.get("success") is True else str(metrics.get("error") or "Evaluation failed"),
    }
    write_json(evaluation_dir / "evaluation_summary.json", summary)
    return summary


def _run_staged_generator_child_pipeline(
    run_dir: Path,
    child: dict[str, Any],
    *,
    pipeline_index: int,
    refinement_config: dict[str, Any],
    evaluation_config: dict[str, Any],
    random_seed: int,
    gpu_device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    engine = str(child.get("engine") or "")
    run_id = str(child.get("run_id") or Path(str(child.get("run_dir") or "")).name)
    label = (
        f"child_{pipeline_index:04d}_"
        f"{_safe_candidate_token(engine)}_"
        f"{_safe_candidate_token(run_id)}"
    )
    pipelines_dir = run_dir / "artifacts" / "design_campaign" / "staged_child_pipelines"
    pipelines_dir.mkdir(parents=True, exist_ok=True)
    update_status(
        run_dir,
        "running",
        current_phase="Staged child pipeline",
        current_engine=ENGINE_LABELS.get(engine, engine),
        progress_label=f"Processing generator child {pipeline_index}: {ENGINE_LABELS.get(engine, engine)}",
    )
    harmonized, workflow_summaries = harmonize_child_candidates(
        [child],
        survivors_per_engine=1_000_000,
        passing_only=False,
        keep_best_failed=True,
    )
    for summary in workflow_summaries:
        summary["child_pipeline_label"] = label
        summary["child_pipeline_index"] = pipeline_index
        summary["run_id"] = run_id
    if not harmonized:
        summary = {
            "mode": "staged_child_pipeline",
            "status": "failed",
            "result": "no_generator_candidates",
            "artifact_label": label,
            "child_pipeline_index": pipeline_index,
            "engine": engine,
            "run_id": run_id,
            "harmonized_candidate_count": 0,
            "output_candidate_count": 0,
            "workflow_summaries": workflow_summaries,
            "error": child.get("error", ""),
        }
        write_json(pipelines_dir / f"{label}.json", summary)
        _write_candidate_jsonl(pipelines_dir / f"{label}.jsonl", [])
        return [], summary

    child_refinement_config = dict(refinement_config)
    child_refinement_config["seed"] = random_seed
    child_refinement_config["artifact_label_prefix"] = label
    if isinstance(child_refinement_config.get("staged_two_step"), dict):
        child_two_step = dict(child_refinement_config["staged_two_step"])
        child_two_step["artifact_label_prefix"] = label
        child_refinement_config["staged_two_step"] = child_two_step

    try:
        refined_candidates, refinement_summary = run_staged_two_step_sequence_refinement(
            run_dir,
            harmonized,
            config=child_refinement_config,
            gpu_device=gpu_device,
        )
    except Exception as exc:
        refined_candidates = []
        refinement_summary = {
            "mode": "staged_two_step",
            "status": "failed",
            "result": "child_refinement_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }

    refined_candidates = [_annotate_campaign_chain_roles(run_dir, candidate) for candidate in refined_candidates]
    child_evaluation_config = dict(evaluation_config)
    child_evaluation_config["seed"] = random_seed
    child_evaluation_config["artifact_label_prefix"] = label
    try:
        evaluation_summary = run_campaign_evaluation(
            run_dir,
            refined_candidates,
            config=child_evaluation_config,
            gpu_device=gpu_device,
        )
    except Exception as exc:
        evaluation_summary = {
            "mode": str(evaluation_config.get("mode") or "none"),
            "status": "failed",
            "candidate_count": len(refined_candidates),
            "error": f"{type(exc).__name__}: {exc}",
        }
        evaluation_dir = run_dir / "artifacts" / "design_campaign" / "evaluation" / _safe_candidate_token(label)
        write_json(evaluation_dir / "evaluation_summary.json", evaluation_summary)

    for candidate in refined_candidates:
        metrics = candidate.setdefault("metrics", {})
        metrics.update(
            {
                "campaign_evaluation_mode": evaluation_summary.get("mode"),
                "campaign_evaluation_status": evaluation_summary.get("status"),
                "campaign_evaluation_refolder": evaluation_summary.get("refolder", ""),
                "staged_child_pipeline_index": pipeline_index,
            }
        )
        metadata = candidate.setdefault("raw_metadata", {})
        metadata["campaign_evaluation_child_run"] = evaluation_summary.get("child_run", "")
        metadata["staged_child_pipeline_label"] = label
        metadata["staged_child_pipeline_index"] = pipeline_index
        metadata["staged_child_generator_run_id"] = run_id

    status = "completed" if refined_candidates else str(refinement_summary.get("status") or "failed")
    summary = {
        "mode": "staged_child_pipeline",
        "status": status,
        "result": "survivors_found" if refined_candidates else "zero_survivors",
        "artifact_label": label,
        "child_pipeline_index": pipeline_index,
        "engine": engine,
        "label": ENGINE_LABELS.get(engine, engine),
        "run_dir": child.get("run_dir", ""),
        "run_id": run_id,
        "harmonized_candidate_count": len(harmonized),
        "output_candidate_count": len(refined_candidates),
        "workflow_summaries": workflow_summaries,
        "sequence_refinement": refinement_summary,
        "evaluation": evaluation_summary,
        "error": child.get("error", ""),
    }
    write_json(pipelines_dir / f"{label}.json", summary)
    _write_candidate_jsonl(pipelines_dir / f"{label}.jsonl", refined_candidates)
    update_status(
        run_dir,
        "running",
        current_phase="Staged child pipeline",
        current_engine=ENGINE_LABELS.get(engine, engine),
        progress_label=(
            f"Finished generator child {pipeline_index}: "
            f"{len(refined_candidates)} final candidate(s)"
        ),
    )
    return refined_candidates, summary


def _run_engine(
    engine: str,
    *,
    target_pdb: Path,
    target_chains: list[str],
    binder_length: str,
    hotspots: str,
    campaign_name: str,
    design_attempts: int,
    sequences_per_backbone: int,
    random_seed: int,
    gpu_device: str,
    config: dict[str, Any],
    parent_run_dir: Path | None = None,
) -> Path | list[Path]:
    runners: dict[str, Callable[..., Path]] = {
        "bindcraft": run_bindcraft,
        "rfdiffusion3_foundry": run_rfdiffusion3_foundry,
        "boltzgen": run_boltzgen,
        "rfdiffusion_classic": run_rfdiffusion_classic,
        "pxdesign": run_pxdesign,
        "genie3": run_genie3,
        "esmfold2_binder_design": run_esmfold2_native_binder_design,
        "protpardelle_1c": run_protpardelle_1c,
        "proteina_complexa": run_proteina_complexa,
    }
    if engine not in runners and engine != "bindcraft2":
        raise ValueError(f"Unsupported design campaign engine: {engine}")
    staged_design_only = str(config.get("campaign_workflow_recipe") or "") == "staged_backbone_sequence_refold"

    shared: dict[str, Any] = {
        "target_pdb": target_pdb,
        "hotspots": hotspots,
        "campaign_name": campaign_name,
        "gpu_device": gpu_device,
    }
    if engine == "bindcraft2":
        from mn_protein_design.workflows.bindcraft2 import run_bindcraft2_campaign

        if parent_run_dir is None:
            raise ValueError("BindCraft 2 campaign workflows require their parent campaign run directory.")
        return run_bindcraft2_campaign(
            parent_run_dir=parent_run_dir,
            target_pdb=target_pdb,
            target_chains=target_chains,
            binder_length=(config.get("binder_length") if "binder_length" in config else binder_length),
            hotspots=str(config.get("hotspots") or hotspots),
            modality=config.get("modality") or "binder",
            design_properties=list(config.get("design_properties") or []),
            number_of_final_designs=int(config.get("number_of_final_designs", design_attempts)),
            max_trajectories=int(config.get("max_trajectories", design_attempts)),
            campaign_seed=int(config.get("campaign_seed", random_seed)),
            workers_per_gpu=config.get("workers_per_gpu", "auto"),
            cpu_cores=int(config.get("cpu_cores", 1)),
            gpu_device=gpu_device,
        )
    if engine == "esmfold2_binder_design":
        requested_length = str(config.get("binder_length") or binder_length)
        sampled_lengths = _sample_fixed_binder_lengths(
            requested_length,
            seed=random_seed,
            engine=engine,
            attempts=design_attempts,
        )
        outputs_per_attempt = max(1, int(config.get("generator_outputs_per_attempt", 1)))
        child_runs: list[Path] = []
        for attempt_index, fixed_length in enumerate(sampled_lengths, start=1):
            for output_index in range(1, outputs_per_attempt + 1):
                child_run = runners[engine](
                    **shared,
                    target_chain=str(config.get("target_chain") or target_chains[0]),
                    binder_length=fixed_length,
                    num_designs=1,
                    optimization_steps=int(config.get("optimization_steps", 150)),
                    learning_rate=float(config.get("learning_rate", 0.1)),
                    seed=random_seed + ((attempt_index - 1) * outputs_per_attempt) + output_index - 1,
                    compile_model=bool(config.get("compile_model", False)),
                    checkpoint_lm=bool(config.get("checkpoint_lm", False)),
                )
                _record_sampled_binder_length(
                    child_run,
                    requested_length=requested_length,
                    sampled_length=fixed_length,
                    attempt_index=attempt_index,
                    total_attempts=len(sampled_lengths),
                    output_index=output_index,
                    outputs_per_attempt=outputs_per_attempt,
                )
                child_runs.append(child_run)
        return child_runs

    shared.update({"target_chains": target_chains, "binder_length": binder_length})
    if engine == "rfdiffusion_classic":
        contig = str(config.get("contig") or default_target_contig(target_pdb, target_chains, binder_length))
        return runners[engine](
            **shared,
            contig=contig,
            num_designs=design_attempts,
            timesteps=int(config.get("timesteps", 50)),
            model_weights=str(config.get("model_weights", "Complex_base")),
            rfdiffusion_guidance_preset=str(config.get("guidance_preset", "none")),
            scaffoldguided=bool(config.get("scaffoldguided", False)),
            mpnn_num_sequences=max(1, int(config.get("mpnn_num_sequences", sequences_per_backbone))),
            sequence_design_method=str(config.get("sequence_design_method", "protein_mpnn")),
            execution_backend=str(config.get("execution_backend", "docker")),
            run_vanilla_pipeline=not staged_design_only,
            monomer_refolding_tool=str(config.get("monomer_refolding_tool", "boltz2_monomer")),
            complex_refolding_tool=str(config.get("complex_refolding_tool", "af2_initial_guess")),
            complex_template_mode=str(config.get("complex_template_mode", "target_template")),
            complex_multimer=bool(config.get("complex_multimer", True)),
            complex_num_recycles=max(1, int(config.get("complex_num_recycles", 3))),
            analysis_keep_top_n=max(1, int(config.get("analysis_keep_top_n", design_attempts))),
            analysis_keep_per_attempt=(
                max(1, int(config["analysis_keep_per_attempt"]))
                if config.get("analysis_keep_per_attempt") not in {None, ""}
                else None
            ),
            analysis_rank_metric=str(config.get("analysis_rank_metric", "iptm")),
            analysis_thresholds=dict(config.get("analysis_thresholds") or {}),
        )
    if engine == "bindcraft":
        bindcraft_settings_file = str(config.get("advanced_settings_file", "default_4stage_multimer_mpnn.json"))
        if staged_design_only:
            bindcraft_settings_file = _bindcraft_generator_settings_file(bindcraft_settings_file)
        return runners[engine](
            **shared,
            number_of_final_designs=min(int(config.get("number_of_final_designs", 1)), design_attempts),
            num_seqs_override=sequences_per_backbone,
            max_mpnn_sequences_override=sequences_per_backbone,
            time_limit_seconds=int(config["time_limit_seconds"]) if config.get("time_limit_seconds") else None,
            enable_mpnn=not staged_design_only,
            max_trajectories=design_attempts,
            filter_settings=str(config.get("filter_settings", "default_filters.json")),
            advanced_settings_file=bindcraft_settings_file,
        )
    if engine == "rfdiffusion3_foundry":
        return runners[engine](
            **shared,
            num_designs=design_attempts,
            timesteps=int(config.get("timesteps", 50)),
            run_vanilla_pipeline=not staged_design_only,
            mpnn_sequences_per_backbone=sequences_per_backbone,
            mpnn_model_type=str(config.get("mpnn_model_type", "protein_mpnn")),
            mpnn_checkpoint_path=str(config.get("mpnn_checkpoint_path", "/weights/proteinmpnn_v_48_020.pt")),
            prepare_target_msa=bool(config.get("prepare_target_msa", True)) and not staged_design_only,
        )
    if engine == "boltzgen":
        return runners[engine](
            **shared,
            num_designs=design_attempts,
            budget=min(int(config.get("budget", 1)), design_attempts),
            sampling_steps=int(config.get("sampling_steps", 20)),
            run_vanilla_pipeline=not staged_design_only,
        )
    if engine == "pxdesign":
        requested_length = str(config.get("binder_length") or binder_length)
        sampled_lengths = _sample_fixed_binder_lengths(
            requested_length,
            seed=random_seed,
            engine=engine,
            attempts=design_attempts,
        )
        child_runs: list[Path] = []
        for attempt_index, fixed_length in enumerate(sampled_lengths, start=1):
            px_shared = {**shared, "binder_length": str(fixed_length)}
            child_run = runners[engine](
                **px_shared,
                num_designs=1,
                n_steps=int(config.get("n_steps", 400)),
                dtype=str(config.get("dtype", "bf16")),
                preset=str(config.get("preset", "preview")),
                run_mode="generation_only" if staged_design_only else "pipeline",
                n_max_runs=int(config.get("n_max_runs", 1)),
                use_fast_ln=bool(config.get("use_fast_ln", True)),
                use_deepspeed_evo_attention=bool(config.get("use_deepspeed_evo_attention", False)),
                prepare_target_msa=bool(config.get("prepare_target_msa", True)) and not staged_design_only,
            )
            _record_sampled_binder_length(
                child_run,
                requested_length=requested_length,
                sampled_length=fixed_length,
                attempt_index=attempt_index,
                total_attempts=len(sampled_lengths),
            )
            child_runs.append(child_run)
        return child_runs
    if engine == "genie3":
        folding_model = str(config.get("folding_model_name", "colabfold"))
        return runners[engine](
            **shared,
            num_designs=design_attempts,
            seed=random_seed,
            direction_scale=float(config.get("direction_scale", 0.0)),
            cond_strategy=str(config.get("cond_strategy", "extended")),
            inverse_folding_num_seq=sequences_per_backbone,
            folding_model_name=folding_model,
            folding_mode=str(config.get("folding_mode", "msa" if folding_model == "boltz2" else "template")),
            folding_num_models=int(config.get("folding_num_models", 5)),
            folding_num_recycles=int(config.get("folding_num_recycles", 20)),
            compile_generation=bool(config.get("compile_generation", False)),
            run_mode="generation_only" if staged_design_only else "full_vanilla_pipeline",
            enable_beam_search=bool(config.get("enable_beam_search", False)),
            beam_width=int(config.get("beam_width", 4)),
        )
    if engine == "protpardelle_1c":
        return runners[engine](
            **shared,
            num_designs=design_attempts,
            num_mpnn_seqs=0 if staged_design_only else sequences_per_backbone,
            model_name=str(config.get("model_name", "cc83")),
            model_epoch=str(config.get("model_epoch", "2616")),
            sampling_config=str(config.get("sampling_config", "sampling_sidechain_conditional")),
            step_scale=float(config.get("step_scale", 1.2)),
            schurn=int(config.get("schurn", 200)),
            crop_cond_start=float(config.get("crop_cond_start", 0.0)),
            batch_size=int(config.get("batch_size", 1)),
            seed=random_seed,
        )
    return runners[engine](
        **shared,
        num_designs=design_attempts * max(1, int(config.get("generator_outputs_per_attempt", 1))),
        n_steps=int(config.get("n_steps", 400)),
        replicas=int(config.get("replicas", 2)),
        seed=random_seed,
        batch_size=int(config.get("batch_size", 1)),
        generator_only=staged_design_only,
    )


def _write_summary_csv(run_dir: Path, candidates: list[dict[str, Any]]) -> Path:
    output = run_dir / "artifacts" / "design_campaign" / "harmonized_candidates.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "design_campaign_rank",
        "engine",
        "engine_rank",
        "candidate_id",
        "stage",
        "native_pass",
        "staged_sequence_level",
        "staged_output_role",
        "target_aligned_binder_rmsd",
        "first_screen_rmsd",
        "first_screen_max_rmsd",
        "first_screen_pass",
        "first_screen_failure_reason",
        "second_screen_rmsd",
        "second_screen_max_rmsd",
        "second_screen_pass",
        "second_screen_failure_reason",
        "binder_sequence",
        "complex_pdb",
        "binder_pdb",
    ]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            metrics = candidate.get("metrics") or {}
            metadata = candidate.get("raw_metadata") or {}
            writer.writerow(
                {
                    "design_campaign_rank": metrics.get("design_campaign_rank"),
                    "engine": metadata.get("design_campaign_engine"),
                    "engine_rank": metrics.get("design_campaign_engine_rank"),
                    "candidate_id": candidate.get("candidate_id"),
                    "stage": candidate.get("stage"),
                    "native_pass": metrics.get("harmonized_native_pass"),
                    "staged_sequence_level": metrics.get("staged_sequence_level"),
                    "staged_output_role": metrics.get("staged_output_role"),
                    "target_aligned_binder_rmsd": metrics.get("target_aligned_binder_rmsd"),
                    "first_screen_rmsd": metrics.get("first_refolding_screen_screen_rmsd"),
                    "first_screen_max_rmsd": metrics.get("first_refolding_screen_screen_max_rmsd"),
                    "first_screen_pass": metrics.get("first_refolding_screen_screen_pass"),
                    "first_screen_failure_reason": metrics.get("first_refolding_screen_screen_failure_reason"),
                    "second_screen_rmsd": metrics.get("second_refolding_screen_screen_rmsd"),
                    "second_screen_max_rmsd": metrics.get("second_refolding_screen_screen_max_rmsd"),
                    "second_screen_pass": metrics.get("second_refolding_screen_screen_pass"),
                    "second_screen_failure_reason": metrics.get("second_refolding_screen_screen_failure_reason"),
                    "binder_sequence": candidate.get("binder_sequence"),
                    "complex_pdb": candidate.get("complex_pdb"),
                    "binder_pdb": candidate.get("binder_pdb"),
                }
            )
    return output


def _write_staged_stream_snapshot(
    run_dir: Path,
    *,
    campaign_name: str,
    workflow_recipe: str,
    target_pdb: Path,
    target_chains: list[str],
    engines: list[str],
    template_config: dict[str, Any],
    child_runs: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    common_summary: dict[str, Any],
    refinement_summary: dict[str, Any],
    evaluation_summary: dict[str, Any],
    harmonized_candidate_count: int,
) -> None:
    for final_rank, candidate in enumerate(candidates, start=1):
        metrics = candidate.setdefault("metrics", {})
        if "design_campaign_rank" in metrics and "design_campaign_parent_rank" not in metrics:
            metrics["design_campaign_parent_rank"] = metrics["design_campaign_rank"]
        metrics["design_campaign_rank"] = final_rank
    normalized = write_candidates(run_dir, "design_campaign", candidates)
    csv_path = _write_summary_csv(run_dir, normalized)
    summary_path = run_dir / "artifacts" / "design_campaign" / "workflow_summary.json"
    write_json(
        summary_path,
        {
            "campaign_name": campaign_name,
            "workflow_recipe": workflow_recipe,
            "target_pdb": str(target_pdb),
            "target_chains": target_chains,
            "engines": engines,
            "template_redesign": template_config,
            "workflows": summaries,
            "candidate_count": len(normalized),
            "harmonized_candidate_count": harmonized_candidate_count,
            "common_validation": common_summary,
            "sequence_refinement": refinement_summary,
            "evaluation": evaluation_summary,
            "streaming_snapshot": True,
        },
    )
    partial_result = {
        "success": False,
        "running": True,
        "outputs": {
            "candidates": normalized,
            "child_runs": child_runs,
            "workflow_summary": str(summary_path.relative_to(run_dir)),
            "harmonized_candidates_csv": str(csv_path.relative_to(run_dir)),
        },
        "metrics": {
            "engine_count": len(engines),
            "completed_engine_count": sum(row.get("status") == "completed" for row in child_runs),
            "failed_engine_count": sum(row.get("status") == "failed" for row in child_runs),
            "candidate_count": len(normalized),
            "harmonized_candidate_count": harmonized_candidate_count,
            "sequence_refinement_status": refinement_summary.get("status"),
            "refined_candidate_count": refinement_summary.get("refined_candidate_count", refinement_summary.get("output_candidate_count", 0)),
            "evaluation_status": evaluation_summary.get("status"),
            "evaluation_mode": evaluation_summary.get("mode"),
            **common_summary,
        },
        "downstream_artifacts": {
            "target_pdb": str(target_pdb),
            "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
            "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
        },
    }
    write_json(run_dir / "result.json", partial_result)


def _staged_stream_summaries(
    *,
    evaluation_config: dict[str, Any],
    readable_child_count: int,
    streamed_pipeline_summaries: list[dict[str, Any]],
    streamed_harmonized_count: int,
    candidate_count: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    common_summary = {
        "status": "skipped",
        "reason": "staged workflow streams each completed generator child through sequence/refold/evaluation",
    }
    completed_pipelines = [
        row for row in streamed_pipeline_summaries
        if str(row.get("status") or "") == "completed"
    ]
    refinement_summary = {
        "mode": "staged_streaming_by_generator_child",
        "status": "completed" if candidate_count or completed_pipelines else "failed",
        "result": "survivors_found" if candidate_count else "zero_survivors",
        "input_child_count": readable_child_count,
        "child_pipeline_count": len(streamed_pipeline_summaries),
        "completed_child_pipeline_count": len(completed_pipelines),
        "failed_child_pipeline_count": sum(
            1 for row in streamed_pipeline_summaries
            if str(row.get("status") or "") == "failed"
        ),
        "harmonized_candidate_count": streamed_harmonized_count,
        "refined_candidate_count": candidate_count,
        "output_candidate_count": candidate_count,
        "child_pipelines": streamed_pipeline_summaries,
        "attempt_summaries": [
            attempt
            for pipeline in streamed_pipeline_summaries
            for attempt in (
                (pipeline.get("sequence_refinement") or {}).get("attempt_summaries") or []
            )
        ],
    }
    evaluation_summary = {
        "mode": str(evaluation_config.get("mode") or "none"),
        "status": "completed" if streamed_pipeline_summaries else "skipped",
        "candidate_count": candidate_count,
        "child_evaluations": [
            row.get("evaluation") or {}
            for row in streamed_pipeline_summaries
        ],
    }
    return common_summary, refinement_summary, evaluation_summary


def run_design_campaign(run_dir: Path) -> Path:
    run_dir = run_dir.expanduser().resolve()
    payload = read_json(run_dir / "input.json")
    inputs = payload.get("inputs") or {}
    params = payload.get("params") or {}
    target_pdb = Path(str(inputs.get("target_pdb") or "")).expanduser()
    target_chains = [str(chain) for chain in inputs.get("target_chains") or []]
    engines = [
        engine
        for engine in params.get("engines") or []
        if engine in ENGINE_ORDER and engine != "template_redesign"
    ]
    engine_configs = params.get("engine_configs") if isinstance(params.get("engine_configs"), dict) else {}
    campaign_name = str(params.get("campaign_name") or "").strip()
    workflow_recipe = str(params.get("workflow_recipe") or "vanilla")
    if workflow_recipe in GENERATOR_ONLY_RECIPES and set(engines).intersection(VANILLA_ONLY_ENGINES):
        unavailable = ", ".join(ENGINE_LABELS[engine] for engine in sorted(set(engines).intersection(VANILLA_ONLY_ENGINES)))
        raise ValueError(f"{unavailable} is currently available only in the vanilla campaign workflow.")
    continue_after_failure = bool(params.get("continue_after_failure", True))
    design_attempts = max(1, int(params.get("design_attempts", 1)))
    sequences_per_backbone = max(1, int(params.get("sequences_per_backbone", 1)))
    random_seed = max(0, int(params.get("random_seed", 0)))
    common_config = params.get("common_validation") if isinstance(params.get("common_validation"), dict) else {}
    template_config = (
        params.get("template_redesign")
        if isinstance(params.get("template_redesign"), dict)
        else {"enabled": False}
    )
    template_enabled = bool(template_config.get("enabled"))
    refinement_config = (
        params.get("sequence_refinement")
        if isinstance(params.get("sequence_refinement"), dict)
        else {"mode": "none"}
    )
    evaluation_config = (
        params.get("evaluation")
        if isinstance(params.get("evaluation"), dict)
        else {"mode": "none"}
    )
    if workflow_recipe == "staged":
        refinement_config = _normalize_staged_sequence_refinement_config(
            refinement_config,
            engines,
            sequences_per_backbone=sequences_per_backbone,
        )
    if template_enabled:
        template_mode_defaults = {
            "interface": "redesign_non_interface",
            "all": "redesign_all",
            "selected_unlocked": "redesign_selected_unlocked",
            "selected_locked": "redesign_selected_locked",
        }
        if str(refinement_config.get("mode") or "none") == "none":
            refinement_config["mode"] = template_mode_defaults.get(
                str(template_config.get("mode") or "interface"),
                "redesign_non_interface",
            )
        refinement_config.setdefault("locked_residues", template_config.get("locked_residues", ""))
        refinement_config.setdefault("unlocked_residues", template_config.get("unlocked_residues", ""))

    if not target_pdb.exists():
        raise FileNotFoundError(f"Target PDB does not exist: {target_pdb}")
    if not target_chains:
        raise ValueError("At least one target chain is required.")
    if not engines and not template_enabled:
        raise ValueError("At least one vanilla workflow or template redesign source is required.")
    refinement_config = _restrict_fragmented_target_refolders(
        refinement_config,
        target_pdb=target_pdb,
        target_chains=target_chains,
    )
    evaluation_config = _restrict_fragmented_target_evaluation(
        evaluation_config,
        target_pdb=target_pdb,
        target_chains=target_chains,
    )

    generator_only_recipe = workflow_recipe in GENERATOR_ONLY_RECIPES
    engine_phase = "Generator workflows" if generator_only_recipe else "Vanilla workflows"
    engine_progress_action = "Generating with" if generator_only_recipe else "Running"
    stream_staged_children = workflow_recipe == "staged"
    update_status(
        run_dir,
        "running",
        campaign_name=campaign_name,
        current_phase=engine_phase,
        current_engine="",
        completed_engines=0,
        total_engines=len(engines) + int(template_enabled),
    )
    child_runs: list[dict[str, Any]] = []
    streamed_candidates: list[dict[str, Any]] = []
    streamed_pipeline_summaries: list[dict[str, Any]] = []
    streamed_workflow_summaries: list[dict[str, Any]] = []
    streamed_harmonized_count = 0
    for index, engine in enumerate(engines, start=1):
        label = ENGINE_LABELS[engine]
        update_status(
            run_dir,
            "running",
            current_phase=engine_phase,
            current_engine=label,
            progress_label=f"{engine_progress_action} {label} ({index}/{len(engines)})",
            completed_engines=index - 1,
        )
        try:
            engine_config = dict(engine_configs.get(engine) or {})
            if generator_only_recipe:
                engine_config = _generator_only_engine_config(
                    engine,
                    engine_config,
                    binder_length=str(params.get("binder_length") or "60"),
                )
            if workflow_recipe == "staged" and engine in NATIVE_SEQUENCE_GENERATOR_ENGINES:
                engine_config.setdefault(
                    "generator_outputs_per_attempt",
                    _staged_native_outputs_per_attempt(refinement_config),
                )
            if engine == "bindcraft" and bool(common_config.get("enabled", False)):
                engine_config.update(
                    {
                        "number_of_final_designs": design_attempts,
                        "filter_settings": "no_filters.json",
                    }
                )
            if engine == "boltzgen" and bool(common_config.get("enabled", False)):
                engine_config["budget"] = design_attempts
            child_run_result = _run_engine(
                engine,
                target_pdb=target_pdb,
                target_chains=target_chains,
                binder_length=str(params.get("binder_length") or "60"),
                hotspots=str(params.get("hotspots") or ""),
                campaign_name=campaign_name,
                design_attempts=design_attempts,
                sequences_per_backbone=sequences_per_backbone,
                random_seed=random_seed,
                gpu_device=str(params.get("gpu_device") or "0"),
                config=engine_config,
                parent_run_dir=run_dir,
            )
            engine_child_runs = child_run_result if isinstance(child_run_result, list) else [child_run_result]
            engine_success = True
            for child_run in engine_child_runs:
                child_result = read_json(child_run / "result.json")
                child_success = child_result.get("success") is True
                engine_success = engine_success and child_success
                child_record = {
                    "engine": engine,
                    "run_dir": str(child_run),
                    "run_id": child_run.name,
                    "status": "completed" if child_success else "failed",
                    "error": "" if child_success else str((child_result.get("metrics") or {}).get("error") or "Workflow failed"),
                }
                child_runs.append(child_record)
                mark_internal_job(
                    child_run,
                    parent_run_dir=run_dir,
                    parent_task_group=DESIGN_CAMPAIGN_GROUP,
                    role="generator_design_workflow" if generator_only_recipe else "vanilla_design_workflow",
                    engine=engine,
                )
                child_metadata = read_json(child_run / "metadata.json")
                child_metadata["design_campaign_engine"] = engine
                write_json(child_run / "metadata.json", child_metadata)
                if stream_staged_children and child_success:
                    pipeline_candidates, pipeline_summary = _run_staged_generator_child_pipeline(
                        run_dir,
                        child_record,
                        pipeline_index=len(streamed_pipeline_summaries) + 1,
                        refinement_config=refinement_config,
                        evaluation_config=evaluation_config,
                        random_seed=random_seed,
                        gpu_device=str(params.get("gpu_device") or "0"),
                    )
                    streamed_candidates.extend(pipeline_candidates)
                    streamed_pipeline_summaries.append(pipeline_summary)
                    streamed_harmonized_count += int(pipeline_summary.get("harmonized_candidate_count") or 0)
                    streamed_workflow_summaries.extend(pipeline_summary.get("workflow_summaries") or [])
                    snapshot_common, snapshot_refinement, snapshot_evaluation = _staged_stream_summaries(
                        evaluation_config=evaluation_config,
                        readable_child_count=len([row for row in child_runs if row.get("run_dir")]),
                        streamed_pipeline_summaries=streamed_pipeline_summaries,
                        streamed_harmonized_count=streamed_harmonized_count,
                        candidate_count=len(streamed_candidates),
                    )
                    _write_staged_stream_snapshot(
                        run_dir,
                        campaign_name=campaign_name,
                        workflow_recipe=workflow_recipe,
                        target_pdb=target_pdb,
                        target_chains=target_chains,
                        engines=engines,
                        template_config=template_config,
                        child_runs=child_runs,
                        summaries=streamed_workflow_summaries,
                        candidates=streamed_candidates,
                        common_summary=snapshot_common,
                        refinement_summary=snapshot_refinement,
                        evaluation_summary=snapshot_evaluation,
                        harmonized_candidate_count=streamed_harmonized_count,
                    )
            if not engine_success and not continue_after_failure:
                break
        except Exception as exc:
            child_runs.append(
                {
                    "engine": engine,
                    "run_dir": "",
                    "run_id": "",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if not continue_after_failure:
                break
        update_status(run_dir, "running", completed_engines=index)
    if template_enabled:
        update_status(
            run_dir,
            "running",
            current_phase="Template redesign",
            current_engine=ENGINE_LABELS["template_redesign"],
            progress_label="Registering uploaded target-binder complex",
        )
        try:
            template_child = _create_template_redesign_source(
                run_dir,
                target_pdb=target_pdb,
                template_config=template_config,
            )
            if template_child is not None:
                child_runs.append(template_child)
                if stream_staged_children and str(template_child.get("status") or "") == "completed":
                    pipeline_candidates, pipeline_summary = _run_staged_generator_child_pipeline(
                        run_dir,
                        template_child,
                        pipeline_index=len(streamed_pipeline_summaries) + 1,
                        refinement_config=refinement_config,
                        evaluation_config=evaluation_config,
                        random_seed=random_seed,
                        gpu_device=str(params.get("gpu_device") or "0"),
                    )
                    streamed_candidates.extend(pipeline_candidates)
                    streamed_pipeline_summaries.append(pipeline_summary)
                    streamed_harmonized_count += int(pipeline_summary.get("harmonized_candidate_count") or 0)
                    streamed_workflow_summaries.extend(pipeline_summary.get("workflow_summaries") or [])
                    snapshot_common, snapshot_refinement, snapshot_evaluation = _staged_stream_summaries(
                        evaluation_config=evaluation_config,
                        readable_child_count=len([row for row in child_runs if row.get("run_dir")]),
                        streamed_pipeline_summaries=streamed_pipeline_summaries,
                        streamed_harmonized_count=streamed_harmonized_count,
                        candidate_count=len(streamed_candidates),
                    )
                    _write_staged_stream_snapshot(
                        run_dir,
                        campaign_name=campaign_name,
                        workflow_recipe=workflow_recipe,
                        target_pdb=target_pdb,
                        target_chains=target_chains,
                        engines=engines,
                        template_config=template_config,
                        child_runs=child_runs,
                        summaries=streamed_workflow_summaries,
                        candidates=streamed_candidates,
                        common_summary=snapshot_common,
                        refinement_summary=snapshot_refinement,
                        evaluation_summary=snapshot_evaluation,
                        harmonized_candidate_count=streamed_harmonized_count,
                    )
                update_status(
                    run_dir,
                    "running",
                    completed_engines=len(engines) + 1,
                )
        except Exception as exc:
            child_runs.append(
                {
                    "engine": "template_redesign",
                    "run_dir": "",
                    "run_id": "",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if not continue_after_failure:
                raise

    readable_children = [row for row in child_runs if row.get("run_dir")]
    update_status(
        run_dir,
        "running",
        current_phase="Collect generator outputs" if generator_only_recipe else "Harmonize",
        current_engine="",
        progress_label=(
            "Collecting generator outputs for staged/scout comparison"
            if generator_only_recipe
            else "Selecting pass-first survivors"
        ),
    )
    common_summary: dict[str, Any] = {}
    if stream_staged_children:
        harmonized = []
        summaries = list(streamed_workflow_summaries)
        common_summary = {
            "status": "skipped",
            "reason": "staged workflow streams each completed generator child through sequence/refold/evaluation",
        }
    elif generator_only_recipe:
        harmonized, summaries = harmonize_child_candidates(
            readable_children,
            survivors_per_engine=(
                int(params.get("survivors_per_engine", 1_000_000))
                if workflow_recipe == "engine_scout"
                else 1_000_000
            ),
            passing_only=False,
            keep_best_failed=True,
        )
        common_summary = {
            "status": "skipped",
            "reason": (
                f"engine scout keeps up to {int(params.get('survivors_per_engine', 1_000_000))} generator outputs per engine"
                if workflow_recipe == "engine_scout"
                else "staged workflow keeps all generator outputs; filtering happens after sequence/refold screening"
            ),
        }
    elif bool(common_config.get("enabled", False)):
        validated, common_summary = run_common_bindcraft_validation(
            run_dir,
            readable_children,
            gpu_device=str(params.get("gpu_device") or "0"),
            num_recycles=int(common_config.get("num_recycles", 3)),
            min_ipsae=float(common_config.get("min_ipsae", 0.0)),
            pyrosetta_nprocs=int(common_config.get("pyrosetta_nprocs", 4)),
        )
        harmonized, summaries = harmonize_validated_candidates(
            validated,
            survivors_per_engine=int(params.get("survivors_per_engine", 1_000_000)),
            keep_best_failed=bool(params.get("keep_best_failed", True)),
        )
    else:
        harmonized, summaries = harmonize_child_candidates(
            readable_children,
            survivors_per_engine=int(params.get("survivors_per_engine", 1_000_000)),
            passing_only=bool(params.get("passing_only", False)),
            keep_best_failed=bool(params.get("keep_best_failed", True)),
        )
    summary_by_engine = {row["engine"]: row for row in summaries}
    for child in child_runs:
        if child["engine"] not in summary_by_engine:
            summaries.append(
                {
                    "engine": child["engine"],
                    "label": ENGINE_LABELS.get(child["engine"], child["engine"]),
                    "run_dir": child.get("run_dir", ""),
                    "run_id": child.get("run_id", ""),
                    "status": child.get("status", "failed"),
                    "candidate_count": 0,
                    "native_passing_count": 0,
                    "selected_count": 0,
                    "selection_mode": "not_available",
                    "error": child.get("error", ""),
                }
            )
    if stream_staged_children:
        refined_candidates = streamed_candidates
        common_summary, refinement_summary, evaluation_summary = _staged_stream_summaries(
            evaluation_config=evaluation_config,
            readable_child_count=len(readable_children),
            streamed_pipeline_summaries=streamed_pipeline_summaries,
            streamed_harmonized_count=streamed_harmonized_count,
            candidate_count=len(refined_candidates),
        )
    else:
        refinement_runner = (
            run_staged_two_step_sequence_refinement
            if workflow_recipe == "staged"
            else run_campaign_sequence_refinement
        )
        refined_candidates, refinement_summary = refinement_runner(
            run_dir,
            harmonized,
            config={
                **refinement_config,
                "seed": random_seed,
            },
            gpu_device=str(params.get("gpu_device") or "0"),
        )
        refined_candidates = [_annotate_campaign_chain_roles(run_dir, candidate) for candidate in refined_candidates]
    for final_rank, candidate in enumerate(refined_candidates, start=1):
        metrics = candidate.setdefault("metrics", {})
        if "design_campaign_rank" in metrics and "design_campaign_parent_rank" not in metrics:
            metrics["design_campaign_parent_rank"] = metrics["design_campaign_rank"]
        metrics["design_campaign_rank"] = final_rank

    if not stream_staged_children:
        try:
            evaluation_summary = run_campaign_evaluation(
                run_dir,
                refined_candidates,
                config={
                    **evaluation_config,
                    "seed": random_seed,
                },
                gpu_device=str(params.get("gpu_device") or "0"),
            )
        except Exception as exc:
            evaluation_summary = {
                "mode": str(evaluation_config.get("mode") or "none"),
                "status": "failed",
                "candidate_count": len(refined_candidates),
                "error": f"{type(exc).__name__}: {exc}",
            }
            evaluation_dir = run_dir / "artifacts" / "design_campaign" / "evaluation"
            write_json(evaluation_dir / "evaluation_summary.json", evaluation_summary)
        for candidate in refined_candidates:
            metrics = candidate.setdefault("metrics", {})
            metrics.update(
                {
                    "campaign_evaluation_mode": evaluation_summary.get("mode"),
                    "campaign_evaluation_status": evaluation_summary.get("status"),
                    "campaign_evaluation_refolder": evaluation_summary.get("refolder", ""),
                }
            )
            metadata = candidate.setdefault("raw_metadata", {})
            metadata["campaign_evaluation_child_run"] = evaluation_summary.get("child_run", "")

    normalized = write_candidates(run_dir, "design_campaign", refined_candidates)
    harmonized_candidate_count = streamed_harmonized_count if stream_staged_children else len(harmonized)
    summary_path = run_dir / "artifacts" / "design_campaign" / "workflow_summary.json"
    write_json(
        summary_path,
        {
            "campaign_name": campaign_name,
            "workflow_recipe": workflow_recipe,
            "target_pdb": str(target_pdb),
            "target_chains": target_chains,
            "engines": engines,
            "template_redesign": template_config,
            "workflows": summaries,
            "candidate_count": len(normalized),
            "harmonized_candidate_count": harmonized_candidate_count,
            "common_validation": common_summary,
            "sequence_refinement": refinement_summary,
            "evaluation": evaluation_summary,
        },
    )
    csv_path = _write_summary_csv(run_dir, normalized)
    finish_job(
        run_dir,
        bool(normalized),
        {
            "outputs": {
                "candidates": normalized,
                "child_runs": child_runs,
                "workflow_summary": str(summary_path.relative_to(run_dir)),
                "harmonized_candidates_csv": str(csv_path.relative_to(run_dir)),
            },
            "metrics": {
                "engine_count": len(engines),
                "completed_engine_count": sum(row.get("status") == "completed" for row in child_runs),
                "failed_engine_count": sum(row.get("status") == "failed" for row in child_runs),
                "candidate_count": len(normalized),
                "harmonized_candidate_count": harmonized_candidate_count,
                "sequence_refinement_status": refinement_summary.get("status"),
                "refined_candidate_count": refinement_summary.get("refined_candidate_count", 0),
                "evaluation_status": evaluation_summary.get("status"),
                "evaluation_mode": evaluation_summary.get("mode"),
                **common_summary,
            },
            "downstream_artifacts": {
                "target_pdb": str(target_pdb),
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
            },
        },
    )
    return run_dir


def create_design_campaign(
    *,
    target_pdb: Path,
    target_chains: list[str],
    binder_length: str,
    hotspots: str,
    campaign_name: str,
    design_attempts: int,
    sequences_per_backbone: int,
    random_seed: int,
    engines: list[str],
    engine_configs: dict[str, dict[str, Any]],
    workflow_recipe: str = "vanilla",
    survivors_per_engine: int = 1_000_000,
    passing_only: bool = False,
    keep_best_failed: bool = True,
    continue_after_failure: bool = True,
    common_validation: dict[str, Any] | None = None,
    sequence_refinement: dict[str, Any] | None = None,
    template_redesign: dict[str, Any] | None = None,
    evaluation: dict[str, Any] | None = None,
    gpu_device: str = "0",
) -> Path:
    selected_engines = [engine for engine in ENGINE_ORDER if engine in engines and engine != "template_redesign"]
    unsupported_generator_engines = set(selected_engines).intersection(VANILLA_ONLY_ENGINES)
    if workflow_recipe in GENERATOR_ONLY_RECIPES and unsupported_generator_engines:
        labels = ", ".join(ENGINE_LABELS[engine] for engine in sorted(unsupported_generator_engines))
        raise ValueError(f"{labels} is currently available only in the vanilla campaign workflow.")
    normalized_engine_configs = {engine: dict(config) for engine, config in (engine_configs or {}).items()}
    if workflow_recipe in GENERATOR_ONLY_RECIPES:
        for engine in selected_engines:
            normalized_engine_configs[engine] = _generator_only_engine_config(
                engine,
                normalized_engine_configs.get(engine),
                binder_length=binder_length,
            )
    if workflow_recipe == "staged":
        sequence_refinement = _normalize_staged_sequence_refinement_config(
            sequence_refinement,
            selected_engines,
            sequences_per_backbone=max(1, int(sequences_per_backbone)),
        )
        native_outputs_per_attempt = _staged_native_outputs_per_attempt(sequence_refinement)
        for engine in selected_engines:
            if engine in NATIVE_SEQUENCE_GENERATOR_ENGINES:
                normalized_engine_configs.setdefault(engine, {}).setdefault(
                    "generator_outputs_per_attempt",
                    native_outputs_per_attempt,
                )
    normalized_workflow_recipe = workflow_recipe if workflow_recipe in {"vanilla", "staged", "engine_scout"} else "vanilla"
    params = {
        "campaign_name": campaign_name.strip(),
        "binder_length": binder_length.strip(),
        "hotspots": hotspots.strip(),
        "design_attempts": max(1, int(design_attempts)),
        "sequences_per_backbone": max(1, int(sequences_per_backbone)),
        "random_seed": max(0, int(random_seed)),
        "engines": selected_engines,
        "engine_configs": normalized_engine_configs,
        "workflow_recipe": normalized_workflow_recipe,
        "survivors_per_engine": int(survivors_per_engine),
        "passing_only": bool(passing_only),
        "keep_best_failed": bool(keep_best_failed),
        "continue_after_failure": bool(continue_after_failure),
        "common_validation": dict(
            common_validation
            or {
                "enabled": False,
                "profile": "bindcraft_default_target_template",
                "num_recycles": 3,
                "min_ipsae": 0.0,
                "pyrosetta_nprocs": 4,
            }
        ),
        "sequence_refinement": dict(sequence_refinement or {"mode": "none"}),
        "template_redesign": dict(template_redesign or {"enabled": False}),
        "evaluation": dict(evaluation or {"mode": "none"}),
        "gpu_device": str(gpu_device),
    }
    if "bindcraft2" in selected_engines:
        try:
            cpu_cores = int(normalized_engine_configs.get("bindcraft2", {}).get("cpu_cores", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("BindCraft 2 campaign CPU core request must be a positive integer.") from exc
        if cpu_cores < 1:
            raise ValueError("BindCraft 2 campaign CPU core request must be a positive integer.")
        params["cpu_cores"] = cpu_cores
    job = create_job(
        DESIGN_CAMPAIGN_GROUP,
        "multi_engine_design_campaign",
        "design_campaign",
        {"target_pdb": str(target_pdb), "target_chains": target_chains},
        params,
    )
    write_json(
        job.run_dir / "worker_request.json",
        {"kind": "design_campaign", "kwargs": {}, "path_kwargs": []},
    )
    write_json(
        job.run_dir / "command.json",
        {
            "mode": "local_worker",
            "command": [
                "python",
                "-m",
                "mn_protein_design.core.local_worker",
                "--run-dir",
                str(job.run_dir),
            ],
        },
    )
    if campaign_name.strip():
        update_status(job.run_dir, "queued", campaign_name=campaign_name.strip())
    spawn_worker_for_run(job.run_dir)
    return job.run_dir


def create_design_campaign_collection(
    *,
    name: str,
    source_run_ids: list[str],
    target_key: str,
) -> Path:
    clean_name = str(name or "Design campaign collection").strip() or "Design campaign collection"
    selected_run_ids = [str(run_id).strip() for run_id in source_run_ids if str(run_id).strip()]
    job = create_job(
        DESIGN_CAMPAIGN_GROUP,
        "design_campaign_collection",
        "design_campaign_collection_merge",
        {"source_runs": selected_run_ids},
        {"collection_name": clean_name, "target_key": str(target_key or "")},
    )
    update_status(job.run_dir, "running", campaign_name=clean_name)

    collection_dir = job.run_dir / "artifacts" / "design_campaign_collection"
    collection_dir.mkdir(parents=True, exist_ok=True)
    source_rows: list[dict[str, Any]] = []
    workflows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    metrics_total = {
        "candidate_count": 0,
        "harmonized_candidate_count": 0,
        "native_candidate_count": 0,
        "common_bindcraft_passing_count": 0,
        "completed_engine_count": 0,
        "failed_engine_count": 0,
    }
    seen_candidate_ids: set[str] = set()

    for order, run_id in enumerate(selected_run_ids, start=1):
        source_run_dir = runs_root() / DESIGN_CAMPAIGN_GROUP / run_id
        input_payload = read_json(source_run_dir / "input.json")
        metadata = read_json(source_run_dir / "metadata.json")
        result = read_json(source_run_dir / "result.json")
        params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
        source_name = str(params.get("campaign_name") or metadata.get("campaign_name") or run_id)
        source_label = f"{metadata.get('job_code') or run_id[:5]} | {source_name}"
        source_candidates = read_candidates(source_run_dir)
        source_summary = read_json(source_run_dir / "artifacts" / "design_campaign" / "workflow_summary.json")
        source_metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        for key in metrics_total:
            try:
                metrics_total[key] += int(source_metrics.get(key) or 0)
            except (TypeError, ValueError):
                pass
        for candidate in source_candidates:
            merged = dict(candidate)
            candidate_id = str(merged.get("candidate_id") or "")
            if candidate_id in seen_candidate_ids:
                candidate_id = f"{candidate_id}__{metadata.get('job_code') or run_id[:5]}"
                merged["candidate_id"] = candidate_id
            seen_candidate_ids.add(candidate_id)
            raw_metadata = (
                dict(merged.get("raw_metadata"))
                if isinstance(merged.get("raw_metadata"), dict)
                else {}
            )
            candidate_metrics = (
                dict(merged.get("metrics"))
                if isinstance(merged.get("metrics"), dict)
                else {}
            )
            raw_metadata.update(
                {
                    "design_campaign_collection": clean_name,
                    "design_campaign_source_run_id": run_id,
                    "design_campaign_source_job_code": str(metadata.get("job_code") or ""),
                    "design_campaign_source_campaign": source_label,
                    "design_campaign_collection_order": order,
                }
            )
            candidate_metrics["source_campaign"] = source_label
            merged["raw_metadata"] = raw_metadata
            merged["metrics"] = candidate_metrics
            candidates.append(merged)
        for workflow in source_summary.get("workflows") if isinstance(source_summary.get("workflows"), list) else []:
            workflow_row = dict(workflow)
            workflow_row["campaign"] = source_label
            workflows.append(workflow_row)
        source_rows.append(
            {
                "order": order,
                "run_id": run_id,
                "job_code": metadata.get("job_code"),
                "campaign": source_name,
                "status": metadata.get("status"),
                "candidates": len(source_candidates),
                "completed_engines": source_metrics.get("completed_engine_count"),
                "failed_engines": source_metrics.get("failed_engine_count"),
            }
        )

    if not candidates:
        finish_job(
            job.run_dir,
            False,
            {"metrics": {"error": "No source candidates were available."}, "outputs": {}},
        )
        raise ValueError("No source candidates were available.")

    normalized = write_candidates(job.run_dir, "design_campaign_collection", candidates)
    sources_path = collection_dir / "design_campaign_collection_sources.csv"
    with sources_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "order",
                "run_id",
                "job_code",
                "campaign",
                "status",
                "candidates",
                "completed_engines",
                "failed_engines",
            ],
        )
        writer.writeheader()
        writer.writerows(source_rows)
    collection_path = collection_dir / "collection.json"
    write_json(
        collection_path,
        {
            "name": clean_name,
            "target_key": target_key,
            "source_run_ids": selected_run_ids,
            "source_count": len(source_rows),
        },
    )
    summary_path = job.run_dir / "artifacts" / "design_campaign" / "workflow_summary.json"
    write_json(
        summary_path,
        {
            "campaign_name": clean_name,
            "target_key": target_key,
            "workflows": workflows,
            "candidate_count": len(normalized),
            "harmonized_candidate_count": len(normalized),
            "common_validation": {
                "profile": "design campaign collection",
                "native_candidate_count": metrics_total["native_candidate_count"],
                "common_bindcraft_passing_count": metrics_total["common_bindcraft_passing_count"],
            },
            "evaluation": {"status": "mixed"},
            "collection_sources": source_rows,
        },
    )
    finish_job(
        job.run_dir,
        True,
        {
            "outputs": {
                "candidates": normalized,
                "workflow_summary": str(summary_path.relative_to(job.run_dir)),
                "collection": str(collection_path.relative_to(job.run_dir)),
                "design_campaign_collection_sources": str(sources_path.relative_to(job.run_dir)),
            },
            "metrics": {
                **metrics_total,
                "collection_name": clean_name,
                "source_run_count": len(source_rows),
                "candidate_count": len(normalized),
                "harmonized_candidate_count": len(normalized),
                "engine_count": len(
                    {
                        str(candidate.get("raw_metadata", {}).get("design_campaign_engine") or candidate.get("source_tool") or "")
                        for candidate in normalized
                    }
                    - {""}
                ),
            },
            "downstream_artifacts": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
                "collection": str(collection_path.relative_to(job.run_dir)),
            },
        },
    )
    return job.run_dir
