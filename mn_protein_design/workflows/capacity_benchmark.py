from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from mn_protein_design.core.candidates import STAGE_COMPLEX_REFOLDING, read_candidates, write_candidates
from mn_protein_design.core.jobs import (
    collect_jobs,
    create_job,
    finish_job,
    prepare_job_for_resume,
    read_json,
    update_status,
    write_json,
)
from mn_protein_design.core.local_worker import spawn_worker_for_run
from mn_protein_design.core.structures import filter_pdb_text, pdb_summary
from mn_protein_design.workflows.benchmark import (
    ALPHAFAST_DB_DIR,
    ALPHAFAST_WEIGHTS_DIR,
    COLABFOLD_CACHE_DIR,
    MSA_REPOSITORY_DIR,
    _child_viewer_prediction_path,
    _safe_id,
    enqueue_candidate_refolding_evaluation,
)
from mn_protein_design.workflows.esm_binder import ESMFOLD2_MODEL_DIR
from mn_protein_design.workflows.refolding import BOLTZ_MODELS_DIR, RF3_CHECKPOINT, _sequences_by_chain


CAPACITY_GROUP = "benchmark"
CAPACITY_JOB_TYPE = "refolding_capacity_benchmark"
CAPACITY_TOOL = "refolding_capacity_matrix"
CAPACITY_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "stopped", "skipped"}
CAPACITY_ACTIVE_STATUSES = {"queued", "running", "preparing"}

AA3 = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}

ENGINE_LABELS = {
    "alphafast_af3": "AlphaFast AF3",
    "colabfold": "ColabFold",
    "af2_initial_guess": "AF2 target-only initial guess",
    "esmfold2": "ESMFold2",
    "boltz2_initial_guess": "Boltz-2",
    "rf3": "RF3",
    "protenix": "Protenix",
    "boltzgen_fold": "BoltzGen Fold",
}


def _sequence(length: int, offset: int = 0) -> str:
    alphabet = "AELKQRAELKQRVSTNPG"
    return "".join(alphabet[(index + offset) % len(alphabet)] for index in range(max(1, int(length))))


def _atom_line(serial: int, atom: str, residue: str, chain: str, resseq: int, x: float, y: float, z: float) -> str:
    return (
        f"ATOM  {serial:5d} {atom:<4s} {residue:>3s} {chain:1s}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 50.00           {atom[0]:>2s}"
    )


def _chain_lines(sequence: str, chain: str, start_serial: int, x_offset: float, y_offset: float) -> tuple[list[str], int]:
    lines: list[str] = []
    serial = start_serial
    for index, aa in enumerate(sequence, start=1):
        residue = AA3.get(aa, "ALA")
        theta = index * 1.745
        x = x_offset + index * 1.5
        y = y_offset + math.cos(theta) * 2.0
        z = math.sin(theta) * 2.0
        for atom, dx, dy, dz in [
            ("N", -0.55, 0.10, 0.00),
            ("CA", 0.00, 0.00, 0.00),
            ("C", 0.55, -0.10, 0.00),
            ("O", 0.85, -0.35, 0.10),
        ]:
            lines.append(_atom_line(serial, atom, residue, chain, index, x + dx, y + dy, z + dz))
            serial += 1
    lines.append("TER")
    return lines, serial


def _write_synthetic_complex(path: Path, target_length: int, binder_length: int) -> tuple[str, str]:
    binder_sequence = _sequence(binder_length, 0)
    target_sequence = _sequence(target_length, 7)
    binder_lines, serial = _chain_lines(binder_sequence, "A", 1, 0.0, 0.0)
    target_lines, _ = _chain_lines(target_sequence, "B", serial, 0.0, 9.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(binder_lines + target_lines + ["END", ""]))
    return binder_sequence, target_sequence


def _copy_target_artifact(job_run_dir: Path, target_pdb: Path, target_chains: list[str]) -> Path:
    out = job_run_dir / "artifacts" / "capacity_target" / "target.pdb"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = target_pdb.read_text(errors="ignore")
    filtered = filter_pdb_text(text, keep_chains=set(target_chains) or None, remove_waters=True, remove_hetero=True)
    out.write_text(filtered if "ATOM" in filtered else text)
    return out


def _target_length(target_pdb: Path, target_chains: list[str]) -> int:
    summary = pdb_summary(target_pdb.read_text(errors="ignore"))
    total = 0
    selected = set(target_chains)
    for row in summary.get("chains") or []:
        if selected and row.get("chain_id") not in selected:
            continue
        total += int(row.get("residue_count") or len(row.get("residues") or []) or 0)
    return total


def _write_real_target_complex(
    path: Path,
    *,
    target_pdb: Path,
    target_chains: list[str],
    binder_length: int,
) -> str:
    binder_sequence = _sequence(binder_length, 0)
    binder_lines, _serial = _chain_lines(binder_sequence, "A", 1, 0.0, 0.0)
    target_text = filter_pdb_text(
        target_pdb.read_text(errors="ignore"),
        keep_chains=set(target_chains) or None,
        remove_waters=True,
        remove_hetero=True,
    )
    target_lines = [
        line
        for line in target_text.splitlines()
        if line.startswith(("ATOM", "TER"))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(binder_lines + target_lines + ["END", ""]))
    return binder_sequence


def _chain_ids_for_copies(copy_count: int) -> list[str]:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if copy_count > len(alphabet):
        raise ValueError(f"Copy-count ladder currently supports up to {len(alphabet)} chains.")
    return list(alphabet[:copy_count])


def _write_sequence_copy_multimer(path: Path, sequence: str, copy_count: int) -> tuple[list[str], list[str]]:
    chain_ids = _chain_ids_for_copies(copy_count)
    lines: list[str] = []
    serial = 1
    radius = max(10.0, len(sequence) * 0.35)
    for index, chain in enumerate(chain_ids):
        angle = (2.0 * math.pi * index) / max(1, copy_count)
        x_offset = math.cos(angle) * radius
        y_offset = math.sin(angle) * radius
        chain_lines, serial = _chain_lines(sequence, chain, serial, x_offset, y_offset)
        lines.extend(chain_lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines + ["END", ""]))
    return [chain_ids[0]], chain_ids[1:]


def _target_panel_system_records(
    *,
    entry: dict[str, Any],
    systems_dir: Path,
    run_dir: Path,
    index: int = 1,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_path = Path(str(entry.get("target_pdb") or "")).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Target-panel PDB does not exist: {source_path}")
    source_chain = str(entry.get("chain") or "").strip()
    if not source_chain:
        raise ValueError("Each target-panel row needs a source chain.")
    source_sequences = _sequences_by_chain(source_path)
    sequence = str(source_sequences.get(source_chain) or "").replace("X", "").strip()
    if not sequence:
        raise ValueError(f"Could not extract chain {source_chain} from {source_path}.")
    copy_count = int(entry.get("copy_count") or 1)
    if copy_count < 1:
        copy_count = 1
    sequence_length = len(sequence)
    target_label = str(entry.get("target_name") or source_path.stem)
    safe_target = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in target_label).strip("_") or f"target_{index}"
    candidate_id = f"cap_{safe_target}_{source_chain}{sequence_length}_x{copy_count}"
    complex_path = systems_dir / f"{candidate_id}.pdb"
    if copy_count == 1:
        complex_text = filter_pdb_text(
            source_path.read_text(errors="ignore"),
            keep_chains={source_chain},
            remove_waters=True,
            remove_hetero=True,
        )
        complex_path.parent.mkdir(parents=True, exist_ok=True)
        complex_path.write_text(complex_text.rstrip() + "\nEND\n")
        binder_chains = [source_chain]
        candidate_target_chains: list[str] = []
    else:
        binder_chains, candidate_target_chains = _write_sequence_copy_multimer(complex_path, sequence, copy_count)
    total_length = sequence_length * copy_count
    target_id = f"{target_label}_{source_chain}_x{copy_count}"
    row = {
        "candidate_id": candidate_id,
        "target_name": target_label,
        "matrix_mode": "target_panel",
        "source_chain": source_chain,
        "copy_count": copy_count,
        "sequence_length": sequence_length,
        "target_length": sequence_length * max(0, copy_count - 1),
        "binder_length": sequence_length,
        "total_length": total_length,
        "complex_pdb": str(complex_path.relative_to(run_dir)),
        "target_pdb": str(source_path if copy_count == 1 else complex_path),
        "target_chains": json.dumps(candidate_target_chains),
        "binder_chains": json.dumps(binder_chains),
    }
    candidate = {
        "candidate_id": candidate_id,
        "stage": STAGE_COMPLEX_REFOLDING,
        "source_tool": CAPACITY_TOOL,
        "target_pdb": str(source_path if copy_count == 1 else complex_path),
        "complex_pdb": str(complex_path.relative_to(run_dir)),
        "binder_sequence": sequence,
        "binder_chains": binder_chains,
        "target_chains": candidate_target_chains,
        "binder_length": str(sequence_length),
        "metrics": {
            "target_id": target_id,
            "matrix_mode": "target_panel",
            "source_chain": source_chain,
            "copy_count": copy_count,
            "sequence_length": sequence_length,
            "target_length": sequence_length * max(0, copy_count - 1),
            "binder_length": sequence_length,
            "total_length": total_length,
        },
        "raw_metadata": {
            "capacity_benchmark": True,
            "matrix_mode": "target_panel",
            "capacity_target_only": copy_count == 1,
            "target_source_pdb": str(source_path),
            "source_chain": source_chain,
        },
    }
    return row, candidate


def _engine_kwargs(engine: str, gpu_device: str, preset: str, params: dict[str, Any]) -> dict[str, Any]:
    capacity_only = preset in {"capacity_only", "minimal", "smoke"}
    practical = preset in {"practical", "standard"}
    full_workflow = preset == "full_workflow"
    if capacity_only:
        alphafast_recycles = 1
        af2_recycles = 1
        colabfold_recycles = 1
        colabfold_models = 1
        boltz_recycles = 1
        boltz_steps = 20
        boltz_samples = 1
        esm_loops = 1
        esm_steps = 16
        rf3_recycles = 2
        rf3_steps = 10
        rf3_samples = 1
        protenix_cycle = 1
        protenix_steps = 10
        protenix_samples = 1
        boltzgen_recycles = 1
        boltzgen_steps = 20
        boltzgen_samples = 1
    elif practical:
        alphafast_recycles = 10
        af2_recycles = 3
        colabfold_recycles = 3
        colabfold_models = 3
        boltz_recycles = 10
        boltz_steps = 200
        boltz_samples = 3
        esm_loops = 10
        esm_steps = 68
        rf3_recycles = 10
        rf3_steps = 50
        rf3_samples = 5
        protenix_cycle = 3
        protenix_steps = 50
        protenix_samples = 5
        boltzgen_recycles = 3
        boltzgen_steps = 200
        boltzgen_samples = 5
    else:
        alphafast_recycles = 10
        af2_recycles = 3
        colabfold_recycles = 3
        colabfold_models = 3
        boltz_recycles = 10
        boltz_steps = 200
        boltz_samples = 3
        esm_loops = 10
        esm_steps = 68
        rf3_recycles = 10
        rf3_steps = 50
        rf3_samples = 5
        protenix_cycle = 3
        protenix_steps = 50
        protenix_samples = 5
        boltzgen_recycles = 3
        boltzgen_steps = 200
        boltzgen_samples = 5
    use_real_target_msa = practical or full_workflow
    base = {
        "models": [],
        "run_common_interface_metrics": bool(full_workflow),
        "run_pyrosetta_input_metrics": bool(full_workflow),
        "run_predicted_rosetta_metrics": bool(full_workflow),
        "run_pymol_metrics": bool(full_workflow),
        "pyrosetta_nprocs": int(params.get("pyrosetta_nprocs", 8 if full_workflow else 1)),
        "gpu_device": gpu_device,
        "alphafast_gpu_device": gpu_device,
        "colabfold_gpu_device": gpu_device,
        "colabfold_msa_source": "msa_repository_then_alphafast_mmseqs_gpu" if use_real_target_msa else "repo_run_csv",
        "msa_repository_dir": MSA_REPOSITORY_DIR,
        "require_real_target_msa": bool(use_real_target_msa),
        "alphafast_db_dir": ALPHAFAST_DB_DIR,
        "alphafast_weights_dir": ALPHAFAST_WEIGHTS_DIR,
        "colabfold_cache_dir": COLABFOLD_CACHE_DIR,
        "alphafast_batch_size": 0,
        "alphafast_num_recycles": int(params.get("alphafast_num_recycles", alphafast_recycles)),
        "alphafast_query_only_msa": not bool(use_real_target_msa),
        "af2_num_recycles": int(params.get("af2_num_recycles", af2_recycles)),
        "af2_multimer": True,
        "af2_use_initial_guess": False,
        "af2_use_binder_template": False,
        "af2_use_interface_template": False,
        "colabfold_num_recycles": int(params.get("colabfold_num_recycles", colabfold_recycles)),
        "colabfold_num_models": int(params.get("colabfold_num_models", colabfold_models)),
        "colabfold_use_target_templates": False,
        "colabfold_max_template_hits": 4,
        "boltz2_use_target_template": True,
        "boltz2_use_target_msa": bool(use_real_target_msa),
        "boltz2_recycling_steps": int(params.get("boltz2_recycling_steps", boltz_recycles)),
        "boltz2_sampling_steps": int(params.get("boltz2_sampling_steps", boltz_steps)),
        "boltz2_diffusion_samples": int(params.get("boltz2_diffusion_samples", boltz_samples)),
        "boltz2_write_full_pae": bool(full_workflow),
        "esmfold2_modes": list(params.get("esmfold2_modes") or ["initial_guess"]),
        "esmfold2_use_target_msa": bool(use_real_target_msa),
        "num_sampling_steps": int(params.get("esmfold2_sampling_steps", esm_steps)),
        "num_loops": int(params.get("esmfold2_loops", esm_loops)),
        "seed": int(params.get("seed", 0)),
        "rf3_checkpoint_path": RF3_CHECKPOINT,
        "rf3_use_target_msa": bool(use_real_target_msa),
        "rf3_recycles": int(params.get("rf3_recycles", rf3_recycles)),
        "rf3_num_steps": int(params.get("rf3_num_steps", rf3_steps)),
        "rf3_diffusion_batch_size": int(params.get("rf3_samples", rf3_samples)),
        "rf3_seed": int(params.get("seed", 0)),
        "protenix_use_msa": bool(use_real_target_msa),
        "protenix_cycle": int(params.get("protenix_cycle", protenix_cycle)),
        "protenix_diffusion_steps": int(params.get("protenix_steps", protenix_steps)),
        "protenix_samples": int(params.get("protenix_samples", protenix_samples)),
        "boltzgen_recycling_steps": int(params.get("boltzgen_recycling_steps", boltzgen_recycles)),
        "boltzgen_sampling_steps": int(params.get("boltzgen_sampling_steps", boltzgen_steps)),
        "boltzgen_diffusion_samples": int(params.get("boltzgen_samples", boltzgen_samples)),
    }
    for key in ENGINE_LABELS:
        base[f"run_{key}"] = False
    if engine == "alphafast_af3":
        base["run_alphafast_af3"] = True
        base["models"] = ["af3"]
    elif engine == "colabfold":
        base["run_colabfold"] = True
        base["models"] = ["colabfold"]
    elif engine == "af2_initial_guess":
        base["run_af2_initial_guess"] = True
    elif engine == "esmfold2":
        base["run_esmfold2"] = True
    elif engine == "boltz2_initial_guess":
        base["run_boltz2_initial_guess"] = True
        base["models"] = ["boltz"]
    elif engine == "rf3":
        base["run_rf3"] = True
    elif engine == "protenix":
        base["run_protenix"] = True
    elif engine == "boltzgen_fold":
        base["run_boltzgen_fold"] = True
    else:
        raise ValueError(f"Unknown capacity engine: {engine}")
    return base


def _write_csv_rows(path: Path, rows: list[dict[str, Any]], fallback_field: str) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or [fallback_field])
        writer.writeheader()
        writer.writerows(rows)


def _enqueue_capacity_child(
    *,
    parent_run_dir: Path,
    parent_run_id: str,
    benchmark_name: str,
    engine: str,
    row: dict[str, Any],
    gpu_device: str,
    preset: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    child = enqueue_candidate_refolding_evaluation(
        source_run_dir=parent_run_dir,
        candidates_jsonl=parent_run_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl",
        selected_candidate_ids=[str(row["candidate_id"])],
        max_candidates=0,
        evaluation_name=f"{benchmark_name} | {ENGINE_LABELS.get(engine, engine)} | {row['candidate_id']}",
        **_engine_kwargs(engine, gpu_device, preset, params),
    )
    child_metadata = read_json(child / "metadata.json")
    child_metadata["capacity_parent_run_id"] = parent_run_id
    child_metadata["capacity_parent_run_dir"] = str(parent_run_dir)
    child_metadata["capacity_engine"] = engine
    child_metadata["capacity_engine_label"] = ENGINE_LABELS.get(engine, engine)
    child_metadata["capacity_candidate_id"] = row["candidate_id"]
    child_metadata["capacity_matrix_mode"] = row.get("matrix_mode", params.get("matrix_mode", ""))
    child_metadata["capacity_copy_count"] = row.get("copy_count", "")
    child_metadata["capacity_sequence_length"] = row.get("sequence_length", "")
    child_metadata["capacity_target_length"] = row.get("target_length", "")
    child_metadata["capacity_binder_length"] = row.get("binder_length", "")
    child_metadata["capacity_total_length"] = row.get("total_length", "")
    child_metadata["capacity_gpu_device"] = gpu_device
    child_metadata["campaign_name"] = benchmark_name
    child_metadata["hidden"] = True
    child_metadata["parent_task_group"] = CAPACITY_GROUP
    child_metadata["parent_run_id"] = parent_run_id
    child_metadata["parent_run_dir"] = str(parent_run_dir)
    child_metadata["parent_role"] = "capacity_cell"
    write_json(child / "metadata.json", child_metadata)
    return {
        "child_run_id": child.name,
        "child_run_dir": str(child),
        "engine": engine,
        "engine_label": ENGINE_LABELS.get(engine, engine),
        **row,
    }


def _capacity_scheduler_files(run_dir: Path) -> None:
    write_json(
        run_dir / "worker_request.json",
        {
            "kind": "capacity_benchmark_scheduler",
            "kwargs": {"poll_seconds": 2},
            "path_kwargs": [],
        },
    )
    write_json(
        run_dir / "command.json",
        {
            "mode": "local_worker",
            "command": ["python", "-m", "mn_protein_design.core.local_worker", "--run-dir", str(run_dir)],
        },
    )


def _worker_is_alive(metadata: dict[str, Any], run_dir: Path) -> bool:
    try:
        pid = int(metadata.get("worker_pid") or 0)
        if pid <= 0:
            return False
        os.kill(pid, 0)
        cmdline = (
            (Path("/proc") / str(pid) / "cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode(errors="replace")
        )
        return (
            "mn_protein_design.core.local_worker" in cmdline
            and str(Path(run_dir).expanduser().resolve()) in cmdline
        )
    except (OSError, TypeError, ValueError):
        return False


def create_refolding_capacity_benchmark(
    *,
    benchmark_name: str,
    target_lengths: list[int] | None = None,
    binder_lengths: list[int] | None = None,
    copy_counts: list[int] | None = None,
    engines: list[str],
    gpu_device: str = "0",
    preset: str = "capacity_only",
    launch: bool = True,
    engine_params: dict[str, Any] | None = None,
    target_pdb: Path | None = None,
    target_chains: list[str] | None = None,
    target_name: str = "",
    matrix_mode: str = "sequence_copy_multimer",
    engine_copy_counts: dict[str, list[int]] | None = None,
    source_capacity_run_id: str = "",
    target_panel_entries: list[dict[str, Any]] | None = None,
) -> Path:
    target_pdb = Path(target_pdb).expanduser().resolve() if target_pdb else None
    target_chains = [str(chain) for chain in (target_chains or []) if str(chain)]
    target_panel_entries = list(target_panel_entries or [])
    if matrix_mode in {"sequence_copy_multimer", "fixed_target_binder_ladder"} and (target_pdb is None or not target_pdb.exists()):
        raise FileNotFoundError("A readable target PDB is required for the selected capacity benchmark.")
    if matrix_mode == "target_panel" and not target_panel_entries:
        raise ValueError("Select at least one prepared target-chain row for target-panel capacity testing.")
    binder_lengths = [int(value) for value in (binder_lengths or [])]
    copy_counts = [int(value) for value in (copy_counts or [])]
    params = {
        "benchmark_name": benchmark_name,
        "matrix_mode": matrix_mode,
        "target_lengths": [int(value) for value in (target_lengths or [])],
        "binder_lengths": binder_lengths,
        "copy_counts": copy_counts,
        "engines": list(engines),
        "capacity_device": str(gpu_device),
        "preset": preset,
        "launch": bool(launch),
        "model_dir": str(ESMFOLD2_MODEL_DIR),
        "target_pdb": str(target_pdb) if target_pdb else "",
        "target_chains": list(target_chains),
        "target_name": target_name,
        "target_panel_entries": target_panel_entries,
        "engine_copy_counts": {
            str(engine): [int(value) for value in values]
            for engine, values in (engine_copy_counts or {}).items()
        },
        "source_capacity_run_id": str(source_capacity_run_id or ""),
        **dict(engine_params or {}),
    }
    job = create_job(
        CAPACITY_GROUP,
        CAPACITY_JOB_TYPE,
        CAPACITY_TOOL,
        inputs={"synthetic": True},
        params={**params, "queue_resource": ""},
    )
    update_status(job.run_dir, "running", campaign_name=benchmark_name, current_phase="Creating capacity matrix")
    systems_dir = job.run_dir / "artifacts" / "capacity_systems"
    rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    if matrix_mode == "sequence_copy_multimer":
        assert target_pdb is not None
        if not target_chains:
            raise ValueError("Select one source chain for the sequence-copy multimer folding benchmark.")
        source_chain = target_chains[0]
        source_sequences = _sequences_by_chain(target_pdb)
        source_sequence = str(source_sequences.get(source_chain) or "").replace("X", "").strip()
        if not source_sequence:
            raise ValueError(f"Could not extract a protein sequence from chain {source_chain} in {target_pdb}.")
        if not copy_counts:
            raise ValueError("Provide at least one multimer copy count.")
        target_artifact = None
        target_length_values = []
    elif matrix_mode == "target_panel":
        target_artifact = None
        target_length_values = []
    elif matrix_mode == "fixed_target_binder_ladder":
        assert target_pdb is not None
        target_artifact = _copy_target_artifact(job.run_dir, target_pdb, target_chains)
        fixed_target_length = _target_length(target_artifact, target_chains)
        shutil.copy2(target_artifact, job.run_dir / "artifacts" / "capacity_target.pdb")
        target_length_values = [fixed_target_length]
    elif matrix_mode != "target_panel":
        target_artifact = None
        target_length_values = params["target_lengths"]
    if matrix_mode == "sequence_copy_multimer":
        assert target_pdb is not None
        source_chain = target_chains[0]
        sequence = source_sequence
        for copy_count in copy_counts:
            if copy_count < 1:
                continue
            sequence_length = len(sequence)
            target_length = sequence_length * (copy_count - 1)
            binder_length = sequence_length
            total_length = sequence_length * copy_count
            candidate_id = f"cap_{source_chain}{sequence_length}_x{copy_count}"
            complex_path = systems_dir / f"{candidate_id}.pdb"
            if copy_count == 1:
                complex_text = filter_pdb_text(
                    target_artifact.read_text(errors="ignore"),
                    keep_chains={source_chain},
                    remove_waters=True,
                    remove_hetero=True,
                )
                complex_path.parent.mkdir(parents=True, exist_ok=True)
                complex_path.write_text(complex_text.rstrip() + "\nEND\n")
                binder_chains = [source_chain]
                candidate_target_chains: list[str] = []
            else:
                binder_chains, candidate_target_chains = _write_sequence_copy_multimer(complex_path, sequence, copy_count)
            candidate_target_pdb = str(target_artifact if copy_count == 1 else complex_path)
            target_id = f"{target_name or target_pdb.stem}_{source_chain}_x{copy_count}"
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "target_name": target_name or target_id,
                    "matrix_mode": matrix_mode,
                    "source_chain": source_chain,
                    "copy_count": copy_count,
                    "sequence_length": sequence_length,
                    "target_length": target_length,
                    "binder_length": binder_length,
                    "total_length": total_length,
                    "complex_pdb": str(complex_path.relative_to(job.run_dir)),
                    "target_pdb": candidate_target_pdb,
                    "target_chains": json.dumps(candidate_target_chains),
                    "binder_chains": json.dumps(binder_chains),
                }
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "stage": STAGE_COMPLEX_REFOLDING,
                    "source_tool": CAPACITY_TOOL,
                    "target_pdb": candidate_target_pdb,
                    "complex_pdb": str(complex_path.relative_to(job.run_dir)),
                    "binder_sequence": sequence,
                    "binder_chains": binder_chains,
                    "target_chains": candidate_target_chains,
                    "binder_length": str(binder_length),
                    "metrics": {
                        "target_id": target_id,
                        "matrix_mode": matrix_mode,
                        "source_chain": source_chain,
                        "copy_count": copy_count,
                        "sequence_length": sequence_length,
                        "target_length": target_length,
                        "binder_length": binder_length,
                        "total_length": total_length,
                    },
                    "raw_metadata": {
                        "capacity_benchmark": True,
                        "matrix_mode": matrix_mode,
                        "capacity_target_only": copy_count == 1,
                        "target_source_pdb": str(target_pdb),
                        "source_chain": source_chain,
                    },
                }
            )
    elif matrix_mode == "target_panel":
        for index, entry in enumerate(target_panel_entries, start=1):
            source_path = Path(str(entry.get("target_pdb") or "")).expanduser()
            if not source_path.exists():
                raise FileNotFoundError(f"Target-panel PDB does not exist: {source_path}")
            source_chain = str(entry.get("chain") or "").strip()
            if not source_chain:
                raise ValueError("Each target-panel row needs a source chain.")
            source_sequences = _sequences_by_chain(source_path)
            sequence = str(source_sequences.get(source_chain) or "").replace("X", "").strip()
            if not sequence:
                raise ValueError(f"Could not extract chain {source_chain} from {source_path}.")
            copy_count = int(entry.get("copy_count") or 1)
            if copy_count < 1:
                copy_count = 1
            sequence_length = len(sequence)
            target_label = str(entry.get("target_name") or source_path.stem)
            safe_target = "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in target_label).strip("_") or f"target_{index}"
            candidate_id = f"cap_{safe_target}_{source_chain}{sequence_length}_x{copy_count}"
            complex_path = systems_dir / f"{candidate_id}.pdb"
            if copy_count == 1:
                complex_text = filter_pdb_text(
                    source_path.read_text(errors="ignore"),
                    keep_chains={source_chain},
                    remove_waters=True,
                    remove_hetero=True,
                )
                complex_path.parent.mkdir(parents=True, exist_ok=True)
                complex_path.write_text(complex_text.rstrip() + "\nEND\n")
                binder_chains = [source_chain]
                candidate_target_chains: list[str] = []
            else:
                binder_chains, candidate_target_chains = _write_sequence_copy_multimer(complex_path, sequence, copy_count)
            total_length = sequence_length * copy_count
            target_id = f"{target_label}_{source_chain}_x{copy_count}"
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "target_name": target_label,
                    "matrix_mode": matrix_mode,
                    "source_chain": source_chain,
                    "copy_count": copy_count,
                    "sequence_length": sequence_length,
                    "target_length": sequence_length * max(0, copy_count - 1),
                    "binder_length": sequence_length,
                    "total_length": total_length,
                    "complex_pdb": str(complex_path.relative_to(job.run_dir)),
                    "target_pdb": str(source_path if copy_count == 1 else complex_path),
                    "target_chains": json.dumps(candidate_target_chains),
                    "binder_chains": json.dumps(binder_chains),
                }
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "stage": STAGE_COMPLEX_REFOLDING,
                    "source_tool": CAPACITY_TOOL,
                    "target_pdb": str(source_path if copy_count == 1 else complex_path),
                    "complex_pdb": str(complex_path.relative_to(job.run_dir)),
                    "binder_sequence": sequence,
                    "binder_chains": binder_chains,
                    "target_chains": candidate_target_chains,
                    "binder_length": str(sequence_length),
                    "metrics": {
                        "target_id": target_id,
                        "matrix_mode": matrix_mode,
                        "source_chain": source_chain,
                        "copy_count": copy_count,
                        "sequence_length": sequence_length,
                        "target_length": sequence_length * max(0, copy_count - 1),
                        "binder_length": sequence_length,
                        "total_length": total_length,
                    },
                    "raw_metadata": {
                        "capacity_benchmark": True,
                        "matrix_mode": matrix_mode,
                        "capacity_target_only": copy_count == 1,
                        "target_source_pdb": str(source_path),
                        "source_chain": source_chain,
                    },
                }
            )
    else:
        for target_length in target_length_values:
            for binder_length in params["binder_lengths"]:
                candidate_id = f"cap_t{target_length}_b{binder_length}"
                complex_path = systems_dir / f"{candidate_id}.pdb"
                if matrix_mode == "fixed_target_binder_ladder":
                    assert target_pdb is not None and target_artifact is not None
                    binder_sequence = _write_real_target_complex(
                        complex_path,
                        target_pdb=target_artifact,
                        target_chains=target_chains,
                        binder_length=binder_length,
                    )
                    candidate_target_pdb = str(target_artifact)
                    candidate_target_chains = list(target_chains)
                    target_id = target_name or target_artifact.stem
                else:
                    binder_sequence, _target_sequence = _write_synthetic_complex(complex_path, target_length, binder_length)
                    candidate_target_pdb = str(complex_path)
                    candidate_target_chains = ["B"]
                    target_id = f"synthetic_target_{target_length}"
                total_length = target_length + binder_length
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "target_name": target_name or target_id,
                        "matrix_mode": matrix_mode,
                        "source_chain": "",
                        "copy_count": "",
                        "sequence_length": "",
                        "target_length": target_length,
                        "binder_length": binder_length,
                        "total_length": total_length,
                        "complex_pdb": str(complex_path.relative_to(job.run_dir)),
                        "target_pdb": candidate_target_pdb,
                        "target_chains": json.dumps(candidate_target_chains),
                        "binder_chains": json.dumps(["A"]),
                    }
                )
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "stage": STAGE_COMPLEX_REFOLDING,
                        "source_tool": CAPACITY_TOOL,
                        "target_pdb": candidate_target_pdb,
                        "complex_pdb": str(complex_path.relative_to(job.run_dir)),
                        "binder_sequence": binder_sequence,
                        "binder_chains": ["A"],
                        "target_chains": candidate_target_chains,
                        "binder_length": str(binder_length),
                        "metrics": {
                            "target_id": target_id,
                            "matrix_mode": matrix_mode,
                            "target_length": target_length,
                            "binder_length": binder_length,
                            "total_length": total_length,
                        },
                        "raw_metadata": {
                            "capacity_benchmark": True,
                            "matrix_mode": matrix_mode,
                            "target_source_pdb": str(target_pdb) if target_pdb else "",
                        },
                    }
                )
    write_candidates(job.run_dir, CAPACITY_TOOL, candidates)
    matrix_path = job.run_dir / "artifacts" / "capacity_matrix.csv"
    _write_csv_rows(matrix_path, rows, "candidate_id")

    children: list[dict[str, Any]] = []
    for engine in engines:
        for row in rows:
            allowed_copy_counts = params["engine_copy_counts"].get(engine)
            if allowed_copy_counts and int(row.get("copy_count") or 0) not in allowed_copy_counts:
                continue
            children.append(
                _enqueue_capacity_child(
                    parent_run_dir=job.run_dir,
                    parent_run_id=job.run_id,
                    benchmark_name=benchmark_name,
                    engine=engine,
                    row=row,
                    gpu_device=str(gpu_device),
                    preset=preset,
                    params=params,
                )
            )

    children_path = job.run_dir / "artifacts" / "capacity_children.csv"
    _write_csv_rows(children_path, children, "child_run_id")
    scheduler_result = {
        "outputs": {
            "capacity_matrix": str(matrix_path.relative_to(job.run_dir)),
            "capacity_children": str(children_path.relative_to(job.run_dir)),
            "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
        },
        "metrics": {
            "cell_count": len(children),
            "system_count": len(rows),
            "engine_count": len(engines),
            "launched": bool(launch),
            "globally_sequential": True,
            "stop_engine_after_first_failure": True,
        },
    }
    write_json(job.run_dir / "result.json", {"success": None, **scheduler_result})
    _capacity_scheduler_files(job.run_dir)
    if launch:
        update_status(job.run_dir, "running", current_phase="Scheduling capacity cells shortest-to-longest")
        spawn_worker_for_run(job.run_dir)
    else:
        finish_job(job.run_dir, True, scheduler_result)
    return job.run_dir


def create_practical_capacity_benchmark_from_parent(
    parent_run_dir: Path,
    *,
    benchmark_name: str | None = None,
    gpu_device: str | None = None,
    launch: bool = True,
) -> Path:
    """Start one practical cell per engine at its largest successful capacity-only system."""
    parent_run_dir = Path(parent_run_dir).expanduser().resolve()
    payload = read_json(parent_run_dir / "input.json")
    if payload.get("job_type") != CAPACITY_JOB_TYPE:
        raise ValueError(f"Not a capacity benchmark run: {parent_run_dir}")
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    matrix_mode = str(params.get("matrix_mode") or "")
    if matrix_mode != "sequence_copy_multimer":
        raise ValueError("Practical seeding currently requires a same-sequence multimer capacity run.")
    successful_by_engine: dict[str, dict[str, Any]] = {}
    failures_by_engine: dict[str, list[dict[str, Any]]] = {}
    for child in capacity_child_rows(parent_run_dir):
        engine = str(child.get("engine_key") or "")
        copy_count = _capacity_int(child.get("copy_count"))
        total_length = _capacity_int(child.get("total_length"))
        if not engine or copy_count is None or total_length is None:
            continue
        status = str(child.get("status") or "")
        if status in {"failed", "skipped"}:
            failures_by_engine.setdefault(engine, []).append(
                {
                    "copy_count": copy_count,
                    "total_length": total_length,
                    "status": status,
                    "run_id": str(child.get("run_id") or ""),
                    "failure_kind": str(child.get("failure_kind") or ""),
                }
            )
            continue
        if status != "completed":
            continue
        current = successful_by_engine.get(engine)
        if current is None or total_length > int(current["total_length"]):
            successful_by_engine[engine] = {
                "copy_count": copy_count,
                "total_length": total_length,
                "source_run_id": str(child.get("run_id") or ""),
            }
    if not successful_by_engine:
        raise ValueError("No completed capacity cells are available to seed a practical benchmark.")
    practical_seed_rows: list[dict[str, Any]] = []
    for engine, row in successful_by_engine.items():
        total_length = int(row["total_length"])
        larger_failures = [
            failure
            for failure in failures_by_engine.get(engine, [])
            if int(failure["total_length"]) > total_length
        ]
        next_failure = min(larger_failures, key=lambda item: int(item["total_length"])) if larger_failures else {}
        practical_seed_rows.append(
            {
                "engine": engine,
                "engine_label": ENGINE_LABELS.get(engine, engine),
                "copy_count": int(row["copy_count"]),
                "total_length": total_length,
                "source_capacity_run_id": row.get("source_run_id", ""),
                "next_failed_total_length": next_failure.get("total_length", ""),
                "next_failed_copy_count": next_failure.get("copy_count", ""),
                "next_failed_status": next_failure.get("status", ""),
                "next_failed_run_id": next_failure.get("run_id", ""),
                "next_failure_kind": next_failure.get("failure_kind", ""),
            }
        )
    engine_copy_counts = {
        engine: [int(row["copy_count"])]
        for engine, row in successful_by_engine.items()
    }
    copy_counts = sorted(
        {copy_count for values in engine_copy_counts.values() for copy_count in values}
    )
    source_name = str(params.get("benchmark_name") or parent_run_dir.name)
    return create_refolding_capacity_benchmark(
        benchmark_name=benchmark_name or f"{source_name} - practical from capacity",
        copy_counts=copy_counts,
        engines=list(successful_by_engine),
        gpu_device=str(gpu_device if gpu_device is not None else params.get("capacity_device") or "0"),
        preset="practical",
        launch=launch,
        target_pdb=Path(str(params.get("target_pdb") or "")),
        target_chains=[str(chain) for chain in (params.get("target_chains") or [])],
        target_name=str(params.get("target_name") or ""),
        matrix_mode=matrix_mode,
        engine_copy_counts=engine_copy_counts,
        source_capacity_run_id=parent_run_dir.name,
        engine_params={
            "practical_seed_policy": "largest_success_per_engine_from_capacity",
            "practical_seed_rows": practical_seed_rows,
            "practical_safe_settings": {
                "explicit_repeated_chains": True,
                "colabfold_use_target_templates": False,
                "alphafast_query_only_msa": True,
                "rf3_min_recycles": 2,
                "viewer_structures_preserve_chains": True,
                "cell_failures_do_not_fail_parent": True,
            },
        },
    )


def extend_refolding_capacity_benchmark(
    parent_run_dir: Path,
    *,
    engines_to_add: list[str] | None = None,
    copy_counts_to_add: list[int] | None = None,
    target_panel_entries_to_add: list[dict[str, Any]] | None = None,
    recalculate_run_ids: list[str] | None = None,
    launch: bool = True,
) -> dict[str, int]:
    """Append engines/systems or replace selected cells inside one capacity parent."""
    parent_run_dir = Path(parent_run_dir).expanduser().resolve()
    payload = read_json(parent_run_dir / "input.json")
    if payload.get("job_type") != CAPACITY_JOB_TYPE:
        raise ValueError(f"Not a capacity benchmark run: {parent_run_dir}")
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    params = dict(params)
    matrix_mode = str(params.get("matrix_mode") or "")
    benchmark_name = str(params.get("benchmark_name") or "Capacity benchmark")
    gpu_device = str(params.get("capacity_device") or "0")
    preset = str(params.get("preset") or "capacity_only")
    parent_run_id = parent_run_dir.name

    matrix_path = parent_run_dir / "artifacts" / "capacity_matrix.csv"
    children_path = parent_run_dir / "artifacts" / "capacity_children.csv"
    with matrix_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    with children_path.open(newline="") as handle:
        children = list(csv.DictReader(handle))
    candidates = read_candidates(parent_run_dir)
    existing_child_results = capacity_child_rows(parent_run_dir)

    added_systems = 0
    added_targets = 0
    requested_copy_counts = sorted({int(value) for value in (copy_counts_to_add or []) if int(value) >= 1})
    existing_copy_counts = {
        int(float(str(row.get("copy_count"))))
        for row in rows
        if str(row.get("copy_count") or "").strip()
    }
    new_copy_counts = [value for value in requested_copy_counts if value not in existing_copy_counts]
    if new_copy_counts and matrix_mode != "sequence_copy_multimer":
        raise ValueError("Additional copy counts are only supported for same-sequence multimer matrices.")
    if new_copy_counts:
        target_pdb = Path(str(params.get("target_pdb") or "")).expanduser().resolve()
        target_chains = [str(chain) for chain in (params.get("target_chains") or []) if str(chain)]
        if not target_pdb.exists() or not target_chains:
            raise ValueError("The original target PDB/source chain is unavailable, so this matrix cannot be extended.")
        source_chain = target_chains[0]
        sequence = str(_sequences_by_chain(target_pdb).get(source_chain) or "").replace("X", "").strip()
        if not sequence:
            raise ValueError(f"Could not extract chain {source_chain} from {target_pdb}.")
        systems_dir = parent_run_dir / "artifacts" / "capacity_systems"
        existing_candidate_ids = {str(row.get("candidate_id") or "") for row in rows}
        for copy_count in new_copy_counts:
            if copy_count < 1:
                continue
            sequence_length = len(sequence)
            candidate_id = f"cap_{source_chain}{sequence_length}_x{copy_count}"
            if candidate_id in existing_candidate_ids:
                continue
            complex_path = systems_dir / f"{candidate_id}.pdb"
            if copy_count == 1:
                complex_text = filter_pdb_text(
                    target_pdb.read_text(errors="ignore"),
                    keep_chains={source_chain},
                    remove_waters=True,
                    remove_hetero=True,
                )
                complex_path.parent.mkdir(parents=True, exist_ok=True)
                complex_path.write_text(complex_text.rstrip() + "\nEND\n")
                binder_chains = [source_chain]
                candidate_target_chains: list[str] = []
            else:
                binder_chains, candidate_target_chains = _write_sequence_copy_multimer(
                    complex_path, sequence, copy_count
                )
            target_id = f"{params.get('target_name') or target_pdb.stem}_{source_chain}_x{copy_count}"
            candidate_target_pdb = str(target_pdb if copy_count == 1 else complex_path)
            row = {
                "candidate_id": candidate_id,
                "target_name": params.get("target_name") or target_id,
                "matrix_mode": matrix_mode,
                "source_chain": source_chain,
                "copy_count": copy_count,
                "sequence_length": sequence_length,
                "target_length": sequence_length * (copy_count - 1),
                "binder_length": sequence_length,
                "total_length": sequence_length * copy_count,
                "complex_pdb": str(complex_path.relative_to(parent_run_dir)),
                "target_pdb": candidate_target_pdb,
                "target_chains": json.dumps(candidate_target_chains),
                "binder_chains": json.dumps(binder_chains),
            }
            rows.append(row)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "stage": STAGE_COMPLEX_REFOLDING,
                    "source_tool": CAPACITY_TOOL,
                    "target_pdb": candidate_target_pdb,
                    "complex_pdb": str(complex_path.relative_to(parent_run_dir)),
                    "binder_sequence": sequence,
                    "binder_chains": binder_chains,
                    "target_chains": candidate_target_chains,
                    "binder_length": str(sequence_length),
                    "metrics": {
                        "target_id": target_id,
                        "matrix_mode": matrix_mode,
                        "source_chain": source_chain,
                        "copy_count": copy_count,
                        "sequence_length": sequence_length,
                        "target_length": sequence_length * (copy_count - 1),
                        "binder_length": sequence_length,
                        "total_length": sequence_length * copy_count,
                    },
                    "raw_metadata": {
                        "capacity_benchmark": True,
                        "matrix_mode": matrix_mode,
                        "capacity_target_only": copy_count == 1,
                        "target_source_pdb": str(target_pdb),
                        "source_chain": source_chain,
                    },
                }
            )
            existing_candidate_ids.add(candidate_id)
            added_systems += 1

    target_panel_entries_to_add = list(target_panel_entries_to_add or [])
    if target_panel_entries_to_add and matrix_mode != "target_panel":
        raise ValueError("Additional target rows are only supported for target-panel matrices.")
    if target_panel_entries_to_add:
        systems_dir = parent_run_dir / "artifacts" / "capacity_systems"
        existing_candidate_ids = {str(row.get("candidate_id") or "") for row in rows}
        existing_target_keys = {
            (
                str(row.get("target_pdb") or ""),
                str(row.get("source_chain") or ""),
                str(row.get("copy_count") or "1"),
            )
            for row in rows
        }
        target_entries = list(params.get("target_panel_entries") or [])
        for index, entry in enumerate(target_panel_entries_to_add, start=len(rows) + 1):
            entry = dict(entry)
            entry["copy_count"] = int(entry.get("copy_count") or 1)
            target_key = (
                str(Path(str(entry.get("target_pdb") or "")).expanduser()),
                str(entry.get("chain") or ""),
                str(entry.get("copy_count") or 1),
            )
            if target_key in existing_target_keys:
                continue
            row, candidate = _target_panel_system_records(
                entry=entry,
                systems_dir=systems_dir,
                run_dir=parent_run_dir,
                index=index,
            )
            if str(row.get("candidate_id") or "") in existing_candidate_ids:
                continue
            rows.append(row)
            candidates.append(candidate)
            target_entries.append(entry)
            existing_candidate_ids.add(str(row.get("candidate_id") or ""))
            existing_target_keys.add(target_key)
            added_systems += 1
            added_targets += 1
        if added_targets:
            params["target_panel_entries"] = target_entries

    existing_engines = [str(engine) for engine in (params.get("engines") or []) if str(engine)]
    added_engines = [
        engine
        for engine in (engines_to_add or [])
        if engine in ENGINE_LABELS and engine not in existing_engines
    ]
    desired_engines = existing_engines + added_engines
    if added_systems:
        write_candidates(parent_run_dir, CAPACITY_TOOL, candidates)
        _write_csv_rows(matrix_path, rows, "candidate_id")

    existing_pairs = {
        (str(child.get("engine") or ""), str(child.get("candidate_id") or ""))
        for child in children
    }
    added_cells = 0
    inherited_failed_cells = 0
    for engine in desired_engines:
        for row in rows:
            pair = (engine, str(row.get("candidate_id") or ""))
            if pair in existing_pairs:
                continue
            child = _enqueue_capacity_child(
                parent_run_dir=parent_run_dir,
                parent_run_id=parent_run_id,
                benchmark_name=benchmark_name,
                engine=engine,
                row=row,
                gpu_device=gpu_device,
                preset=preset,
                params=params,
            )
            row_copy_count = _capacity_int(row.get("copy_count"))
            predecessors = [
                existing
                for existing in existing_child_results
                if str(existing.get("engine_key") or "") == engine
                and _capacity_int(existing.get("copy_count")) is not None
                and row_copy_count is not None
                and int(_capacity_int(existing.get("copy_count")) or 0) < row_copy_count
            ]
            predecessor = (
                max(predecessors, key=lambda item: int(_capacity_int(item.get("copy_count")) or 0))
                if predecessors
                else None
            )
            if predecessor and str(predecessor.get("status") or "") in {"failed", "skipped"}:
                _mark_capacity_child_inherited_failure(
                    Path(str(child["child_run_dir"])),
                    predecessor=predecessor,
                )
                inherited_failed_cells += 1
            children.append(child)
            existing_pairs.add(pair)
            added_cells += 1

    recalculated_cells = 0
    selected_run_ids = {str(value) for value in (recalculate_run_ids or []) if str(value)}
    if selected_run_ids:
        row_by_candidate = {str(row.get("candidate_id") or ""): row for row in rows}
        for child in list(children):
            old_run_id = str(child.get("child_run_id") or "")
            if old_run_id not in selected_run_ids:
                continue
            engine = str(child.get("engine") or "")
            row = row_by_candidate.get(str(child.get("candidate_id") or ""))
            if engine not in ENGINE_LABELS or row is None:
                continue
            replacement = _enqueue_capacity_child(
                parent_run_dir=parent_run_dir,
                parent_run_id=parent_run_id,
                benchmark_name=f"{benchmark_name} recalculation",
                engine=engine,
                row=row,
                gpu_device=gpu_device,
                preset=preset,
                params=params,
            )
            old_run_dir = Path(str(child.get("child_run_dir") or ""))
            old_metadata = read_json(old_run_dir / "metadata.json")
            old_metadata["capacity_superseded_by"] = replacement["child_run_id"]
            old_metadata["capacity_superseded_at"] = time.strftime(
                "%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()
            )
            write_json(old_run_dir / "metadata.json", old_metadata)
            children.append(replacement)
            recalculated_cells += 1

    if not (added_cells or recalculated_cells):
        return {
            "added_systems": 0,
            "added_targets": 0,
            "added_engines": 0,
            "added_cells": 0,
            "inherited_failed_cells": 0,
            "recalculated_cells": 0,
        }

    params["engines"] = desired_engines
    if matrix_mode == "sequence_copy_multimer":
        params["copy_counts"] = sorted(existing_copy_counts | set(new_copy_counts))
    payload["params"] = params
    write_json(parent_run_dir / "input.json", payload)
    _write_csv_rows(children_path, children, "child_run_id")

    result = read_json(parent_run_dir / "result.json")
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    result["success"] = None
    result["metrics"] = {
        **metrics,
        "cell_count": len(children) - recalculated_cells,
        "system_count": len(rows),
        "engine_count": len(desired_engines),
        "globally_sequential": True,
        "extended_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
    }
    write_json(parent_run_dir / "result.json", result)
    _capacity_scheduler_files(parent_run_dir)

    parent_metadata = read_json(parent_run_dir / "metadata.json")
    scheduler_alive = _worker_is_alive(parent_metadata, parent_run_dir)
    if launch and not scheduler_alive:
        parent_metadata.pop("worker_pid", None)
        parent_metadata.pop("worker_started_at", None)
        write_json(parent_run_dir / "metadata.json", parent_metadata)
        update_status(parent_run_dir, "running", current_phase="Scheduling added capacity cells")
        spawn_worker_for_run(parent_run_dir)
    elif not launch:
        update_status(parent_run_dir, "queued", current_phase="Added capacity cells waiting to launch")

    return {
        "added_systems": added_systems,
        "added_targets": added_targets,
        "added_engines": len(added_engines),
        "added_cells": added_cells,
        "inherited_failed_cells": inherited_failed_cells,
        "recalculated_cells": recalculated_cells,
    }


def _capacity_child_status(child_run_dir: Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
    metadata = read_json(child_run_dir / "metadata.json")
    result = read_json(child_run_dir / "result.json")
    status = str(metadata.get("status") or "")
    if not status:
        if result.get("success") is True:
            status = "completed"
        elif result.get("success") is False:
            status = "failed"
        else:
            status = "queued"
    if status == "completed" and _capacity_child_missing_native_prediction(child_run_dir, metadata):
        status = "failed"
        message = (
            f"{metadata.get('capacity_engine_label') or metadata.get('capacity_engine') or 'Engine'} "
            "completed without producing a native prediction CIF/PDB; staged input complex is not a valid capacity prediction."
        )
        metadata = {
            **metadata,
            "status": "failed",
            "worker_error": metadata.get("worker_error") or message,
            "worker_exception_type": metadata.get("worker_exception_type") or "MissingPredictionStructure",
            "current_phase": metadata.get("current_phase") or "No native prediction structure produced",
        }
        result_metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        result = {
            **result,
            "success": False,
            "metrics": {
                **result_metrics,
                "worker_error": result_metrics.get("worker_error") or message,
                "worker_exception_type": result_metrics.get("worker_exception_type") or "MissingPredictionStructure",
                "native_prediction_structure_count": 0,
            },
        }
        try:
            write_json(child_run_dir / "metadata.json", metadata)
            write_json(child_run_dir / "result.json", result)
        except OSError:
            pass
    return status, metadata, result


def _capacity_child_missing_native_prediction(child_run_dir: Path, metadata: dict[str, Any]) -> bool:
    engine = str(metadata.get("capacity_engine") or "")
    source_label = {
        "alphafast_af3": "af3",
        "af2_initial_guess": "af2",
        "colabfold": "colabfold",
        "boltz2_initial_guess": "boltz2",
        "protenix": "protenix",
        "boltzgen_fold": "boltzgen_fold",
    }.get(engine)
    if not source_label:
        return False
    candidate_id = str(metadata.get("capacity_candidate_id") or "").strip()
    if not candidate_id:
        return False
    return _child_viewer_prediction_path(child_run_dir, source_label, _safe_id(candidate_id).lower()) is None


def _mark_capacity_child_skipped(child_run_dir: Path, reason: str) -> None:
    metadata = read_json(child_run_dir / "metadata.json")
    now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    metadata["status"] = "skipped"
    metadata["updated_at"] = now
    metadata["completed_at"] = now
    metadata["skip_reason"] = reason
    metadata["current_phase"] = reason
    write_json(child_run_dir / "metadata.json", metadata)
    result = read_json(child_run_dir / "result.json")
    result.update({"success": None, "skipped": True, "skip_reason": reason})
    write_json(child_run_dir / "result.json", result)


def _mark_capacity_child_inherited_failure(
    child_run_dir: Path,
    *,
    predecessor: dict[str, Any],
) -> None:
    predecessor_copy = predecessor.get("copy_count")
    predecessor_run_id = predecessor.get("run_id")
    reason = (
        f"Inherited capacity failure: copy count {predecessor_copy} failed, "
        "so this larger system was not launched."
    )
    metadata = read_json(child_run_dir / "metadata.json")
    metadata["capacity_inherited_failure"] = True
    metadata["capacity_failure_predecessor_run_id"] = predecessor_run_id
    metadata["capacity_failure_predecessor_copy_count"] = predecessor_copy
    metadata["worker_error"] = reason
    metadata["current_phase"] = reason
    write_json(child_run_dir / "metadata.json", metadata)
    result = read_json(child_run_dir / "result.json")
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    result["success"] = False
    result["metrics"] = {
        **metrics,
        "worker_error": reason,
        "capacity_inherited_failure": True,
        "capacity_failure_predecessor_run_id": predecessor_run_id,
        "capacity_failure_predecessor_copy_count": predecessor_copy,
    }
    write_json(child_run_dir / "result.json", result)
    update_status(
        child_run_dir,
        "failed",
        capacity_inherited_failure=True,
        capacity_failure_predecessor_run_id=predecessor_run_id,
        capacity_failure_predecessor_copy_count=predecessor_copy,
        worker_error=reason,
        current_phase=reason,
    )


def _capacity_parent_summary(parent_run_dir: Path) -> dict[str, Any]:
    children = capacity_child_rows(parent_run_dir)
    total = len(children)
    completed = sum(1 for child in children if child.get("status") == "completed")
    failed = sum(1 for child in children if child.get("status") == "failed")
    skipped = sum(1 for child in children if child.get("status") == "skipped")
    active = sum(1 for child in children if child.get("status") in CAPACITY_ACTIVE_STATUSES)
    terminal = completed + failed + skipped + sum(1 for child in children if child.get("status") in {"cancelled", "stopped"})
    if total and active:
        status = "running"
    elif total and terminal >= total:
        status = "completed"
    else:
        status = "queued"
    return {
        "status": status,
        "cell_count": total,
        "completed_cells": completed,
        "failed_cells": failed,
        "skipped_cells": skipped,
        "running_cells": active,
    }


def run_capacity_benchmark_scheduler(run_dir: Path, poll_seconds: int = 10) -> None:
    run_dir = Path(run_dir)
    children_path = run_dir / "artifacts" / "capacity_children.csv"
    if not children_path.exists():
        finish_job(run_dir, False, {"metrics": {"error": "capacity_children.csv missing"}})
        return
    while True:
        parent_metadata = read_json(run_dir / "metadata.json")
        if str(parent_metadata.get("status") or "") in {"cancelled", "stopped", "paused"}:
            return
        with children_path.open(newline="") as handle:
            all_children = list(csv.DictReader(handle))
        children = []
        for child in all_children:
            child_run_dir = Path(str(child.get("child_run_dir") or ""))
            child_metadata = read_json(child_run_dir / "metadata.json")
            if child_metadata.get("capacity_superseded_by"):
                continue
            children.append(child)
        by_engine: dict[str, list[dict[str, Any]]] = {}
        for row in children:
            by_engine.setdefault(str(row.get("engine") or ""), []).append(row)
        for engine_rows in by_engine.values():
            engine_rows.sort(key=lambda row: int(float(str(row.get("total_length") or 0))))

        launched_any = False
        all_done = True
        summary = _capacity_parent_summary(run_dir)
        update_status(
            run_dir,
            "running",
            current_phase=(
                f"Capacity cells: {summary['completed_cells']} completed, "
                f"{summary['failed_cells']} failed, {summary['skipped_cells']} skipped"
            ),
        )
        global_active = False
        for child in children:
            child_run_dir = Path(str(child.get("child_run_dir") or ""))
            status, metadata, _result = _capacity_child_status(child_run_dir)
            if status in {"running", "preparing"} or (status == "queued" and metadata.get("worker_pid")):
                _record_capacity_gpu_memory(child_run_dir, metadata.get("capacity_gpu_device") or "0")
                global_active = True
                break
        if global_active:
            time.sleep(max(1, int(poll_seconds or 10)))
            continue

        for engine, engine_rows in by_engine.items():
            child_infos = []
            for row in engine_rows:
                child_run_dir = Path(str(row.get("child_run_dir") or ""))
                status, metadata, _result = _capacity_child_status(child_run_dir)
                child_infos.append((row, child_run_dir, status, metadata))

            capacity_failures = []
            transient_failures = []
            for row, child_run_dir, status, metadata in child_infos:
                if status != "failed":
                    continue
                detail = _capacity_child_failure_detail(child_run_dir)
                result = read_json(child_run_dir / "result.json")
                metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
                failure_kind = _capacity_failure_kind(
                    metrics.get("worker_error") or metadata.get("worker_error"),
                    metrics.get("worker_exception_type") or metadata.get("worker_exception_type"),
                    detail,
                )
                if failure_kind in {
                    "model_load_oom",
                    "inference_oom",
                    "output_oom",
                    "inherited_failure",
                    "missing_prediction_structure",
                }:
                    capacity_failures.append((row, child_run_dir, metadata))
                else:
                    transient_failures.append((row, child_run_dir, metadata, failure_kind))

            retried = False
            for _row, child_run_dir, metadata, failure_kind in transient_failures:
                retry_count = int(metadata.get("capacity_retry_count") or 0)
                if retry_count >= 1:
                    continue
                prepare_job_for_resume(child_run_dir)
                retry_metadata = read_json(child_run_dir / "metadata.json")
                retry_metadata["capacity_retry_count"] = retry_count + 1
                retry_metadata["capacity_retry_reason"] = failure_kind or "non-capacity failure"
                write_json(child_run_dir / "metadata.json", retry_metadata)
                spawn_worker_for_run(child_run_dir)
                retried = True
                launched_any = True
                all_done = False
                break
            if retried:
                break

            if capacity_failures:
                for _row, child_run_dir, status, metadata in child_infos:
                    if status == "queued" and not metadata.get("worker_pid"):
                        _mark_capacity_child_skipped(child_run_dir, f"Skipped after earlier {engine} capacity failure")
                continue

            if any(
                status in {"running", "preparing"} or (status == "queued" and metadata.get("worker_pid"))
                for _row, _path, status, metadata in child_infos
            ):
                all_done = False
                continue

            pending = [
                (row, child_run_dir, metadata)
                for row, child_run_dir, status, metadata in child_infos
                if status == "queued" and not metadata.get("worker_pid")
            ]
            if pending:
                all_done = False
                _row, child_run_dir, _metadata = pending[0]
                spawn_worker_for_run(child_run_dir)
                launched_any = True
                break

            if any(status not in CAPACITY_TERMINAL_STATUSES for _row, _path, status, _metadata in child_infos):
                all_done = False

        if all_done:
            summary = _capacity_parent_summary(run_dir)
            result = read_json(run_dir / "result.json")
            outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
            finish_job(
                run_dir,
                True,
                {
                    "outputs": outputs,
                    "metrics": {
                        **(result.get("metrics") if isinstance(result.get("metrics"), dict) else {}),
                        **summary,
                    },
                },
            )
            return
        time.sleep(1 if launched_any else max(1, int(poll_seconds or 10)))


def capacity_parent_rows() -> list[dict[str, Any]]:
    rows = []
    for row in collect_jobs(CAPACITY_GROUP):
        run_dir = Path(str(row.get("run_dir") or ""))
        payload = read_json(run_dir / "input.json")
        if payload.get("job_type") != CAPACITY_JOB_TYPE:
            continue
        result = read_json(run_dir / "result.json")
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
        child_rows = capacity_child_rows(run_dir)
        child_summary = _capacity_parent_summary(run_dir) if child_rows else {}
        sequence_lengths = sorted(
            {
                value
                for value in (_capacity_int(child.get("sequence_length")) for child in child_rows)
                if value is not None
            }
        )
        target_name = str(params.get("target_name") or "")
        target_chains = [str(chain) for chain in (params.get("target_chains") or []) if str(chain)]
        rows.append(
            {
                **row,
                "status": child_summary.get("status") or row.get("status"),
                "result": f"/results?task_group={row['task_group']}&run_id={row['run_id']}",
                "benchmark": params.get("benchmark_name") or row.get("campaign_name") or row.get("job_code"),
                "umbrella_type": "Folding capacity",
                "preset": params.get("preset") or "",
                "matrix_mode": params.get("matrix_mode") or "",
                "target": target_name or Path(str(params.get("target_pdb") or "")).stem,
                "target_chains": ", ".join(target_chains),
                "sequence_length": ", ".join(str(value) for value in sequence_lengths),
                "source_capacity_run_id": params.get("source_capacity_run_id") or "",
                "engines": ", ".join(ENGINE_LABELS.get(engine, engine) for engine in params.get("engines", [])),
                "systems": metrics.get("system_count", ""),
                "cells": metrics.get("cell_count", ""),
                "completed_cells": child_summary.get("completed_cells", sum(1 for child in child_rows if child.get("status") == "completed")),
                "failed_cells": child_summary.get("failed_cells", sum(1 for child in child_rows if child.get("status") == "failed")),
                "skipped_cells": child_summary.get("skipped_cells", sum(1 for child in child_rows if child.get("status") == "skipped")),
                "running_cells": child_summary.get("running_cells", sum(1 for child in child_rows if child.get("status") in CAPACITY_ACTIVE_STATUSES)),
            }
        )
    return rows


def capacity_child_rows(parent_run_dir: Path) -> list[dict[str, Any]]:
    parent_run_dir = Path(parent_run_dir)
    parent_id = parent_run_dir.name
    rows: list[dict[str, Any]] = []
    for row in collect_jobs("benchmark", include_hidden=True):
        run_dir = Path(str(row.get("run_dir") or ""))
        status, metadata, result = _capacity_child_status(run_dir)
        if str(metadata.get("capacity_parent_run_id") or "") != parent_id:
            continue
        if metadata.get("capacity_superseded_by"):
            continue
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        failure_detail = _capacity_child_failure_detail(run_dir) if status == "failed" else ""
        failure_kind = _capacity_failure_kind(
            metrics.get("worker_error") or metadata.get("worker_error"),
            metrics.get("worker_exception_type") or metadata.get("worker_exception_type"),
            failure_detail,
        )
        rows.append(
            {
                **row,
                "status": status,
                "result": f"/results?task_group={row['task_group']}&run_id={row['run_id']}",
                "engine": metadata.get("capacity_engine_label") or metadata.get("capacity_engine"),
                "engine_key": metadata.get("capacity_engine"),
                "candidate_id": metadata.get("capacity_candidate_id"),
                "matrix_mode": metadata.get("capacity_matrix_mode"),
                "copy_count": metadata.get("capacity_copy_count"),
                "sequence_length": metadata.get("capacity_sequence_length"),
                "target_length": metadata.get("capacity_target_length"),
                "binder_length": metadata.get("capacity_binder_length"),
                "total_length": metadata.get("capacity_total_length"),
                "peak_gpu_memory_mib": metadata.get("capacity_peak_gpu_memory_mib"),
                "gpu_total_memory_mib": metadata.get("capacity_gpu_total_memory_mib"),
                "worker_error": metrics.get("worker_error") or metadata.get("worker_error") or "",
                "exception": metrics.get("worker_exception_type") or metadata.get("worker_exception_type") or "",
                "failure_kind": failure_kind,
            }
        )
    return sorted(rows, key=lambda item: (str(item.get("engine") or ""), int(item.get("total_length") or 0)))


def _capacity_int(value: object) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _capacity_failure_kind(error: object, exception: object = "", detail: object = "") -> str:
    text = f"{error or ''} {exception or ''} {detail or ''}".lower()
    if "inherited capacity failure" in text:
        return "inherited_failure"
    if "missingpredictionstructure" in text or "native prediction" in text:
        return "missing_prediction_structure"
    if "mmseqs" in text and any(token in text for token in ["out of memory", "cuda oom", "cuda error"]):
        return "msa_oom"
    if any(token in text for token in ["out of memory", "cuda oom", "cublas", "cudnn", "memoryerror"]):
        if "predicted_aligned_error" in text or "full pae" in text or "write pae" in text:
            return "output_oom"
        if any(token in text for token in ["load model", "loading model", "model weights", "weights"]):
            return "model_load_oom"
        return "inference_oom"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if text.strip():
        return "failed"
    return ""


def _capacity_child_failure_detail(child_run_dir: Path) -> str:
    parts: list[str] = []
    for name in ("stderr.log", "worker_stderr.log"):
        path = child_run_dir / name
        if not path.exists():
            continue
        try:
            parts.append(path.read_text(errors="replace")[-20000:])
        except OSError:
            continue
    return "\n".join(parts)


def _gpu_memory_snapshot(gpu_device: object) -> tuple[int | None, int | None]:
    device = str(gpu_device or "").removeprefix("device=").strip()
    if not device.isdigit():
        return None, None
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device}",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
        )
        first_line = next((line for line in proc.stdout.splitlines() if line.strip()), "")
        used_text, total_text = [part.strip() for part in first_line.split(",", 1)]
        return int(used_text), int(total_text)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None, None


def _record_capacity_gpu_memory(child_run_dir: Path, gpu_device: object) -> None:
    used_mib, total_mib = _gpu_memory_snapshot(gpu_device)
    if used_mib is None:
        return
    metadata = read_json(child_run_dir / "metadata.json")
    previous = int(metadata.get("capacity_peak_gpu_memory_mib") or 0)
    metadata["capacity_peak_gpu_memory_mib"] = max(previous, used_mib)
    if total_mib is not None:
        metadata["capacity_gpu_total_memory_mib"] = total_mib
    metadata["capacity_gpu_memory_sampled_at"] = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    write_json(child_run_dir / "metadata.json", metadata)


def capacity_limit_rows(
    *,
    parent_run_dir: Path | None = None,
    matrix_mode: str | None = None,
    preset: str | None = None,
    gpu_device: str | None = None,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    parents = capacity_parent_rows()
    if parent_run_dir is not None:
        selected = Path(parent_run_dir).expanduser().resolve()
        parents = [
            parent
            for parent in parents
            if Path(str(parent.get("run_dir") or "")).expanduser().resolve() == selected
        ]
    for parent in parents:
        parent_run_dir = Path(str(parent.get("run_dir") or ""))
        payload = read_json(parent_run_dir / "input.json")
        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
        parent_preset = str(params.get("preset") or "")
        parent_gpu = str(params.get("capacity_device") or params.get("gpu_device") or "")
        if preset is not None and parent_preset != str(preset):
            continue
        if gpu_device is not None and parent_gpu != str(gpu_device):
            continue
        for child in capacity_child_rows(parent_run_dir):
            child_mode = str(child.get("matrix_mode") or "")
            if matrix_mode is not None and child_mode != str(matrix_mode):
                continue
            engine_key = str(child.get("engine_key") or "")
            if not engine_key:
                continue
            key = (engine_key, child_mode, parent_preset, parent_gpu)
            group = groups.setdefault(
                key,
                {
                    "engine_key": engine_key,
                    "engine": child.get("engine") or ENGINE_LABELS.get(engine_key, engine_key),
                    "matrix_mode": child_mode,
                    "preset": parent_preset,
                    "gpu_device": parent_gpu,
                    "tested_cells": 0,
                    "completed_cells": 0,
                    "failed_cells": 0,
                    "max_success_total_length": None,
                    "min_failed_total_length": None,
                    "min_oom_total_length": None,
                    "last_failure_kind": "",
                },
            )
            total_length = _capacity_int(child.get("total_length"))
            status = str(child.get("status") or "")
            if status not in {"completed", "failed"} or total_length is None:
                continue
            group["tested_cells"] += 1
            if status == "completed":
                group["completed_cells"] += 1
                current = group.get("max_success_total_length")
                group["max_success_total_length"] = total_length if current is None else max(int(current), total_length)
            elif status == "failed":
                group["failed_cells"] += 1
                current = group.get("min_failed_total_length")
                group["min_failed_total_length"] = total_length if current is None else min(int(current), total_length)
                failure_kind = str(child.get("failure_kind") or "")
                group["last_failure_kind"] = failure_kind or group.get("last_failure_kind") or "failed"
                if failure_kind in {"model_load_oom", "inference_oom", "output_oom"}:
                    oom_current = group.get("min_oom_total_length")
                    group["min_oom_total_length"] = total_length if oom_current is None else min(int(oom_current), total_length)
    return sorted(
        [row for row in groups.values() if int(row.get("tested_cells") or 0) > 0],
        key=lambda row: (str(row.get("engine") or ""), str(row.get("preset") or ""), str(row.get("gpu_device") or "")),
    )


def capacity_warnings(
    *,
    engine_keys: list[str],
    total_length: int | None,
    matrix_mode: str = "sequence_copy_multimer",
    preset: str | None = None,
    gpu_device: str | None = None,
) -> list[dict[str, Any]]:
    requested = _capacity_int(total_length)
    if requested is None or requested <= 0:
        return []
    exact_rows = capacity_limit_rows(matrix_mode=matrix_mode, preset=preset, gpu_device=gpu_device)
    gpu_rows = capacity_limit_rows(matrix_mode=matrix_mode, gpu_device=gpu_device) if gpu_device is not None else []
    all_rows = capacity_limit_rows(matrix_mode=matrix_mode)

    def _pick(engine_key: str) -> tuple[dict[str, Any] | None, str]:
        for rows, scope in [
            (exact_rows, "same GPU/run depth"),
            (gpu_rows, "same GPU"),
            (all_rows, "any capacity run"),
        ]:
            candidates = [row for row in rows if str(row.get("engine_key")) == str(engine_key)]
            if not candidates:
                continue
            candidates.sort(
                key=lambda row: (
                    _capacity_int(row.get("min_failed_total_length")) is None,
                    _capacity_int(row.get("min_failed_total_length")) or 10**12,
                    -int(row.get("tested_cells") or 0),
                    -(_capacity_int(row.get("max_success_total_length")) or 0),
                )
            )
            return candidates[0], scope
        return None, ""

    messages: list[dict[str, Any]] = []
    for engine_key in engine_keys:
        row, evidence_scope = _pick(str(engine_key))
        label = ENGINE_LABELS.get(str(engine_key), str(engine_key))
        if not row or int(row.get("tested_cells") or 0) == 0:
            messages.append(
                {
                    "severity": "info",
                    "engine": label,
                    "message": f"{label}: no capacity result yet.",
                }
            )
            continue
        max_success = _capacity_int(row.get("max_success_total_length"))
        min_failed = _capacity_int(row.get("min_failed_total_length"))
        min_oom = _capacity_int(row.get("min_oom_total_length"))
        failure_limit = min_oom
        if failure_limit is not None and requested >= failure_limit:
            messages.append(
                {
                    "severity": "error",
                    "engine": label,
                    "message": f"{label}: requested {requested} residues is at/above known model OOM threshold ({failure_limit}; evidence: {evidence_scope}).",
                    "evidence_scope": evidence_scope,
                    **row,
                }
            )
        elif max_success is not None and requested > max_success:
            messages.append(
                {
                    "severity": "warning",
                    "engine": label,
                    "message": f"{label}: requested {requested} residues is above the largest successful capacity test ({max_success}; evidence: {evidence_scope}).",
                    "evidence_scope": evidence_scope,
                    **row,
                }
            )
        else:
            messages.append(
                {
                    "severity": "ok",
                    "engine": label,
                    "message": f"{label}: requested {requested} residues is within the tested successful range ({evidence_scope}).",
                    "evidence_scope": evidence_scope,
                    **row,
                }
            )
    return messages
