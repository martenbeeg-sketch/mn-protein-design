from __future__ import annotations

import gzip
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from mn_protein_design.core.artifacts import Artifact, artifact_path
from mn_protein_design.core.candidates import STAGE_GENERATION_BACKBONE, STAGE_SEQUENCE_DESIGN, read_candidates, write_candidates
from mn_protein_design.core.jobs import create_job, finish_job, update_status, write_json
from mn_protein_design.core.manifests import load_manifest


DESIGN_GROUP = "design"


def _rel_path(run_dir: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def _resolve_candidate_path(source_run_dir: Path, path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text)
    return path if path.is_absolute() else source_run_dir / path


def _chain_ids_from_pdb(path: Path) -> list[str]:
    chains: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        chain = line[21].strip() or "_"
        if chain not in seen:
            chains.append(chain)
            seen.add(chain)
    return chains


def _infer_design_chains(candidate: dict[str, Any], pdb_path: Path, requested: str = "") -> list[str]:
    if requested.strip():
        return [token.strip() for token in re.split(r"[,;\s]+", requested) if token.strip()]
    target_chains = set(candidate.get("target_chains") or [])
    chains = _chain_ids_from_pdb(pdb_path)
    binder_chains = [chain for chain in chains if chain not in target_chains]
    return binder_chains or chains[-1:]


def _read_fasta_sequences(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    header = ""
    seq: list[str] = []
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith(">"):
            if header or seq:
                rows.append({"header": header, "sequence": "".join(seq)})
            header = line[1:].strip()
            seq = []
        else:
            seq.append(line.strip())
    if header or seq:
        rows.append({"header": header, "sequence": "".join(seq)})
    parsed: list[dict[str, Any]] = []
    for row in rows:
        metrics: dict[str, Any] = {}
        for key in ["score", "global_score", "seq_recovery"]:
            match = re.search(rf"{key}=(-?\d+(?:\.\d+)?)", row["header"])
            if match:
                metrics[key] = float(match.group(1))
        parsed.append({**row, "metrics": metrics})
    return parsed


def _run_shell_steps(run_dir: Path, steps: list[dict]) -> int:
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


def _collect_artifacts(run_dir: Path, root: Path, patterns: list[tuple[str, str]]) -> list[dict]:
    artifacts: list[dict] = []
    for pattern, artifact_type in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                artifacts.append(Artifact(path.stem, path, artifact_type).to_json(run_dir))
    return artifacts


def _normalize_ligandmpnn_candidates(
    run_dir: Path,
    source_candidates: list[dict[str, Any]],
    candidate_input_paths: dict[str, Path],
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    raw_root = run_dir / "artifacts" / "raw" / "ligandmpnn"
    for source in source_candidates:
        source_id = source.get("candidate_id")
        if not source_id:
            continue
        seq_dir = raw_root / "output" / source_id / "seqs"
        fasta_files = sorted(seq_dir.glob("*.fa"))
        input_pdb = candidate_input_paths.get(source_id)
        design_chains = params.get("design_chains_by_candidate", {}).get(source_id, [])
        for fasta_path in fasta_files:
            for index, seq_row in enumerate(_read_fasta_sequences(fasta_path), start=1):
                normalized.append(
                    {
                        "candidate_id": f"{source_id}_{params['model_type']}_{index:03d}",
                        "stage": STAGE_SEQUENCE_DESIGN,
                        "source_tool": params["model_type"],
                        "target_pdb": source.get("target_pdb"),
                        "complex_pdb": _rel_path(run_dir, input_pdb),
                        "binder_pdb": None,
                        "binder_sequence": seq_row.get("sequence"),
                        "target_chains": source.get("target_chains", []),
                        "binder_chains": design_chains,
                        "hotspots": source.get("hotspots", []),
                        "binder_length": source.get("binder_length"),
                        "contig": source.get("contig"),
                        "metrics": seq_row.get("metrics") or {},
                        "parents": [source_id],
                        "raw_metadata": {
                            "source_candidate": source,
                            "fasta": _rel_path(run_dir, fasta_path),
                            "fasta_header": seq_row.get("header"),
                        },
                    }
                )
    return write_candidates(run_dir, params["model_type"], normalized)


def run_ligandmpnn_sequence_design(
    source_run_dir: Path,
    candidates_jsonl: Path,
    model_type: str = "protein_mpnn",
    design_chains: str = "",
    num_seq_per_target: int = 1,
    sampling_temp: float = 0.0001,
    omit_aas: str = "CX",
    seed: int | None = None,
) -> Path:
    source_run_dir = Path(source_run_dir)
    candidates_jsonl = Path(candidates_jsonl)
    if model_type not in {"protein_mpnn", "ligand_mpnn", "soluble_mpnn"}:
        raise ValueError("model_type must be protein_mpnn, ligand_mpnn, or soluble_mpnn.")
    if num_seq_per_target < 1:
        raise ValueError("Sequences per backbone must be at least 1.")
    source_candidates = [
        candidate for candidate in read_candidates(candidates_jsonl) if candidate.get("stage") == STAGE_GENERATION_BACKBONE
    ]
    if not source_candidates:
        raise ValueError("No generation.backbone candidates were found in the selected candidate set.")

    manifest = load_manifest("ligandmpnn")
    params = {
        "source_run_dir": str(source_run_dir),
        "candidates_jsonl": str(candidates_jsonl),
        "model_type": model_type,
        "design_chains": design_chains,
        "num_seq_per_target": num_seq_per_target,
        "sampling_temp": sampling_temp,
        "omit_AAs": omit_aas,
        "seed": seed,
    }
    job = create_job(
        DESIGN_GROUP,
        job_type="sequence_design",
        tool="ligandmpnn",
        inputs={"source_run_dir": str(source_run_dir), "candidates_jsonl": str(candidates_jsonl)},
        params=params,
    )

    raw_root = job.run_dir / "artifacts" / "raw" / "ligandmpnn"
    input_dir = raw_root / "inputs"
    output_dir = raw_root / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_input_paths: dict[str, Path] = {}
    design_chains_by_candidate: dict[str, list[str]] = {}
    steps: list[dict] = []

    for candidate in source_candidates:
        source_id = str(candidate["candidate_id"])
        source_pdb = _resolve_candidate_path(source_run_dir, candidate.get("complex_pdb") or candidate.get("binder_pdb"))
        if source_pdb is None or not source_pdb.exists() or source_pdb.suffix.lower() != ".pdb":
            continue
        staged_pdb = artifact_path(job.run_dir, "raw", "ligandmpnn", "inputs", f"{source_id}.pdb")
        shutil.copy2(source_pdb, staged_pdb)
        candidate_input_paths[source_id] = staged_pdb
        chains = _infer_design_chains(candidate, staged_pdb, design_chains)
        design_chains_by_candidate[source_id] = chains
        (output_dir / source_id).mkdir(parents=True, exist_ok=True)
        args = [
            "python",
            "/opt/LigandMPNN/run.py",
            "--pdb_path",
            f"/work/artifacts/raw/ligandmpnn/inputs/{source_id}.pdb",
            "--out_folder",
            f"/work/artifacts/raw/ligandmpnn/output/{source_id}",
            "--model_type",
            model_type,
            "--chains_to_design",
            ",".join(chains),
            "--number_of_batches",
            str(num_seq_per_target),
            "--batch_size",
            "1",
            "--temperature",
            str(sampling_temp),
            "--omit_AA",
            omit_aas,
            "--verbose",
            "1",
        ]
        if seed is not None:
            args.extend(["--seed", str(seed)])
        command = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "-v",
            f"{job.run_dir}:/work",
            "-w",
            "/work",
            manifest["image"],
            *args,
        ]
        steps.append({"name": f"{model_type}-{source_id}", "command": command})

    if not steps:
        raise ValueError("No selected candidates had an existing PDB structure file. Foundry CIF outputs should use the Foundry-native MPNN path or be converted first.")

    params["design_chains_by_candidate"] = design_chains_by_candidate
    write_json(raw_root / "ligandmpnn_params.json", params)
    rc = _run_shell_steps(job.run_dir, steps)
    candidates = _normalize_ligandmpnn_candidates(job.run_dir, source_candidates, candidate_input_paths, params) if rc == 0 else []
    artifacts = _collect_artifacts(
        job.run_dir,
        job.run_dir / "artifacts",
        [
            ("raw/ligandmpnn/**/*.fa", "fasta"),
            ("raw/ligandmpnn/**/*.pdb", "pdb"),
            ("raw/ligandmpnn/**/*.json", "sequence_design_input"),
            ("normalized_candidates/*.jsonl", "normalized_candidates"),
            ("normalized_candidates/*.json", "campaign_result"),
        ],
    )
    finish_job(
        job.run_dir,
        rc == 0 and bool(candidates),
        {
            "outputs": {"artifacts": artifacts, "candidates": candidates},
            "metrics": {"return_code": rc, "artifact_count": len(artifacts), "candidate_count": len(candidates)},
            "downstream_artifacts": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
            },
        },
    )
    return job.run_dir


def _stage_foundry_structure(source_path: Path, staged_path: Path) -> None:
    staged_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.name.endswith(".gz"):
        with gzip.open(source_path, "rb") as src, staged_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    else:
        shutil.copy2(source_path, staged_path)


def _normalize_foundry_mpnn_candidates(
    run_dir: Path,
    source_candidates: list[dict[str, Any]],
    candidate_input_paths: dict[str, Path],
    params: dict[str, Any],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    raw_root = run_dir / "artifacts" / "raw" / "foundry_mpnn"
    for source in source_candidates:
        source_id = source.get("candidate_id")
        if not source_id:
            continue
        output_dir = raw_root / "output" / source_id
        cif_files = sorted(output_dir.glob("*.cif"))
        input_path = candidate_input_paths.get(source_id)
        for index, cif_path in enumerate(cif_files, start=1):
            normalized.append(
                {
                    "candidate_id": f"{source_id}_foundry_mpnn_{index:03d}",
                    "stage": STAGE_SEQUENCE_DESIGN,
                    "source_tool": "foundry_mpnn",
                    "target_pdb": source.get("target_pdb"),
                    "complex_pdb": _rel_path(run_dir, cif_path),
                    "binder_pdb": None,
                    "binder_sequence": None,
                    "target_chains": source.get("target_chains", []),
                    "binder_chains": [],
                    "hotspots": source.get("hotspots", []),
                    "binder_length": source.get("binder_length"),
                    "contig": source.get("contig"),
                    "metrics": {},
                    "parents": [source_id],
                    "raw_metadata": {
                        "source_candidate": source,
                        "input_cif": _rel_path(run_dir, input_path),
                        "mpnn_cif": _rel_path(run_dir, cif_path),
                    },
                }
            )
    return write_candidates(run_dir, "foundry_mpnn", normalized)


def run_foundry_mpnn_sequence_design(
    source_run_dir: Path,
    candidates_jsonl: Path,
    number_of_batches: int = 1,
    batch_size: int = 10,
    checkpoint_path: str = "/weights/ligandmpnn_v_32_010_25.pt",
) -> Path:
    source_run_dir = Path(source_run_dir)
    candidates_jsonl = Path(candidates_jsonl)
    if number_of_batches < 1:
        raise ValueError("Number of batches must be at least 1.")
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")
    source_candidates = [
        candidate for candidate in read_candidates(candidates_jsonl) if candidate.get("stage") == STAGE_GENERATION_BACKBONE
    ]
    if not source_candidates:
        raise ValueError("No generation.backbone candidates were found in the selected candidate set.")

    manifest = load_manifest("foundry_mpnn")
    params = {
        "source_run_dir": str(source_run_dir),
        "candidates_jsonl": str(candidates_jsonl),
        "number_of_batches": number_of_batches,
        "batch_size": batch_size,
        "checkpoint_path": checkpoint_path,
        "model_type": "ligand_mpnn",
    }
    job = create_job(
        DESIGN_GROUP,
        job_type="sequence_design",
        tool="foundry_mpnn",
        inputs={"source_run_dir": str(source_run_dir), "candidates_jsonl": str(candidates_jsonl)},
        params=params,
    )

    raw_root = job.run_dir / "artifacts" / "raw" / "foundry_mpnn"
    input_dir = raw_root / "inputs"
    output_dir = raw_root / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_input_paths: dict[str, Path] = {}
    steps: list[dict] = []

    for candidate in source_candidates:
        source_id = str(candidate["candidate_id"])
        source_structure = _resolve_candidate_path(source_run_dir, candidate.get("complex_pdb") or candidate.get("binder_pdb"))
        if source_structure is None or not source_structure.exists():
            continue
        if not (source_structure.name.endswith(".cif") or source_structure.name.endswith(".cif.gz")):
            continue
        staged_cif = artifact_path(job.run_dir, "raw", "foundry_mpnn", "inputs", f"{source_id}.cif")
        _stage_foundry_structure(source_structure, staged_cif)
        candidate_input_paths[source_id] = staged_cif
        (output_dir / source_id).mkdir(parents=True, exist_ok=True)
        command = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "-v",
            f"{job.run_dir}:/work",
            "-v",
            "/mnt/db/reference_files/foundry:/weights:ro",
            "-w",
            "/work",
            manifest["image"],
            "mpnn",
            "--structure_path",
            f"/work/artifacts/raw/foundry_mpnn/inputs/{source_id}.cif",
            "--checkpoint_path",
            checkpoint_path,
            "--is_legacy_weights",
            "True",
            "--model_type",
            "ligand_mpnn",
            "--batch_size",
            str(batch_size),
            "--number_of_batches",
            str(number_of_batches),
            "--remove_waters",
            "True",
            "--out_directory",
            f"/work/artifacts/raw/foundry_mpnn/output/{source_id}",
        ]
        steps.append({"name": f"foundry-mpnn-{source_id}", "command": command})

    if not steps:
        raise ValueError("No selected candidates had an existing CIF structure file. RFdiffusion PDB outputs should use the shared LigandMPNN backend.")

    write_json(raw_root / "foundry_mpnn_params.json", params)
    rc = _run_shell_steps(job.run_dir, steps)
    candidates = _normalize_foundry_mpnn_candidates(job.run_dir, source_candidates, candidate_input_paths, params) if rc == 0 else []
    artifacts = _collect_artifacts(
        job.run_dir,
        job.run_dir / "artifacts",
        [
            ("raw/foundry_mpnn/**/*.cif", "cif"),
            ("raw/foundry_mpnn/**/*.json", "sequence_design_input"),
            ("normalized_candidates/*.jsonl", "normalized_candidates"),
            ("normalized_candidates/*.json", "campaign_result"),
        ],
    )
    finish_job(
        job.run_dir,
        rc == 0 and bool(candidates),
        {
            "outputs": {"artifacts": artifacts, "candidates": candidates},
            "metrics": {"return_code": rc, "artifact_count": len(artifacts), "candidate_count": len(candidates)},
            "downstream_artifacts": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
            },
        },
    )
    return job.run_dir
