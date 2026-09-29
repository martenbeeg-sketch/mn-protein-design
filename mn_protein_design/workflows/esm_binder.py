from __future__ import annotations

import json
import random
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from mn_protein_design.core.candidates import STAGE_COMPLEX_REFOLDING, write_candidates
from mn_protein_design.core.gpu import docker_gpu_args, normalize_gpu_device
from mn_protein_design.core.jobs import create_job, finish_job, update_status, write_json
from mn_protein_design.core.scheduler import apply_docker_cpu_limit
from mn_protein_design.core.structures import filter_pdb_text
from mn_protein_design.runtime import reference_root
from mn_protein_design.workflows.esmfold2_runtime import run_esmfold2_batch


DESIGN_GROUP = "design"
BIOHUB_ESM_ROOT = reference_root() / "biohub-esm"
ESMFOLD2_MODEL_DIR = BIOHUB_ESM_ROOT / "ESMFold2"
ESMC_MODEL_DIR = BIOHUB_ESM_ROOT / "ESMC-6B"
ESMFOLD2_BINDER_MODEL_ROOT = BIOHUB_ESM_ROOT / "binder-design"
ESMFOLD2_BINDER_IMAGE = "mn-biohub-esm:3.4.1-cu128"
DEFAULT_BINDER_MODEL = "ESMFold2-Experimental-Fast"
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
SOLUBLE_AA_WEIGHTS = {
    "A": 0.07,
    "C": 0.01,
    "D": 0.08,
    "E": 0.10,
    "F": 0.03,
    "G": 0.06,
    "H": 0.02,
    "I": 0.04,
    "K": 0.10,
    "L": 0.08,
    "M": 0.02,
    "N": 0.05,
    "P": 0.04,
    "Q": 0.07,
    "R": 0.07,
    "S": 0.07,
    "T": 0.06,
    "V": 0.05,
    "W": 0.01,
    "Y": 0.04,
}


def _safe_id(value: object, fallback: str = "esmfold2") -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip()).strip("_")
    return text or fallback


def _parse_binder_lengths(text: str) -> tuple[int, int]:
    values = [int(token) for token in re.findall(r"\d+", text)]
    if not values:
        raise ValueError("Binder length is required.")
    if len(values) == 1:
        return values[0], values[0]
    low, high = values[0], values[1]
    if low < 1 or high < 1 or high < low:
        raise ValueError("Binder length must be a positive value or min-max range.")
    return low, high


def _pdb_sequence_records(path: Path) -> dict[str, list[tuple[int, str, str]]]:
    aa3_to_1 = {
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
    records: dict[str, list[tuple[int, str, str]]] = {}
    seen: set[tuple[str, str, str]] = set()
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM  "):
            continue
        chain = line[21].strip() or "_"
        residue = line[22:26].strip()
        insertion = line[26].strip()
        key = (chain, residue, insertion)
        if key in seen:
            continue
        seen.add(key)
        try:
            residue_number = int(residue)
        except ValueError:
            continue
        aa = aa3_to_1.get(line[17:20].strip().upper(), "X")
        records.setdefault(chain, []).append((residue_number, insertion, aa))
    return {
        chain: sorted(chain_records, key=lambda item: (item[0], item[1]))
        for chain, chain_records in records.items()
    }


def _target_sequences(path: Path, target_chains: list[str]) -> tuple[dict[str, str], dict[str, dict[int, int]]]:
    records = _pdb_sequence_records(path)
    chains = [chain for chain in target_chains if chain in records] or list(records)
    sequences: dict[str, str] = {}
    residue_maps: dict[str, dict[int, int]] = {}
    for chain in chains:
        chain_records = records.get(chain, [])
        sequences[chain] = "".join(record[2] for record in chain_records)
        residue_maps[chain] = {
            residue_number: index
            for index, (residue_number, _insertion, _aa) in enumerate(chain_records, start=1)
        }
    if not sequences:
        raise ValueError(f"No protein sequence could be read from {path}.")
    return sequences, residue_maps


def _parse_hotspots(hotspots: str | list[str] | None) -> list[tuple[str, int]]:
    if not hotspots:
        return []
    if isinstance(hotspots, str):
        tokens = [token for token in re.split(r"[\s,;]+", hotspots) if token]
    else:
        tokens = [str(token) for token in hotspots if str(token)]
    parsed: list[tuple[str, int]] = []
    for token in tokens:
        match = re.fullmatch(r"([A-Za-z])(\d+)", token.strip().replace(":", ""))
        if match:
            parsed.append((match.group(1).upper(), int(match.group(2))))
    return parsed


def _mapped_hotspots(hotspots: str | list[str] | None, residue_maps: dict[str, dict[int, int]]) -> list[tuple[str, int]]:
    mapped: list[tuple[str, int]] = []
    for chain, residue_number in _parse_hotspots(hotspots):
        mapped_residue = residue_maps.get(chain, {}).get(residue_number)
        if mapped_residue is not None:
            mapped.append((chain, mapped_residue))
    return mapped


def _parse_sequence_text(text: str) -> list[str]:
    sequences: list[str] = []
    current: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
                current = []
            continue
        tokens = re.split(r"[\s,;]+", line)
        if len(tokens) > 1 and all(re.fullmatch(r"[A-Za-z]+", token) for token in tokens if token):
            if current:
                sequences.append("".join(current))
                current = []
            sequences.extend(tokens)
            continue
        current.append(line)
    if current:
        sequences.append("".join(current))
    cleaned = []
    for sequence in sequences:
        sequence = re.sub(r"[^A-Za-z]", "", sequence).upper()
        if sequence and all(aa in AA_ALPHABET for aa in sequence):
            cleaned.append(sequence)
    return cleaned


def _sample_binder_sequence(length: int, rng: random.Random) -> str:
    amino_acids = list(AA_ALPHABET)
    weights = [SOLUBLE_AA_WEIGHTS[aa] for aa in amino_acids]
    return "".join(rng.choices(amino_acids, weights=weights, k=length))


def _binder_sequences(
    sequence_text: str,
    binder_length: str,
    num_designs: int,
    seed: int,
) -> list[str]:
    pasted = _parse_sequence_text(sequence_text)
    if pasted:
        return pasted[:num_designs]
    min_length, max_length = _parse_binder_lengths(binder_length)
    rng = random.Random(seed)
    return [
        _sample_binder_sequence(rng.randint(min_length, max_length), rng)
        for _ in range(num_designs)
    ]


def _choose_binder_chain(target_chains: list[str]) -> str:
    for chain in ["B", "C", "D", "E", "F", "Z", "Y", "X"]:
        if chain not in set(target_chains):
            return chain
    return "Z"


def _chain_residue_atoms(complex_obj: Any) -> dict[str, dict[int, list[np.ndarray]]]:
    chain_lookup = dict(getattr(complex_obj.metadata, "chain_lookup", {}) or {})
    residues: dict[str, dict[int, list[np.ndarray]]] = {}
    counters: dict[str, int] = {}
    atom_positions = np.asarray(complex_obj.atom_positions, dtype=float)
    atom_names = list(complex_obj.atom_names)
    atom_elements = list(getattr(complex_obj, "atom_elements", []))
    for token_index, chain_numeric in enumerate(complex_obj.chain_id):
        chain = str(chain_lookup.get(int(chain_numeric), chain_numeric))
        counters[chain] = counters.get(chain, 0) + 1
        residue_number = counters[chain]
        start, end = [int(value) for value in complex_obj.token_to_atoms[token_index]]
        for atom_index in range(start, end):
            element = str(atom_elements[atom_index]).upper() if atom_index < len(atom_elements) else ""
            atom_name = str(atom_names[atom_index]).strip()
            if element == "H" or atom_name.startswith("H"):
                continue
            residues.setdefault(chain, {}).setdefault(residue_number, []).append(atom_positions[atom_index])
    return residues


def _min_distance(coords_a: list[np.ndarray], coords_b: list[np.ndarray]) -> float | None:
    if not coords_a or not coords_b:
        return None
    coords_b_array = np.asarray(coords_b, dtype=float)
    best = min(float(np.sqrt(np.sum((coords_b_array - coord_a) ** 2, axis=1)).min()) for coord_a in coords_a)
    return best


def _hotspot_metrics_from_complex(
    complex_obj: Any,
    *,
    binder_chain: str,
    target_chains: list[str],
    mapped_hotspots: list[tuple[str, int]],
    contact_cutoff: float,
) -> dict[str, Any]:
    residues = _chain_residue_atoms(complex_obj)
    binder_residues = residues.get(binder_chain, {})
    target_residues = {
        chain: chain_residues
        for chain, chain_residues in residues.items()
        if chain in set(target_chains)
    }
    binder_coords = [coord for residue_atoms in binder_residues.values() for coord in residue_atoms]
    binder_ca = []
    # MolecularComplex does not preserve CA grouping directly here; use all-atom
    # contact metrics and leave CA geometry to downstream validation if needed.
    metrics: dict[str, Any] = {
        "hotspot_count": len(mapped_hotspots),
        "hotspots_contacted": 0,
        "hotspot_contact_fraction": None,
        "min_binder_to_hotspot_distance": None,
        "mean_binder_to_hotspot_distance": None,
        "binder_interface_contacts": 0,
        "hotspot_contact_cutoff": contact_cutoff,
        "binder_ca_count": len(binder_ca),
    }
    if not binder_residues or not target_residues:
        metrics["hotspot_metrics_error"] = "binder or target chain missing"
        return metrics

    for binder_atoms in binder_residues.values():
        for chain_residues in target_residues.values():
            for target_atoms in chain_residues.values():
                distance = _min_distance(binder_atoms, target_atoms)
                if distance is not None and distance <= contact_cutoff:
                    metrics["binder_interface_contacts"] += 1

    if not mapped_hotspots:
        return metrics

    contacted: set[tuple[str, int]] = set()
    distances: list[float] = []
    for hotspot in mapped_hotspots:
        chain, residue_number = hotspot
        hotspot_atoms = target_residues.get(chain, {}).get(residue_number, [])
        distance = _min_distance(binder_coords, hotspot_atoms)
        if distance is None:
            continue
        distances.append(distance)
        if distance <= contact_cutoff:
            contacted.add(hotspot)
    metrics["hotspots_contacted"] = len(contacted)
    metrics["hotspot_contact_fraction"] = len(contacted) / len(mapped_hotspots) if mapped_hotspots else None
    metrics["min_binder_to_hotspot_distance"] = min(distances) if distances else None
    metrics["mean_binder_to_hotspot_distance"] = sum(distances) / len(distances) if distances else None
    if not distances:
        metrics["hotspot_metrics_error"] = "hotspot residues were not found in the folded target chain"
    return metrics


def _mean_plddt(result: Any) -> float | None:
    plddt = getattr(result, "plddt", None)
    if plddt is None:
        return None
    try:
        return float(plddt.float().mean().item())
    except Exception:
        return float(np.asarray(plddt, dtype=float).mean())


def _ranking_score(metrics: dict[str, Any]) -> float:
    iptm = float(metrics.get("iptm") or 0.0)
    plddt = float(metrics.get("plddt_mean") or 0.0) / 100.0
    hotspot_fraction = metrics.get("hotspot_contact_fraction")
    hotspot_bonus = float(hotspot_fraction) if hotspot_fraction is not None else 0.0
    interface_contacts = min(float(metrics.get("binder_interface_contacts") or 0.0) / 20.0, 1.0)
    return (0.55 * iptm) + (0.20 * plddt) + (0.20 * hotspot_bonus) + (0.05 * interface_contacts)


def run_esmfold2_binder_screening(
    *,
    target_pdb: Path,
    target_chains: list[str],
    hotspots: str = "",
    binder_length: str = "55-55",
    num_designs: int = 4,
    binder_sequences_text: str = "",
    campaign_name: str = "",
    num_loops: int = 3,
    num_sampling_steps: int = 32,
    seed: int = 0,
    device: str = "auto",
    contact_cutoff: float = 8.0,
    gpu_device: object | None = None,
) -> Path:
    """Run an experimental ESMFold2 target+binder screening loop.

    This is a pragmatic approximation of a binder-design workflow: it proposes
    simple binder sequences or accepts user-supplied sequences, folds each
    target+binder complex with ESMFold2, ranks the results, and writes the
    app's normalized candidate format. It is not Biohub's unreleased ESMFold2
    inversion protocol.
    """
    target_pdb = Path(target_pdb)
    target_sequences, residue_maps = _target_sequences(target_pdb, target_chains)
    folded_target_chains = list(target_sequences)
    binder_chain = _choose_binder_chain(folded_target_chains)
    mapped_hotspot_ids = _mapped_hotspots(hotspots, residue_maps)
    sequences = _binder_sequences(binder_sequences_text, binder_length, int(num_designs), int(seed))
    if not sequences:
        raise ValueError("No binder sequences were supplied or generated.")

    job = create_job(
        DESIGN_GROUP,
        "esmfold2_binder_screening",
        "esmfold2_binder",
        {"target_pdb": str(target_pdb), "target_chains": target_chains},
        {
            "campaign_name": campaign_name,
            "hotspots": hotspots,
            "binder_length": binder_length,
            "num_designs": num_designs,
            "num_loops": num_loops,
            "num_sampling_steps": num_sampling_steps,
            "seed": seed,
            "device": device,
            "gpu_device": gpu_device,
            "contact_cutoff": contact_cutoff,
            "binder_sequences_supplied": bool(binder_sequences_text.strip()),
        },
    )
    if campaign_name.strip():
        update_status(job.run_dir, "queued", campaign_name=campaign_name.strip())
    update_status(job.run_dir, "running")
    raw_dir = job.run_dir / "artifacts" / "raw" / "esmfold2_binder"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_artifact = raw_dir / "target.pdb"
    target_artifact.write_text(target_pdb.read_text(errors="ignore"))
    (raw_dir / "binder_sequences.fasta").write_text(
        "".join(f">esm_binder_{index:05d}\n{sequence}\n" for index, sequence in enumerate(sequences, start=1))
    )
    write_json(
        raw_dir / "input_mapping.json",
        {
            "target_chains": folded_target_chains,
            "binder_chain": binder_chain,
            "requested_hotspots": [f"{chain}{residue}" for chain, residue in _parse_hotspots(hotspots)],
            "mapped_hotspots": [f"{chain}{residue}" for chain, residue in mapped_hotspot_ids],
            "residue_maps": residue_maps,
        },
    )

    try:
        requests = [
            {
                "request_id": f"esmfold2_{index:05d}",
                "sequences": [
                    *[
                        {"id": chain, "sequence": sequence}
                        for chain, sequence in target_sequences.items()
                    ],
                    {"id": binder_chain, "sequence": binder_sequence},
                ],
                "num_loops": int(num_loops),
                "num_sampling_steps": int(num_sampling_steps),
                "seed": int(seed) + index - 1,
            }
            for index, binder_sequence in enumerate(sequences, start=1)
        ]
        predictions = run_esmfold2_batch(
            run_dir=job.run_dir,
            requests=requests,
            gpu_device=gpu_device,
            device=device,
            num_loops=int(num_loops),
            num_sampling_steps=int(num_sampling_steps),
            seed=int(seed),
        )
        candidates: list[dict[str, Any]] = []
        with (job.run_dir / "stdout.log").open("a") as stdout:
            stdout.write(
                "Experimental ESMFold2 binder screening. This is not the unreleased Biohub inversion protocol.\n"
            )
            stdout.write(f"Target chains: {','.join(folded_target_chains)} | binder chain: {binder_chain}\n")
            stdout.flush()
            for index, binder_sequence in enumerate(sequences, start=1):
                candidate_id = f"esmfold2_{index:05d}"
                stdout.write(f"Folding {candidate_id} length={len(binder_sequence)}\n")
                stdout.flush()
                prediction = predictions[candidate_id]
                result = prediction.result
                complex_path = prediction.complex_path
                metrics = {
                        "complex_refolding_backend": "esmfold2_container",
                    "result_kind": "experimental_screening",
                    "iptm": float(result.iptm) if result.iptm is not None else None,
                    "ptm": float(result.ptm) if result.ptm is not None else None,
                    "plddt_mean": _mean_plddt(result),
                    "binder_length": len(binder_sequence),
                    **_hotspot_metrics_from_complex(
                        result.complex,
                        binder_chain=binder_chain,
                        target_chains=folded_target_chains,
                        mapped_hotspots=mapped_hotspot_ids,
                        contact_cutoff=float(contact_cutoff),
                    ),
                }
                metrics["esmfold2_screening_score"] = _ranking_score(metrics)
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "source_tool": "esmfold2_binder",
                        "stage": STAGE_COMPLEX_REFOLDING,
                        "target_pdb": str(target_artifact.relative_to(job.run_dir)),
                        "complex_pdb": str(complex_path.relative_to(job.run_dir)),
                        "binder_pdb": None,
                        "binder_sequence": binder_sequence,
                        "target_chains": folded_target_chains,
                        "binder_chains": [binder_chain],
                        "hotspots": [token for token in re.split(r"[\s,;]+", hotspots) if token],
                        "binder_length": str(len(binder_sequence)),
                        "contig": None,
                        "metrics": metrics,
                        "raw_metadata": {
                            "result_kind": "experimental_screening",
                            "warning": "Not Biohub's gradient-guided ESMFold2 inversion protocol.",
                            "mapped_hotspots": [f"{chain}{residue}" for chain, residue in mapped_hotspot_ids],
                        },
                    }
                )
        candidates.sort(key=lambda candidate: candidate["metrics"].get("esmfold2_screening_score") or 0.0, reverse=True)
        for rank, candidate in enumerate(candidates, start=1):
            candidate["metrics"]["esmfold2_screening_rank"] = rank
        normalized = write_candidates(job.run_dir, "esmfold2_binder", candidates)
        finish_job(
            job.run_dir,
            True,
            {
                "outputs": {
                    "candidates": str(job.run_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl"),
                    "raw_output": str(raw_dir),
                },
                "metrics": {
                    "candidate_count": len(normalized),
                    "best_score": normalized[0]["metrics"].get("esmfold2_screening_score") if normalized else None,
                },
            },
        )
        return job.run_dir
    except Exception as exc:
        with (job.run_dir / "stderr.log").open("a") as stderr:
            stderr.write(f"{type(exc).__name__}: {exc}\n")
        finish_job(job.run_dir, False, {"metrics": {"error": str(exc)}})
        raise


def run_esmfold2_native_binder_design(
    *,
    target_pdb: Path,
    target_chain: str,
    hotspots: str = "",
    binder_length: int = 60,
    num_designs: int = 1,
    campaign_name: str = "",
    optimization_steps: int = 150,
    learning_rate: float = 0.1,
    seed: int = 0,
    model_name: str = DEFAULT_BINDER_MODEL,
    compile_model: bool = False,
    checkpoint_lm: bool = False,
    gpu_device: object = "0",
) -> Path:
    """Run Biohub's released gradient-guided ESMFold2 binder design protocol."""
    target_pdb = Path(target_pdb)
    target_sequences, _ = _target_sequences(target_pdb, [target_chain])
    if target_chain not in target_sequences:
        raise ValueError(f"Target chain {target_chain!r} was not found in {target_pdb}.")
    if binder_length < 1:
        raise ValueError("Binder length must be at least 1.")
    if num_designs < 1:
        raise ValueError("Design attempts must be at least 1.")
    if optimization_steps < 1:
        raise ValueError("Optimization steps must be at least 1.")
    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive.")

    model_dir = ESMFOLD2_BINDER_MODEL_ROOT / model_name
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Required ESMFold2 binder-design checkpoint is missing: {model_dir}. "
            "Run the Biohub ESM model setup command first."
        )

    params = {
        "target_chain": target_chain,
        "target_chains": [target_chain],
        "hotspots": hotspots,
        "binder_length": int(binder_length),
        "num_designs": int(num_designs),
        "campaign_name": campaign_name.strip(),
        "optimization_steps": int(optimization_steps),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "model_name": model_name,
        "compile_model": bool(compile_model),
        "checkpoint_lm": bool(checkpoint_lm),
        "gpu_device": normalize_gpu_device(gpu_device),
        "pipeline_mode": "biohub_esmfold2_gradient_binder_design",
        "memory_profile": "lean_shared_model",
    }
    job = create_job(
        DESIGN_GROUP,
        "design_campaign",
        "esmfold2_binder_design",
        {"target_pdb": str(target_pdb), "target_chains": [target_chain]},
        params,
    )
    if campaign_name.strip():
        update_status(job.run_dir, "queued", campaign_name=campaign_name.strip())

    raw_dir = job.run_dir / "artifacts" / "raw" / "esmfold2_binder_design"
    output_dir = raw_dir / "output"
    config_dir = raw_dir / "config"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    target_artifact = raw_dir / "target.pdb"
    target_artifact.write_text(
        filter_pdb_text(
            target_pdb.read_text(errors="ignore"),
            keep_chains={target_chain},
        )
    )
    config = {
        "target_sequence": target_sequences[target_chain],
        "binder_length": int(binder_length),
        "num_designs": int(num_designs),
        "optimization_steps": int(optimization_steps),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "model_name": model_name,
        "compile": bool(compile_model),
        "checkpoint_lm": bool(checkpoint_lm),
        "tutorial_path": "/opt/esm-binder-design/binder_design.py",
        "reference_root": "/ref/biohub-esm",
        "output_dir": "/work/artifacts/raw/esmfold2_binder_design/output",
    }
    config_path = config_dir / "binder_design.json"
    write_json(config_path, config)

    command = [
        "docker",
        "run",
        "--rm",
        *docker_gpu_args(gpu_device),
        "--shm-size=32G",
        "-v",
        f"{job.run_dir}:/work",
        "-v",
        f"{BIOHUB_ESM_ROOT}:/ref/biohub-esm:ro",
        "-v",
        "mn-protein-design_biohub-esm-hf-cache:/cache/huggingface",
        ESMFOLD2_BINDER_IMAGE,
        "biohub-esm-binder-design",
        "--config",
        "/work/artifacts/raw/esmfold2_binder_design/config/binder_design.json",
    ]
    update_status(job.run_dir, "running")
    command = apply_docker_cpu_limit(command, job.run_dir)
    write_json(job.run_dir / "command.json", {"mode": "docker", "steps": [{"name": "esmfold2_binder_design", "command": command}]})
    with (job.run_dir / "stdout.log").open("a") as stdout, (job.run_dir / "stderr.log").open("a") as stderr:
        proc = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    rc = int(proc.returncode)

    candidates: list[dict[str, Any]] = []
    metrics_path = output_dir / "design_metrics.json"
    if rc == 0 and metrics_path.exists():
        payload = json.loads(metrics_path.read_text())
        for design in payload.get("designs") or []:
            design_index = int(design.get("design_index") or len(candidates) + 1)
            critics = design.get("critics") or []
            critic = max(
                critics,
                key=lambda row: float(row.get("iptm") or row.get("distogram_iptm_proxy") or 0.0),
                default={},
            )
            complex_value = critic.get("complex_path")
            complex_path = None
            if complex_value:
                complex_path = output_dir / Path(str(complex_value)).name
            candidates.append(
                {
                    "candidate_id": f"esmfold2_design_{design_index:05d}",
                    "source_tool": "esmfold2_binder_design",
                    "stage": STAGE_COMPLEX_REFOLDING,
                    "target_pdb": str(target_artifact.relative_to(job.run_dir)),
                    "complex_pdb": str(complex_path.relative_to(job.run_dir)) if complex_path and complex_path.exists() else None,
                    "binder_pdb": None,
                    "binder_sequence": design.get("binder_sequence"),
                    "target_chains": ["A"],
                    "binder_chains": ["B"],
                    "hotspots": [token for token in re.split(r"[\s,;]+", hotspots) if token],
                    "binder_length": str(design.get("binder_length") or binder_length),
                    "contig": None,
                    "metrics": {
                        "result_kind": "native_pipeline",
                        "complex_refolding_backend": "biohub_esmfold2_gradient_design",
                        "final_loss": critic.get("final_loss"),
                        "iptm": critic.get("iptm"),
                        "distogram_iptm_proxy": critic.get("distogram_iptm_proxy"),
                        "native_design_rank": design_index,
                    },
                    "raw_metadata": {
                        "result_kind": "native_pipeline",
                        "model_name": model_name,
                        "memory_profile": payload.get("profile"),
                        "input_target_chain": target_chain,
                        "output_target_chain": "A",
                        "output_binder_chain": "B",
                        "critics": critics,
                    },
                }
            )
        write_candidates(job.run_dir, "esmfold2_binder_design", candidates)

    finish_job(
        job.run_dir,
        rc == 0 and bool(candidates),
        {
            "outputs": {
                "candidates": str(job.run_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl"),
                "raw_output": str(output_dir),
            },
            "metrics": {
                "return_code": rc,
                "candidate_count": len(candidates),
                "model_name": model_name,
            },
        },
    )
    return job.run_dir
