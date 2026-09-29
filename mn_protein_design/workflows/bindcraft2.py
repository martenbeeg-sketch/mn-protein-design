from __future__ import annotations

import csv
import math
import shlex
import subprocess
from pathlib import Path
from typing import Any

from mn_protein_design.core.artifacts import Artifact, artifact_path
from mn_protein_design.core.candidates import (
    STAGE_COMPLEX_REFOLDING,
    write_candidates,
)
from mn_protein_design.core.gpu import docker_gpu_args, normalize_gpu_device
from mn_protein_design.core.jobs import (
    create_job,
    finish_job,
    read_json,
    update_status,
    write_json,
)
from mn_protein_design.core.structures import filter_pdb_text
from mn_protein_design.core.local_worker import spawn_worker_for_run
from mn_protein_design.core.portable_paths import portable_path
from mn_protein_design.core.scheduler import apply_docker_cpu_limit, configured_cpu_slots
from mn_protein_design.runtime import reference_root
from mn_protein_design.workflows.design import (
    _infer_generated_chain_roles,
    _pdb_sequences_by_chain,
    _sequences_by_chain,
    normalize_hotspots,
    parse_binder_lengths,
)


DESIGN_GROUP = "design"
BINDCRAFT2_TOOL = "bindcraft2"
BINDCRAFT2_JOB_KIND = "bindcraft2_design"
BINDCRAFT2_IMAGE = "mn-bindcraft2:4a56313-cu13"
BINDCRAFT2_SOURCE_COMMIT = "4a56313e96afc13c443d88427281cf2169a1c9ca"
BINDCRAFT2_SOURCE_URL = "https://github.com/PacesaLab/BindCraft2.git"
BINDCRAFT2_REFERENCE_SUBDIR = "alphafold_models"
BINDCRAFT2_MODELS = tuple(
    [f"model_{index}_multimer_v3" for index in range(1, 6)] + ["model_1_ptm", "model_2_ptm"]
)
ALPHAFOLD_MINIMUM_BYTES = 100 * 1024 * 1024
BC2_BINDER_FORMATS = (
    "binder",
    "large_binder",
    "peptide",
    "cyclic_peptide",
    "homo_oligomer",
    "multidomain",
    "VHH",
    "ARP",
    "scFv",
    "Fab",
)
BC2_CONFORMATIONAL_OBJECTIVES = ("induced_fit", "fold_switch")
SUPPORTED_MODALITIES = BC2_BINDER_FORMATS + BC2_CONFORMATIONAL_OBJECTIVES
BC2_PROPERTY_PRESETS = {
    "bigbang": "Start gradient design from the coordinates on hand.",
    "disulfide_staple": "Allow cysteine and require a geometrically detected disulfide.",
    "forced_targeting": "Concentrate the paratope on declared hotspots (requires hotspots).",
    "humanize": "Add human sequence preferences and an MHC-II anchor-score objective.",
    "initial_guess": "Use the trajectory pose as the starting guess for validation.",
    "mixed_topology": "Encourage mixed helix and beta-sheet topology.",
    "protease_stable": "Penalize protease motifs, exposed loops, and exposed termini.",
    "termini_accessible": "Orient both chain ends away from the target.",
    "termini_together": "Bring the N and C termini close together.",
}
BC2_SCAFFOLD_FORMATS = frozenset({"VHH", "scFv", "Fab", "ARP"})
BC2_PRESET_LENGTHS = {
    "binder": (60, 180),
    "large_binder": (250, 600),
    "peptide": (12, 25),
    "cyclic_peptide": (6, 16),
    "homo_oligomer": (40, 120),
    "multidomain": (120, 300),
}
WORKERS_PER_GPU_OPTIONS: tuple[str | int, ...] = ("auto", 1, 2, 3, 4)


def bindcraft2_cpu_options() -> tuple[list[int], int]:
    capacity = configured_cpu_slots()
    options = sorted({1, min(2, capacity), min(4, capacity), min(8, capacity)})
    return options, min(4, capacity)


def bindcraft2_reference_directory(root: Path | None = None) -> Path:
    return (root or reference_root()) / BINDCRAFT2_REFERENCE_SUBDIR


def missing_bindcraft2_parameters(
    root: Path | None = None,
    *,
    minimum_bytes: int = ALPHAFOLD_MINIMUM_BYTES,
) -> list[str]:
    """Return the required BC2 AF2 checkpoint files that are absent or incomplete."""
    directory = bindcraft2_reference_directory(root)
    missing: list[str] = []
    for model in BINDCRAFT2_MODELS:
        possible = (directory / f"params_{model}.npz", directory / "params" / f"params_{model}.npz")
        if not any(path.is_file() and path.stat().st_size >= minimum_bytes for path in possible):
            missing.append(f"params_{model}.npz")
    return missing


def bindcraft2_image_available(image: str = BINDCRAFT2_IMAGE) -> bool:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _validate_hotspots(hotspots: str, target_chains: list[str]) -> str:
    normalized = normalize_hotspots(hotspots) if hotspots else ""
    allowed = {str(chain).strip().upper() for chain in target_chains}
    invalid = [token for token in normalized.split(",") if token and token[0] not in allowed]
    if invalid:
        raise ValueError(f"Hotspots must belong to selected target chains: {', '.join(invalid)}")
    return normalized


def normalize_bindcraft2_modalities(modality: str | list[str] | tuple[str, ...]) -> list[str]:
    """Normalize a BC2 binder format and its optional conformational objective."""
    names = [str(value).strip() for value in ([modality] if isinstance(modality, str) else modality) if str(value).strip()]
    if not names:
        names = ["binder"]
    if len(set(names)) != len(names):
        raise ValueError("BindCraft 2 modalities cannot be repeated.")
    unknown = [name for name in names if name not in SUPPORTED_MODALITIES]
    if unknown:
        raise ValueError(f"Unknown BindCraft 2 modality: {', '.join(unknown)}")
    formats = [name for name in names if name in BC2_BINDER_FORMATS]
    objectives = [name for name in names if name in BC2_CONFORMATIONAL_OBJECTIVES]
    if len(formats) > 1:
        raise ValueError("Choose one BindCraft 2 binder format.")
    if len(objectives) > 1:
        raise ValueError("Choose at most one conformational objective.")
    if objectives and formats and formats[0] != "binder":
        raise ValueError("BindCraft 2 induced_fit and fold_switch objectives currently require the binder format.")
    if objectives and not formats:
        formats = ["binder"]
    return formats + objectives


def build_bindcraft2_settings(
    *,
    target_chains: list[str],
    binder_length: str | None,
    hotspots: str = "",
    campaign_name: str = "",
    modality: str | list[str] | tuple[str, ...] = "binder",
    design_properties: list[str] | tuple[str, ...] = (),
    number_of_final_designs: int = 1,
    max_trajectories: int = 1,
    campaign_seed: int = 0,
    workers_per_gpu: str | int = "auto",
) -> dict[str, Any]:
    chains = [str(chain).strip() for chain in target_chains if str(chain).strip()]
    if not chains:
        raise ValueError("At least one target chain is required.")
    lengths = parse_binder_lengths(binder_length) if str(binder_length or "").strip() else None
    modalities = normalize_bindcraft2_modalities(modality)
    properties = list(dict.fromkeys(str(value).strip() for value in design_properties if str(value).strip()))
    unknown_properties = [name for name in properties if name not in BC2_PROPERTY_PRESETS]
    if unknown_properties:
        raise ValueError(f"Unknown BindCraft 2 design property: {', '.join(unknown_properties)}")
    if "forced_targeting" in properties and not hotspots:
        raise ValueError("BindCraft 2 forced_targeting requires at least one target hotspot.")
    if modalities[0] in BC2_SCAFFOLD_FORMATS and "mixed_topology" in properties:
        raise ValueError("BindCraft 2 mixed_topology cannot be combined with a fixed scaffold modality.")
    if int(number_of_final_designs) < 1:
        raise ValueError("Desired final designs must be at least 1.")
    if int(max_trajectories) < 1:
        raise ValueError("Design attempts must be at least 1.")
    if int(campaign_seed) < 0:
        raise ValueError("Campaign seed must be zero or greater.")
    if workers_per_gpu != "auto":
        try:
            worker_count = int(workers_per_gpu)
        except (TypeError, ValueError) as exc:
            raise ValueError("Workers per GPU must be auto or a positive integer.") from exc
        if worker_count < 1:
            raise ValueError("Workers per GPU must be auto or a positive integer.")
        workers_per_gpu = worker_count

    hotspot_text = _validate_hotspots(hotspots, chains)
    target = {"name": "app_target", "target_path": "target.pdb", "chains": ",".join(chains)}
    if hotspot_text:
        target["hotspots"] = hotspot_text
    name = str(campaign_name or "bindcraft2").strip()
    settings: dict[str, Any] = {
        "targets": [target],
        "modality": modalities[0] if len(modalities) == 1 else modalities,
        "campaign_name": name,
        "number_of_final_designs": int(number_of_final_designs),
        "max_trajectories": int(max_trajectories),
        "campaign_seed": int(campaign_seed),
        "workers_per_gpu": workers_per_gpu,
        "project_folder": "output",
        "resume": True,
        "trajectory_only": False,
        "save_design_trajectory": True,
        "save_design_animations": False,
        "save_design_frames": False,
    }
    if lengths is not None:
        settings["binder_lengths"] = lengths
    settings.update({name: True for name in properties})
    return settings


def _portable_command(command: list[str], reference_dir: Path) -> list[str]:
    source = str(reference_dir.expanduser().resolve())
    replacement = portable_path(reference_dir)
    return [token.replace(source, replacement) for token in command]


def build_bindcraft2_docker_command(
    *,
    run_dir: Path,
    reference_dir: Path,
    gpu_device: object = "0",
    image: str = BINDCRAFT2_IMAGE,
) -> list[str]:
    gpu = normalize_gpu_device(gpu_device)
    reference = reference_dir.expanduser().resolve()
    workdir = "/work/artifacts/raw/bindcraft2"
    command = [
        "docker",
        "run",
        "--rm",
        *docker_gpu_args(gpu),
        "--shm-size=32G",
        "-e",
        f"BINDCRAFT_AF2_PARAMS=/ref/{BINDCRAFT2_REFERENCE_SUBDIR}",
        "-v",
        f"{run_dir.expanduser().resolve()}:/work:rw",
        "-v",
        f"{reference}:/ref:ro",
        "-w",
        workdir,
        image,
        "design",
        "settings.json",
    ]
    return apply_docker_cpu_limit(command, run_dir)


def _prepare_bindcraft2_run(
    *,
    target_pdb: Path,
    target_chains: list[str],
    binder_length: str | None,
    hotspots: str = "",
    campaign_name: str = "",
    modality: str | list[str] | tuple[str, ...] = "binder",
    design_properties: list[str] | tuple[str, ...] = (),
    number_of_final_designs: int = 1,
    max_trajectories: int = 1,
    campaign_seed: int = 0,
    workers_per_gpu: str | int = "auto",
    cpu_cores: int = 4,
    gpu_device: object = "0",
    dispatch_worker: bool,
    parent_run_dir: Path | None = None,
) -> Path:
    target_pdb = Path(target_pdb).expanduser().resolve()
    if not target_pdb.is_file():
        raise FileNotFoundError(f"Target structure not found: {target_pdb}")
    if int(cpu_cores) < 1:
        raise ValueError("CPU cores must be at least 1.")
    settings = build_bindcraft2_settings(
        target_chains=target_chains,
        binder_length=binder_length,
        hotspots=hotspots,
        campaign_name=campaign_name,
        modality=modality,
        design_properties=design_properties,
        number_of_final_designs=number_of_final_designs,
        max_trajectories=max_trajectories,
        campaign_seed=campaign_seed,
        workers_per_gpu=workers_per_gpu,
    )
    available_target_chains = _pdb_sequences_by_chain(target_pdb)
    missing_chains = [chain for chain in settings["targets"][0]["chains"].split(",") if chain not in available_target_chains]
    if missing_chains:
        raise ValueError(f"Target structure does not contain selected chain(s): {', '.join(missing_chains)}")
    parameters = missing_bindcraft2_parameters()
    if parameters:
        raise FileNotFoundError(
            "BindCraft 2 AlphaFold parameters are missing or incomplete under "
            f"{bindcraft2_reference_directory()}: {', '.join(parameters)}"
        )
    if not bindcraft2_image_available():
        raise RuntimeError(
            f"Docker image {BINDCRAFT2_IMAGE} is missing. Build it with "
            "bash containers/bindcraft2/build.sh."
        )

    selected_gpu = normalize_gpu_device(gpu_device)
    params = {
        "target_chains": settings["targets"][0]["chains"].split(","),
        "binder_length": binder_length,
        "hotspots": settings["targets"][0].get("hotspots", ""),
        "campaign_name": settings["campaign_name"],
        "modality": settings["modality"],
        "design_properties": [name for name in BC2_PROPERTY_PRESETS if settings.get(name)],
        "number_of_final_designs": int(number_of_final_designs),
        "max_trajectories": int(max_trajectories),
        "campaign_seed": int(campaign_seed),
        "workers_per_gpu": settings["workers_per_gpu"],
        "cpu_cores": int(cpu_cores),
        "gpu_device": selected_gpu,
        "docker_image": BINDCRAFT2_IMAGE,
        "bindcraft2_source_commit": BINDCRAFT2_SOURCE_COMMIT,
        "bindcraft2_source_url": BINDCRAFT2_SOURCE_URL,
        "reference_key": f"reference:///{BINDCRAFT2_REFERENCE_SUBDIR}",
    }
    job = create_job(
        DESIGN_GROUP,
        "bindcraft2_design",
        BINDCRAFT2_TOOL,
        inputs={"target_pdb": "artifacts/raw/bindcraft2/target.pdb", "target_chains": params["target_chains"]},
        params=params,
    )
    staged_target = artifact_path(job.run_dir, "raw", "bindcraft2", "target.pdb")
    raw_dir = staged_target.parent
    target_text = target_pdb.read_text(errors="ignore")
    staged_target.write_text(filter_pdb_text(target_text, keep_chains=set(params["target_chains"])))
    settings_path = raw_dir / "settings.json"
    write_json(settings_path, settings)
    if dispatch_worker:
        write_json(job.run_dir / "worker_request.json", {"kind": BINDCRAFT2_JOB_KIND, "kwargs": {}, "path_kwargs": []})
        write_json(
            job.run_dir / "command.json",
            {
                "mode": "local_worker",
                "command": ["python", "-m", "mn_protein_design.core.local_worker", "--run-dir", "runs:///design/" + job.run_dir.name],
            },
        )
    else:
        parent_metadata = read_json(Path(parent_run_dir) / "metadata.json") if parent_run_dir else {}
        parent_allocation = parent_metadata.get("resource_allocation") if isinstance(parent_metadata.get("resource_allocation"), dict) else {}
        child_metadata = read_json(job.run_dir / "metadata.json")
        child_metadata["resource_allocation"] = {
            **parent_allocation,
            "cpu_cores": int(parent_allocation.get("cpu_cores") or cpu_cores),
            "gpu_device": selected_gpu,
            "scheduler": str(parent_allocation.get("scheduler") or "design-campaign-parent"),
        }
        write_json(job.run_dir / "metadata.json", child_metadata)
        write_json(
            job.run_dir / "command.json",
            {
                "mode": "embedded_campaign",
                "parent_run": portable_path(Path(parent_run_dir)) if parent_run_dir else "",
            },
        )
    if campaign_name.strip():
        update_status(job.run_dir, "queued", campaign_name=campaign_name.strip())
    if dispatch_worker:
        spawn_worker_for_run(job.run_dir)
    return job.run_dir


def enqueue_bindcraft2_design(
    *,
    target_pdb: Path,
    target_chains: list[str],
    binder_length: str | None,
    hotspots: str = "",
    campaign_name: str = "",
    modality: str | list[str] | tuple[str, ...] = "binder",
    design_properties: list[str] | tuple[str, ...] = (),
    number_of_final_designs: int = 1,
    max_trajectories: int = 1,
    campaign_seed: int = 0,
    workers_per_gpu: str | int = "auto",
    cpu_cores: int = 4,
    gpu_device: object = "0",
) -> Path:
    return _prepare_bindcraft2_run(
        target_pdb=target_pdb,
        target_chains=target_chains,
        binder_length=binder_length,
        hotspots=hotspots,
        campaign_name=campaign_name,
        modality=modality,
        design_properties=design_properties,
        number_of_final_designs=number_of_final_designs,
        max_trajectories=max_trajectories,
        campaign_seed=campaign_seed,
        workers_per_gpu=workers_per_gpu,
        cpu_cores=cpu_cores,
        gpu_device=gpu_device,
        dispatch_worker=True,
    )


def run_bindcraft2_campaign(
    *,
    parent_run_dir: Path,
    target_pdb: Path,
    target_chains: list[str],
    binder_length: str | None,
    hotspots: str = "",
    campaign_name: str = "",
    modality: str | list[str] | tuple[str, ...] = "binder",
    design_properties: list[str] | tuple[str, ...] = (),
    number_of_final_designs: int = 1,
    max_trajectories: int = 1,
    campaign_seed: int = 0,
    workers_per_gpu: str | int = "auto",
    cpu_cores: int = 1,
    gpu_device: object = "0",
) -> Path:
    """Run BC2 as an inline child of an already scheduled vanilla campaign."""
    child_run = _prepare_bindcraft2_run(
        target_pdb=target_pdb,
        target_chains=target_chains,
        binder_length=binder_length,
        hotspots=hotspots,
        campaign_name=campaign_name,
        modality=modality,
        design_properties=design_properties,
        number_of_final_designs=number_of_final_designs,
        max_trajectories=max_trajectories,
        campaign_seed=campaign_seed,
        workers_per_gpu=workers_per_gpu,
        cpu_cores=cpu_cores,
        gpu_device=gpu_device,
        dispatch_worker=False,
        parent_run_dir=parent_run_dir,
    )
    run_bindcraft2_job(child_run)
    return child_run


def _coerce_metric(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        return value
    if not math.isfinite(number):
        return value
    return int(number) if number.is_integer() else number


def normalize_bindcraft2_candidates(run_dir: Path, settings: dict[str, Any]) -> list[dict[str, Any]]:
    raw_dir = run_dir / "artifacts" / "raw" / "bindcraft2"
    target_artifact = raw_dir / "target.pdb"
    output_dir = raw_dir / "output"
    ranked_csv = output_dir / "3_Ranked" / "!_Ranked.csv"
    if not ranked_csv.is_file():
        return write_candidates(run_dir, BINDCRAFT2_TOOL, [])

    input_target_chains = [str(chain) for chain in settings.get("target_chains") or []]
    requested_hotspots = [token for token in str(settings.get("hotspots") or "").split(",") if token]
    requested_length = str(settings.get("binder_length") or "")
    candidates: list[dict[str, Any]] = []
    with ranked_csv.open(newline="", encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            design_name = Path(str(row.get("design") or "")).name
            if not design_name:
                continue
            structures = sorted(
                path
                for path in (output_dir / "3_Ranked").glob(f"{design_name}*.cif")
                if not path.stem.endswith("_monomer")
            )
            if not structures:
                continue
            complex_path = next((path for path in structures if path.stem == design_name), structures[0])
            binder_chains, inferred_target_chains, chain_inference = _infer_generated_chain_roles(
                target_artifact,
                complex_path,
                input_target_chains,
            )
            sequences = _sequences_by_chain(complex_path)
            sequence = str(row.get("Binder_Sequence") or "").strip() or "".join(
                sequences.get(chain, "") for chain in binder_chains
            )
            metrics = {
                key: value
                for key, raw_value in row.items()
                if key not in {"design", "Binder_Sequence"} and (value := _coerce_metric(str(raw_value or ""))) is not None
            }
            rank = _coerce_metric(str(row.get("rank") or index))
            metrics.update(
                {
                    "bindcraft2_rank": rank,
                    "native_pass_filters": True,
                    "result_kind": "native_pipeline",
                    "target_chain_inference": chain_inference,
                }
            )
            # Store stable app-managed URIs here; the candidate writer and result.json both retain
            # them without recording this machine's active run-root path.
            complex_ref = portable_path(complex_path)
            target_ref = portable_path(target_artifact)
            candidates.append(
                {
                    "candidate_id": f"bindcraft2_{index:05d}",
                    "stage": STAGE_COMPLEX_REFOLDING,
                    "source_tool": BINDCRAFT2_TOOL,
                    "target_pdb": target_ref,
                    "complex_pdb": complex_ref,
                    "binder_pdb": None,
                    "binder_sequence": sequence or None,
                    "target_chains": inferred_target_chains or input_target_chains,
                    "binder_chains": binder_chains,
                    "hotspots": requested_hotspots,
                    "binder_length": str(row.get("length") or requested_length),
                    "metrics": metrics,
                    "raw_metadata": {
                        "bindcraft2_design": design_name,
                        "bindcraft2_ranked_csv": ranked_csv.relative_to(run_dir).as_posix(),
                        "bindcraft2_output_structure": complex_ref,
                        "bindcraft2_row": row,
                        "input_target_chains": input_target_chains,
                        "target_chain_inference": chain_inference,
                    },
                }
            )
    return write_candidates(run_dir, BINDCRAFT2_TOOL, candidates)


def _bindcraft2_artifacts(run_dir: Path) -> list[dict[str, Any]]:
    raw = run_dir / "artifacts" / "raw" / "bindcraft2"
    candidates = run_dir / "artifacts" / "normalized_candidates"
    selected: list[tuple[Path, str]] = []
    patterns = (
        (raw / "output" / "3_Ranked" / "!_Ranked.csv", "bindcraft2_ranked_table"),
        (raw / "output" / "2_Refolded" / "!_Refolded.csv", "bindcraft2_refolded_table"),
        (raw / "output" / "1_Trajectories" / "!_Trajectories.csv", "bindcraft2_trajectory_table"),
        (raw / "output" / "summary.csv", "bindcraft2_summary"),
        (raw / "output" / "campaign_metadata.json", "bindcraft2_campaign_metadata"),
        (raw / "settings.json", "bindcraft2_input_settings"),
        (raw / "target.pdb", "bindcraft2_input_target"),
        (candidates / "candidates.jsonl", "normalized_candidates"),
        (candidates / "campaign_result.json", "campaign_result"),
    )
    for path, artifact_type in patterns:
        if path.is_file():
            selected.append((path, artifact_type))
    metadata_files = sorted((raw / "output").glob("campaign_metadata*.json"))
    selected.extend((path, "bindcraft2_campaign_metadata") for path in metadata_files if path.is_file())
    ranked_dir = raw / "output" / "3_Ranked"
    if ranked_dir.is_dir():
        selected.extend((path, "designed_complex_mmcif") for path in sorted(ranked_dir.glob("*.cif")) if not path.stem.endswith("_monomer"))
    return [Artifact(path.stem, path, artifact_type).to_json(run_dir) for path, artifact_type in selected]


def run_bindcraft2_job(run_dir: Path) -> Path:
    run_dir = Path(run_dir).expanduser().resolve()
    payload = read_json(run_dir / "input.json")
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    reference_dir = reference_root()
    missing = missing_bindcraft2_parameters(reference_dir)
    if missing:
        raise FileNotFoundError(
            f"BindCraft 2 AlphaFold parameters are missing under {bindcraft2_reference_directory(reference_dir)}: "
            + ", ".join(missing)
        )
    image = str(params.get("docker_image") or BINDCRAFT2_IMAGE)
    command = build_bindcraft2_docker_command(
        run_dir=run_dir,
        reference_dir=reference_dir,
        gpu_device=params.get("gpu_device", "0"),
        image=image,
    )
    command_for_record = _portable_command(command, reference_dir)
    write_json(
        run_dir / "command.json",
        {"mode": "docker", "steps": [{"name": BINDCRAFT2_TOOL, "command": command_for_record}]},
    )
    update_status(run_dir, "running")
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        stdout.write(f"$ {shlex.join(command_for_record)}\n")
        stdout.flush()
        process = subprocess.run(command, cwd=run_dir, stdout=stdout, stderr=stderr, check=False)

    candidates = normalize_bindcraft2_candidates(run_dir, params)
    artifacts = _bindcraft2_artifacts(run_dir)
    finish_job(
        run_dir,
        process.returncode == 0,
        {
            "outputs": {"artifacts": artifacts, "candidates": candidates},
            "metrics": {
                "return_code": int(process.returncode),
                "candidate_count": len(candidates),
                "artifact_count": len(artifacts),
                "accepted_design_count": len(candidates),
                "bindcraft2_image": image,
                "bindcraft2_source_commit": str(params.get("bindcraft2_source_commit") or BINDCRAFT2_SOURCE_COMMIT),
                "bindcraft2_source_url": str(params.get("bindcraft2_source_url") or BINDCRAFT2_SOURCE_URL),
                "bindcraft2_workers_per_gpu": params.get("workers_per_gpu", "auto"),
                "bindcraft2_cpu_cores": params.get("cpu_cores", 4),
            },
            "downstream_artifacts": {
                "target_pdb": "artifacts/raw/bindcraft2/target.pdb",
                "settings_json": "artifacts/raw/bindcraft2/settings.json",
                "campaign_output": "artifacts/raw/bindcraft2/output",
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
            },
        },
    )
    return run_dir
