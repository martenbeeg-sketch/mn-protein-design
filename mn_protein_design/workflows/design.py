from __future__ import annotations

import glob
import csv
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from mn_protein_design.core.artifacts import Artifact, artifact_path
from mn_protein_design.core.candidates import (
    STAGE_GENERATION_BACKBONE,
    STAGE_GENERATION_BACKBONE_SEQUENCE,
    write_candidates,
)
from mn_protein_design.core.jobs import collect_jobs, create_job, finish_job, read_json, update_status, write_json
from mn_protein_design.core.manifests import load_manifest
from mn_protein_design.core.structures import filter_pdb_text, pdb_summary


DESIGN_GROUP = "design"


def prepared_design_targets() -> list[dict]:
    from mn_protein_design.workflows.detection import completed_target_jobs

    return completed_target_jobs()


def target_label(row: dict) -> str:
    chains = ",".join(row.get("chains") or [])
    return f"{row['target_name']} ({row['job_code']}, chains {chains or 'unknown'})"


def design_jobs_for_target(target_pdb: Path) -> list[dict]:
    target_path = str(target_pdb)
    rows: list[dict] = []
    for row in collect_jobs(DESIGN_GROUP):
        run_dir = Path(row["run_dir"])
        payload = read_json(run_dir / "input.json")
        inputs = payload.get("inputs") or {}
        if inputs.get("target_pdb") != target_path:
            continue
        result = read_json(run_dir / "result.json")
        rows.append(
            {
                **row,
                "success": result.get("success"),
                "candidate_count": (result.get("metrics") or {}).get("candidate_count", ""),
            }
        )
    return rows


def default_target_contig(target_pdb: Path, target_chains: list[str], binder_length: str) -> str:
    summary = pdb_summary(target_pdb.read_text(errors="ignore"))
    chain_rows = {row["chain_id"]: row for row in summary.get("chains", [])}
    segments: list[str] = []
    for chain_id in target_chains:
        row = chain_rows.get(chain_id)
        residues = sorted(row.get("residues") or []) if row else []
        if not residues:
            continue
        start = end = residues[0]
        for residue in residues[1:]:
            if residue == end + 1:
                end = residue
                continue
            segments.append(f"{chain_id}{start}-{end}")
            start = end = residue
        segments.append(f"{chain_id}{start}-{end}")
    return "/".join(segments) + f"/0 {binder_length}" if segments else ""


def normalize_hotspots(text: str) -> str:
    tokens = []
    for raw in re.split(r"[,;\s]+", text.strip()):
        token = raw.strip()
        if not token:
            continue
        token = token.replace(":", "")
        if not re.fullmatch(r"[A-Za-z][0-9]+", token):
            raise ValueError(f"Hotspot '{raw}' must look like A123.")
        tokens.append(token.upper())
    return ",".join(tokens)


def validate_rfdiffusion_inputs(contig: str, hotspots: str, binder_length: str, num_designs: int, timesteps: int) -> None:
    if not contig.strip():
        raise ValueError("Contig is required.")
    if "/0 " not in contig:
        raise ValueError("Binder contig should look like A10-150/0 55-55.")
    if not re.fullmatch(r"\d+(-\d+)?", binder_length.strip()):
        raise ValueError("Binder length should look like 55 or 55-80.")
    if num_designs < 1:
        raise ValueError("Number of designs must be at least 1.")
    if timesteps < 1:
        raise ValueError("Timesteps must be at least 1.")
    if hotspots:
        normalize_hotspots(hotspots)


def validate_rfdiffusion_optional_params(contigmap_length: str = "", inpaint_seq: str = "") -> None:
    if contigmap_length and not re.fullmatch(r"\d+(-\d+)?", contigmap_length.strip()):
        raise ValueError("contigmap.length should look like 123 or 123-456.")
    if inpaint_seq:
        for segment in inpaint_seq.split("/"):
            if not re.fullmatch(r"[A-Za-z][0-9]+(-[0-9]+)?", segment.strip()):
                raise ValueError("contigmap.inpaint_seq should look like A10-20/A22/B30-40.")


def build_rfdiffusion_run_parameters(
    timesteps: int,
    partial_diffusion: bool = False,
    contigmap_length: str = "",
    inpaint_seq: str = "",
    model_weights: str = "Complex_base",
    deterministic: bool = False,
    noise_scale_ca: str = "",
    noise_scale_frame: str = "",
    extra_run_parameters: str = "",
) -> str:
    validate_rfdiffusion_optional_params(contigmap_length, inpaint_seq)
    parts = [f"diffuser.partial_T={timesteps}" if partial_diffusion else f"diffuser.T={timesteps}"]
    if contigmap_length.strip():
        length = contigmap_length.strip()
        if "-" not in length:
            length = f"{length}-{length}"
        parts.append(f"contigmap.length={length}")
    if inpaint_seq.strip():
        parts.append(f"contigmap.inpaint_seq=[{inpaint_seq.strip()}]")
    if model_weights not in {"", "Base", "Complex_base"}:
        parts.append(f"inference.ckpt_override_path=/models/{model_weights}_ckpt.pt")
    if deterministic:
        parts.append("inference.deterministic=True")
    if noise_scale_ca.strip():
        parts.append(f"denoiser.noise_scale_ca={noise_scale_ca.strip()}")
    if noise_scale_frame.strip():
        parts.append(f"denoiser.noise_scale_frame={noise_scale_frame.strip()}")
    if extra_run_parameters.strip():
        parts.append(extra_run_parameters.strip())
    return " ".join(parts)


def parse_binder_lengths(text: str) -> list[int]:
    tokens = [token for token in re.split(r"[-,\s]+", text.strip()) if token]
    if not tokens:
        raise ValueError("Binder length is required.")
    lengths = [int(token) for token in tokens]
    if any(length < 1 for length in lengths):
        raise ValueError("Binder lengths must be positive.")
    if len(lengths) > 2:
        raise ValueError("Binder length should be a single value or a min-max range.")
    if len(lengths) == 2 and lengths[1] < lengths[0]:
        raise ValueError("Binder length max must be greater than or equal to min.")
    return [lengths[0]] if len(lengths) == 2 and lengths[0] == lengths[1] else lengths


def _ovo_bindcraft_resource(*parts: str) -> Path:
    return Path("/home/user/programs/ovo-git/ovo/resources/bindcraft").joinpath(*parts)


def _read_json_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"BindCraft settings file not found: {path}")
    return json.loads(path.read_text())


def _bindcraft_advanced_settings(enable_mpnn: bool, max_trajectories: int) -> dict:
    settings = _read_json_file(_ovo_bindcraft_resource("settings_advanced", "default_4stage_multimer.json"))
    settings.update(
        {
            "enable_mpnn": enable_mpnn,
            "num_seqs": 1,
            "max_mpnn_sequences": 1,
            "max_trajectories": max_trajectories,
            "save_design_animations": False,
            "save_design_trajectory_plots": False,
            "save_trajectory_pickle": False,
            "remove_unrelaxed_trajectory": False,
            "remove_unrelaxed_complex": False,
            "remove_binder_monomer": False,
            "af_params_dir": "alphafold_models_path",
        }
    )
    if not enable_mpnn:
        settings["enable_rejection_check"] = False
    return settings


def _copy_bindcraft_settings(run_dir: Path, params: dict) -> None:
    bindcraft_dir = run_dir / "artifacts" / "raw" / "bindcraft"
    bindcraft_dir.mkdir(parents=True, exist_ok=True)
    input_dict = {
        "design_path": "output",
        "starting_pdb": "target.pdb",
        "binder_name": "design",
        "chains": ",".join(params["target_chains"]),
        "target_hotspot_residues": params.get("hotspots") or "",
        "lengths": parse_binder_lengths(params["binder_length"]),
        "number_of_final_designs": params["number_of_final_designs"],
    }
    write_json(bindcraft_dir / "input.json", input_dict)
    write_json(
        bindcraft_dir / "settings_advanced.json",
        _bindcraft_advanced_settings(params["enable_mpnn"], params["max_trajectories"]),
    )
    filters_path = _ovo_bindcraft_resource("settings_filters", params["filter_settings"])
    write_json(bindcraft_dir / "settings_filters.json", _read_json_file(filters_path))


def _read_bindcraft_stats(output_dir: Path) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for csv_name in ["final_design_stats.csv", "trajectory_stats.csv", "mpnn_design_stats.csv"]:
        path = output_dir / csv_name
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                design_id = row.get("Design") or row.get("Name")
                if not design_id:
                    continue
                stats.setdefault(design_id, {}).update({k: v for k, v in row.items() if v not in {"", None}})
    return stats


def _normalize_bindcraft_candidates(run_dir: Path, params: dict, target_artifact: Path) -> list[dict]:
    output_dir = run_dir / "artifacts" / "raw" / "bindcraft" / "output"
    stats = _read_bindcraft_stats(output_dir)
    pdb_paths = []
    for pattern in [
        "Accepted/*.pdb",
        "Trajectory/Relaxed/*.pdb",
        "Trajectory/*.pdb",
        "Trajectory/Clashing/*.pdb",
        "Rejected/*.pdb",
    ]:
        pdb_paths.extend(sorted(output_dir.glob(pattern)))

    seen: set[Path] = set()
    candidates: list[dict] = []
    for index, pdb_path in enumerate([path for path in pdb_paths if not (path in seen or seen.add(path))], start=1):
        design_name = pdb_path.stem
        try:
            complex_path = str(pdb_path.relative_to(run_dir))
            target_path = str(target_artifact.relative_to(run_dir))
        except ValueError:
            complex_path = str(pdb_path)
            target_path = str(target_artifact)
        candidates.append(
            {
                "candidate_id": f"bindcraft_{index:05d}",
                "source_tool": "bindcraft",
                "stage": STAGE_GENERATION_BACKBONE_SEQUENCE if params.get("enable_mpnn") else STAGE_GENERATION_BACKBONE,
                "target_pdb": target_path,
                "complex_pdb": complex_path,
                "binder_pdb": None,
                "binder_sequence": (stats.get(design_name) or {}).get("Sequence"),
                "target_chains": params.get("target_chains", []),
                "hotspots": [token for token in params.get("hotspots", "").split(",") if token],
                "binder_length": params.get("binder_length"),
                "contig": None,
                "metrics": stats.get(design_name, {}),
                "raw_metadata": {"bindcraft_design": design_name},
            }
        )
    return write_candidates(run_dir, "bindcraft", candidates)


def _rel_path(run_dir: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def _write_candidates(run_dir: Path, tool: str, candidates: list[dict]) -> None:
    write_candidates(run_dir, tool, candidates)


def _generic_candidate_records(
    run_dir: Path,
    tool: str,
    stage: str,
    params: dict,
    target_artifact: Path | None,
    structure_patterns: list[str],
    metadata_patterns: list[str] | None = None,
) -> list[dict]:
    metadata_patterns = metadata_patterns or []
    structure_paths: list[Path] = []
    for pattern in structure_patterns:
        structure_paths.extend(sorted((run_dir / "artifacts").glob(pattern)))
    metadata_paths: list[Path] = []
    for pattern in metadata_patterns:
        metadata_paths.extend(sorted((run_dir / "artifacts").glob(pattern)))
    candidates: list[dict] = []
    for index, structure_path in enumerate(structure_paths, start=1):
        metadata_path = metadata_paths[index - 1] if index - 1 < len(metadata_paths) else None
        candidates.append(
            {
                "candidate_id": f"{tool}_{index:05d}",
                "source_tool": tool,
                "stage": stage,
                "target_pdb": _rel_path(run_dir, target_artifact),
                "complex_pdb": _rel_path(run_dir, structure_path),
                "binder_pdb": None,
                "binder_sequence": None,
                "target_chains": params.get("target_chains", []),
                "hotspots": [token for token in params.get("hotspots", "").split(",") if token],
                "binder_length": params.get("binder_length"),
                "contig": params.get("contig"),
                "metrics": {},
                "raw_metadata": {"metadata_path": _rel_path(run_dir, metadata_path)},
            }
        )
    _write_candidates(run_dir, tool, candidates)
    return candidates


def _finish_design_job(run_dir: Path, success: bool, rc: int, candidates: list[dict], artifact_patterns: list[tuple[str, str]]) -> None:
    artifacts = _collect_artifacts(
        run_dir,
        run_dir / "artifacts",
        artifact_patterns
        + [
            ("normalized_candidates/*.jsonl", "normalized_candidates"),
            ("normalized_candidates/*.json", "campaign_result"),
        ],
    )
    finish_job(
        run_dir,
        success and bool(candidates),
        {
            "outputs": {"artifacts": artifacts, "candidates": candidates},
            "metrics": {"return_code": rc, "artifact_count": len(artifacts), "candidate_count": len(candidates)},
            "downstream_artifacts": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
            },
        },
    )


def _copy_target_for_design(job_run_dir: Path, target_pdb: Path, target_chains: list[str], tool_dir: str = "input") -> Path:
    target_artifact = artifact_path(job_run_dir, tool_dir, "target.pdb")
    target_artifact.write_text(filter_pdb_text(target_pdb.read_text(errors="ignore"), keep_chains=set(target_chains) or None))
    return target_artifact


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
        for path_text in glob.glob(str(root / pattern), recursive=True):
            path = Path(path_text)
            if path.is_file():
                artifacts.append(Artifact(path.stem, path, artifact_type).to_json(run_dir))
    return artifacts


def _candidate_record(
    run_dir: Path,
    candidate_id: str,
    complex_pdb: Path,
    trb_path: Path | None,
    params: dict,
    target_artifact: Path,
) -> dict:
    def rel(path: Path | None) -> str | None:
        if path is None:
            return None
        try:
            return str(path.relative_to(run_dir))
        except ValueError:
            return str(path)

    return {
        "candidate_id": candidate_id,
        "source_tool": "rfdiffusion_classic",
        "stage": STAGE_GENERATION_BACKBONE,
        "target_pdb": rel(target_artifact),
        "complex_pdb": rel(complex_pdb),
        "binder_pdb": None,
        "binder_sequence": None,
        "target_chains": params.get("target_chains", []),
        "hotspots": [token for token in params.get("hotspots", "").split(",") if token],
        "binder_length": params.get("binder_length"),
        "contig": params.get("contig"),
        "metrics": {},
        "raw_metadata": {"trb_path": rel(trb_path)},
    }


def _normalize_rfdiffusion_candidates(run_dir: Path, params: dict, target_artifact: Path) -> list[dict]:
    raw_dir = run_dir / "artifacts" / "raw" / "rfdiffusion" / "output"
    candidates: list[dict] = []
    for index, pdb_path in enumerate(sorted(raw_dir.glob("design_*.pdb")), start=1):
        trb_path = pdb_path.with_suffix(".trb")
        candidates.append(
            _candidate_record(
                run_dir,
                candidate_id=f"rfdiffusion_classic_{index:05d}",
                complex_pdb=pdb_path,
                trb_path=trb_path if trb_path.exists() else None,
                params=params,
                target_artifact=target_artifact,
            )
        )
    return write_candidates(run_dir, "rfdiffusion_classic", candidates)


def _rfdiffusion_pipeline_payload(
    *,
    target_pdb_path: str,
    contig: str,
    hotspots: str,
    num_designs: int,
    final_run_parameters: str,
    backbone_filters: str,
    mpnn_num_sequences: int,
    mpnn_sampling_temp: float,
    mpnn_omit_aa: str,
    mpnn_bias_aa: str,
    mpnn_run_parameters: str,
    sequence_design_method: str,
    mpnn_fastrelax_cycles: int,
    refolding_test: str,
    disable_pyrosetta_scoring: bool = True,
) -> dict:
    if sequence_design_method == "fastrelax":
        mpnn_args = f'-omit_AAs "{mpnn_omit_aa.strip()}" -temperature {mpnn_sampling_temp}'
        if mpnn_bias_aa.strip():
            mpnn_args += f' -bias_AA "{mpnn_bias_aa.strip()}"'
    else:
        mpnn_args = f'--omit_AA "{mpnn_omit_aa.strip()}" --temperature {mpnn_sampling_temp}'
        if mpnn_bias_aa.strip():
            mpnn_args += f' --bias_AA "{mpnn_bias_aa.strip()}"'
    if mpnn_run_parameters.strip():
        mpnn_args += f" {mpnn_run_parameters.strip()}"
    return {
        "pipeline": "mn-protein-design-rfdiffusion",
        "design_type": "binder",
        "rfdiffusion_input_pdb": target_pdb_path,
        "rfdiffusion_num_designs": num_designs,
        "rfdiffusion_contig": contig,
        "rfdiffusion_run_parameters": final_run_parameters,
        "hotspot": hotspots,
        "backbone_filters": backbone_filters.strip() or "none",
        "mpnn_num_sequences": 1 if sequence_design_method == "fastrelax" else mpnn_num_sequences,
        "mpnn_fastrelax_cycles": mpnn_fastrelax_cycles if sequence_design_method == "fastrelax" else 0,
        "mpnn_run_parameters": mpnn_args.strip(),
        "disable_pyrosetta_scoring": disable_pyrosetta_scoring,
        "refolding_tests": refolding_test or None,
        "planned_steps": [
            "rfdiffusion_backbone",
            "backbone_metrics",
            "sequence_design",
            "refolding_validation" if refolding_test else "skip_refolding_validation",
        ],
    }


def run_rfdiffusion_classic(
    target_pdb: Path,
    target_chains: list[str],
    contig: str,
    binder_length: str,
    hotspots: str = "",
    num_designs: int = 1,
    timesteps: int = 50,
    model_weights: str = "Complex_base",
    extra_run_parameters: str = "",
    partial_diffusion: bool = False,
    contigmap_length: str = "",
    inpaint_seq: str = "",
    backbone_filters: str = "",
    deterministic: bool = False,
    noise_scale_ca: str = "",
    noise_scale_frame: str = "",
    save_trajectory: bool = False,
    editable_run_parameters: str = "",
    edited_target_pdb_text: str | None = None,
    mpnn_num_sequences: int = 1,
    mpnn_sampling_temp: float = 0.0001,
    mpnn_omit_aa: str = "CX",
    mpnn_bias_aa: str = "",
    mpnn_run_parameters: str = "",
    sequence_design_method: str = "ligandmpnn",
    mpnn_fastrelax_cycles: int = 0,
    refolding_test: str = "af2_model_1_multimer_tt_3rec",
    execution_backend: str = "docker",
) -> Path:
    hotspots = normalize_hotspots(hotspots) if hotspots else ""
    validate_rfdiffusion_inputs(contig, hotspots, binder_length, num_designs, timesteps)
    validate_rfdiffusion_optional_params(contigmap_length, inpaint_seq)
    manifest = load_manifest("rfdiffusion_classic")
    final_run_parameters = editable_run_parameters.strip() or build_rfdiffusion_run_parameters(
        timesteps=timesteps,
        partial_diffusion=partial_diffusion,
        contigmap_length=contigmap_length,
        inpaint_seq=inpaint_seq,
        model_weights=model_weights,
        deterministic=deterministic,
        noise_scale_ca=noise_scale_ca,
        noise_scale_frame=noise_scale_frame,
        extra_run_parameters=extra_run_parameters,
    )
    params = {
        "target_chains": target_chains,
        "contig": contig,
        "binder_length": binder_length,
        "hotspots": hotspots,
        "num_designs": num_designs,
        "timesteps": timesteps,
        "model_weights": model_weights,
        "extra_run_parameters": extra_run_parameters.strip(),
        "partial_diffusion": partial_diffusion,
        "contigmap_length": contigmap_length.strip(),
        "inpaint_seq": inpaint_seq.strip(),
        "backbone_filters": backbone_filters.strip(),
        "deterministic": deterministic,
        "noise_scale_ca": noise_scale_ca.strip(),
        "noise_scale_frame": noise_scale_frame.strip(),
        "save_trajectory": save_trajectory,
        "rfdiffusion_run_parameters": final_run_parameters,
        "mpnn_num_sequences": mpnn_num_sequences,
        "mpnn_sampling_temp": mpnn_sampling_temp,
        "mpnn_omit_aa": mpnn_omit_aa.strip(),
        "mpnn_bias_aa": mpnn_bias_aa.strip(),
        "mpnn_run_parameters": mpnn_run_parameters.strip(),
        "sequence_design_method": sequence_design_method,
        "mpnn_fastrelax_cycles": mpnn_fastrelax_cycles,
        "refolding_test": refolding_test,
        "execution_backend": execution_backend,
        "pipeline_mode": "mn_protein_design_rfdiffusion",
    }
    job = create_job(
        DESIGN_GROUP,
        job_type="design_campaign",
        tool="rfdiffusion_classic",
        inputs={"target_pdb": str(target_pdb), "target_chains": target_chains},
        params=params,
    )

    target_artifact = artifact_path(job.run_dir, "input", "target.pdb")
    target_text = edited_target_pdb_text if edited_target_pdb_text is not None else target_pdb.read_text(errors="ignore")
    target_artifact.write_text(filter_pdb_text(target_text, keep_chains=set(target_chains) or None))

    raw_dir = job.run_dir / "artifacts" / "raw" / "rfdiffusion"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pipeline_dir = artifact_path(job.run_dir, "raw", "rfdiffusion", "pipeline_inputs")
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    pipeline_payload = _rfdiffusion_pipeline_payload(
        target_pdb_path="/work/artifacts/input/target.pdb",
        contig=contig,
        hotspots=hotspots,
        num_designs=num_designs,
        final_run_parameters=final_run_parameters,
        backbone_filters=backbone_filters,
        mpnn_num_sequences=mpnn_num_sequences,
        mpnn_sampling_temp=mpnn_sampling_temp,
        mpnn_omit_aa=mpnn_omit_aa,
        mpnn_bias_aa=mpnn_bias_aa,
        mpnn_run_parameters=mpnn_run_parameters,
        sequence_design_method=sequence_design_method,
        mpnn_fastrelax_cycles=mpnn_fastrelax_cycles,
        refolding_test=refolding_test,
    )
    write_json(pipeline_dir / "mn_rfdiffusion_pipeline_params.json", pipeline_payload)
    (pipeline_dir / "rfdiffusion_run_parameters.txt").write_text(final_run_parameters + "\n")
    (pipeline_dir / "rfdiffusion_contig.txt").write_text(contig + "\n")
    (pipeline_dir / "hotspots.txt").write_text(hotspots + "\n")

    hotspot_arg = f' "ppi.hotspot_res=[{hotspots}]"' if hotspots else ""
    traj_arg = "true" if save_trajectory else "false"
    script = (
        "set -euxo pipefail; "
        "rm -rf /work/artifacts/raw/rfdiffusion/output; "
        "export PYTHONPATH=/opt/RFdiffusion:/opt/RFdiffusion/env/SE3Transformer; "
        "mkdir -p /work/artifacts/raw/rfdiffusion/output; "
        "cd /work/artifacts/raw/rfdiffusion; "
        "python3 /opt/RFdiffusion/scripts/run_inference.py "
        "inference.output_prefix=output/design "
        "inference.model_directory_path=/models "
        "inference.schedule_directory_path=/opt/RFdiffusion/schedules "
        "inference.input_pdb=/work/artifacts/input/target.pdb "
        f"inference.num_designs={num_designs} "
        f'"contigmap.contigs=[{contig}]"'
        f"{hotspot_arg} "
        f"inference.write_trajectory={traj_arg} "
        f"{final_run_parameters}; "
        'test -n "$(find /work/artifacts/raw/rfdiffusion/output -maxdepth 1 -type f -name \"design_*.pdb\" -print -quit)"'
    )
    if execution_backend == "nextflow":
        if shutil.which("nextflow") is None:
            raise RuntimeError("Nextflow is not on PATH. Update the app env with `conda env update -f environment.yml --prune`.")
        nextflow_dir = raw_dir / "nextflow"
        nextflow_dir.mkdir(parents=True, exist_ok=True)
        pipeline_path = Path(__file__).resolve().parents[1] / "pipelines" / "rfdiffusion_backbone"
        command = [
            "bash",
            "-lc",
            " ".join(
                [
                    f"cd {shlex.quote(str(nextflow_dir))}",
                    "&&",
                    "nextflow",
                    "run",
                    shlex.quote(str(pipeline_path)),
                    "-config",
                    shlex.quote(str(pipeline_path / "nextflow.config")),
                    "-work-dir",
                    shlex.quote(str(nextflow_dir / "work")),
                    "-with-trace",
                    "trace.txt",
                    "-with-report",
                    "report.html",
                    "-with-timeline",
                    "timeline.html",
                    "--input_pdb",
                    shlex.quote(str(target_artifact.resolve())),
                    "--publish_dir",
                    shlex.quote(str(raw_dir / "output")),
                    "--rfdiffusion_image",
                    shlex.quote(str(manifest["image"])),
                    "--rfdiffusion_num_designs",
                    str(num_designs),
                    "--rfdiffusion_contig",
                    shlex.quote(contig),
                    "--hotspot",
                    shlex.quote(hotspots),
                    "--write_trajectory",
                    str(save_trajectory).lower(),
                    "--rfdiffusion_run_parameters",
                    shlex.quote(final_run_parameters),
                ]
            ),
        ]
        steps = [{"name": "rfdiffusion-nextflow-backbone", "command": command}]
    else:
        command = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "-v",
            f"{job.run_dir}:/work",
            "-v",
            "/mnt/db/reference_files/rfdiffusion_models:/models:ro",
            "-w",
            "/work",
            manifest["image"],
            "bash",
            "-lc",
            script,
        ]
        steps = [{"name": "rfdiffusion-docker-backbone", "command": command}]
    rc = _run_shell_steps(job.run_dir, steps)

    candidates = _normalize_rfdiffusion_candidates(job.run_dir, params, target_artifact) if rc == 0 else []
    artifacts = _collect_artifacts(
        job.run_dir,
        job.run_dir / "artifacts",
        [
            ("raw/rfdiffusion/output/*.pdb", "designed_complex_pdb"),
            ("raw/rfdiffusion/output/*.trb", "rfdiffusion_trb"),
            ("raw/rfdiffusion/pipeline_inputs/*", "rfdiffusion_pipeline_input"),
            ("raw/rfdiffusion/nextflow/*.txt", "nextflow_trace"),
            ("raw/rfdiffusion/nextflow/*.html", "nextflow_report"),
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
                "target_pdb": str(target_artifact.relative_to(job.run_dir)),
                "pipeline_inputs": "artifacts/raw/rfdiffusion/pipeline_inputs",
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
            },
        },
    )
    return job.run_dir


def run_bindcraft(
    target_pdb: Path,
    target_chains: list[str],
    binder_length: str,
    hotspots: str = "",
    number_of_final_designs: int = 1,
    time_limit_seconds: int = 900,
    single_af_model: bool = True,
    enable_mpnn: bool = False,
    max_trajectories: int = 1,
    filter_settings: str = "no_filters.json",
) -> Path:
    if not target_chains:
        raise ValueError("At least one target chain is required.")
    hotspots = normalize_hotspots(hotspots) if hotspots else ""
    parse_binder_lengths(binder_length)
    if number_of_final_designs < 1:
        raise ValueError("Desired final designs must be at least 1.")
    if time_limit_seconds < 60:
        raise ValueError("Time limit should be at least 60 seconds.")
    if max_trajectories < 1:
        raise ValueError("Max trajectories must be at least 1.")

    manifest = load_manifest("bindcraft")
    params = {
        "target_chains": target_chains,
        "binder_length": binder_length,
        "hotspots": hotspots,
        "number_of_final_designs": number_of_final_designs,
        "time_limit_seconds": time_limit_seconds,
        "single_af_model": single_af_model,
        "enable_mpnn": enable_mpnn,
        "max_trajectories": max_trajectories,
        "filter_settings": filter_settings,
    }
    job = create_job(
        DESIGN_GROUP,
        job_type="design_campaign",
        tool="bindcraft",
        inputs={"target_pdb": str(target_pdb), "target_chains": target_chains},
        params=params,
    )
    target_artifact = artifact_path(job.run_dir, "raw", "bindcraft", "input_pdb.pdb")
    target_text = target_pdb.read_text(errors="ignore")
    target_artifact.write_text(filter_pdb_text(target_text, keep_chains=set(target_chains) or None))
    _copy_bindcraft_settings(job.run_dir, params)

    model_patch = ""
    if single_af_model:
        model_patch = (
            'sed -i "s/design_models = \\[0,1,2,3,4\\]/design_models = [0]/" /content/bindcraft/functions/generic_utils.py; '
            'sed -i "s/prediction_models = \\[0,1\\]/prediction_models = [0]/" /content/bindcraft/functions/generic_utils.py; '
        )
    script = (
        "set -euxo pipefail; "
        "cd /work/artifacts/raw/bindcraft; "
        "ln -sf input_pdb.pdb target.pdb; "
        "ln -sfn /af_models alphafold_models_path; "
        "cp settings_advanced.json advanced.json; "
        f"{model_patch}"
        f"timeout {time_limit_seconds} python /content/bindcraft/bindcraft.py "
        "--settings input.json --advanced advanced.json --filters settings_filters.json; "
        'test -n "$(find output -type f -name \"*.pdb\" -print -quit)"'
    )
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-v",
        f"{job.run_dir}:/work",
        "-v",
        "/mnt/db/reference_files/alphafold_models:/af_models:ro",
        "-w",
        "/work",
        manifest["image"],
        "bash",
        "-lc",
        script,
    ]
    rc = _run_shell_steps(job.run_dir, [{"name": "bindcraft", "command": command}])
    candidates = _normalize_bindcraft_candidates(job.run_dir, params, target_artifact) if rc == 0 else []
    artifacts = _collect_artifacts(
        job.run_dir,
        job.run_dir / "artifacts",
        [
            ("raw/bindcraft/output/**/*.pdb", "designed_complex_pdb"),
            ("raw/bindcraft/output/**/*.csv", "bindcraft_metrics_table"),
            ("normalized_candidates/*.jsonl", "normalized_candidates"),
            ("normalized_candidates/*.json", "campaign_result"),
            ("raw/bindcraft/*.json", "bindcraft_input_settings"),
        ],
    )
    finish_job(
        job.run_dir,
        rc == 0 and bool(candidates),
        {
            "outputs": {"artifacts": artifacts, "candidates": candidates},
            "metrics": {"return_code": rc, "artifact_count": len(artifacts), "candidate_count": len(candidates)},
            "downstream_artifacts": {
                "target_pdb": str(target_artifact.relative_to(job.run_dir)),
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
            },
        },
    )
    return job.run_dir


def run_rfdiffusion3_foundry(
    target_pdb: Path,
    target_chains: list[str],
    binder_length: str,
    hotspots: str = "",
    num_designs: int = 1,
    timesteps: int = 50,
) -> Path:
    hotspots = normalize_hotspots(hotspots) if hotspots else ""
    manifest = load_manifest("rfdiffusion3_foundry")
    params = {
        "target_chains": target_chains,
        "binder_length": binder_length,
        "hotspots": hotspots,
        "num_designs": num_designs,
        "timesteps": timesteps,
    }
    job = create_job(DESIGN_GROUP, "design_campaign", "rfdiffusion3_foundry", {"target_pdb": str(target_pdb), "target_chains": target_chains}, params)
    target_artifact = _copy_target_for_design(job.run_dir, target_pdb, target_chains, "raw/rfdiffusion3_foundry")
    contig = default_target_contig(target_artifact, target_chains, binder_length).replace("/0 ", ",/0,")
    design_input = {
        "design_1": {
            "dialect": 2,
            "input": "target.pdb",
            "contig": contig,
            "is_non_loopy": True,
            "infer_ori_strategy": "hotspots" if hotspots else "default",
        }
    }
    if hotspots:
        design_input["design_1"]["select_hotspots"] = {token: "CA" for token in hotspots.split(",") if token}
    write_json(job.run_dir / "artifacts" / "raw" / "rfdiffusion3_foundry" / "rfd3_inputs_staged.json", design_input)
    script = (
        "set -euxo pipefail; cd /work/artifacts/raw/rfdiffusion3_foundry; rm -rf rfd3; "
        f"rfd3 design out_dir=rfd3 inputs=rfd3_inputs_staged.json ckpt_path=/weights/rfd3_latest.ckpt "
        f"diffusion_batch_size=1 n_batches={num_designs} inference_sampler.num_timesteps={timesteps} "
        "skip_existing=False prevalidate_inputs=True; "
        'test -n "$(find rfd3 -type f \\( -name \"*.cif\" -o -name \"*.cif.gz\" \\) -print -quit)"'
    )
    command = ["docker", "run", "--rm", "--gpus", "all", "-v", f"{job.run_dir}:/work", "-v", "/mnt/db/reference_files/foundry:/weights:ro", "-w", "/work", manifest["image"], "bash", "-lc", script]
    rc = _run_shell_steps(job.run_dir, [{"name": "rfdiffusion3-foundry", "command": command}])
    candidates = _generic_candidate_records(job.run_dir, "rfdiffusion3_foundry", "backbone", {**params, "contig": contig}, target_artifact, ["raw/rfdiffusion3_foundry/rfd3/*.cif", "raw/rfdiffusion3_foundry/rfd3/*.cif.gz"], ["raw/rfdiffusion3_foundry/rfd3/*.json"]) if rc == 0 else []
    _finish_design_job(job.run_dir, rc == 0, rc, candidates, [("raw/rfdiffusion3_foundry/rfd3/**/*", "rfdiffusion3_output"), ("raw/rfdiffusion3_foundry/*.json", "rfdiffusion3_input")])
    return job.run_dir


def run_boltzgen(
    target_pdb: Path,
    target_chains: list[str],
    binder_length: str,
    hotspots: str = "",
    num_designs: int = 1,
    sampling_steps: int = 20,
) -> Path:
    hotspots = normalize_hotspots(hotspots) if hotspots else ""
    lengths = parse_binder_lengths(binder_length)
    binder_len = lengths[0]
    manifest = load_manifest("boltzgen")
    params = {"target_chains": target_chains, "binder_length": binder_length, "hotspots": hotspots, "num_designs": num_designs, "sampling_steps": sampling_steps}
    job = create_job(DESIGN_GROUP, "design_campaign", "boltzgen", {"target_pdb": str(target_pdb), "target_chains": target_chains}, params)
    raw_dir = job.run_dir / "artifacts" / "raw" / "boltzgen"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_artifact = _copy_target_for_design(job.run_dir, target_pdb, target_chains, "raw/boltzgen/input")
    first_chain = target_chains[0] if target_chains else "A"
    residues = pdb_summary(target_artifact.read_text(errors="ignore"))["chains"][0]["residues"] if pdb_summary(target_artifact.read_text(errors="ignore")).get("chains") else []
    start = min(residues) if residues else 1
    end = max(residues) if residues else 9999
    binding = ",".join(token[1:] for token in hotspots.split(",") if token.startswith(first_chain)) if hotspots else ""
    spec = [
        "entities:",
        "  - protein:",
        "      id: B",
        f"      sequence: {binder_len}",
        "  - file:",
        "      path: target.pdb",
        "      include:",
        "        - chain:",
        f"            id: {first_chain}",
        f"            res_index: {start}..{end}",
    ]
    if binding:
        spec.extend(["      binding_types:", "        - chain:", f"            id: {first_chain}", f"            binding: {binding}"])
    spec.append('      structure_groups: "all"')
    (raw_dir / "input" / "target_binder.yaml").write_text("\n".join(spec) + "\n")
    script = (
        "set -euxo pipefail; cd /work/artifacts/raw/boltzgen; "
        "boltzgen check input/target_binder.yaml --output checked --cache /cache; rm -rf run-design-only; "
        f"boltzgen run input/target_binder.yaml --output run-design-only --protocol protein-anything --steps design "
        f"--num_designs {num_designs} --diffusion_batch_size 1 --devices 1 --num_workers 1 --cache /cache "
        f"--config design sampling_steps={sampling_steps} compile_pairformer=false compile_structure=false data.num_workers=1; "
        'test -n "$(find run-design-only/intermediate_designs -maxdepth 1 -type f -name \"*.cif\" -print -quit)"'
    )
    command = ["docker", "run", "--rm", "--gpus", "all", "--entrypoint", "/bin/bash", "-v", "/mnt/db/reference_files/boltzgen-cache:/cache", "-v", f"{job.run_dir}:/work", "-w", "/work", manifest["image"], "-lc", script]
    rc = _run_shell_steps(job.run_dir, [{"name": "boltzgen", "command": command}])
    candidates = _generic_candidate_records(job.run_dir, "boltzgen", "design_only", params, target_artifact, ["raw/boltzgen/run-design-only/intermediate_designs/*.cif"], ["raw/boltzgen/run-design-only/intermediate_designs/*.npz"]) if rc == 0 else []
    _finish_design_job(job.run_dir, rc == 0, rc, candidates, [("raw/boltzgen/**/*.cif", "cif"), ("raw/boltzgen/**/*.npz", "npz"), ("raw/boltzgen/**/*.yaml", "yaml")])
    return job.run_dir


def run_genie3(num_designs: int = 1, config_name: str = "genie3_pdl1_binder_smoke.yaml") -> Path:
    manifest = load_manifest("genie3")
    params = {"num_designs": num_designs, "config_name": config_name, "contract": "BinderBench PDL1"}
    job = create_job(DESIGN_GROUP, "design_campaign", "genie3", {"target_pdb": "bundled:04_pdl1"}, params)
    out_root = job.run_dir / "artifacts" / "raw" / "genie3"
    out_root.mkdir(parents=True, exist_ok=True)
    command = ["docker", "run", "--rm", "--gpus", "all", "-v", "/mnt/db/reference_files/genie3/pretrained:/opt/genie3/pretrained:ro", "-v", f"{Path.cwd() / 'smoke_tests'}:/work/config:ro", "-v", f"{out_root}:/work/output", manifest["image"], "genie3", "generate", "-c", f"/work/config/{config_name}", "--num-devices", "1", "--verbose"]
    rc = _run_shell_steps(job.run_dir, [{"name": "genie3", "command": command}])
    candidates = _generic_candidate_records(job.run_dir, "genie3", "backbone", params, None, ["raw/genie3/**/*.pdb"], ["raw/genie3/**/*.json"]) if rc == 0 else []
    _finish_design_job(job.run_dir, rc == 0, rc, candidates, [("raw/genie3/**/*.pdb", "pdb"), ("raw/genie3/**/*.json", "json")])
    return job.run_dir


def run_pxdesign(num_designs: int = 1, n_steps: int = 400, dtype: str = "bf16", use_msa: bool = False) -> Path:
    manifest = load_manifest("pxdesign")
    params = {"num_designs": num_designs, "n_steps": n_steps, "dtype": dtype, "use_msa": use_msa, "contract": "PDL1_quick_start"}
    job = create_job(DESIGN_GROUP, "design_campaign", "pxdesign", {"target_pdb": "bundled:PDL1_quick_start"}, params)
    out_dir = job.run_dir / "artifacts" / "raw" / "pxdesign"
    command = ["docker", "run", "--rm", "--gpus", "all", "-v", f"{job.run_dir / 'artifacts' / 'raw'}:/work", "-v", "/mnt/db/reference_files/pxdesign:/ref/pxdesign", manifest["image"], "pxdesign", "infer", "-i", "/opt/PXDesign/examples/PDL1_quick_start.yaml", "-o", "/work/pxdesign", "--N_sample", str(num_designs), "--N_step", str(n_steps), "--dtype", dtype, "--use_msa", str(use_msa).lower(), "--load_checkpoint_dir", "/ref/pxdesign/release_data/checkpoint"]
    rc = _run_shell_steps(job.run_dir, [{"name": "pxdesign", "command": command}])
    candidates = _generic_candidate_records(job.run_dir, "pxdesign", "backbone", params, None, ["raw/pxdesign/**/*.cif"]) if rc == 0 else []
    _finish_design_job(job.run_dir, rc == 0, rc, candidates, [("raw/pxdesign/**/*.cif", "cif"), ("raw/pxdesign/**/*SUCCESS_FILE", "success_file"), ("raw/pxdesign/**/*.json", "json")])
    return job.run_dir


def run_protpardelle_1c(num_designs: int = 1, seed: int = 7) -> Path:
    manifest = load_manifest("protpardelle_1c")
    params = {"num_designs": num_designs, "seed": seed, "contract": "protpardelle_pdl1_smoke"}
    job = create_job(DESIGN_GROUP, "design_campaign", "protpardelle_1c", {"target_pdb": "bundled:17_PDL1(AAV)"}, params)
    out_root = job.run_dir / "artifacts" / "raw" / "protpardelle_1c"
    out_root.mkdir(parents=True, exist_ok=True)
    command = ["docker", "run", "--rm", "--gpus", "all", "-v", "/mnt/db/reference_files/protpardelle-1c:/ref/protpardelle-1c:ro", "-v", f"{Path.cwd() / 'smoke_tests'}:/work/config:ro", "-v", f"{out_root}:/work/output", manifest["image"], "python", "-m", "protpardelle.sample", "/work/config/protpardelle_pdl1_smoke.yaml", "--motif-dir", "/opt/protpardelle-1c/examples/motifs/bindcraft", "--num-samples", str(num_designs), "--num-mpnn-seqs", "0", "--batch-size", "1", "--seed", str(seed)]
    rc = _run_shell_steps(job.run_dir, [{"name": "protpardelle-1c", "command": command}])
    candidates = _generic_candidate_records(job.run_dir, "protpardelle_1c", "backbone", params, None, ["raw/protpardelle_1c/**/*.pdb"], ["raw/protpardelle_1c/**/*.csv"]) if rc == 0 else []
    _finish_design_job(job.run_dir, rc == 0, rc, candidates, [("raw/protpardelle_1c/**/*.pdb", "pdb"), ("raw/protpardelle_1c/**/*.csv", "csv")])
    return job.run_dir


def run_proteina_complexa(run_name: str, n_steps: int = 20, replicas: int = 2) -> Path:
    manifest = load_manifest("proteina_complexa")
    params = {"run_name": run_name, "n_steps": n_steps, "replicas": replicas, "contract": "search_binder_local_pipeline PDL1"}
    job = create_job(DESIGN_GROUP, "design_campaign", "proteina_complexa", {"target_pdb": "bundled:02_PDL1"}, params)
    out_root = job.run_dir / "artifacts" / "raw" / "proteina_complexa"
    tool = Path.cwd() / "tools_to_implement" / "Proteina-Complexa"
    common_env = "export COMPLEXA_INIT=1 DATA_PATH=/workspace/protein-foundation-models/assets CKPT_PATH=/workspace/protein-foundation-models/ckpts AF2_DIR=/ref/af2 RF3_CKPT_PATH=/workspace/protein-foundation-models/community_models/ckpts/RF3/rf3_foundry_01_24_latest_remapped.ckpt RF3_EXEC_PATH=/workspace/.venv/bin/rf3 FOLDSEEK_EXEC=/workspace/.venv/bin/foldseek MMSEQS_EXEC=/workspace/.venv/bin/mmseqs SC_EXEC=/usr/local/bin/sc DSSP_EXEC=/usr/local/bin/dssp"
    common_args = f"configs/search_binder_local_pipeline.yaml ++run_name={run_name} ++generation.task_name=02_PDL1 ++generation.dataloader.dataset.nres.nsamples=1 ++generation.search.best_of_n.replicas={replicas} ++generation.args.nsteps={n_steps} ++ckpt_path=/workspace/protein-foundation-models/ckpts ++ckpt_name=complexa.ckpt ++autoencoder_ckpt_path=/workspace/protein-foundation-models/ckpts/complexa_ae.ckpt"
    steps = []
    for stage in ["generate", "filter", "evaluate", "analyze"]:
        steps.append(
            {
                "name": f"proteina-complexa-{stage}",
                "command": ["docker", "run", "--rm", "--gpus", "all", "-v", f"{tool}:/workspace/protein-foundation-models", "-v", "/mnt/db/reference_files/proteina-complexa/ckpts:/workspace/protein-foundation-models/ckpts:ro", "-v", "/mnt/db/reference_files/proteina-complexa/hf-cache:/workspace/shared-community/hf-cache", "-v", "/mnt/db/reference_files/alphafold_models:/ref/af2:ro", manifest["image"], "bash", "-lc", f"{common_env}; complexa {stage} {common_args}"],
            }
        )
    rc = _run_shell_steps(job.run_dir, steps)
    if rc == 0:
        out_root.mkdir(parents=True, exist_ok=True)
        for dirname in ["inference", "evaluation_results", "logs"]:
            src = tool / dirname
            if src.exists():
                shutil.copytree(src, out_root / dirname, dirs_exist_ok=True)
    candidates = _generic_candidate_records(job.run_dir, "proteina_complexa", "backbone+validation", params, None, ["raw/proteina_complexa/**/*.pdb"], ["raw/proteina_complexa/**/*.csv"]) if rc == 0 else []
    _finish_design_job(job.run_dir, rc == 0, rc, candidates, [("raw/proteina_complexa/**/*.pdb", "pdb"), ("raw/proteina_complexa/**/*.csv", "csv"), ("raw/proteina_complexa/**/*.json", "json"), ("raw/proteina_complexa/**/*.log", "log")])
    return job.run_dir
