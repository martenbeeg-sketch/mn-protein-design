from __future__ import annotations

import glob
import csv
import gzip
import json
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from mn_protein_design.core.artifacts import Artifact, artifact_path
from mn_protein_design.core.candidates import (
    STAGE_COMPLEX_REFOLDING,
    STAGE_GENERATION_BACKBONE,
    STAGE_GENERATION_BACKBONE_SEQUENCE,
    write_candidates,
)
from mn_protein_design.core.hotspot_metrics import calculate_hotspot_metrics, passes_hotspot_prefilter
from mn_protein_design.core.jobs import collect_jobs, create_job, finish_job, read_json, update_status, write_json
from mn_protein_design.core.manifests import load_manifest
from mn_protein_design.core.structures import filter_pdb_text, pdb_summary


DESIGN_GROUP = "design"
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


def _bindcraft_advanced_settings(
    settings_file: str,
    max_trajectories: int,
    num_seqs_override: int | None = None,
    max_mpnn_sequences_override: int | None = None,
) -> dict:
    settings = _read_json_file(_ovo_bindcraft_resource("settings_advanced", settings_file))
    settings.update(
        {
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
    if num_seqs_override is not None:
        settings["num_seqs"] = num_seqs_override
    if max_mpnn_sequences_override is not None:
        settings["max_mpnn_sequences"] = max_mpnn_sequences_override
    return settings


def _copy_bindcraft_settings(run_dir: Path, params: dict) -> None:
    bindcraft_dir = run_dir / "artifacts" / "raw" / "bindcraft"
    bindcraft_dir.mkdir(parents=True, exist_ok=True)
    binder_name = params.get("campaign_name") or "design"
    binder_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", binder_name.strip()).strip("_") or "design"
    input_dict = {
        "design_path": "output",
        "starting_pdb": "target.pdb",
        "binder_name": binder_name,
        "chains": ",".join(params["target_chains"]),
        "target_hotspot_residues": params.get("hotspots") or "",
        "lengths": parse_binder_lengths(params["binder_length"]),
        "number_of_final_designs": params["number_of_final_designs"],
    }
    write_json(bindcraft_dir / "input.json", input_dict)
    write_json(
        bindcraft_dir / "settings_advanced.json",
        _bindcraft_advanced_settings(
            params["advanced_settings_file"],
            params["max_trajectories"],
            params.get("num_seqs_override"),
            params.get("max_mpnn_sequences_override"),
        ),
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
                cleaned = {k: v for k, v in row.items() if v not in {"", None}}
                for key, value in list(cleaned.items()):
                    if key.startswith("Average_"):
                        cleaned.setdefault(key.removeprefix("Average_"), value)
                stats.setdefault(design_id, {}).update(cleaned)
    return stats


def _bindcraft_stats_for_design(stats: dict[str, dict], design_name: str) -> dict:
    if design_name in stats:
        return stats[design_name]
    model_stripped = re.sub(r"_model\d+$", "", design_name)
    if model_stripped in stats:
        return stats[model_stripped]
    for stats_name, row in stats.items():
        if design_name.startswith(stats_name) or stats_name.startswith(model_stripped):
            return row
    return {}


def _normalize_bindcraft_candidates(run_dir: Path, params: dict, target_artifact: Path) -> list[dict]:
    output_dir = run_dir / "artifacts" / "raw" / "bindcraft" / "output"
    stats = _read_bindcraft_stats(output_dir)
    pdb_by_design: dict[str, Path] = {}
    for pattern in ["Accepted/*.pdb"]:
        for pdb_path in sorted(output_dir.glob(pattern)):
            pdb_by_design.setdefault(pdb_path.stem, pdb_path)
    candidates: list[dict] = []
    for index, pdb_path in enumerate(pdb_by_design.values(), start=1):
        design_name = pdb_path.stem
        metrics = _bindcraft_stats_for_design(stats, design_name)
        binder_chains, target_chains, chain_inference = _infer_generated_chain_roles(
            target_artifact,
            pdb_path,
            params.get("target_chains", []),
        )
        binder_sequence = metrics.get("Sequence")
        if not binder_sequence and binder_chains:
            sequences = _pdb_sequences_by_chain(pdb_path)
            binder_sequence = "".join(sequences.get(chain, "") for chain in binder_chains) or None
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
                "stage": STAGE_COMPLEX_REFOLDING,
                "target_pdb": target_path,
                "complex_pdb": complex_path,
                "binder_pdb": None,
                "binder_sequence": binder_sequence,
                "target_chains": target_chains,
                "binder_chains": binder_chains,
                "hotspots": [token for token in params.get("hotspots", "").split(",") if token],
                "binder_length": params.get("binder_length"),
                "contig": None,
                "metrics": {
                    **metrics,
                    "complex_refolding_backend": "bindcraft",
                    "result_kind": "native_pipeline",
                    "target_chain_inference": chain_inference,
                },
                "raw_metadata": {
                    "result_kind": "native_pipeline",
                    "bindcraft_design": design_name,
                    "input_target_chains": params.get("target_chains", []),
                    "target_chain_inference": chain_inference,
                    "bindcraft_output_kind": pdb_path.parent.name,
                },
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


def _read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_boltzgen_generation_candidates(run_dir: Path, params: dict, target_artifact: Path) -> list[dict]:
    structure_paths = sorted((run_dir / "artifacts").glob("raw/boltzgen/run-generation-only/intermediate_designs/*.cif"))
    metadata_paths = {
        path.stem: path
        for path in sorted((run_dir / "artifacts").glob("raw/boltzgen/run-generation-only/intermediate_designs/*.npz"))
    }
    candidates: list[dict] = []
    for index, structure_path in enumerate(structure_paths, start=1):
        binder_chains, target_chains, chain_inference = _infer_generated_chain_roles(
            target_artifact,
            structure_path,
            params.get("target_chains", []),
        )
        sequences = _sequences_by_chain(structure_path)
        binder_sequence = "".join(sequences.get(chain, "") for chain in binder_chains) or None
        metadata_path = metadata_paths.get(structure_path.stem)
        candidates.append(
            {
                "candidate_id": f"boltzgen_{index:05d}",
                "source_tool": "boltzgen",
                "stage": STAGE_GENERATION_BACKBONE_SEQUENCE,
                "target_pdb": _rel_path(run_dir, target_artifact),
                "complex_pdb": _rel_path(run_dir, structure_path),
                "binder_pdb": None,
                "binder_sequence": binder_sequence,
                "target_chains": target_chains,
                "binder_chains": binder_chains,
                "hotspots": [token for token in params.get("hotspots", "").split(",") if token],
                "binder_length": params.get("binder_length"),
                "contig": None,
                "metrics": {"target_chain_inference": chain_inference},
                "raw_metadata": {
                    "result_kind": "generation_only",
                    "metadata_path": _rel_path(run_dir, metadata_path),
                    "input_target_chains": params.get("target_chains", []),
                    "target_chain_inference": chain_inference,
                },
            }
        )
    return write_candidates(run_dir, "boltzgen", candidates)


def _boltzgen_metric_rows(run_dir: Path, budget: int) -> list[dict[str, str]]:
    ranked_dir = run_dir / "artifacts" / "raw" / "boltzgen" / "run-vanilla" / "final_ranked_designs"
    all_metrics = ranked_dir / "all_designs_metrics.csv"
    if all_metrics.exists():
        return _read_csv_rows(all_metrics)
    preferred = ranked_dir / f"final_designs_metrics_{budget}.csv"
    if preferred.exists():
        return _read_csv_rows(preferred)
    candidates = sorted(ranked_dir.glob("final_designs_metrics_*.csv"))
    return _read_csv_rows(candidates[0] if candidates else None)


def _boltzgen_final_design_path(run_dir: Path, budget: int, row: dict[str, str]) -> Path | None:
    ranked_dir = run_dir / "artifacts" / "raw" / "boltzgen" / "run-vanilla" / "final_ranked_designs"
    final_dirs = [ranked_dir / f"final_{budget}_designs", *sorted(ranked_dir.glob("final_*_designs"))]
    design_id = str(row.get("id") or "").strip()
    file_name = str(row.get("file_name") or "").strip()
    stems = [Path(file_name).stem] if file_name else []
    if design_id:
        stems.append(design_id)
    for final_dir in final_dirs:
        if not final_dir.exists():
            continue
        for stem in stems:
            for path in sorted(final_dir.glob(f"*{stem}*.cif")):
                if path.is_file():
                    return path
        if file_name:
            direct = final_dir / file_name
            if direct.exists():
                return direct
    return None


def _numeric_or_text(value: str) -> object:
    if value is None:
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return int(number)
    return number


def _normalize_boltzgen_vanilla_candidates(run_dir: Path, params: dict, target_artifact: Path) -> list[dict]:
    budget = int(params.get("budget") or 1)
    rows = _boltzgen_metric_rows(run_dir, budget)
    candidates: list[dict] = []
    for index, row in enumerate(rows, start=1):
        structure_path = _boltzgen_final_design_path(run_dir, budget, row)
        if structure_path is None:
            continue
        binder_chains, target_chains, chain_inference = _infer_generated_chain_roles(
            target_artifact,
            structure_path,
            params.get("target_chains", []),
        )
        sequences = _sequences_by_chain(structure_path)
        binder_sequence = (
            row.get("designed_sequence")
            or row.get("designed_chain_sequence")
            or row.get("sequence")
            or "".join(sequences.get(chain, "") for chain in binder_chains)
            or None
        )
        metrics = {key: _numeric_or_text(value) for key, value in row.items() if value not in {"", None}}
        metrics.update(
            {
                "complex_refolding_backend": "boltzgen_vanilla",
                "result_kind": "native_pipeline",
                "target_chain_inference": chain_inference,
            }
        )
        candidates.append(
            {
                "candidate_id": f"boltzgen_{index:05d}",
                "source_tool": "boltzgen",
                "stage": STAGE_COMPLEX_REFOLDING,
                "target_pdb": _rel_path(run_dir, target_artifact),
                "complex_pdb": _rel_path(run_dir, structure_path),
                "binder_pdb": None,
                "binder_sequence": binder_sequence,
                "target_chains": target_chains,
                "binder_chains": binder_chains,
                "hotspots": [token for token in params.get("hotspots", "").split(",") if token],
                "binder_length": params.get("binder_length"),
                "contig": None,
                "metrics": metrics,
                "raw_metadata": {
                    "result_kind": "native_pipeline",
                    "boltzgen_design_id": row.get("id"),
                    "boltzgen_file_name": row.get("file_name"),
                    "input_target_chains": params.get("target_chains", []),
                    "target_chain_inference": chain_inference,
                    "output_kind": "final_ranked_design",
                },
            }
        )
    return write_candidates(run_dir, "boltzgen", candidates)


def _pxdesign_summary_paths(run_dir: Path) -> list[Path]:
    raw_dir = run_dir / "artifacts" / "raw" / "pxdesign"
    paths = [
        *raw_dir.glob("output/design_outputs/*/summary.csv"),
        *raw_dir.glob("design_outputs/*/summary.csv"),
    ]
    return sorted(path for path in paths if path.is_file())


def _pxdesign_structure_path(summary_path: Path, row: dict[str, str]) -> Path | None:
    chosen = str(row.get("chosen_struct_path") or "").strip()
    if chosen:
        candidate = summary_path.parent / chosen
        if candidate.exists():
            return candidate
    rank = str(row.get("rank") or "").strip()
    if rank:
        for subdir in ["passing-Protenix-basic", "passing-AF2-IG-easy", "orig_designed"]:
            candidate = summary_path.parent / subdir / f"rank_{rank}.cif"
            if candidate.exists():
                return candidate
    for path in sorted(summary_path.parent.glob("**/*.cif")):
        if path.is_file():
            return path
    return None


def _pxdesign_bool(value: object) -> bool | None:
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _normalize_pxdesign_candidates(run_dir: Path, params: dict, target_artifact: Path) -> list[dict]:
    candidates: list[dict] = []
    for summary_path in _pxdesign_summary_paths(run_dir):
        rows = _read_csv_rows(summary_path)
        for row in rows:
            structure_path = _pxdesign_structure_path(summary_path, row)
            if structure_path is None:
                continue
            index = len(candidates) + 1
            binder_chains, target_chains, chain_inference = _infer_generated_chain_roles(
                target_artifact,
                structure_path,
                params.get("target_chains", []),
            )
            sequences = _sequences_by_chain(structure_path)
            binder_sequence = row.get("sequence") or "".join(sequences.get(chain, "") for chain in binder_chains) or None
            metrics = {key: _numeric_or_text(value) for key, value in row.items() if value not in {"", None}}
            success_values = [
                _pxdesign_bool(row.get("Protenix-success")),
                _pxdesign_bool(row.get("Protenix-basic-success")),
                _pxdesign_bool(row.get("AF2-IG-success")),
                _pxdesign_bool(row.get("AF2-IG-easy-success")),
            ]
            native_success = any(value is True for value in success_values) if any(value is not None for value in success_values) else None
            metrics.update(
                {
                    "complex_refolding_backend": f"pxdesign_{params.get('preset', 'preview')}",
                    "target_chain_inference": chain_inference,
                    "pass_filters": native_success,
                    "result_kind": "native_pipeline",
                    "binder_plddt": metrics.get("af2_plddt"),
                    "iptm": metrics.get("af2_iptm"),
                    "ipae": metrics.get("af2_ipAE"),
                    "binder_rmsd": metrics.get("af2_bound_unbound_RMSD"),
                }
            )
            candidates.append(
                {
                    "candidate_id": f"pxdesign_{index:05d}",
                    "source_tool": "pxdesign",
                    "stage": STAGE_COMPLEX_REFOLDING,
                    "target_pdb": _rel_path(run_dir, target_artifact),
                    "complex_pdb": _rel_path(run_dir, structure_path),
                    "binder_pdb": None,
                    "binder_sequence": binder_sequence,
                    "target_chains": target_chains,
                    "binder_chains": binder_chains,
                    "hotspots": [token for token in params.get("hotspots", "").split(",") if token],
                    "binder_length": params.get("binder_length"),
                    "contig": None,
                    "metrics": metrics,
                    "raw_metadata": {
                        "result_kind": "native_pipeline",
                        "summary_csv": _rel_path(run_dir, summary_path),
                        "chosen_struct_path": row.get("chosen_struct_path"),
                        "chosen_struct_type": row.get("chosen_struct_type"),
                        "input_target_chains": params.get("target_chains", []),
                        "target_chain_inference": chain_inference,
                    },
                }
            )
    return write_candidates(run_dir, "pxdesign", candidates)


def _normalize_pxdesign_generation_candidates(run_dir: Path, params: dict, target_artifact: Path) -> list[dict]:
    raw_dir = run_dir / "artifacts" / "raw" / "pxdesign" / "output"
    structure_paths = sorted(path for path in raw_dir.glob("**/*.cif") if path.is_file())
    candidates: list[dict] = []
    for index, structure_path in enumerate(structure_paths, start=1):
        binder_chains, target_chains, chain_inference = _infer_generated_chain_roles(
            target_artifact,
            structure_path,
            params.get("target_chains", []),
        )
        sequences = _sequences_by_chain(structure_path)
        binder_sequence = "".join(sequences.get(chain, "") for chain in binder_chains) or None
        candidates.append(
            {
                "candidate_id": f"pxdesign_{index:05d}",
                "source_tool": "pxdesign",
                "stage": STAGE_GENERATION_BACKBONE,
                "target_pdb": _rel_path(run_dir, target_artifact),
                "complex_pdb": _rel_path(run_dir, structure_path),
                "binder_pdb": None,
                "binder_sequence": binder_sequence,
                "target_chains": target_chains,
                "binder_chains": binder_chains,
                "hotspots": [token for token in params.get("hotspots", "").split(",") if token],
                "binder_length": params.get("binder_length"),
                "contig": None,
                "metrics": {
                    "target_chain_inference": chain_inference,
                    "pxdesign_mode": "generation_only",
                    "result_kind": "generation_only",
                },
                "raw_metadata": {
                    "result_kind": "generation_only",
                    "input_target_chains": params.get("target_chains", []),
                    "target_chain_inference": chain_inference,
                    "output_kind": "infer_prediction",
                },
            }
        )
    return write_candidates(run_dir, "pxdesign", candidates)


def _normalize_rfdiffusion3_contig(contig: str) -> str:
    normalized = contig.strip()
    if not normalized:
        return normalized
    if ",/0," in normalized:
        return normalized
    return re.sub(r"/0\s+", ",/0,", normalized)


def _cif_chain_lengths(path: Path) -> dict[str, int]:
    text = gzip.open(path, "rt", errors="ignore").read() if path.name.endswith(".gz") else path.read_text(errors="ignore")
    atom_headers: list[str] = []
    in_atom_loop = False
    residues: set[tuple[str, str, str]] = set()
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
        atom_name = row.get("auth_atom_id") or row.get("label_atom_id")
        if atom_name != "CA":
            continue
        chain = row.get("auth_asym_id") or row.get("label_asym_id") or "_"
        residue = row.get("auth_seq_id") or row.get("label_seq_id") or "0"
        insertion = row.get("pdbx_PDB_ins_code") or ""
        residues.add((chain, residue, insertion))
    chain_lengths: dict[str, int] = {}
    for chain, _, _ in residues:
        chain_lengths[chain] = chain_lengths.get(chain, 0) + 1
    return chain_lengths


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
    text = gzip.open(path, "rt", errors="ignore").read() if path.name.endswith(".gz") else path.read_text(errors="ignore")
    atom_headers: list[str] = []
    in_atom_loop = False
    residues: dict[str, list[tuple[int, str, str]]] = {}
    seen: set[tuple[str, str, str]] = set()
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
        atom_name = row.get("auth_atom_id") or row.get("label_atom_id")
        if atom_name != "CA":
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


def _target_chains_in_generated_structure(
    target_artifact: Path,
    generated_structure: Path,
    requested_target_chains: list[str],
    output_chains: list[str],
) -> list[str]:
    requested = [chain for chain in requested_target_chains if chain in output_chains]
    if not target_artifact.exists():
        return requested
    target_sequences = _pdb_sequences_by_chain(target_artifact)
    generated_sequences = _sequences_by_chain(generated_structure)
    target_chains: list[str] = []
    used_target_sequences: set[str] = set()
    for output_chain, output_sequence in generated_sequences.items():
        if not output_sequence:
            continue
        for target_chain, target_sequence in target_sequences.items():
            if target_chain in used_target_sequences or not target_sequence:
                continue
            if output_sequence == target_sequence:
                target_chains.append(output_chain)
                used_target_sequences.add(target_chain)
                break
    return target_chains or requested


def _chain_ids_from_structure(path: Path) -> list[str]:
    if path.suffix.lower() == ".cif" or path.name.endswith(".cif.gz"):
        return list(_cif_chain_lengths(path))
    return list(_pdb_sequences_by_chain(path))


def _infer_generated_chain_roles(target_artifact: Path, generated_structure: Path, requested_target_chains: list[str]) -> tuple[list[str], list[str], str]:
    output_chains = _chain_ids_from_structure(generated_structure)
    requested = [chain for chain in requested_target_chains if chain in output_chains]
    target_chains = _target_chains_in_generated_structure(
        target_artifact,
        generated_structure,
        requested_target_chains,
        output_chains,
    )
    inference = "sequence_match" if target_chains and target_artifact.exists() else "requested_chain_ids"
    if not target_chains:
        if "B" in output_chains:
            target_chains = ["B"]
            inference = "fallback_B"
        elif len(output_chains) > 1:
            target_chains = output_chains[1:]
            inference = "fallback_nonfirst"
    target_set = set(target_chains)
    binder_chains = [chain for chain in output_chains if chain not in target_set]
    if not binder_chains and output_chains:
        binder_chains = [output_chains[0]]
        target_chains = [chain for chain in output_chains if chain not in set(binder_chains)]
        inference = "fallback_first_binder"
    return binder_chains, target_chains, inference


def _normalize_rfdiffusion3_candidates(run_dir: Path, params: dict, target_artifact: Path) -> list[dict]:
    raw_dir = run_dir / "artifacts" / "raw" / "rfdiffusion3_foundry" / "rfd3"
    structure_paths = sorted([*raw_dir.glob("*.cif"), *raw_dir.glob("*.cif.gz")])
    metadata_paths = {path.stem.replace(".cif", ""): path for path in raw_dir.glob("*.json")}
    target_chains = params.get("target_chains", [])
    candidates: list[dict] = []
    for index, structure_path in enumerate(structure_paths, start=1):
        chain_lengths = _cif_chain_lengths(structure_path)
        output_chains = list(chain_lengths)
        binder_chains, inferred_target_chains, inference = _infer_generated_chain_roles(target_artifact, structure_path, target_chains)
        generated_binder_present = bool(binder_chains)
        metadata_path = metadata_paths.get(structure_path.stem.replace(".cif", ""))
        candidates.append(
            {
                "candidate_id": f"rfdiffusion3_foundry_{index:05d}",
                "source_tool": "rfdiffusion3_foundry",
                "stage": STAGE_GENERATION_BACKBONE,
                "target_pdb": _rel_path(run_dir, target_artifact),
                "complex_pdb": _rel_path(run_dir, structure_path),
                "binder_pdb": None,
                "binder_sequence": None,
                "target_chains": inferred_target_chains,
                "binder_chains": binder_chains,
                "hotspots": [token for token in params.get("hotspots", "").split(",") if token],
                "binder_length": params.get("binder_length"),
                "contig": params.get("contig"),
                "metrics": {
                    "chain_lengths": chain_lengths,
                    "generated_binder_present": generated_binder_present,
                },
                "raw_metadata": {
                    "metadata_path": _rel_path(run_dir, metadata_path),
                    "output_chains": output_chains,
                    "input_target_chains": target_chains,
                    "target_chain_inference": inference,
                    "normalization_warning": "" if generated_binder_present else "RFdiffusion3 output has no non-target binder chain.",
                },
            }
        )
    _write_candidates(run_dir, "rfdiffusion3_foundry", candidates)
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

    binder_chains, target_chains, chain_inference = _infer_generated_chain_roles(
        target_artifact,
        complex_pdb,
        params.get("target_chains", []),
    )
    binder_chain_for_metrics = binder_chains[0] if binder_chains else "A"
    target_chain_for_metrics = target_chains[0] if target_chains else "B"
    metrics = calculate_hotspot_metrics(
        complex_pdb,
        params.get("hotspots", ""),
        binder_chain=binder_chain_for_metrics,
        target_chain=target_chain_for_metrics,
        contact_cutoff=float(params.get("hotspot_contact_cutoff") or 8.0),
        contig=str(params.get("contig") or ""),
    )
    metrics["binder_chains"] = ",".join(binder_chains)
    metrics["target_chains"] = ",".join(target_chains)
    metrics["target_chain_inference"] = chain_inference
    if params.get("apply_backbone_hotspot_prefilter"):
        passed, failures = passes_hotspot_prefilter(
            metrics,
            min_contact_fraction=float(params.get("backbone_min_hotspot_contact_fraction") or 0.25),
            max_min_distance=float(params.get("backbone_max_hotspot_distance") or 10.0),
        )
        metrics["passes_backbone_hotspot_filter"] = passed
        metrics["backbone_hotspot_filter_failures"] = "; ".join(failures)
    else:
        metrics["passes_backbone_hotspot_filter"] = None
        metrics["backbone_hotspot_filter_failures"] = ""

    return {
        "candidate_id": candidate_id,
        "source_tool": "rfdiffusion_classic",
        "stage": STAGE_GENERATION_BACKBONE,
        "target_pdb": rel(target_artifact),
        "complex_pdb": rel(complex_pdb),
        "binder_pdb": None,
        "binder_sequence": None,
        "target_chains": target_chains,
        "binder_chains": binder_chains,
        "hotspots": [token for token in params.get("hotspots", "").split(",") if token],
        "binder_length": params.get("binder_length"),
        "contig": params.get("contig"),
        "metrics": metrics,
        "raw_metadata": {
            "trb_path": rel(trb_path),
            "input_target_chains": params.get("target_chains", []),
            "output_chains": _chain_ids_from_structure(complex_pdb),
            "target_chain_inference": chain_inference,
            "chain_convention": "dynamic_target_sequence_match_with_classic_fallback",
        },
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
    rfdiffusion_guidance_preset: str,
    backbone_filters: str,
    apply_backbone_hotspot_prefilter: bool,
    backbone_min_hotspot_contact_fraction: float,
    backbone_max_hotspot_distance: float,
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
        "rfdiffusion_guidance_preset": rfdiffusion_guidance_preset,
        "rfdiffusion_run_parameters": final_run_parameters,
        "hotspot": hotspots,
        "backbone_filters": backbone_filters.strip() or "none",
        "backbone_hotspot_prefilter": {
            "enabled": apply_backbone_hotspot_prefilter,
            "min_hotspot_contact_fraction": backbone_min_hotspot_contact_fraction,
            "max_min_binder_to_hotspot_distance": backbone_max_hotspot_distance,
            "contact_cutoff": 8.0,
        },
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


def _hydra_overrides_for_bash(overrides: str) -> str:
    return overrides.replace('"', '\\"')


def run_rfdiffusion_classic(
    target_pdb: Path,
    target_chains: list[str],
    contig: str,
    binder_length: str,
    hotspots: str = "",
    num_designs: int = 1,
    timesteps: int = 50,
    model_weights: str = "Complex_base",
    rfdiffusion_guidance_preset: str = "none",
    extra_run_parameters: str = "",
    partial_diffusion: bool = False,
    contigmap_length: str = "",
    inpaint_seq: str = "",
    backbone_filters: str = "",
    apply_backbone_hotspot_prefilter: bool = False,
    backbone_min_hotspot_contact_fraction: float = 0.25,
    backbone_max_hotspot_distance: float = 10.0,
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
        "rfdiffusion_guidance_preset": rfdiffusion_guidance_preset,
        "extra_run_parameters": extra_run_parameters.strip(),
        "partial_diffusion": partial_diffusion,
        "contigmap_length": contigmap_length.strip(),
        "inpaint_seq": inpaint_seq.strip(),
        "backbone_filters": backbone_filters.strip(),
        "apply_backbone_hotspot_prefilter": apply_backbone_hotspot_prefilter,
        "backbone_min_hotspot_contact_fraction": backbone_min_hotspot_contact_fraction,
        "backbone_max_hotspot_distance": backbone_max_hotspot_distance,
        "hotspot_contact_cutoff": 8.0,
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
        rfdiffusion_guidance_preset=rfdiffusion_guidance_preset,
        backbone_filters=backbone_filters,
        apply_backbone_hotspot_prefilter=apply_backbone_hotspot_prefilter,
        backbone_min_hotspot_contact_fraction=backbone_min_hotspot_contact_fraction,
        backbone_max_hotspot_distance=backbone_max_hotspot_distance,
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
    shell_run_parameters = _hydra_overrides_for_bash(final_run_parameters)
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
        f"{shell_run_parameters}; "
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
    campaign_name: str = "",
    number_of_final_designs: int = 1,
    num_seqs_override: int | None = None,
    max_mpnn_sequences_override: int | None = None,
    time_limit_seconds: int | None = None,
    enable_mpnn: bool = True,
    max_trajectories: int = 1,
    filter_settings: str = "default_filters.json",
    advanced_settings_file: str = "default_4stage_multimer_mpnn.json",
) -> Path:
    if not target_chains:
        raise ValueError("At least one target chain is required.")
    hotspots = normalize_hotspots(hotspots) if hotspots else ""
    parse_binder_lengths(binder_length)
    if number_of_final_designs < 1:
        raise ValueError("Desired final designs must be at least 1.")
    if num_seqs_override is not None and num_seqs_override < 1:
        raise ValueError("num_seqs must be at least 1.")
    if max_mpnn_sequences_override is not None and max_mpnn_sequences_override < 1:
        raise ValueError("max_mpnn_sequences must be at least 1.")
    if time_limit_seconds is not None and time_limit_seconds < 60:
        raise ValueError("Time limit should be at least 60 seconds.")
    if max_trajectories < 1:
        raise ValueError("Max trajectories must be at least 1.")
    advanced_settings_path = _ovo_bindcraft_resource("settings_advanced", advanced_settings_file)
    if not advanced_settings_path.exists():
        raise ValueError(f"Unknown BindCraft advanced settings file: {advanced_settings_file}")
    if filter_settings == "no_filters.json":
        filter_settings = "default_filters.json"

    manifest = load_manifest("bindcraft")
    params = {
        "target_chains": target_chains,
        "binder_length": binder_length,
        "hotspots": hotspots,
        "campaign_name": campaign_name.strip(),
        "number_of_final_designs": number_of_final_designs,
        "num_seqs_override": num_seqs_override,
        "max_mpnn_sequences_override": max_mpnn_sequences_override,
        "time_limit_seconds": time_limit_seconds,
        "enable_mpnn": enable_mpnn,
        "max_trajectories": max_trajectories,
        "filter_settings": filter_settings,
        "advanced_settings_file": advanced_settings_file,
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

    bindcraft_command = "python /content/bindcraft/bindcraft.py"
    if time_limit_seconds is not None:
        bindcraft_command = f"timeout {int(time_limit_seconds)} {bindcraft_command}"
    script = (
        "set -euxo pipefail; "
        "cd /work/artifacts/raw/bindcraft; "
        "ln -sf input_pdb.pdb target.pdb; "
        "ln -sfn /af_models alphafold_models_path; "
        "cp settings_advanced.json advanced.json; "
        f"{bindcraft_command} "
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
    candidates = _normalize_bindcraft_candidates(job.run_dir, params, target_artifact)
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
        bool(candidates),
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
    contig: str | None = None,
    is_non_loopy: bool = True,
    infer_ori_strategy: str = "",
) -> Path:
    hotspots = normalize_hotspots(hotspots) if hotspots else ""
    manifest = load_manifest("rfdiffusion3_foundry")
    final_contig = _normalize_rfdiffusion3_contig(contig) if contig and contig.strip() else ""
    params = {
        "target_chains": target_chains,
        "binder_length": binder_length,
        "hotspots": hotspots,
        "num_designs": num_designs,
        "timesteps": timesteps,
        "contig": final_contig,
        "is_non_loopy": is_non_loopy,
        "infer_ori_strategy": infer_ori_strategy.strip() or ("hotspots" if hotspots else "default"),
    }
    job = create_job(DESIGN_GROUP, "design_campaign", "rfdiffusion3_foundry", {"target_pdb": str(target_pdb), "target_chains": target_chains}, params)
    target_artifact = _copy_target_for_design(job.run_dir, target_pdb, target_chains, "raw/rfdiffusion3_foundry")
    if not final_contig:
        final_contig = _normalize_rfdiffusion3_contig(default_target_contig(target_artifact, target_chains, binder_length))
        params["contig"] = final_contig
        write_json(
            job.run_dir / "input.json",
            {
                "job_type": "design_campaign",
                "tool": "rfdiffusion3_foundry",
                "inputs": {"target_pdb": str(target_pdb), "target_chains": target_chains},
                "params": params,
            },
        )
    design_input = {
        "design_1": {
            "dialect": 2,
            "input": "target.pdb",
            "contig": final_contig,
            "is_non_loopy": is_non_loopy,
            "infer_ori_strategy": params["infer_ori_strategy"],
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
    candidates = _normalize_rfdiffusion3_candidates(job.run_dir, params, target_artifact) if rc == 0 else []
    _finish_design_job(job.run_dir, rc == 0, rc, candidates, [("raw/rfdiffusion3_foundry/rfd3/**/*", "rfdiffusion3_output"), ("raw/rfdiffusion3_foundry/*.json", "rfdiffusion3_input")])
    return job.run_dir


def run_boltzgen(
    target_pdb: Path,
    target_chains: list[str],
    binder_length: str,
    hotspots: str = "",
    campaign_name: str = "",
    num_designs: int = 1,
    budget: int = 1,
    sampling_steps: int = 20,
    run_vanilla_pipeline: bool = True,
) -> Path:
    hotspots = normalize_hotspots(hotspots) if hotspots else ""
    lengths = parse_binder_lengths(binder_length)
    binder_sequence_spec = str(lengths[0]) if len(lengths) == 1 else f"{lengths[0]}..{lengths[1]}"
    manifest = load_manifest("boltzgen")
    if budget < 1:
        raise ValueError("BoltzGen final design budget must be at least 1.")
    if budget > num_designs:
        budget = num_designs
    params = {
        "target_chains": target_chains,
        "binder_length": binder_length,
        "hotspots": hotspots,
        "campaign_name": campaign_name.strip(),
        "num_designs": num_designs,
        "budget": budget,
        "sampling_steps": sampling_steps,
        "run_vanilla_pipeline": run_vanilla_pipeline,
    }
    job = create_job(DESIGN_GROUP, "design_campaign", "boltzgen", {"target_pdb": str(target_pdb), "target_chains": target_chains}, params)
    raw_dir = job.run_dir / "artifacts" / "raw" / "boltzgen"
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_artifact = _copy_target_for_design(job.run_dir, target_pdb, target_chains, "raw/boltzgen/input")
    summary = pdb_summary(target_artifact.read_text(errors="ignore"))
    residues_by_chain = {row["chain_id"]: sorted(row.get("residues") or []) for row in summary.get("chains", [])}
    residue_index_by_chain = {
        chain_id: {residue: index for index, residue in enumerate(residues, start=1)}
        for chain_id, residues in residues_by_chain.items()
    }
    binding_by_chain: dict[str, list[str]] = {}
    for token in hotspots.split(",") if hotspots else []:
        if not token:
            continue
        chain_id = token[0]
        residue_number = int(token[1:])
        chain_local_index = residue_index_by_chain.get(chain_id, {}).get(residue_number)
        if chain_local_index is not None:
            binding_by_chain.setdefault(chain_id, []).append(str(chain_local_index))
    spec = [
        "entities:",
        "  - protein:",
        "      id: B",
        f"      sequence: {binder_sequence_spec}",
        "  - file:",
        "      path: target.pdb",
        "      include:",
    ]
    for chain_id in target_chains or ["A"]:
        spec.extend(["        - chain:", f"            id: {chain_id}"])
    if binding_by_chain:
        spec.append("      binding_types:")
        for chain_id in sorted(binding_by_chain):
            spec.extend(["        - chain:", f"            id: {chain_id}", f"            binding: {','.join(binding_by_chain[chain_id])}"])
    spec.append('      structure_groups: "all"')
    (raw_dir / "input" / "target_binder.yaml").write_text("\n".join(spec) + "\n")
    output_dir = "run-vanilla" if run_vanilla_pipeline else "run-generation-only"
    run_args = (
        f"boltzgen run input/target_binder.yaml --output {output_dir} --protocol protein-anything "
        f"--num_designs {num_designs} --diffusion_batch_size 1 --devices 1 --num_workers 1 --cache /cache "
    )
    if run_vanilla_pipeline:
        run_args += f"--budget {budget} "
    else:
        run_args += "--steps design "
    run_args += (
        f"--config design sampling_steps={sampling_steps} "
        "compile_pairformer=false compile_structure=false data.num_workers=1"
    )
    output_check = (
        f"test -n \"$(find {output_dir}/final_ranked_designs -type f -path '*/final_*_designs/*.cif' -print -quit)\""
        if run_vanilla_pipeline
        else f"test -n \"$(find {output_dir}/intermediate_designs -maxdepth 1 -type f -name '*.cif' -print -quit)\""
    )
    script = (
        "set -euxo pipefail; cd /work/artifacts/raw/boltzgen; "
        f"boltzgen check input/target_binder.yaml --output checked --cache /cache; rm -rf {output_dir}; "
        f"{run_args}; "
        f"{output_check}"
    )
    command = ["docker", "run", "--rm", "--gpus", "all", "--entrypoint", "/bin/bash", "-v", "/mnt/db/reference_files/boltzgen-cache:/cache", "-v", f"{job.run_dir}:/work", "-w", "/work", manifest["image"], "-lc", script]
    rc = _run_shell_steps(job.run_dir, [{"name": "boltzgen", "command": command}])
    if rc == 0 and run_vanilla_pipeline:
        candidates = _normalize_boltzgen_vanilla_candidates(job.run_dir, params, target_artifact)
    elif rc == 0:
        candidates = _normalize_boltzgen_generation_candidates(job.run_dir, params, target_artifact)
    else:
        candidates = []
    _finish_design_job(
        job.run_dir,
        rc == 0 and bool(candidates),
        rc,
        candidates,
        [
            ("raw/boltzgen/**/*.cif", "cif"),
            ("raw/boltzgen/**/*.npz", "npz"),
            ("raw/boltzgen/**/*.csv", "csv"),
            ("raw/boltzgen/**/*.pdf", "pdf"),
            ("raw/boltzgen/**/*.yaml", "yaml"),
        ],
    )
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


def run_pxdesign(
    target_pdb: Path,
    target_chains: list[str],
    binder_length: str,
    hotspots: str = "",
    campaign_name: str = "",
    num_designs: int = 1,
    n_steps: int = 400,
    dtype: str = "bf16",
    preset: str = "preview",
    run_mode: str = "pipeline",
    n_max_runs: int = 1,
    use_fast_ln: bool = True,
    use_deepspeed_evo_attention: bool = False,
) -> Path:
    hotspots = normalize_hotspots(hotspots) if hotspots else ""
    lengths = parse_binder_lengths(binder_length)
    px_binder_length = int(lengths[0])
    manifest = load_manifest("pxdesign")
    params = {
        "target_chains": target_chains,
        "binder_length": binder_length,
        "pxdesign_binder_length": px_binder_length,
        "hotspots": hotspots,
        "campaign_name": campaign_name.strip(),
        "num_designs": num_designs,
        "n_steps": n_steps,
        "dtype": dtype,
        "preset": preset,
        "run_mode": run_mode,
        "n_max_runs": n_max_runs,
        "use_fast_ln": use_fast_ln,
        "use_deepspeed_evo_attention": use_deepspeed_evo_attention,
    }
    job = create_job(DESIGN_GROUP, "design_campaign", "pxdesign", {"target_pdb": str(target_pdb), "target_chains": target_chains}, params)
    if campaign_name.strip():
        update_status(job.run_dir, "queued", campaign_name=campaign_name.strip())
    raw_dir = job.run_dir / "artifacts" / "raw" / "pxdesign"
    input_dir = raw_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_artifact = _copy_target_for_design(job.run_dir, target_pdb, target_chains, "raw/pxdesign/input")
    hotspots_by_chain: dict[str, list[int]] = {}
    for token in hotspots.split(",") if hotspots else []:
        if not token:
            continue
        try:
            hotspots_by_chain.setdefault(token[0], []).append(int(token[1:]))
        except ValueError:
            continue
    yaml_lines = [
        "target:",
        "  file: input/target.pdb",
        "  chains:",
    ]
    for chain_id in target_chains or ["A"]:
        chain_hotspots = hotspots_by_chain.get(chain_id, [])
        if chain_hotspots:
            yaml_lines.extend(
                [
                    f"    {chain_id}:",
                    f"      hotspots: [{', '.join(str(item) for item in chain_hotspots)}]",
                ]
            )
        else:
            yaml_lines.append(f"    {chain_id}: all")
    yaml_lines.extend(["", f"binder_length: {px_binder_length}"])
    (input_dir / "target_binder.yaml").write_text("\n".join(yaml_lines) + "\n")
    output_dir = "output"
    common_prefix = (
        "set -euxo pipefail; cd /work/artifacts/raw/pxdesign; "
        "export PROTENIX_DATA_ROOT_DIR=/ref/pxdesign/release_data/ccd_cache; "
        "export TOOL_WEIGHTS_ROOT=/ref/pxdesign/tool_weights; "
        "pxdesign check-input --yaml input/target_binder.yaml; "
        "rm -rf parsed_target output; "
        "pxdesign parse-target --yaml input/target_binder.yaml -o parsed_target; "
    )
    if run_mode == "generation_only":
        script = (
            common_prefix +
            f"pxdesign infer -i input/target_binder.yaml -o {output_dir} "
            f"--N_sample {int(num_designs)} "
            f"--N_step {int(n_steps)} "
            f"--dtype {shlex.quote(dtype)} "
            f"--use_fast_ln {str(bool(use_fast_ln))} "
            f"--use_deepspeed_evo_attention {str(bool(use_deepspeed_evo_attention))} "
            "--load_checkpoint_dir /ref/pxdesign/release_data/checkpoint; "
            'test -n "$(find output -type f -name \"*.cif\" -print -quit)"'
        )
    else:
        script = (
            common_prefix +
            f"pxdesign pipeline --preset {shlex.quote(preset)} "
            "-i input/target_binder.yaml "
            f"-o {output_dir} "
            f"--N_sample {int(num_designs)} "
            f"--N_step {int(n_steps)} "
            f"--N_max_runs {int(n_max_runs)} "
            f"--dtype {shlex.quote(dtype)} "
            f"--use_fast_ln {str(bool(use_fast_ln))} "
            f"--use_deepspeed_evo_attention {str(bool(use_deepspeed_evo_attention))} "
            "--load_checkpoint_dir /ref/pxdesign/release_data/checkpoint; "
            "test -s output/design_outputs/target_binder/summary.csv"
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
        "/mnt/db/reference_files/pxdesign:/ref/pxdesign",
        "-w",
        "/work",
        manifest["image"],
        "bash",
        "-lc",
        script,
    ]
    rc = _run_shell_steps(job.run_dir, [{"name": "pxdesign", "command": command}])
    if rc == 0 and run_mode == "generation_only":
        candidates = _normalize_pxdesign_generation_candidates(job.run_dir, params, target_artifact)
    elif rc == 0:
        candidates = _normalize_pxdesign_candidates(job.run_dir, params, target_artifact)
    else:
        candidates = []
    _finish_design_job(
        job.run_dir,
        rc == 0 and bool(candidates),
        rc,
        candidates,
        [
            ("raw/pxdesign/**/*.cif", "cif"),
            ("raw/pxdesign/**/*.pdb", "pdb"),
            ("raw/pxdesign/**/*.csv", "csv"),
            ("raw/pxdesign/**/*.json", "json"),
            ("raw/pxdesign/**/*.yaml", "yaml"),
            ("raw/pxdesign/**/*.png", "plot"),
        ],
    )
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
