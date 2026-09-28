from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from mn_protein_design.runtime import runs_root


ENGINE_LABELS = {
    "alphafast_af3": "AlphaFast AF3",
    "alphafast_msa": "AlphaFast MMseqs MSA",
    "colabfold": "ColabFold",
    "af2_initial_guess": "AF2 initial guess",
    "boltz2_initial_guess": "Boltz-2",
    "esmfold2": "ESMFold2",
    "openfold3": "OpenFold-3",
    "rf3": "RF3",
    "protenix": "Protenix",
    "protenix_v1": "Protenix v1",
    "protenix_v2": "Protenix v2",
    "boltzgen_fold": "BoltzGen target-template fold",
    "postprocessing": "Metric postprocessing",
    "design_campaign": "Design campaign",
    "rfdiffusion_classic": "RFdiffusion classic",
    "bindcraft": "BindCraft",
    "rfdiffusion3_foundry": "RFdiffusion3 / Foundry",
    "boltzgen": "BoltzGen",
    "pxdesign": "PXDesign",
    "genie3": "Genie3",
    "esmfold2_binder_design": "ESMFold2 binder design",
    "protpardelle_1c": "Protpardelle-1c",
    "proteina_complexa": "Proteina-Complexa",
    "protein_mpnn": "ProteinMPNN",
    "ligand_mpnn": "LigandMPNN",
    "soluble_mpnn": "Soluble ProteinMPNN",
    "scannet": "ScanNet",
    "surf2spot": "Surf2Spot",
    "masif_seed": "MaSIF-seed",
    "pesto": "PeSTo",
}

FALLBACK_SECONDS_PER_CANDIDATE = {
    "esmfold2": 4.0,
    "af2_initial_guess": 18.0,
    "boltz2_initial_guess": 21.0,
    "colabfold": 35.0,
    "alphafast_af3": 55.0,
    "alphafast_msa": 15.0,
    "postprocessing": 8.0,
    "rf3": 30.0,
    "openfold3": 75.0,
    "protenix": 45.0,
    "protenix_v1": 75.0,
    "protenix_v2": 90.0,
    "boltzgen_fold": 25.0,
    "rfdiffusion_classic": 180.0,
    "bindcraft": 1800.0,
    "rfdiffusion3_foundry": 240.0,
    "boltzgen": 300.0,
    "pxdesign": 240.0,
    "genie3": 180.0,
    "esmfold2_binder_design": 1800.0,
    "protpardelle_1c": 300.0,
    "proteina_complexa": 300.0,
    "protein_mpnn": 10.0,
    "ligand_mpnn": 12.0,
    "soluble_mpnn": 10.0,
    "scannet": 300.0,
    "surf2spot": 180.0,
    "masif_seed": 600.0,
    "pesto": 90.0,
    "design_campaign": 600.0,
}

ESMFOLD2_BASELINE_LOOPS = 10
ESMFOLD2_BASELINE_SAMPLING_STEPS = 68

TOOL_TO_ENGINE = {
    "esmfold2_benchmark": "esmfold2",
    "esmfold2_complex_validation": "esmfold2",
    "esmfold2_initial_guess_validation": "esmfold2",
    "af2_initial_guess": "af2_initial_guess",
    "boltz2_initial_guess": "boltz2_initial_guess",
    "rf3": "rf3",
    "openfold3": "openfold3",
    "protenix": "protenix",
    "protenix_v1": "protenix_v1",
    "protenix_v2": "protenix_v2",
    "boltzgen_fold": "boltzgen_fold",
    "design_campaign": "design_campaign",
    "multi_engine_design_campaign": "design_campaign",
    "sequence_design": "protein_mpnn",
    "scannet": "scannet",
    "surf2spot": "surf2spot",
    "masif_seed": "masif_seed",
    "pesto": "pesto",
}


@dataclass(frozen=True)
class RuntimeObservation:
    engine: str
    seconds: float
    candidate_count: int
    total_residues: int | None = None
    run_id: str = ""
    source: str = "metadata"

    @property
    def seconds_per_candidate(self) -> float | None:
        if self.candidate_count <= 0:
            return None
        return self.seconds / float(self.candidate_count)

    @property
    def seconds_per_residue(self) -> float | None:
        if not self.total_residues or self.total_residues <= 0:
            return None
        return self.seconds / float(self.total_residues)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    seconds = max(0, int(round(seconds)))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if minutes or hours or days:
        parts.append(f"{minutes}m")
    if not parts:
        parts.append(f"{secs}s")
    return " ".join(parts)


def collect_runtime_observations(root: Path | None = None) -> list[RuntimeObservation]:
    root = root or runs_root()
    observations: list[RuntimeObservation] = []
    if not root.exists():
        return observations
    for metadata_path in root.glob("*/*/metadata.json"):
        run_dir = metadata_path.parent
        metadata = _read_json(metadata_path)
        if metadata.get("status") != "completed":
            continue
        result = _read_json(run_dir / "result.json")
        run_id = str(metadata.get("run_id") or run_dir.name)
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        runtime_timings = metrics.get("runtime_engine_timings")
        if isinstance(runtime_timings, dict):
            for engine, timing in runtime_timings.items():
                if not isinstance(timing, dict):
                    continue
                seconds = _float_or_none(timing.get("seconds"))
                candidate_count = _int_or_none(timing.get("candidate_count"))
                if seconds is None or candidate_count is None or candidate_count <= 0:
                    continue
                observations.append(
                    RuntimeObservation(
                        engine=str(engine),
                        seconds=seconds,
                        candidate_count=candidate_count,
                        total_residues=_int_or_none(timing.get("total_residues")),
                        run_id=run_id,
                        source="runtime_engine_timings",
                    )
                )
        engine = TOOL_TO_ENGINE.get(str(metadata.get("tool") or ""))
        if not engine:
            continue
        duration = _metadata_duration_seconds(metadata)
        candidate_count = _candidate_count(result)
        if duration is None or not candidate_count:
            continue
        total_residues = _int_or_none(metrics.get("total_residues") or metrics.get("total_length"))
        observations.append(
            RuntimeObservation(
                engine=engine,
                seconds=duration,
                candidate_count=candidate_count,
                total_residues=total_residues,
                run_id=run_id,
                source="job_metadata",
            )
        )
    return observations


def estimate_engines(
    *,
    engines: list[str],
    candidate_count: int,
    total_residues: int | None = None,
    engine_params: dict[str, dict[str, Any]] | None = None,
    observations: list[RuntimeObservation] | None = None,
) -> dict[str, Any]:
    observations = observations if observations is not None else collect_runtime_observations()
    engine_params = engine_params or {}
    candidate_count = max(0, int(candidate_count or 0))
    rows: list[dict[str, Any]] = []
    total_seconds = 0.0
    for engine in engines:
        engine_obs = [obs for obs in observations if obs.engine == engine and obs.candidate_count > 0]
        residue_rates = [obs.seconds_per_residue for obs in engine_obs if obs.seconds_per_residue]
        candidate_rates = [obs.seconds_per_candidate for obs in engine_obs if obs.seconds_per_candidate]
        basis = "fallback"
        if total_residues and residue_rates:
            rate = statistics.median(residue_rates)
            estimated_seconds = float(total_residues) * rate
            basis = "history/residue"
        elif candidate_rates:
            rate = statistics.median(candidate_rates)
            estimated_seconds = float(candidate_count) * rate
            basis = "history/candidate"
        else:
            rate = FALLBACK_SECONDS_PER_CANDIDATE.get(engine, 30.0)
            estimated_seconds = float(candidate_count) * rate
        multiplier = _engine_runtime_multiplier(engine, engine_params.get(engine) or {})
        if multiplier != 1.0:
            estimated_seconds *= multiplier
            basis = f"{basis} x{multiplier:.2f}"
        total_seconds += estimated_seconds
        rows.append(
            {
                "engine": engine,
                "label": ENGINE_LABELS.get(engine, engine),
                "estimated_seconds": estimated_seconds,
                "estimated_time": format_duration(estimated_seconds),
                "basis": basis,
                "history_runs": len(engine_obs),
                "seconds_per_candidate": estimated_seconds / candidate_count if candidate_count else None,
                "seconds_per_residue": estimated_seconds / total_residues if total_residues else None,
            }
        )
    return {
        "candidate_count": candidate_count,
        "total_residues": total_residues,
        "total_seconds": total_seconds,
        "total_time": format_duration(total_seconds),
        "rows": rows,
    }


def estimate_job_runtime(row: dict[str, Any], input_payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return estimate_job_runtime_with_observations(row, input_payload=input_payload, observations=None)


def estimate_job_runtime_with_observations(
    row: dict[str, Any],
    input_payload: dict[str, Any] | None = None,
    observations: list[RuntimeObservation] | None = None,
) -> dict[str, Any]:
    input_payload = input_payload or {}
    params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
    tool = str(row.get("tool") or input_payload.get("tool") or "")
    job_type = str(row.get("job_type") or input_payload.get("job_type") or "")
    if job_type == "refolding_evaluation" or tool in {"refolding_evaluation_engines", "target_refolding_evaluation_engines"}:
        engines = _refolding_engine_keys(params)
        candidate_count = (
            _int_or_none(params.get("candidate_count"))
            or _int_or_none(params.get("selected_candidate_count"))
            or _int_or_none(params.get("max_candidates"))
            or 1
        )
        total_residues = _int_or_none(params.get("total_residues"))
        engine_params = _refolding_engine_params(params, engines)
        estimate = estimate_engines(
            engines=engines or [tool or job_type],
            candidate_count=max(1, candidate_count),
            total_residues=total_residues,
            engine_params=engine_params,
            observations=observations,
        )
        return {
            "estimated_seconds": estimate.get("total_seconds"),
            "estimated_time": estimate.get("total_time", "n/a"),
            "basis": "refolding engine fallbacks/history",
        }
    if tool == "design_campaign" or job_type == "multi_engine_design_campaign":
        engines = [str(engine) for engine in params.get("engines") or []]
        template_enabled = bool((params.get("template_redesign") or {}).get("enabled")) if isinstance(params.get("template_redesign"), dict) else False
        if template_enabled:
            engines = ["protein_mpnn", *engines]
        attempts = max(1, _int_or_none(params.get("design_attempts")) or 1)
        sequences = max(1, _int_or_none(params.get("sequences_per_backbone")) or 1)
        common_validation = params.get("common_validation") if isinstance(params.get("common_validation"), dict) else {}
        evaluation = params.get("evaluation") if isinstance(params.get("evaluation"), dict) else {}
        estimate = estimate_engines(engines=engines, candidate_count=attempts * sequences, observations=observations)
        extra_seconds = 0.0
        if bool(common_validation.get("enabled", True)):
            extra_seconds += attempts * max(1, len(engines)) * FALLBACK_SECONDS_PER_CANDIDATE["af2_initial_guess"]
        if str(evaluation.get("mode") or "none") != "none":
            extra_seconds += attempts * max(1, len(engines)) * FALLBACK_SECONDS_PER_CANDIDATE["postprocessing"]
        total_seconds = float(estimate["total_seconds"]) + extra_seconds
        return {
            "estimated_seconds": total_seconds,
            "estimated_time": format_duration(total_seconds),
            "basis": "campaign engine fallbacks/history",
        }
    engine = TOOL_TO_ENGINE.get(tool) or TOOL_TO_ENGINE.get(job_type) or tool
    candidate_count = (
        _int_or_none(params.get("candidate_count"))
        or _int_or_none(params.get("num_designs"))
        or _int_or_none(params.get("num_samples"))
        or _int_or_none(params.get("num_seq_per_target"))
        or _int_or_none(params.get("design_attempts"))
        or 1
    )
    estimate = estimate_engines(
        engines=[engine],
        candidate_count=max(1, candidate_count),
        engine_params={engine: params},
        observations=observations,
    )
    row_estimate = estimate["rows"][0] if estimate["rows"] else {}
    return {
        "estimated_seconds": row_estimate.get("estimated_seconds"),
        "estimated_time": row_estimate.get("estimated_time", "n/a"),
        "basis": row_estimate.get("basis", "fallback"),
    }


def _refolding_engine_keys(params: dict[str, Any]) -> list[str]:
    engines: list[str] = []
    if bool(params.get("run_alphafast_af3")):
        engines.append("alphafast_af3")
    if bool(params.get("run_colabfold")):
        engines.append("colabfold")
    if bool(params.get("run_af2_initial_guess")):
        engines.append("af2_initial_guess")
    if bool(params.get("run_boltz2_initial_guess")):
        engines.append("boltz2_initial_guess")
    if bool(params.get("run_esmfold2")):
        engines.append("esmfold2")
    if bool(params.get("run_rf3")):
        engines.append("rf3")
    if bool(params.get("run_openfold3")):
        engines.append("openfold3")
    if bool(params.get("run_protenix")):
        engines.append("protenix")
    if bool(params.get("run_protenix_v1")):
        engines.append("protenix_v1")
    if bool(params.get("run_protenix_v2")):
        engines.append("protenix_v2")
    if bool(params.get("run_boltzgen_fold")):
        engines.append("boltzgen_fold")
    msa_needed = (
        bool(params.get("run_alphafast_af3"))
        or bool(params.get("run_colabfold"))
        or (bool(params.get("run_boltz2_initial_guess")) and bool(params.get("boltz2_use_target_msa", True)))
        or (bool(params.get("run_esmfold2")) and bool(params.get("esmfold2_use_target_msa")))
        or (bool(params.get("run_rf3")) and bool(params.get("rf3_use_target_msa")))
        or (bool(params.get("run_openfold3")) and bool(params.get("openfold3_use_target_msa")))
        or (bool(params.get("run_protenix")) and bool(params.get("protenix_use_msa", True)))
        or (bool(params.get("run_protenix_v1")) and bool(params.get("protenix_v1_use_msa", True)))
        or (bool(params.get("run_protenix_v2")) and bool(params.get("protenix_v2_use_msa", True)))
    )
    msa_source = str(params.get("shared_msa_source") or params.get("colabfold_msa_source") or "")
    query_only = bool(params.get("alphafast_query_only_msa"))
    if msa_needed and not query_only and ("alphafast_mmseqs_gpu" in msa_source or not msa_source):
        engines.insert(0, "alphafast_msa")
    if (
        bool(params.get("run_common_interface_metrics"))
        or bool(params.get("run_predicted_rosetta_metrics"))
        or bool(params.get("run_pymol_metrics"))
        or bool(params.get("run_pyrosetta_input_metrics"))
    ):
        engines.append("postprocessing")
    return engines


def _refolding_engine_params(params: dict[str, Any], engines: list[str]) -> dict[str, dict[str, Any]]:
    engine_params: dict[str, dict[str, Any]] = {}
    if "esmfold2" in engines:
        engine_params["esmfold2"] = {
            "num_loops": params.get("num_loops"),
            "num_sampling_steps": params.get("num_sampling_steps"),
        }
    return engine_params


def _engine_runtime_multiplier(engine: str, params: dict[str, Any]) -> float:
    if engine != "esmfold2":
        return 1.0
    loops = _int_or_none(params.get("num_loops"))
    steps = _int_or_none(params.get("num_sampling_steps"))
    if not loops and not steps:
        return 1.0
    loop_ratio = float(loops or ESMFOLD2_BASELINE_LOOPS) / float(ESMFOLD2_BASELINE_LOOPS)
    step_ratio = float(steps or ESMFOLD2_BASELINE_SAMPLING_STEPS) / float(ESMFOLD2_BASELINE_SAMPLING_STEPS)
    # ESMFold2 runtime is split between recurrent pair folding loops and diffusion sampling.
    # This weighted model keeps the paper default (10 loops / 68 steps) at 1.0 while making
    # preset changes visible in rough runtime estimates.
    return max(0.05, (0.65 * loop_ratio) + (0.35 * step_ratio))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _metadata_duration_seconds(metadata: dict[str, Any]) -> float | None:
    created = str(metadata.get("created_at") or "")
    completed = str(metadata.get("completed_at") or metadata.get("updated_at") or "")
    if not created or not completed:
        return None
    try:
        start = datetime.fromisoformat(created)
        end = datetime.fromisoformat(completed)
    except ValueError:
        return None
    return max(0.0, (end - start).total_seconds())


def _candidate_count(result: dict[str, Any]) -> int | None:
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
    for key in ("candidate_count", "record_count", "colabfold_selected_count"):
        value = _int_or_none(metrics.get(key))
        if value:
            return value
    candidates = outputs.get("candidates")
    if isinstance(candidates, list):
        return len(candidates)
    return None


def _float_or_none(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: object) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None
