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
    "rf3": "RF3",
    "protenix": "Protenix",
    "boltzgen_fold": "BoltzGen target-template fold",
    "postprocessing": "Metric postprocessing",
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
    "protenix": 45.0,
    "boltzgen_fold": 25.0,
}

TOOL_TO_ENGINE = {
    "esmfold2_benchmark": "esmfold2",
    "esmfold2_complex_validation": "esmfold2",
    "esmfold2_initial_guess_validation": "esmfold2",
    "af2_initial_guess": "af2_initial_guess",
    "boltz2_initial_guess": "boltz2_initial_guess",
    "rf3": "rf3",
    "protenix": "protenix",
    "boltzgen_fold": "boltzgen_fold",
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
        observations.append(
            RuntimeObservation(
                engine=engine,
                seconds=duration,
                candidate_count=candidate_count,
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
    observations: list[RuntimeObservation] | None = None,
) -> dict[str, Any]:
    observations = observations if observations is not None else collect_runtime_observations()
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
            }
        )
    return {
        "candidate_count": candidate_count,
        "total_residues": total_residues,
        "total_seconds": total_seconds,
        "total_time": format_duration(total_seconds),
        "rows": rows,
    }


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
