"""Local automation services backed by the app's normal job and workflow code.

This module is the supported Python boundary for local scripts. It deliberately
delegates campaign creation, scheduling, cancellation, and result parsing to the
same functions used by the Streamlit app.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from mn_protein_design.core import jobs
from mn_protein_design.core.candidates import campaign_result_path, read_candidates
from mn_protein_design.core.portable_paths import resolve_stored_path
from mn_protein_design.core.structures import pdb_summary
from mn_protein_design.workflows.design import normalize_hotspots, parse_binder_lengths
from mn_protein_design.workflows.design_campaigns import (
    ENGINE_ORDER,
    GENERATOR_ONLY_RECIPES,
    VANILLA_ONLY_ENGINES,
    create_design_campaign,
)


REQUEST_SCHEMA_VERSION = 1
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled", "stopped"})
ACTIVE_STATUSES = frozenset({"queued", "preparing", "running", "paused", "holding"})
_REQUEST_KEYS = frozenset(
    {
        "schema_version",
        "campaign_name",
        "target_pdb",
        "target_chains",
        "binder_length",
        "hotspots",
        "design_attempts",
        "sequences_per_backbone",
        "random_seed",
        "engines",
        "engine_configs",
        "workflow_recipe",
        "gpu_device",
        "survivors_per_engine",
        "passing_only",
        "keep_best_failed",
        "continue_after_failure",
        "common_validation",
        "sequence_refinement",
        "template_redesign",
        "evaluation",
    }
)


class AutomationError(ValueError):
    """An actionable request or job-reference error for local automation."""


def _positive_int(value: object, name: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise AutomationError(f"{name} must be an integer greater than or equal to {minimum}.")
    return value


def _bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise AutomationError(f"{name} must be true or false.")
    return value


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AutomationError(f"{name} must be a JSON object.")
    return value


def _resolve_target(value: object, request_file: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise AutomationError("target_pdb must be a non-empty path or managed app URI.")
    raw = value.strip()
    if raw.startswith(("app://", "reference://", "runs://")):
        resolved = resolve_stored_path(raw, must_exist=True)
        if resolved is None or not resolved.is_file():
            raise AutomationError(f"Managed target path does not exist: {raw}")
        return resolved.expanduser().resolve()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = request_file.parent / path
    path = path.resolve()
    if not path.is_file():
        raise AutomationError(f"Target PDB does not exist: {path}")
    return path


def validate_campaign_request(payload: object, *, request_file: Path) -> dict[str, Any]:
    """Validate and normalize a versioned campaign request without submitting it."""
    request = _object(payload, "request")
    unknown = sorted(set(request) - _REQUEST_KEYS)
    if unknown:
        raise AutomationError(f"Unsupported request field(s): {', '.join(unknown)}")
    version = request.get("schema_version")
    if type(version) is not int or version != REQUEST_SCHEMA_VERSION:
        raise AutomationError(f"schema_version must be {REQUEST_SCHEMA_VERSION}.")

    target_pdb = _resolve_target(request.get("target_pdb"), request_file)
    summary = pdb_summary(target_pdb.read_text(errors="ignore"))
    available_residues = {
        str(row["chain_id"]): {int(residue) for residue in row.get("residues", [])}
        for row in summary.get("chains", [])
    }
    if not available_residues or not summary.get("atom_count"):
        raise AutomationError(f"Target does not contain PDB ATOM records with chain IDs: {target_pdb}")

    chains_value = request.get("target_chains")
    if not isinstance(chains_value, list) or not chains_value:
        raise AutomationError("target_chains must be a non-empty list of PDB chain IDs.")
    if any(not isinstance(chain, str) or len(chain.strip()) != 1 for chain in chains_value):
        raise AutomationError("Each target chain must be a single-character PDB chain ID.")
    target_chains = [chain.strip() for chain in chains_value]
    if len(set(target_chains)) != len(target_chains):
        raise AutomationError("target_chains must not contain duplicates.")
    missing_chains = [chain for chain in target_chains if chain not in available_residues]
    if missing_chains:
        raise AutomationError(f"Target chain(s) absent from PDB: {', '.join(missing_chains)}")

    binder_length = request.get("binder_length", "60")
    if not isinstance(binder_length, (str, int)) or isinstance(binder_length, bool):
        raise AutomationError("binder_length must be a length or min-max range, such as '60-100'.")
    binder_length = str(binder_length).strip()
    try:
        parse_binder_lengths(binder_length)
    except (ValueError, TypeError) as exc:
        raise AutomationError(f"Invalid binder_length: {exc}") from exc

    hotspots_value = request.get("hotspots", "")
    if not isinstance(hotspots_value, (str, list)):
        raise AutomationError("hotspots must be a string or list such as 'A56,A59'.")
    if isinstance(hotspots_value, list):
        if any(not isinstance(item, str) for item in hotspots_value):
            raise AutomationError("Each hotspot must be a residue tag such as 'A56'.")
        hotspots_value = ",".join(hotspots_value)
    try:
        hotspots = normalize_hotspots(hotspots_value) if hotspots_value.strip() else ""
    except ValueError as exc:
        raise AutomationError(str(exc)) from exc
    for hotspot in filter(None, hotspots.split(",")):
        chain, residue_text = hotspot[0], hotspot[1:]
        if chain not in target_chains:
            raise AutomationError(f"Hotspot {hotspot} is not in the selected target chains.")
        if int(residue_text) not in available_residues[chain]:
            raise AutomationError(f"Hotspot {hotspot} does not exist in the target PDB.")

    engines_value = request.get("engines")
    if not isinstance(engines_value, list) or not engines_value:
        raise AutomationError("engines must be a non-empty list of workflow IDs.")
    if any(not isinstance(engine, str) for engine in engines_value):
        raise AutomationError("Every engine must be a workflow ID string.")
    engines = list(dict.fromkeys(engines_value))
    unknown_engines = sorted(set(engines) - (set(ENGINE_ORDER) - {"template_redesign"}))
    if unknown_engines:
        raise AutomationError(f"Unknown campaign engine(s): {', '.join(unknown_engines)}")

    workflow_recipe = request.get("workflow_recipe", "vanilla")
    if not isinstance(workflow_recipe, str) or workflow_recipe not in {"vanilla", *GENERATOR_ONLY_RECIPES}:
        raise AutomationError("workflow_recipe must be 'vanilla', 'staged', or 'engine_scout'.")
    unsupported = sorted(set(engines).intersection(VANILLA_ONLY_ENGINES))
    if workflow_recipe in GENERATOR_ONLY_RECIPES and unsupported:
        raise AutomationError(
            f"{', '.join(unsupported)} is available only in the vanilla campaign workflow."
        )

    configs = _object(request.get("engine_configs", {}), "engine_configs")
    unused_configs = sorted(set(configs) - set(engines))
    if unused_configs:
        raise AutomationError(f"engine_configs contains unselected engine(s): {', '.join(unused_configs)}")
    if any(not isinstance(config, dict) for config in configs.values()):
        raise AutomationError("Each engine_configs entry must be a JSON object.")

    campaign_name = request.get("campaign_name", request_file.stem)
    if not isinstance(campaign_name, str) or not campaign_name.strip():
        raise AutomationError("campaign_name must be a non-empty string.")
    gpu_device = request.get("gpu_device", "0")
    if not isinstance(gpu_device, (str, int)) or isinstance(gpu_device, bool) or not str(gpu_device).strip():
        raise AutomationError("gpu_device must be a GPU ID such as '0', or 'cpu' where supported.")

    common_validation = _object(request.get("common_validation", {"enabled": False}), "common_validation")
    sequence_refinement = _object(request.get("sequence_refinement", {"mode": "none"}), "sequence_refinement")
    template_redesign = _object(request.get("template_redesign", {"enabled": False}), "template_redesign")
    evaluation = _object(request.get("evaluation", {"mode": "none"}), "evaluation")
    if "enabled" in common_validation:
        _bool(common_validation["enabled"], "common_validation.enabled")

    attempts = _positive_int(request.get("design_attempts", 1), "design_attempts")
    sequences = _positive_int(request.get("sequences_per_backbone", 1), "sequences_per_backbone")
    seed = _positive_int(request.get("random_seed", 0), "random_seed", minimum=0)
    survivors = _positive_int(request.get("survivors_per_engine", 1_000_000), "survivors_per_engine")
    passing_only = _bool(request.get("passing_only", False), "passing_only")
    keep_best_failed = _bool(request.get("keep_best_failed", True), "keep_best_failed")
    continue_after_failure = _bool(request.get("continue_after_failure", True), "continue_after_failure")

    return {
        "target_pdb": target_pdb,
        "target_chains": target_chains,
        "binder_length": binder_length,
        "hotspots": hotspots,
        "campaign_name": campaign_name.strip(),
        "design_attempts": attempts,
        "sequences_per_backbone": sequences,
        "random_seed": seed,
        "engines": engines,
        "engine_configs": {engine: dict(configs.get(engine) or {}) for engine in engines},
        "workflow_recipe": workflow_recipe,
        "survivors_per_engine": survivors,
        "passing_only": passing_only,
        "keep_best_failed": keep_best_failed,
        "continue_after_failure": continue_after_failure,
        "common_validation": dict(common_validation),
        "sequence_refinement": dict(sequence_refinement),
        "template_redesign": dict(template_redesign),
        "evaluation": dict(evaluation),
        "gpu_device": str(gpu_device).strip(),
    }


def submit_design_campaign_request(request_file: str | Path) -> dict[str, Any]:
    """Submit a validated local campaign and return its stable job identity."""
    request_path = Path(request_file).expanduser().resolve()
    normalized = load_campaign_request(request_path)
    run_dir = create_design_campaign(**normalized)
    return job_reference(run_dir)


def load_campaign_request(request_file: str | Path) -> dict[str, Any]:
    """Read and validate a versioned campaign JSON file without submitting it."""
    request_path = Path(request_file).expanduser().resolve()
    try:
        payload = json.loads(request_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AutomationError(f"Could not read campaign request JSON {request_path}: {exc}") from exc
    return validate_campaign_request(payload, request_file=request_path)


def _visible_jobs(*, include_hidden: bool = False) -> list[dict[str, Any]]:
    return jobs.collect_jobs(include_hidden=include_hidden)


def resolve_job(reference: str, *, include_hidden: bool = False) -> dict[str, Any]:
    """Resolve a full run ID, job code, or task-group/run-ID reference."""
    ref = str(reference or "").strip()
    if not ref:
        raise AutomationError("A job ID or job code is required.")
    if "/" in ref:
        group, run_id = ref.split("/", 1)
        matches = [row for row in _visible_jobs(include_hidden=include_hidden) if row["task_group"] == group and row["run_id"] == run_id]
    else:
        matches = [
            row
            for row in _visible_jobs(include_hidden=include_hidden)
            if row["run_id"].lower() == ref.lower() or row["job_code"].lower() == ref.lower()
        ]
    if not matches:
        raise AutomationError(f"No job matches '{ref}'. Use a full run ID or a unique job code.")
    if len(matches) > 1:
        ids = ", ".join(f"{row['task_group']}/{row['run_id']}" for row in matches[:10])
        raise AutomationError(f"Job reference '{ref}' is ambiguous; use a full run ID. Matches: {ids}")
    return matches[0]


def job_reference(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    row = next((item for item in _visible_jobs(include_hidden=True) if Path(item["run_dir"]).resolve() == run_path), None)
    if row is None:
        raise AutomationError(f"Submitted job is not discoverable in the local run store: {run_path}")
    return {key: row[key] for key in ("job_code", "run_id", "task_group", "status", "run_dir", "created_at")}


def list_jobs(
    *,
    task_group: str | None = None,
    status: str | None = None,
    limit: int = 100,
    include_hidden: bool = False,
) -> list[dict[str, Any]]:
    limit = _positive_int(limit, "limit")
    if task_group is not None and not isinstance(task_group, str):
        raise AutomationError("task_group must be a string when provided.")
    if status is not None and not isinstance(status, str):
        raise AutomationError("status must be a string when provided.")
    rows = _visible_jobs(include_hidden=include_hidden)
    if task_group:
        rows = [row for row in rows if row["task_group"] == task_group]
    if status:
        rows = [row for row in rows if row["status"].lower() == status.lower()]
    return rows[:limit]


def inspect_job(reference: str, *, include_hidden: bool = False) -> dict[str, Any]:
    row = resolve_job(reference, include_hidden=include_hidden)
    run_dir = Path(row["run_dir"])
    metadata = jobs.read_json(run_dir / "metadata.json")
    input_payload = jobs.read_json(run_dir / "input.json")
    result = jobs.read_json(run_dir / "result.json")
    candidates_path = campaign_result_path(run_dir)
    candidates = read_candidates(run_dir)
    return {
        "job": row,
        "metadata": metadata,
        "input": input_payload,
        "result": result,
        "candidate_count": len(candidates),
        "campaign_result_path": str(candidates_path) if candidates_path.exists() else None,
    }


def job_results(
    reference: str,
    *,
    include_candidates: bool = False,
    include_hidden: bool = False,
) -> dict[str, Any]:
    details = inspect_job(reference, include_hidden=include_hidden)
    run_dir = Path(details["job"]["run_dir"])
    response: dict[str, Any] = {
        "job": details["job"],
        "result": details["result"],
        "candidate_count": details["candidate_count"],
        "campaign_result_path": details["campaign_result_path"],
    }
    if include_candidates:
        response["candidates"] = read_candidates(run_dir)
    return response


def cancel_job(reference: str, *, reason: str = "Cancelled through local automation CLI") -> dict[str, Any]:
    if not isinstance(reason, str) or not reason.strip():
        raise AutomationError("Cancellation reason must be a non-empty string.")
    row = resolve_job(reference)
    if row["status"] not in ACTIVE_STATUSES:
        raise AutomationError(f"Job {row['run_id']} is {row['status']}; only active or paused jobs can be cancelled.")
    jobs.stop_job(Path(row["run_dir"]), reason=reason)
    return resolve_job(row["run_id"])


def wait_for_job(
    reference: str,
    *,
    timeout_seconds: float = 3600,
    poll_seconds: float = 2,
    include_hidden: bool = False,
) -> dict[str, Any] | None:
    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise AutomationError("timeout_seconds must be a finite non-negative number.")
    if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
        raise AutomationError("timeout_seconds must be a finite non-negative number.")
    if isinstance(poll_seconds, bool) or not isinstance(poll_seconds, (int, float)):
        raise AutomationError("poll_seconds must be a finite positive number.")
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise AutomationError("poll_seconds must be a finite positive number.")
    deadline = time.monotonic() + timeout_seconds
    while True:
        row = resolve_job(reference, include_hidden=include_hidden)
        if row["status"] in TERMINAL_STATUSES:
            return row
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(poll_seconds, remaining))
