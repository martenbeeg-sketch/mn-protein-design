from __future__ import annotations

from pathlib import Path
from typing import Any

from mn_protein_design.core.jobs import read_json, utc_now, write_json


CAMPAIGN_GROUP = "campaign"
PIPELINE_SCHEMA_VERSION = "mn-protein-design.pipeline.v1"


def pipeline_path(run_dir: Path) -> Path:
    return run_dir / "pipeline.json"


def read_pipeline(run_dir: Path) -> dict[str, Any]:
    return read_json(pipeline_path(run_dir))


def write_pipeline(run_dir: Path, payload: dict[str, Any]) -> None:
    payload.setdefault("schema_version", PIPELINE_SCHEMA_VERSION)
    payload["updated_at"] = utc_now()
    write_json(pipeline_path(run_dir), payload)


def new_step(step_id: str, module: str, tool: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "step_id": step_id,
        "module": module,
        "tool": tool,
        "status": "pending",
        "params": params or {},
        "job_ref": None,
        "created_at": utc_now(),
        "updated_at": utc_now(),
    }


def step_source(payload: dict[str, Any], step_index: int) -> dict[str, Any]:
    if step_index <= 0:
        return dict(payload.get("initial_source") or {})
    previous = dict((payload.get("steps") or [])[step_index - 1])
    return dict(previous.get("job_ref") or {})


def step_summary_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, step in enumerate(payload.get("steps") or [], start=1):
        job_ref = step.get("job_ref") or {}
        rows.append(
            {
                "order": index,
                "module": step.get("module"),
                "tool": step.get("tool"),
                "status": step.get("status"),
                "job_code": job_ref.get("job_code", ""),
                "run_id": job_ref.get("run_id", ""),
                "candidate_count": job_ref.get("candidate_count", ""),
            }
        )
    return rows
