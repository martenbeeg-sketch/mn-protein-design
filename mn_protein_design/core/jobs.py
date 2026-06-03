from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from mn_protein_design.runtime import runs_root


ACTIVE_STATUSES = {"queued", "running", "preparing"}
ACTIVE_OWNER_STALE_SECONDS = 6 * 60 * 60
GPU_TOOL_KEYWORDS = (
    "alphafold",
    "af2",
    "af3",
    "binder_scoring",
    "boltz",
    "colabfold",
    "esmfold",
    "foundry",
    "ligandmpnn",
    "proteina",
    "protpardelle",
    "pxdesign",
    "rfdiffusion",
)
_PROCESS_RESOURCE_LOCKS: dict[str, Path] = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def short_job_code(run_id: str) -> str:
    suffix = run_id.rsplit("-", 1)[-1]
    compact = "".join(ch for ch in suffix.upper() if ch.isalnum())
    if len(compact) >= 5:
        return compact[:5]
    fallback = "".join(ch for ch in run_id.upper() if ch.isalnum())
    return (compact + fallback)[:5] or "JOB00"


def display_job_code(metadata_code: object, run_id: str) -> str:
    code = str(metadata_code or "").strip().upper()
    if len(code) == 5 and code.isalnum():
        return code
    return short_job_code(run_id)


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _queue_resource_for(tool: str, params: dict | None) -> str | None:
    params = params or {}
    explicit = str(params.get("queue_resource") or "").strip().lower()
    if explicit:
        return explicit
    if str(params.get("device") or "").strip().lower() == "cuda":
        return "gpu"
    if any("gpu" in str(key).lower() for key in params):
        return "gpu"
    tool_text = str(tool or "").lower()
    if any(keyword in tool_text for keyword in GPU_TOOL_KEYWORDS):
        return "gpu"
    return None


def _lock_dir(resource: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in resource.strip().lower())
    return runs_root() / "_locks" / f"{safe or 'resource'}.lock"


def _lock_is_stale(lock_dir: Path) -> bool:
    owner = read_json(lock_dir / "owner.json")
    owner_run_dir_text = str(owner.get("run_dir") or "").strip()
    if not owner_run_dir_text:
        return True
    owner_run_dir = Path(owner_run_dir_text)
    if not owner_run_dir.exists():
        return True
    owner_metadata = read_json(owner_run_dir / "metadata.json")
    return str(owner_metadata.get("status") or "") not in ACTIVE_STATUSES


def _active_resource_owner(resource: str, requester_run_dir: Path) -> Path | None:
    root = runs_root()
    if not root.exists():
        return None
    for metadata_path in root.glob("*/*/metadata.json"):
        run_dir = metadata_path.parent
        if run_dir == requester_run_dir:
            continue
        metadata = read_json(metadata_path)
        status = str(metadata.get("status") or "")
        if status not in {"running", "preparing"}:
            continue
        updated_at = str(metadata.get("updated_at") or metadata.get("created_at") or "")
        if updated_at:
            try:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(updated_at)).total_seconds()
            except ValueError:
                age = 0.0
            if age > ACTIVE_OWNER_STALE_SECONDS:
                continue
        input_payload = read_json(run_dir / "input.json")
        params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
        active_resource = str(metadata.get("queue_resource") or _queue_resource_for(str(metadata.get("tool") or ""), params) or "")
        if active_resource == resource:
            return run_dir
    return None


def _acquire_resource_lock(run_dir: Path, resource: str, poll_seconds: float = 5.0) -> None:
    lock_dir = _lock_dir(resource)
    metadata = read_json(run_dir / "metadata.json")
    if metadata.get("queue_lock_acquired") == resource:
        return
    process_owner = _PROCESS_RESOURCE_LOCKS.get(resource)
    if process_owner is not None and process_owner != run_dir:
        metadata["queue_resource"] = resource
        metadata["queue_shared_with_run_dir"] = str(process_owner)
        metadata["updated_at"] = utc_now()
        write_json(run_dir / "metadata.json", metadata)
        return
    metadata["queue_resource"] = resource
    metadata.setdefault("queue_wait_started_at", utc_now())
    metadata["status"] = "queued"
    metadata["updated_at"] = utc_now()
    write_json(run_dir / "metadata.json", metadata)
    while True:
        if _active_resource_owner(resource, run_dir) is not None:
            time.sleep(max(0.5, poll_seconds))
            continue
        try:
            lock_dir.parent.mkdir(parents=True, exist_ok=True)
            lock_dir.mkdir()
            write_json(
                lock_dir / "owner.json",
                {
                    "resource": resource,
                    "run_dir": str(run_dir),
                    "task_group": metadata.get("task_group"),
                    "run_id": metadata.get("run_id"),
                    "acquired_at": utc_now(),
                },
            )
            metadata = read_json(run_dir / "metadata.json")
            metadata["queue_lock_acquired"] = resource
            metadata["queue_started_at"] = utc_now()
            metadata["updated_at"] = metadata["queue_started_at"]
            write_json(run_dir / "metadata.json", metadata)
            _PROCESS_RESOURCE_LOCKS[resource] = run_dir
            return
        except FileExistsError:
            if _lock_is_stale(lock_dir):
                shutil.rmtree(lock_dir, ignore_errors=True)
                continue
            time.sleep(max(0.5, poll_seconds))


def _release_resource_lock(run_dir: Path) -> None:
    metadata = read_json(run_dir / "metadata.json")
    resource = str(metadata.get("queue_lock_acquired") or "").strip()
    if not resource:
        return
    lock_dir = _lock_dir(resource)
    owner = read_json(lock_dir / "owner.json")
    if str(owner.get("run_dir") or "") == str(run_dir):
        shutil.rmtree(lock_dir, ignore_errors=True)
    if _PROCESS_RESOURCE_LOCKS.get(resource) == run_dir:
        _PROCESS_RESOURCE_LOCKS.pop(resource, None)
    metadata.pop("queue_lock_acquired", None)
    metadata["updated_at"] = utc_now()
    write_json(run_dir / "metadata.json", metadata)


@dataclass(frozen=True)
class JobPaths:
    task_group: str
    run_id: str
    run_dir: Path

    @property
    def artifacts_dir(self) -> Path:
        return self.run_dir / "artifacts"


def create_job(task_group: str, job_type: str, tool: str, inputs: dict, params: dict | None = None) -> JobPaths:
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    run_dir = runs_root() / task_group / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "artifacts").mkdir()
    for name in ("stdout.log", "stderr.log"):
        (run_dir / name).write_text("")
    created_at = utc_now()
    queue_resource = _queue_resource_for(tool, params)
    write_json(run_dir / "input.json", {"job_type": job_type, "tool": tool, "inputs": inputs, "params": params or {}})
    metadata = {
        "run_id": run_id,
        "job_code": short_job_code(run_id),
        "task_group": task_group,
        "job_type": job_type,
        "tool": tool,
        "status": "queued",
        "created_at": created_at,
        "updated_at": created_at,
    }
    if queue_resource:
        metadata["queue_resource"] = queue_resource
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "command.json", {"mode": "internal", "command": []})
    return JobPaths(task_group=task_group, run_id=run_id, run_dir=run_dir)


def update_status(run_dir: Path, status: str, **extra: object) -> None:
    metadata = read_json(run_dir / "metadata.json")
    if status == "running":
        queue_resource = str(extra.get("queue_resource") or metadata.get("queue_resource") or "").strip()
        if queue_resource:
            _acquire_resource_lock(run_dir, queue_resource)
            metadata = read_json(run_dir / "metadata.json")
    metadata.update(extra)
    metadata["status"] = status
    metadata["updated_at"] = utc_now()
    if status in {"completed", "failed", "cancelled"}:
        metadata.setdefault("completed_at", metadata["updated_at"])
    write_json(run_dir / "metadata.json", metadata)


def finish_job(run_dir: Path, success: bool, result: dict) -> None:
    input_payload = read_json(run_dir / "input.json")
    normalized = {
        "success": success,
        "job_type": input_payload.get("job_type"),
        "tool": input_payload.get("tool"),
        "inputs": input_payload.get("inputs", {}),
        "outputs": result.get("outputs", {}),
        "metrics": result.get("metrics", {}),
        "downstream_artifacts": result.get("downstream_artifacts", {}),
    }
    for key, value in result.items():
        normalized.setdefault(key, value)
    write_json(run_dir / "result.json", normalized)
    update_status(run_dir, "completed" if success else "failed")
    _release_resource_lock(run_dir)


def collect_jobs(task_group: str | None = None) -> list[dict]:
    root = runs_root()
    root.mkdir(parents=True, exist_ok=True)
    groups = [task_group] if task_group else [p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")]
    rows: list[dict] = []
    for group in groups:
        group_dir = root / group
        if not group_dir.exists():
            continue
        for run_dir in sorted([p for p in group_dir.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True):
            metadata = read_json(run_dir / "metadata.json")
            if metadata.get("hidden"):
                continue
            result = read_json(run_dir / "result.json")
            input_payload = read_json(run_dir / "input.json")
            input_params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
            status = metadata.get("status") or ("completed" if result.get("success") else "unknown")
            if result.get("success") is False:
                status = "failed"
            rows.append(
                {
                    "job_code": display_job_code(metadata.get("job_code"), run_dir.name),
                    "run_id": run_dir.name,
                    "task_group": group,
                    "job_type": metadata.get("job_type") or input_payload.get("job_type", ""),
                    "tool": metadata.get("tool") or input_payload.get("tool", ""),
                    "status": status,
                    "queue_resource": metadata.get("queue_resource", ""),
                    "current_phase": metadata.get("current_phase", ""),
                    "current_engine": metadata.get("current_engine", ""),
                    "campaign_name": metadata.get("campaign_name") or input_params.get("campaign_name", ""),
                    "campaign_id": metadata.get("campaign_id", ""),
                    "campaign_step": metadata.get("campaign_step", ""),
                    "campaign_step_index": metadata.get("campaign_step_index", ""),
                    "created_at": metadata.get("created_at", ""),
                    "updated_at": metadata.get("updated_at", ""),
                    "success": result.get("success"),
                    "run_dir": str(run_dir),
                }
            )
    return rows


def get_run_dir(task_group: str, run_id: str) -> Path:
    return runs_root() / task_group / run_id


def delete_job_run(task_group: str, run_id: str) -> Path:
    root = runs_root().resolve()
    run_dir = (root / task_group / run_id).resolve()
    run_dir.relative_to(root)
    if not run_dir.exists():
        raise FileNotFoundError(f"Job run does not exist: {task_group}/{run_id}")
    if not run_dir.is_dir():
        raise NotADirectoryError(f"Job run is not a directory: {run_dir}")
    shutil.rmtree(run_dir)
    return run_dir


def _iter_payload_strings(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        strings: list[str] = []
        for child in value.values():
            strings.extend(_iter_payload_strings(child))
        return strings
    if isinstance(value, list):
        strings: list[str] = []
        for child in value:
            strings.extend(_iter_payload_strings(child))
        return strings
    return []


def _job_identity(row: dict) -> tuple[str, str]:
    return str(row["task_group"]), str(row["run_id"])


def _job_depends_on(run_dir: Path, upstream_run_dir: Path) -> bool:
    upstream = upstream_run_dir.resolve()
    for name in ("input.json", "pipeline.json", "result.json"):
        payload_path = run_dir / name
        if not payload_path.exists():
            continue
        payload = read_json(payload_path)
        if name == "pipeline.json":
            initial_source = payload.get("initial_source") or {}
            source_run_dir = initial_source.get("run_dir")
            if source_run_dir:
                try:
                    Path(str(source_run_dir)).resolve().relative_to(upstream)
                    return True
                except Exception:
                    pass
            continue
        for text in _iter_payload_strings(payload):
            if not text:
                continue
            try:
                path = Path(text)
                resolved = path.resolve() if path.is_absolute() else (run_dir / path).resolve()
            except Exception:
                resolved = None
            if resolved is not None:
                try:
                    resolved.relative_to(upstream)
                    return True
                except ValueError:
                    pass
            if str(upstream) in text:
                return True
    return False


def find_downstream_jobs(task_group: str, run_id: str) -> list[dict]:
    root = runs_root().resolve()
    upstream_run_dir = (root / task_group / run_id).resolve()
    upstream_identity = (task_group, run_id)
    downstream: list[dict] = []
    seen: set[tuple[str, str]] = set()
    frontier = [upstream_run_dir]
    all_jobs = collect_jobs()
    while frontier:
        current = frontier.pop(0)
        for row in all_jobs:
            identity = _job_identity(row)
            if identity == upstream_identity or identity in seen:
                continue
            run_dir = Path(str(row["run_dir"]))
            if _job_depends_on(run_dir, current):
                seen.add(identity)
                downstream.append(row)
                frontier.append(run_dir)
    return downstream


def deletion_plan(selections: list[tuple[str, str]], include_downstream: bool = False) -> list[dict]:
    rows_by_identity = {_job_identity(row): row for row in collect_jobs()}
    planned: dict[tuple[str, str], dict] = {}
    for task_group, run_id in selections:
        identity = (task_group, run_id)
        if identity in rows_by_identity:
            planned[identity] = rows_by_identity[identity]
        if include_downstream:
            for row in find_downstream_jobs(task_group, run_id):
                planned[_job_identity(row)] = row
    return sorted(
        planned.values(),
        key=lambda row: len(str(row.get("run_dir") or "")),
        reverse=True,
    )
