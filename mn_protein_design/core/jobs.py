from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from mn_protein_design.core.gpu import gpu_queue_resource
from mn_protein_design.core.portable_paths import resolve_managed_paths, store_managed_paths
from mn_protein_design.runtime import runs_root


ACTIVE_STATUSES = {"queued", "running", "preparing"}
PAUSED_STATUSES = {"paused", "holding"}
STOPPED_STATUSES = {"cancelled", "stopped"}
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
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    if Path(path).name == "command.json":
        return payload
    return resolve_managed_paths(payload, run_dir=_job_dir_for_path(Path(path)))


def _job_dir_for_path(path: Path) -> Path | None:
    try:
        relative = Path(path).expanduser().resolve().relative_to(runs_root().expanduser().resolve())
    except ValueError:
        return None
    if len(relative.parts) < 2:
        return None
    return runs_root() / relative.parts[0] / relative.parts[1]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if Path(path).name == "command.json":
        stored_payload = payload
    else:
        stored_payload = store_managed_paths(payload, run_dir=_job_dir_for_path(Path(path)))
    path.write_text(json.dumps(stored_payload, indent=2, sort_keys=True) + "\n")


def _warning_metric_active(value: object) -> bool:
    if value in {None, "", False, 0, 0.0}:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "no", "ok"}
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _metric_label(key: str) -> str:
    return key.replace("_", " ")


def _param_bool(params: dict, key: str, default: bool = False) -> bool:
    value = params.get(key)
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _refolding_requests_target_msa(params: dict) -> bool:
    """Return whether any enabled prediction engine is configured to consume target MSAs."""
    return any(
        (
            _param_bool(params, "run_alphafast_af3") and not _param_bool(params, "alphafast_query_only_msa"),
            _param_bool(params, "run_colabfold") and _param_bool(params, "colabfold_use_target_msa", True),
            _param_bool(params, "run_boltz2_initial_guess") and _param_bool(params, "boltz2_use_target_msa", True),
            _param_bool(params, "run_rf3") and _param_bool(params, "rf3_use_target_msa", True),
            _param_bool(params, "run_openfold3") and _param_bool(params, "openfold3_use_target_msa", True),
            _param_bool(params, "run_esmfold2") and _param_bool(params, "esmfold2_use_target_msa", False),
            _param_bool(params, "run_protenix") and _param_bool(params, "protenix_use_msa", True),
            _param_bool(params, "run_protenix_v1") and _param_bool(params, "protenix_v1_use_msa", True),
            _param_bool(params, "run_protenix_v2") and _param_bool(params, "protenix_v2_use_msa", True),
        )
    )


def _strip_expected_template_only_msa_warning(text: str, params: dict) -> str:
    if _refolding_requests_target_msa(params):
        return text
    expected_fragments = (
        "alphafast af3 is configured for query-only msa mode",
        "real target-msa requirement is disabled",
        "chain msa target msa blocking nonreal count",
        "chain msa target msa missing count",
        "chain msa target msa available count",
        "chain msa target msa real count",
        "chain msa target msa query only count",
        "chain msa target msa short nonreal count",
    )
    prefix = "Degraded run:"
    suffix = text.strip()
    had_prefix = suffix.lower().startswith(prefix.lower())
    if had_prefix:
        suffix = suffix[len(prefix) :].strip()
    suffix = suffix.rstrip(".")
    kept = [
        part.strip()
        for part in suffix.split(";")
        if part.strip() and not any(fragment in part.strip().lower() for fragment in expected_fragments)
    ]
    if not kept:
        return ""
    return f"{prefix} {'; '.join(kept)}." if had_prefix else "; ".join(kept)


def derive_job_warning(metadata: dict | None, input_payload: dict | None, result: dict | None) -> str:
    """Return a user-facing warning for completed/running jobs with degraded behavior."""
    metadata = metadata or {}
    input_payload = input_payload or {}
    result = result or {}
    params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
    target_msa_requested = _refolding_requests_target_msa(params)
    explicit = str(metadata.get("warning") or result.get("warning") or "").strip()
    if explicit:
        return _strip_expected_template_only_msa_warning(explicit, params)
    warnings = result.get("warnings")
    if isinstance(warnings, list):
        text_warnings = [str(item).strip() for item in warnings if str(item).strip()]
        if text_warnings:
            return _strip_expected_template_only_msa_warning("; ".join(text_warnings[:3]), params)
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    messages: list[str] = []
    if metrics.get("require_real_target_msa_disabled_after_msa_failure"):
        messages.append("Real target-MSA requirement was disabled after MSA generation failed")
    if metrics.get("alphafast_msa_pipeline_fallback"):
        messages.append("AlphaFast/MMseqs MSA fallback was used")
    if target_msa_requested and params.get("alphafast_query_only_msa"):
        messages.append("AlphaFast AF3 is configured for query-only MSA mode")
    if target_msa_requested and params.get("require_real_target_msa") is False:
        messages.append("Real target-MSA requirement is disabled")
    warning_terms = (
        "fallback",
        "query_only",
        "nonreal",
        "missing_after",
        "skipped",
        "degraded",
    )
    ignored_terms = (
        "skipped_count",
        "selected_candidate_count",
    )
    for key, value in metrics.items():
        key_text = str(key)
        key_lower = key_text.lower()
        if any(term in key_lower for term in ignored_terms):
            continue
        if not target_msa_requested and "msa" in key_lower:
            continue
        if any(term in key_lower for term in warning_terms) and _warning_metric_active(value):
            messages.append(f"{_metric_label(key_text)}: {value}")
        elif key_lower.endswith("_error") and _warning_metric_active(value):
            messages.append(f"{_metric_label(key_text)}: {value}")
    deduped: list[str] = []
    seen: set[str] = set()
    for message in messages:
        if message not in seen:
            seen.add(message)
            deduped.append(message)
    if not deduped:
        return ""
    suffix = "; ".join(deduped[:4])
    if len(deduped) > 4:
        suffix += f"; +{len(deduped) - 4} more"
    return f"Degraded run: {suffix}."


def _queue_resource_for(tool: str, params: dict | None) -> str | None:
    params = params or {}
    explicit = str(params.get("queue_resource") or "").strip().lower()
    if explicit:
        return explicit
    gpu_device = str(params.get("gpu_device") or "").strip().lower()
    if gpu_device:
        if gpu_device in {"none", "cpu", "off", "false", "0-gpu"}:
            return None
        if gpu_device in {"all", "auto", "any"}:
            return "gpu"
        if gpu_device.startswith("device="):
            gpu_device = gpu_device.removeprefix("device=").strip()
        return f"gpu:{gpu_device or '0'}"
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
    if str(owner_metadata.get("status") or "") not in ACTIVE_STATUSES:
        return True
    updated_at = str(owner_metadata.get("updated_at") or owner_metadata.get("created_at") or "")
    if updated_at:
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(updated_at)).total_seconds()
        except ValueError:
            age = 0.0
        return age > ACTIVE_OWNER_STALE_SECONDS
    return False


def _active_resource_owner(resource: str, requester_run_dir: Path) -> Path | None:
    root = runs_root()
    if not root.exists():
        return None
    try:
        requester_resolved = requester_run_dir.resolve()
    except Exception:
        requester_resolved = requester_run_dir
    for metadata_path in root.glob("*/*/metadata.json"):
        run_dir = metadata_path.parent
        try:
            same_run = run_dir.resolve() == requester_resolved
        except Exception:
            same_run = run_dir == requester_run_dir
        if same_run:
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
        active_resource = str(_queue_resource_for(str(metadata.get("tool") or input_payload.get("tool") or ""), params) or metadata.get("queue_resource") or "")
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
            try:
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
            except FileNotFoundError:
                # Another worker can reap a stale lock directory in the small
                # window between mkdir() and owner.json creation. Retry cleanly.
                continue
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


@dataclass
class _JobCreationAdoption:
    job: JobPaths
    consumed: bool = False


_JOB_CREATION_ADOPTION: ContextVar[_JobCreationAdoption | None] = ContextVar(
    "mn_protein_design_job_creation_adoption",
    default=None,
)


@contextmanager
def adopt_next_job_creation(job: JobPaths):
    """Let one workflow-created job use an already queued run directory."""
    token = _JOB_CREATION_ADOPTION.set(_JobCreationAdoption(job=job))
    try:
        yield
    finally:
        _JOB_CREATION_ADOPTION.reset(token)


def _adopt_queued_job(task_group: str, job_type: str, tool: str, inputs: dict, params: dict) -> JobPaths | None:
    adoption = _JOB_CREATION_ADOPTION.get()
    if adoption is None or adoption.consumed:
        return None
    adoption.consumed = True
    job = adoption.job
    if task_group != job.task_group:
        raise ValueError(
            f"Queued workflow expected task group {job.task_group!r}, but {tool!r} creates {task_group!r}."
        )
    queue_resource = _queue_resource_for(tool, params)
    write_json(
        job.run_dir / "input.json",
        {"job_type": job_type, "tool": tool, "inputs": inputs, "params": params},
    )
    metadata = read_json(job.run_dir / "metadata.json")
    metadata.update({"task_group": task_group, "job_type": job_type, "tool": tool})
    if queue_resource:
        metadata["queue_resource"] = queue_resource
    else:
        metadata.pop("queue_resource", None)
    metadata["updated_at"] = utc_now()
    write_json(job.run_dir / "metadata.json", metadata)
    return job


def create_job(task_group: str, job_type: str, tool: str, inputs: dict, params: dict | None = None) -> JobPaths:
    params = params or {}
    adopted_job = _adopt_queued_job(task_group, job_type, tool, inputs, params)
    if adopted_job is not None:
        return adopted_job
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    run_dir = runs_root() / task_group / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "artifacts").mkdir()
    for name in ("stdout.log", "stderr.log"):
        (run_dir / name).write_text("")
    created_at = utc_now()
    queue_resource = _queue_resource_for(tool, params)
    write_json(run_dir / "input.json", {"job_type": job_type, "tool": tool, "inputs": inputs, "params": params})
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


def mark_internal_job(
    run_dir: Path,
    *,
    parent_run_dir: Path,
    parent_task_group: str,
    role: str,
    engine: str | None = None,
) -> None:
    metadata = read_json(run_dir / "metadata.json")
    metadata["hidden"] = True
    metadata["parent_task_group"] = parent_task_group
    metadata["parent_run_id"] = parent_run_dir.name
    metadata["parent_run_dir"] = str(parent_run_dir)
    metadata["parent_role"] = role
    if engine:
        metadata["benchmark_engine"] = engine
    metadata["updated_at"] = utc_now()
    write_json(run_dir / "metadata.json", metadata)


def update_status(run_dir: Path, status: str, **extra: object) -> None:
    metadata = read_json(run_dir / "metadata.json")
    if status == "running":
        input_payload = read_json(run_dir / "input.json")
        params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
        derived_queue_resource = _queue_resource_for(str(metadata.get("tool") or input_payload.get("tool") or ""), params)
        queue_resource = str(extra.get("queue_resource") or derived_queue_resource or metadata.get("queue_resource") or "").strip()
        if queue_resource:
            _acquire_resource_lock(run_dir, queue_resource)
            metadata = read_json(run_dir / "metadata.json")
    metadata.update(extra)
    metadata["status"] = status
    if status == "running":
        metadata.pop("scheduler_dispatch_requested", None)
        metadata.pop("scheduler_wait_reason", None)
    metadata["updated_at"] = utc_now()
    if status in {"completed", "failed", "cancelled"}:
        metadata.setdefault("completed_at", metadata["updated_at"])
    write_json(run_dir / "metadata.json", metadata)
    if status in {"completed", "failed", "cancelled", *PAUSED_STATUSES}:
        _release_resource_lock(run_dir)


def finish_job(run_dir: Path, success: bool, result: dict) -> None:
    input_payload = read_json(run_dir / "input.json")
    metadata = read_json(run_dir / "metadata.json")
    if success:
        for key in (
            "warning",
            "warning_level",
            "worker_error",
            "worker_exception_type",
            "cancel_reason",
            "pause_reason",
        ):
            metadata.pop(key, None)
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
    warning = derive_job_warning(metadata, input_payload, normalized)
    if warning:
        metadata["warning"] = warning
        metadata["warning_level"] = metadata.get("warning_level") or "degraded"
        metadata["updated_at"] = utc_now()
        write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "result.json", normalized)
    update_status(run_dir, "completed" if success else "failed")
    _release_resource_lock(run_dir)


def _signal_worker_process(run_dir: Path, sig: int = signal.SIGTERM) -> bool:
    metadata = read_json(run_dir / "metadata.json")
    try:
        pid = int(metadata.get("worker_pid") or 0)
    except (TypeError, ValueError):
        pid = 0
    if pid <= 0:
        return False
    try:
        os.killpg(pid, sig)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        try:
            os.kill(pid, sig)
            return True
        except Exception:
            return False
    except Exception:
        return False


def _stop_run_docker_containers(run_dir: Path) -> list[str]:
    """Stop running Docker containers whose inspect metadata references run_dir."""
    run_text = str(run_dir.expanduser().resolve())
    try:
        ps = subprocess.run(
            ["docker", "ps", "-q"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    container_ids = [line.strip() for line in ps.stdout.splitlines() if line.strip()]
    if not container_ids:
        return []

    matching: list[str] = []
    for container_id in container_ids:
        try:
            inspected = subprocess.run(
                ["docker", "inspect", container_id],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            continue
        if inspected.returncode == 0 and run_text in inspected.stdout:
            matching.append(container_id)

    stopped: list[str] = []
    for container_id in matching:
        try:
            result = subprocess.run(
                ["docker", "stop", container_id],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except Exception:
            continue
        if result.returncode == 0:
            stopped.append(container_id)
    return stopped


def stop_job(run_dir: Path, reason: str = "Stopped by user") -> None:
    """Stop a queued/running job and mark it cancelled without deleting artifacts."""
    _signal_worker_process(run_dir, signal.SIGTERM)
    stopped_containers = _stop_run_docker_containers(run_dir)
    _release_resource_lock(run_dir)
    metadata = read_json(run_dir / "metadata.json")
    stopped_at = utc_now()
    metadata["status"] = "cancelled"
    metadata["cancelled_at"] = stopped_at
    metadata["completed_at"] = stopped_at
    metadata["updated_at"] = stopped_at
    metadata["progress_label"] = reason
    metadata["cancel_reason"] = reason
    if stopped_containers:
        metadata["stopped_container_ids"] = stopped_containers
    write_json(run_dir / "metadata.json", metadata)
    result = read_json(run_dir / "result.json")
    result.update(
        {
            "success": False,
            "cancelled": True,
            "cancel_reason": reason,
        }
    )
    write_json(run_dir / "result.json", result)


def pause_job(run_dir: Path, reason: str = "") -> None:
    """Park a resumable job without treating it as a failed result."""
    _signal_worker_process(run_dir, signal.SIGTERM)
    stopped_containers = _stop_run_docker_containers(run_dir)
    _release_resource_lock(run_dir)
    metadata = read_json(run_dir / "metadata.json")
    metadata.pop("completed_at", None)
    metadata["status"] = "paused"
    metadata["paused_at"] = utc_now()
    metadata["updated_at"] = metadata["paused_at"]
    if reason:
        metadata["pause_reason"] = reason
        metadata["progress_label"] = reason
    if stopped_containers:
        metadata["stopped_container_ids"] = stopped_containers
    write_json(run_dir / "metadata.json", metadata)
    result = read_json(run_dir / "result.json")
    if result:
        result["success"] = None
        result["paused"] = True
        if reason:
            result["pause_reason"] = reason
        write_json(run_dir / "result.json", result)


def prepare_job_for_resume(run_dir: Path) -> None:
    """Return a stopped job to the queue without discarding its artifacts."""
    _release_resource_lock(run_dir)
    metadata = read_json(run_dir / "metadata.json")
    metadata.pop("completed_at", None)
    metadata.pop("worker_error", None)
    metadata.pop("worker_exception_type", None)
    metadata["status"] = "queued"
    metadata["resume_requested_at"] = utc_now()
    metadata["current_phase"] = "Waiting to resume"
    metadata["current_engine"] = ""
    metadata["progress_label"] = "Waiting to resume from completed artifacts"
    metadata["updated_at"] = metadata["resume_requested_at"]
    write_json(run_dir / "metadata.json", metadata)
    result = read_json(run_dir / "result.json")
    if result:
        result["success"] = None
        result["resuming"] = True
        result.pop("worker_error", None)
        result.pop("worker_exception_type", None)
        write_json(run_dir / "result.json", result)


def set_job_resume_gpu(run_dir: Path, gpu_device: object | None) -> None:
    """Retarget a paused/stopped worker job before resuming it."""
    if gpu_device is None:
        return
    gpu_text = str(gpu_device).strip()
    if not gpu_text:
        return
    queue_resource = gpu_queue_resource(gpu_text)

    metadata = read_json(run_dir / "metadata.json")
    metadata.pop("queue_lock_acquired", None)
    metadata.pop("queue_shared_with_run_dir", None)
    if queue_resource:
        metadata["queue_resource"] = queue_resource
    else:
        metadata.pop("queue_resource", None)
    metadata["updated_at"] = utc_now()
    write_json(run_dir / "metadata.json", metadata)

    input_path = run_dir / "input.json"
    input_payload = read_json(input_path)
    params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
    params["gpu_device"] = gpu_text
    params["alphafast_gpu_device"] = gpu_text
    if queue_resource:
        params["queue_resource"] = queue_resource
    else:
        params.pop("queue_resource", None)
    input_payload["params"] = params
    write_json(input_path, input_payload)

    worker_request_path = run_dir / "worker_request.json"
    if worker_request_path.exists():
        worker_request = read_json(worker_request_path)
        kwargs = worker_request.get("kwargs") if isinstance(worker_request.get("kwargs"), dict) else {}
        kwargs["gpu_device"] = gpu_text
        kwargs["alphafast_gpu_device"] = gpu_text
        kwargs["colabfold_gpu_device"] = gpu_text
        validation_kwargs = kwargs.get("validation_kwargs") if isinstance(kwargs.get("validation_kwargs"), dict) else None
        if validation_kwargs is not None:
            validation_kwargs["gpu_device"] = gpu_text
            validation_kwargs["alphafast_gpu_device"] = gpu_text
            validation_kwargs["colabfold_gpu_device"] = gpu_text
        # worker kwargs are passed directly to workflow functions; queue_resource
        # belongs in metadata/input params and would be an unexpected kwarg.
        kwargs.pop("queue_resource", None)
        worker_request["kwargs"] = kwargs
        write_json(worker_request_path, worker_request)


def mark_job_for_artifact_resume(run_dir: Path) -> None:
    """Make a worker request prefer completed artifacts after pause/stop/failure."""
    input_path = run_dir / "input.json"
    input_payload = read_json(input_path)
    params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
    params["resume"] = True
    input_payload["params"] = params
    if input_payload:
        write_json(input_path, input_payload)

    worker_request_path = run_dir / "worker_request.json"
    if not worker_request_path.exists():
        return
    worker_request = read_json(worker_request_path)
    kind = str(worker_request.get("kind") or "")
    kwargs = worker_request.get("kwargs") if isinstance(worker_request.get("kwargs"), dict) else {}
    if kind in {"de_novo_binder_scoring_dataset", "candidate_refolding_evaluation"}:
        kwargs["resume"] = True
    elif kind == "sequence_design_pipeline":
        kwargs["resume"] = True
        validation_kwargs = kwargs.get("validation_kwargs") if isinstance(kwargs.get("validation_kwargs"), dict) else {}
        if validation_kwargs:
            validation_kwargs["resume"] = True
            kwargs["validation_kwargs"] = validation_kwargs
    kwargs.pop("queue_resource", None)
    worker_request["kwargs"] = kwargs
    write_json(worker_request_path, worker_request)


def resume_job(run_dir: Path, *, gpu_device: object | None = None) -> None:
    """Queue a paused/stopped local-worker job again."""
    if not (run_dir / "worker_request.json").exists():
        raise FileNotFoundError(f"Job is not resumable because worker_request.json is missing: {run_dir}")
    set_job_resume_gpu(run_dir, gpu_device)
    mark_job_for_artifact_resume(run_dir)
    prepare_job_for_resume(run_dir)
    from mn_protein_design.core.local_worker import spawn_worker_for_run

    spawn_worker_for_run(run_dir)


def collect_jobs(task_group: str | None = None, *, include_hidden: bool = False) -> list[dict]:
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
            if not include_hidden and (metadata.get("hidden") or metadata.get("capacity_parent_run_id")):
                continue
            result = read_json(run_dir / "result.json")
            input_payload = read_json(run_dir / "input.json")
            input_params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
            warning = derive_job_warning(metadata, input_payload, result)
            queue_resource = (
                _queue_resource_for(str(metadata.get("tool") or input_payload.get("tool", "")), input_params)
                or metadata.get("queue_resource", "")
            )
            allocation = metadata.get("resource_allocation") if isinstance(metadata.get("resource_allocation"), dict) else {}
            resource_allocation = ""
            if allocation:
                try:
                    cpu_cores = int(allocation.get("cpu_cores") or 0)
                except (TypeError, ValueError):
                    cpu_cores = 0
                gpu_device = str(allocation.get("gpu_device") or "cpu")
                cpu_text = f"{cpu_cores} CPU core{'s' if cpu_cores != 1 else ''}" if cpu_cores else "coordination"
                resource_allocation = f"{cpu_text} · {gpu_device}"
            metadata_status = str(metadata.get("status") or "")
            status = metadata_status or ("completed" if result.get("success") else "unknown")
            if (
                result.get("success") is False
                and metadata_status not in ACTIVE_STATUSES
                and metadata_status not in PAUSED_STATUSES
                and metadata_status not in STOPPED_STATUSES
            ):
                status = "failed"
            rows.append(
                {
                    "job_code": display_job_code(metadata.get("job_code"), run_dir.name),
                    "run_id": run_dir.name,
                    "task_group": group,
                    "job_type": metadata.get("job_type") or input_payload.get("job_type", ""),
                    "tool": metadata.get("tool") or input_payload.get("tool", ""),
                    "status": status,
                    "queue_resource": queue_resource,
                    "resource_allocation": resource_allocation,
                    "scheduler_wait_reason": metadata.get("scheduler_wait_reason", ""),
                    "current_phase": metadata.get("current_phase", ""),
                    "current_engine": metadata.get("current_engine", ""),
                    "warning": warning,
                    "warning_level": metadata.get("warning_level", "degraded" if warning else ""),
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
