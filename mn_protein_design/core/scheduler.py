from __future__ import annotations

import os
import math
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from mn_compute_scheduler.resources import (
    AdmissionRequest,
    CPULease,
    FileLease,
    MemoryLease,
    ResourceSnapshot,
    acquire_cpu_lease,
    acquire_first_gpu_lease,
    acquire_gpu_lease,
    acquire_memory_lease,
    active_resource_reservations,
    assess_resource_admission,
    capture_resource_snapshot,
    cpu_pool_capacity,
    discover_gpu_capacity,
    discover_gpu_ids,
    shared_state_dir,
)

from mn_protein_design.core.jobs import (
    ACTIVE_STATUSES,
    ACTIVE_OWNER_STALE_SECONDS,
    _queue_resource_for,
    finish_job,
    read_json,
    update_status,
    utc_now,
    write_json,
)
from mn_protein_design.runtime import runs_root


SERVICE_DIR_NAME = "_worker_service"
DEFAULT_POLL_SECONDS = 2.0
COORDINATOR_JOB_TYPES = {
    "refolding_capacity_benchmark",
    "design_generator_capacity_benchmark",
}
CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


@dataclass(frozen=True)
class ResourceRequest:
    cpu_cores: int
    gpu_resource: str | None
    gpu_device: str | None
    min_vram_gb: float = 0.0
    ram_gb: float = 0.0
    scratch_gb: float = 0.0


@dataclass
class ActiveWorker:
    process: subprocess.Popen
    run_dir: Path
    request: ResourceRequest
    memory_lease: MemoryLease | None = None
    cpu_lease: CPULease | None = None
    gpu_leases: tuple[FileLease, ...] = ()
    gpu_devices: tuple[str, ...] = ()


def worker_service_dir(root: Path | None = None) -> Path:
    return (root or runs_root()) / SERVICE_DIR_NAME


def worker_service_status_path(root: Path | None = None) -> Path:
    return worker_service_dir(root) / "status.json"


def worker_service_lock_path(root: Path | None = None) -> Path:
    return worker_service_dir(root) / "service.lock"


def configured_cpu_slots(value: int | None = None) -> int:
    if value is None:
        raw = os.getenv("MN_PROTEIN_DESIGN_WORKER_CPU_SLOTS", "").strip()
        if raw:
            try:
                value = int(raw)
            except ValueError as exc:
                raise ValueError("MN_PROTEIN_DESIGN_WORKER_CPU_SLOTS must be a positive integer") from exc
    shared_capacity = cpu_pool_capacity()
    if value is None:
        slots = shared_capacity
    else:
        try:
            requested = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Worker CPU slots must be a positive integer") from exc
        if requested < 1:
            raise ValueError("Worker CPU slots must be at least 1")
        slots = min(requested, shared_capacity)
    if slots < 1:
        raise ValueError("Worker CPU slots must be at least 1")
    return slots


def detect_gpu_devices(timeout_seconds: float = 2.0) -> list[str]:
    """Return host GPU indices reported by nvidia-smi; [] means no visible GPUs."""
    _ = timeout_seconds
    return [str(device) for device in discover_gpu_ids()]


def _read_run_payloads(run_dir: Path) -> tuple[dict, dict, dict]:
    metadata = read_json(run_dir / "metadata.json")
    input_payload = read_json(run_dir / "input.json")
    request = read_json(run_dir / "worker_request.json")
    return metadata, input_payload, request


def _nested_cpu_thread_requests(value: object) -> list[int]:
    requests: list[int] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in {"num_threads", "nprocs", "pyrosetta_nprocs"}:
                try:
                    parsed = int(child)
                except (TypeError, ValueError):
                    parsed = 0
                if parsed > 0:
                    requests.append(parsed)
            requests.extend(_nested_cpu_thread_requests(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            requests.extend(_nested_cpu_thread_requests(child))
    return requests


def request_for_run(run_dir: Path, cpu_capacity: int | None = None) -> ResourceRequest:
    metadata, input_payload, worker_request = _read_run_payloads(run_dir)
    params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
    worker_resources = worker_request.get("resources") if isinstance(worker_request.get("resources"), dict) else {}
    metadata_resources = metadata.get("resource_request") if isinstance(metadata.get("resource_request"), dict) else {}
    params_resources = params.get("resource_request") if isinstance(params.get("resource_request"), dict) else {}
    resource_sources = (worker_resources, metadata_resources, params_resources, params)

    def resource_value(*keys: str, default: object = 0) -> object:
        return next(
            (
                source[key]
                for source in resource_sources
                for key in keys
                if source.get(key) not in (None, "")
            ),
            default,
        )

    raw_cores = next(
        (
            item
            for item in (
                worker_resources.get("cpu_cores"),
                metadata_resources.get("cpu_cores"),
                params.get("cpu_cores"),
                params.get("cpu_slots"),
            )
            if item not in (None, "")
        ),
        max(_nested_cpu_thread_requests(params), default=None),
    )
    job_type = str(metadata.get("job_type") or input_payload.get("job_type") or "")
    gpu_resource = str(metadata.get("queue_resource") or "").strip().lower() or _queue_resource_for(
        str(metadata.get("tool") or input_payload.get("tool") or ""), params
    )
    explicit_gpu = resource_value("gpu", default=None)
    if explicit_gpu is not None and (
        explicit_gpu is False
        or explicit_gpu == 0
        or str(explicit_gpu).strip().lower() in {"false", "no", "none", "cpu", ""}
    ):
        gpu_resource = None
    raw_gpu_device = resource_value("gpu_device", default=None)
    if raw_gpu_device in (None, ""):
        raw_gpu_ids = resource_value("gpu_ids", default="")
        if isinstance(raw_gpu_ids, (list, tuple)):
            raw_gpu_device = ",".join(str(item) for item in raw_gpu_ids)
        else:
            raw_gpu_device = raw_gpu_ids
    gpu_raw = str(raw_gpu_device or "").strip()
    if gpu_resource and gpu_resource.startswith("gpu:"):
        gpu_device = gpu_resource.removeprefix("gpu:")
    elif gpu_resource == "gpu":
        gpu_device = gpu_raw if gpu_raw and gpu_raw.lower() not in {"auto", "any"} else "all"
    else:
        gpu_device = None

    if raw_cores is None:
        capacity = cpu_capacity or configured_cpu_slots()
        if job_type in COORDINATOR_JOB_TYPES:
            # These workers mostly monitor and dispatch child jobs; consuming
            # every CPU slot would prevent their children from starting.
            cpu_cores = 0
        else:
            cpu_cores = min(4, capacity)
    else:
        try:
            cpu_cores = int(raw_cores)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid CPU core request in {run_dir}: {raw_cores!r}") from exc
    if cpu_cores < 0 or (cpu_cores == 0 and job_type not in COORDINATOR_JOB_TYPES):
        raise ValueError(f"CPU core request must be positive in {run_dir}")
    try:
        min_vram_gb = float(resource_value("min_vram_gb", default=0) or 0)
        ram_gb = float(resource_value("ram_gb", default=0) or 0)
        scratch_gb = float(resource_value("scratch_gb", default=0) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid RAM/VRAM/scratch request in {run_dir}: {exc}") from exc
    if min(min_vram_gb, ram_gb, scratch_gb) < 0:
        raise ValueError(f"RAM/VRAM/scratch requests cannot be negative in {run_dir}")
    if not all(math.isfinite(value) for value in (min_vram_gb, ram_gb, scratch_gb)):
        raise ValueError(f"RAM/VRAM/scratch requests must be finite in {run_dir}")
    return ResourceRequest(
        cpu_cores=cpu_cores,
        gpu_resource=gpu_resource,
        gpu_device=gpu_device,
        min_vram_gb=min_vram_gb,
        ram_gb=ram_gb,
        scratch_gb=scratch_gb,
    )


def _gpu_indices(resource: str | None, device: str | None) -> set[str] | None:
    """Return selected GPU indices, or None when the request occupies every GPU."""
    if not resource:
        return set()
    raw = str(device or "all").strip().lower()
    if resource == "gpu" or raw in {"", "all", "auto", "any"}:
        return None
    if raw.startswith("device="):
        raw = raw.removeprefix("device=")
    return {part.strip() for part in raw.split(",") if part.strip()}


def gpu_requests_conflict(left: ResourceRequest, right: ResourceRequest) -> bool:
    if not left.gpu_resource or not right.gpu_resource:
        return False
    left_devices = _gpu_indices(left.gpu_resource, left.gpu_device)
    right_devices = _gpu_indices(right.gpu_resource, right.gpu_device)
    if left_devices is None or right_devices is None:
        return True
    return bool(left_devices.intersection(right_devices))


def _pid_alive(pid: object) -> bool:
    try:
        value = int(pid or 0)
    except (TypeError, ValueError):
        return False
    if value <= 0:
        return False
    try:
        os.kill(value, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def _run_age_seconds(metadata: dict, now: datetime) -> float:
    updated_at = str(metadata.get("updated_at") or metadata.get("created_at") or "")
    if not updated_at:
        return 0.0
    try:
        return max(0.0, (now - datetime.fromisoformat(updated_at)).total_seconds())
    except ValueError:
        return 0.0


def _iter_run_dirs(root: Path | None = None):
    root = root or runs_root()
    if not root.exists():
        return
    for group_dir in root.iterdir():
        if not group_dir.is_dir() or group_dir.name.startswith("_"):
            continue
        for run_dir in group_dir.iterdir():
            if run_dir.is_dir() and (run_dir / "metadata.json").is_file():
                yield run_dir


def queued_worker_runs(root: Path | None = None) -> list[Path]:
    pending: list[tuple[str, int, str, Path]] = []
    for run_dir in _iter_run_dirs(root):
        if not (run_dir / "worker_request.json").is_file():
            continue
        metadata = read_json(run_dir / "metadata.json")
        status = str(metadata.get("status") or "")
        if status != "queued" and not (
            status in {"running", "preparing"} and metadata.get("scheduler_dispatch_requested")
        ):
            continue
        if _pid_alive(metadata.get("worker_pid")):
            continue
        pending.append(
            (
                str(metadata.get("created_at") or run_dir.name),
                run_dir.stat().st_mtime_ns,
                str(run_dir),
                run_dir,
            )
        )
    return [row[3] for row in sorted(pending, key=lambda item: item[:3])]


def _resource_request_from_metadata(run_dir: Path, cpu_capacity: int) -> ResourceRequest:
    metadata, _, _ = _read_run_payloads(run_dir)
    allocation = metadata.get("resource_allocation") if isinstance(metadata.get("resource_allocation"), dict) else {}
    if allocation:
        request = request_for_run(run_dir, cpu_capacity)
        return ResourceRequest(
            cpu_cores=int(allocation.get("cpu_cores") or request.cpu_cores),
            gpu_resource=request.gpu_resource,
            gpu_device=str(allocation.get("gpu_device") or request.gpu_device or "") or None,
        )
    return request_for_run(run_dir, cpu_capacity)


def _active_allocations(cpu_capacity: int, root: Path | None = None) -> list[tuple[Path, ResourceRequest]]:
    allocations: list[tuple[Path, ResourceRequest]] = []
    now = datetime.now(timezone.utc)
    for run_dir in _iter_run_dirs(root):
        metadata = read_json(run_dir / "metadata.json")
        status = str(metadata.get("status") or "")
        worker_alive = _pid_alive(metadata.get("worker_pid"))
        shared_parent_text = str(metadata.get("queue_shared_with_run_dir") or "").strip()
        if shared_parent_text:
            shared_parent = Path(shared_parent_text)
            parent_metadata = read_json(shared_parent / "metadata.json")
            parent_status = str(parent_metadata.get("status") or "")
            parent_alive = _pid_alive(parent_metadata.get("worker_pid"))
            parent_recent = _run_age_seconds(parent_metadata, now) < ACTIVE_OWNER_STALE_SECONDS
            if parent_status in {"running", "preparing"} and (parent_alive or parent_recent):
                # Internal campaign steps run inside their parent's worker and
                # share its CPU/GPU allocation. Do not count both directories.
                continue
        if metadata.get("scheduler_dispatch_requested") and not worker_alive:
            # A few existing scheduler workflows set their parent to running
            # before requesting a background worker. It is not executing yet.
            continue
        if status not in {"running", "preparing"} and not (status == "queued" and worker_alive):
            continue
        if (
            status in {"running", "preparing"}
            and not worker_alive
            and not metadata.get("worker_pid")
            and _run_age_seconds(metadata, now) >= ACTIVE_OWNER_STALE_SECONDS
        ):
            # Ignore old active records without a live worker PID. Keep their
            # files and status intact so the user can inspect and resume them.
            continue
        if str(run_dir) in {str(path) for path, _ in allocations}:
            continue
        try:
            request = _resource_request_from_metadata(run_dir, cpu_capacity)
        except (TypeError, ValueError):
            request = ResourceRequest(cpu_cores=1, gpu_resource=None, gpu_device=None)
        allocations.append((run_dir, request))
        updated_at = str(metadata.get("updated_at") or metadata.get("created_at") or "")
        if status in {"running", "preparing"} and (run_dir / "worker_request.json").exists() and not worker_alive:
            # Worker PIDs can be absent for old jobs started synchronously by the UI.
            # Recover only runs which explicitly have a worker PID recorded.
            if metadata.get("worker_pid"):
                try:
                    age = (now - datetime.fromisoformat(updated_at)).total_seconds() if updated_at else 0
                except ValueError:
                    age = 0
                if age >= 10:
                    _mark_orphaned_worker(run_dir, metadata)
                    allocations.pop()
    return allocations


def _mark_orphaned_worker(run_dir: Path, metadata: dict) -> None:
    reason = "Background worker exited before recording a terminal job status. Review logs, then resume if appropriate."
    if not read_json(run_dir / "result.json"):
        finish_job(
            run_dir,
            False,
            {"metrics": {"worker_error": reason, "worker_exception_type": "WorkerProcessExited"}},
        )
        return
    update_status(run_dir, "failed", worker_error=reason, worker_exception_type="WorkerProcessExited")


def _available_for_gpu_request(request: ResourceRequest, devices: list[str], occupied: list[ResourceRequest]) -> tuple[bool, str | None]:
    if not request.gpu_resource:
        return True, None
    if not devices:
        return False, "Waiting for a visible NVIDIA GPU (nvidia-smi found none)."
    wanted = _gpu_indices(request.gpu_resource, request.gpu_device)
    if wanted is not None:
        missing = wanted.difference(devices)
        if missing:
            return False, f"Requested GPU device(s) unavailable: {', '.join(sorted(missing))}."
    if any(gpu_requests_conflict(request, current) for current in occupied):
        return False, "Waiting for the requested GPU allocation to become available."
    return True, None


def apply_docker_cpu_limit(command: list[str], run_dir: Path) -> list[str]:
    """Pass the scheduled CPU allocation into Docker and enforce a quota when supported."""
    if len(command) < 2 or command[0] != "docker" or command[1] != "run":
        return command
    metadata = read_json(run_dir / "metadata.json")
    allocation = metadata.get("resource_allocation") if isinstance(metadata.get("resource_allocation"), dict) else {}
    resource_request = metadata.get("resource_request") if isinstance(metadata.get("resource_request"), dict) else {}
    input_payload = read_json(run_dir / "input.json")
    params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
    try:
        cores = int(
            allocation.get("cpu_cores")
            or resource_request.get("cpu_cores")
            or params.get("cpu_cores")
            or params.get("cpu_slots")
            or params.get("num_threads")
            or os.getenv("MN_PROTEIN_DESIGN_JOB_CPU_CORES")
            or 0
        )
    except (TypeError, ValueError):
        cores = 0
    if cores < 1:
        return command
    options: list[str] = []
    if "--cpus" not in command and not any(value.startswith("--cpus=") for value in command):
        if _docker_cpu_quota_supported():
            options.extend(["--cpus", str(cores)])
    for name in CPU_THREAD_ENV_VARS:
        if _docker_env_is_set(command, name):
            continue
        options.extend(["--env", f"{name}={cores}"])
    return [*command[:2], *options, *command[2:]] if options else command


def _docker_env_is_set(command: list[str], name: str) -> bool:
    for index, value in enumerate(command):
        if value == "-e" or value == "--env":
            if index + 1 < len(command) and command[index + 1].split("=", 1)[0] == name:
                return True
        if value.startswith("--env=") and value[len("--env="):].split("=", 1)[0] == name:
            return True
    return False


def apply_docker_cpu_limits_to_steps(run_dir: Path, steps: list[dict]) -> list[dict]:
    """Return workflow steps with scheduler CPU settings applied to Docker commands."""
    prepared: list[dict] = []
    for step in steps:
        row = dict(step)
        command = row.get("command")
        if isinstance(command, list):
            row["command"] = apply_docker_cpu_limit(command, run_dir)
        prepared.append(row)
    return prepared


def _docker_cpu_quota_supported() -> bool:
    """Check whether the host exposes the CFS quota controls used by Docker --cpus."""
    cgroup_root = Path("/sys/fs/cgroup")
    if (cgroup_root / "cpu.max").is_file():
        return True
    return any(
        candidate.is_file()
        for candidate in (
            cgroup_root / "cpu" / "cpu.cfs_quota_us",
            cgroup_root / "cpu,cpuacct" / "cpu.cfs_quota_us",
        )
    )


def scheduler_snapshot() -> dict:
    return read_json(worker_service_status_path())


def _gpu_candidates(request: ResourceRequest, devices: list[str]) -> tuple[list[str], bool]:
    if not request.gpu_resource:
        return [], False
    raw = str(request.gpu_device or "all").strip().lower()
    if raw.startswith("device="):
        raw = raw.removeprefix("device=")
    if raw in {"", "all"} or (raw in {"auto", "any"} and not devices):
        return list(dict.fromkeys(str(device) for device in devices)), True
    if raw in {"auto", "any"}:
        return list(dict.fromkeys(str(device) for device in devices)), False
    if request.gpu_resource == "gpu" and raw == "all":
        return list(dict.fromkeys(str(device) for device in devices)), True
    selected = [part.strip() for part in raw.split(",") if part.strip()]
    return [device for device in dict.fromkeys(selected) if device in devices], len(selected) > 1


def _admission_request(request: ResourceRequest) -> AdmissionRequest:
    return AdmissionRequest(
        gpu=bool(request.gpu_resource),
        min_vram_gb=request.min_vram_gb,
        cpu_threads=request.cpu_cores,
        ram_gb=request.ram_gb,
        scratch_gb=request.scratch_gb,
    )


def _scratch_path() -> Path:
    configured = os.getenv("TMPDIR", "").strip()
    return Path(configured).expanduser() if configured else Path("/tmp")


class WorkerScheduler:
    def __init__(
        self,
        *,
        cpu_slots: int | None = None,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        gpu_devices: list[str] | None = None,
        root: Path | None = None,
        launcher: Callable[[Path, ResourceRequest], subprocess.Popen] | None = None,
    ) -> None:
        self.root = (root or runs_root()).expanduser().resolve()
        self.cpu_slots = configured_cpu_slots(cpu_slots)
        self.shared_cpu_capacity = cpu_pool_capacity()
        self.poll_seconds = max(0.25, float(poll_seconds))
        self.gpu_devices = gpu_devices
        self.launcher = launcher or self._launch_worker
        self.active: dict[str, ActiveWorker] = {}
        self._stop = threading.Event()
        self.started_at = utc_now()

    def _devices(self) -> list[str]:
        return detect_gpu_devices() if self.gpu_devices is None else list(self.gpu_devices)

    def _launch_worker(self, run_dir: Path, request: ResourceRequest) -> subprocess.Popen:
        logs_dir = run_dir
        stdout = (logs_dir / "worker_stdout.log").open("a")
        stderr = (logs_dir / "worker_stderr.log").open("a")
        env = os.environ.copy()
        for key in CPU_THREAD_ENV_VARS:
            env[key] = str(max(1, request.cpu_cores))
        env["MN_PROTEIN_DESIGN_JOB_CPU_CORES"] = str(max(1, request.cpu_cores))
        command = [sys.executable, "-m", "mn_protein_design.core.local_worker", "--run-dir", str(run_dir)]
        metadata = read_json(run_dir / "metadata.json")
        metadata["worker_started_at"] = utc_now()
        metadata["resource_allocation"] = {
            "cpu_cores": request.cpu_cores,
            "gpu_device": request.gpu_device or "cpu",
            "gpu_resource": request.gpu_resource,
            "min_vram_gb": request.min_vram_gb,
            "ram_gb": request.ram_gb,
            "scratch_gb": request.scratch_gb,
            "scheduler": "local-worker-service",
        }
        metadata.pop("scheduler_wait_reason", None)
        metadata["updated_at"] = metadata["worker_started_at"]
        write_json(run_dir / "metadata.json", metadata)
        try:
            proc = subprocess.Popen(
                command,
                cwd=str(Path(__file__).resolve().parents[2]),
                stdout=stdout,
                stderr=stderr,
                env=env,
                start_new_session=True,
            )
        finally:
            stdout.close()
            stderr.close()
        metadata = read_json(run_dir / "metadata.json")
        metadata["worker_pid"] = proc.pid
        metadata["updated_at"] = utc_now()
        write_json(run_dir / "metadata.json", metadata)
        return proc

    def _refresh_active(self) -> None:
        for key, worker in list(self.active.items()):
            return_code = worker.process.poll()
            if return_code is None:
                continue
            self.active.pop(key, None)
            if worker.memory_lease is not None:
                worker.memory_lease.release()
            if worker.cpu_lease is not None:
                worker.cpu_lease.release()
            for lease in worker.gpu_leases:
                lease.release()
            metadata = read_json(worker.run_dir / "metadata.json")
            if str(metadata.get("status") or "") in ACTIVE_STATUSES:
                if return_code != 0:
                    _mark_orphaned_worker(worker.run_dir, metadata)
                else:
                    # A successful runner should persist a terminal status itself.
                    update_status(worker.run_dir, "failed", worker_error="Worker exited without a terminal job status.")

    def _occupied_requests(self) -> list[ResourceRequest]:
        by_path: dict[str, ResourceRequest] = {}
        for worker in self.active.values():
            by_path[str(worker.run_dir)] = worker.request
        for run_dir, request in _active_allocations(self.cpu_slots, self.root):
            by_path.setdefault(str(run_dir), request)
        return list(by_path.values())

    def _set_wait_reason(self, run_dir: Path, reason: str | None) -> None:
        metadata = read_json(run_dir / "metadata.json")
        current = metadata.get("scheduler_wait_reason")
        if current == reason:
            return
        if reason:
            metadata["scheduler_wait_reason"] = reason
        else:
            metadata.pop("scheduler_wait_reason", None)
        metadata["updated_at"] = utc_now()
        write_json(run_dir / "metadata.json", metadata)

    def schedule_once(self) -> list[Path]:
        self._refresh_active()
        launched: list[Path] = []
        devices = self._devices()
        snapshot = capture_resource_snapshot(_scratch_path())
        state_root = shared_state_dir()
        pending = queued_worker_runs(self.root)
        for run_dir in pending:
            key = str(run_dir)
            if key in self.active:
                continue
            try:
                request = request_for_run(run_dir, self.cpu_slots)
            except ValueError as exc:
                self._set_wait_reason(run_dir, str(exc))
                continue
            metadata = read_json(run_dir / "metadata.json")
            if _pid_alive(metadata.get("worker_pid")):
                continue
            if request.cpu_cores > self.cpu_slots:
                self._set_wait_reason(
                    run_dir,
                    f"Needs {request.cpu_cores} CPU slots; shared scheduler capacity is {self.cpu_slots}. Set MN_COMPUTE_SCHEDULER_CPU_SLOTS to change it.",
                )
                continue
            candidates, all_devices = _gpu_candidates(request, devices)
            admission = assess_resource_admission(
                _admission_request(request),
                snapshot,
                candidate_gpu_ids=tuple(int(device) for device in candidates if device.isdigit()),
            )
            if not admission.allowed:
                self._set_wait_reason(run_dir, "; ".join(admission.reasons))
                continue
            if request.cpu_cores > 0:
                protein_cpu_use = sum(
                    int(row.get("cpu_threads") or 0)
                    for row in active_resource_reservations(state_root)
                    if row.get("app_id") == "mn-protein-design"
                )
                if protein_cpu_use + request.cpu_cores > self.cpu_slots:
                    self._set_wait_reason(
                        run_dir,
                        f"Waiting for CPU slots ({min(protein_cpu_use, self.cpu_slots)}/{self.cpu_slots} in use).",
                    )
                    continue
            owner_id = f"protein-design-{os.getpid()}-{run_dir.name}"
            memory_lease, memory_decision = acquire_memory_lease(
                _admission_request(request),
                snapshot,
                run_id=run_dir.name,
                owner_id=owner_id,
                app_id="mn-protein-design",
                run_dir=run_dir,
                state_root=state_root,
                scratch_path=_scratch_path(),
            )
            if memory_lease is None:
                self._set_wait_reason(run_dir, "; ".join(memory_decision.reasons))
                continue

            cpu_lease = None
            if request.cpu_cores > 0:
                cpu_lease = acquire_cpu_lease(
                    request.cpu_cores,
                    run_id=run_dir.name,
                    worker_id=owner_id,
                    workflow=str(read_json(run_dir / "metadata.json").get("tool") or ""),
                    app_id="mn-protein-design",
                    state_root=state_root,
                    capacity=self.shared_cpu_capacity,
                )
                if cpu_lease is None:
                    memory_lease.release()
                    used = sum(
                        int(row.get("cpu_threads") or 0)
                        for row in active_resource_reservations(state_root)
                    )
                    self._set_wait_reason(
                        run_dir,
                        f"Waiting for CPU slots ({min(used, self.shared_cpu_capacity)}/{self.shared_cpu_capacity} in use).",
                    )
                    continue

            gpu_leases: list[FileLease] = []
            chosen_devices: list[str] = []
            if request.gpu_resource:
                eligible = [str(gpu) for gpu in admission.eligible_gpu_ids]
                if all_devices:
                    wanted = candidates
                    if len(eligible) != len(wanted):
                        if cpu_lease is not None:
                            cpu_lease.release()
                        memory_lease.release()
                        self._set_wait_reason(run_dir, "Waiting until every requested GPU meets the VRAM requirement.")
                        continue
                    for device in wanted:
                        lease = acquire_gpu_lease(
                            int(device),
                            run_id=run_dir.name,
                            worker_id=owner_id,
                            workflow=str(read_json(run_dir / "metadata.json").get("tool") or ""),
                            app_id="mn-protein-design",
                            state_root=state_root,
                        )
                        if lease is None:
                            break
                        gpu_leases.append(lease)
                        chosen_devices.append(device)
                else:
                    for device in eligible:
                        lease = acquire_gpu_lease(
                            int(device),
                            run_id=run_dir.name,
                            worker_id=owner_id,
                            workflow=str(read_json(run_dir / "metadata.json").get("tool") or ""),
                            app_id="mn-protein-design",
                            state_root=state_root,
                        )
                        if lease is not None:
                            gpu_leases.append(lease)
                            chosen_devices.append(device)
                            break
                if not chosen_devices:
                    if cpu_lease is not None:
                        cpu_lease.release()
                    for lease in gpu_leases:
                        lease.release()
                    memory_lease.release()
                    self._set_wait_reason(run_dir, "Waiting for the requested GPU allocation to become available.")
                    continue
                if len(chosen_devices) != len(candidates) and all_devices:
                    for lease in gpu_leases:
                        lease.release()
                    if cpu_lease is not None:
                        cpu_lease.release()
                    memory_lease.release()
                    self._set_wait_reason(run_dir, "Waiting for all requested GPUs to become available.")
                    continue

            allocated_request = request
            if request.gpu_resource and not all_devices and chosen_devices:
                allocated_request = replace(request, gpu_device=chosen_devices[0])
            try:
                proc = self.launcher(run_dir, allocated_request)
            except Exception as exc:
                if cpu_lease is not None:
                    cpu_lease.release()
                for lease in gpu_leases:
                    lease.release()
                memory_lease.release()
                self._set_wait_reason(run_dir, f"Worker launch failed: {exc}")
                continue
            memory_lease.set_owner_pid(proc.pid)
            if cpu_lease is not None:
                cpu_lease.set_owner_pid(proc.pid)
            for lease in gpu_leases:
                lease.set_owner_pid(proc.pid)
            memory_lease.update_allocation(
                cpu_cores=request.cpu_cores,
                cpu_slots=list(cpu_lease.slot_ids) if cpu_lease is not None else [],
                gpu_devices=chosen_devices,
                run_dir=str(run_dir),
            )
            metadata = read_json(run_dir / "metadata.json")
            allocation = metadata.get("resource_allocation")
            if not isinstance(allocation, dict):
                allocation = {}
            allocation.update(
                {
                    "cpu_slots": list(cpu_lease.slot_ids) if cpu_lease is not None else [],
                    "gpu_devices": chosen_devices,
                    "ram_gb": request.ram_gb,
                    "scratch_gb": request.scratch_gb,
                    "allocated_at": utc_now(),
                }
            )
            metadata["resource_allocation"] = allocation
            metadata["updated_at"] = utc_now()
            write_json(run_dir / "metadata.json", metadata)
            self.active[key] = ActiveWorker(
                process=proc,
                run_dir=run_dir,
                request=allocated_request,
                memory_lease=memory_lease,
                cpu_lease=cpu_lease,
                gpu_leases=tuple(gpu_leases),
                gpu_devices=tuple(chosen_devices),
            )
            launched.append(run_dir)
            self._set_wait_reason(run_dir, None)
        return launched

    def stop(self) -> None:
        self._stop.set()

    def run_forever(self) -> None:
        while not self._stop.is_set():
            self.schedule_once()
            for worker in self.active.values():
                if worker.memory_lease is not None:
                    worker.memory_lease.heartbeat()
                if worker.cpu_lease is not None:
                    worker.cpu_lease.heartbeat()
                for lease in worker.gpu_leases:
                    lease.heartbeat()
            self.write_health()
            self._stop.wait(self.poll_seconds)
        self._refresh_active()
        self.write_health(stopping=True)

    def write_health(self, *, stopping: bool = False) -> None:
        active_workers = []
        for run_dir, request in _active_allocations(self.cpu_slots, self.root):
            metadata = read_json(run_dir / "metadata.json")
            active_workers.append(
                {
                    "run_dir": str(run_dir),
                    "pid": metadata.get("worker_pid"),
                    "status": metadata.get("status"),
                    "cpu_cores": request.cpu_cores,
                    "gpu_device": request.gpu_device or "cpu",
                }
            )
        status = {
            "pid": os.getpid(),
            "status": "stopping" if stopping else "running",
            "started_at": self.started_at,
            "heartbeat_at": utc_now(),
            "cpu_slots": self.cpu_slots,
            "shared_cpu_pool_capacity": self.shared_cpu_capacity,
            "visible_gpus": self._devices(),
            "active_workers": active_workers,
            "resource_snapshot": capture_resource_snapshot(_scratch_path()).to_dict(),
            "shared_resource_reservations": active_resource_reservations(shared_state_dir()),
        }
        write_json(worker_service_status_path(self.root), status)


def service_owner_pid() -> int | None:
    owner = read_json(worker_service_lock_path() / "owner.json")
    try:
        return int(owner.get("pid") or 0) or None
    except (TypeError, ValueError):
        return None


def _acquire_service_lock() -> bool:
    lock_dir = worker_service_lock_path()
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(4):
        try:
            lock_dir.mkdir()
        except FileExistsError:
            owner_pid = service_owner_pid()
            if owner_pid and _pid_alive(owner_pid):
                return False
            # Give another just-started service time to publish its PID.
            time.sleep(0.2)
            owner_pid = service_owner_pid()
            if owner_pid and _pid_alive(owner_pid):
                return False
            shutil.rmtree(lock_dir, ignore_errors=True)
            continue
        write_json(lock_dir / "owner.json", {"pid": os.getpid(), "started_at": utc_now()})
        return True
    return False


def run_worker_service(*, cpu_slots: int | None = None, poll_seconds: float = DEFAULT_POLL_SECONDS) -> int:
    worker_service_dir().mkdir(parents=True, exist_ok=True)
    if not _acquire_service_lock():
        owner_pid = service_owner_pid()
        raise RuntimeError(f"A worker service is already running (PID {owner_pid or 'unknown'}).")
    scheduler = WorkerScheduler(cpu_slots=cpu_slots, poll_seconds=poll_seconds)
    previous_handlers: dict[int, object] = {}

    def request_stop(signum, frame) -> None:
        scheduler.stop()

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, request_stop)
    try:
        scheduler.write_health()
        scheduler.run_forever()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        lock_dir = worker_service_lock_path()
        owner_pid = service_owner_pid()
        if owner_pid == os.getpid():
            shutil.rmtree(lock_dir, ignore_errors=True)
    return 0
