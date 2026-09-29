from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from mn_protein_design.core.jobs import create_job, read_json, update_status, write_json
from mn_protein_design.core.scheduler import (
    ResourceRequest,
    WorkerScheduler,
    _active_allocations,
    apply_docker_cpu_limit,
    configured_cpu_slots,
    gpu_requests_conflict,
    request_for_run,
)


class FakeProcess:
    next_pid = 100_000

    def __init__(self) -> None:
        self.pid = self.next_pid
        type(self).next_pid += 1
        self.returncode = None

    def poll(self) -> int | None:
        return self.returncode


def _queued_worker_job(task_group: str, tool: str, params: dict) -> Path:
    job = create_job(task_group, "test_job", tool, {}, params)
    write_json(job.run_dir / "worker_request.json", {"kind": "test_worker", "kwargs": {}})
    return job.run_dir


def test_cpu_only_jobs_use_the_configured_cpu_capacity(run_store) -> None:
    first = _queued_worker_job("analysis", "local_analysis", {"gpu_device": "none"})
    second = _queued_worker_job("analysis", "local_analysis", {"gpu_device": "none"})
    processes: list[FakeProcess] = []

    def launch(run_dir: Path, request: ResourceRequest) -> FakeProcess:
        assert request.cpu_cores == 2
        process = FakeProcess()
        processes.append(process)
        return process

    scheduler = WorkerScheduler(cpu_slots=2, gpu_devices=[], root=run_store, launcher=launch)

    first_batch = scheduler.schedule_once()
    assert len(first_batch) == 1
    assert first_batch[0] in {first, second}
    waiting_run = ({first, second} - set(first_batch)).pop()
    assert scheduler.schedule_once() == []
    assert read_json(waiting_run / "metadata.json")["scheduler_wait_reason"] == "Waiting for CPU slots (2/2 in use)."
    assert processes[0].poll() is None


def test_gpu_jobs_share_distinct_devices_but_wait_for_the_same_device(run_store) -> None:
    gpu_zero_a = _queued_worker_job("design", "rfdiffusion", {"gpu_device": "0"})
    gpu_zero_b = _queued_worker_job("design", "rfdiffusion", {"gpu_device": "0"})
    gpu_one = _queued_worker_job("design", "rfdiffusion", {"gpu_device": "1"})
    launched: list[Path] = []

    def launch(run_dir: Path, request: ResourceRequest) -> FakeProcess:
        launched.append(run_dir)
        return FakeProcess()

    scheduler = WorkerScheduler(
        cpu_slots=12,
        gpu_devices=["0", "1"],
        root=run_store,
        launcher=launch,
    )

    first_batch = scheduler.schedule_once()
    assert len(first_batch) == 2
    assert gpu_one in first_batch
    assert set(first_batch).intersection({gpu_zero_a, gpu_zero_b})
    waiting_gpu_zero = ({gpu_zero_a, gpu_zero_b} - set(first_batch)).pop()
    assert launched == first_batch
    assert "requested GPU allocation" in read_json(waiting_gpu_zero / "metadata.json")["scheduler_wait_reason"]

    scheduler.write_health()
    health = read_json(run_store / "_worker_service" / "status.json")
    reservations = health["shared_resource_reservations"]
    assert len(reservations) == 2
    assert {row["app_id"] for row in reservations} == {"mn-protein-design"}
    assert {row["run_id"] for row in reservations} == {path.name for path in first_batch}
    assert {tuple(row["gpu_devices"]) for row in reservations} == {("0",), ("1",)}
    assert all(row["cpu_threads"] == 4 for row in reservations)


def test_all_gpu_request_conflicts_with_a_specific_device() -> None:
    all_gpus = ResourceRequest(cpu_cores=1, gpu_resource="gpu", gpu_device="all")
    gpu_one = ResourceRequest(cpu_cores=1, gpu_resource="gpu:1", gpu_device="1")

    assert gpu_requests_conflict(all_gpus, gpu_one)
    assert gpu_requests_conflict(gpu_one, all_gpus)


def test_scheduler_dispatches_legacy_parent_marked_running_before_worker_start(run_store) -> None:
    run_dir = _queued_worker_job("benchmark", "local_analysis", {"gpu_device": "none"})
    update_status(run_dir, "running")
    metadata = read_json(run_dir / "metadata.json")
    metadata["scheduler_dispatch_requested"] = True
    write_json(run_dir / "metadata.json", metadata)
    launched: list[Path] = []

    def launch(path: Path, request: ResourceRequest) -> FakeProcess:
        launched.append(path)
        return FakeProcess()

    scheduler = WorkerScheduler(cpu_slots=2, gpu_devices=[], root=run_store, launcher=launch)

    assert scheduler.schedule_once() == [run_dir]
    assert launched == [run_dir]


def test_capacity_coordinator_does_not_consume_child_cpu_slots(run_store) -> None:
    coordinator = create_job(
        "benchmark",
        "refolding_capacity_benchmark",
        "refolding_capacity_matrix",
        {},
        {"queue_resource": ""},
    )
    write_json(coordinator.run_dir / "worker_request.json", {"kind": "test_worker"})
    child = _queued_worker_job("benchmark", "colabfold", {"gpu_device": "0"})
    assert request_for_run(coordinator.run_dir, cpu_capacity=1).cpu_cores == 0
    launched: list[Path] = []

    def launch(run_dir: Path, request: ResourceRequest) -> FakeProcess:
        launched.append(run_dir)
        return FakeProcess()

    scheduler = WorkerScheduler(cpu_slots=1, gpu_devices=["0"], root=run_store, launcher=launch)

    assert set(scheduler.schedule_once()) == {coordinator.run_dir, child}
    assert set(launched) == {coordinator.run_dir, child}


def test_resource_request_and_docker_command_use_allocated_cpu_limit(run_store, monkeypatch) -> None:
    run_dir = _queued_worker_job("analysis", "local_analysis", {"gpu_device": "none"})
    request = request_for_run(run_dir, cpu_capacity=3)
    assert request.cpu_cores == 3
    assert request.gpu_resource is None

    write_json(
        run_dir / "metadata.json",
        {
            **read_json(run_dir / "metadata.json"),
            "resource_allocation": {"cpu_cores": 3},
        },
    )
    command = ["docker", "run", "--rm", "image", "python", "run.py"]
    monkeypatch.setattr("mn_protein_design.core.scheduler._docker_cpu_quota_supported", lambda: True)

    assert apply_docker_cpu_limit(command, run_dir) == [
        "docker",
        "run",
        "--cpus",
        "3",
        "--env",
        "OMP_NUM_THREADS=3",
        "--env",
        "MKL_NUM_THREADS=3",
        "--env",
        "OPENBLAS_NUM_THREADS=3",
        "--env",
        "NUMEXPR_NUM_THREADS=3",
        "--env",
        "VECLIB_MAXIMUM_THREADS=3",
        "--env",
        "BLIS_NUM_THREADS=3",
        "--rm",
        "image",
        "python",
        "run.py",
    ]


def test_docker_cpu_limit_is_skipped_without_cfs_quota_support(run_store, monkeypatch) -> None:
    run_dir = _queued_worker_job("analysis", "local_analysis", {"gpu_device": "none"})
    write_json(
        run_dir / "metadata.json",
        {**read_json(run_dir / "metadata.json"), "resource_allocation": {"cpu_cores": 4}},
    )
    command = ["docker", "run", "--rm", "image", "python", "run.py"]
    monkeypatch.setattr("mn_protein_design.core.scheduler._docker_cpu_quota_supported", lambda: False)

    limited = apply_docker_cpu_limit(command, run_dir)
    assert "--cpus" not in limited
    assert "--env" in limited
    assert "OMP_NUM_THREADS=4" in limited


def test_default_worker_cpu_slots_scale_to_host_capacity(monkeypatch) -> None:
    monkeypatch.delenv("MN_PROTEIN_DESIGN_WORKER_CPU_SLOTS", raising=False)
    monkeypatch.setattr("mn_protein_design.core.scheduler.os.sched_getaffinity", lambda _pid: set(range(64)))
    assert configured_cpu_slots() == 63

    monkeypatch.setattr("mn_protein_design.core.scheduler.os.sched_getaffinity", lambda _pid: set(range(8)))
    assert configured_cpu_slots() == 7


def test_worker_cpu_cap_never_exceeds_the_shared_pool(monkeypatch) -> None:
    monkeypatch.setenv("MN_COMPUTE_SCHEDULER_CPU_SLOTS", "4")

    assert configured_cpu_slots(8) == 4
    assert configured_cpu_slots(2) == 2


def test_thread_count_parameters_request_enough_cpu_slots(run_store) -> None:
    run_dir = _queued_worker_job(
        "refolding-validation",
        "refolding_evaluation",
        {"gpu_device": "0", "metrics": {"pyrosetta_nprocs": 12}},
    )
    assert request_for_run(run_dir, cpu_capacity=32).cpu_cores == 12


def test_run_resource_request_includes_ram_scratch_and_gpu_memory(run_store) -> None:
    run_dir = _queued_worker_job("design", "rfdiffusion", {"gpu_device": "0"})
    write_json(
        run_dir / "worker_request.json",
        {
            "kind": "test_worker",
            "resources": {
                "cpu_cores": 6,
                "min_vram_gb": 12,
                "ram_gb": 24,
                "scratch_gb": 80,
            },
        },
    )

    request = request_for_run(run_dir, cpu_capacity=16)

    assert request == ResourceRequest(
        cpu_cores=6,
        gpu_resource="gpu:0",
        gpu_device="0",
        min_vram_gb=12,
        ram_gb=24,
        scratch_gb=80,
    )


def test_child_step_shares_parent_scheduler_allocation(run_store) -> None:
    parent = _queued_worker_job("design", "bindcraft", {"gpu_device": "0"})
    child = _queued_worker_job("design", "proteinmpnn", {"gpu_device": "0"})
    parent_metadata = read_json(parent / "metadata.json")
    parent_metadata.update(
        {
            "status": "running",
            "worker_pid": 99999999,
            "resource_allocation": {"cpu_cores": 8, "gpu_device": "0", "gpu_resource": "gpu:0"},
        }
    )
    write_json(parent / "metadata.json", parent_metadata)
    child_metadata = read_json(child / "metadata.json")
    child_metadata.update({"status": "running", "queue_shared_with_run_dir": str(parent)})
    write_json(child / "metadata.json", child_metadata)

    allocations = _active_allocations(32, run_store)

    assert [run_dir for run_dir, _request in allocations] == [parent]


def test_stale_active_record_without_worker_pid_does_not_hold_resources(run_store) -> None:
    old_run = create_job("benchmark", "legacy_job", "legacy_gpu_tool", {}, {"gpu_device": "0"})
    old_time = (datetime.now(timezone.utc) - timedelta(hours=7)).isoformat()
    metadata = read_json(old_run.run_dir / "metadata.json")
    metadata.update({"status": "running", "created_at": old_time, "updated_at": old_time})
    write_json(old_run.run_dir / "metadata.json", metadata)

    assert _active_allocations(32, run_store) == []
    assert read_json(old_run.run_dir / "metadata.json")["status"] == "running"
