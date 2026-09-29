from __future__ import annotations

import inspect
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from mn_protein_design.core.jobs import create_job, read_json, write_json
from mn_protein_design.core.local_worker import spawn_worker_for_run
from mn_protein_design.core.portable_paths import portable_path
from mn_protein_design.core.scheduler import configured_cpu_slots


_INPUT_KEYS = {
    "target_pdb",
    "target_chains",
    "source_run_dir",
    "candidates_jsonl",
    "candidates_csv",
    "chain_ids",
    "chain_id",
    "candidate_ids",
}


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return portable_path(value)
    if isinstance(value, dict):
        return {str(key): _json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(child) for child in value]
    if isinstance(value, set):
        return sorted(_json_value(child) for child in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Workflow argument is not JSON serializable: {type(value).__name__}")


def _invocation_value(value: Any) -> Any:
    if isinstance(value, Path):
        return {"$path": portable_path(value)}
    if isinstance(value, dict):
        return {str(key): _invocation_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_invocation_value(child) for child in value]
    if isinstance(value, set):
        return sorted(_invocation_value(child) for child in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Workflow argument is not serializable for background execution: {type(value).__name__}")


def _bound_arguments(function: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(function)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return {
        name: value
        for name, value in bound.arguments.items()
        if name not in {"existing_job", "progress_callback"}
    }


def _nested_cpu_thread_requests(value: Any) -> list[int]:
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


def enqueue_workflow_call(
    function: Callable[..., Any],
    *,
    task_group: str,
    tool: str,
    job_type: str,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    cpu_cores: int | None = None,
    gpu_device: object | None = None,
) -> Path:
    """Queue one normal workflow function and let it create its job in this run directory."""
    module = str(getattr(function, "__module__", ""))
    function_name = str(getattr(function, "__name__", ""))
    if not module.startswith("mn_protein_design.workflows.") or not function_name.startswith("run_"):
        raise ValueError("Only app workflow run_* functions can be dispatched through the local worker.")

    call_kwargs = _bound_arguments(function, args, kwargs or {})
    signature = inspect.signature(function)
    if gpu_device is None:
        gpu_device = call_kwargs.get("gpu_device")
    device_choice = str(call_kwargs.get("device") or "").lower()
    if gpu_device is None and device_choice in {"cuda", "auto"}:
        gpu_device = "all"
    elif gpu_device is None and device_choice == "cpu":
        gpu_device = "none"
    elif gpu_device is None:
        probe_params = {**call_kwargs, "gpu_device": ""}
        from mn_protein_design.core.jobs import _queue_resource_for

        gpu_device = "all" if _queue_resource_for(tool, probe_params) else "none"
    normalized_gpu = str(gpu_device)

    requested_cores = cpu_cores
    if requested_cores is None:
        value = call_kwargs.get("cpu_cores")
        if value not in (None, ""):
            requested_cores = int(value)
        else:
            requested_cores = max(
                _nested_cpu_thread_requests(call_kwargs),
                default=min(4, configured_cpu_slots()),
            )
    if int(requested_cores) < 1:
        raise ValueError("Queued workflow CPU allocation must be at least one slot.")
    if "cpu_cores" in signature.parameters:
        call_kwargs["cpu_cores"] = int(requested_cores)

    params = {key: _json_value(value) for key, value in call_kwargs.items()}
    params["cpu_cores"] = int(requested_cores)
    params["gpu_device"] = normalized_gpu
    input_payload = {key: value for key, value in call_kwargs.items() if key in _INPUT_KEYS}
    job = create_job(
        task_group,
        job_type,
        tool,
        {key: _json_value(value) for key, value in input_payload.items()},
        params,
    )
    metadata = read_json(job.run_dir / "metadata.json")
    metadata["resource_request"] = {"cpu_cores": int(requested_cores)}
    write_json(job.run_dir / "metadata.json", metadata)
    positional_only = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY
    }
    positional_names = [name for name in call_kwargs if name in positional_only]
    serialized_kwargs = {
        name: _invocation_value(value)
        for name, value in call_kwargs.items()
        if name not in positional_only
    }
    serialized_args = [_invocation_value(call_kwargs[name]) for name in positional_names]
    write_json(
        job.run_dir / "worker_request.json",
        {
            "kind": "workflow_call",
            "module": module,
            "function": function_name,
            "args": serialized_args,
            "kwargs": serialized_kwargs,
            "resources": {"cpu_cores": int(requested_cores), "gpu_device": normalized_gpu},
        },
    )
    write_json(
        job.run_dir / "command.json",
        {"mode": "local_worker", "command": ["mn-protein-design", "worker", "--run-dir", str(job.run_dir)]},
    )
    spawn_worker_for_run(job.run_dir)
    return job.run_dir


def queued_workflow_alias(
    function: Callable[..., Any],
    *,
    task_group: str,
    tool: str,
    job_type: str,
) -> Callable[..., Path]:
    """Build a UI-facing alias that queues a workflow using its regular signature."""

    @wraps(function)
    def queue(*args: Any, **kwargs: Any) -> Path:
        requested_cpu_cores = kwargs.pop("queue_cpu_cores", None)
        return enqueue_workflow_call(
            function,
            task_group=task_group,
            tool=tool,
            job_type=job_type,
            args=args,
            kwargs=kwargs,
            cpu_cores=int(requested_cpu_cores) if requested_cpu_cores not in (None, "") else None,
        )

    return queue


def _restore_invocation_value(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"$path"}:
            return Path(str(value["$path"])).expanduser()
        return {key: _restore_invocation_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_restore_invocation_value(child) for child in value]
    return value


def run_workflow_call(run_dir: Path, request: dict[str, Any]) -> Path:
    """Execute a queued call; its first create_job() adopts the queue's run folder."""
    module_name = str(request.get("module") or "")
    function_name = str(request.get("function") or "")
    if not module_name.startswith("mn_protein_design.workflows.") or not function_name.startswith("run_"):
        raise ValueError("Worker request names an unsupported workflow function.")
    import importlib

    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise ValueError(f"Workflow function is unavailable: {module_name}.{function_name}")
    call_args = [_restore_invocation_value(value) for value in request.get("args") or []]
    call_kwargs = {
        key: _restore_invocation_value(value)
        for key, value in (request.get("kwargs") or {}).items()
    }
    from mn_protein_design.core.jobs import JobPaths, adopt_next_job_creation

    metadata = read_json(run_dir / "metadata.json")
    job = JobPaths(
        task_group=str(metadata.get("task_group") or run_dir.parent.name),
        run_id=run_dir.name,
        run_dir=run_dir,
    )
    with adopt_next_job_creation(job):
        result = function(*call_args, **call_kwargs)
    if isinstance(result, (list, tuple)):
        return Path(result[0]) if result else run_dir
    return Path(result) if isinstance(result, (str, Path)) else run_dir
