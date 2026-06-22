from __future__ import annotations

import subprocess


def docker_gpu_args(gpu_device: object = "all") -> list[str]:
    """Return Docker GPU arguments for a selected host GPU/device set."""
    value = str(gpu_device if gpu_device is not None else "all").strip()
    if not value or value.lower() in {"all", "auto", "any"}:
        return ["--gpus", "all"]
    if value.lower() in {"none", "cpu", "off", "false", "0-gpu"}:
        return []
    if value.startswith("device="):
        return ["--gpus", value]
    return ["--gpus", f"device={value}"]


def normalize_gpu_device(gpu_device: object = "all") -> str:
    value = str(gpu_device if gpu_device is not None else "all").strip()
    return value or "all"


def gpu_queue_resource(gpu_device: object = "all") -> str | None:
    value = normalize_gpu_device(gpu_device).lower()
    if value in {"none", "cpu", "off", "false", "0-gpu"}:
        return None
    if value in {"all", "auto", "any"}:
        return "gpu"
    if value.startswith("device="):
        value = value.removeprefix("device=").strip()
    return f"gpu:{value or '0'}"


def available_gpu_devices() -> list[str]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=2,
        )
    except Exception:
        return ["0", "1", "all", "none"]
    devices = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not devices:
        return ["0", "1", "all", "none"]
    return [*devices, "all", "none"]
