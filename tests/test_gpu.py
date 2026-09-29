from __future__ import annotations

import pytest

from mn_protein_design.core.gpu import (
    docker_gpu_args,
    gpu_queue_resource,
    normalize_gpu_device,
)


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("0", ["--gpus", "device=0"]),
        ("1", ["--gpus", "device=1"]),
        ("device=2", ["--gpus", "device=2"]),
        ("all", ["--gpus", "all"]),
        ("auto", ["--gpus", "all"]),
        ("none", []),
        ("cpu", []),
    ],
)
def test_docker_gpu_args(device: str, expected: list[str]) -> None:
    assert docker_gpu_args(device) == expected


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("0", "gpu:0"),
        ("device=1", "gpu:1"),
        ("all", "gpu"),
        ("auto", "gpu"),
        ("none", None),
        ("cpu", None),
    ],
)
def test_gpu_queue_resource(device: str, expected: str | None) -> None:
    assert gpu_queue_resource(device) == expected


def test_empty_gpu_device_normalizes_to_all() -> None:
    assert normalize_gpu_device("") == "all"
    assert normalize_gpu_device(None) == "all"
