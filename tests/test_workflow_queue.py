from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import importlib

import pytest

from mn_protein_design.core.jobs import create_job, read_json, write_json
from mn_protein_design.core.scheduler import request_for_run
from mn_protein_design.core import workflow_queue


def test_enqueue_workflow_serializes_paths_and_resource_request(run_store, monkeypatch, tmp_path: Path) -> None:
    spawned: list[Path] = []

    def run_example(target_pdb: Path, gpu_device: str = "0", cpu_cores: int = 2) -> Path:
        return Path(target_pdb)

    run_example.__module__ = "mn_protein_design.workflows.test_stub"
    monkeypatch.setattr(workflow_queue, "spawn_worker_for_run", lambda run_dir: spawned.append(run_dir))
    target = tmp_path / "target.pdb"
    target.write_text("ATOM\n")

    run_dir = workflow_queue.enqueue_workflow_call(
        run_example,
        task_group="design",
        tool="example_gpu_tool",
        job_type="design_campaign",
        kwargs={"target_pdb": target, "gpu_device": "1"},
        cpu_cores=6,
    )

    request = read_json(run_dir / "worker_request.json")
    assert request["kind"] == "workflow_call"
    assert request["module"] == "mn_protein_design.workflows.test_stub"
    assert request["kwargs"]["target_pdb"]["$path"] == str(target)
    assert request["kwargs"]["cpu_cores"] == 6
    assert request["resources"] == {"cpu_cores": 6, "gpu_device": "1"}
    assert request_for_run(run_dir, cpu_capacity=32).cpu_cores == 6
    assert request_for_run(run_dir, cpu_capacity=32).gpu_device == "1"
    assert spawned == [run_dir]


def test_run_workflow_call_adopts_the_queued_run_folder(run_store, monkeypatch) -> None:
    queued = create_job("design", "queued", "placeholder", {}, {})

    def run_stub(target_name: str) -> Path:
        job = create_job("design", "design_campaign", "stub_engine", {"target_name": target_name}, {})
        return job.run_dir

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(run_stub=run_stub),
    )
    result = workflow_queue.run_workflow_call(
        queued.run_dir,
        {
            "module": "mn_protein_design.workflows.stub",
            "function": "run_stub",
            "args": [],
            "kwargs": {"target_name": "target-A"},
        },
    )

    assert result == queued.run_dir
    assert read_json(queued.run_dir / "input.json")["inputs"]["target_name"] == "target-A"
    assert read_json(queued.run_dir / "metadata.json")["tool"] == "stub_engine"


def test_enqueue_rejects_non_workflow_functions(tmp_path: Path) -> None:
    def run_fake() -> None:
        return None

    with pytest.raises(ValueError, match=r"Only app workflow run_\* functions"):
        workflow_queue.enqueue_workflow_call(
            run_fake,
            task_group="design",
            tool="fake",
            job_type="fake",
        )
