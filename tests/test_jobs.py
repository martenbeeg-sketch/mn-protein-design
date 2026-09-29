from __future__ import annotations

from mn_protein_design.core.jobs import adopt_next_job_creation, create_job, finish_job, read_json, write_json


def test_create_job_writes_file_contract_and_selected_gpu(run_store) -> None:
    job = create_job(
        task_group="design",
        job_type="backbone_generation",
        tool="rfdiffusion",
        inputs={"target_id": "target-1"},
        params={"gpu_device": "1", "attempts": 2},
    )

    assert job.run_dir.is_relative_to(run_store)
    assert (job.run_dir / "artifacts").is_dir()
    assert (job.run_dir / "stdout.log").exists()
    assert (job.run_dir / "stderr.log").exists()
    assert read_json(job.run_dir / "input.json")["params"]["attempts"] == 2
    assert read_json(job.run_dir / "command.json")["mode"] == "internal"

    metadata = read_json(job.run_dir / "metadata.json")
    assert metadata["status"] == "queued"
    assert metadata["queue_resource"] == "gpu:1"


def test_cpu_job_does_not_claim_a_gpu_queue_resource(run_store) -> None:
    job = create_job(
        task_group="analysis",
        job_type="target_analysis",
        tool="local_analysis",
        inputs={},
        params={"gpu_device": "none"},
    )

    assert "queue_resource" not in read_json(job.run_dir / "metadata.json")


def test_finish_job_persists_result_and_terminal_status(run_store) -> None:
    job = create_job("analysis", "target_analysis", "local_analysis", {}, {})

    finish_job(
        job.run_dir,
        success=True,
        result={"outputs": {"summary": "artifacts/summary.json"}, "metrics": {"items": 3}},
    )

    result = read_json(job.run_dir / "result.json")
    metadata = read_json(job.run_dir / "metadata.json")
    assert result["success"] is True
    assert result["outputs"]["summary"] == "artifacts/summary.json"
    assert result["metrics"]["items"] == 3
    assert metadata["status"] == "completed"


def test_queued_workflow_adopts_first_created_job_and_preserves_resource_request(run_store) -> None:
    queued = create_job("design", "queued_workflow", "pending", {}, {"cpu_cores": 8})
    metadata = read_json(queued.run_dir / "metadata.json")
    metadata["resource_request"] = {"cpu_cores": 8}
    write_json(queued.run_dir / "metadata.json", metadata)

    with adopt_next_job_creation(queued):
        adopted = create_job(
            "design",
            "design_campaign",
            "rfdiffusion_classic",
            {"target_pdb": "/tmp/target.pdb"},
            {"gpu_device": "1"},
        )
        unrelated = create_job("design", "another_job", "another_tool", {}, {})

    assert adopted.run_dir == queued.run_dir
    assert unrelated.run_dir != queued.run_dir
    assert read_json(queued.run_dir / "input.json")["tool"] == "rfdiffusion_classic"
    assert read_json(queued.run_dir / "metadata.json")["resource_request"] == {"cpu_cores": 8}
