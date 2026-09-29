from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mn_protein_design.cli import app
from mn_protein_design.services.local_automation import AutomationError
from mn_protein_design.services import workflow_automation as automation


def _pdb_file(path: Path) -> Path:
    path.write_text(
        "ATOM      1  CA  ALA A  56      10.000  11.000  12.000  1.00 20.00           C\n"
        "ATOM      2  CA  GLY A  57      11.000  12.000  13.000  1.00 20.00           C\n"
        "END\n"
    )
    return path


def _write_request(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


def test_catalog_covers_major_workflow_groups_and_schema_is_versioned() -> None:
    rows = automation.workflow_catalog()
    ids = {row["workflow"] for row in rows}

    assert len(rows) >= 45
    assert {
        "target.prepare",
        "detection.scannet",
        "design.bindcraft",
        "design.bindcraft2",
        "sequence.ligandmpnn",
        "refolding.boltz2_complex",
        "analysis.rank",
        "import.table",
        "benchmark.refold_candidates",
        "benchmark.design_capacity",
    }.issubset(ids)

    schema = automation.workflow_schema("sequence.ligandmpnn")
    assert schema["schema_version"] == 1
    assert "source_job" in schema["request_fields"]
    assert "gpu_device" in {parameter["name"] for parameter in schema["parameters"]}
    refolding_schema = automation.workflow_schema("benchmark.refold_candidates")
    option_names = {parameter["name"] for parameter in refolding_schema["options_schema"]}
    assert "run_esmfold2" in option_names
    assert "require_monomer_success" not in option_names
    assert [row["workflow"] for row in automation.workflow_catalog(category="sequence")] == [
        "sequence.ligandmpnn",
        "sequence.foundry_mpnn",
        "sequence.pipeline",
    ]


def test_validate_resolves_paths_and_applies_resources_without_creating_jobs(tmp_path: Path) -> None:
    target = _pdb_file(tmp_path / "target.pdb")
    request = _write_request(
        tmp_path / "workflow.json",
        {
            "schema_version": 1,
            "workflow": "detection.scannet",
            "parameters": {"target_pdb": "target.pdb", "chain_ids": ["A"]},
            "resources": {"cpu_cores": 3, "gpu_device": "1"},
        },
    )

    report = automation.validate_workflow_request(request)

    assert report["valid"] is True
    assert report["parameters"]["target_pdb"] == str(target.resolve())
    assert report["parameters"]["gpu_device"] == "1"
    assert report["resources"] == {"cpu_cores": 3, "gpu_device": "1"}


def test_validate_candidate_workflow_supports_source_job_shorthand(tmp_path: Path, monkeypatch) -> None:
    source_run = tmp_path / "source-run"
    candidates = source_run / "artifacts" / "normalized_candidates" / "candidates.jsonl"
    candidates.parent.mkdir(parents=True)
    candidates.write_text('{"candidate_id":"one"}\n')
    monkeypatch.setattr(
        automation,
        "resolve_job",
        lambda reference: {
            "run_id": "run-1",
            "run_dir": str(source_run),
        },
    )
    request = _write_request(
        tmp_path / "sequence.json",
        {"schema_version": 1, "workflow": "sequence.ligandmpnn", "source_job": "ABCDE"},
    )

    report = automation.validate_workflow_request(request)

    assert report["parameters"]["source_run_dir"] == str(source_run.resolve())
    assert report["parameters"]["candidates_jsonl"] == str(candidates.resolve())


def test_validate_campaign_lineage_resolves_source_job_and_checks_steps(tmp_path: Path, monkeypatch) -> None:
    source_run = tmp_path / "source-run"
    candidates = source_run / "artifacts" / "normalized_candidates" / "candidates.jsonl"
    candidates.parent.mkdir(parents=True)
    candidates.write_text('{"candidate_id":"one"}\n')
    (source_run / "metadata.json").write_text(json.dumps({"task_group": "design", "tool": "rfdiffusion_classic"}))
    monkeypatch.setattr(
        automation,
        "resolve_job",
        lambda reference: {"run_id": "run-1", "run_dir": str(source_run), "job_code": "ABCDE"},
    )
    request = _write_request(
        tmp_path / "lineage.json",
        {
            "schema_version": 1,
            "workflow": "campaign.validation_sequence",
            "source_job": "ABCDE",
            "parameters": {
                "campaign_name": "local validation",
                "steps": [
                    {"module": "sequence_design", "tool": "ligandmpnn", "params": {"num_seq_per_target": 2}},
                    {"module": "analysis", "tool": "ranking", "params": {"keep_top_n": 20}},
                ],
            },
            "resources": {"cpu_cores": 4, "gpu_device": "0"},
        },
    )

    report = automation.validate_workflow_request(request)

    assert report["parameters"]["initial_source"]["run_dir"] == str(source_run.resolve())
    assert report["parameters"]["initial_source"]["candidates_jsonl"] == str(candidates.resolve())
    assert report["parameters"]["steps"][0]["tool"] == "ligandmpnn"
    assert report["parameters"]["cpu_cores"] == 4


def test_validate_campaign_lineage_rejects_unwired_step(tmp_path: Path) -> None:
    request = _write_request(
        tmp_path / "lineage.json",
        {
            "schema_version": 1,
            "workflow": "campaign.validation_sequence",
            "source_job": "ABCDE",
            "parameters": {"steps": [{"module": "unknown", "tool": "unknown"}]},
        },
    )
    with pytest.raises(AutomationError, match="Unsupported campaign step"):
        automation.validate_workflow_request(request)


def test_validate_refolding_options_do_not_collide_with_app_owned_arguments(tmp_path: Path) -> None:
    source_run = tmp_path / "source-run"
    source_run.mkdir()
    candidates = source_run / "candidates.jsonl"
    candidates.write_text("{}\n")
    request = _write_request(
        tmp_path / "refold.json",
        {
            "schema_version": 1,
            "workflow": "benchmark.refold_candidates",
            "parameters": {
                "source_run_dir": str(source_run),
                "candidates_jsonl": str(candidates),
                "options": {
                    "run_esmfold2": True,
                    "run_pyrosetta_input_metrics": True,
                    "pyrosetta_nprocs": 8,
                },
            },
            "resources": {"cpu_cores": 2, "gpu_device": "0"},
        },
    )

    report = automation.validate_workflow_request(request)

    assert report["parameters"]["options"]["gpu_device"] == "0"
    assert report["parameters"]["options"]["run_esmfold2"] is True
    assert report["parameters"]["options"]["pyrosetta_nprocs"] == 2
    assert report["resources"]["cpu_cores"] == 2


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"schema_version": 1, "workflow": "unknown.tool", "parameters": {}}, "Unknown workflow"),
        (
            {"schema_version": 1, "workflow": "detection.scannet", "parameters": {"target_pdb": "x.pdb", "bad": 2}},
            "Unsupported parameter",
        ),
        (
            {"schema_version": 1, "workflow": "detection.scannet", "parameters": {"target_pdb": "x.pdb", "chain_ids": [1]}},
            r"chain_ids\[\] must be a string",
        ),
        (
            {
                "schema_version": 1,
                "workflow": "detection.scannet",
                "parameters": {"target_pdb": "x.pdb", "chain_ids": ["A"], "gpu_device": "0"},
                "resources": {"gpu_device": "1"},
            },
            "conflicts with parameters.gpu_device",
        ),
        (
            {"schema_version": 1, "workflow": "target.mask", "parameters": {}, "source_job": "ABCDE"},
            "source_job is not supported",
        ),
    ],
)
def test_validate_rejects_unknown_or_inconsistent_requests(tmp_path: Path, payload: dict, error: str) -> None:
    _pdb_file(tmp_path / "x.pdb")
    request = _write_request(tmp_path / "invalid.json", payload)
    with pytest.raises(AutomationError, match=error):
        automation.validate_workflow_request(request)


def test_submit_uses_existing_worker_queue_with_allocations(tmp_path: Path, monkeypatch) -> None:
    target = _pdb_file(tmp_path / "target.pdb")
    request = _write_request(
        tmp_path / "workflow.json",
        {
            "schema_version": 1,
            "workflow": "detection.scannet",
            "parameters": {"target_pdb": str(target), "chain_ids": ["A"]},
            "resources": {"cpu_cores": 4, "gpu_device": "0"},
        },
    )
    captured: dict = {}
    run_dir = tmp_path / "runs" / "detection" / "run-1"

    def enqueue(function, **kwargs):
        captured["function"] = function
        captured.update(kwargs)
        return run_dir

    monkeypatch.setattr(automation, "enqueue_workflow_call", enqueue)
    monkeypatch.setattr(automation, "job_reference", lambda path: {"run_dir": str(path), "status": "queued"})

    result = automation.submit_workflow_request(request)

    assert result == {"workflow": "detection.scannet", "job": {"run_dir": str(run_dir), "status": "queued"}}
    assert captured["function"].__name__ == "run_scannet"
    assert captured["task_group"] == "detection"
    assert captured["kwargs"]["target_pdb"] == target.resolve()
    assert captured["cpu_cores"] == 4
    assert captured["gpu_device"] == "0"


def test_submit_custom_refolding_queues_and_wakes_worker(tmp_path: Path, monkeypatch) -> None:
    from mn_protein_design.workflows import benchmark

    source_run = tmp_path / "source-run"
    source_run.mkdir()
    candidates = source_run / "candidates.jsonl"
    candidates.write_text("{}\n")
    evaluation_run = tmp_path / "runs" / "benchmark" / "evaluation-1"
    evaluation_run.mkdir(parents=True)
    (evaluation_run / "metadata.json").write_text("{}")
    request = _write_request(
        tmp_path / "refold.json",
        {
            "schema_version": 1,
            "workflow": "benchmark.refold_candidates",
            "parameters": {
                "source_run_dir": str(source_run),
                "candidates_jsonl": str(candidates),
                "options": {"run_esmfold2": True},
            },
            "resources": {"cpu_cores": 2, "gpu_device": "0"},
        },
    )
    captured: dict = {}

    def enqueue(
        *,
        source_run_dir,
        candidates_jsonl,
        max_candidates=0,
        selected_candidate_ids=None,
        evaluation_name="Refolding evaluation",
        **kwargs,
    ):
        captured.update(
            {
                "source_run_dir": source_run_dir,
                "candidates_jsonl": candidates_jsonl,
                "max_candidates": max_candidates,
                "selected_candidate_ids": selected_candidate_ids,
                "evaluation_name": evaluation_name,
                **kwargs,
            }
        )
        return evaluation_run

    started: list[Path] = []
    monkeypatch.setattr(benchmark, "enqueue_candidate_refolding_evaluation", enqueue)
    monkeypatch.setattr(automation, "spawn_worker_for_run", lambda path: started.append(Path(path)))
    monkeypatch.setattr(automation, "job_reference", lambda path: {"run_dir": str(path), "status": "queued"})

    result = automation.submit_workflow_request(request)

    assert result["job"]["run_dir"] == str(evaluation_run)
    assert started == [evaluation_run]
    assert captured["run_esmfold2"] is True
    assert captured["gpu_device"] == "0"
    assert captured["task_group"] == "benchmark"
    metadata = json.loads((evaluation_run / "metadata.json").read_text())
    assert metadata["resource_request"]["cpu_cores"] == 2
    assert metadata["queue_resource"] == "gpu:0"


def test_cli_lists_and_validates_workflows(tmp_path: Path) -> None:
    listed = CliRunner().invoke(app, ["workflow", "list", "--category", "design"])
    assert listed.exit_code == 0, listed.output
    catalog = json.loads(listed.output)
    assert any(row["workflow"] == "design.bindcraft2" for row in catalog)

    _pdb_file(tmp_path / "target.pdb")
    request = _write_request(
        tmp_path / "workflow.json",
        {
            "schema_version": 1,
            "workflow": "detection.scannet",
            "parameters": {"target_pdb": "target.pdb", "chain_ids": ["A"]},
        },
    )
    validated = CliRunner().invoke(app, ["workflow", "validate", "--request", str(request)])
    assert validated.exit_code == 0, validated.output
    assert json.loads(validated.output)["valid"] is True
