from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mn_protein_design.cli import app
from mn_protein_design.services import local_automation as automation


def _pdb_file(path: Path) -> Path:
    path.write_text(
        "ATOM      1  CA  ALA A  56      10.000  11.000  12.000  1.00 20.00           C\n"
        "ATOM      2  CA  GLY A  57      11.000  12.000  13.000  1.00 20.00           C\n"
        "ATOM      3  CA  GLY B   1      12.000  13.000  14.000  1.00 20.00           C\n"
        "END\n"
    )
    return path


def _request(target: str = "target.pdb") -> dict:
    return {
        "schema_version": 1,
        "campaign_name": "local automation test",
        "target_pdb": target,
        "target_chains": ["A"],
        "binder_length": "60-90",
        "hotspots": ["A56"],
        "design_attempts": 10,
        "engines": ["bindcraft", "bindcraft2"],
        "workflow_recipe": "vanilla",
        "gpu_device": "0",
        "engine_configs": {
            "bindcraft": {"filter_settings": "default_filters.json"},
            "bindcraft2": {"max_trajectories": 10, "modality": ["binder"]},
        },
    }


def test_validate_request_resolves_relative_target_and_normalizes_hotspots(tmp_path: Path) -> None:
    target = _pdb_file(tmp_path / "target.pdb")
    request_path = tmp_path / "campaign.json"
    normalized = automation.validate_campaign_request(_request(), request_file=request_path)

    assert normalized["target_pdb"] == target.resolve()
    assert normalized["target_chains"] == ["A"]
    assert normalized["hotspots"] == "A56"
    assert normalized["engines"] == ["bindcraft", "bindcraft2"]
    assert normalized["engine_configs"]["bindcraft2"]["max_trajectories"] == 10


def test_cli_validate_prints_machine_readable_request_without_submitting(tmp_path: Path) -> None:
    _pdb_file(tmp_path / "target.pdb")
    request_path = tmp_path / "campaign.json"
    request_path.write_text(json.dumps(_request()))

    result = CliRunner().invoke(app, ["campaign", "validate", "--request", str(request_path)])

    assert result.exit_code == 0, result.output
    response = json.loads(result.output)
    assert response["valid"] is True
    assert response["request"]["target_pdb"] == str((tmp_path / "target.pdb").resolve())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"schema_version": 2}, "schema_version must be 1"),
        ({"target_chains": ["Z"]}, r"Target chain\(s\) absent"),
        ({"hotspots": "A58"}, "does not exist in the target PDB"),
        ({"hotspots": "B1"}, "not in the selected target chains"),
        ({"workflow_recipe": "staged", "engines": ["bindcraft2"]}, "available only in the vanilla"),
        ({"unknown_option": 1}, "Unsupported request field"),
    ],
)
def test_validate_request_rejects_invalid_configuration(
    tmp_path: Path, change: dict, message: str
) -> None:
    _pdb_file(tmp_path / "target.pdb")
    payload = _request()
    payload.update(change)

    with pytest.raises(automation.AutomationError, match=message):
        automation.validate_campaign_request(payload, request_file=tmp_path / "campaign.json")


def test_submit_delegates_to_existing_campaign_workflow(tmp_path: Path, monkeypatch) -> None:
    _pdb_file(tmp_path / "target.pdb")
    request_path = tmp_path / "campaign.json"
    request_path.write_text(json.dumps(_request()))
    created: dict = {}
    run_dir = tmp_path / "runs" / "design-campaign" / "run-12345"
    row = {
        "job_code": "12345",
        "run_id": run_dir.name,
        "task_group": "design-campaign",
        "status": "queued",
        "run_dir": str(run_dir),
        "created_at": "now",
    }

    def create_campaign(**kwargs):
        created.update(kwargs)
        return run_dir

    monkeypatch.setattr(automation, "create_design_campaign", create_campaign)
    monkeypatch.setattr(automation.jobs, "collect_jobs", lambda **_: [row])
    result = automation.submit_design_campaign_request(request_path)

    assert result["run_id"] == run_dir.name
    assert created["target_pdb"] == (tmp_path / "target.pdb").resolve()
    assert created["design_attempts"] == 10
    assert created["engines"] == ["bindcraft", "bindcraft2"]


def test_job_lookup_results_and_hidden_child_visibility(monkeypatch, tmp_path: Path) -> None:
    run_dir = tmp_path / "job"
    run_dir.mkdir()
    row = {
        "job_code": "ABCDE",
        "run_id": "20260929-120000-abcde123",
        "task_group": "design-campaign",
        "status": "completed",
        "success": True,
        "run_dir": str(run_dir),
    }
    monkeypatch.setattr(automation.jobs, "collect_jobs", lambda **_: [row])
    monkeypatch.setattr(automation.jobs, "read_json", lambda path: {"success": True} if path.name == "result.json" else {})
    monkeypatch.setattr(automation, "read_candidates", lambda _path: [{"candidate_id": "candidate-1"}])
    monkeypatch.setattr(automation, "campaign_result_path", lambda _path: run_dir / "campaign_result.json")
    (run_dir / "campaign_result.json").write_text("{}")

    result = automation.job_results("abcde", include_candidates=True)
    assert result["job"]["run_id"] == row["run_id"]
    assert result["candidate_count"] == 1
    assert result["candidates"] == [{"candidate_id": "candidate-1"}]


def test_hidden_child_requires_explicit_lookup(monkeypatch, tmp_path: Path) -> None:
    child = {
        "job_code": "CHILD",
        "run_id": "child-run",
        "task_group": "design",
        "status": "running",
        "run_dir": str(tmp_path),
    }
    monkeypatch.setattr(
        automation.jobs,
        "collect_jobs",
        lambda **options: [child] if options.get("include_hidden") else [],
    )

    with pytest.raises(automation.AutomationError, match="No job matches"):
        automation.resolve_job("CHILD")
    assert automation.resolve_job("CHILD", include_hidden=True) == child


def test_cancel_only_accepts_active_jobs(monkeypatch, tmp_path: Path) -> None:
    row = {
        "job_code": "ABCDE",
        "run_id": "run-abcde",
        "task_group": "design-campaign",
        "status": "running",
        "run_dir": str(tmp_path),
    }
    stopped: list[tuple[Path, str]] = []
    monkeypatch.setattr(automation.jobs, "collect_jobs", lambda **_: [row])
    monkeypatch.setattr(automation.jobs, "stop_job", lambda path, reason: stopped.append((path, reason)))
    assert automation.cancel_job("ABCDE")["status"] == "running"
    assert stopped == [(tmp_path, "Cancelled through local automation CLI")]
    row["status"] = "completed"
    with pytest.raises(automation.AutomationError, match="only active or paused jobs"):
        automation.cancel_job("ABCDE")


def test_wait_returns_terminal_row_or_timeout(monkeypatch) -> None:
    rows = iter(
        [
            {"run_id": "run", "status": "running"},
            {"run_id": "run", "status": "completed", "success": True},
        ]
    )
    monkeypatch.setattr(automation, "resolve_job", lambda *_, **__: next(rows))
    monkeypatch.setattr(automation.time, "sleep", lambda _: None)
    result = automation.wait_for_job("run", timeout_seconds=1, poll_seconds=0.01)
    assert result and result["status"] == "completed"

    monkeypatch.setattr(automation, "resolve_job", lambda *_, **__: {"run_id": "run", "status": "running"})
    monkeypatch.setattr(automation.time, "monotonic", iter([0.0, 2.0]).__next__)
    assert automation.wait_for_job("run", timeout_seconds=0.5, poll_seconds=0.01) is None
