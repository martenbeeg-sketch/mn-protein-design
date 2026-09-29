from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from mn_protein_design.core.jobs import collect_jobs, create_job, finish_job, read_json, update_status, write_json
from mn_protein_design.core.portability import (
    PortabilityError,
    export_portable_workdir,
    import_portable_workdir,
    verify_portable_export,
)
from mn_protein_design.core.portable_paths import resolve_stored_path
from mn_protein_design.core.candidates import read_candidates


def _tree_digest(root: Path) -> dict[str, tuple[int, str]]:
    result = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        content = path.read_bytes()
        result[path.relative_to(root).as_posix()] = (len(content), hashlib.sha256(content).hexdigest())
    return result


def _configure_roots(monkeypatch, app_home: Path, runs: Path, references: Path) -> None:
    monkeypatch.setenv("MN_PROTEIN_DESIGN_APP_HOME", str(app_home))
    monkeypatch.setenv("MN_PROTEIN_DESIGN_RUN_DIR", str(runs))
    monkeypatch.setenv("MN_PROTEIN_DESIGN_REFERENCE_DIR", str(references))


def test_new_job_json_stores_managed_paths_portably_and_reads_them_locally(monkeypatch, tmp_path) -> None:
    app_home = tmp_path / "app-home"
    runs = app_home / "workdir" / "runs"
    references = tmp_path / "references"
    _configure_roots(monkeypatch, app_home, runs, references)
    (app_home / "workdir" / "targets").mkdir(parents=True)
    references.mkdir()
    app_target = app_home / "workdir" / "targets" / "target.pdb"
    reference_target = references / "reference.pdb"
    app_target.write_text("TARGET\n")
    reference_target.write_text("REFERENCE\n")

    job = create_job(
        "design",
        "test_design",
        "rfdiffusion",
        {"target_pdb": str(app_target), "reference_pdb": str(reference_target)},
        {"gpu_device": "none"},
    )
    staged_input = job.run_dir / "artifacts" / "input.pdb"
    staged_input.write_text("INPUT\n")
    run_root_input = job.run_dir / "root-input.pdb"
    run_root_input.write_text("ROOT INPUT\n")
    write_json(job.run_dir / "input.json", {
        "job_type": "test_design",
        "tool": "rfdiffusion",
        "inputs": {"target_pdb": str(app_target), "reference_pdb": str(reference_target), "target": "1abc/A"},
        "params": {"input_path": str(staged_input), "source_pdb": str(run_root_input)},
    })

    stored = json.loads((job.run_dir / "input.json").read_text())
    assert stored["inputs"]["target_pdb"] == "app:///workdir/targets/target.pdb"
    assert stored["inputs"]["reference_pdb"] == "reference:///reference.pdb"
    assert stored["params"]["input_path"] == "artifacts/input.pdb"
    assert stored["params"]["source_pdb"] == "root-input.pdb"
    assert read_json(job.run_dir / "input.json")["inputs"]["target_pdb"] == str(app_target)
    assert read_json(job.run_dir / "input.json")["inputs"]["reference_pdb"] == str(reference_target)
    assert read_json(job.run_dir / "input.json")["inputs"]["target"] == "1abc/A"
    assert read_json(job.run_dir / "input.json")["params"]["input_path"] == str(staged_input)
    assert read_json(job.run_dir / "input.json")["params"]["source_pdb"] == str(run_root_input)

    worker_request = job.run_dir / "worker_request.json"
    write_json(worker_request, {"kind": "test", "kwargs": {"target_pdb": str(app_target)}})
    assert json.loads(worker_request.read_text())["kwargs"]["target_pdb"] == "app:///workdir/targets/target.pdb"
    assert read_json(worker_request)["kwargs"]["target_pdb"] == str(app_target)


def test_export_import_preserves_source_and_jobs_page_reads_copied_data(monkeypatch, tmp_path) -> None:
    source_home = tmp_path / "source-home"
    source_runs = tmp_path / "source-runs"
    source_references = tmp_path / "source-references"
    workdir = source_home / "workdir"
    targets = workdir / "targets"
    targets.mkdir(parents=True)
    source_runs.mkdir()
    source_references.mkdir()
    _configure_roots(monkeypatch, source_home, source_runs, source_references)

    # The source installation has the default workdir/runs mount as well as a
    # separately configured active runs directory. Both roots must survive.
    legacy_run = workdir / "runs" / "analysis" / "legacy-job"
    legacy_run.mkdir(parents=True)
    (legacy_run / "metadata.json").write_text(json.dumps({
        "run_id": "legacy-job",
        "task_group": "analysis",
        "job_type": "legacy_analysis",
        "tool": "local_analysis",
        "status": "completed",
        "created_at": "2026-01-01T00:00:00+00:00",
    }))
    (legacy_run / "result.json").write_text(json.dumps({"success": True, "metrics": {"items": 1}}))
    workdir_target = targets / "target.pdb"
    workdir_target.write_text("WORKDIR TARGET\n")
    reference_target = source_references / "target.pdb"
    reference_target.write_text("REFERENCE TARGET\n")
    write_json(workdir / "target-record.json", {"target_pdb": str(workdir_target)})

    job = create_job("design", "backbone_generation", "rfdiffusion", {}, {"gpu_device": "none"})
    model = job.run_dir / "artifacts" / "model.pdb"
    model.write_text("MODEL\n")
    finish_job(job.run_dir, True, {"outputs": {"structure": "artifacts/model.pdb"}, "metrics": {"models": 1}})
    update_status(job.run_dir, "completed")
    # Represent historical input metadata with absolute paths. Export should
    # rewrite these only in its copy and leave this source tree byte-for-byte.
    (job.run_dir / "input.json").write_text(json.dumps({
        "job_type": "backbone_generation",
        "tool": "rfdiffusion",
        "inputs": {"target_pdb": str(workdir_target), "reference_pdb": str(reference_target)},
        "params": {"model_path": str(model)},
    }, indent=2) + "\n")
    (job.run_dir / "artifacts" / "normalized_candidates").mkdir(parents=True)
    (job.run_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl").write_text(
        json.dumps({
            "candidate_id": "candidate-1",
            "stage": "generation.backbone",
            "complex_pdb": str(model),
            "target_pdb": str(reference_target),
            "raw_metadata": {"source_run_dir": str(job.run_dir), "source_candidate": {"complex_pdb": str(model)}},
        }) + "\n"
    )

    source_before = _tree_digest(source_home) | {
        f"external-runs/{key}": value for key, value in _tree_digest(source_runs).items()
    } | {
        f"reference-root/{key}": value for key, value in _tree_digest(source_references).items()
    }
    bundle = tmp_path / "portable-bundle"
    export_report = export_portable_workdir(
        bundle,
        app_home_path=source_home,
        runs_path=source_runs,
        references_path=source_references,
    )
    assert export_report["verification"]["valid"] is True
    assert _tree_digest(source_home) | {
        f"external-runs/{key}": value for key, value in _tree_digest(source_runs).items()
    } | {
        f"reference-root/{key}": value for key, value in _tree_digest(source_references).items()
    } == source_before
    assert not (bundle / "reference_files").exists()
    exported_input = json.loads((bundle / "workdir" / "runs" / "design" / job.run_id / "input.json").read_text())
    assert exported_input["inputs"]["target_pdb"] == "app:///workdir/targets/target.pdb"
    assert exported_input["inputs"]["reference_pdb"] == "reference:///target.pdb"
    assert exported_input["params"]["model_path"] == f"runs:///design/{job.run_id}/artifacts/model.pdb"

    _configure_roots(monkeypatch, source_home, source_runs, source_references)
    assert read_json(job.run_dir / "input.json")["inputs"]["target_pdb"] == str(workdir_target)
    assert read_candidates(job.run_dir)[0]["complex_pdb"] == str(model)
    assert [row["run_id"] for row in collect_jobs()] == [job.run_id]

    destination_home = tmp_path / "destination-home"
    imported = import_portable_workdir(bundle, destination_home)
    destination_runs = Path(imported["runs_dir"])
    destination_references = tmp_path / "destination-references"
    shutil.copytree(source_references, destination_references)
    _configure_roots(monkeypatch, destination_home, destination_runs, destination_references)

    assert (destination_home / "migration-report.json").is_file()
    assert (destination_home / "workdir" / "target-record.json").is_file()
    assert read_json(destination_runs / "design" / job.run_id / "input.json")["inputs"]["target_pdb"] == str(destination_home / "workdir" / "targets" / "target.pdb")
    assert read_json(destination_runs / "design" / job.run_id / "input.json")["inputs"]["reference_pdb"] == str(destination_references / "target.pdb")
    assert resolve_stored_path(f"runs:///design/{job.run_id}/artifacts/model.pdb") == destination_runs / "design" / job.run_id / "artifacts" / "model.pdb"
    copied_candidates = read_candidates(destination_runs / "design" / job.run_id)
    assert copied_candidates[0]["complex_pdb"] == str(destination_runs / "design" / job.run_id / "artifacts" / "model.pdb")
    assert copied_candidates[0]["target_pdb"] == str(destination_references / "target.pdb")
    assert copied_candidates[0]["raw_metadata"]["source_run_dir"] == str(destination_runs / "design" / job.run_id)
    assert copied_candidates[0]["raw_metadata"]["source_candidate"]["complex_pdb"] == str(destination_runs / "design" / job.run_id / "artifacts" / "model.pdb")
    assert {row["run_id"] for row in collect_jobs()} == {job.run_id, "legacy-job"}

    app_test = AppTest.from_file(str(Path(__file__).parents[1] / "mn_protein_design" / "app" / "pages" / "jobs.py"))
    app_test.run()
    assert not app_test.exception
    assert any("Showing 2 of 2 jobs." in item.value for item in app_test.caption)


def test_export_rejects_active_jobs_and_external_operational_paths(monkeypatch, tmp_path) -> None:
    app_home = tmp_path / "app-home"
    runs = app_home / "workdir" / "runs"
    references = tmp_path / "references"
    outside = tmp_path / "outside.pdb"
    outside.write_text("OUTSIDE\n")
    _configure_roots(monkeypatch, app_home, runs, references)
    references.mkdir()
    job = create_job("design", "test_design", "rfdiffusion", {"target_pdb": str(outside)}, {})
    with pytest.raises(PortabilityError, match="queued, active, or paused"):
        export_portable_workdir(tmp_path / "active-bundle", app_home_path=app_home, runs_path=runs, references_path=references)

    update_status(job.run_dir, "completed")
    with pytest.raises(PortabilityError, match="outside the workdir/runs/reference roots"):
        export_portable_workdir(tmp_path / "external-bundle", app_home_path=app_home, runs_path=runs, references_path=references)


def test_verify_detects_tampering_and_import_refuses_existing_app_home(monkeypatch, tmp_path) -> None:
    app_home = tmp_path / "app-home"
    runs = app_home / "workdir" / "runs"
    references = tmp_path / "references"
    (app_home / "workdir").mkdir(parents=True)
    runs.mkdir()
    references.mkdir()
    _configure_roots(monkeypatch, app_home, runs, references)
    bundle = tmp_path / "bundle"
    export_portable_workdir(bundle, app_home_path=app_home, runs_path=runs, references_path=references)
    assert verify_portable_export(bundle)["valid"] is True

    destination = tmp_path / "existing-app-home"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep\n")
    with pytest.raises(PortabilityError, match="already exists"):
        import_portable_workdir(bundle, destination)
    assert sentinel.read_text() == "keep\n"

    (bundle / "workdir" / "changed.txt").write_text("tampered\n")
    verification = verify_portable_export(bundle)
    assert verification["valid"] is False
    assert any("inventory" in item["error"] or "SHA-256" in item["error"] for item in verification["errors"])


def test_export_refuses_conflicting_runs_without_changing_either_source(monkeypatch, tmp_path) -> None:
    app_home = tmp_path / "app-home"
    workdir_runs = app_home / "workdir" / "runs" / "design" / "same-job"
    configured_runs = tmp_path / "configured-runs" / "design" / "same-job"
    references = tmp_path / "references"
    workdir_runs.mkdir(parents=True)
    configured_runs.mkdir(parents=True)
    references.mkdir()
    _configure_roots(monkeypatch, app_home, tmp_path / "configured-runs", references)
    (workdir_runs / "result.json").write_text('{"success": true}\n')
    (configured_runs / "result.json").write_text('{"success": false}\n')
    before = {
        "workdir": _tree_digest(app_home),
        "runs": _tree_digest(tmp_path / "configured-runs"),
    }

    with pytest.raises(PortabilityError, match="conflicting files"):
        export_portable_workdir(
            tmp_path / "conflicting-bundle",
            app_home_path=app_home,
            runs_path=tmp_path / "configured-runs",
            references_path=references,
        )

    assert _tree_digest(app_home) == before["workdir"]
    assert _tree_digest(tmp_path / "configured-runs") == before["runs"]
