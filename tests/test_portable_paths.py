from __future__ import annotations

import json

from mn_protein_design.core.artifacts import Artifact
from mn_protein_design.core.candidates import read_candidates, write_candidates
from mn_protein_design.core.portable_paths import portable_path, resolve_managed_paths, resolve_stored_path


def test_portable_path_encodes_managed_roots_and_preserves_external_paths(monkeypatch, tmp_path) -> None:
    app_home = tmp_path / "app-home"
    runs = app_home / "workdir" / "runs"
    references = tmp_path / "references"
    run_dir = runs / "design" / "job-1"
    run_dir.mkdir(parents=True)
    references.mkdir()
    monkeypatch.setenv("MN_PROTEIN_DESIGN_APP_HOME", str(app_home))
    monkeypatch.setenv("MN_PROTEIN_DESIGN_RUN_DIR", str(runs))
    monkeypatch.setenv("MN_PROTEIN_DESIGN_REFERENCE_DIR", str(references))

    run_file = run_dir / "artifacts" / "target.pdb"
    reference_file = references / "targets" / "target.pdb"
    app_file = app_home / "workdir" / "targets" / "target.json"
    external_file = tmp_path / "external" / "target.pdb"

    assert portable_path(run_file, run_dir=run_dir) == "artifacts/target.pdb"
    assert portable_path("artifacts/target.pdb", run_dir=run_dir) == "artifacts/target.pdb"
    assert portable_path(reference_file) == "reference:///targets/target.pdb"
    assert portable_path(app_file) == "app:///workdir/targets/target.json"
    assert portable_path(external_file) == str(external_file.resolve())


def test_resolve_stored_path_handles_uris_relative_paths_and_legacy_paths(monkeypatch, tmp_path) -> None:
    app_home = tmp_path / "new-app-home"
    runs = tmp_path / "new-runs"
    references = tmp_path / "new-references"
    run_dir = runs / "design" / "job-1"
    run_dir.mkdir(parents=True)
    references.mkdir()
    monkeypatch.setenv("MN_PROTEIN_DESIGN_APP_HOME", str(app_home))
    monkeypatch.setenv("MN_PROTEIN_DESIGN_RUN_DIR", str(runs))
    monkeypatch.setenv("MN_PROTEIN_DESIGN_REFERENCE_DIR", str(references))

    (references / "target.pdb").touch()
    (run_dir / "artifact.pdb").touch()
    (runs / "design" / "old-job" / "artifact.pdb").parent.mkdir(parents=True)
    (runs / "design" / "old-job" / "artifact.pdb").touch()
    original = tmp_path / "old-machine" / "workdir" / "runs" / "design" / "old-job" / "artifact.pdb"
    original.parent.mkdir(parents=True)
    original.touch()

    assert resolve_stored_path("reference:///target.pdb") == references / "target.pdb"
    assert resolve_stored_path("artifact.pdb", run_dir=run_dir) == run_dir / "artifact.pdb"
    assert resolve_stored_path(
        "/old-machine/mn-protein-design-workdir/workdir/runs/design/old-job/artifact.pdb",
        must_exist=True,
    ) == runs / "design" / "old-job" / "artifact.pdb"
    assert resolve_stored_path(str(original), must_exist=True) == original
    assert resolve_stored_path("runs:///../../etc/passwd", must_exist=True) is None


def test_named_advanced_settings_preset_is_not_resolved_as_run_relative_path(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "design-campaign" / "job-1"
    run_dir.mkdir(parents=True)

    resolved = resolve_managed_paths(
        {"advanced_settings_file": "default_4stage_multimer_mpnn.json"},
        run_dir=run_dir,
    )

    assert resolved["advanced_settings_file"] == "default_4stage_multimer_mpnn.json"


def test_candidate_and_artifact_records_store_managed_references(monkeypatch, tmp_path) -> None:
    app_home = tmp_path / "app-home"
    runs = app_home / "workdir" / "runs"
    references = tmp_path / "references"
    run_dir = runs / "design" / "job-1"
    run_dir.mkdir(parents=True)
    references.mkdir()
    monkeypatch.setenv("MN_PROTEIN_DESIGN_APP_HOME", str(app_home))
    monkeypatch.setenv("MN_PROTEIN_DESIGN_RUN_DIR", str(runs))
    monkeypatch.setenv("MN_PROTEIN_DESIGN_REFERENCE_DIR", str(references))
    target = references / "target.pdb"
    target.touch()
    source_run = runs / "design" / "source-job"
    source_run.mkdir(parents=True)
    source_model = source_run / "artifacts" / "source.pdb"
    source_model.parent.mkdir()
    source_model.write_text("MODEL\n")

    write_candidates(run_dir, "test", [{
        "candidate_id": "c1",
        "target_pdb": str(target),
        "raw_metadata": {"source_run_dir": str(source_run), "source_candidate": {"complex_pdb": str(source_model)}},
    }])

    stored = json.loads((run_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl").read_text())
    assert stored["target_pdb"] == "reference:///target.pdb"
    assert stored["raw_metadata"]["source_run_dir"] == "runs:///design/source-job"
    assert stored["raw_metadata"]["source_candidate"]["complex_pdb"] == "runs:///design/source-job/artifacts/source.pdb"
    assert read_candidates(run_dir)[0]["target_pdb"] == str(target)
    assert read_candidates(run_dir)[0]["raw_metadata"]["source_run_dir"] == str(source_run)
    assert read_candidates(run_dir)[0]["raw_metadata"]["source_candidate"]["complex_pdb"] == str(source_model)
    assert Artifact("target", target, "pdb").to_json(run_dir)["path"] == "reference:///target.pdb"
