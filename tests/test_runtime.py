from __future__ import annotations

from mn_protein_design.runtime import app_home, reference_root, runs_root


def test_runtime_paths_follow_explicit_environment(monkeypatch, tmp_path) -> None:
    app_home_path = tmp_path / "app-home"
    runs_path = tmp_path / "runs"
    references_path = tmp_path / "references"
    monkeypatch.setenv("MN_PROTEIN_DESIGN_APP_HOME", str(app_home_path))
    monkeypatch.setenv("MN_PROTEIN_DESIGN_RUN_DIR", str(runs_path))
    monkeypatch.setenv("MN_PROTEIN_DESIGN_REFERENCE_DIR", str(references_path))

    assert app_home() == app_home_path
    assert runs_root() == runs_path
    assert reference_root() == references_path
