from __future__ import annotations

import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from mn_protein_design.core.candidates import write_candidates


APP_ROOT = Path(__file__).parents[1]
PAGES = [
    "analysis",
    "benchmark",
    "candidate_sets",
    "design",
    "design_campaigns",
    "jobs",
    "ppi_detection",
    "refolding",
    "results",
    "sequence_design",
    "settings",
    "target_preparation",
]


def _configure_empty_data_roots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    app_home = tmp_path / "app-home"
    runs = tmp_path / "runs"
    references = tmp_path / "references"
    app_home.mkdir()
    runs.mkdir()
    references.mkdir()
    monkeypatch.setenv("MN_PROTEIN_DESIGN_APP_HOME", str(app_home))
    monkeypatch.setenv("MN_PROTEIN_DESIGN_RUN_DIR", str(runs))
    monkeypatch.setenv("MN_PROTEIN_DESIGN_REFERENCE_DIR", str(references))


@pytest.mark.parametrize("page_name", PAGES)
def test_major_streamlit_page_starts_with_isolated_empty_data_roots(
    page_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure_empty_data_roots(monkeypatch, tmp_path)

    page = APP_ROOT / "mn_protein_design" / "app" / "pages" / f"{page_name}.py"
    app = AppTest.from_file(str(page)).run(timeout=30)

    assert not app.exception, "\n".join(str(error.message) for error in app.exception)


def test_candidate_sets_available_sets_tab_renders_when_selected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_empty_data_roots(monkeypatch, tmp_path)

    page = APP_ROOT / "mn_protein_design" / "app" / "pages" / "candidate_sets.py"
    app = AppTest.from_file(str(page))
    app.session_state["candidate_sets_tabs"] = "Available Sets"
    app.run(timeout=30)

    assert not app.exception, "\n".join(str(error.message) for error in app.exception)
    assert any(element.value == "Available Candidate Sets" for element in app.subheader)


@pytest.mark.parametrize(
    ("selected_tab", "expected_message"),
    [
        ("Target Analysis", "Select one target structure to inspect its fragment map."),
        ("Target Refolding", "Select one target structure to configure a refolding run."),
    ],
)
def test_target_preparation_lazy_panels_render_when_selected(
    selected_tab: str,
    expected_message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_empty_data_roots(monkeypatch, tmp_path)
    page = APP_ROOT / "mn_protein_design" / "app" / "pages" / "target_preparation.py"
    app = AppTest.from_file(str(page))
    app.session_state["target_preparation_tabs"] = selected_tab
    app.run(timeout=30)

    assert not app.exception, "\n".join(str(error.message) for error in app.exception)
    assert app.session_state["target_preparation_tabs"] == selected_tab
    assert any(element.value == expected_message for element in app.info)


def _pdb_atom(serial: int, atom: str, residue: str, chain: str, number: int, x: float, y: float) -> str:
    element = atom[0]
    return (
        f"ATOM  {serial:5d} {atom:>4s} {residue:>3s} {chain}{number:4d}    "
        f"{x:8.3f}{y:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}          {element:>2s}\n"
    )


def _write_scout_fixture_structures(run_dir: Path) -> tuple[Path, Path]:
    target = run_dir / "artifacts" / "target.pdb"
    complex_path = run_dir / "artifacts" / "candidate.pdb"
    target.parent.mkdir(parents=True)
    lines: list[str] = []
    serial = 1
    for chain, residues in (
        ("A", [(1, "ALA", 0.0), (2, "GLY", 4.0), (3, "SER", 8.0)]),
        ("Z", [(1, "GLY", 1.0), (2, "ALA", 1.5), (3, "GLY", 2.0)]),
    ):
        for resnum, residue, x in residues:
            for atom, offset in (("N", -0.4), ("CA", 0.0), ("C", 0.4), ("O", 0.7), ("CB", 0.0)):
                lines.append(_pdb_atom(serial, atom, residue, chain, resnum, x + offset, float(resnum)))
                serial += 1
        lines.append("TER\n")
    complex_path.write_text("".join(lines) + "END\n")
    target.write_text("".join(line for line in lines if line.startswith("ATOM") and line[21] == "A") + "END\n")
    return target, complex_path


def test_design_campaign_scout_plots_load_only_when_selected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _configure_empty_data_roots(monkeypatch, tmp_path)
    run_dir = tmp_path / "runs" / "design-campaign" / "lazy-fixture"
    run_dir.mkdir(parents=True)
    target, complex_path = _write_scout_fixture_structures(run_dir)
    (run_dir / "metadata.json").write_text(json.dumps({"task_group": "design-campaign", "status": "completed"}))
    (run_dir / "input.json").write_text(
        json.dumps({
            "job_type": "multi_engine_design_campaign",
            "inputs": {"target_pdb": "artifacts/target.pdb", "target_chains": ["A"]},
            "params": {"workflow_recipe": "engine_scout", "engines": ["bindcraft"], "hotspots": "A1"},
        })
    )
    (run_dir / "result.json").write_text(json.dumps({"status": "completed", "tool": "design_campaign", "metrics": {"engine_count": 1}}))
    summary_dir = run_dir / "artifacts" / "design_campaign"
    summary_dir.mkdir(parents=True)
    (summary_dir / "workflow_summary.json").write_text(json.dumps({"workflow_recipe": "engine_scout", "workflows": []}))
    write_candidates(
        run_dir,
        "bindcraft",
        [
            {
                "candidate_id": "fixture-001",
                "source_tool": "bindcraft",
                "stage": "generation.backbone",
                "binder_sequence": "GAG",
                "target_pdb": target,
                "complex_pdb": complex_path,
                "target_chains": ["A"],
                "binder_chains": ["Z"],
                "metrics": {
                    "result_kind": "generation_only",
                    "hotspot_atom_contact_fraction": 0.75,
                    "binder_helix_residues": 1,
                    "binder_sheet_residues": 1,
                    "binder_coil_residues": 1,
                },
            }
        ],
    )

    page = APP_ROOT / "mn_protein_design" / "app" / "pages" / "results.py"
    app = AppTest.from_file(str(page))
    app.query_params.update({"task_group": "design-campaign", "run_id": "lazy-fixture"})
    app.run(timeout=30)

    assert not app.exception, "\n".join(str(error.message) for error in app.exception)
    assert app.segmented_control[0].value == "Scout Overview"
    assert not any(element.label == "Candidates to summarize" for element in app.number_input)

    app.segmented_control[0].set_value("Scout Plots").run(timeout=30)

    assert not app.exception, "\n".join(str(error.message) for error in app.exception)
    assert any(element.label == "Candidates to summarize" for element in app.number_input)
