from __future__ import annotations

import json
from pathlib import Path

import pytest

from mn_protein_design.core.structures import pdb_summary
from mn_protein_design.workflows.target_prep import (
    analyze_target_chain,
    split_target_chain_fragments,
    target_chain_break_summary,
)


def _atom(serial: int, name: str, resname: str, chain: str, resseq: int, xyz: tuple[float, float, float]) -> str:
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d} {name:>4s} {resname:>3s} {chain:1s}{resseq:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}          {name[0]:>2s}"
    )


def _write_chain(path: Path, residue_groups: list[tuple[int, float]]) -> Path:
    lines: list[str] = []
    serial = 1
    for residue_number, x in residue_groups:
        for atom_name, offset in (("N", 0.0), ("CA", 0.5), ("C", 1.0), ("O", 1.2)):
            lines.append(_atom(serial, atom_name, "ALA", "A", residue_number, (x + offset, 0.0, 0.0)))
            serial += 1
    path.write_text("\n".join(lines + ["TER", "END", ""]))
    return path


@pytest.mark.parametrize(
    ("residues", "expected_reason", "expected_number_gaps", "expected_coordinate_breaks"),
    [
        ([(1, 0.0), (2, 3.0)], None, 0, 0),
        ([(1, 0.0), (3, 3.0)], "residue_number_gap", 1, 0),
        ([(1, 0.0), (2, 20.0)], "peptide_bond_gap", 0, 1),
        ([(1, 0.0), (2, 20.0), (3, 23.0)], "peptide_bond_gap", 0, 1),
    ],
)
def test_chain_break_summary_classifies_number_and_coordinate_gaps(
    tmp_path: Path, residues, expected_reason, expected_number_gaps, expected_coordinate_breaks
) -> None:
    pdb_path = _write_chain(tmp_path / "target.pdb", residues)

    summary = target_chain_break_summary(pdb_path, "A")

    assert summary["fragment_count"] == (1 if expected_reason is None else 2)
    assert summary["sequence_length"] == len(residues)
    assert summary["residue_number_break_count"] == expected_number_gaps
    assert summary["coordinate_break_count"] == expected_coordinate_breaks
    if expected_reason:
        assert summary["breaks"][0]["reason"] == expected_reason


@pytest.mark.parametrize("chain", ["B", "", "missing"])
def test_chain_break_summary_returns_empty_for_absent_chain(tmp_path: Path, chain: str) -> None:
    pdb_path = _write_chain(tmp_path / "target.pdb", [(1, 0.0)])

    summary = target_chain_break_summary(pdb_path, chain)

    assert summary["fragment_count"] == 0
    assert summary["break_count"] == 0
    assert summary["fragments"] == []


@pytest.mark.parametrize(
    ("residues", "warning_text"),
    [
        ([(1, 0.0), (3, 3.0)], "Residue numbering has explicit gaps"),
        ([(1, 0.0), (2, 20.0)], "Backbone coordinates are discontinuous"),
        ([(1, 0.0), (2, 20.0)], "Downstream folding should treat disconnected fragments"),
    ],
)
def test_target_analysis_reports_actionable_fragment_warnings(tmp_path: Path, residues, warning_text: str) -> None:
    pdb_path = _write_chain(tmp_path / "target.pdb", residues)

    report = analyze_target_chain(pdb_path, "A")

    assert any(warning_text in warning for warning in report["warnings"])


def test_target_analysis_identifies_likely_concatenated_domains(tmp_path: Path) -> None:
    pdb_path = _write_chain(tmp_path / "target.pdb", [(1, 0.0), (2, 30.0)])

    report = analyze_target_chain(pdb_path, "A")

    assert any("separate structural chains or domains" in warning for warning in report["warnings"])


def test_split_target_fragments_writes_role_map_and_stable_artifacts(tmp_path: Path, run_store: Path) -> None:
    source = _write_chain(tmp_path / "source.pdb", [(4, 0.0), (5, 3.0), (9, 8.0), (10, 11.0)])

    run_dir = split_target_chain_fragments(source, "A", target_name="PDL1", source_label="fixture")

    payload = json.loads((run_dir / "artifacts" / "target.json").read_text())
    split_path = run_dir / "artifacts" / "target_split_fragments.pdb"
    summary = pdb_summary(split_path.read_text())
    assert payload["target_name"] == "PDL1"
    assert payload["source_label"] == "fixture"
    assert payload["chain_roles"]["target_chains"] == ["A", "B"]
    assert payload["chain_roles"]["binder_chains"] == []
    assert [row["chain_id"] for row in summary["chains"]] == ["A", "B"]
    assert [row["residue_count"] for row in summary["chains"]] == [2, 2]
    assert payload["fragment_split"]["fragments"][1]["break_before"]["reason"] == "residue_number_gap"
    assert (run_dir / "artifacts" / "target_clean.pdb").read_text() == split_path.read_text()
    assert (run_dir / "artifacts" / "target_trimmed.pdb").read_text() == split_path.read_text()


@pytest.mark.parametrize("chain", ["", "B"])
def test_split_target_fragments_rejects_missing_or_unsplittable_chains(
    tmp_path: Path, run_store: Path, chain: str
) -> None:
    source = _write_chain(tmp_path / "single.pdb", [(1, 0.0), (2, 3.0)])

    with pytest.raises(ValueError):
        split_target_chain_fragments(source, chain)


def test_split_target_fragments_rejects_missing_source(tmp_path: Path, run_store: Path) -> None:
    with pytest.raises(FileNotFoundError):
        split_target_chain_fragments(tmp_path / "missing.pdb", "A")
