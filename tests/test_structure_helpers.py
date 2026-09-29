from __future__ import annotations

from pathlib import Path

import pytest

from mn_protein_design.core.residue_selection import ResidueSelection
from mn_protein_design.core.structures import (
    detect_nonstandard_residues,
    download_pdb,
    filter_pdb_text,
    map_modified_residues_to_standard,
    pdb_chains,
    pdb_summary,
    residue_inventory,
    residues_within_spheres,
)


def _atom(serial: int, name: str, resname: str, chain: str, residue: int,
          xyz: tuple[float, float, float], record: str = "ATOM") -> str:
    x, y, z = xyz
    return (
        f"{record:<6}{serial:5d} {name:>4s} {resname:>3s} {chain}{residue:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}          {name[0]:>2s}"
    )


def _fixture_pdb() -> str:
    return "\n".join([
        "HEADER    TEST STRUCTURE",
        _atom(1, "CA", "ALA", "A", 1, (0, 0, 0)),
        _atom(2, "CA", "GLY", "A", 2, (4, 0, 0)),
        _atom(3, "CA", "SER", "B", 1, (1, 0, 0)),
        _atom(4, "O", "HOH", "A", 3, (2, 0, 0), "HETATM"),
        _atom(5, "C1", "LIG", "A", 9, (3, 0, 0), "HETATM"),
        "END", "",
    ])


def test_pdb_chains_returns_sorted_nonempty_chain_ids(tmp_path: Path) -> None:
    path = tmp_path / "structure.pdb"
    path.write_text(_fixture_pdb())
    assert pdb_chains(path) == ["A", "B"]


def test_pdb_summary_counts_atom_types_waters_and_unique_residues() -> None:
    summary = pdb_summary(_fixture_pdb())
    assert summary["atom_count"] == 3
    assert summary["hetero_count"] == 2
    assert summary["water_count"] == 1
    assert summary["chains"] == [
        {"chain_id": "A", "residue_count": 4, "start": 1, "end": 9, "residues": [1, 2, 3, 9]},
        {"chain_id": "B", "residue_count": 1, "start": 1, "end": 1, "residues": [1]},
    ]


@pytest.mark.parametrize(
    ("keep_chains", "remove_waters", "remove_hetero", "selection", "expected_records"),
    [({"A"}, True, False, [], 3), ({"B"}, True, False, [], 1),
     (None, True, True, [], 3), (None, False, False, [ResidueSelection("A", 2, 2)], 1)],
)
def test_filter_pdb_text_applies_chain_water_hetero_and_range_rules(
    keep_chains, remove_waters, remove_hetero, selection, expected_records: int
) -> None:
    filtered = filter_pdb_text(_fixture_pdb(), keep_chains=keep_chains, remove_waters=remove_waters,
                               remove_hetero=remove_hetero, residue_selections=selection)
    assert sum(line.startswith(("ATOM  ", "HETATM")) for line in filtered.splitlines()) == expected_records
    assert filtered.rstrip().endswith("END")


@pytest.mark.parametrize(
    ("diameter", "expected"),
    [(0, set()), (-1, set()), (2, {("A", 1), ("B", 1)}),
     (8, {("A", 1), ("A", 2), ("A", 9), ("B", 1)})],
)
def test_residue_sphere_selection_uses_ca_anchor_and_diameter(diameter: float, expected: set[tuple[str, int]]) -> None:
    assert residues_within_spheres(_fixture_pdb(), {("A", 1)}, diameter) == expected


def test_residue_sphere_selection_handles_unknown_seed() -> None:
    assert residues_within_spheres(_fixture_pdb(), {("Z", 999)}, 20) == set()


def test_residue_inventory_keeps_chain_resname_and_atom_counts() -> None:
    inventory = residue_inventory(_fixture_pdb())
    assert len(inventory) == 5
    water = next(row for row in inventory if row["resname"] == "HOH")
    assert water["chain"] == "A"
    assert water["atom_count"] == 1


def test_detect_nonstandard_residues_reports_protein_like_hetero_residue() -> None:
    text = "\n".join(
        _atom(i, name, "CAS", "A", 12, (float(i), 0, 0), "HETATM")
        for i, name in enumerate(["N", "CA", "C", "SG", "AS"], start=1)
    )
    assert detect_nonstandard_residues(text) == [
        {"record": "HETATM", "chain": "A", "residue_number": "12", "insertion_code": "",
         "resname": "CAS", "known_mapping": True, "mapped_to": "CYS"}
    ]


def test_known_modified_residue_mapping_converts_backbone_and_drops_unknown_atoms() -> None:
    text = "\n".join(
        _atom(i, name, "CAS", "A", 1, (i, 0, 0), "HETATM")
        for i, name in enumerate(["N", "CA", "C", "O", "CB", "SG", "AS"], start=1)
    )
    converted, report = map_modified_residues_to_standard(text)
    atom_lines = [line for line in converted.splitlines() if line.startswith("ATOM")]
    assert all(line[17:20] == "CYS" for line in atom_lines)
    assert len(atom_lines) == 6
    assert " AS " not in converted
    assert report["modified_residue_mapping_count"] == 1
    assert report["modified_residue_mappings"][0]["dropped_atoms"] == 1


@pytest.mark.parametrize("pdb_id", ["", "1AB", "12!4", "ABCDE"])
def test_download_pdb_validates_identifier_before_network(pdb_id: str, monkeypatch) -> None:
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("network should not be called for an invalid PDB id")

    monkeypatch.setattr("urllib.request.urlopen", fail_if_called)
    with pytest.raises(ValueError, match="four-character code"):
        download_pdb(pdb_id)


@pytest.mark.parametrize("text", ["", "not pdb", "HEADER only\n", "ATOM response\n"])
def test_modified_residue_mapping_is_a_noop_without_supported_residues(text: str) -> None:
    converted, report = map_modified_residues_to_standard(text)
    assert report["modified_residue_mapping_count"] == 0
    assert converted == (text.rstrip("\n") + "\n" if text else "")
