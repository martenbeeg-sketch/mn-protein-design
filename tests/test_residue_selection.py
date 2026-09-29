from __future__ import annotations

import pytest

from mn_protein_design.core.residue_selection import ResidueSelection, parse_selection


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", []),
        ("   ", []),
        ("A:10-30", [ResidueSelection("A", 10, 30)]),
        ("A:42", [ResidueSelection("A", 42, 42)]),
        (" A:1-1 ", [ResidueSelection("A", 1, 1)]),
        ("A:1-3,B:10-12", [ResidueSelection("A", 1, 3), ResidueSelection("B", 10, 12)]),
        ("A:1-3;B:10-12", [ResidueSelection("A", 1, 3), ResidueSelection("B", 10, 12)]),
        ("AA:5-7", [ResidueSelection("AA", 5, 7)]),
    ],
)
def test_parse_selection_accepts_supported_range_forms(text: str, expected: list[ResidueSelection]) -> None:
    assert parse_selection(text) == expected


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("A", "must look like"),
        (":1-4", "missing a chain ID"),
        ("A:x-y", "invalid literal"),
        ("A:9-3", "end before start"),
        ("A:1-2-3", "invalid literal"),
        ("A:", "invalid literal"),
        ("A:-2--1", "invalid literal"),
    ],
)
def test_parse_selection_rejects_malformed_ranges(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_selection(text)


@pytest.mark.parametrize(
    ("selection", "chain", "residue", "expected"),
    [
        (ResidueSelection("A", 2, 4), "A", 2, True),
        (ResidueSelection("A", 2, 4), "A", 4, True),
        (ResidueSelection("A", 2, 4), "A", 1, False),
        (ResidueSelection("A", 2, 4), "B", 3, False),
        (ResidueSelection("A", -2, 0), "A", -1, True),
    ],
)
def test_residue_selection_contains_checks_chain_and_inclusive_range(selection, chain: str, residue: int, expected: bool) -> None:
    assert selection.contains(chain, residue) is expected


def test_residue_selection_json_roundtrip_shape() -> None:
    assert ResidueSelection("Z", 5, 8).to_json() == {"chain_id": "Z", "start": 5, "end": 8}
