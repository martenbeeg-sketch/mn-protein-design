from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mn_protein_design.core.candidates import read_candidates
from mn_protein_design.workflows import candidate_import as importer


def _pdb_atom(serial: int, name: str, resname: str, chain: str, residue: int, x: float) -> str:
    return (
        f"ATOM  {serial:5d} {name:>4s} {resname:>3s} {chain}{residue:4d}    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}          {name[0]:>2s}"
    )


def _complex_pdb(path: Path) -> Path:
    lines = []
    serial = 1
    for chain, residue, resname, x in [("A", 1, "ALA", 0.0), ("Z", 1, "GLY", 4.0)]:
        for atom, offset in [("N", 0.0), ("CA", 0.5), ("C", 1.0)]:
            lines.append(_pdb_atom(serial, atom, resname, chain, residue, x + offset))
            serial += 1
    path.write_text("\n".join(lines + ["END", ""]))
    return path


@pytest.mark.parametrize(
    ("value", "expected"),
    [(" ACd-ef 12* ", "ACDEF"), (None, ""), (float("nan"), ""), ("", ""), ("m.k l?", "MKL")],
)
def test_clean_sequence_normalizes_text(value, expected: str) -> None:
    assert importer._clean_sequence(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, []), (float("nan"), []), ("", []), ("A, Z;B", ["A", "Z", "B"])],
)
def test_split_chains_handles_empty_cells_and_separators(value, expected: list[str]) -> None:
    assert importer._split_chains(value) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("  candidate / 1 ", "candidate___1"), (None, "candidate"), (float("nan"), "candidate"),
     ("", "candidate"), ("A.b-c_1", "A.b-c_1")],
)
def test_safe_name_makes_candidate_ids_filesystem_safe(raw, expected: str) -> None:
    assert importer._safe_name(raw) == expected


@pytest.mark.parametrize(
    ("path", "candidate_id", "rank"),
    [
        ("ranked/001_binder-x.pdb", "binder-x", 1),
        ("ranked/12_binder-x.cif", "binder-x", 12),
        ("accepted/binder-x_rank_001.pdb", "binder-x", None),
        ("binder-x_model.pdb", "binder-x", None),
        ("nested/4_binder-x.pdb", "4_binder-x", None),
    ],
)
def test_candidate_ids_preserve_rank_folder_semantics(path: str, candidate_id: str, rank: int | None) -> None:
    candidate_path = Path(path)
    assert importer._candidate_id_from_path(candidate_path) == candidate_id
    ranked = importer._ranked_candidate_from_path(candidate_path)
    assert (ranked[0] if ranked else None) == rank


@pytest.mark.parametrize(
    ("complex_chains", "binder_chains", "explicit", "expected"),
    [
        (["A", "Z"], ["Z"], [], ["A"]),
        (["A", "B", "Z"], ["Z"], ["B"], ["B"]),
        (["Z"], ["Z"], ["A"], []),
        ([], ["Z"], [], ["B"]),
    ],
)
def test_target_chain_inference_respects_explicit_roles(complex_chains, binder_chains, explicit, expected) -> None:
    assert importer._target_chain_list(
        complex_chains=complex_chains,
        binder_chains=binder_chains,
        explicit_target_chains=explicit,
    ) == expected


@pytest.mark.parametrize(
    ("filename", "data", "expected_columns"),
    [
        ("candidates.csv", b"id,seq\na,ACD\n", ["id", "seq"]),
        ("candidates.tsv", b"id\tseq\na\tACD\n", ["id", "seq"]),
        ("candidates.txt", b"id\tseq\na\tACD\n", ["id", "seq"]),
    ],
)
def test_read_table_accepts_csv_and_delimited_uploads(filename, data: bytes, expected_columns: list[str]) -> None:
    table = importer._read_table(data=data, filename=filename)
    assert list(table.columns) == expected_columns
    assert table.iloc[0].to_dict() == {"id": "a", "seq": "ACD"}


def test_copy_or_link_can_copy_or_symlink_structure(tmp_path: Path) -> None:
    source = tmp_path / "source.pdb"
    source.write_text("PDB\n")
    copied = importer._copy_or_link(source, tmp_path / "copy" / "structure.pdb", copy_files=True)
    linked = importer._copy_or_link(source, tmp_path / "link" / "structure.pdb", copy_files=False)

    assert copied.read_text() == "PDB\n"
    assert not copied.is_symlink()
    assert linked.resolve() == source.resolve()
    assert linked.is_symlink() or linked.read_text() == "PDB\n"


def test_structure_discovery_filters_hidden_and_non_structure_files(tmp_path: Path) -> None:
    for name in ["a.pdb", "b.cif", "c.mmcif", "notes.csv", ".hidden.pdb"]:
        (tmp_path / name).write_text("data\n")

    assert [path.name for path in importer._find_structure_files(tmp_path, "*")] == ["a.pdb", "b.cif", "c.mmcif"]


@pytest.mark.parametrize("target_chains", [None, [], ["A"], ["A", "B"]])
@pytest.mark.parametrize("binder_chains", [None, [], ["Z"]])
def test_generic_import_persists_sequence_roles_metrics_and_source_table(
    target_chains, binder_chains, tmp_path: Path, run_store: Path
) -> None:
    table = pd.DataFrame(
        [
            {"name": "same / id", "seq": "acD-ef*", "score": 0.75, "binder_chains": "", "target_chains": ""},
            {"name": "same / id", "seq": "GGH", "score": 0.5, "binder_chains": "Z", "target_chains": "A"},
            {"name": "no-sequence", "seq": float("nan"), "score": 0.1, "binder_chains": "Z", "target_chains": "A"},
        ]
    ).to_csv(index=False).encode()
    kwargs = {
        "table_bytes": table,
        "table_filename": "upload.csv",
        "id_column": "name",
        "sequence_column": "seq",
        "binder_chains_column": "binder_chains",
        "target_chains_column": "target_chains",
        "default_binder_chains": binder_chains,
        "default_target_chains": target_chains,
        "source_tool": "fixture-import",
    }

    run_dir = importer.run_generic_table_import(**kwargs)

    candidates = read_candidates(run_dir)
    assert len(candidates) == 2
    assert candidates[0]["candidate_id"] == "same___id"
    assert candidates[1]["candidate_id"] == "same___id_002"
    assert [row["binder_sequence"] for row in candidates] == ["ACDEF", "GGH"]
    assert candidates[0]["binder_chains"] == (binder_chains or ["A"])
    assert candidates[0]["target_chains"] == (target_chains or [])
    assert candidates[1]["binder_chains"] == ["Z"]  # row-level values override the defaults
    assert candidates[1]["target_chains"] == ["A"]
    assert candidates[0]["metrics"]["import_score"] == 0.75
    assert candidates[0]["source_tool"] == "fixture-import"
    assert json.loads((run_dir / "artifacts" / "raw" / "generic_import" / "import_source.json").read_text())["candidate_count"] == 2
    summary = pd.read_csv(run_dir / "artifacts" / "normalized_candidates" / "import_summary.csv")
    assert summary["status"].tolist() == ["imported", "imported", "skipped"]


@pytest.mark.parametrize(
    ("filename", "payload", "error"),
    [
        ("data.json", b"{}", "must be CSV"),
        ("empty.csv", b"id,seq\n", "no rows"),
    ],
)
def test_generic_import_rejects_unsupported_or_empty_tables(filename: str, payload: bytes, error: str, run_store: Path) -> None:
    with pytest.raises(ValueError, match=error):
        importer.run_generic_table_import(table_bytes=payload, table_filename=filename, sequence_column="seq")


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({}, "Provide a CSV/Excel"),
        ({"table_bytes": b"id,seq\na,ACD\n", "table_filename": "x.csv"}, "Select a sequence column"),
        ({"table_bytes": b"id,seq\na,ACD\n", "table_filename": "x.csv", "sequence_column": "seq", "id_column": "missing"}, None),
    ],
)
def test_generic_import_validates_required_inputs(kwargs: dict, error: str | None, run_store: Path) -> None:
    if error:
        with pytest.raises(ValueError, match=error):
            importer.run_generic_table_import(**kwargs)
    else:
        # An absent optional ID column falls back to stable row-based IDs.
        run_dir = importer.run_generic_table_import(**kwargs)
        assert read_candidates(run_dir)[0]["candidate_id"] == "candidate_00001"


def test_generic_import_fails_when_every_candidate_lacks_a_sequence(run_store: Path) -> None:
    with pytest.raises(ValueError, match="No importable candidates"):
        importer.run_generic_table_import(
            table_bytes=b"id,seq\na,\nb,\n",
            table_filename="empty-sequences.csv",
            id_column="id",
            sequence_column="seq",
        )


def test_generic_import_uses_row_id_when_table_id_is_blank(run_store: Path) -> None:
    run_dir = importer.run_generic_table_import(
        table_bytes=b"id,seq\n,ACD\n",
        table_filename="blank-id.csv",
        id_column="id",
        sequence_column="seq",
    )
    assert read_candidates(run_dir)[0]["candidate_id"] == "candidate_00001"


def test_generic_import_stages_complex_and_target_and_infers_chain_roles(
    tmp_path: Path, run_store: Path
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    complex_path = _complex_pdb(source_dir / "complex.pdb")
    target_path = _complex_pdb(tmp_path / "target.pdb")
    run_dir = importer.run_generic_table_import(
        table_bytes=b"id,complex,binder_chains\nbinder-1,complex.pdb,Z\n",
        table_filename="with-structure.csv",
        base_dir=source_dir,
        id_column="id",
        complex_path_column="complex",
        binder_chains_column="binder_chains",
        target_structure=target_path,
        copy_files=True,
    )

    candidate = read_candidates(run_dir)[0]
    assert candidate["binder_sequence"] == "G"
    assert candidate["binder_chains"] == ["Z"]
    assert candidate["target_chains"] == ["A"]
    assert Path(candidate["complex_pdb"]).read_text() == complex_path.read_text()
    assert Path(candidate["target_pdb"]).read_text() == target_path.read_text()
    assert Path(candidate["complex_pdb"]).is_relative_to(run_dir)
    assert Path(candidate["target_pdb"]).is_relative_to(run_dir)


def test_bindcraft_import_preserves_ranked_structure_sequence_and_metrics(
    tmp_path: Path, run_store: Path
) -> None:
    source_dir = tmp_path / "bindcraft-output"
    ranked_dir = source_dir / "Accepted" / "Ranked"
    ranked_dir.mkdir(parents=True)
    structure = _complex_pdb(ranked_dir / "001_design-1.pdb")
    (source_dir / "final_design_stats.csv").write_text(
        "Design,Rank,Sequence,Average_i_pTM\ndesign-1,1,GGG,0.82\n"
    )

    run_dir = importer.run_bindcraft_import(
        input_dir=source_dir,
        binder_chains=["Z"],
        max_candidates=1,
    )

    candidate = read_candidates(run_dir)[0]
    assert candidate["candidate_id"] == "design-1"
    assert candidate["binder_sequence"] == "GGG"
    assert candidate["binder_chains"] == ["Z"]
    assert candidate["target_chains"] == ["A"]
    assert candidate["metrics"]["bindcraft_final_rank"] == 1
    assert candidate["metrics"]["bindcraft_Average_i_pTM"] == 0.82
    assert candidate["metrics"]["bindcraft_ranked_pdb_rank"] == 1
    assert candidate["metrics"]["bindcraft_rank_consistent"] is True
    assert Path(candidate["complex_pdb"]).is_file()
