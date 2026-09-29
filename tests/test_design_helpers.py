from __future__ import annotations

from pathlib import Path

import pytest

from mn_protein_design.core.candidates import STAGE_GENERATION_BACKBONE
from mn_protein_design.workflows import design


@pytest.mark.parametrize(
    ("value", "expected"),
    [("", ""), (" A56-120/0 50-50 ", "A56-120,/0,50-50"),
     ("A1-10/0 20-30/B2-9/0 40-50", "A1-10,/0,20-30/B2-9,/0,40-50"),
     ("A1-10/0,20-30", "A1-10/0,20-30")],
)
def test_rfdiffusion3_contig_normalization_handles_native_spacing(value: str, expected: str) -> None:
    assert design._normalize_rfdiffusion3_contig(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("55", [55]), ("55-80", [55, 80]), ("55 - 80", [55, 80]), ("55,80", [55, 80]),
     ("55 80", [55, 80]), ("55-55", [55]), (" 64 ", [64])],
)
def test_binder_length_parser_accepts_single_values_and_ranges(value: str, expected: list[int]) -> None:
    assert design.parse_binder_lengths(value) == expected


@pytest.mark.parametrize("value", ["", "0", "-1", "10-9", "10-11-12", "55--80"])
def test_binder_length_parser_rejects_invalid_or_reversed_ranges(value: str) -> None:
    with pytest.raises(ValueError):
        design.parse_binder_lengths(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("A56", "A56"), ("a56; B129", "A56,B129"), ("A:56 B129", "A56,B129"),
     ("Z9,A2,A2", "Z9,A2,A2"), ("", "")],
)
def test_hotspot_normalization_cleans_separators_and_uppercases(value: str, expected: str) -> None:
    assert design.normalize_hotspots(value) == expected


@pytest.mark.parametrize("value", ["56", "A", "AA5", "A-1", "A5.5"])
def test_hotspot_normalization_rejects_non_residue_tokens(value: str) -> None:
    with pytest.raises(ValueError, match="must look like"):
        design.normalize_hotspots(value)


@pytest.mark.parametrize(
    ("contig", "hotspots", "binder_length", "designs", "steps", "message"),
    [
        ("", "", "55", 1, 50, "Contig is required"),
        ("A10-100", "", "55", 1, 50, "Binder contig"),
        ("A10-100/0 55-55", "", "55-70", 1, 50, None),
        ("A10-100/0 55-55", "B56", "55", 0, 50, "at least 1"),
        ("A10-100/0 55-55", "B56", "55", 1, 0, "Timesteps"),
        ("A10-100/0 55-55", "bad", "55", 1, 50, "must look like"),
    ],
)
def test_rfdiffusion_input_validation_covers_required_fields(contig, hotspots, binder_length, designs, steps, message) -> None:
    if message:
        with pytest.raises(ValueError, match=message):
            design.validate_rfdiffusion_inputs(contig, hotspots, binder_length, designs, steps)
    else:
        design.validate_rfdiffusion_inputs(contig, hotspots, binder_length, designs, steps)


@pytest.mark.parametrize(
    ("length", "inpaint", "message"),
    [("", "", None), ("120", "A10-20/B3", None), ("120-240", "A10", None),
     ("1-2-3", "", "contigmap.length"), ("", "A10-/B2", "inpaint_seq")],
)
def test_rfdiffusion_optional_parameter_validation(length, inpaint, message) -> None:
    if message:
        with pytest.raises(ValueError, match=message):
            design.validate_rfdiffusion_optional_params(length, inpaint)
    else:
        design.validate_rfdiffusion_optional_params(length, inpaint)


def test_rfdiffusion_run_parameters_emit_only_enabled_overrides() -> None:
    params = design.build_rfdiffusion_run_parameters(
        timesteps=40, partial_diffusion=True, contigmap_length="120", inpaint_seq="A10-20",
        model_weights="Complex_beta", deterministic=True, noise_scale_ca="0.5",
    )
    assert params == (
        "diffuser.partial_T=40 contigmap.length=120-120 contigmap.inpaint_seq=[A10-20] "
        "inference.ckpt_override_path=/models/Complex_beta_ckpt.pt inference.deterministic=True "
        "denoiser.noise_scale_ca=0.5"
    )


def test_scaffold_guided_parameters_respect_enabled_fields() -> None:
    assert design.build_rfdiffusion_scaffoldguided_parameters(False) == ""
    params = design.build_rfdiffusion_scaffoldguided_parameters(
        True, scaffold_dir="/models/scaffolds/", target_pdb=False, target_path="target.pdb", target_ss="target.ss"
    )
    assert "++scaffoldguided.scaffoldguided=True" in params
    assert "++scaffoldguided.scaffold_dir=/models/scaffolds/" in params
    assert "++scaffoldguided.target_path=target.pdb" in params
    assert "++scaffoldguided.target_ss=target.ss" in params
    assert "++scaffoldguided.target_pdb=True" not in params


def test_hydra_shell_overrides_escape_double_quotes() -> None:
    assert design._hydra_overrides_for_bash('a="value"') == 'a=\\"value\\"'


@pytest.mark.parametrize(
    ("native", "raw", "expected_ids", "expected_levels"),
    [
        ([], [], [], []),
        ([{"candidate_id": "n1", "complex_pdb": "same.pdb"}],
         [{"candidate_id": "r1", "complex_pdb": "same.pdb"}, {"candidate_id": "r2", "complex_pdb": "raw.pdb"}],
         ["n1", "tool_prefilter_00002"], ["native_pipeline", "prefilter_generated"]),
        ([{"candidate_id": "n1", "binder_pdb": "binder.pdb"}],
         [{"candidate_id": "r1", "binder_pdb": "binder.pdb"}], ["n1"], ["native_pipeline"]),
    ],
)
def test_candidate_pool_merge_deduplicates_structures_and_marks_provenance(native, raw, expected_ids, expected_levels) -> None:
    merged = design._merge_candidate_pools(
        tool="tool", native_candidates=native, raw_candidates=raw,
        native_coverage="ranked", raw_coverage="unfiltered",
    )
    assert [row["candidate_id"] for row in merged] == expected_ids
    assert [row["metrics"]["candidate_pool_level"] for row in merged] == expected_levels
    assert all(row["raw_metadata"]["candidate_pool_coverage"] for row in merged)


@pytest.mark.parametrize(
    ("row", "expected"),
    [({"self_complex_i_pAE": "0.2", "self_complex_pLDDT": "0.95", "self_binder_scRMSD_ca": "1.4"}, True),
     ({"self_complex_i_pAE": "0.3", "self_complex_pLDDT": "0.95", "self_binder_scRMSD_ca": "1.4"}, False),
     ({"self_complex_i_pAE": "0.2", "self_complex_pLDDT": "0.89", "self_binder_scRMSD_ca": "1.4"}, False),
     ({"self_complex_i_pAE": "0.2", "self_complex_pLDDT": "0.95", "self_binder_scRMSD_ca": "1.5"}, False),
     ({"self_complex_i_pAE": "bad"}, None)],
)
def test_proteina_complexa_native_filter_obeys_thresholds(row: dict, expected: bool | None) -> None:
    assert design._proteina_complexa_native_pass(row) is expected


def test_proteina_complexa_ranking_uses_lowest_interface_pae_and_stable_ties() -> None:
    rows = [{"id": "first", "self_complex_i_pAE": "2.0"},
            {"id": "second", "self_complex_i_pAE": "1.0"},
            {"id": "third", "self_complex_i_pAE": "1.0"},
            {"id": "missing", "self_complex_i_pAE": ""}]
    ranked = design._proteina_complexa_ranked_rows(rows)
    assert [(rank, row["id"], metric, value) for rank, row, metric, value in ranked] == [
        (1, "second", "self_complex_i_pAE", 1.0), (2, "third", "self_complex_i_pAE", 1.0),
        (3, "first", "self_complex_i_pAE", 2.0), (4, "missing", "self_complex_i_pAE", None)]


@pytest.mark.parametrize(
    ("rows", "expected_order", "metric"),
    [
        ([{"score": "0.2", "self_complex_i_pTM": "0.5"}, {"score": "0.1", "self_complex_i_pTM": "0.9"}],
         ["0.9", "0.5"], "self_complex_i_pTM"),
        ([{"score": "a"}, {"score": "b"}], ["a", "b"], None),
    ],
)
def test_proteina_complexa_ranking_falls_back_to_iptm_or_input_order(rows, expected_order, metric) -> None:
    ranked = design._proteina_complexa_ranked_rows(rows)
    assert [row.get("self_complex_i_pTM") if metric else row["score"] for _, row, _, _ in ranked] == expected_order
    assert all(rank_metric == metric for _, _, rank_metric, _ in ranked)


def test_generic_candidate_discovery_normalizes_structure_paths(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "run"
    artifacts = run_dir / "artifacts"
    (artifacts / "raw").mkdir(parents=True)
    structure = artifacts / "raw" / "candidate.pdb"
    structure.write_text("END\n")
    target = tmp_path / "target.pdb"
    target.write_text("END\n")
    monkeypatch.setattr(design, "write_candidates", lambda _run, _tool, rows: rows)

    rows = design._generic_candidate_records(
        run_dir, "engine", STAGE_GENERATION_BACKBONE,
        {"target_chains": ["A"], "hotspots": "A56,B129", "binder_length": "55", "contig": "A1-80/0 55-55"},
        target, ["raw/*.pdb"],
    )

    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "engine_00001"
    assert rows[0]["complex_pdb"] == "artifacts/raw/candidate.pdb"
    assert rows[0]["target_pdb"] == str(target)
    assert rows[0]["hotspots"] == ["A56", "B129"]
