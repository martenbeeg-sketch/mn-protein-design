from __future__ import annotations

import pytest

from mn_protein_design.workflows.chain_roles import (
    CHAIN_ROLE_SCHEMA_EXPLICIT_V2,
    CHAIN_ROLE_SCHEMA_LEGACY_INFERRED,
    assign_binder_engine_chains,
    assign_target_engine_chains,
    build_explicit_role_map,
    normalize_legacy_role_payload,
    problem_findings,
    validate_chain_roles,
)


@pytest.mark.parametrize(
    ("source", "reserved", "expected"),
    [([], [], []), (["A", "B"], [], ["A", "B"]), (["Z", "A", "B"], [], ["Z", "A", "B"]),
     (["A", "A"], [], ["A"]), (["A", "C"], ["A"], ["B", "C"]),
     (["long-chain", "B"], [], ["A", "B"])],
)
def test_target_engine_chain_assignment_is_valid_unique_and_reserved_aware(source, reserved, expected) -> None:
    assert assign_target_engine_chains(source, reserved=reserved) == expected


@pytest.mark.parametrize(
    ("source", "reserved", "expected"),
    [([], [], []), (["Z", "Y"], [], ["Z", "Y"]), (["A", "Z", "Y"], [], ["A", "Z", "Y"]),
     (["Z", "Z"], [], ["Z"]), (["Z", "A"], ["Z"], ["Y", "A"])],
)
def test_binder_engine_chain_assignment_counts_down_from_z(source, reserved, expected) -> None:
    assert assign_binder_engine_chains(source, reserved=reserved) == expected


def test_explicit_role_map_keeps_target_and_binder_mapping_rows() -> None:
    role_map = build_explicit_role_map(
        binder_source_chains=["B"], binder_engine_chains=["Z"],
        target_source_chains=["A", "C"], target_engine_chains=["A", "B"],
        target_fragments=[{"source_chain": "A", "engine_chain": "A"}],
    )
    assert role_map.schema == CHAIN_ROLE_SCHEMA_EXPLICIT_V2
    assert role_map.target_chains == ("A", "B")
    assert role_map.binder_chains == ("Z",)
    assert role_map.roles == {"A": "target", "B": "target", "Z": "binder"}
    assert role_map.chain_map["target_fragments"][0]["engine_chain"] == "A"


def test_role_map_deduplicates_ids_and_drops_unpaired_mapping_rows() -> None:
    role_map = build_explicit_role_map(
        binder_source_chains=["Z", "Z", "Y"], binder_engine_chains=["Z"],
        target_source_chains=["A"], target_engine_chains=["A", "B"],
    )
    assert role_map.binder_chains == ("Z",)
    assert role_map.target_chains == ("A",)
    assert len(role_map.chain_map["binder"]) == 1


def test_legacy_role_payload_is_normalized_with_compatibility_schema() -> None:
    role_map = normalize_legacy_role_payload({
        "binder": [{"original_chain": "A", "engine_chain": "A"}],
        "targets": [{"original_chain": "B", "engine_chain": "A"}],
        "target_fragments": [{"engine_chain": "A"}],
    })
    assert role_map.schema == CHAIN_ROLE_SCHEMA_LEGACY_INFERRED
    assert role_map.binder_chains == ("A",)
    assert role_map.target_chains == ("A",)
    assert role_map.roles["A"] == "binder"  # binder role wins a malformed collision
    assert role_map.chain_map["targets"][0]["role"] == "target"


@pytest.mark.parametrize(
    ("binder", "target", "structure", "expected_code"),
    [
        (["Z"], ["A"], ["A", "Z"], None),
        (["A"], ["A"], ["A"], "binder_target_chain_overlap"),
        (["Z"], ["A"], ["A"], "declared_chains_absent_from_structure"),
        ([], ["A"], ["A"], "missing_binder_chains"),
        (["Z"], [], ["Z"], "missing_target_chains"),
    ],
)
def test_validate_chain_roles_reports_invalid_role_metadata(binder, target, structure, expected_code) -> None:
    findings = validate_chain_roles(
        candidate_id="fixture", binder_chains=binder, target_chains=target,
        structure_chains=structure, schema=CHAIN_ROLE_SCHEMA_EXPLICIT_V2,
    )
    codes = {row["code"] for row in findings}
    if expected_code:
        assert expected_code in codes
    else:
        assert not any(row["severity"] in {"warning", "error"} for row in findings)


@pytest.mark.parametrize("severity", ["warning", "error", "WARNING", "info", ""])
def test_problem_findings_returns_only_actionable_severity(severity: str) -> None:
    findings = [{"severity": severity, "code": severity or "empty"}]
    assert problem_findings(findings) == (findings if severity.lower() in {"warning", "error", ""} else [])
