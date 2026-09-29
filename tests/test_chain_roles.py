from __future__ import annotations

from mn_protein_design.workflows.chain_roles import (
    CHAIN_ROLE_SCHEMA_EXPLICIT_V2,
    assign_binder_engine_chains,
    assign_target_engine_chains,
    build_explicit_role_map,
    validate_chain_roles,
)


def test_target_and_binder_chains_avoid_role_overlap() -> None:
    target_chains = assign_target_engine_chains(["A", "B"])
    binder_chains = assign_binder_engine_chains(["A"], reserved=target_chains)

    assert target_chains == ["A", "B"]
    assert binder_chains == ["Z"]
    assert set(target_chains).isdisjoint(binder_chains)


def test_explicit_role_map_validates_against_staged_structure() -> None:
    role_map = build_explicit_role_map(
        binder_source_chains=["D"],
        binder_engine_chains=["Z"],
        target_source_chains=["A", "B"],
        target_engine_chains=["A", "B"],
    )

    assert role_map.schema == CHAIN_ROLE_SCHEMA_EXPLICIT_V2
    assert role_map.roles == {"A": "target", "B": "target", "Z": "binder"}
    findings = validate_chain_roles(
        candidate_id="candidate-1",
        binder_chains=role_map.binder_chains,
        target_chains=role_map.target_chains,
        structure_chains=["A", "B", "Z"],
        schema=role_map.schema,
    )
    assert findings == []


def test_chain_role_validation_reports_binder_target_overlap() -> None:
    findings = validate_chain_roles(
        candidate_id="candidate-2",
        binder_chains=["A"],
        target_chains=["A"],
        structure_chains=["A"],
        schema=CHAIN_ROLE_SCHEMA_EXPLICIT_V2,
    )

    assert any(
        row["code"] == "binder_target_chain_overlap" and row["severity"] == "warning"
        for row in findings
    )
