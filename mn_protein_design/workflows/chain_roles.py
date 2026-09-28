from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


CHAIN_ROLE_SCHEMA_EXPLICIT_V2 = "explicit_v2"
CHAIN_ROLE_SCHEMA_LEGACY_INFERRED = "legacy_inferred"

TARGET_CHAIN_ORDER = [chr(code) for code in range(ord("A"), ord("Z") + 1)]
BINDER_CHAIN_ORDER = [chr(code) for code in range(ord("Z"), ord("A") - 1, -1)]


@dataclass(frozen=True)
class ChainRoleMap:
    schema: str
    target_chains: tuple[str, ...]
    binder_chains: tuple[str, ...]
    roles: dict[str, str]
    chain_map: dict[str, list[dict[str, Any]]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "target_chains": list(self.target_chains),
            "binder_chains": list(self.binder_chains),
            "roles": dict(self.roles),
            "chain_map": self.chain_map,
        }


def _clean_chains(chains: Iterable[object]) -> list[str]:
    values: list[str] = []
    for chain in chains:
        text = str(chain or "").strip()
        if not text or text in values:
            continue
        values.append(text[:1])
    return values


def assign_target_engine_chains(source_chains: Iterable[object], *, reserved: Iterable[object] = ()) -> list[str]:
    reserved_set = set(_clean_chains(reserved))
    assigned: list[str] = []
    for source_chain in _clean_chains(source_chains):
        candidate = source_chain if source_chain in TARGET_CHAIN_ORDER and source_chain not in reserved_set else ""
        if not candidate or candidate in assigned:
            candidate = next(
                chain for chain in TARGET_CHAIN_ORDER if chain not in reserved_set and chain not in assigned
            )
        assigned.append(candidate)
    return assigned


def assign_binder_engine_chains(source_chains: Iterable[object], *, reserved: Iterable[object] = ()) -> list[str]:
    reserved_set = set(_clean_chains(reserved))
    assigned: list[str] = []
    for source_chain in _clean_chains(source_chains):
        candidate = source_chain if source_chain in BINDER_CHAIN_ORDER and source_chain not in reserved_set else ""
        if not candidate or candidate in assigned:
            candidate = next(
                chain for chain in BINDER_CHAIN_ORDER if chain not in reserved_set and chain not in assigned
            )
        assigned.append(candidate)
    return assigned


def build_explicit_role_map(
    *,
    binder_source_chains: Iterable[object],
    binder_engine_chains: Iterable[object],
    target_source_chains: Iterable[object],
    target_engine_chains: Iterable[object],
    target_fragments: list[dict[str, Any]] | None = None,
) -> ChainRoleMap:
    binder_rows = [
        {"original_chain": original, "engine_chain": engine, "role": "binder"}
        for original, engine in zip(_clean_chains(binder_source_chains), _clean_chains(binder_engine_chains))
    ]
    target_rows = [
        {"original_chain": original, "engine_chain": engine, "role": "target"}
        for original, engine in zip(_clean_chains(target_source_chains), _clean_chains(target_engine_chains))
    ]
    roles = {row["engine_chain"]: "target" for row in target_rows}
    roles.update({row["engine_chain"]: "binder" for row in binder_rows})
    chain_map: dict[str, list[dict[str, Any]]] = {
        "binder": binder_rows,
        "targets": target_rows,
    }
    if target_fragments:
        chain_map["target_fragments"] = target_fragments
    return ChainRoleMap(
        schema=CHAIN_ROLE_SCHEMA_EXPLICIT_V2,
        target_chains=tuple(row["engine_chain"] for row in target_rows),
        binder_chains=tuple(row["engine_chain"] for row in binder_rows),
        roles=roles,
        chain_map=chain_map,
    )


def normalize_legacy_role_payload(payload: dict[str, Any]) -> ChainRoleMap:
    binder_rows = [row for row in payload.get("binder") or [] if isinstance(row, dict)]
    target_rows = [row for row in payload.get("targets") or [] if isinstance(row, dict)]
    binder_chains = _clean_chains(row.get("engine_chain") for row in binder_rows)
    target_chains = _clean_chains(row.get("engine_chain") for row in target_rows)
    roles = {chain: "target" for chain in target_chains}
    roles.update({chain: "binder" for chain in binder_chains})
    chain_map = {
        "binder": [{**row, "role": "binder"} for row in binder_rows],
        "targets": [{**row, "role": "target"} for row in target_rows],
    }
    fragments = payload.get("target_fragments")
    if isinstance(fragments, list):
        chain_map["target_fragments"] = fragments
    return ChainRoleMap(
        schema=CHAIN_ROLE_SCHEMA_LEGACY_INFERRED,
        target_chains=tuple(target_chains),
        binder_chains=tuple(binder_chains),
        roles=roles,
        chain_map=chain_map,
    )


def validate_chain_roles(
    *,
    candidate_id: object = "",
    binder_chains: Iterable[object] = (),
    target_chains: Iterable[object] = (),
    structure_chains: Iterable[object] = (),
    schema: object = "",
    target_only: bool = False,
) -> list[dict[str, Any]]:
    """Return non-blocking findings for role metadata consumed by scoring/refolding.

    Legacy rows are allowed and reported as informational compatibility notes.
    Only warning/error findings should be treated as user-facing problems.
    """
    candidate_text = str(candidate_id or "").strip()
    binder = _clean_chains(binder_chains)
    target = _clean_chains(target_chains)
    structure = _clean_chains(structure_chains)
    structure_set = set(structure)
    schema_text = str(schema or "").strip()
    findings: list[dict[str, Any]] = []

    def add(code: str, message: str, severity: str = "warning", **extra: Any) -> None:
        findings.append(
            {
                "candidate_id": candidate_text,
                "code": code,
                "severity": severity,
                "message": message,
                **extra,
            }
        )

    if not schema_text:
        add(
            "legacy_role_compatible",
            "Legacy chain-role layout detected; using declared binder_chains/target_chains.",
            "info",
            compatibility_schema=CHAIN_ROLE_SCHEMA_LEGACY_INFERRED,
        )
    elif schema_text != CHAIN_ROLE_SCHEMA_EXPLICIT_V2:
        add(
            "unknown_chain_role_schema",
            f"Unknown chain_role_schema {schema_text!r}; falling back to declared binder/target chains.",
        )

    if target_only:
        effective_target = target or binder
        if binder and not target:
            add(
                "target_only_legacy_binder_chains",
                "Target-only input stores target fragments in binder_chains for legacy compatibility.",
                "info",
            )
        elif binder and target:
            add(
                "target_only_legacy_binder_chains",
                "Target-only input duplicates target fragments in binder_chains for legacy compatibility.",
                "info",
            )
        if not effective_target:
            add("missing_target_chains", "Target-only input has no declared target/source chains.")
    else:
        if not binder:
            add("missing_binder_chains", "Complex input has no declared binder/design chains.")
        if not target:
            add("missing_target_chains", "Complex input has no declared target/reference chains.")

    overlap = [] if target_only else sorted(set(binder).intersection(target))
    if overlap:
        add(
            "binder_target_chain_overlap",
            "The same chain is declared as both binder and target.",
            chains=overlap,
        )

    if structure:
        expected = (target or binder) if target_only else [*binder, *target]
        missing = [chain for chain in expected if chain not in structure_set]
        if missing:
            add(
                "declared_chains_absent_from_structure",
                "Some declared role chains are not present in the staged structure.",
                chains=missing,
                structure_chains=structure,
            )

    return findings


def problem_findings(findings: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        finding
        for finding in findings
        if str(finding.get("severity") or "warning").lower() in {"warning", "error"}
    ]
