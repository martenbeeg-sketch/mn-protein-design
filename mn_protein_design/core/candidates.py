from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mn_protein_design.core.jobs import write_json


CANDIDATE_SCHEMA_VERSION = "mn-protein-design.candidate.v1"


STAGE_GENERATION_BACKBONE = "generation.backbone"
STAGE_GENERATION_BACKBONE_SEQUENCE = "generation.backbone_sequence"
STAGE_SEQUENCE_DESIGN = "sequence_design"
STAGE_MONOMER_REFOLDING = "monomer_refolding"
STAGE_COMPLEX_REFOLDING = "complex_refolding"
STAGE_ANALYSIS = "analysis"
STAGE_BENCHMARK = "benchmark"

LEGACY_STAGE_MAP = {
    "backbone": STAGE_GENERATION_BACKBONE,
    "trajectory": STAGE_GENERATION_BACKBONE,
    "backbone+sequence": STAGE_GENERATION_BACKBONE_SEQUENCE,
    "backbone+validation": STAGE_COMPLEX_REFOLDING,
    "backbone+sequence+refolding": STAGE_COMPLEX_REFOLDING,
    "design_only": STAGE_GENERATION_BACKBONE,
}


@dataclass
class Candidate:
    candidate_id: str
    stage: str
    source_tool: str
    target_pdb: str | None = None
    complex_pdb: str | None = None
    binder_pdb: str | None = None
    binder_sequence: str | None = None
    target_chains: list[str] = field(default_factory=list)
    binder_chains: list[str] = field(default_factory=list)
    hotspots: list[str] = field(default_factory=list)
    binder_length: str | None = None
    contig: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    parents: list[str] = field(default_factory=list)
    raw_metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: str = CANDIDATE_SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def normalize_stage(stage: object) -> str:
    text = str(stage or "").strip()
    return LEGACY_STAGE_MAP.get(text, text or STAGE_GENERATION_BACKBONE)


def normalize_candidate(payload: dict[str, Any]) -> dict[str, Any]:
    source_tool = str(payload.get("source_tool") or payload.get("tool") or "")
    binder_sequence = payload.get("binder_sequence")
    if binder_sequence is None:
        binder_sequence = payload.get("sequence")
    normalized = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "candidate_id": str(payload.get("candidate_id") or ""),
        "stage": normalize_stage(payload.get("stage")),
        "source_tool": source_tool,
        "tool": source_tool,
        "target_pdb": payload.get("target_pdb"),
        "complex_pdb": payload.get("complex_pdb"),
        "binder_pdb": payload.get("binder_pdb"),
        "binder_sequence": binder_sequence,
        "target_chains": list(payload.get("target_chains") or []),
        "binder_chains": list(payload.get("binder_chains") or []),
        "hotspots": list(payload.get("hotspots") or []),
        "binder_length": payload.get("binder_length"),
        "contig": payload.get("contig"),
        "metrics": dict(payload.get("metrics") or {}),
        "parents": list(payload.get("parents") or []),
        "raw_metadata": dict(payload.get("raw_metadata") or {}),
    }
    if payload.get("tool") and not normalized["raw_metadata"].get("legacy_tool"):
        normalized["raw_metadata"]["legacy_tool"] = payload.get("tool")
    if payload.get("sequence") and normalized["binder_sequence"] == payload.get("sequence"):
        normalized["raw_metadata"].setdefault("legacy_sequence_key", "sequence")
    return normalized


def candidates_dir(run_dir: Path) -> Path:
    return run_dir / "artifacts" / "normalized_candidates"


def candidates_jsonl_path(run_dir: Path) -> Path:
    return candidates_dir(run_dir) / "candidates.jsonl"


def campaign_result_path(run_dir: Path) -> Path:
    return candidates_dir(run_dir) / "campaign_result.json"


def write_candidates(run_dir: Path, source_tool: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [normalize_candidate(candidate) for candidate in candidates]
    out_dir = candidates_dir(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_jsonl_path(run_dir).write_text(
        "".join(json.dumps(candidate, sort_keys=True) + "\n" for candidate in normalized)
    )
    write_json(campaign_result_path(run_dir), {"schema_version": CANDIDATE_SCHEMA_VERSION, "source_tool": source_tool, "candidates": normalized})
    return normalized


def read_candidates(path: Path) -> list[dict[str, Any]]:
    if path.is_dir():
        path = candidates_jsonl_path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(normalize_candidate(json.loads(line)))
        except json.JSONDecodeError:
            continue
    return rows


def candidate_stage_counts(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        stage = normalize_stage(candidate.get("stage"))
        counts[stage] = counts.get(stage, 0) + 1
    return counts
