from __future__ import annotations

import json

from mn_protein_design.core.candidates import (
    STAGE_GENERATION_BACKBONE,
    normalize_candidate,
    read_candidates,
)


def test_normalize_candidate_preserves_legacy_sequence_and_tool_fields() -> None:
    candidate = normalize_candidate(
        {
            "candidate_id": "candidate-1",
            "tool": "legacy-generator",
            "stage": "backbone",
            "sequence": "ACDE",
            "target_chains": ["A"],
            "binder_chains": ["Z"],
        }
    )

    assert candidate["stage"] == STAGE_GENERATION_BACKBONE
    assert candidate["source_tool"] == "legacy-generator"
    assert candidate["binder_sequence"] == "ACDE"
    assert candidate["raw_metadata"]["legacy_tool"] == "legacy-generator"
    assert candidate["raw_metadata"]["legacy_sequence_key"] == "sequence"


def test_read_candidates_normalizes_legacy_jsonl_and_skips_bad_rows(tmp_path) -> None:
    candidates_path = tmp_path / "candidates.jsonl"
    candidates_path.write_text(
        json.dumps(
            {
                "candidate_id": "legacy-1",
                "tool": "old-engine",
                "stage": "backbone",
                "sequence": "ACDE",
            }
        )
        + "\nnot-json\n"
    )

    candidates = read_candidates(candidates_path)

    assert len(candidates) == 1
    assert candidates[0]["candidate_id"] == "legacy-1"
    assert candidates[0]["binder_sequence"] == "ACDE"
    assert candidates[0]["stage"] == STAGE_GENERATION_BACKBONE
