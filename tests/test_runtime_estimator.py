from __future__ import annotations

import json
from pathlib import Path

import pytest

from mn_protein_design.core.runtime_estimator import (
    FALLBACK_SECONDS_PER_CANDIDATE,
    RuntimeObservation,
    collect_runtime_observations,
    estimate_engines,
    estimate_job_runtime_with_observations,
    _refolding_engine_keys,
    format_duration,
)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(None, "n/a"), (-8, "0s"), (0, "0s"), (59.4, "59s"), (59.6, "1m"),
     (60, "1m"), (3599, "59m"), (3600, "1h 0m"), (90061, "1d 1h 1m")],
)
def test_format_duration_handles_boundaries(seconds, expected: str) -> None:
    assert format_duration(seconds) == expected


def test_runtime_observation_rates_guard_invalid_denominators() -> None:
    observation = RuntimeObservation("rf3", seconds=120.0, candidate_count=0, total_residues=0)
    assert observation.seconds_per_candidate is None
    assert observation.seconds_per_residue is None


@pytest.mark.parametrize(
    ("engine", "count", "residues", "expected_seconds"),
    [("rf3", 3, None, 90.0), ("unknown_engine", 2, None, 60.0), ("rf3", -5, None, 0.0),
     ("rf3", 2, 600, 60.0)],
)
def test_estimate_engines_uses_fallback_and_normalizes_counts(engine, count, residues, expected_seconds: float) -> None:
    estimate = estimate_engines(engines=[engine], candidate_count=count, total_residues=residues, observations=[])
    assert estimate["rows"][0]["estimated_seconds"] == expected_seconds
    assert estimate["rows"][0]["basis"] == "fallback"
    assert estimate["total_seconds"] == expected_seconds


@pytest.mark.parametrize(
    ("observations", "residues", "expected_basis", "expected"),
    [([RuntimeObservation("rf3", 100, 2)], None, "history/candidate", 150.0),
     ([RuntimeObservation("rf3", 100, 2, total_residues=200)], 300, "history/residue", 150.0),
     ([RuntimeObservation("rf3", 100, 2), RuntimeObservation("rf3", 300, 2)], None, "history/candidate", 300.0)],
)
def test_estimate_engines_uses_median_history(observations, residues, expected_basis: str, expected: float) -> None:
    estimate = estimate_engines(engines=["rf3"], candidate_count=3, total_residues=residues, observations=observations)
    assert estimate["rows"][0]["basis"] == expected_basis
    assert estimate["rows"][0]["estimated_seconds"] == expected


@pytest.mark.parametrize(
    ("params", "expected_multiplier"),
    [({}, 1.0), ({"num_loops": 10, "num_sampling_steps": 68}, 1.0),
     ({"num_loops": 20, "num_sampling_steps": 68}, 1.65),
     ({"num_loops": 10, "num_sampling_steps": 136}, 1.35), ({"num_loops": 0, "num_sampling_steps": 0}, 1.0)],
)
def test_esmfold2_estimate_scales_with_loop_and_sampling_settings(params, expected_multiplier: float) -> None:
    baseline = estimate_engines(engines=["esmfold2"], candidate_count=1, observations=[])["total_seconds"]
    scaled = estimate_engines(engines=["esmfold2"], candidate_count=1, engine_params={"esmfold2": params}, observations=[])
    assert scaled["total_seconds"] == pytest.approx(baseline * expected_multiplier)


def test_estimate_engines_reports_per_engine_and_aggregate_fields() -> None:
    estimate = estimate_engines(engines=["rf3", "esmfold2"], candidate_count=4, observations=[])
    assert estimate["candidate_count"] == 4
    assert len(estimate["rows"]) == 2
    assert estimate["total_seconds"] == sum(row["estimated_seconds"] for row in estimate["rows"])
    assert all(row["estimated_time"] for row in estimate["rows"])
    assert estimate["rows"][0]["seconds_per_candidate"] == FALLBACK_SECONDS_PER_CANDIDATE["rf3"]


def _write_observation(root: Path, run_id: str, metadata: dict, result: dict) -> None:
    run_dir = root / "design" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(json.dumps(metadata))
    (run_dir / "result.json").write_text(json.dumps(result))


def test_collect_runtime_observations_reads_completed_tool_and_engine_timings(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_observation(root, "timed", {
        "run_id": "timed", "status": "completed", "tool": "rf3",
        "created_at": "2026-01-01T00:00:00+00:00", "completed_at": "2026-01-01T00:02:00+00:00",
    }, {"metrics": {"candidate_count": 2, "total_residues": 80,
                     "runtime_engine_timings": {"rf3": {"seconds": 90, "candidate_count": 3, "total_residues": 60}}}})
    _write_observation(root, "active", {"status": "running", "tool": "rf3"}, {"metrics": {"candidate_count": 5}})
    observations = collect_runtime_observations(root)
    assert [(row.engine, row.seconds, row.candidate_count, row.source) for row in observations] == [
        ("rf3", 90.0, 3, "runtime_engine_timings"), ("rf3", 120.0, 2, "job_metadata")]
    assert observations[1].total_residues == 80


@pytest.mark.parametrize("result", [{}, {"metrics": {}}, {"outputs": {"candidates": []}}])
def test_collect_runtime_observations_ignores_missing_candidate_counts(tmp_path: Path, result: dict) -> None:
    _write_observation(tmp_path / "runs", "no-count", {
        "status": "completed", "tool": "rf3", "created_at": "2026-01-01T00:00:00",
        "completed_at": "2026-01-01T00:01:00",
    }, result)
    assert collect_runtime_observations(tmp_path / "runs") == []


@pytest.mark.parametrize(
    ("params", "expected_engines"),
    [({"run_rf3": True, "rf3_use_target_msa": True}, ["alphafast_msa", "rf3"]),
     ({"run_rf3": True, "alphafast_query_only_msa": True}, ["rf3"]),
     ({"run_rf3": True, "rf3_use_target_msa": True, "shared_msa_source": "alphafast_mmseqs_gpu"}, ["alphafast_msa", "rf3"]),
     ({"run_colabfold": True, "run_predicted_rosetta_metrics": True}, ["alphafast_msa", "colabfold", "postprocessing"]),
     ({"run_esmfold2": True, "esmfold2_use_target_msa": False}, ["esmfold2"]),
     ({"run_protenix_v2": True, "protenix_v2_use_msa": False}, ["protenix_v2"])],
)
def test_refolding_runtime_engine_selection_tracks_enabled_steps(params: dict, expected_engines: list[str]) -> None:
    estimate = estimate_job_runtime_with_observations(
        {"tool": "refolding_evaluation_engines", "job_type": "refolding_evaluation"},
        {"params": {**params, "candidate_count": 1}}, observations=[])
    assert estimate["estimated_seconds"] > 0
    assert estimate["basis"] == "refolding engine fallbacks/history"
    assert _refolding_engine_keys(params) == expected_engines


@pytest.mark.parametrize(
    ("job", "input_payload", "expected"),
    [({"tool": "rf3"}, {"params": {"num_designs": 3}}, 90.0),
     ({"tool": "rf3"}, {"params": {"candidate_count": 4, "num_designs": 99}}, 120.0),
     ({"tool": "protein_mpnn"}, {"params": {"num_seq_per_target": 5}}, 50.0)],
)
def test_job_runtime_estimates_select_candidate_count(job: dict, input_payload: dict, expected: float) -> None:
    estimate = estimate_job_runtime_with_observations(job, input_payload, observations=[])
    assert estimate["estimated_seconds"] == expected


def test_design_campaign_runtime_adds_common_validation_and_metrics() -> None:
    job = {"tool": "design_campaign", "job_type": "multi_engine_design_campaign"}
    base_params = {"engines": ["rf3"], "design_attempts": 2, "sequences_per_backbone": 3,
                   "common_validation": {"enabled": False}, "evaluation": {"mode": "none"}}
    base = estimate_job_runtime_with_observations(job, {"params": base_params}, observations=[])
    full_params = {**base_params, "common_validation": {"enabled": True}, "evaluation": {"mode": "full"}}
    full = estimate_job_runtime_with_observations(job, {"params": full_params}, observations=[])
    assert full["estimated_seconds"] > base["estimated_seconds"]
