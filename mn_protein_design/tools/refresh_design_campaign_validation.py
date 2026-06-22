#!/usr/bin/env python
from __future__ import annotations

import argparse
import shutil
from datetime import datetime, timezone
from pathlib import Path

from mn_protein_design.core.candidates import write_candidates
from mn_protein_design.core.jobs import read_json, update_status, write_json
from mn_protein_design.workflows.design_campaigns import (
    _write_summary_csv,
    harmonize_validated_candidates,
    run_common_bindcraft_validation,
)


def _archive_current_results(run_dir: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    archive = run_dir / "artifacts" / "design_campaign" / "validation_history" / timestamp
    archive.mkdir(parents=True, exist_ok=False)
    paths = {
        "common_validation": run_dir / "artifacts" / "design_campaign" / "common_validation",
        "workflow_summary.json": run_dir / "artifacts" / "design_campaign" / "workflow_summary.json",
        "candidates.jsonl": run_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl",
        "campaign_result.json": run_dir / "artifacts" / "normalized_candidates" / "campaign_result.json",
        "result.json": run_dir / "result.json",
    }
    for name, source in paths.items():
        if not source.exists():
            continue
        destination = archive / name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
    return archive


def refresh_campaign(run_dir: Path, *, gpu_device: str) -> None:
    run_dir = run_dir.expanduser().resolve()
    payload = read_json(run_dir / "input.json")
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    result = read_json(run_dir / "result.json")
    outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
    child_runs = [
        row
        for row in outputs.get("child_runs") or []
        if isinstance(row, dict) and row.get("run_dir")
    ]
    if not child_runs:
        raise ValueError(f"No reusable child design runs found in {run_dir}")

    archive = _archive_current_results(run_dir)
    common_config = (
        params.get("common_validation")
        if isinstance(params.get("common_validation"), dict)
        else {}
    )
    update_status(
        run_dir,
        "running",
        current_phase="Common validation refresh",
        current_engine="AF2-multimer-v3",
        progress_label="Reusing existing designs; refreshing harmonized validation",
    )
    validated, common_summary = run_common_bindcraft_validation(
        run_dir,
        child_runs,
        gpu_device=gpu_device,
        num_recycles=int(common_config.get("num_recycles", 3)),
        min_ipsae=float(common_config.get("min_ipsae", 0.0)),
        pyrosetta_nprocs=int(common_config.get("pyrosetta_nprocs", 4)),
    )
    harmonized, refreshed_summaries = harmonize_validated_candidates(
        validated,
        survivors_per_engine=int(params.get("survivors_per_engine", 2)),
        keep_best_failed=bool(params.get("keep_best_failed", True)),
    )
    for candidate in harmonized:
        metrics = candidate.setdefault("metrics", {})
        metrics.update(
            {
                "campaign_evaluation_mode": "none",
                "campaign_evaluation_status": "skipped",
                "campaign_evaluation_refolder": "",
            }
        )

    normalized = write_candidates(run_dir, "design_campaign", harmonized)
    summary_path = run_dir / "artifacts" / "design_campaign" / "workflow_summary.json"
    summary = read_json(summary_path)
    existing_workflows = (
        summary.get("workflows")
        if isinstance(summary.get("workflows"), list)
        else []
    )
    refreshed_by_engine = {row["engine"]: row for row in refreshed_summaries}
    summary["workflows"] = [
        {**row, **refreshed_by_engine.get(str(row.get("engine")), {})}
        for row in existing_workflows
    ]
    summary["common_validation"] = {
        **common_summary,
        "refreshed_from_existing_designs": True,
        "previous_results_archive": str(archive),
    }
    summary["candidate_count"] = len(normalized)
    summary["harmonized_candidate_count"] = len(harmonized)
    write_json(summary_path, summary)
    csv_path = _write_summary_csv(run_dir, normalized)

    result_outputs = dict(outputs)
    result_outputs.update(
        {
            "candidates": normalized,
            "workflow_summary": str(summary_path.relative_to(run_dir)),
            "harmonized_candidates_csv": str(csv_path.relative_to(run_dir)),
        }
    )
    result_metrics = (
        dict(result.get("metrics"))
        if isinstance(result.get("metrics"), dict)
        else {}
    )
    result_metrics.update(
        {
            "candidate_count": len(normalized),
            "harmonized_candidate_count": len(harmonized),
            **common_summary,
            "validation_refreshed_from_existing_designs": True,
            "previous_results_archive": str(archive),
        }
    )
    result.update(
        {
            "success": bool(normalized),
            "outputs": result_outputs,
            "metrics": result_metrics,
        }
    )
    write_json(run_dir / "result.json", result)
    update_status(
        run_dir,
        "completed" if normalized else "failed",
        current_phase="",
        current_engine="",
        progress_label="",
        validation_refreshed_from_existing_designs=True,
        previous_results_archive=str(archive),
    )
    print(f"{run_dir.name}: {len(normalized)} candidates; archived previous results at {archive}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--gpu-device", default="0")
    options = parser.parse_args()
    for run_dir in options.run_dirs:
        refresh_campaign(run_dir, gpu_device=str(options.gpu_device))


if __name__ == "__main__":
    main()
