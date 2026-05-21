from __future__ import annotations

from pathlib import Path

from mn_protein_design.core.candidates import candidate_stage_counts, read_candidates
from mn_protein_design.core.jobs import collect_jobs, read_json
from mn_protein_design.core.modules import list_module_specs


def module_specs() -> list[dict]:
    return list_module_specs()


def candidate_sources(task_group: str | None = None) -> list[dict]:
    groups = [task_group] if task_group else ["design", "refolding-validation", "analysis"]
    rows: list[dict] = []
    for group in groups:
        for job in collect_jobs(group):
            run_dir = Path(job["run_dir"])
            candidates = read_candidates(run_dir)
            if not candidates:
                continue
            result = read_json(run_dir / "result.json")
            rows.append(
                {
                    **job,
                    "candidate_count": len(candidates),
                    "stage_counts": candidate_stage_counts(candidates),
                    "candidates_jsonl": str(run_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl"),
                    "campaign_result": str(run_dir / "artifacts" / "normalized_candidates" / "campaign_result.json"),
                    "downstream_artifacts": result.get("downstream_artifacts", {}),
                }
            )
    return rows


def load_source_candidates(source: dict) -> list[dict]:
    path = Path(str(source.get("candidates_jsonl") or ""))
    return read_candidates(path)
