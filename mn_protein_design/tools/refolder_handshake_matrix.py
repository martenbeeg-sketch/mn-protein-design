from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from mn_protein_design.core.candidates import read_candidates
from mn_protein_design.core.jobs import create_job, finish_job, write_json
from mn_protein_design.workflows.design_campaigns import (
    _screen_candidates_with_refolder,
    run_campaign_evaluation,
)


SOURCE_CAMPAIGN = Path(
    "/mnt/data/RESULTS/mn-protein-design-workdir/workdir/runs/"
    "design-campaign/20260715-194154-ed304983"
)
CHILD_LABEL = "child_0001_rfdiffusion_classic_20260715-194154-c488a83d"


def _label(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "")


def _load_boundary_candidates() -> dict[str, list[dict]]:
    return {
        "level1": list(
            read_candidates(
                SOURCE_CAMPAIGN
                / "artifacts/design_campaign/staged_two_step"
                / f"{CHILD_LABEL}_first_pass/source_candidates.jsonl"
            )
        )[:1],
        "level2": list(
            read_candidates(
                SOURCE_CAMPAIGN
                / "artifacts/design_campaign/staged_two_step"
                / f"{CHILD_LABEL}_second_pass/source_candidates.jsonl"
            )
        )[:1],
        "evaluation": list(read_candidates(SOURCE_CAMPAIGN / "artifacts/normalized_candidates/candidates.jsonl"))[:1],
    }


def _tests(group: str, candidates: dict[str, list[dict]]) -> list[tuple[str, str, dict, list[dict]]]:
    common = {"num_recycles": 1, "num_sampling_steps": 10, "num_samples": 1, "seed": 3, "use_target_msa": True}
    af2 = {
        **common,
        "use_target_msa": False,
        "af2_multimer": True,
        "af2_use_initial_guess": True,
        "af2_use_binder_template": False,
        "af2_use_interface_template": False,
    }
    esm = {**common, "use_target_msa": False}
    boltzgen = {**common, "use_target_msa": False}
    protenix_template = {
        **common,
        "protenix_v1_use_template": True,
        "protenix_v2_use_template": True,
    }
    groups = {
        "gpu0": [
            ("level1", "AF2-IG", af2, candidates["level1"]),
            ("level1", "Boltz-2", common, candidates["level1"]),
            ("level1", "ColabFold", common, candidates["level1"]),
            ("level2", "ESMFold2", esm, candidates["level2"]),
            ("evaluation", "Protenix", common, candidates["evaluation"]),
        ],
        "gpu1": [
            ("level1", "RF3", common, candidates["level1"]),
            ("level2", "RF3", common, candidates["level2"]),
            ("level2", "OpenFold-3", common, candidates["level2"]),
            ("evaluation", "AlphaFast AF3", common, candidates["evaluation"]),
            ("evaluation", "Protenix v1", protenix_template, candidates["evaluation"]),
            ("evaluation", "Protenix v2", protenix_template, candidates["evaluation"]),
            ("evaluation", "BoltzGen Fold", boltzgen, candidates["evaluation"]),
        ],
    }
    return groups[group]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--group", choices=["gpu0", "gpu1"], required=True)
    parser.add_argument("--only", action="append", default=[], help="Run rows whose boundary or refolder contains this text.")
    parser.add_argument("--boundary", action="append", default=[], help="Run only this boundary, e.g. level1.")
    args = parser.parse_args()

    candidates = _load_boundary_candidates()
    job = create_job(
        "design-campaign",
        f"refolder_handshake_matrix_{args.group}",
        "design_campaign_refolder_handshake",
        {"source_campaign": str(SOURCE_CAMPAIGN)},
        {
            "gpu_device": "none",
            "child_gpu_device": args.gpu,
            "group": args.group,
            "target": "same as RFdiffusion staged run",
        },
    )
    print(f"SUITE {job.run_dir}", flush=True)
    write_json(
        job.run_dir / "command.json",
        {
            "mode": "manual_codex_refolder_handshake_matrix",
            "source_campaign": str(SOURCE_CAMPAIGN),
            "group": args.group,
            "gpu_device": args.gpu,
            "note": "RFdiffusion-classic candidate; minimal level/evaluation refolder handshake tests",
        },
    )

    results: list[dict] = []
    matrix_path = job.run_dir / "artifacts/design_campaign/refolder_handshake_matrix.json"
    selected_tests = _tests(args.group, candidates)
    if args.only:
        wanted = [text.lower() for text in args.only]
        selected_tests = [
            row
            for row in selected_tests
            if any(text in row[0].lower() or text in row[1].lower() for text in wanted)
        ]
    if args.boundary:
        boundaries = {text.lower() for text in args.boundary}
        selected_tests = [row for row in selected_tests if row[0].lower() in boundaries]
    for index, (boundary, refolder, settings, boundary_candidates) in enumerate(selected_tests, start=1):
        print(f"START {index} {boundary} {refolder}", flush=True)
        row = {
            "boundary": boundary,
            "refolder": refolder,
            "status": "failed",
            "candidate_count_in": len(boundary_candidates),
            "settings": settings,
        }
        try:
            if boundary in {"level1", "level2"}:
                screened, summary = _screen_candidates_with_refolder(
                    job.run_dir,
                    boundary_candidates,
                    label=f"{boundary}_{_label(refolder)}",
                    gpu_device=args.gpu,
                    refolder=refolder,
                    settings=settings,
                )
                row.update(
                    {
                        "status": summary.get("status"),
                        "candidate_count_out": len(screened),
                        "summary": summary,
                        "child_run": summary.get("child_run", ""),
                        "child_run_id": summary.get("child_run_id", ""),
                    }
                )
            else:
                config = {
                    **settings,
                    "mode": "refold_and_metrics",
                    "refolder": refolder,
                    "ipsae": False,
                    "rosetta": False,
                    "pymol": False,
                    "artifact_label_prefix": _label(refolder),
                }
                summary = run_campaign_evaluation(job.run_dir, boundary_candidates, config=config, gpu_device=args.gpu)
                row.update(
                    {
                        "status": summary.get("status"),
                        "candidate_count_out": summary.get("candidate_count"),
                        "summary": summary,
                        "child_run": summary.get("child_run", ""),
                        "child_run_id": summary.get("child_run_id", ""),
                    }
                )
        except BaseException as exc:
            row.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-4000:],
                }
            )
        results.append(row)
        write_json(matrix_path, {"suite_run": str(job.run_dir), "results": results})
        print(
            f"DONE {boundary} {refolder} {row.get('status')} "
            f"{row.get('candidate_count_out')} {row.get('error', '')}",
            flush=True,
        )

    finish_job(
        job.run_dir,
        any(row.get("status") == "completed" for row in results),
        {
            "outputs": {"matrix": "artifacts/design_campaign/refolder_handshake_matrix.json"},
            "metrics": {
                "test_count": len(results),
                "completed_count": sum(row.get("status") == "completed" for row in results),
                "failed_count": sum(row.get("status") != "completed" for row in results),
            },
            "results": results,
        },
    )
    print(f"FINAL {job.run_dir}", flush=True)
    print(
        json.dumps(
            {
                "completed": sum(row.get("status") == "completed" for row in results),
                "failed": [
                    {
                        "boundary": row["boundary"],
                        "refolder": row["refolder"],
                        "status": row["status"],
                        "error": row.get("error", ""),
                        "child_run": row.get("child_run", ""),
                    }
                    for row in results
                    if row.get("status") != "completed"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
