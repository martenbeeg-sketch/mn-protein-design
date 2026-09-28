from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

from mn_protein_design.core.jobs import JobPaths, finish_job, read_json, update_status, utc_now, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]


def _worker_log(run_dir: Path, name: str, text: str) -> None:
    path = run_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def _restore_worker_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    kwargs = dict(payload.get("kwargs") or {})
    path_kwargs = set(payload.get("path_kwargs") or [])
    for key in path_kwargs:
        value = kwargs.get(key)
        if value not in {None, ""}:
            kwargs[key] = Path(str(value)).expanduser()
    kwargs.pop("progress_callback", None)
    kwargs.pop("existing_job", None)
    return kwargs


def run_worker_job(run_dir: Path) -> int:
    run_dir = run_dir.expanduser().resolve()
    request = read_json(run_dir / "worker_request.json")
    kind = str(request.get("kind") or "").strip()
    if not kind:
        raise RuntimeError(f"No worker_request.json found for {run_dir}")

    try:
        if kind == "design_campaign":
            from mn_protein_design.workflows.design_campaigns import run_design_campaign

            run_design_campaign(run_dir)
            return 0
        if kind == "target_preparation":
            from mn_protein_design.workflows.target_prep import run_queued_target_preparation

            kwargs = _restore_worker_kwargs(request)
            run_queued_target_preparation(run_dir, **kwargs)
            return 0
        if kind == "sequence_design_pipeline":
            from mn_protein_design.core.candidates import read_candidates
            from mn_protein_design.workflows.benchmark import enqueue_candidate_refolding_evaluation
            from mn_protein_design.workflows.sequence_design import (
                run_foundry_mpnn_sequence_design,
                run_ligandmpnn_sequence_design,
            )

            kwargs = dict(request.get("kwargs") or {})
            backend = str(kwargs.pop("backend") or "")
            source_run_dir = Path(str(kwargs.pop("source_run_dir"))).expanduser()
            candidates_jsonl = Path(str(kwargs.pop("candidates_jsonl"))).expanduser()
            design_kwargs = dict(kwargs.pop("design_kwargs") or {})
            validation_kwargs = dict(kwargs.pop("validation_kwargs") or {})
            resume = bool(kwargs.pop("resume", False))
            job = JobPaths(task_group="design", run_id=run_dir.name, run_dir=run_dir)
            designed_candidates = run_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl"
            if not (resume and designed_candidates.exists()):
                if backend == "shared_ligandmpnn":
                    run_ligandmpnn_sequence_design(
                        source_run_dir=source_run_dir,
                        candidates_jsonl=candidates_jsonl,
                        existing_job=job,
                        **design_kwargs,
                    )
                elif backend == "foundry_mpnn":
                    run_foundry_mpnn_sequence_design(
                        source_run_dir=source_run_dir,
                        candidates_jsonl=candidates_jsonl,
                        existing_job=job,
                        **design_kwargs,
                    )
                else:
                    raise ValueError(f"Unsupported sequence-design backend: {backend}")

            if validation_kwargs:
                if not designed_candidates.exists():
                    raise RuntimeError("Sequence design completed without normalized candidates for validation.")
                metadata = read_json(run_dir / "metadata.json")
                existing_validation_run = Path(str(metadata.get("validation_run_dir") or "")).expanduser()
                if resume and existing_validation_run.exists() and (existing_validation_run / "worker_request.json").exists():
                    validation_metadata = read_json(existing_validation_run / "metadata.json")
                    validation_result = read_json(existing_validation_run / "result.json")
                    if str(validation_metadata.get("status") or "") != "completed" or validation_result.get("success") is not True:
                        from mn_protein_design.core.jobs import resume_job

                        resume_job(existing_validation_run, gpu_device=validation_kwargs.get("gpu_device"))
                else:
                    validation_kwargs["resume"] = bool(validation_kwargs.get("resume") or resume)
                    validation_run = enqueue_candidate_refolding_evaluation(
                        source_run_dir=run_dir,
                        candidates_jsonl=designed_candidates,
                        selected_candidate_ids=[str(row.get("candidate_id") or "") for row in read_candidates(run_dir)],
                        evaluation_name="Sequence design validation",
                        job_type="sequence_design_validation",
                        tool_name="sequence_design_validation",
                        **validation_kwargs,
                    )
                    metadata["validation_run_id"] = validation_run.name
                    metadata["validation_run_dir"] = str(validation_run)
                    write_json(run_dir / "metadata.json", metadata)
                    spawn_worker_for_run(validation_run)
            return 0
        if kind in {"de_novo_binder_scoring_dataset", "candidate_refolding_evaluation"}:
            from mn_protein_design.workflows.benchmark import run_de_novo_binder_scoring_dataset

            kwargs = _restore_worker_kwargs(request)
            metadata = read_json(run_dir / "metadata.json")
            job = JobPaths(
                task_group=str(metadata.get("task_group") or "benchmark"),
                run_id=run_dir.name,
                run_dir=run_dir,
            )
            run_de_novo_binder_scoring_dataset(existing_job=job, **kwargs)
            return 0
        if kind == "monomer_refolding":
            from mn_protein_design.workflows.refolding import run_monomer_refolding_contract

            kwargs = _restore_worker_kwargs(request)
            job = JobPaths(task_group="refolding-validation", run_id=run_dir.name, run_dir=run_dir)
            run_monomer_refolding_contract(existing_job=job, **kwargs)
            return 0
        if kind == "benchmark_pyrosetta_backfill":
            from mn_protein_design.workflows.benchmark import run_missing_pyrosetta_benchmark_metrics

            kwargs = _restore_worker_kwargs(request)
            run_missing_pyrosetta_benchmark_metrics(run_dir, **kwargs)
            return 0
        if kind == "capacity_benchmark_scheduler":
            from mn_protein_design.workflows.capacity_benchmark import run_capacity_benchmark_scheduler

            kwargs = _restore_worker_kwargs(request)
            run_capacity_benchmark_scheduler(run_dir, **kwargs)
            return 0
        if kind == "design_capacity_scheduler":
            from mn_protein_design.workflows.capacity_benchmark import run_design_capacity_scheduler

            kwargs = _restore_worker_kwargs(request)
            run_design_capacity_scheduler(run_dir, **kwargs)
            return 0
        if kind == "detection_tool":
            from mn_protein_design.workflows.detection import run_masif_seed, run_pesto, run_scannet, run_surf2spot

            metadata = read_json(run_dir / "metadata.json")
            input_payload = read_json(run_dir / "input.json")
            tool = str(input_payload.get("tool") or metadata.get("tool") or "")
            inputs = input_payload.get("inputs") if isinstance(input_payload.get("inputs"), dict) else {}
            params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
            job = JobPaths(task_group=str(metadata.get("task_group") or ""), run_id=run_dir.name, run_dir=run_dir)
            target_pdb = Path(str(inputs.get("target_pdb") or "")).expanduser()
            gpu_device = params.get("gpu_device", "0")
            if tool == "scannet":
                run_scannet(
                    target_pdb,
                    [str(chain) for chain in inputs.get("chains") or []],
                    mode=str(params.get("mode") or "interface"),
                    use_msa=bool(params.get("use_msa")),
                    gpu_device=gpu_device,
                    existing_job=job,
                )
                return 0
            if tool == "surf2spot":
                run_surf2spot(target_pdb, gpu_device=gpu_device, existing_job=job)
                return 0
            if tool == "pesto":
                run_pesto(
                    target_pdb,
                    [str(chain) for chain in inputs.get("chains") or []],
                    gpu_device=gpu_device,
                    existing_job=job,
                )
                return 0
            if tool == "masif_seed":
                run_masif_seed(
                    target_pdb,
                    str(inputs.get("chain_id") or ""),
                    gpu_device=gpu_device,
                    existing_job=job,
                )
                return 0
            raise RuntimeError(f"Unknown detection tool for local worker: {tool}")
        raise RuntimeError(f"Unknown local worker request kind: {kind}")
    except Exception as exc:
        _worker_log(run_dir, "worker_stderr.log", traceback.format_exc())
        result = read_json(run_dir / "result.json")
        if not result:
            finish_job(
                run_dir,
                False,
                {
                    "metrics": {
                        "worker_error": str(exc),
                        "worker_exception_type": type(exc).__name__,
                    }
                },
            )
        else:
            update_status(run_dir, "failed", worker_error=str(exc), worker_exception_type=type(exc).__name__)
        return 1


def spawn_worker_for_run(run_dir: Path) -> subprocess.Popen:
    run_dir = run_dir.expanduser().resolve()
    stdout = (run_dir / "worker_stdout.log").open("a")
    stderr = (run_dir / "worker_stderr.log").open("a")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "mn_protein_design.core.local_worker",
            "--run-dir",
            str(run_dir),
        ],
        cwd=str(REPO_ROOT),
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    metadata = read_json(run_dir / "metadata.json")
    metadata["worker_pid"] = proc.pid
    metadata["worker_started_at"] = utc_now()
    metadata["updated_at"] = metadata["worker_started_at"]
    write_json(run_dir / "metadata.json", metadata)
    return proc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a queued mn-protein-design local worker job.")
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(run_worker_job(args.run_dir))


if __name__ == "__main__":
    main()
