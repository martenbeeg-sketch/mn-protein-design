from __future__ import annotations

from pathlib import Path
from typing import Any

from mn_protein_design.core.candidates import candidate_stage_counts, read_candidates
from mn_protein_design.core.jobs import collect_jobs, create_job, read_json, update_status, utc_now, write_json
from mn_protein_design.core.pipeline import (
    CAMPAIGN_GROUP,
    PIPELINE_SCHEMA_VERSION,
    new_step,
    read_pipeline,
    step_source,
    write_pipeline,
)
from mn_protein_design.workflows.analysis import run_analysis_contract
from mn_protein_design.workflows.refolding import (
    run_complex_refolding_contract,
    run_monomer_refolding_contract,
)
from mn_protein_design.workflows.sequence_design import (
    run_foundry_mpnn_sequence_design,
    run_ligandmpnn_sequence_design,
)


def _source_snapshot(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_group": source.get("task_group"),
        "run_id": source.get("run_id"),
        "run_dir": source.get("run_dir"),
        "job_code": source.get("job_code"),
        "tool": source.get("tool"),
        "candidates_jsonl": source.get("candidates_jsonl"),
        "candidate_count": source.get("candidate_count"),
        "stage_counts": source.get("stage_counts", {}),
    }


def _job_ref_from_run(run_dir: Path) -> dict[str, Any]:
    metadata = read_json(run_dir / "metadata.json")
    candidates = read_candidates(run_dir)
    return {
        "task_group": metadata.get("task_group"),
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "job_code": metadata.get("job_code"),
        "tool": metadata.get("tool"),
        "candidates_jsonl": str(run_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl"),
        "candidate_count": len(candidates),
        "stage_counts": candidate_stage_counts(candidates),
    }


def _tag_lineage_job(run_dir: Path, campaign_name: str, campaign_id: str, step: dict[str, Any], step_index: int, source: dict[str, Any]) -> None:
    metadata = read_json(run_dir / "metadata.json")
    metadata.update(
        {
            "campaign_name": campaign_name,
            "campaign_id": campaign_id,
            "campaign_step": str(step.get("module") or ""),
            "campaign_step_index": step_index,
            "upstream_task_group": source.get("task_group"),
            "upstream_run_id": source.get("run_id"),
            "upstream_job_code": source.get("job_code"),
        }
    )
    write_json(run_dir / "metadata.json", metadata)


def _run_module_step(source: dict[str, Any], step: dict[str, Any]) -> Path:
    module = str(step.get("module"))
    tool = str(step.get("tool"))
    params = dict(step.get("params") or {})
    if module == "sequence_design" and tool == "ligandmpnn":
        return run_ligandmpnn_sequence_design(
            source_run_dir=Path(str(source["run_dir"])),
            candidates_jsonl=Path(str(source["candidates_jsonl"])),
            model_type=str(params.get("model_type") or "protein_mpnn"),
            design_chains=str(params.get("design_chains") or ""),
            num_seq_per_target=int(params.get("num_seq_per_target") or 1),
            sampling_temp=float(params.get("sampling_temp") or 0.0001),
            omit_aas=str(params.get("omit_aas") or "CX"),
            seed=int(params["seed"]) if params.get("seed") else None,
            require_backbone_hotspot_filter_pass=bool(params.get("require_backbone_hotspot_filter_pass", False)),
        )
    if module == "sequence_design" and tool == "foundry_mpnn":
        return run_foundry_mpnn_sequence_design(
            source_run_dir=Path(str(source["run_dir"])),
            candidates_jsonl=Path(str(source["candidates_jsonl"])),
            number_of_batches=int(params.get("number_of_batches") or 1),
            batch_size=int(params.get("batch_size") or 10),
            model_type=str(params.get("model_type") or "ligand_mpnn"),
            checkpoint_path=str(params.get("checkpoint_path") or "/weights/ligandmpnn_v_32_010_25.pt"),
        )
    if module == "monomer_refolding":
        return run_monomer_refolding_contract(
            source_run_dir=Path(str(source["run_dir"])),
            candidates_jsonl=Path(str(source["candidates_jsonl"])),
            tool=tool,
            min_plddt=float(params.get("min_plddt") or 70.0),
        )
    if module == "complex_refolding":
        return run_complex_refolding_contract(
            source_run_dir=Path(str(source["run_dir"])),
            candidates_jsonl=Path(str(source["candidates_jsonl"])),
            tool=tool,
            require_monomer_success=bool(params.get("require_monomer_success", True)),
            template_mode=str(params.get("template_mode") or "target_template"),
            num_recycles=int(params.get("num_recycles") or 3),
            multimer=bool(params.get("multimer", True)),
        )
    if module == "analysis":
        return run_analysis_contract(
            source_run_dir=Path(str(source["run_dir"])),
            candidates_jsonl=Path(str(source["candidates_jsonl"])),
            tool=tool,
            keep_top_n=int(params.get("keep_top_n") or 100),
            thresholds=dict(params.get("thresholds") or {}),
        )
    raise NotImplementedError(f"{module}/{tool} is registered, but no runner is wired yet.")


def run_lineage_steps(campaign_name: str, initial_source: dict[str, Any], steps: list[dict[str, Any]]) -> list[Path]:
    child_runs: list[Path] = []
    source = dict(initial_source)
    campaign_label = campaign_name.strip() or f"Pipeline from {source.get('job_code') or source.get('run_id') or 'source'}"
    campaign_id = str(source.get("run_id") or source.get("job_code") or "").strip() or utc_now()

    source_run_dir = Path(str(source.get("run_dir") or ""))
    if source_run_dir.exists():
        _tag_lineage_job(
            source_run_dir,
            campaign_label,
            campaign_id,
            {"module": "backbone_generation"},
            0,
            {},
        )

    for index, step in enumerate(steps, start=1):
        child_run_dir = _run_module_step(source, step)
        _tag_lineage_job(child_run_dir, campaign_label, campaign_id, step, index, source)
        child_runs.append(child_run_dir)
        result = read_json(child_run_dir / "result.json")
        if result.get("success") is not True:
            break
        source = _job_ref_from_run(child_run_dir)
    return child_runs


def _pipeline_result(payload: dict[str, Any]) -> dict[str, Any]:
    completed_refs = [
        step.get("job_ref")
        for step in payload.get("steps", [])
        if step.get("status") == "completed" and step.get("job_ref")
    ]
    return {
        "success": True if payload.get("status") == "completed" else False if payload.get("status") == "failed" else None,
        "job_type": "campaign_pipeline",
        "tool": "campaign",
        "outputs": {
            "pipeline": "pipeline.json",
            "completed_jobs": completed_refs,
        },
        "metrics": {
            "step_count": len(payload.get("steps", [])),
            "completed_steps": len(completed_refs),
        },
    }


def create_campaign(name: str, initial_source: dict[str, Any], steps: list[dict[str, Any]]) -> Path:
    if not steps:
        raise ValueError("A campaign needs at least one module step.")
    job = create_job(
        CAMPAIGN_GROUP,
        job_type="campaign_pipeline",
        tool="campaign",
        inputs={"initial_source": _source_snapshot(initial_source)},
        params={"name": name},
    )
    pipeline = {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "name": name.strip() or job.run_id,
        "status": "created",
        "initial_source": _source_snapshot(initial_source),
        "steps": [
            new_step(
                step_id=f"{step['module']}_{index:02d}",
                module=str(step["module"]),
                tool=str(step["tool"]),
                params=dict(step.get("params") or {}),
            )
            for index, step in enumerate(steps, start=1)
        ],
        "created_at": utc_now(),
    }
    write_pipeline(job.run_dir, pipeline)
    write_json(job.run_dir / "result.json", _pipeline_result(pipeline))
    update_status(job.run_dir, "queued")
    return job.run_dir


def list_campaigns() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in collect_jobs(CAMPAIGN_GROUP):
        run_dir = Path(job["run_dir"])
        pipeline = read_pipeline(run_dir)
        rows.append({**job, "pipeline": pipeline, "name": pipeline.get("name") or job["run_id"]})
    return rows


def run_next_step(campaign_run_dir: Path) -> Path | None:
    campaign_run_dir = Path(campaign_run_dir)
    pipeline = read_pipeline(campaign_run_dir)
    if pipeline.get("status") == "failed":
        raise ValueError("Campaign is failed. Create a new campaign or reset the failed step before running more steps.")
    steps = list(pipeline.get("steps") or [])
    next_index = None
    for index, step in enumerate(steps):
        if step.get("status") == "failed":
            raise ValueError(f"Campaign step {step.get('step_id') or index + 1} is failed. Create a new campaign or reset it before retrying.")
        if step.get("status") != "completed":
            next_index = index
            break
    if next_index is None:
        pipeline["status"] = "completed"
        write_pipeline(campaign_run_dir, pipeline)
        write_json(campaign_run_dir / "result.json", _pipeline_result(pipeline))
        update_status(campaign_run_dir, "completed")
        return None

    source = step_source(pipeline, next_index)
    if not source.get("run_dir") or not source.get("candidates_jsonl"):
        raise ValueError("The selected step does not have a usable upstream candidate source.")

    step = steps[next_index]
    step["status"] = "running"
    step["updated_at"] = utc_now()
    pipeline["status"] = "running"
    pipeline["steps"] = steps
    write_pipeline(campaign_run_dir, pipeline)
    update_status(campaign_run_dir, "running")

    try:
        module = str(step.get("module"))
        tool = str(step.get("tool"))
        params = dict(step.get("params") or {})
        if module == "sequence_design" and tool == "ligandmpnn":
            child_run_dir = run_ligandmpnn_sequence_design(
                source_run_dir=Path(str(source["run_dir"])),
                candidates_jsonl=Path(str(source["candidates_jsonl"])),
                model_type=str(params.get("model_type") or "protein_mpnn"),
                design_chains=str(params.get("design_chains") or ""),
                num_seq_per_target=int(params.get("num_seq_per_target") or 1),
                sampling_temp=float(params.get("sampling_temp") or 0.0001),
                omit_aas=str(params.get("omit_aas") or "CX"),
                seed=int(params["seed"]) if params.get("seed") else None,
                require_backbone_hotspot_filter_pass=bool(params.get("require_backbone_hotspot_filter_pass", False)),
            )
        elif module == "sequence_design" and tool == "foundry_mpnn":
            child_run_dir = run_foundry_mpnn_sequence_design(
                source_run_dir=Path(str(source["run_dir"])),
                candidates_jsonl=Path(str(source["candidates_jsonl"])),
                number_of_batches=int(params.get("number_of_batches") or 1),
                batch_size=int(params.get("batch_size") or 10),
                model_type=str(params.get("model_type") or "ligand_mpnn"),
                checkpoint_path=str(params.get("checkpoint_path") or "/weights/ligandmpnn_v_32_010_25.pt"),
            )
        elif module == "monomer_refolding":
            child_run_dir = run_monomer_refolding_contract(
                source_run_dir=Path(str(source["run_dir"])),
                candidates_jsonl=Path(str(source["candidates_jsonl"])),
                tool=tool,
                min_plddt=float(params.get("min_plddt") or 70.0),
            )
        elif module == "complex_refolding":
            child_run_dir = run_complex_refolding_contract(
                source_run_dir=Path(str(source["run_dir"])),
                candidates_jsonl=Path(str(source["candidates_jsonl"])),
                tool=tool,
                require_monomer_success=bool(params.get("require_monomer_success", True)),
                template_mode=str(params.get("template_mode") or "target_template"),
                num_recycles=int(params.get("num_recycles") or 3),
                multimer=bool(params.get("multimer", True)),
            )
        elif module == "analysis":
            child_run_dir = run_analysis_contract(
                source_run_dir=Path(str(source["run_dir"])),
                candidates_jsonl=Path(str(source["candidates_jsonl"])),
                tool=tool,
                keep_top_n=int(params.get("keep_top_n") or 100),
                thresholds=dict(params.get("thresholds") or {}),
            )
        else:
            raise NotImplementedError(f"{module}/{tool} is registered in the campaign, but no runner is wired yet.")
    except Exception:
        pipeline = read_pipeline(campaign_run_dir)
        pipeline_steps = list(pipeline.get("steps") or [])
        pipeline_steps[next_index]["status"] = "failed"
        pipeline_steps[next_index]["updated_at"] = utc_now()
        pipeline["status"] = "failed"
        pipeline["steps"] = pipeline_steps
        write_pipeline(campaign_run_dir, pipeline)
        write_json(campaign_run_dir / "result.json", _pipeline_result(pipeline))
        update_status(campaign_run_dir, "failed")
        raise

    child_result = read_json(child_run_dir / "result.json")
    if child_result.get("success") is not True:
        pipeline = read_pipeline(campaign_run_dir)
        pipeline_steps = list(pipeline.get("steps") or [])
        pipeline_steps[next_index]["status"] = "failed"
        pipeline_steps[next_index]["updated_at"] = utc_now()
        pipeline_steps[next_index]["job_ref"] = _job_ref_from_run(child_run_dir)
        pipeline["status"] = "failed"
        pipeline["steps"] = pipeline_steps
        write_pipeline(campaign_run_dir, pipeline)
        write_json(campaign_run_dir / "result.json", _pipeline_result(pipeline))
        update_status(campaign_run_dir, "failed")
        return child_run_dir

    pipeline = read_pipeline(campaign_run_dir)
    pipeline_steps = list(pipeline.get("steps") or [])
    job_ref = _job_ref_from_run(child_run_dir)
    pipeline_steps[next_index]["status"] = "completed"
    pipeline_steps[next_index]["updated_at"] = utc_now()
    pipeline_steps[next_index]["job_ref"] = job_ref
    pipeline["steps"] = pipeline_steps
    pipeline["status"] = "completed" if all(step.get("status") == "completed" for step in pipeline_steps) else "running"
    write_pipeline(campaign_run_dir, pipeline)
    write_json(campaign_run_dir / "result.json", _pipeline_result(pipeline))
    update_status(campaign_run_dir, pipeline["status"])
    return child_run_dir
