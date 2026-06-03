from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from mn_protein_design.core.candidates import candidate_stage_counts, read_candidates
from mn_protein_design.core.jobs import ACTIVE_STATUSES, collect_jobs, delete_job_run, deletion_plan, find_downstream_jobs
from mn_protein_design.workflows.campaigns import run_lineage_steps


def render_job_table(task_group: str | list[str] | None = None, rows_override: list[dict] | None = None) -> None:
    table_key = (
        "all"
        if task_group is None
        else "_".join(task_group)
        if isinstance(task_group, list)
        else task_group
    )
    table_key = "".join(ch if ch.isalnum() else "_" for ch in table_key)

    if rows_override is not None:
        rows = rows_override
    elif isinstance(task_group, list):
        rows = []
        for group in task_group:
            rows.extend(collect_jobs(group))
        rows.sort(
            key=lambda row: row.get("updated_at") or row.get("created_at") or "",
            reverse=True,
        )
    else:
        rows = collect_jobs(task_group)
    if not rows:
        st.info("No jobs yet.")
        return
    df = pd.DataFrame(rows)
    df["job_code_link"] = df.apply(
        lambda row: (
            f"/results?task_group={row['task_group']}"
            f"&run_id={row['run_id']}"
            f"&job_code={row['job_code']}"
        ),
        axis=1,
    )
    df["delete"] = False
    display_df = df[
        [
            "delete",
            "job_code_link",
            "campaign_name",
            "task_group",
            "job_type",
            "tool",
            "status",
            "queue_resource",
            "current_phase",
            "current_engine",
            "created_at",
            "updated_at",
            "run_id",
        ]
    ].rename(columns={"job_code_link": "job_code"})
    disabled_delete_rows = [
        idx for idx, row in display_df.iterrows() if row["status"] in ACTIVE_STATUSES
    ]
    edited_df = st.data_editor(
        display_df,
        hide_index=True,
        width="stretch",
        key=f"{table_key}_jobs_table",
        num_rows="fixed",
        disabled=[
            "job_code",
            "campaign_name",
            "task_group",
            "job_type",
            "tool",
            "status",
            "queue_resource",
            "current_phase",
            "current_engine",
            "created_at",
            "updated_at",
            "run_id",
        ],
        column_config={
            "delete": st.column_config.CheckboxColumn(
                "delete",
                help="Tick finished jobs to delete.",
                default=False,
            ),
            "job_code": st.column_config.LinkColumn(
                "job_code",
                display_text=r".*job_code=([A-Z0-9]+)$",
                help="Open this job's result page.",
            ),
        },
    )

    if disabled_delete_rows and edited_df.loc[disabled_delete_rows, "delete"].any():
        st.warning("Running, queued, or preparing jobs cannot be deleted.")
        edited_df.loc[disabled_delete_rows, "delete"] = False

    edited_df["delete"] = edited_df["delete"].fillna(False).astype(bool)
    editor_state = st.session_state.get(f"{table_key}_jobs_table")
    if isinstance(editor_state, dict):
        for raw_index, changes in (editor_state.get("edited_rows") or {}).items():
            if not isinstance(changes, dict) or "delete" not in changes:
                continue
            try:
                row_index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if row_index in edited_df.index:
                edited_df.at[row_index, "delete"] = bool(changes["delete"])

    selected_rows = edited_df[
        edited_df["delete"] & ~edited_df["status"].isin(ACTIVE_STATUSES)
    ]
    selected_refs = [
        (str(row["task_group"]), str(row["run_id"]))
        for row in selected_rows.to_dict(orient="records")
    ]
    st.session_state[f"{table_key}_delete_refs"] = selected_refs

    if selected_rows.empty:
        st.caption("Tick finished jobs in the delete column to show delete options.")
    else:
        downstream_rows = []
        downstream_seen = set()
        for task, run_id in selected_refs:
            for downstream in find_downstream_jobs(task, run_id):
                identity = (str(downstream["task_group"]), str(downstream["run_id"]))
                if identity in downstream_seen or identity in selected_refs:
                    continue
                downstream_seen.add(identity)
                downstream_rows.append(downstream)
        include_downstream = False
        if downstream_rows:
            st.warning(f"{len(downstream_rows)} downstream job(s) depend on the selected job(s).")
            with st.expander("Downstream jobs that would become orphaned", expanded=True):
                st.dataframe(
                    pd.DataFrame(downstream_rows)[
                        ["job_code", "task_group", "job_type", "tool", "status", "run_id"]
                    ],
                    hide_index=True,
                    width="stretch",
                )
            include_downstream = st.checkbox(
                "Also delete downstream dependent jobs",
                key=f"{table_key}_include_downstream_delete_jobs",
            )
        planned_rows = deletion_plan(selected_refs, include_downstream=include_downstream)
        active_planned = [row for row in planned_rows if row.get("status") in ACTIVE_STATUSES]
        if active_planned:
            st.error("The deletion plan includes running, queued, or preparing jobs. Stop those jobs first.")
            st.dataframe(
                pd.DataFrame(active_planned)[["job_code", "task_group", "job_type", "tool", "status", "run_id"]],
                hide_index=True,
                width="stretch",
            )
        st.caption(f"Deletion plan: {len(planned_rows)} run director{'y' if len(planned_rows) == 1 else 'ies'}.")
        confirm = st.checkbox(
            "I understand this permanently deletes the planned run directories.",
            key=f"{table_key}_confirm_delete_jobs",
        )

        if st.button(
            "Delete selected jobs",
            type="primary",
            disabled=not confirm or bool(active_planned),
            key=f"{table_key}_delete_selected_jobs",
        ):
            deleted = []
            try:
                for row in planned_rows:
                    delete_job_run(str(row["task_group"]), str(row["run_id"]))
                    code = str(row["job_code"]).rsplit("job_code=", 1)[-1]
                    deleted.append(code)
            except Exception as exc:
                st.error(f"Delete failed: {exc}")
            else:
                st.session_state[f"{table_key}_delete_refs"] = []
                st.success(f"Deleted {len(deleted)} job(s): {', '.join(deleted)}")
                st.rerun()


def result_link(task_group: str, run_id: str, label: str = "Open result") -> str:
    return f"/results?task_group={task_group}&run_id={run_id}"


def source_from_run_dir(run_dir: Path) -> dict:
    from mn_protein_design.core.jobs import read_json

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


def run_steps_after_source(campaign_name: str, source: dict, steps: list[dict]) -> tuple[Path | None, list[Path]]:
    return None, run_lineage_steps(campaign_name, source, steps)


def show_pipeline_links(campaign_run: Path | None, child_runs: list[Path]) -> None:
    from mn_protein_design.core.jobs import read_json

    for child_run in child_runs:
        metadata = read_json(child_run / "metadata.json")
        task_group = str(metadata.get("task_group") or "")
        label = str(metadata.get("job_type") or metadata.get("tool") or child_run.name).replace("_", " ")
        if task_group:
            st.link_button(f"Open {label}", result_link(task_group, child_run.name))


def show_contract_files(run_dir: Path) -> None:
    cols = st.columns(6)
    for col, name in zip(cols, ["input.json", "metadata.json", "command.json", "stdout.log", "stderr.log", "result.json"]):
        path = run_dir / name
        col.caption(name)
        col.write("ok" if path.exists() else "missing")
