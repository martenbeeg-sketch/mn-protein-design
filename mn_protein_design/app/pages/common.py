from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from mn_protein_design.core.candidates import candidate_stage_counts, read_candidates
from mn_protein_design.core.jobs import (
    ACTIVE_STATUSES,
    collect_jobs,
    delete_job_run,
    deletion_plan,
    display_job_code,
    find_downstream_jobs,
    read_json,
)
from mn_protein_design.runtime import runs_root
from mn_protein_design.workflows.campaigns import run_lineage_steps


def selected_dataframe_rows(event: object, key: str | None = None) -> list[int]:
    """Return selected row positions from Streamlit dataframe selection events."""
    selection = getattr(event, "selection", None)
    if selection is not None:
        rows = [int(row) for row in (getattr(selection, "rows", []) or [])]
        if rows:
            return rows
    if isinstance(event, dict):
        rows = (event.get("selection") or {}).get("rows") or []
        rows = [int(row) for row in rows]
        if rows:
            return rows
    if key:
        state = st.session_state.get(key)
        if isinstance(state, dict):
            rows = (state.get("selection") or {}).get("rows") or []
            return [int(row) for row in rows]
    return []


def show_delete_jobs_dialog(
    *,
    table_key: str,
    pending_refs: list[tuple[str, str]],
    selected_refs_key: str,
    label: str = "job",
) -> None:
    def selected_job_summary() -> pd.DataFrame:
        rows = []
        root = runs_root()
        for task, run_id in pending_refs:
            metadata = read_json(root / str(task) / str(run_id) / "metadata.json")
            rows.append(
                {
                    "job_code": display_job_code(metadata.get("job_code"), str(run_id)),
                    "task_group": str(task),
                    "status": str(metadata.get("status") or ""),
                    "run_id": str(run_id),
                }
            )
        return pd.DataFrame(rows)

    def render_confirmation() -> None:
        st.warning(
            f"Delete {len(pending_refs)} selected {label}{'' if len(pending_refs) == 1 else 's'}? "
            "This permanently removes their run directories."
        )
        summary_df = selected_job_summary()
        if not summary_df.empty:
            st.markdown("**Selected jobs**")
            st.dataframe(summary_df, hide_index=True, width="stretch")

        include_downstream = False
        check_downstream = st.checkbox(
            "Check downstream dependent jobs",
            value=False,
            key=f"{table_key}_check_downstream_delete_jobs",
        )
        if check_downstream:
            downstream_rows = []
            downstream_seen = set()
            for task, run_id in pending_refs:
                for downstream in find_downstream_jobs(task, run_id):
                    identity = (str(downstream["task_group"]), str(downstream["run_id"]))
                    if identity in downstream_seen or identity in pending_refs:
                        continue
                    downstream_seen.add(identity)
                    downstream_rows.append(downstream)
            if downstream_rows:
                st.warning(f"{len(downstream_rows)} downstream job(s) depend on the selected job(s).")
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
            else:
                st.caption("No downstream dependent jobs found.")

        planned_rows = deletion_plan(pending_refs, include_downstream=include_downstream)
        active_planned = [row for row in planned_rows if row.get("status") in ACTIVE_STATUSES]
        if active_planned:
            st.error("The deletion includes running, queued, or preparing jobs. Stop those jobs first.")
            st.dataframe(
                pd.DataFrame(active_planned)[["job_code", "task_group", "job_type", "tool", "status", "run_id"]],
                hide_index=True,
                width="stretch",
            )
        st.caption(f"Will delete {len(planned_rows)} run director{'y' if len(planned_rows) == 1 else 'ies'}.")
        confirm = st.checkbox(
            "Yes, permanently delete these run directories.",
            key=f"{table_key}_confirm_delete_jobs",
        )
        cancel_col, delete_col = st.columns([1, 1])
        if cancel_col.button("Cancel", key=f"{table_key}_cancel_delete_jobs"):
            st.session_state[f"{table_key}_delete_refs"] = []
            st.session_state[selected_refs_key] = []
            st.rerun()
        if delete_col.button(
            "Confirm delete",
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
                st.session_state[f"{table_key}_delete_result"] = ("error", f"Delete failed: {exc}")
            else:
                st.session_state[f"{table_key}_delete_refs"] = []
                st.session_state[selected_refs_key] = []
                st.session_state[f"{table_key}_delete_result"] = (
                    "success",
                    f"Deleted {len(deleted)} job(s): {', '.join(deleted)}",
                )
            st.rerun()

    if hasattr(st, "dialog"):
        @st.dialog("Confirm deletion")
        def confirmation_dialog() -> None:
            render_confirmation()

        confirmation_dialog()
    else:
        render_confirmation()


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
    display_df = df[
        [
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
    event = st.dataframe(
        display_df,
        hide_index=True,
        width="stretch",
        key=f"{table_key}_jobs_table",
        on_select="rerun",
        selection_mode="multi-row",
        column_config={
            "job_code": st.column_config.LinkColumn(
                "job_code",
                display_text=r".*job_code=([A-Z0-9]+)$",
                help="Open this job's result page.",
            ),
        },
    )
    delete_result = st.session_state.pop(f"{table_key}_delete_result", None)
    if delete_result:
        level, message = delete_result
        if level == "success":
            st.success(str(message))
        else:
            st.error(str(message))

    table_widget_key = f"{table_key}_jobs_table"
    selected_indices = [idx for idx in selected_dataframe_rows(event, table_widget_key) if 0 <= idx < len(display_df)]
    selected_rows = display_df.iloc[selected_indices].copy() if selected_indices else display_df.iloc[0:0].copy()
    active_selected = selected_rows[selected_rows["status"].isin(ACTIVE_STATUSES)]
    if not active_selected.empty:
        st.warning("Running, queued, or preparing jobs cannot be deleted.")
    selected_rows = selected_rows[~selected_rows["status"].isin(ACTIVE_STATUSES)]
    selected_refs = [
        (str(row["task_group"]), str(row["run_id"]))
        for row in selected_rows.to_dict(orient="records")
    ]
    selected_refs_key = f"{table_key}_selected_delete_refs"
    if selected_refs:
        st.session_state[selected_refs_key] = selected_refs
    cached_selected_refs = st.session_state.get(selected_refs_key) or []

    delete_clicked = st.button(
        "Delete selected jobs",
        type="primary",
        disabled=not cached_selected_refs,
        key=f"{table_key}_request_delete_jobs",
    )
    if delete_clicked and cached_selected_refs:
        show_delete_jobs_dialog(
            table_key=table_key,
            pending_refs=cached_selected_refs,
            selected_refs_key=selected_refs_key,
            label="job",
        )
    else:
        st.caption("Select finished rows in the table to enable deletion.")


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
