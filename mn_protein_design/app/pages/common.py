from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from mn_protein_design.core.jobs import ACTIVE_STATUSES, collect_jobs, delete_job_run


def render_job_table(task_group: str | list[str] | None = None) -> None:
    table_key = (
        "all"
        if task_group is None
        else "_".join(task_group)
        if isinstance(task_group, list)
        else task_group
    )
    table_key = "".join(ch if ch.isalnum() else "_" for ch in table_key)

    if isinstance(task_group, list):
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
            "task_group",
            "job_type",
            "tool",
            "status",
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
        disabled=[
            "job_code",
            "task_group",
            "job_type",
            "tool",
            "status",
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

    selected_rows = edited_df[
        edited_df["delete"] & ~edited_df["status"].isin(ACTIVE_STATUSES)
    ]
    if not selected_rows.empty:
        confirm = st.checkbox(
            "I understand this permanently deletes the selected run directories.",
            key=f"{table_key}_confirm_delete_jobs",
        )

        if st.button(
            "Delete selected jobs",
            type="primary",
            disabled=not confirm,
            key=f"{table_key}_delete_selected_jobs",
        ):
            deleted = []
            for row in selected_rows.to_dict(orient="records"):
                delete_job_run(str(row["task_group"]), str(row["run_id"]))
                code = str(row["job_code"]).rsplit("job_code=", 1)[-1]
                deleted.append(code)
            st.success(f"Deleted {len(deleted)} job(s): {', '.join(deleted)}")
            st.rerun()


def result_link(task_group: str, run_id: str, label: str = "Open result") -> str:
    return f"/results?task_group={task_group}&run_id={run_id}"


def show_contract_files(run_dir: Path) -> None:
    cols = st.columns(6)
    for col, name in zip(cols, ["input.json", "metadata.json", "command.json", "stdout.log", "stderr.log", "result.json"]):
        path = run_dir / name
        col.caption(name)
        col.write("ok" if path.exists() else "missing")
