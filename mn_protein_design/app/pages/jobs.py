from __future__ import annotations

import streamlit as st

from mn_protein_design.app.pages.common import render_job_table
from mn_protein_design.core.jobs import collect_jobs


st.title("Jobs")
st.caption("All file-backed jobs across target preparation, detection, design, refolding, and analysis.")

all_rows = collect_jobs()
if not all_rows:
    st.info("No jobs yet.")
    st.stop()

task_groups = sorted({str(row.get("task_group")) for row in all_rows if row.get("task_group")})
job_types = sorted({str(row.get("job_type")) for row in all_rows if row.get("job_type")})
tools = sorted({str(row.get("tool")) for row in all_rows if row.get("tool")})
statuses = sorted({str(row.get("status")) for row in all_rows if row.get("status")})

selected_task_groups = st.multiselect("Task group", task_groups, default=task_groups)
selected_job_types = st.multiselect("Job type", job_types, default=job_types)
selected_tools = st.multiselect("Tool", tools, default=tools)
selected_statuses = st.multiselect("Status", statuses, default=statuses)

filtered_rows = [
    row
    for row in all_rows
    if str(row.get("task_group")) in set(selected_task_groups)
    and str(row.get("job_type")) in set(selected_job_types)
    and str(row.get("tool")) in set(selected_tools)
    and str(row.get("status")) in set(selected_statuses)
]
filtered_rows.sort(key=lambda row: str(row.get("created_at") or ""), reverse=True)
st.caption(
    f"Showing {len(filtered_rows)} of {len(all_rows)} jobs."
)
render_job_table(rows_override=filtered_rows)
