from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from mn_protein_design.app.pages.common import show_contract_files
from mn_protein_design.core.artifacts import build_run_zip
from mn_protein_design.core.jobs import get_run_dir, read_json


task_group = st.query_params.get("task_group", "")
run_id = st.query_params.get("run_id", "")

st.title("Result Details")
if not task_group or not run_id:
    st.info("Select a job from a jobs page.")
    st.stop()

run_dir = get_run_dir(task_group, run_id)
if not run_dir.exists():
    st.error(f"Run not found: {run_dir}")
    st.stop()

st.caption(str(run_dir))
show_contract_files(run_dir)

zip_bytes = build_run_zip(run_dir)
st.download_button(
    "Download result zip",
    data=zip_bytes,
    file_name=f"{task_group}_{run_id}.zip",
    mime="application/zip",
    help="Creates a fresh zip from this run's contract files, logs, and artifacts. The zip is not stored in the workdir.",
)

for name in ["metadata.json", "input.json", "command.json", "result.json"]:
    with st.expander(name, expanded=name in {"metadata.json", "result.json"}):
        st.json(read_json(run_dir / name))

for log_name in ["stdout.log", "stderr.log"]:
    log_path = run_dir / log_name
    with st.expander(log_name):
        st.code(log_path.read_text(errors="ignore") if log_path.exists() else "", language="text")

artifacts_dir = run_dir / "artifacts"
if artifacts_dir.exists():
    st.subheader("Artifacts")
    for path in sorted(p for p in artifacts_dir.rglob("*") if p.is_file()):
        st.write(str(path.relative_to(run_dir)))
        if path.suffix.lower() == ".csv":
            with st.expander(f"Preview {path.name}"):
                try:
                    st.dataframe(pd.read_csv(path).head(200), hide_index=True, width="stretch")
                except Exception as exc:
                    st.warning(f"Could not preview CSV: {exc}")
