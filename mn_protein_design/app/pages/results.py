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

analysis_csv = run_dir / "artifacts" / "analysis" / "ranked_candidates.csv"
if analysis_csv.exists():
    st.subheader("Analysis Results")
    try:
        df = pd.read_csv(analysis_csv)
        if df.empty:
            st.info("The ranked candidates table is empty.")
        else:
            metric_cols = st.columns(4)
            metric_cols[0].metric("Candidates", len(df))
            if "passes_filters" in df:
                metric_cols[1].metric("Passing", int(df["passes_filters"].fillna(False).sum()))
            if "analysis_score" in df:
                metric_cols[2].metric("Best score", f"{pd.to_numeric(df['analysis_score'], errors='coerce').max():.2f}")
            if "ipsae_error" in df:
                metric_cols[3].metric("IPSAE errors", int(df["ipsae_error"].fillna("").astype(bool).sum()))
            visible_cols = [
                "analysis_rank",
                "candidate_id",
                "passes_filters",
                "analysis_score",
                "binder_plddt",
                "confidence",
                "iptm",
                "ipsae",
                "ipsae_min",
                "ipsae_max",
                "ipsae_avg",
                "lis",
                "ipsae_min_in_calculation",
                "ipae",
                "pdockq_min",
                "pdockq_max",
                "pdockq2_min",
                "pdockq2_max",
                "ipde",
                "monomer_rmsd",
                "binder_rmsd",
                "hotspot_contact_fraction",
                "hotspots_contacted",
                "min_binder_to_hotspot_distance",
                "binder_interface_contacts",
                "hotspot_interface_contact_fraction",
                "filter_failures",
                "ipsae_error",
                "complex_pdb",
            ]
            st.dataframe(df[[col for col in visible_cols if col in df.columns]], hide_index=True, width="stretch")
            if "analysis_rank" in df and "analysis_score" in df:
                st.caption("Score by rank")
                st.line_chart(df.set_index("analysis_rank")[["analysis_score"]])
    except Exception as exc:
        st.warning(f"Could not preview analysis CSV: {exc}")

for name in ["metadata.json", "input.json", "command.json", "result.json"]:
    with st.expander(name, expanded=False):
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
