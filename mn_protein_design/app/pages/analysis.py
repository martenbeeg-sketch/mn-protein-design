from __future__ import annotations

import pandas as pd
import streamlit as st

from mn_protein_design.workflows.modules import candidate_sources, load_source_candidates, module_specs


st.title("Analysis")
st.caption("Consumes normalized candidates from any module and prepares filters, ranking, clustering, and reports.")

analysis_spec = [row for row in module_specs() if row["module_id"] == "analysis"][0]
st.json(analysis_spec)

sources = candidate_sources()
if not sources:
    st.info("No normalized candidate sets are available yet.")
    st.stop()

labels = [
    f"{source['task_group']} / {source['job_code']} | {source['tool']} | {source['candidate_count']} candidates"
    for source in sources
]
selected = st.selectbox("Candidate set", range(len(sources)), format_func=lambda index: labels[index])
source = sources[selected]
candidates = load_source_candidates(source)

rows = []
for candidate in candidates:
    metrics = candidate.get("metrics") or {}
    rows.append(
        {
            "candidate_id": candidate.get("candidate_id"),
            "stage": candidate.get("stage"),
            "source_tool": candidate.get("source_tool"),
            "complex_pdb": candidate.get("complex_pdb"),
            "binder_sequence": candidate.get("binder_sequence"),
            "metric_count": len(metrics),
        }
    )

st.subheader("Normalized Candidate Table")
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

st.info("Next implementation step: add saved filters and ranking outputs as an analysis job that writes normalized analysis candidates.")
