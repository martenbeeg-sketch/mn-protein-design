from __future__ import annotations

import streamlit as st

from mn_protein_design.app.pages.common import render_job_table


st.title("Jobs / Results")
st.caption("All file-backed jobs across target preparation, detection, design, refolding, and analysis.")
render_job_table()
