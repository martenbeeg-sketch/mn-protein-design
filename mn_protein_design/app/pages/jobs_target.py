from __future__ import annotations

import streamlit as st

from mn_protein_design.app.pages.common import render_job_table


st.title("Target Prep Jobs")
st.caption("Target preparation, cropping, and target-refolding jobs that produce prepared target structures.")
render_job_table(["target-prep", "target-crop", "target-refolding"])
