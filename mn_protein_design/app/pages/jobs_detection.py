from __future__ import annotations

import streamlit as st

from mn_protein_design.app.pages.common import render_job_table


st.title("Detection Jobs")
st.subheader("PPI / Surface Detection")
render_job_table("detection")
st.subheader("Hotspot Detection")
render_job_table("hotspot-detection")
st.subheader("Target Cropping")
render_job_table("target-crop")
