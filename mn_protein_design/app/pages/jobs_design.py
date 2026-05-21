from __future__ import annotations

import streamlit as st

from mn_protein_design.app.pages.common import render_job_table


st.title("Design Jobs")
render_job_table("design")
