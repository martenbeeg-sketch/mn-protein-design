from __future__ import annotations

import os

import pandas as pd
import streamlit as st

_previous_import_flag = os.environ.get("MN_PROTEIN_DESIGN_IMPORT_CAPACITY_HELPERS")
os.environ["MN_PROTEIN_DESIGN_IMPORT_CAPACITY_HELPERS"] = "1"
from mn_protein_design.app.pages.capacity_benchmark import (  # noqa: E402
    _design_capacity_batch_table,
    _design_capacity_rows,
    _query_param_value,
    _render_design_capacity_batch_detail,
)
if _previous_import_flag is None:
    os.environ.pop("MN_PROTEIN_DESIGN_IMPORT_CAPACITY_HELPERS", None)
else:
    os.environ["MN_PROTEIN_DESIGN_IMPORT_CAPACITY_HELPERS"] = _previous_import_flag

from mn_protein_design.app.pages.common import refresh_results_button  # noqa: E402


st.title("Design Capacity Batch Results")
refresh_results_button("design_capacity_batch_results_refresh")

selected_batch = _query_param_value("design_capacity_batch")
if not selected_batch:
    st.info("Open this page from a Design Capacity batch row.")
    st.link_button("Back to Capacity Benchmark", "/capacity-benchmark")
    st.stop()

rows = _design_capacity_rows()
if not rows:
    st.info("No design capacity runs are available.")
    st.link_button("Back to Capacity Benchmark", "/capacity-benchmark")
    st.stop()

df = pd.DataFrame(rows)
batch_df = _design_capacity_batch_table(df)
_render_design_capacity_batch_detail(df, batch_df, selected_batch, back_url="/capacity-benchmark")
