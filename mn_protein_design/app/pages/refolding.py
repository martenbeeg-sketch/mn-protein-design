from __future__ import annotations

import pandas as pd
import streamlit as st

from mn_protein_design.core.candidates import (
    STAGE_GENERATION_BACKBONE_SEQUENCE,
    STAGE_SEQUENCE_DESIGN,
)
from mn_protein_design.workflows.modules import candidate_sources, load_source_candidates, module_specs


st.title("Refolding / Validation")
st.caption("Consumes normalized sequence-designed candidates and produces monomer or complex refolding candidates.")

spec_rows = [row for row in module_specs() if row["module_id"] in {"monomer_refolding", "complex_refolding"}]
st.dataframe(
    pd.DataFrame(
        [
            {
                "Module": row["label"],
                "Tools": ", ".join(row["tools"]),
                "Consumes": ", ".join(row["consumes"]),
                "Produces": ", ".join(row["produces"]),
                "Backends": ", ".join(row["supported_backends"]),
            }
            for row in spec_rows
        ]
    ),
    use_container_width=True,
    hide_index=True,
)

sources = candidate_sources("design")
usable_sources = [
    source
    for source in sources
    if any(stage in source["stage_counts"] for stage in [STAGE_SEQUENCE_DESIGN, STAGE_GENERATION_BACKBONE_SEQUENCE])
]

if not usable_sources:
    st.info("No sequence-designed candidate sets are available yet. Run a generation tool that produces sequence candidates, or add the sequence-design module next.")
    st.stop()

labels = [
    f"{source['job_code']} | {source['tool']} | {source['candidate_count']} candidates | {source['stage_counts']}"
    for source in usable_sources
]
selected = st.selectbox("Candidate set", range(len(usable_sources)), format_func=lambda index: labels[index])
source = usable_sources[selected]
candidates = load_source_candidates(source)

st.subheader("Normalized Inputs")
st.dataframe(
    pd.DataFrame(
        [
            {
                "candidate_id": row.get("candidate_id"),
                "stage": row.get("stage"),
                "source_tool": row.get("source_tool"),
                "complex_pdb": row.get("complex_pdb"),
                "binder_sequence": row.get("binder_sequence"),
            }
            for row in candidates
        ]
    ),
    use_container_width=True,
    hide_index=True,
)

st.info("Next implementation step: add AF2 initial-guess and Boltz2 initial-guess runners that consume this normalized candidate set.")
