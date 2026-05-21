from __future__ import annotations

import pandas as pd
import streamlit as st

from mn_protein_design.core.candidates import STAGE_GENERATION_BACKBONE
from mn_protein_design.app.pages.common import result_link
from mn_protein_design.workflows.modules import candidate_sources, load_source_candidates, module_specs
from mn_protein_design.workflows.sequence_design import run_foundry_mpnn_sequence_design, run_ligandmpnn_sequence_design


st.title("Sequence Design / Optimization")
st.caption("Consumes normalized backbone candidates and produces sequence-designed binder candidates.")

sequence_spec = [row for row in module_specs() if row["module_id"] == "sequence_design"][0]
st.dataframe(
    pd.DataFrame(
        [
            {
                "Module": sequence_spec["label"],
                "Tools": ", ".join(sequence_spec["tools"]),
                "Consumes": ", ".join(sequence_spec["consumes"]),
                "Produces": ", ".join(sequence_spec["produces"]),
                "Backends": ", ".join(sequence_spec["supported_backends"]),
            }
        ]
    ),
    use_container_width=True,
    hide_index=True,
)

sources = [
    source
    for source in candidate_sources("design")
    if STAGE_GENERATION_BACKBONE in source["stage_counts"]
]

if not sources:
    st.info("No backbone candidate sets are available yet. Run RFdiffusion, RFdiffusion3, BoltzGen, or another generation tool first.")
    st.stop()

labels = [
    f"{source['job_code']} | {source['tool']} | {source['candidate_count']} candidates | {source['stage_counts']}"
    for source in sources
]
selected = st.selectbox("Backbone candidate set", range(len(sources)), format_func=lambda index: labels[index])
source = sources[selected]
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
                "contig": row.get("contig"),
            }
            for row in candidates
            if row.get("stage") == STAGE_GENERATION_BACKBONE
        ]
    ),
    use_container_width=True,
    hide_index=True,
)

source_tool = str(source.get("tool") or "")
default_backend = "foundry_native" if source_tool == "rfdiffusion3_foundry" else "shared_ligandmpnn"
backend = st.radio(
    "Sequence backend",
    ["shared_ligandmpnn", "foundry_native"],
    format_func={
        "shared_ligandmpnn": "Shared LigandMPNN container for PDB candidates",
        "foundry_native": "Foundry-native MPNN for CIF candidates",
    }.get,
    index=1 if default_backend == "foundry_native" else 0,
    horizontal=True,
)

if backend == "shared_ligandmpnn":
    model_type = st.radio(
        "MPNN model",
        ["protein_mpnn", "ligand_mpnn", "soluble_mpnn"],
        format_func={
            "protein_mpnn": "ProteinMPNN weights through LigandMPNN",
            "ligand_mpnn": "LigandMPNN",
            "soluble_mpnn": "SolubleMPNN",
        }.get,
        horizontal=True,
    )
    settings_cols = st.columns(3)
    with settings_cols[0]:
        num_seq_per_target = st.number_input("Sequences per backbone", min_value=1, max_value=1000, value=1, step=1)
    with settings_cols[1]:
        sampling_temp = st.number_input("Sampling temperature", min_value=0.0, max_value=5.0, value=0.0001, step=0.0001, format="%g")
    with settings_cols[2]:
        omit_aas = st.text_input("Omit amino acids", value="CX")
    design_chains = st.text_input(
        "Chains to design",
        value="",
        placeholder="Leave empty to infer non-target binder chain",
    )
    seed = st.number_input("Seed", min_value=0, max_value=999999999, value=0, step=1)
else:
    model_type = "ligand_mpnn"
    settings_cols = st.columns(2)
    with settings_cols[0]:
        foundry_batches = st.number_input("Number of batches", min_value=1, max_value=1000, value=1, step=1)
    with settings_cols[1]:
        foundry_batch_size = st.number_input("Batch size", min_value=1, max_value=1000, value=10, step=1)
    foundry_checkpoint = st.text_input("Foundry MPNN checkpoint", value="/weights/ligandmpnn_v_32_010_25.pt")

st.json(
    {
        "module": "sequence_design",
        "tool": "ligandmpnn" if backend == "shared_ligandmpnn" else "foundry_mpnn",
        "model_type": model_type,
        "backend": backend,
        "input_candidates": source["candidates_jsonl"],
        "produces": sequence_spec["produces"],
        "note": "RFdiffusion PDB outputs use the shared LigandMPNN container. RFdiffusion3 / Foundry CIF outputs can use Foundry-native MPNN.",
    }
)

if st.button("Run sequence design", type="primary"):
    try:
        with st.spinner("Running sequence design..."):
            if backend == "shared_ligandmpnn":
                run_dir = run_ligandmpnn_sequence_design(
                    source_run_dir=source["run_dir"],
                    candidates_jsonl=source["candidates_jsonl"],
                    model_type=model_type,
                    design_chains=design_chains,
                    num_seq_per_target=int(num_seq_per_target),
                    sampling_temp=float(sampling_temp),
                    omit_aas=omit_aas,
                    seed=int(seed) if int(seed) else None,
                )
            else:
                run_dir = run_foundry_mpnn_sequence_design(
                    source_run_dir=source["run_dir"],
                    candidates_jsonl=source["candidates_jsonl"],
                    number_of_batches=int(foundry_batches),
                    batch_size=int(foundry_batch_size),
                    checkpoint_path=foundry_checkpoint,
                )
        st.success("Sequence design job finished.")
        st.link_button("Open result", result_link("design", run_dir.name))
    except Exception as exc:
        st.error(str(exc))
