from __future__ import annotations

import pandas as pd
import streamlit as st

from mn_protein_design.core.candidates import STAGE_GENERATION_BACKBONE
from mn_protein_design.app.pages.common import result_link, run_steps_after_source, show_pipeline_links, source_from_run_dir
from mn_protein_design.workflows.modules import candidate_sources, load_source_candidates
from mn_protein_design.workflows.sequence_design import run_foundry_mpnn_sequence_design, run_ligandmpnn_sequence_design


st.title("Sequence Design / Optimization")
st.caption("Consumes normalized backbone candidates and produces sequence-designed binder candidates.")

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

st.subheader("Downstream Workflow")
continue_downstream = st.checkbox(
    "Continue with monomer refolding, complex refolding, and analysis after sequence design",
    value=False,
)
downstream_steps: list[dict] = []
if continue_downstream:
    st.markdown("**Monomer Refolding**")
    monomer_tool = st.selectbox(
        "Monomer refolding",
        ["boltz2_monomer", "esmfold", "af2_monomer"],
        format_func={
            "boltz2_monomer": "Boltz2 monomer",
            "esmfold": "ESMFold",
            "af2_monomer": "AF2 monomer contract",
        }.get,
        key="sequence_continue_monomer_tool",
    )
    min_plddt = st.number_input("Minimum monomer pLDDT", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    downstream_steps.append(
        {
            "module": "monomer_refolding",
            "tool": str(monomer_tool),
            "params": {"min_plddt": float(min_plddt)},
        }
    )

    st.markdown("**Complex Refolding**")
    complex_tool = st.selectbox(
        "Complex refolding",
        ["af2_initial_guess", "boltz2_initial_guess"],
        format_func={
            "af2_initial_guess": "AF2 initial guess",
            "boltz2_initial_guess": "Boltz2 initial guess",
        }.get,
        key="sequence_continue_complex_tool",
    )
    if complex_tool == "boltz2_initial_guess":
        template_mode = st.segmented_control(
            "Boltz2 template mode",
            ["target_template", "no_template"],
            selection_mode="single",
            default="target_template",
            format_func={"target_template": "Target template", "no_template": "No template"}.get,
        )
        multimer = True
    else:
        af2_options = {
            "af2_model_1_multimer_tt_3rec": ("AF2 multimer, target template", "target_template", True, 3),
            "af2_model_1_ptm_tt_3rec": ("AF2 monomer model, target template", "target_template", False, 3),
            "af2_model_1_multimer_tbt_3rec": ("AF2 multimer, target + binder templates", "target_binder_template", True, 3),
            "af2_model_1_multimer_ct_3rec": ("AF2 multimer, complex template", "complex_template", True, 3),
        }
        af2_choice = st.selectbox(
            "AF2 complex refolding model",
            list(af2_options),
            format_func=lambda key: af2_options[key][0],
            key="sequence_continue_af2_model",
        )
        _label, template_mode, multimer, _recycles = af2_options[af2_choice]
    downstream_steps.append(
        {
            "module": "complex_refolding",
            "tool": str(complex_tool),
            "params": {
                "require_monomer_success": True,
                "template_mode": str(template_mode or "target_template"),
                "multimer": bool(multimer),
                "num_recycles": 3,
            },
        }
    )

    st.markdown("**Analysis**")
    analysis_cols = st.columns(5)
    with analysis_cols[0]:
        keep_top_n = st.number_input("Keep top", min_value=1, max_value=10000, value=100, step=10)
    with analysis_cols[1]:
        min_binder_plddt = st.number_input("Min binder pLDDT", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    with analysis_cols[2]:
        max_ipae = st.number_input("Max iPAE", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    with analysis_cols[3]:
        min_ipsae = st.number_input("Min ipSAE", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    with analysis_cols[4]:
        max_binder_rmsd = st.number_input("Max binder RMSD", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    downstream_steps.append(
        {
            "module": "analysis",
            "tool": "ranking",
            "params": {
                "keep_top_n": int(keep_top_n),
                "thresholds": {
                    "min_binder_plddt": float(min_binder_plddt),
                    "min_confidence": 0.0,
                    "min_iptm": 0.0,
                    "min_ipsae": float(min_ipsae),
                    "max_ipae": float(max_ipae),
                    "max_ipde": 20.0,
                    "max_binder_rmsd": float(max_binder_rmsd),
                },
            },
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
        if continue_downstream and downstream_steps:
            with st.spinner("Continuing downstream workflow..."):
                campaign_run, child_runs = run_steps_after_source(
                    f"Continue from sequence design {run_dir.name}",
                    source_from_run_dir(run_dir),
                    downstream_steps,
                )
            st.success("Downstream workflow finished.")
            show_pipeline_links(campaign_run, child_runs)
    except Exception as exc:
        st.error(str(exc))
