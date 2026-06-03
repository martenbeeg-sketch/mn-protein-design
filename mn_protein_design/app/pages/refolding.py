from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from mn_protein_design.app.pages.common import run_steps_after_source, show_pipeline_links, source_from_run_dir
from mn_protein_design.core.candidates import (
    STAGE_COMPLEX_REFOLDING,
    STAGE_GENERATION_BACKBONE_SEQUENCE,
    STAGE_SEQUENCE_DESIGN,
)
from mn_protein_design.core.jobs import read_json
from mn_protein_design.workflows import refolding as refolding_workflow
from mn_protein_design.workflows.modules import candidate_sources, load_source_candidates


st.title("Refolding / Validation")
st.caption("Consumes normalized candidates and runs refolding/validation engines on existing designs.")

sources = candidate_sources()
usable_sources = [
    source
    for source in sources
    if any(
        stage in source["stage_counts"]
        for stage in [STAGE_SEQUENCE_DESIGN, STAGE_GENERATION_BACKBONE_SEQUENCE, STAGE_COMPLEX_REFOLDING]
    )
]

if not usable_sources:
    st.info("No sequence-designed or accepted-complex candidate sets are available yet. Run a generation tool first.")
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

st.subheader("ESMFold2 Complex Validation")
st.caption(
    "Refolds existing target+binder candidates with local Biohub ESMFold2 and writes a new normalized validation job."
)
esm_mode = st.segmented_control(
    "ESMFold2 mode",
    ["sequence", "initial_guess"],
    selection_mode="single",
    default="sequence",
    format_func={
        "sequence": "Sequence only",
        "initial_guess": "Initial guess",
    }.get,
    key="esmfold2_validation_mode",
)
esm_cols = st.columns(5)
with esm_cols[0]:
    esm_max_candidates = st.number_input(
        "Max candidates",
        min_value=1,
        max_value=1000,
        value=min(20, max(1, int(source["candidate_count"]))),
        step=1,
        key="esmfold2_validation_max_candidates",
    )
with esm_cols[1]:
    esm_sampling_steps = st.number_input(
        "Sampling steps",
        min_value=1,
        max_value=256,
        value=32,
        step=1,
        key="esmfold2_validation_sampling_steps",
    )
with esm_cols[2]:
    esm_loops = st.number_input(
        "Recycling loops",
        min_value=1,
        max_value=16,
        value=3,
        step=1,
        key="esmfold2_validation_loops",
    )
with esm_cols[3]:
    esm_seed = st.number_input(
        "Seed",
        min_value=0,
        max_value=999999,
        value=0,
        step=1,
        key="esmfold2_validation_seed",
    )
with esm_cols[4]:
    esm_contact_cutoff = st.number_input(
        "Contact cutoff",
        min_value=2.0,
        max_value=20.0,
        value=8.0,
        step=0.5,
        key="esmfold2_validation_contact_cutoff",
    )
esm_device = st.segmented_control(
    "ESMFold2 device",
    ["auto", "cuda", "cpu"],
    selection_mode="single",
    default="auto",
    format_func={"auto": "Auto", "cuda": "CUDA", "cpu": "CPU"}.get,
    key="esmfold2_validation_device",
)
if st.button("Run ESMFold2 validation", type="primary"):
    try:
        with st.spinner("Running ESMFold2 complex validation..."):
            run_dir = refolding_workflow.run_esmfold2_complex_validation(
                source_run_dir=Path(str(source["run_dir"])),
                candidates_jsonl=Path(str(source["candidates_jsonl"])),
                max_candidates=int(esm_max_candidates),
                num_loops=int(esm_loops),
                num_sampling_steps=int(esm_sampling_steps),
                seed=int(esm_seed),
                device=str(esm_device or "auto"),
                contact_cutoff=float(esm_contact_cutoff),
                use_initial_guess=str(esm_mode or "sequence") == "initial_guess",
            )
        result = read_json(run_dir / "result.json")
        if result.get("success") is True:
            st.success("ESMFold2 validation finished.")
        else:
            st.error("ESMFold2 validation failed. Open the job logs for details.")
        show_pipeline_links(run_dir, [run_dir])
    except Exception as exc:
        st.error(str(exc))

st.subheader("Run Full Refolding / Validation")
st.caption(
    "This always runs the validation pack in order: monomer refolding, complex refolding, then analysis."
)

st.markdown("**1. Monomer Refolding**")
monomer_tool = st.segmented_control(
    "Monomer tool",
    ["af2_monomer", "boltz2_monomer", "esmfold"],
    selection_mode="single",
    default="boltz2_monomer",
    format_func={
        "af2_monomer": "AF2 monomer",
        "boltz2_monomer": "Boltz2 monomer",
        "esmfold": "ESMFold",
    }.get,
)
min_plddt = st.number_input("Minimum monomer pLDDT", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
if monomer_tool in {"esmfold", "boltz2_monomer"}:
    st.info("This monomer refolding backend is Docker-backed.")
else:
    st.info("AF2 monomer currently writes the normalized monomer-refolding job contract. The ColabDesign evaluator is not wired yet.")

st.markdown("**2. Complex Refolding**")
complex_tool = st.segmented_control(
    "Complex tool",
    ["af2_initial_guess", "boltz2_initial_guess", "esmfold2_complex_validation", "esmfold2_initial_guess_validation"],
    selection_mode="single",
    default="af2_initial_guess",
    format_func={
        "af2_initial_guess": "AF2 initial guess",
        "boltz2_initial_guess": "Boltz2 initial guess",
        "esmfold2_complex_validation": "ESMFold2 validation",
        "esmfold2_initial_guess_validation": "ESMFold2 initial guess",
    }.get,
)
selected_complex_tool = str(complex_tool or "af2_initial_guess")
if selected_complex_tool == "boltz2_initial_guess":
    template_mode = st.segmented_control(
        "Boltz2 template mode",
        ["target_template", "no_template"],
        selection_mode="single",
        default="target_template",
        format_func={"target_template": "Target template", "no_template": "No template"}.get,
    )
    multimer = True
elif selected_complex_tool in {"esmfold2_complex_validation", "esmfold2_initial_guess_validation"}:
    template_mode = "target_template"
    multimer = True
    if selected_complex_tool == "esmfold2_initial_guess_validation":
        st.info("ESMFold2 uses the target structure as a target distogram prior, matching the target-template idea of AF2 initial guess.")
    else:
        st.info("ESMFold2 refolds each existing candidate sequence against its target; no AF2/Boltz template is used.")
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
        key="full_validation_af2_model",
    )
    _label, template_mode, multimer, _recycles = af2_options[af2_choice]
st.info("Complex refolding is started from the monomer-refolding result, so failed monomer candidates do not continue.")

st.markdown("**3. Analysis**")
analysis_cols = st.columns(4)
with analysis_cols[0]:
    keep_top_n = st.number_input("Keep top", min_value=1, max_value=10000, value=100, step=10, key="full_validation_keep_top")
with analysis_cols[1]:
    min_binder_plddt = st.number_input("Min binder pLDDT", min_value=0.0, max_value=100.0, value=70.0, step=1.0, key="full_validation_plddt")
with analysis_cols[2]:
    max_ipae = st.number_input("Max iPAE", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key="full_validation_ipae")
with analysis_cols[3]:
    max_binder_rmsd = st.number_input("Max binder RMSD", min_value=0.0, max_value=100.0, value=5.0, step=0.5, key="full_validation_rmsd")

steps = [
    {
        "module": "monomer_refolding",
        "tool": str(monomer_tool or "boltz2_monomer"),
        "params": {"min_plddt": float(min_plddt)},
    },
    {
        "module": "complex_refolding",
        "tool": selected_complex_tool,
        "params": {
            "require_monomer_success": True,
            "template_mode": str(template_mode or "target_template"),
            "multimer": bool(multimer),
            "num_recycles": 3,
            "max_candidates": int(esm_max_candidates),
            "num_sampling_steps": int(esm_sampling_steps),
            "seed": int(esm_seed),
            "device": str(esm_device or "auto"),
            "contact_cutoff": float(esm_contact_cutoff),
        },
    },
    {
        "module": "analysis",
        "tool": "ranking",
        "params": {
            "keep_top_n": int(keep_top_n),
            "thresholds": {
                "min_binder_plddt": float(min_binder_plddt),
                "min_confidence": 0.0,
                "min_iptm": 0.0,
                "min_ipsae": 0.0,
                "max_ipae": float(max_ipae),
                "max_ipde": 20.0,
                "max_binder_rmsd": float(max_binder_rmsd),
            },
        },
    },
]

if st.button("Run full validation", type="primary"):
    try:
        with st.spinner("Running monomer refolding, complex refolding, and analysis..."):
            campaign_run, child_runs = run_steps_after_source(
                f"Validation from {source['job_code']} {source['tool']}",
                source_from_run_dir(Path(str(source["run_dir"]))),
                steps,
            )
        if child_runs:
            final_result = read_json(child_runs[-1] / "result.json")
            if final_result.get("success") is not True:
                st.error("Validation stopped on a failed step. Open the child job for logs.")
            else:
                st.success("Full validation workflow finished.")
        show_pipeline_links(campaign_run, child_runs)
    except Exception as exc:
        st.error(str(exc))
