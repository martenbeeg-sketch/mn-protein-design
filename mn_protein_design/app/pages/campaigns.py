from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from mn_protein_design.app.pages.common import result_link
from mn_protein_design.core.candidates import STAGE_GENERATION_BACKBONE
from mn_protein_design.core.pipeline import read_pipeline, step_summary_rows
from mn_protein_design.workflows.campaigns import create_campaign, list_campaigns, run_next_step
from mn_protein_design.workflows.modules import candidate_sources, load_source_candidates


def _source_label(source: dict) -> str:
    return f"{source['job_code']} | {source['tool']} | {source['candidate_count']} candidates | {source['stage_counts']}"


def _default_sequence_tool(source: dict) -> str:
    return "foundry_mpnn" if str(source.get("tool") or "") == "rfdiffusion3_foundry" else "ligandmpnn"


def _sequence_params(tool: str) -> dict:
    if tool == "foundry_mpnn":
        cols = st.columns(3)
        with cols[0]:
            number_of_batches = st.number_input("Number of batches", min_value=1, max_value=1000, value=1, step=1)
        with cols[1]:
            batch_size = st.number_input("Batch size", min_value=1, max_value=1000, value=10, step=1)
        with cols[2]:
            checkpoint_path = st.text_input("Checkpoint", value="/weights/ligandmpnn_v_32_010_25.pt")
        return {
            "number_of_batches": int(number_of_batches),
            "batch_size": int(batch_size),
            "checkpoint_path": checkpoint_path,
        }

    model_type = st.radio(
        "MPNN model",
        ["protein_mpnn", "ligand_mpnn", "soluble_mpnn"],
        format_func={
            "protein_mpnn": "ProteinMPNN",
            "ligand_mpnn": "LigandMPNN",
            "soluble_mpnn": "SolubleMPNN",
        }.get,
        horizontal=True,
    )
    cols = st.columns(4)
    with cols[0]:
        num_seq_per_target = st.number_input("Sequences per candidate", min_value=1, max_value=1000, value=1, step=1)
    with cols[1]:
        sampling_temp = st.number_input("Sampling temperature", min_value=0.0, max_value=5.0, value=0.0001, step=0.0001, format="%g")
    with cols[2]:
        omit_aas = st.text_input("Omit amino acids", value="CX")
    with cols[3]:
        seed = st.number_input("Seed", min_value=0, max_value=999999999, value=0, step=1)
    return {
        "model_type": model_type,
        "design_chains": "",
        "num_seq_per_target": int(num_seq_per_target),
        "sampling_temp": float(sampling_temp),
        "omit_aas": omit_aas,
        "seed": int(seed) if int(seed) else None,
    }


def _monomer_params() -> tuple[str, dict]:
    tool = st.segmented_control(
        "Monomer refolding tool",
        ["af2_monomer", "boltz2_monomer", "esmfold"],
        selection_mode="single",
        default="af2_monomer",
        format_func={
            "af2_monomer": "AF2 monomer",
            "boltz2_monomer": "Boltz2 monomer",
            "esmfold": "ESMFold",
        }.get,
    )
    min_plddt = st.number_input("Minimum monomer pLDDT", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
    return str(tool or "af2_monomer"), {"min_plddt": float(min_plddt)}


def _complex_params() -> tuple[str, dict]:
    tool = st.segmented_control(
        "Complex refolding tool",
        ["af2_initial_guess", "boltz2_initial_guess"],
        selection_mode="single",
        default="af2_initial_guess",
        format_func={
            "af2_initial_guess": "AF2 initial guess",
            "boltz2_initial_guess": "Boltz2 initial guess",
        }.get,
    )
    require_monomer = st.checkbox("Only run after monomer refolding step succeeds", value=True)
    selected_tool = str(tool or "af2_initial_guess")
    if selected_tool == "boltz2_initial_guess":
        template_mode = st.segmented_control(
            "Boltz2 template mode",
            ["target_template", "no_template"],
            selection_mode="single",
            default="target_template",
            format_func={
                "target_template": "Target template",
                "no_template": "No template",
            }.get,
        )
        return selected_tool, {
            "require_monomer_success": bool(require_monomer),
            "template_mode": str(template_mode or "target_template"),
            "multimer": True,
            "num_recycles": 3,
        }
    else:
        af2_options = {
            "af2_model_1_ptm_tt_3rec": ("AF2 monomer model, target template", "target_template", False, 3),
            "af2_model_1_ptm_tbt_3rec": ("AF2 monomer model, target + binder templates", "target_binder_template", False, 3),
            "af2_model_1_ptm_ct_3rec": ("AF2 monomer model, complex template", "complex_template", False, 3),
            "af2_model_1_multimer_tt_3rec": ("AF2 multimer, target template", "target_template", True, 3),
            "af2_model_1_multimer_tbt_3rec": ("AF2 multimer, target + binder templates", "target_binder_template", True, 3),
            "af2_model_1_multimer_ct_3rec": ("AF2 multimer, complex template", "complex_template", True, 3),
        }
        af2_choice = st.selectbox(
            "AF2 complex refolding model",
            list(af2_options),
            format_func=lambda key: af2_options[key][0],
            index=list(af2_options).index("af2_model_1_multimer_tt_3rec"),
        )
        _, template_mode, multimer, num_recycles = af2_options[af2_choice]
    return selected_tool, {
        "require_monomer_success": bool(require_monomer),
        "template_mode": str(template_mode or "target_template"),
        "multimer": bool(multimer),
        "num_recycles": int(num_recycles),
    }


def _analysis_params() -> tuple[str, dict]:
    tool = st.segmented_control(
        "Analysis mode",
        ["ranking", "filters", "reports"],
        selection_mode="single",
        default="ranking",
        format_func={
            "ranking": "Ranking",
            "filters": "Filters",
            "reports": "Reports",
        }.get,
    )
    keep_top_n = st.number_input("Keep top candidates", min_value=1, max_value=10000, value=100, step=10)
    cols = st.columns(3)
    with cols[0]:
        min_binder_plddt = st.number_input("Min binder pLDDT", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        max_ipae = st.number_input("Max iPAE", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    with cols[1]:
        min_iptm = st.number_input("Min ipTM", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        max_ipde = st.number_input("Max iPDE", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
    with cols[2]:
        min_ipsae = st.number_input("Min ipSAE", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        min_confidence = st.number_input("Min confidence", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    max_binder_rmsd = st.number_input("Max binder RMSD", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
    return str(tool or "ranking"), {
        "keep_top_n": int(keep_top_n),
        "thresholds": {
            "min_binder_plddt": float(min_binder_plddt),
            "min_confidence": float(min_confidence),
            "min_iptm": float(min_iptm),
            "min_ipsae": float(min_ipsae),
            "max_ipae": float(max_ipae),
            "max_ipde": float(max_ipde),
            "max_binder_rmsd": float(max_binder_rmsd),
        },
    }


st.title("Campaigns")
st.caption("String normalized design modules together. Each module writes candidates that the next module can consume.")

new_tab, existing_tab = st.tabs(["New campaign", "Existing campaigns"])

with new_tab:
    sources = [
        source
        for source in candidate_sources("design")
        if STAGE_GENERATION_BACKBONE in source["stage_counts"]
    ]
    if not sources:
        st.info("No backbone candidate sets are available yet. Run a generation tool first.")
    else:
        campaign_name = st.text_input("Campaign name", value="")
        source_index = st.selectbox("Starting candidate set", range(len(sources)), format_func=lambda index: _source_label(sources[index]))
        source = sources[source_index]

        candidates = [
            row
            for row in load_source_candidates(source)
            if row.get("stage") == STAGE_GENERATION_BACKBONE
        ]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "candidate_id": row.get("candidate_id"),
                        "source_tool": row.get("source_tool"),
                        "structure": row.get("complex_pdb") or row.get("binder_pdb"),
                        "hotspots": ",".join(row.get("hotspots") or []),
                        "contig": row.get("contig"),
                    }
                    for row in candidates
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Pipeline")
        st.caption("Default path: sequence design, monomer refolding, complex refolding, then analysis.")
        sequence_enabled = st.checkbox("Add sequence design step", value=True)
        monomer_enabled = st.checkbox("Add monomer refolding step", value=True)
        complex_enabled = st.checkbox("Add complex refolding step", value=True)
        analysis_enabled = st.checkbox("Add analysis step", value=True)
        steps: list[dict] = []
        if sequence_enabled:
            st.markdown("**Sequence Design**")
            default_tool = _default_sequence_tool(source)
            tool_options = ["ligandmpnn", "foundry_mpnn"]
            sequence_tool = st.segmented_control(
                "Sequence tool",
                tool_options,
                selection_mode="single",
                default=default_tool,
                format_func={
                    "ligandmpnn": "LigandMPNN / ProteinMPNN",
                    "foundry_mpnn": "Foundry MPNN",
                }.get,
            )
            params = _sequence_params(str(sequence_tool or default_tool))
            steps.append({"module": "sequence_design", "tool": str(sequence_tool or default_tool), "params": params})
        if monomer_enabled:
            st.markdown("**Monomer Refolding**")
            tool, params = _monomer_params()
            steps.append({"module": "monomer_refolding", "tool": tool, "params": params})
        if complex_enabled:
            st.markdown("**Complex Refolding**")
            tool, params = _complex_params()
            steps.append({"module": "complex_refolding", "tool": tool, "params": params})
        if analysis_enabled:
            st.markdown("**Analysis**")
            tool, params = _analysis_params()
            steps.append({"module": "analysis", "tool": tool, "params": params})

        st.info(
            "ESMFold and Boltz2 monomer refolding are Docker-backed. AF2 monomer, complex refolding, "
            "AF2 initial guess and Boltz2 complex refolding are Docker-backed. Analysis still creates "
            "ranked tables and metric summaries from normalized outputs."
        )

        if st.button("Create campaign", type="primary", disabled=not steps):
            try:
                run_dir = create_campaign(campaign_name, source, steps)
                st.success("Campaign created.")
                st.link_button("Open campaign result", result_link("campaign", run_dir.name))
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

with existing_tab:
    campaigns = list_campaigns()
    if not campaigns:
        st.info("No campaigns yet.")
    else:
        campaign_labels = [
            f"{row['job_code']} | {row['name']} | {row['status']} | {row['updated_at']}"
            for row in campaigns
        ]
        selected_campaign = st.selectbox(
            "Campaign",
            range(len(campaigns)),
            format_func=lambda index: campaign_labels[index],
        )
        campaign = campaigns[selected_campaign]
        pipeline = campaign.get("pipeline") or {}
        st.link_button("Open campaign result", result_link("campaign", campaign["run_id"]))

        st.subheader("Steps")
        rows = step_summary_rows(pipeline)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        for step in pipeline.get("steps") or []:
            job_ref = step.get("job_ref") or {}
            if job_ref.get("task_group") and job_ref.get("run_id"):
                st.link_button(
                    f"Open {step.get('module')} result {job_ref.get('job_code') or job_ref.get('run_id')}",
                    result_link(str(job_ref["task_group"]), str(job_ref["run_id"])),
                )

        done = all(step.get("status") == "completed" for step in pipeline.get("steps") or [])
        if st.button("Run next pending step", type="primary", disabled=done):
            try:
                with st.spinner("Running next campaign step..."):
                    child_run_dir = run_next_step(Path(campaign["run_dir"]))
                updated_pipeline = read_pipeline(Path(campaign["run_dir"]))
                if child_run_dir is None:
                    st.success("Campaign is already complete.")
                elif updated_pipeline.get("status") == "failed":
                    st.error("Step job failed. Open the child job result for logs.")
                    st.link_button("Open failed child job result", result_link("design", child_run_dir.name))
                else:
                    st.success("Step finished.")
                    st.link_button("Open child job result", result_link("design", child_run_dir.name))
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
