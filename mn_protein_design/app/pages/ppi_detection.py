from __future__ import annotations

from pathlib import Path

import streamlit as st

from mn_protein_design.app.components.molstar_viewer import StructureVisualization, molstar_custom_component
from mn_protein_design.core.structures import pdb_summary
from mn_protein_design.workflows.detection import (
    chains_for_pdb,
    completed_target_jobs,
    detection_jobs_for_target,
    run_masif_seed,
    run_scannet,
    run_surf2spot,
    target_label,
)


st.title("PPI / Hotspot Detection")
st.caption("Run interface and hotspot detection from a completed prepared-target job.")

targets = completed_target_jobs()
if not targets:
    st.info("Prepare a target first, then return here to run ScanNet, MaSIF, or Surf2Spot.")
    st.stop()

target = st.selectbox("Prepared target", options=targets, format_func=target_label)
target_pdb = Path(target["target_pdb"])
if not target_pdb.exists():
    st.error(f"Prepared target file is missing: {target_pdb}")
    st.stop()
chains = chains_for_pdb(target_pdb)
if not chains:
    st.error("The selected prepared target has no detectable protein chains.")
    st.stop()
summary = pdb_summary(target_pdb.read_text(errors="ignore"))

left, right = st.columns([0.9, 1.2], gap="large")

with left:
    st.subheader("Input")
    st.write(f"Target: `{target['target_name']}`")
    st.write(f"Source job: `{target['job_code']}`")
    st.write(f"Prepared PDB: `{target_pdb.name}`")
    st.caption(f"Chains: {', '.join(chains) if chains else 'unknown'}")

    existing = detection_jobs_for_target(target_pdb)
    if existing:
        st.subheader("Existing Results")
        for job in existing:
            st.markdown(
                f"- [{job['tool']} {job['job_code']}]"
                f"(/results?task_group={job['task_group']}&run_id={job['run_id']}) "
                f"`{job['status']}`"
            )

    st.subheader("ScanNet PPI")
    scannet_chains = st.multiselect("Chains for ScanNet", options=chains, default=chains, key="scannet_chains")
    scannet_mode = st.selectbox("Prediction mode", ["interface", "epitope", "idp"], key="scannet_mode")
    use_msa = st.checkbox("Use MSA", value=False, help="Slower and requires HHblits/database paths inside the ScanNet container.")
    if st.button("Run ScanNet", type="primary", disabled=not scannet_chains):
        with st.spinner("Running ScanNet PPI prediction..."):
            run_dir = run_scannet(target_pdb, scannet_chains, mode=scannet_mode, use_msa=use_msa)
        st.success(f"ScanNet job finished: {run_dir.name}")
        st.markdown(f"[Open result](/results?task_group=detection&run_id={run_dir.name})")

    st.subheader("Surf2Spot Hotspots")
    st.caption("Runs HS-preprocess, HS-craft, HS-predict, and HS-draw on the prepared structure.")
    if st.button("Run Surf2Spot HS", disabled=not target_pdb.exists()):
        with st.spinner("Running Surf2Spot hotspot prediction..."):
            run_dir = run_surf2spot(target_pdb)
        st.success(f"Surf2Spot job finished: {run_dir.name}")
        st.markdown(f"[Open result](/results?task_group=hotspot-detection&run_id={run_dir.name})")

    st.subheader("MaSIF Target Surface")
    st.caption("MaSIF-seed target/site prediction currently runs one prepared chain at a time.")
    masif_chain = st.selectbox("Target chain for MaSIF", options=chains, key="masif_chain")
    if st.button("Run MaSIF target surface", disabled=not masif_chain):
        with st.spinner("Running MaSIF-seed target surface/site prediction..."):
            run_dir = run_masif_seed(target_pdb, masif_chain)
        st.success(f"MaSIF job finished: {run_dir.name}")
        st.markdown(f"[Open result](/results?task_group=detection&run_id={run_dir.name})")

with right:
    st.subheader("Prepared Structure")
    cols = st.columns(4)
    cols[0].metric("Chains", len(summary["chains"]))
    cols[1].metric("Residues", sum(chain["residue_count"] for chain in summary["chains"]))
    cols[2].metric("Atoms", summary["atom_count"])
    cols[3].metric("Waters", summary["water_count"])
    molstar_custom_component(
        structures=[
            StructureVisualization(
                pdb=target_pdb.read_text(errors="ignore"),
                color="chain-id",
                representation_type="cartoon+ball-and-stick",
            )
        ],
        key=f"detection_target_{target['run_id']}",
        height=680,
        show_controls=True,
        download_filename=f"{target['target_name']}_prepared",
    )
