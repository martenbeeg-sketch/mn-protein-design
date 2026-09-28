from __future__ import annotations

from pathlib import Path
import hashlib

import pandas as pd
import streamlit as st

from mn_protein_design.app.components.molstar_viewer import StructureVisualization, molstar_custom_component
from mn_protein_design.app.pages.common import (
    gpu_run_panel,
    refresh_results_button,
    result_link,
    selected_dataframe_rows,
    show_delete_jobs_dialog,
    show_pipeline_links,
)
from mn_protein_design.core.jobs import ACTIVE_STATUSES, collect_jobs, read_json
from mn_protein_design.core.local_worker import spawn_worker_for_run
from mn_protein_design.core.structures import pdb_summary
from mn_protein_design.workflows.detection import (
    DETECTION_GROUP,
    HOTSPOT_GROUP,
    chains_for_pdb,
    detection_jobs_for_target,
    enqueue_masif_seed,
    enqueue_pesto,
    enqueue_scannet,
    enqueue_surf2spot,
    ppi_target_jobs,
    target_label,
)


def _prepared_target_kind(target: dict, target_pdb: Path) -> str:
    explicit = str(target.get("prepared_kind") or "").strip()
    if explicit:
        return explicit
    task_group = str(target.get("task_group") or "")
    if task_group == "target-crop":
        return "cropped"
    name = target_pdb.name.lower()
    if "cropped" in name or "crop" in name:
        return "cropped"
    if "trimmed" in name:
        return "trimmed"
    if "clean" in name:
        return "cleaned"
    return task_group.replace("target-", "") or "prepared"


def _target_info_for_pdb(target_pdb: Path) -> dict[str, str]:
    target_path = target_pdb.expanduser()
    target_resolved = str(target_path.resolve()) if target_path.exists() else str(target_path)
    for target in targets:
        candidate_path = Path(str(target.get("target_pdb") or "")).expanduser()
        candidate_resolved = str(candidate_path.resolve()) if candidate_path.exists() else str(candidate_path)
        if candidate_path == target_path or candidate_resolved == target_resolved:
            kind = _prepared_target_kind(target, candidate_path)
            target_name = str(target.get("target_name") or target.get("job_code") or target_pdb.name)
            job_code = str(target.get("job_code") or "")
            task_group = str(target.get("task_group") or "")
            source_category = str(target.get("source_category") or kind or "")
            source_label = str(target.get("source_label") or target.get("tool") or "")
            return {
                "target": target_name,
                "source_job": job_code,
                "source_type": task_group,
                "source_category": source_category,
                "source_label": source_label,
                "prepared_kind": kind,
                "prepared_pdb": target_pdb.name,
                "target_filter": f"{target_name} | {source_category} | {job_code} | {kind} | {target_pdb.name}",
                "target_pdb": str(candidate_path),
            }
    return {
        "target": target_pdb.name or "unknown",
        "source_job": "",
        "source_type": "",
        "source_category": "unknown",
        "source_label": "",
        "prepared_kind": "",
        "prepared_pdb": target_pdb.name,
        "target_filter": f"unknown | {target_pdb.name}",
        "target_pdb": str(target_pdb),
    }


def _detection_results_table() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in [DETECTION_GROUP, HOTSPOT_GROUP]:
        for job in collect_jobs(group):
            run_dir = Path(str(job.get("run_dir") or ""))
            input_json = read_json(run_dir / "input.json")
            result = read_json(run_dir / "result.json")
            inputs = input_json.get("inputs") if isinstance(input_json.get("inputs"), dict) else {}
            params = input_json.get("params") if isinstance(input_json.get("params"), dict) else {}
            metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
            tool = str(job.get("tool") or input_json.get("tool") or "")
            target_pdb = Path(str(inputs.get("target_pdb") or ""))
            target_info = _target_info_for_pdb(target_pdb)
            chains = inputs.get("chains") or inputs.get("chain_id") or ""
            if isinstance(chains, list):
                chains = ",".join(str(chain) for chain in chains)
            rows.append(
                {
                    "result": result_link(group, str(job.get("run_id")), "Open"),
                    "tool": tool,
                    "target": target_info["target"],
                    "source_job": target_info["source_job"],
                    "source_type": target_info["source_type"],
                    "source_category": target_info["source_category"],
                    "source_label": target_info["source_label"],
                    "prepared_kind": target_info["prepared_kind"],
                    "prepared_pdb": target_info["prepared_pdb"],
                    "target_filter": target_info["target_filter"],
                    "target_pdb": target_info["target_pdb"],
                    "chains": chains,
                    "mode": params.get("mode"),
                    "status": job.get("status"),
                    "artifacts": metrics.get("artifact_count"),
                    "created_at": job.get("created_at"),
                    "job_code": job.get("job_code"),
                    "run_id": job.get("run_id"),
                    "task_group": group,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("created_at", ascending=False, na_position="last")


st.title("PPI / Hotspot Detection")
st.caption("Run interface and hotspot detection from a completed prepared-target job.")


@st.cache_data(ttl=60, show_spinner=False)
def _load_detection_targets() -> list[dict]:
    return ppi_target_jobs()


targets = _load_detection_targets()
if not targets:
    st.info("Prepare a target first, then return here to run ScanNet, PeSTo, MaSIF, or Surf2Spot.")
    st.stop()


def _category_label(category: str) -> str:
    labels = {
        "trimmed": "Trimmed prepared targets",
        "cleaned": "Cleaned prepared targets",
        "mutated": "Mutated targets",
        "cropped": "Cropped targets",
        "benchmark": "Benchmark targets",
        "imported": "Imported dataset targets",
        "unknown": "Unknown source",
    }
    return labels.get(category, category.replace("_", " ").title())


def _source_categories(rows: list[dict]) -> list[str]:
    preferred = ["imported", "trimmed", "cleaned", "mutated", "cropped", "benchmark", "unknown"]
    present = {str(row.get("source_category") or row.get("prepared_kind") or "unknown") for row in rows}
    present.add("mutated")
    ordered = [category for category in preferred if category in present]
    ordered.extend(sorted(category for category in present if category not in preferred))
    return ordered


def _target_state_token(target_path: Path, chains: list[str]) -> str:
    payload = f"{target_path}|{','.join(chains)}"
    return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()[:10]

dataset_tab, methods_tab, run_tab, results_tab = st.tabs(["Dataset", "Methods", "Run", "Results"])
target = targets[0]
target_pdb = Path(str(target.get("target_pdb") or ""))
chains: list[str] = []
summary: dict[str, object] = {}
target_ready = False

with dataset_tab:
    categories = _source_categories(targets)
    previous_category_filter = st.session_state.get("ppi_detection_dataset_source_filter")
    if previous_category_filter and any(category not in categories for category in previous_category_filter):
        st.session_state.pop("ppi_detection_dataset_source_filter", None)
    selected_categories = st.multiselect(
        "Target source filter",
        options=categories,
        default=categories,
        format_func=_category_label,
        key="ppi_detection_dataset_source_filter",
    )
    if not selected_categories:
        selected_categories = categories
        st.info("No source filter selected; showing all target sources.")
    visible_targets = [
        target_row
        for target_row in targets
        if str(target_row.get("source_category") or target_row.get("prepared_kind") or "unknown") in selected_categories
    ]
    if not visible_targets:
        visible_targets = targets
        st.warning("No targets matched the selected source filter, so all targets are shown.")
    selected_target_path = str(st.session_state.get("ppi_detection_selected_target_pdb") or "")
    default_visible_index = next(
        (
            index
            for index, target_row in enumerate(visible_targets)
            if str(target_row.get("target_pdb") or "") == selected_target_path
        ),
        0,
    )
    target_table_rows = []
    for index, target_row in enumerate(visible_targets):
        chains_text = ",".join(str(chain) for chain in target_row.get("chains") or [])
        target_table_rows.append(
            {
                "target": target_row.get("target_name"),
                "source_category": _category_label(str(target_row.get("source_category") or target_row.get("prepared_kind") or "unknown")),
                "source": target_row.get("source_label") or target_row.get("tool") or target_row.get("task_group"),
                "source_job": target_row.get("job_code"),
                "chains": chains_text or "unknown",
                "records": target_row.get("records"),
                "pdb": Path(str(target_row.get("target_pdb") or "")).name,
                "_target_index": index,
            }
        )
    st.caption("Select one target row for preview and downstream PPI/hotspot detection.")
    target_table_df = pd.DataFrame(target_table_rows)
    event = st.dataframe(
        target_table_df.drop(columns=["_target_index"]),
        width="stretch",
        hide_index=True,
        key="ppi_detection_target_table",
        on_select="rerun",
        selection_mode="single-row",
    )
    selected_indices = selected_dataframe_rows(event, "ppi_detection_target_table")
    selected_visible_index = default_visible_index
    if selected_indices and 0 <= selected_indices[0] < len(target_table_df):
        selected_visible_index = int(target_table_df.iloc[selected_indices[0]]["_target_index"])
    target = visible_targets[selected_visible_index]
    st.session_state["ppi_detection_selected_target_pdb"] = str(target.get("target_pdb") or "")
    target_pdb = Path(str(target.get("target_pdb") or ""))
    if not target_pdb.exists():
        st.error(f"Target file is missing: {target_pdb}")
    else:
        chains = chains_for_pdb(target_pdb)
    if target_pdb.exists() and not chains:
        st.error("The selected prepared target has no detectable protein chains.")
    target_ready = target_pdb.exists() and bool(chains)
    if target_ready:
        summary = pdb_summary(target_pdb.read_text(errors="ignore"))
    target_token = _target_state_token(target_pdb, chains)
    if st.session_state.get("ppi_detection_active_target_token") != target_token:
        st.session_state["ppi_detection_active_target_token"] = target_token
        st.session_state["ppi_detection_scannet_chains"] = list(chains)
        st.session_state["ppi_detection_pesto_chains"] = list(chains)
        st.session_state["ppi_detection_masif_chain"] = chains[0] if chains else ""

    st.subheader("Input")
    st.write(f"Target: `{target['target_name']}`")
    st.write(f"Source job: `{target['job_code']}`")
    st.write(f"Source type: `{target.get('source_category') or target.get('prepared_kind') or 'target'}`")
    st.write(f"Source: `{target.get('source_label') or target.get('tool') or target.get('task_group') or 'unknown'}`")
    st.write(f"PDB: `{target_pdb.name}`")
    st.caption(f"Chains: {', '.join(chains) if chains else 'unknown'}")

    if target_ready:
        cols = st.columns(4)
        summary_chains = summary.get("chains") if isinstance(summary.get("chains"), list) else []
        cols[0].metric("Chains", len(summary_chains))
        cols[1].metric("Residues", sum(chain["residue_count"] for chain in summary_chains))
        cols[2].metric("Atoms", summary.get("atom_count", 0))
        cols[3].metric("Waters", summary.get("water_count", 0))
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

        existing = detection_jobs_for_target(target_pdb)
        if existing:
            st.subheader("Existing Results")
            for job in existing:
                st.markdown(
                    f"- [{job['tool']} {job['job_code']}]"
                    f"(/results?task_group={job['task_group']}&run_id={job['run_id']}) "
                    f"`{job['status']}`"
                )

with methods_tab:
    st.subheader("Methods")
    st.caption("Enable one or more detection tools. Each tool keeps its own parameters grouped below.")

    tool_keys = ["ppi_enable_scannet", "ppi_enable_pesto", "ppi_enable_surf2spot", "ppi_enable_masif"]
    defaults = {
        "ppi_enable_scannet": True,
        "ppi_enable_pesto": True,
        "ppi_enable_surf2spot": True,
        "ppi_enable_masif": True,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    bulk_cols = st.columns([1, 1, 5])
    if bulk_cols[0].button("Select all tools", key="ppi_select_all_tools"):
        for key in tool_keys:
            st.session_state[key] = True
        st.rerun()
    if bulk_cols[1].button("Deselect all tools", key="ppi_deselect_all_tools"):
        for key in tool_keys:
            st.session_state[key] = False
        st.rerun()

    with st.expander("ScanNet PPI Settings", expanded=bool(st.session_state.get("ppi_enable_scannet"))):
        enable_scannet = st.checkbox(
            "Run ScanNet PPI",
            value=True,
            key="ppi_enable_scannet",
            help="Predicts protein-protein interface, epitope, or disordered interaction residues on selected chains.",
        )
        scannet_cols = st.columns(3)
        scannet_chain_key = f"scannet_chains_{st.session_state.get('ppi_detection_active_target_token', 'none')}"
        if scannet_chain_key not in st.session_state:
            st.session_state[scannet_chain_key] = [
                chain for chain in st.session_state.get("ppi_detection_scannet_chains", chains) if chain in chains
            ] or list(chains)
        scannet_chains = scannet_cols[0].multiselect(
            "Chains",
            options=chains,
            key=scannet_chain_key,
            disabled=not enable_scannet or not target_ready,
        )
        st.session_state["ppi_detection_scannet_chains"] = list(scannet_chains)
        scannet_mode = scannet_cols[1].selectbox(
            "Prediction mode",
            ["interface", "epitope", "idp"],
            key="scannet_mode",
            disabled=not enable_scannet or not target_ready,
        )
        use_msa = scannet_cols[2].checkbox(
            "Use MSA",
            value=False,
            key="scannet_use_msa",
            disabled=not enable_scannet or not target_ready,
            help="Slower and requires HHblits/database paths inside the ScanNet container.",
        )
        st.caption("Outputs residue-level PPI/interface predictions for the selected chain set.")
        if not enable_scannet:
            scannet_chains = []

    with st.expander("PeSTo PPI Settings", expanded=bool(st.session_state.get("ppi_enable_pesto"))):
        enable_pesto = st.checkbox(
            "Run PeSTo PPI",
            value=True,
            key="ppi_enable_pesto",
            help="Predicts residue-level protein-protein interaction interfaces with the PeSTo i_v4_1 model.",
        )
        pesto_cols = st.columns(3)
        pesto_chain_key = f"pesto_chains_{st.session_state.get('ppi_detection_active_target_token', 'none')}"
        if pesto_chain_key not in st.session_state:
            st.session_state[pesto_chain_key] = [
                chain for chain in st.session_state.get("ppi_detection_pesto_chains", chains) if chain in chains
            ] or list(chains)
        pesto_chains = pesto_cols[0].multiselect(
            "Chains",
            options=chains,
            key=pesto_chain_key,
            disabled=not enable_pesto or not target_ready,
        )
        st.session_state["ppi_detection_pesto_chains"] = list(pesto_chains)
        pesto_cols[1].text_input("Interface class", value="protein-protein", disabled=True)
        pesto_cols[2].text_input("Model", value="i_v4_1", disabled=True)
        st.caption("Outputs original-numbered residue probabilities and a score-annotated PDB for Mol* visualization.")
        if not enable_pesto:
            pesto_chains = []

    with st.expander("Surf2Spot Hotspot Detection Settings", expanded=bool(st.session_state.get("ppi_enable_surf2spot"))):
        enable_surf2spot = st.checkbox(
            "Run Surf2Spot hotspot detection",
            value=True,
            key="ppi_enable_surf2spot",
            help="Runs the Surf2Spot HS pipeline to identify binding hotspots and draw PyMOL-ready outputs.",
        )
        surf_cols = st.columns(3)
        surf_cols[0].text_input("Mode", value="HS", disabled=True)
        surf_cols[1].text_input("Chains", value="all prepared chains", disabled=True)
        surf_cols[2].text_input("Output", value="hotspot table, surface mesh, PyMOL session", disabled=True)
        st.caption("Runs HS-preprocess, HS-craft, HS-predict, and HS-draw on the prepared structure.")

    with st.expander("MaSIF Target Surface Settings", expanded=bool(st.session_state.get("ppi_enable_masif"))):
        enable_masif = st.checkbox(
            "Run MaSIF target surface",
            value=True,
            key="ppi_enable_masif",
            help="Runs MaSIF-seed target/site prediction for one prepared target chain.",
        )
        masif_cols = st.columns(3)
        masif_chain_key = f"masif_chain_{st.session_state.get('ppi_detection_active_target_token', 'none')}"
        if masif_chain_key not in st.session_state:
            previous_masif_chain = str(st.session_state.get("ppi_detection_masif_chain") or "")
            st.session_state[masif_chain_key] = previous_masif_chain if previous_masif_chain in chains else (chains[0] if chains else "")
        masif_chain = masif_cols[0].selectbox(
            "Target chain",
            options=chains,
            key=masif_chain_key,
            disabled=not enable_masif or not target_ready,
        )
        st.session_state["ppi_detection_masif_chain"] = masif_chain
        masif_cols[1].text_input("Mode", value="target_site_surface", disabled=True)
        masif_cols[2].text_input("Output", value="surface mesh and site scores", disabled=True)
        st.caption("MaSIF-seed target/site prediction currently runs one prepared chain at a time.")
        if not enable_masif:
            masif_chain = ""

with run_tab:
    st.subheader("Run")
    st.subheader("Input")
    st.write(f"Target: `{target['target_name']}`")
    st.write(f"Source job: `{target['job_code']}`")
    st.write(f"Source type: `{target.get('source_category') or target.get('prepared_kind') or 'target'}`")
    st.write(f"PDB: `{target_pdb.name}`")
    st.caption(f"Chains: {', '.join(chains) if chains else 'unknown'}")
    with st.expander("Compute", expanded=True):
        ppi_gpu_device = gpu_run_panel(key="ppi_detection_run", default="0")
    run_rows = []
    if enable_scannet:
        run_rows.append({"method": "ScanNet", "chains": ",".join(scannet_chains), "mode": scannet_mode})
    if enable_pesto:
        run_rows.append({"method": "PeSTo", "chains": ",".join(pesto_chains), "mode": "protein_interface"})
    if enable_surf2spot:
        run_rows.append({"method": "Surf2Spot", "chains": ",".join(chains), "mode": "HS"})
    if enable_masif:
        run_rows.append({"method": "MaSIF-seed", "chains": masif_chain, "mode": "target_site_surface"})
    if run_rows:
        st.dataframe(pd.DataFrame(run_rows), hide_index=True, width="stretch")
    else:
        st.info("Enable at least one method in the Methods tab.")

    run_disabled = (
        not target_ready
        or not run_rows
        or (enable_scannet and not scannet_chains)
        or (enable_pesto and not pesto_chains)
        or (enable_masif and not masif_chain)
    )
    if not target_ready:
        st.warning("Select a readable target with at least one protein chain in the Dataset tab before running methods.")
    if st.button("Run selected methods", type="primary", disabled=run_disabled):
        queued_runs: list[Path] = []
        launch_status: list[dict[str, str]] = []
        if enable_scannet:
            with st.spinner("Queueing ScanNet PPI prediction..."):
                try:
                    run_dir = enqueue_scannet(target_pdb, scannet_chains, mode=scannet_mode, use_msa=use_msa, gpu_device=ppi_gpu_device)
                    spawn_worker_for_run(run_dir)
                    queued_runs.append(run_dir)
                    launch_status.append({"method": "ScanNet", "status": "queued", "run_id": run_dir.name, "message": ""})
                except Exception as exc:
                    launch_status.append({"method": "ScanNet", "status": "failed to queue", "run_id": "", "message": str(exc)})
        if enable_pesto:
            with st.spinner("Queueing PeSTo PPI prediction..."):
                try:
                    run_dir = enqueue_pesto(target_pdb, pesto_chains, gpu_device=ppi_gpu_device)
                    spawn_worker_for_run(run_dir)
                    queued_runs.append(run_dir)
                    launch_status.append({"method": "PeSTo", "status": "queued", "run_id": run_dir.name, "message": ""})
                except Exception as exc:
                    launch_status.append({"method": "PeSTo", "status": "failed to queue", "run_id": "", "message": str(exc)})
        if enable_surf2spot:
            with st.spinner("Queueing Surf2Spot hotspot prediction..."):
                try:
                    run_dir = enqueue_surf2spot(target_pdb, gpu_device=ppi_gpu_device)
                    spawn_worker_for_run(run_dir)
                    queued_runs.append(run_dir)
                    launch_status.append({"method": "Surf2Spot", "status": "queued", "run_id": run_dir.name, "message": ""})
                except Exception as exc:
                    launch_status.append({"method": "Surf2Spot", "status": "failed to queue", "run_id": "", "message": str(exc)})
        if enable_masif:
            with st.spinner("Queueing MaSIF-seed target surface/site prediction..."):
                try:
                    run_dir = enqueue_masif_seed(target_pdb, masif_chain, gpu_device=ppi_gpu_device)
                    spawn_worker_for_run(run_dir)
                    queued_runs.append(run_dir)
                    launch_status.append({"method": "MaSIF-seed", "status": "queued", "run_id": run_dir.name, "message": ""})
                except Exception as exc:
                    launch_status.append({"method": "MaSIF-seed", "status": "failed to queue", "run_id": "", "message": str(exc)})
        st.session_state["ppi_detection_last_launch_status"] = launch_status
        if queued_runs:
            st.success(f"Queued {len(queued_runs)} detection job(s). They will continue outside the UI session.")
            show_pipeline_links(None, queued_runs)
        failed_launches = [row for row in launch_status if row["status"] != "queued"]
        if failed_launches:
            st.error(f"{len(failed_launches)} selected method(s) could not be queued. See launch status below.")

    launch_status = st.session_state.get("ppi_detection_last_launch_status") or []
    if launch_status:
        st.subheader("Last Launch Status")
        st.dataframe(pd.DataFrame(launch_status), hide_index=True, width="stretch")

with results_tab:
    st.subheader("Results")
    refresh_results_button("ppi_detection_refresh_results")
    df = _detection_results_table()
    if df.empty:
        st.info("No PPI/hotspot detection jobs yet.")
    else:
        source_categories = ["All sources", *_source_categories(df.to_dict(orient="records"))]
        selected_source_category = st.selectbox(
            "Source filter",
            source_categories,
            format_func=lambda value: value if value == "All sources" else _category_label(str(value)),
            key="ppi_detection_results_source_filter",
        )
        filtered_df = df.copy()
        if selected_source_category != "All sources":
            filtered_df = filtered_df[filtered_df["source_category"].astype(str) == str(selected_source_category)].copy()

        target_filters = ["All targets", *filtered_df["target_filter"].dropna().astype(str).drop_duplicates().tolist()]
        selected_target_filter = st.selectbox(
            "Target filter",
            target_filters,
            index=0,
            key="ppi_detection_results_target_filter_v2",
        )
        if selected_target_filter != "All targets":
            filtered_df = filtered_df[filtered_df["target_filter"].astype(str) == selected_target_filter].copy()
        if filtered_df.empty:
            st.info("No PPI/hotspot detection jobs match the selected target filter.")
        display_cols = [
            "result",
            "tool",
            "target",
            "source_job",
            "source_type",
            "source_category",
            "source_label",
            "prepared_kind",
            "prepared_pdb",
            "chains",
            "mode",
            "status",
            "artifacts",
            "created_at",
            "job_code",
            "run_id",
        ]
        display_df = filtered_df[[col for col in display_cols if col in filtered_df.columns]].copy()
        table_key = "ppi_detection_results"
        event = st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            key=f"{table_key}_jobs_table",
            on_select="rerun",
            selection_mode="multi-row",
            column_config={"result": st.column_config.LinkColumn("result", display_text="Open")},
        )
        delete_result = st.session_state.pop(f"{table_key}_delete_result", None)
        if delete_result:
            level, message = delete_result
            if level == "success":
                st.success(str(message))
            else:
                st.error(str(message))

        table_widget_key = f"{table_key}_jobs_table"
        selected_indices = [idx for idx in selected_dataframe_rows(event, table_widget_key) if 0 <= idx < len(display_df)]
        selected_rows = filtered_df.iloc[selected_indices].copy() if selected_indices else filtered_df.iloc[0:0].copy()
        active_selected = selected_rows[selected_rows["status"].isin(ACTIVE_STATUSES)]
        if not active_selected.empty:
            st.warning("Running, queued, or preparing detection jobs cannot be deleted.")
        selected_rows = selected_rows[~selected_rows["status"].isin(ACTIVE_STATUSES)]
        selected_refs = [
            (str(row["task_group"]), str(row["run_id"]))
            for row in selected_rows.to_dict(orient="records")
        ]
        selected_refs_key = f"{table_key}_selected_delete_refs"
        if selected_refs:
            st.session_state[selected_refs_key] = selected_refs
        cached_selected_refs = st.session_state.get(selected_refs_key) or []
        delete_clicked = st.button(
            "Delete selected detection jobs",
            type="primary",
            disabled=not cached_selected_refs,
            key=f"{table_key}_request_delete_jobs",
        )
        if delete_clicked and cached_selected_refs:
            show_delete_jobs_dialog(
                table_key=table_key,
                pending_refs=cached_selected_refs,
                selected_refs_key=selected_refs_key,
                label="detection job",
            )
        else:
            st.caption("Select finished detection rows in the table to enable deletion.")
