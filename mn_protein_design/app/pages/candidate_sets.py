from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from mn_protein_design.app.pages.common import result_link, show_pipeline_links
from mn_protein_design.core.jobs import read_json
from mn_protein_design.workflows import candidate_import
from mn_protein_design.workflows.modules import candidate_sources, load_source_candidates


st.title("Candidate Sets")
st.caption(
    "Import external design results into the app's normalized candidate format. "
    "Imported sets can then be used by Refolding / Validation and Analysis without modifying the original design folders."
)

tabs = st.tabs(["Import", "Available Sets"])

with tabs[0]:
    st.subheader("Import BindCraft Results")
    st.caption(
        "Point this at a BindCraft output folder containing predicted complex PDB/CIF files. "
        "The importer stages the structures and writes a normalized candidate table."
    )
    input_dir = st.text_input(
        "BindCraft output folder",
        value="",
        placeholder="/path/to/bindcraft/output",
        key="bindcraft_import_input_dir",
    )
    target_pdb = st.text_input(
        "Target PDB",
        value="",
        placeholder="/path/to/target.pdb",
        help="Recommended. Used later by refolding/evaluation tasks for target-template and target-chain context.",
        key="bindcraft_import_target_pdb",
    )
    cols = st.columns(4)
    with cols[0]:
        binder_chains = st.text_input(
            "Binder chain(s)",
            value="A",
            help="Comma-separated chain IDs in the imported complex that belong to the designed binder.",
            key="bindcraft_import_binder_chains",
        )
    with cols[1]:
        target_chains = st.text_input(
            "Target chain(s)",
            value="",
            help="Comma-separated target chains. Leave empty to use every non-binder chain in each complex.",
            key="bindcraft_import_target_chains",
        )
    with cols[2]:
        file_pattern = st.text_input(
            "Structure file pattern",
            value="**/*.pdb",
            help="Glob pattern relative to the BindCraft folder. Use a narrower pattern if the folder contains trajectories.",
            key="bindcraft_import_file_pattern",
        )
    with cols[3]:
        max_candidates = st.number_input(
            "Max candidates",
            min_value=0,
            max_value=100000,
            value=0,
            step=1,
            help="0 imports all matching structures.",
            key="bindcraft_import_max_candidates",
        )
    copy_files = st.checkbox(
        "Copy structures into the app workdir",
        value=True,
        help="Recommended. If off, the import run uses symlinks when possible.",
        key="bindcraft_import_copy_files",
    )
    import_name = st.text_input("Import name", value="BindCraft import", key="bindcraft_import_name")
    if st.button("Import BindCraft candidate set", type="primary", disabled=not bool(input_dir.strip())):
        try:
            run_dir = candidate_import.run_bindcraft_import(
                input_dir=Path(input_dir),
                target_pdb=Path(target_pdb) if target_pdb.strip() else None,
                binder_chains=candidate_import.split_chains(binder_chains) or ["A"],
                target_chains=candidate_import.split_chains(target_chains),
                file_pattern=file_pattern.strip() or "**/*.pdb",
                max_candidates=int(max_candidates),
                copy_files=bool(copy_files),
                import_name=import_name,
            )
            st.success("Candidate set imported.")
            show_pipeline_links(run_dir, [run_dir])
        except Exception as exc:
            st.error(str(exc))

with tabs[1]:
    st.subheader("Available Candidate Sets")
    sources = candidate_sources()
    if not sources:
        st.info("No normalized candidate sets are available yet.")
    else:
        rows = []
        for source in sources:
            rows.append(
                {
                    "result": result_link(str(source.get("task_group")), str(source.get("run_id")), "Open"),
                    "job_code": source.get("job_code"),
                    "task_group": source.get("task_group"),
                    "tool": source.get("tool"),
                    "status": source.get("status"),
                    "candidate_count": source.get("candidate_count"),
                    "stage_counts": source.get("stage_counts"),
                    "created_at": source.get("created_at"),
                    "run_id": source.get("run_id"),
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            width="stretch",
            column_config={"result": st.column_config.LinkColumn("result", display_text="Open")},
        )
        selected_index = st.selectbox(
            "Preview candidate set",
            range(len(sources)),
            format_func=lambda index: f"{sources[index]['job_code']} | {sources[index]['tool']} | {sources[index]['candidate_count']} candidates",
        )
        source = sources[selected_index]
        candidates = load_source_candidates(source)
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "candidate_id": candidate.get("candidate_id"),
                        "stage": candidate.get("stage"),
                        "source_tool": candidate.get("source_tool"),
                        "binder_chains": ",".join(candidate.get("binder_chains") or []),
                        "target_chains": ",".join(candidate.get("target_chains") or []),
                        "binder_sequence": candidate.get("binder_sequence"),
                        "complex_pdb": candidate.get("complex_pdb"),
                        "target_pdb": candidate.get("target_pdb"),
                    }
                    for candidate in candidates[:200]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        source_run_dir = Path(str(source.get("run_dir") or ""))
        result = read_json(source_run_dir / "result.json")
        if result.get("metrics"):
            with st.expander("Source metrics", expanded=False):
                st.json(result.get("metrics"))
