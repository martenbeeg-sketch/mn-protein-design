from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from mn_protein_design.app.pages.common import result_link, show_pipeline_links
from mn_protein_design.core.jobs import read_json
from mn_protein_design.workflows import candidate_import
from mn_protein_design.workflows.modules import candidate_sources, load_source_candidates


def _read_table_preview(path: Path | None, data: bytes | None, filename: str) -> pd.DataFrame:
    name = str(filename or path or "").lower()
    source = BytesIO(data) if data is not None else path
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(source)
    sep = "\t" if name.endswith((".tsv", ".txt")) else ","
    return pd.read_csv(source, sep=sep)


def _guess_column(columns: list[str], candidates: list[str]) -> str:
    lowered = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for column in columns:
        normalized = column.lower().replace(" ", "_").replace("-", "_")
        if any(candidate.lower() in normalized for candidate in candidates):
            return column
    return ""


def _column_select(label: str, columns: list[str], guessed: str, *, key: str, help: str | None = None) -> str:
    options = [""] + columns
    index = options.index(guessed) if guessed in options else 0
    return st.selectbox(label, options, index=index, format_func=lambda value: value or "None", key=key, help=help)


st.title("Candidate Sets")
st.caption(
    "Import external design results into the app's normalized candidate format. "
    "Imported sets can then be used by Refolding / Validation and Analysis without modifying the original design folders."
)

tabs = st.tabs(["Import", "Available Sets"])

with tabs[0]:
    st.subheader("Import Generic Candidate Table")
    st.caption(
        "Use this for CSV/Excel tables with candidate IDs and binder sequences. "
        "Optional structure columns are preserved for result alignment and comparison."
    )
    generic_upload = st.file_uploader(
        "Candidate CSV/Excel",
        type=["csv", "tsv", "txt", "xlsx", "xls"],
        key="generic_import_upload",
    )
    generic_table_path = st.text_input(
        "Or candidate table path",
        value="",
        placeholder="/path/to/candidates.csv",
        key="generic_import_table_path",
    )
    preview_df = None
    preview_error = ""
    generic_table_bytes = generic_upload.getvalue() if generic_upload is not None else None
    generic_table_filename = generic_upload.name if generic_upload is not None else Path(generic_table_path).name
    if generic_upload is not None or generic_table_path.strip():
        try:
            preview_df = _read_table_preview(
                Path(generic_table_path).expanduser() if generic_upload is None else None,
                generic_table_bytes,
                generic_table_filename,
            )
        except Exception as exc:
            preview_error = str(exc)
    if preview_error:
        st.error(preview_error)
    if preview_df is not None:
        columns = [str(column) for column in preview_df.columns]
        with st.expander("Table preview", expanded=True):
            st.dataframe(preview_df.head(25), hide_index=True, width="stretch")
        id_guess = _guess_column(columns, ["candidate_id", "binder_id", "design_id", "design", "name", "id"])
        sequence_guess = _guess_column(columns, ["binder_sequence", "sequence", "seq", "binder_seq", "aa_sequence"])
        complex_guess = _guess_column(columns, ["complex_pdb", "complex_path", "pdb", "model", "structure", "structure_path"])
        binder_guess = _guess_column(columns, ["binder_pdb", "binder_path", "binder_structure"])
        binder_chain_guess = _guess_column(columns, ["binder_chains", "binder_chain", "designed_chain"])
        target_chain_guess = _guess_column(columns, ["target_chains", "target_chain"])
        map_cols = st.columns(3)
        with map_cols[0]:
            generic_id_column = _column_select("Candidate ID column", columns, id_guess, key="generic_import_id_column")
            generic_sequence_column = _column_select("Binder sequence column", columns, sequence_guess, key="generic_import_sequence_column")
        with map_cols[1]:
            generic_complex_column = _column_select(
                "Complex structure path column",
                columns,
                complex_guess,
                key="generic_import_complex_column",
                help="Optional. Used as the original binder-target complex for alignment/comparison.",
            )
            generic_binder_column = _column_select(
                "Binder-only structure path column",
                columns,
                binder_guess,
                key="generic_import_binder_column",
            )
        with map_cols[2]:
            generic_binder_chains_column = _column_select(
                "Binder chains column",
                columns,
                binder_chain_guess,
                key="generic_import_binder_chains_column",
            )
            generic_target_chains_column = _column_select(
                "Target chains column",
                columns,
                target_chain_guess,
                key="generic_import_target_chains_column",
            )
        generic_cols = st.columns(4)
        with generic_cols[0]:
            generic_base_dir = st.text_input(
                "Path base folder",
                value="",
                placeholder="/path/for/relative/structure/paths",
                key="generic_import_base_dir",
            )
        with generic_cols[1]:
            generic_target = st.text_input(
                "Optional target PDB/CIF",
                value="",
                placeholder="/path/to/target.pdb",
                key="generic_import_target",
                help="Can be replaced later in Refolding / Validation.",
            )
        with generic_cols[2]:
            generic_default_binder_chains = st.text_input(
                "Default binder chain(s)",
                value="A",
                key="generic_import_default_binder_chains",
            )
        with generic_cols[3]:
            generic_default_target_chains = st.text_input(
                "Default target chain(s)",
                value="",
                key="generic_import_default_target_chains",
            )
        generic_settings = st.columns(3)
        with generic_settings[0]:
            generic_source_tool = st.text_input("Source/tool label", value="generic_import", key="generic_import_source_tool")
        with generic_settings[1]:
            generic_max_candidates = st.number_input(
                "Max table rows",
                min_value=0,
                max_value=100000,
                value=0,
                step=1,
                key="generic_import_max_candidates",
                help="0 imports all rows.",
            )
        with generic_settings[2]:
            generic_copy_files = st.checkbox(
                "Copy files into the app workdir",
                value=True,
                key="generic_import_copy_files",
            )
        generic_import_name = st.text_input("Generic import name", value="Generic candidate import", key="generic_import_name")
        generic_ready = bool(generic_sequence_column or generic_complex_column or generic_binder_column)
        if not generic_ready:
            st.warning("Select a sequence column or a structure path column before importing.")
        if st.button("Import generic candidate set", type="primary", disabled=not generic_ready):
            try:
                run_dir = candidate_import.run_generic_table_import(
                    table_path=Path(generic_table_path) if generic_upload is None and generic_table_path.strip() else None,
                    table_bytes=generic_table_bytes,
                    table_filename=generic_table_filename,
                    base_dir=Path(generic_base_dir) if generic_base_dir.strip() else None,
                    id_column=generic_id_column,
                    sequence_column=generic_sequence_column,
                    complex_path_column=generic_complex_column,
                    binder_path_column=generic_binder_column,
                    binder_chains_column=generic_binder_chains_column,
                    target_chains_column=generic_target_chains_column,
                    target_structure=Path(generic_target) if generic_target.strip() else None,
                    default_binder_chains=candidate_import.split_chains(generic_default_binder_chains) or ["A"],
                    default_target_chains=candidate_import.split_chains(generic_default_target_chains),
                    max_candidates=int(generic_max_candidates),
                    copy_files=bool(generic_copy_files),
                    import_name=generic_import_name,
                    source_tool=generic_source_tool.strip() or "generic_import",
                )
                st.success("Candidate set imported.")
                show_pipeline_links(run_dir, [run_dir])
            except Exception as exc:
                st.error(str(exc))

    st.divider()
    st.subheader("Import BindCraft Results")
    st.caption(
        "Point this at a BindCraft output folder. When final_design_stats.csv is present, it defines the "
        "canonical accepted designs, original BindCraft rank, sequence, and structure selection."
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
            value="B",
            help="Comma-separated chain IDs in the imported complex that belong to the designed binder.",
            key="bindcraft_import_binder_chains",
        )
    with cols[1]:
        target_chains = st.text_input(
            "Target chain(s)",
            value="A",
            help="Comma-separated target chains. BindCraft accepted complexes commonly use A for target and B for binder; leave empty to use every non-binder chain.",
            key="bindcraft_import_target_chains",
        )
    with cols[2]:
        file_pattern = st.text_input(
            "Structure file pattern",
            value="**/*.pdb",
            help=(
                "Glob pattern relative to the BindCraft folder. If final_design_stats.csv exists, matching "
                "structures are still selected from Accepted/Ranked first, then Accepted."
            ),
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
                binder_chains=candidate_import.split_chains(binder_chains) or ["B"],
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
