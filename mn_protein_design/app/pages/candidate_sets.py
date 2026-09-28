from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from mn_protein_design.app.pages.common import gpu_run_panel, result_link, show_pipeline_links
from mn_protein_design.core.jobs import create_job, read_json, write_json
from mn_protein_design.core.local_worker import spawn_worker_for_run
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


def _first_text(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _run_metadata(run_dir_text: object) -> dict:
    run_dir = Path(str(run_dir_text or ""))
    if not str(run_dir):
        return {}
    return read_json(run_dir / "metadata.json")


def _source_result_link(run_dir_text: object) -> str:
    run_dir = Path(str(run_dir_text or ""))
    if not str(run_dir):
        return ""
    metadata = _run_metadata(run_dir)
    task_group = _first_text(metadata.get("task_group"))
    run_id = _first_text(metadata.get("run_id"), run_dir.name)
    if not task_group or not run_id:
        return ""
    return result_link(task_group, run_id, "Open source")


def _format_threshold_value(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value or "").strip()
    return f"{number:.4g}"


def _selection_rule_text(manifest: dict) -> str:
    filter_config = manifest.get("filter_config") if isinstance(manifest.get("filter_config"), dict) else {}
    selected_feature = _first_text(manifest.get("selected_feature"), filter_config.get("selected_feature"))
    direction = _first_text(manifest.get("direction"), filter_config.get("direction"))
    parts: list[str] = []
    manual = filter_config.get("manual_threshold") if isinstance(filter_config.get("manual_threshold"), dict) else {}
    if selected_feature and manual.get("enabled"):
        condition = _first_text(manual.get("condition"), ">=" if direction == "higher" else "<=")
        parts.append(f"{selected_feature} {condition} {_format_threshold_value(manual.get('value'))}")
    elif selected_feature and filter_config.get("manual_threshold_enabled"):
        condition = _first_text(filter_config.get("manual_threshold_operator"), ">=" if direction == "higher" else "<=")
        parts.append(f"{selected_feature} {condition} {_format_threshold_value(filter_config.get('manual_threshold'))}")
    elif selected_feature:
        parts.append(f"{selected_feature} ({direction or 'ranked'})")
    keep_per_parent = filter_config.get("keep_per_parent")
    if keep_per_parent not in {None, "", 0, "0"}:
        parts.append(f"top {keep_per_parent} per parent")
    sequence_filters = filter_config.get("sequence_filters") if isinstance(filter_config.get("sequence_filters"), dict) else {}
    for feature, rule in sequence_filters.items():
        if not isinstance(rule, dict) or not rule.get("enabled"):
            continue
        if "threshold" in rule:
            parts.append(f"sequence filter {feature} {rule.get('operator', '<=')} {_format_threshold_value(rule.get('threshold'))}")
        elif "min" in rule or "max" in rule:
            parts.append(
                f"sequence filter {feature} "
                f"{_format_threshold_value(rule.get('min'))}-{_format_threshold_value(rule.get('max'))}"
            )
    prefilters = filter_config.get("prefilters") if isinstance(filter_config.get("prefilters"), list) else []
    for prefilter in prefilters:
        if not isinstance(prefilter, dict):
            continue
        feature = _first_text(prefilter.get("feature"))
        operator = _first_text(prefilter.get("operator"))
        threshold = _format_threshold_value(prefilter.get("threshold"))
        if feature and operator and threshold:
            parts.append(f"prefilter {feature} {operator} {threshold}")
    return "; ".join(parts)


def _candidate_set_provenance(source: dict) -> dict:
    run_dir = Path(str(source.get("run_dir") or ""))
    normalized_dir = run_dir / "artifacts" / "normalized_candidates"
    manifest = read_json(normalized_dir / "result_selection_manifest.json")
    import_sources = sorted((run_dir / "artifacts" / "raw").glob("*/import_source.json")) if run_dir.exists() else []
    import_source = read_json(import_sources[0]) if import_sources else {}
    source_run_dir = _first_text(manifest.get("source_run_dir"), import_source.get("source_run_dir"))
    source_metadata = _run_metadata(source_run_dir)
    candidates = load_source_candidates(source)
    first_candidate = candidates[0] if candidates else {}
    raw_metadata = first_candidate.get("raw_metadata") if isinstance(first_candidate.get("raw_metadata"), dict) else {}
    metrics = first_candidate.get("metrics") if isinstance(first_candidate.get("metrics"), dict) else {}
    source_table = _first_text(
        import_source.get("table_filename"),
        Path(str(import_source.get("table_path") or "")).name,
        Path(str(raw_metadata.get("source_table") or metrics.get("import_source_table") or "")).name,
    )
    source_label = _first_text(
        manifest.get("export_name"),
        raw_metadata.get("import_name"),
        source_table,
        source_metadata.get("job_code"),
        source_run_dir,
    )
    return {
        "source_result": _source_result_link(source_run_dir),
        "source_job": _first_text(source_metadata.get("job_code")),
        "source_run_id": _first_text(source_metadata.get("run_id"), Path(source_run_dir).name if source_run_dir else ""),
        "source_task": _first_text(source_metadata.get("task_group")),
        "source_tool": _first_text(source_metadata.get("tool"), first_candidate.get("source_tool"), source.get("tool")),
        "source_label": source_label,
        "source_table": source_table,
        "target_id": _first_text(first_candidate.get("target_id"), raw_metadata.get("target_id")),
        "selected_engine": _first_text(manifest.get("selected_engine")),
        "selected_feature": _first_text(manifest.get("selected_feature")),
        "selection_kind": _first_text(manifest.get("selection_kind")),
        "selection_rule": _selection_rule_text(manifest),
        "provenance_manifest": manifest or import_source,
    }


def _candidate_role_summary(candidates: list[dict]) -> dict[str, int]:
    explicit = 0
    with_chain_roles = 0
    with_binder_target = 0
    for candidate in candidates:
        raw_metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
        schema = _first_text(candidate.get("chain_role_schema"), raw_metadata.get("chain_role_schema"))
        if schema:
            explicit += 1
        if isinstance(raw_metadata.get("chain_roles"), dict) or isinstance(candidate.get("chain_roles"), dict):
            with_chain_roles += 1
        if candidate.get("binder_chains") and candidate.get("target_chains"):
            with_binder_target += 1
    return {
        "candidates": len(candidates),
        "with_binder_and_target_chains": with_binder_target,
        "with_chain_role_schema": explicit,
        "with_chain_role_map": with_chain_roles,
    }


st.title("Candidate Sets")
st.caption(
    "Import external design results into the app's normalized candidate format. "
    "Imported sets can then be used by Refolding / Validation and Analysis without modifying the original design folders. "
    "Declare the chains as they appear in the source files here; downstream jobs normalize targets to A... and binders to Z..."
)

tabs = st.tabs(["Import", "Available Sets"])

with tabs[0]:
    st.subheader("Import Generic Candidate Table")
    st.caption(
        "Use this for CSV/Excel tables with candidate IDs and binder sequences. "
        "Optional structure columns are preserved for result alignment and comparison. "
        "Binder/target chain fields should describe the source structure, even if it uses an older binder-A target-B layout."
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
                help=(
                    "Fallback source-chain ID(s) for rows without a binder chain column. "
                    "Use the chain letters in the imported structure, not the downstream normalized Z/Y binder convention."
                ),
            )
        with generic_cols[3]:
            generic_default_target_chains = st.text_input(
                "Default target chain(s)",
                value="",
                key="generic_import_default_target_chains",
                help=(
                    "Fallback source-chain ID(s) for target chains. Leave empty only when every non-binder chain "
                    "in the source complex should be treated as target."
                ),
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
        if generic_complex_column and not (generic_binder_chains_column or generic_default_binder_chains.strip()):
            st.warning("Complex imports need a binder-chain column or a correct default binder chain.")
        if generic_complex_column and not generic_target_chains_column and not generic_default_target_chains.strip():
            st.info("No target-chain field/default is set; import will treat every non-binder source chain as target.")
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
        "canonical accepted designs, original BindCraft rank, sequence, and structure selection. "
        "BindCraft-native complexes commonly use target A and binder B; downstream jobs will stage them into the app convention."
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
        provenances = [_candidate_set_provenance(source) for source in sources]
        rows = []
        for source, provenance in zip(sources, provenances, strict=False):
            rows.append(
                {
                    "result": result_link(str(source.get("task_group")), str(source.get("run_id")), "Open"),
                    "job_code": source.get("job_code"),
                    "source": provenance.get("source_label"),
                    "source_result": provenance.get("source_result"),
                    "source_job": provenance.get("source_job"),
                    "source_task": provenance.get("source_task"),
                    "source_tool": provenance.get("source_tool"),
                    "source_table": provenance.get("source_table"),
                    "target_id": provenance.get("target_id"),
                    "selected_engine": provenance.get("selected_engine"),
                    "selected_feature": provenance.get("selected_feature"),
                    "selection_kind": provenance.get("selection_kind"),
                    "selection_rule": provenance.get("selection_rule"),
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
            column_config={
                "result": st.column_config.LinkColumn("result", display_text="Open"),
                "source_result": st.column_config.LinkColumn("source_result", display_text="Open source"),
            },
        )
        selected_index = st.selectbox(
            "Preview candidate set",
            range(len(sources)),
            format_func=lambda index: (
                f"{sources[index]['job_code']} | "
                f"{provenances[index].get('source_label') or sources[index]['tool']} | "
                f"{sources[index]['candidate_count']} candidates"
            ),
        )
        source = sources[selected_index]
        provenance = provenances[selected_index]
        if provenance.get("provenance_manifest"):
            with st.expander("Candidate set provenance", expanded=False):
                st.json(provenance.get("provenance_manifest"))
        candidates = load_source_candidates(source)
        role_summary = _candidate_role_summary(candidates)
        role_cols = st.columns(4)
        role_cols[0].metric("Candidates", f"{role_summary['candidates']:,}")
        role_cols[1].metric("Binder + target chains", f"{role_summary['with_binder_and_target_chains']:,}")
        role_cols[2].metric("Role schema", f"{role_summary['with_chain_role_schema']:,}")
        role_cols[3].metric("Role maps", f"{role_summary['with_chain_role_map']:,}")
        if candidates and role_summary["with_binder_and_target_chains"] < len(candidates):
            structured_missing_roles = [
                candidate
                for candidate in candidates
                if candidate.get("complex_pdb") and not (candidate.get("binder_chains") and candidate.get("target_chains"))
            ]
            if structured_missing_roles:
                st.warning(
                    "Some structured candidates do not declare both binder and target chains. "
                    "Downstream staging can infer simple cases, but explicit source-chain roles are safer for old/new mixed data."
                )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "candidate_id": candidate.get("candidate_id"),
                        "stage": candidate.get("stage"),
                        "source_tool": candidate.get("source_tool"),
                        "binder_chains": ",".join(candidate.get("binder_chains") or []),
                        "target_chains": ",".join(candidate.get("target_chains") or []),
                        "chain_role_schema": (
                            (candidate.get("raw_metadata") or {}).get("chain_role_schema")
                            if isinstance(candidate.get("raw_metadata"), dict)
                            else ""
                        ),
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
        with st.expander("Run binder monomer refolding", expanded=False):
            candidates_jsonl = Path(str(source.get("candidates_jsonl") or ""))
            st.caption(
                "Queues this candidate set for binder-only monomer folding. "
                "Use this to check whether selected binders keep a plausible monomer fold before another complex-refolding round."
            )
            show_experimental_esmfold = st.checkbox(
                "Show experimental ESMFold monomer backend",
                value=False,
                key=f"candidate_set_monomer_show_esmfold_{source.get('run_id')}",
                help=(
                    "The current ovo-esm image must include the ESMFold Python dependencies. "
                    "If einops is missing, ESMFold jobs fail before folding."
                ),
            )
            monomer_tool_options = ["boltz2_monomer", "esmfold2_monomer"]
            if show_experimental_esmfold:
                monomer_tool_options.append("esmfold")
            if show_experimental_esmfold:
                st.warning(
                    "ESMFold monomer is experimental on this installation. "
                    "The `ovo-esm:latest` image must contain `einops`; otherwise it exits before producing structures."
                )
            monomer_tool = st.selectbox(
                "Monomer folding engine",
                monomer_tool_options,
                format_func=lambda value: {
                    "boltz2_monomer": "Boltz-2 monomer",
                    "esmfold2_monomer": "ESMFold2 monomer",
                    "esmfold": "Legacy ESMFold monomer",
                }.get(value, value),
                key=f"candidate_set_monomer_tool_{source.get('run_id')}",
            )
            monomer_min_confidence = 0.7 if monomer_tool == "boltz2_monomer" else 70.0
            monomer_gpu_device = gpu_run_panel(key=f"candidate_set_monomer_{source.get('run_id')}", default="0")
            can_launch_monomer = bool(candidates_jsonl.exists() and candidates)
            if not candidates_jsonl.exists():
                st.warning("This candidate set has no normalized candidates.jsonl file.")
            if st.button(
                "Queue binder monomer refolding",
                type="primary",
                disabled=not can_launch_monomer,
                key=f"candidate_set_monomer_launch_{source.get('run_id')}",
            ):
                job = create_job(
                    "refolding-validation",
                    job_type="monomer_refolding",
                    tool=monomer_tool,
                    inputs={
                        "source_run_dir": str(source_run_dir),
                        "candidates_jsonl": str(candidates_jsonl),
                    },
                    params={
                        "min_plddt": float(monomer_min_confidence),
                        "gpu_device": monomer_gpu_device,
                        "source_job_code": source.get("job_code"),
                        "source_run_id": source.get("run_id"),
                        "source_candidate_count": source.get("candidate_count"),
                        "source_selection_rule": provenance.get("selection_rule"),
                    },
                )
                metadata = read_json(job.run_dir / "metadata.json")
                metadata.update(
                    {
                        "title": "Binder monomer refolding",
                        "description": (
                            f"from {source.get('job_code') or source.get('run_id')}; "
                            f"{source.get('candidate_count')} selected; {monomer_tool}"
                        ),
                        "source_job_code": source.get("job_code"),
                        "source_run_id": source.get("run_id"),
                        "source_task_group": source.get("task_group"),
                        "source_tool": source.get("tool"),
                        "source_selection_rule": provenance.get("selection_rule"),
                        "comments": "Active job; do not delete while running or queued.",
                    }
                )
                write_json(job.run_dir / "metadata.json", metadata)
                write_json(
                    job.run_dir / "worker_request.json",
                    {
                        "kind": "monomer_refolding",
                        "kwargs": {
                            "source_run_dir": str(source_run_dir),
                            "candidates_jsonl": str(candidates_jsonl),
                            "tool": monomer_tool,
                            "min_plddt": float(monomer_min_confidence),
                            "gpu_device": monomer_gpu_device,
                        },
                        "path_kwargs": ["source_run_dir", "candidates_jsonl"],
                    },
                )
                spawn_worker_for_run(job.run_dir)
                st.success(f"Queued binder monomer refolding job {metadata.get('job_code')}.")
                st.link_button(
                    "Open monomer refolding job",
                    result_link("refolding-validation", job.run_dir.name),
                )
