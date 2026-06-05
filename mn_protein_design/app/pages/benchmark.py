from __future__ import annotations

import os
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from mn_protein_design.app.pages.common import (
    result_link,
    selected_dataframe_rows,
    show_delete_jobs_dialog,
    show_pipeline_links,
)
from mn_protein_design.core.jobs import ACTIVE_STATUSES, collect_jobs, finish_job, read_json
from mn_protein_design.core.local_worker import spawn_worker_for_run
from mn_protein_design.core.runtime_estimator import estimate_engines
from mn_protein_design.workflows.benchmark import (
    ALPHAFAST_DB_DIR,
    ALPHAFAST_WEIGHTS_DIR,
    COLABFOLD_CACHE_DIR,
    MSA_REPOSITORY_DIR,
    PUBLISHED_DATASET,
    benchmark_engines_in_metrics,
    create_benchmark_collection,
    enqueue_de_novo_binder_scoring_dataset,
    run_esmfold2_binder_benchmark,
    run_precomputed_metric_benchmark,
)
from mn_protein_design.workflows.esm_binder import ESMFOLD2_MODEL_DIR
from mn_protein_design.workflows.refolding import BOLTZ_MODELS_DIR, RF3_CHECKPOINT


KNOWN_BENCHMARK_ROOT = Path("/mnt/db/reference_files/de_novo_binder_scoring_overath_2025")
KNOWN_BENCHMARK_CSV = KNOWN_BENCHMARK_ROOT / "final_dataset.csv"
KNOWN_BENCHMARK_PDB_DIR = KNOWN_BENCHMARK_ROOT / "input_pdbs"


def _safe_sort_token(value: object) -> str:
    token = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value or "sort"))
    return token.strip("_") or "sort"


@st.cache_data(show_spinner=False)
def _known_benchmark_summary(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["binder_id", "target_id", "binder", "source", "A_length", "B_length"])
    binder_values = df["binder"].astype(str).str.lower().isin({"1", "true", "yes", "y", "binder", "positive"})
    summary = (
        df.assign(binder_bool=binder_values)
        .groupby("target_id", dropna=False)
        .agg(
            records=("binder_id", "count"),
            binders=("binder_bool", "sum"),
            sources=("source", "nunique"),
            median_binder_len=("A_length", "median"),
            median_target_len=("B_length", "median"),
        )
        .reset_index()
        .sort_values(["records", "target_id"], ascending=[False, True])
    )
    summary["nonbinders"] = summary["records"] - summary["binders"]
    return summary[["target_id", "records", "binders", "nonbinders", "sources", "median_binder_len", "median_target_len"]]


@st.cache_data(show_spinner=False)
def _known_benchmark_subset(csv_path: str, targets: tuple[str, ...], max_per_target: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if targets:
        df = df[df["target_id"].astype(str).isin(set(targets))]
    if max_per_target > 0 and not df.empty:
        df = df.groupby("target_id", group_keys=False).head(max_per_target)
    return df


def _dataset_runtime_size(df: pd.DataFrame | None, fallback_count: int = 0) -> dict[str, int | None]:
    if df is None or df.empty:
        return {"candidate_count": int(fallback_count or 0), "total_residues": None}
    candidate_count = int(len(df))
    length_columns = [column for column in ["A_length", "B_length"] if column in df.columns]
    total_residues = None
    if length_columns:
        total = 0
        for column in length_columns:
            total += int(pd.to_numeric(df[column], errors="coerce").fillna(0).sum())
        total_residues = total or None
    return {"candidate_count": candidate_count, "total_residues": total_residues}


def _estimate_rows_dataframe(estimate: dict) -> pd.DataFrame:
    rows = []
    for row in estimate.get("rows") or []:
        rows.append(
            {
                "engine": row.get("label"),
                "estimated_time": row.get("estimated_time"),
                "seconds_per_candidate": row.get("seconds_per_candidate"),
                "basis": row.get("basis"),
                "history_runs": row.get("history_runs"),
            }
        )
    return pd.DataFrame(rows)


def _enabled_engines(params: dict) -> str:
    if isinstance(params.get("selections"), list):
        collection_engines: list[str] = []
        for selection in params.get("selections") or []:
            if isinstance(selection, dict):
                collection_engines.extend(str(engine) for engine in (selection.get("engines") or []))
        if collection_engines:
            return ", ".join(dict.fromkeys(collection_engines))
    engines: list[str] = []
    if params.get("run_alphafast_af3"):
        engines.append("AF3")
    if params.get("run_colabfold"):
        engines.append("ColabFold")
    if params.get("run_af2_initial_guess"):
        engines.append("AF2-IG")
    if params.get("run_boltz2_initial_guess"):
        engines.append("Boltz-2")
    if params.get("run_esmfold2"):
        engines.append("ESMFold2")
    if params.get("run_rf3"):
        engines.append("RF3")
    if params.get("run_protenix"):
        engines.append("Protenix")
    if params.get("run_boltzgen_fold"):
        engines.append("BoltzGen fold")
    if not engines and params.get("modes"):
        engines.append("ESMFold2")
    return ", ".join(engines) or "metrics only"


def _job_dataset_label(input_payload: dict) -> str:
    job_type = str(input_payload.get("job_type") or "")
    params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
    if job_type == "benchmark_collection":
        return f"Collection: {params.get('collection_name') or 'Benchmark collection'}"
    inputs = input_payload.get("inputs") if isinstance(input_payload.get("inputs"), dict) else {}
    if job_type == "de_novo_binder_scoring_dataset":
        pdb_dir = str(inputs.get("input_pdb_dir") or "")
        if "de_novo_binder_scoring_overath_2025" in pdb_dir:
            return "Overath 2025"
        return "repo-format"
    if job_type == "binder_benchmark":
        return "ESMFold2 CSV"
    if job_type == "metric_dataset_benchmark":
        return "metric table"
    return job_type or "benchmark"


def _has_plot_outputs(run_dir: Path) -> str:
    benchmark_dir = run_dir / "artifacts" / "benchmark"
    if not benchmark_dir.exists():
        return "no"
    has_ranking = (benchmark_dir / "merged_benchmark_feature_ranking.csv").exists() or any(
        benchmark_dir.glob("*feature*benchmark*.csv")
    )
    has_metrics = (benchmark_dir / "merged_benchmark_metrics.csv").exists() or any(
        benchmark_dir.glob("*_metrics.csv")
    )
    return "yes" if has_ranking and has_metrics else "partial" if has_ranking else "no"


def _first_feature_summary(run_dir: Path) -> dict:
    benchmark_dir = run_dir / "artifacts" / "benchmark"
    preferred = [
        "merged_benchmark_feature_summary.json",
        "metric_feature_summary.json",
        "esmfold2_balanced_candidate_feature_summary.json",
        "esmfold2_feature_summary.json",
        "alphafast_af3_feature_summary.json",
        "colabfold_feature_summary.json",
        "af2_initial_guess_feature_summary.json",
        "boltz2_initial_guess_feature_summary.json",
        "common_interface_feature_summary.json",
        "esmfold2_common_interface_feature_summary.json",
        "af2_common_interface_feature_summary.json",
    ]
    for name in preferred:
        summary = read_json(benchmark_dir / name)
        if summary:
            return summary
    for path in sorted(benchmark_dir.glob("*feature_summary.json")):
        summary = read_json(path)
        if summary:
            return summary
    return {}


def _benchmark_job_table(rows: list[dict]) -> pd.DataFrame:
    enriched: list[dict] = []
    for row in rows:
        run_dir = Path(str(row.get("run_dir") or ""))
        input_payload = read_json(run_dir / "input.json")
        job_type = str(input_payload.get("job_type") or "")
        result = read_json(run_dir / "result.json")
        feature_summary = _first_feature_summary(run_dir)
        params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
        top_ap = (
            feature_summary.get("top_feature_average_precision")
            or metrics.get("merged_benchmark_top_feature_average_precision")
            or metrics.get("top_feature_average_precision")
            or metrics.get("average_precision")
        )
        top_feature = feature_summary.get("top_feature") or metrics.get("merged_benchmark_top_feature") or metrics.get("top_feature")
        records = metrics.get("record_count")
        if records is None and isinstance(outputs.get("candidates"), list):
            records = len(outputs.get("candidates") or [])
        description_parts = [
            _job_dataset_label(input_payload),
            _enabled_engines(params),
        ]
        if records:
            description_parts.append(f"{records} rows")
        enriched.append(
            {
                "delete": False,
                "result": result_link("benchmark", str(row.get("run_id")), "Open result"),
                "kind": "collection" if job_type == "benchmark_collection" else "benchmark",
                "description": " | ".join(description_parts),
                "dataset": _job_dataset_label(input_payload),
                "engines": _enabled_engines(params),
                "records": records,
                "top_feature": top_feature,
                "top_ap": top_ap,
                "plots": _has_plot_outputs(run_dir),
                "status": row.get("status"),
                "created_at": row.get("created_at"),
                "task_group": "benchmark",
                "run_id": row.get("run_id"),
                "job_code": row.get("job_code"),
            }
        )
    return pd.DataFrame(enriched)


st.title("Binder Benchmark")
st.caption("Run labeled binder/nonbinder datasets through prediction engines, then compare shared interface metrics.")

dataset_source = "Installed Overath 2025"
known_csv_text: str | None = None
known_pdb_dir: Path | None = None
repo_zip = None
repo_zip_path = ""
repo_input_csv = None
repo_input_csv_path = ""
repo_pdb_dir = ""
repo_csv_text: str | None = None
repo_zip_bytes: bytes | None = None
simple_csv_text: str | None = None
simple_csv_path = ""
metric_text: str | None = None
metric_path = ""
use_published_metrics = True

run_repo_esm = True
run_repo_af2ig = True
run_boltz2_ig = True
run_colabfold = True
run_alphafast_af3 = True
run_rf3 = False
run_protenix = False
run_boltzgen_fold = False
generate_inputs = True
repo_models: list[str] = ["af3", "boltz", "colabfold"]
run_pyrosetta_input = True
default_pyrosetta_nprocs = max(1, min(64, os.cpu_count() or 1))
pyrosetta_nprocs = default_pyrosetta_nprocs
run_common_interface_metrics = True
run_predicted_rosetta_metrics = True
run_pymol_metrics = True
repo_mode = "pdb_only"
repo_max_records = 0
repo_esm_modes = ["initial_guess"]
repo_sampling = 32
repo_loops = 3
repo_seed = 0
af2_recycles = 3
af2_multimer = True
af2_binder_template = False
af2_interface_template = False
boltz2_target_template = True
boltz2_use_target_msa = bool(st.session_state.get("benchmark_boltz2_use_target_msa", True))
boltz2_recycling_steps = 10
boltz2_sampling_steps = 200
boltz2_diffusion_samples = 3
boltz2_write_full_pae = True
rf3_checkpoint_path = str(RF3_CHECKPOINT)
rf3_use_target_msa = bool(st.session_state.get("benchmark_rf3_use_target_msa", True))
rf3_recycles = 10
rf3_num_steps = 50
rf3_diffusion_batch_size = 5
rf3_seed = 0
protenix_use_msa = bool(st.session_state.get("benchmark_protenix_use_msa", True))
protenix_cycle = 3
protenix_diffusion_steps = 50
protenix_samples = 5
boltzgen_recycling_steps = 3
boltzgen_sampling_steps = 200
boltzgen_diffusion_samples = 5
esmfold2_use_target_msa = bool(st.session_state.get("benchmark_esmfold2_use_target_msa", False))
colabfold_cache_dir = str(COLABFOLD_CACHE_DIR)
colabfold_recycles = 3
colabfold_models = 3
colabfold_use_target_templates = bool(st.session_state.get("benchmark_colabfold_use_target_templates", True))
colabfold_max_template_hits = 4
colabfold_msa_source = "msa_repository_then_alphafast_mmseqs_gpu"
msa_repository_dir = str(MSA_REPOSITORY_DIR)
alphafast_db_dir = str(ALPHAFAST_DB_DIR)
alphafast_weights_dir = str(ALPHAFAST_WEIGHTS_DIR)
alphafast_batch_size = 0
alphafast_recycles = 10
gpu_device = 0
simple_modes = ["sequence", "initial_guess"]
simple_max_records = 0
simple_sampling_steps = 32
simple_recycling_loops = 3
simple_seed = 0
simple_contact_cutoff = 8.0
simple_device = "cuda"
label_column = "binder"
metric_max_rows = 0
metric_max_cols = 500
runtime_size = {"candidate_count": 0, "total_residues": None}
selected_targets: list[str] = []
selected_known_count = 0

tabs = st.tabs(["Dataset", "Engines", "Metrics", "Run", "Results", "Collections"])

with tabs[0]:
    st.subheader("Dataset")
    dataset_source = st.segmented_control(
        "Input source",
        ["Installed Overath 2025", "Repo-format upload/path", "Simple ESMFold2 CSV", "Metric table only"],
        selection_mode="single",
        default="Installed Overath 2025",
    )

    if dataset_source == "Installed Overath 2025":
        known_available = KNOWN_BENCHMARK_CSV.exists() and KNOWN_BENCHMARK_PDB_DIR.exists()
        if not known_available:
            st.error(f"Installed dataset was not found at {KNOWN_BENCHMARK_ROOT}.")
        else:
            summary = _known_benchmark_summary(str(KNOWN_BENCHMARK_CSV))
            cols = st.columns(4)
            cols[0].metric("Rows", f"{int(summary['records'].sum()):,}")
            cols[1].metric("Targets", f"{len(summary):,}")
            cols[2].metric("Binders", f"{int(summary['binders'].sum()):,}")
            cols[3].metric("Nonbinders", f"{int(summary['nonbinders'].sum()):,}")
            with st.expander("Target Summary", expanded=True):
                st.dataframe(summary, width="stretch", hide_index=True)

            target_options = [
                target
                for target in summary["target_id"].dropna().astype(str).tolist()
                if target.strip() and target.strip().lower() != "nan"
            ]
            use_all_targets = st.checkbox(
                "Include all targets",
                value=False,
                help="Show records from every target in the selectable record table.",
            )
            if use_all_targets:
                selected_targets = target_options
                st.caption(f"All {len(selected_targets):,} targets are included.")
            else:
                selected_targets = st.multiselect(
                    "Targets to benchmark",
                    target_options,
                    default=[],
                    help="Select one or more targets from the installed benchmark dataset.",
                )
            if not selected_targets:
                st.warning("Select at least one target to enable a rerun from the installed dataset.")
            else:
                subset = _known_benchmark_subset(str(KNOWN_BENCHMARK_CSV), tuple(selected_targets), 0).copy()
                subset["binder_id"] = subset["binder_id"].astype(str)
                available_ids = set(subset["binder_id"])
                selected_ids_key = "known_benchmark_selected_ids"
                selected_ids = set(st.session_state.get(selected_ids_key, set())) & available_ids

                target_signature = tuple(sorted(selected_targets))
                previous_signature = st.session_state.get("known_benchmark_target_signature")
                select_all = st.checkbox(
                    "Select all records for selected targets",
                    value=False,
                    key="known_benchmark_select_all",
                    help="Checking selects every currently filtered record. Unchecking clears the selection.",
                )
                previous_select_all = bool(st.session_state.get("known_benchmark_select_all_previous", False))
                selection_changed = select_all != previous_select_all or target_signature != previous_signature
                if selection_changed:
                    if select_all:
                        selected_ids = available_ids
                    elif select_all != previous_select_all:
                        selected_ids = set()
                    st.session_state["known_benchmark_editor_version"] = (
                        int(st.session_state.get("known_benchmark_editor_version", 0)) + 1
                    )
                st.session_state["known_benchmark_select_all_previous"] = select_all
                st.session_state["known_benchmark_target_signature"] = target_signature

                preview_cols = [
                    "binder_id",
                    "target_id",
                    "binder",
                    "source",
                    "binder_chain",
                    "target_chains",
                    "A_length",
                    "B_length",
                ]
                display_cols = [col for col in preview_cols if col in subset.columns]
                sort_options = display_cols or ["binder_id"]
                sort_cols = st.columns([2, 1, 4])
                sort_col = sort_cols[0].selectbox(
                    "Record table order",
                    sort_options,
                    index=sort_options.index("binder_id") if "binder_id" in sort_options else 0,
                    key="known_benchmark_record_sort_col",
                    help="Use this instead of the grid header sort; checkbox edits rerun the page and reset browser-only sorting.",
                )
                sort_descending = sort_cols[1].checkbox(
                    "Descending",
                    value=False,
                    key="known_benchmark_record_sort_descending",
                )
                ordered_subset = subset.copy()
                if sort_col in ordered_subset.columns:
                    sort_values = ordered_subset[sort_col]
                    if not pd.api.types.is_numeric_dtype(sort_values) and not pd.api.types.is_bool_dtype(sort_values):
                        ordered_subset = ordered_subset.assign(_record_sort_value=sort_values.astype(str).str.lower())
                        sort_by = "_record_sort_value"
                    else:
                        sort_by = sort_col
                    ordered_subset = ordered_subset.sort_values(
                        by=sort_by,
                        ascending=not sort_descending,
                        kind="mergesort",
                        na_position="last",
                    ).drop(columns=["_record_sort_value"], errors="ignore")
                editor_df = ordered_subset[display_cols].copy()
                editor_df.insert(0, "selected", editor_df["binder_id"].isin(selected_ids))
                previous_selected_ids = set(selected_ids)
                edited_df = st.data_editor(
                    editor_df,
                    width="stretch",
                    height=520,
                    hide_index=True,
                    disabled=[col for col in editor_df.columns if col != "selected"],
                    column_config={
                        "selected": st.column_config.CheckboxColumn("Select", default=False),
                        "binder": st.column_config.CheckboxColumn("Known binder"),
                    },
                    key=(
                        f"known_benchmark_record_editor_"
                        f"{int(st.session_state.get('known_benchmark_editor_version', 0))}_"
                        f"{_safe_sort_token(sort_col)}_{int(sort_descending)}"
                    ),
                )
                selected_ids = set(
                    edited_df.loc[edited_df["selected"].fillna(False), "binder_id"].astype(str)
                )
                st.session_state[selected_ids_key] = selected_ids
                if selected_ids != previous_selected_ids:
                    st.session_state["known_benchmark_editor_version"] = (
                        int(st.session_state.get("known_benchmark_editor_version", 0)) + 1
                    )
                    st.rerun()
                selected_subset = subset[subset["binder_id"].isin(selected_ids)].copy()
                selected_known_count = len(selected_subset)
                st.info(
                    f"{selected_known_count:,} of {len(subset):,} filtered records selected. "
                    "Only checked rows will be included in the benchmark."
                )
                if selected_known_count:
                    known_csv_text = selected_subset.to_csv(index=False)
                    known_pdb_dir = KNOWN_BENCHMARK_PDB_DIR
                    runtime_size = _dataset_runtime_size(selected_subset)

    elif dataset_source == "Repo-format upload/path":
        st.caption("Use a repo-format dataset containing `input.csv` and `input_pdbs/*.pdb`.")
        repo_zip = st.file_uploader("Repo-format ZIP", type=["zip"], help="ZIP may contain input.csv and input_pdbs/*.pdb")
        repo_zip_path = st.text_input("Or ZIP path", key="repo_zip_path")
        repo_input_csv = st.file_uploader("Optional input.csv", type=["csv"], key="repo_input_csv")
        repo_input_csv_path = st.text_input("Or input.csv path", key="repo_input_csv_path")
        repo_pdb_dir = st.text_input("Input PDB folder path", placeholder="/path/to/input_pdbs")
        repo_csv_text = repo_input_csv.getvalue().decode("utf-8-sig", errors="replace") if repo_input_csv is not None else None
        repo_zip_bytes = repo_zip.getvalue() if repo_zip is not None else None
        preview_df = None
        if repo_csv_text:
            try:
                preview_df = pd.read_csv(StringIO(repo_csv_text))
            except Exception:
                preview_df = None
        elif repo_input_csv_path.strip() and Path(repo_input_csv_path).expanduser().exists():
            try:
                preview_df = pd.read_csv(Path(repo_input_csv_path).expanduser())
            except Exception:
                preview_df = None
        runtime_size = _dataset_runtime_size(preview_df, fallback_count=repo_max_records)

    elif dataset_source == "Simple ESMFold2 CSV":
        uploaded_csv = st.file_uploader("Benchmark CSV", type=["csv"], key="esmfold2_csv_upload")
        simple_csv_path = st.text_input("Or CSV path on this machine", placeholder="/path/to/benchmark.csv", key="esmfold2_csv_path")
        if uploaded_csv is not None:
            simple_csv_text = uploaded_csv.getvalue().decode("utf-8-sig", errors="replace")
            try:
                st.dataframe(pd.read_csv(StringIO(simple_csv_text)).head(20), width="stretch", hide_index=True)
            except Exception as exc:
                st.warning(f"Could not preview uploaded CSV: {exc}")
        elif simple_csv_path.strip():
            path = Path(simple_csv_path).expanduser()
            if path.exists():
                try:
                    simple_preview = pd.read_csv(path)
                    st.dataframe(simple_preview.head(20), width="stretch", hide_index=True)
                    runtime_size = _dataset_runtime_size(simple_preview.head(int(simple_max_records)))
                except Exception as exc:
                    st.warning(f"Could not preview CSV: {exc}")
            else:
                st.warning("CSV path does not exist.")
        if simple_csv_text:
            try:
                runtime_size = _dataset_runtime_size(pd.read_csv(StringIO(simple_csv_text)).head(int(simple_max_records)))
            except Exception:
                runtime_size = {"candidate_count": int(simple_max_records), "total_residues": None}
        with st.expander("Accepted simple CSV columns", expanded=False):
            st.markdown(
                "Use one row per candidate. Accepted columns include `candidate_id`, `target_pdb`, "
                "`complex_pdb`, `binder_pdb`, `binder_sequence`, `target_chains`, `binder_chains`, "
                "`hotspots`, `label`, `source`, and `target_id`."
            )

    else:
        use_published_metrics = st.checkbox(
            "Use included published prepared_training_dataset.csv",
            value=True,
            help=str(PUBLISHED_DATASET),
        )
        metric_upload = st.file_uploader("Metric CSV", type=["csv"], disabled=use_published_metrics, key="metric_csv_upload")
        metric_path = st.text_input("Or metric CSV path", disabled=use_published_metrics, key="metric_csv_path")
        if use_published_metrics and PUBLISHED_DATASET.exists():
            st.dataframe(pd.read_csv(PUBLISHED_DATASET, nrows=20), width="stretch", hide_index=True)
        elif metric_upload is not None:
            metric_text = metric_upload.getvalue().decode("utf-8-sig", errors="replace")
            st.dataframe(pd.read_csv(StringIO(metric_text)).head(20), width="stretch", hide_index=True)
        elif metric_path.strip() and Path(metric_path).expanduser().exists():
            st.dataframe(pd.read_csv(Path(metric_path).expanduser()).head(20), width="stretch", hide_index=True)

with tabs[1]:
    st.subheader("Prediction / Refolding Engines")
    st.caption("These generate structures or confidence outputs. Shared interface metrics are configured separately.")

    engine_cols = st.columns(8)
    with engine_cols[0]:
        run_alphafast_af3 = st.checkbox("AlphaFast AF3", value=True, disabled=dataset_source in {"Simple ESMFold2 CSV", "Metric table only"})
    with engine_cols[1]:
        run_colabfold = st.checkbox("ColabFold", value=True, disabled=dataset_source in {"Simple ESMFold2 CSV", "Metric table only"})
    with engine_cols[2]:
        run_repo_af2ig = st.checkbox("AF2 initial guess", value=True, disabled=dataset_source in {"Simple ESMFold2 CSV", "Metric table only"})
    with engine_cols[3]:
        run_repo_esm = st.checkbox("ESMFold2", value=True, disabled=dataset_source == "Metric table only")
    with engine_cols[4]:
        run_boltz2_ig = st.checkbox("Boltz-2", value=True, disabled=dataset_source in {"Simple ESMFold2 CSV", "Metric table only"})
    with engine_cols[5]:
        run_rf3 = st.checkbox("RF3", value=False, disabled=dataset_source in {"Simple ESMFold2 CSV", "Metric table only"})
    with engine_cols[6]:
        run_protenix = st.checkbox("Protenix", value=False, disabled=dataset_source in {"Simple ESMFold2 CSV", "Metric table only"})
    with engine_cols[7]:
        run_boltzgen_fold = st.checkbox(
            "BoltzGen fold",
            value=False,
            disabled=dataset_source in {"Simple ESMFold2 CSV", "Metric table only"},
            help="Runs BoltzGen's target-template-conditioned folding stage. This is not MSA-based sequence-only refolding.",
        )

    if dataset_source not in {"Simple ESMFold2 CSV", "Metric table only"}:
        repo_mode = st.segmented_control("Input mode", ["pdb_only", "hybrid", "seq_only_csv"], selection_mode="single", default="pdb_only")
        selected_record_count = int(runtime_size.get("candidate_count") or 0)
        if selected_record_count:
            st.caption(f"Prediction engines will process all {selected_record_count:,} checked records.")
        generate_inputs = True

        with st.expander("Compute", expanded=True):
            gpu_device = st.number_input(
                "GPU device",
                min_value=0,
                max_value=15,
                value=0,
                step=1,
                key="benchmark_gpu_device",
                help="Used by GPU-backed benchmark engines that expose device selection, currently AlphaFast AF3 and ColabFold.",
            )

        with st.expander("MSA Reference Data", expanded=True):
            msa_consuming_engine_selected = bool(
                run_alphafast_af3
                or run_colabfold
                or (run_boltz2_ig and boltz2_use_target_msa)
                or (run_repo_esm and esmfold2_use_target_msa)
                or (run_rf3 and rf3_use_target_msa)
                or (run_protenix and protenix_use_msa)
            )
            msa_cols = st.columns(4)
            colabfold_msa_source = msa_cols[0].segmented_control(
                "Shared target MSA source",
                ["msa_repository_then_alphafast_mmseqs_gpu", "msa_repository", "alphafast_mmseqs_gpu", "repo_run_csv"],
                selection_mode="single",
                default="msa_repository_then_alphafast_mmseqs_gpu",
                disabled=not msa_consuming_engine_selected,
                format_func={
                    "msa_repository_then_alphafast_mmseqs_gpu": "Repository, then AlphaFast/MMseqs",
                    "msa_repository": "Repository only",
                    "alphafast_mmseqs_gpu": "AlphaFast/MMseqs GPU",
                    "repo_run_csv": "Existing run.csv paths",
                }.get,
                help="Prepared once before engine input generation. AF3, ColabFold, Boltz-style inputs, and ESMFold2 can adapt these target-chain MSAs; binder chains stay no-MSA.",
            )
            msa_repository_dir = msa_cols[1].text_input(
                "MSA repository",
                value=str(MSA_REPOSITORY_DIR),
                help="Shared target-chain MSA cache. Used first when repository-backed MSA mode is selected.",
            )
            alphafast_db_dir = msa_cols[2].text_input(
                "Alignment/MMseqs DB dir",
                value=str(ALPHAFAST_DB_DIR),
                help="AlphaFast database root containing the MMseqs alignment databases.",
            )
            alphafast_batch_size = msa_cols[3].number_input(
                "AlphaFast/MMseqs batch size",
                min_value=0,
                max_value=10000,
                value=0,
                step=1,
                key="benchmark_alphafast_batch_size",
                disabled=not msa_consuming_engine_selected,
                help=(
                    "0 means auto/default. This controls AlphaFast data-pipeline batching for target MSA/template "
                    "preparation; it is not the number of AF3 structures predicted at once."
                ),
            )
        default_models = []
        if run_alphafast_af3 or run_colabfold:
            default_models.append("af3")
        if run_boltz2_ig:
            default_models.append("boltz")
        if run_colabfold:
            default_models.append("colabfold")
        repo_models = default_models or ["af3", "boltz", "colabfold"]

        with st.expander("AlphaFast AF3 Settings", expanded=run_alphafast_af3):
            af3_cols = st.columns(2)
            alphafast_weights_dir = af3_cols[0].text_input("AF3 weights dir", value=str(ALPHAFAST_WEIGHTS_DIR), disabled=not run_alphafast_af3)
            alphafast_recycles = af3_cols[1].number_input("AF3 recycles", min_value=1, max_value=48, value=10, step=1, key="benchmark_alphafast_recycles", disabled=not run_alphafast_af3)

        with st.expander("ColabFold Settings", expanded=run_colabfold):
            colab_cols = st.columns(5)
            colabfold_cache_dir = colab_cols[0].text_input("ColabFold / AF2 model cache", value=str(COLABFOLD_CACHE_DIR), disabled=not run_colabfold)
            colabfold_recycles = colab_cols[1].number_input("ColabFold recycles", min_value=1, max_value=48, value=3, step=1, key="benchmark_colabfold_recycles", disabled=not run_colabfold)
            colabfold_models = colab_cols[2].number_input("ColabFold models", min_value=1, max_value=5, value=3, step=1, key="benchmark_colabfold_models", disabled=not run_colabfold)
            colabfold_use_target_templates = colab_cols[3].checkbox(
                "Use target PDB templates",
                value=colabfold_use_target_templates,
                key="benchmark_colabfold_use_target_templates",
                disabled=not run_colabfold,
                help="Passes target-chain-only PDB templates to ColabFold. Binder chains and binder-interface geometry are not templated.",
            )
            colabfold_max_template_hits = colab_cols[4].number_input(
                "Max template hits",
                min_value=1,
                max_value=20,
                value=4,
                step=1,
                key="benchmark_colabfold_max_template_hits",
                disabled=not run_colabfold or not colabfold_use_target_templates,
            )

        with st.expander("AF2 Initial Guess Settings", expanded=run_repo_af2ig):
            af2_cols = st.columns(4)
            af2_recycles = af2_cols[0].number_input("AF2-IG recycles", min_value=1, max_value=24, value=3, step=1, key="benchmark_af2ig_recycles", disabled=not run_repo_af2ig)
            af2_multimer = af2_cols[1].checkbox("AF2 multimer", value=True, disabled=not run_repo_af2ig)
            af2_binder_template = af2_cols[2].checkbox("Binder template", value=False, disabled=not run_repo_af2ig)
            af2_interface_template = af2_cols[3].checkbox("Interface template", value=False, disabled=not run_repo_af2ig or not af2_binder_template)

        with st.expander("ESMFold2 Settings", expanded=run_repo_esm):
            repo_esm_modes = st.multiselect(
                "Modes",
                ["sequence", "initial_guess"],
                default=["initial_guess"],
                disabled=not run_repo_esm,
                format_func={"sequence": "Sequence only", "initial_guess": "Initial guess"}.get,
            )
            esm_cols = st.columns(5)
            esm_cols[0].text_input("ESMFold2 model dir", value=str(ESMFOLD2_MODEL_DIR), disabled=True)
            esmfold2_use_target_msa = esm_cols[1].checkbox(
                "Use target MSAs",
                value=False,
                key="benchmark_esmfold2_use_target_msa",
                disabled=not run_repo_esm,
                help="Default off. Enable to pass prepared per-target-chain A3M files into ESMFold2 ProteinInput objects.",
            )
            repo_sampling = esm_cols[2].number_input("ESMFold2 sampling steps", min_value=1, max_value=256, value=32, step=1, key="benchmark_esmfold2_sampling_steps", disabled=not run_repo_esm)
            repo_loops = esm_cols[3].number_input("ESMFold2 recycling loops", min_value=1, max_value=16, value=3, step=1, key="benchmark_esmfold2_recycling_loops", disabled=not run_repo_esm)
            repo_seed = esm_cols[4].number_input("ESMFold2 seed", min_value=0, max_value=999999, value=0, step=1, key="benchmark_esmfold2_seed", disabled=not run_repo_esm)

        with st.expander("Boltz-2 Settings", expanded=run_boltz2_ig):
            boltz_cols = st.columns(6)
            boltz_cols[0].text_input("Boltz-2 model cache", value=str(BOLTZ_MODELS_DIR), disabled=True)
            boltz2_target_template = boltz_cols[1].checkbox("Use target template", value=True, disabled=not run_boltz2_ig)
            boltz2_use_target_msa = boltz_cols[2].checkbox(
                "Use target MSAs",
                value=True,
                key="benchmark_boltz2_use_target_msa",
                disabled=not run_boltz2_ig,
                help="Default on. Binder chains stay no-MSA; each declared target chain uses its prepared MSA when available.",
            )
            boltz2_recycling_steps = boltz_cols[3].number_input("Recycling steps", min_value=1, max_value=48, value=10, step=1, key="benchmark_boltz2_recycling_steps", disabled=not run_boltz2_ig)
            boltz2_sampling_steps = boltz_cols[4].number_input("Sampling steps", min_value=1, max_value=1000, value=200, step=1, key="benchmark_boltz2_sampling_steps", disabled=not run_boltz2_ig)
            boltz2_diffusion_samples = boltz_cols[5].number_input("Diffusion samples", min_value=1, max_value=20, value=3, step=1, key="benchmark_boltz2_diffusion_samples", disabled=not run_boltz2_ig)
            boltz2_write_full_pae = st.checkbox("Write full PAE", value=True, disabled=not run_boltz2_ig)

        with st.expander("RF3 Settings", expanded=run_rf3):
            rf3_cols = st.columns(6)
            rf3_checkpoint_path = rf3_cols[0].text_input("RF3 checkpoint", value=str(RF3_CHECKPOINT), disabled=not run_rf3)
            rf3_use_target_msa = rf3_cols[1].checkbox(
                "Use target MSAs",
                value=True,
                key="benchmark_rf3_use_target_msa",
                disabled=not run_rf3,
                help="Default on. Each declared target chain receives its prepared A3M; binder chains remain MSA-free.",
            )
            rf3_recycles = rf3_cols[2].number_input("RF3 recycles", min_value=1, max_value=48, value=10, step=1, key="benchmark_rf3_recycles", disabled=not run_rf3)
            rf3_num_steps = rf3_cols[3].number_input("RF3 diffusion steps", min_value=1, max_value=1000, value=50, step=1, key="benchmark_rf3_num_steps", disabled=not run_rf3)
            rf3_diffusion_batch_size = rf3_cols[4].number_input("RF3 samples", min_value=1, max_value=20, value=5, step=1, key="benchmark_rf3_diffusion_batch_size", disabled=not run_rf3)
            rf3_seed = rf3_cols[5].number_input("RF3 seed", min_value=0, max_value=999999, value=0, step=1, key="benchmark_rf3_seed", disabled=not run_rf3)
            st.caption("RF3 folds the complex from sequences and per-chain target MSAs. These controls map to RF3 Hydra settings: n_recycles, num_steps, diffusion_batch_size, and seed.")

        with st.expander("Protenix Settings", expanded=run_protenix):
            protenix_cols = st.columns(4)
            protenix_use_msa = protenix_cols[0].checkbox(
                "Use target MSAs",
                value=True,
                key="benchmark_protenix_use_msa",
                disabled=not run_protenix,
                help=(
                    "Default on. Reuses the app's prepared target-chain A3Ms, packaged as Protenix pairing and "
                    "non-pairing MSA directories. Binder chains receive query-only single-sequence MSA files so "
                    "Protenix stays fully local and does not launch its online MSA search."
                ),
            )
            protenix_cycle = protenix_cols[1].number_input("Pairformer cycles", min_value=1, max_value=48, value=3, step=1, key="benchmark_protenix_cycle", disabled=not run_protenix)
            protenix_diffusion_steps = protenix_cols[2].number_input("Diffusion steps", min_value=1, max_value=1000, value=50, step=1, key="benchmark_protenix_diffusion_steps", disabled=not run_protenix)
            protenix_samples = protenix_cols[3].number_input("Samples", min_value=1, max_value=20, value=5, step=1, key="benchmark_protenix_samples", disabled=not run_protenix)

        with st.expander("BoltzGen Fold Settings", expanded=run_boltzgen_fold):
            boltzgen_cols = st.columns(3)
            boltzgen_recycling_steps = boltzgen_cols[0].number_input("Recycling steps", min_value=1, max_value=48, value=3, step=1, key="benchmark_boltzgen_recycling_steps", disabled=not run_boltzgen_fold)
            boltzgen_sampling_steps = boltzgen_cols[1].number_input("Sampling steps", min_value=1, max_value=1000, value=200, step=1, key="benchmark_boltzgen_sampling_steps", disabled=not run_boltzgen_fold)
            boltzgen_diffusion_samples = boltzgen_cols[2].number_input("Diffusion samples", min_value=1, max_value=20, value=5, step=1, key="benchmark_boltzgen_diffusion_samples", disabled=not run_boltzgen_fold)
            st.caption("BoltzGen uses the declared target chains as a structural template and predicts the designed binder region. It does not use an MSA in this stage.")

    if dataset_source == "Simple ESMFold2 CSV":
        with st.expander("ESMFold2 Settings", expanded=True):
            simple_modes = st.multiselect(
                "Modes",
                ["sequence", "initial_guess"],
                default=["sequence", "initial_guess"],
                format_func={"sequence": "Sequence only", "initial_guess": "Initial guess"}.get,
            )
            simple_cols = st.columns(6)
            simple_max_records = simple_cols[0].number_input(
                "Max records",
                min_value=0,
                max_value=10000,
                value=0,
                step=1,
                key="benchmark_simple_max_records",
                help="0 runs every record in the input CSV.",
            )
            simple_sampling_steps = simple_cols[1].number_input("Sampling steps", min_value=1, max_value=256, value=32, step=1, key="benchmark_simple_sampling_steps")
            simple_recycling_loops = simple_cols[2].number_input("Recycling loops", min_value=1, max_value=16, value=3, step=1, key="benchmark_simple_recycling_loops")
            simple_seed = simple_cols[3].number_input("Seed", min_value=0, max_value=999999, value=0, step=1, key="benchmark_simple_seed")
            simple_contact_cutoff = simple_cols[4].number_input("Contact cutoff", min_value=2.0, max_value=20.0, value=8.0, step=0.5, key="benchmark_simple_contact_cutoff")
            simple_device = simple_cols[5].selectbox("Device", ["cuda", "auto", "cpu"], index=0)

with tabs[2]:
    st.subheader("Metrics")
    if dataset_source == "Metric table only":
        metric_cols = st.columns(3)
        label_column = metric_cols[0].text_input("Label column", value="binder")
        metric_max_rows = metric_cols[1].number_input("Max rows", min_value=0, max_value=1000000, value=0, step=100, key="benchmark_metric_max_rows", help="0 means all rows.")
        metric_max_cols = metric_cols[2].number_input("Max numeric columns", min_value=1, max_value=10000, value=500, step=50, key="benchmark_metric_max_cols")
    else:
        st.caption("Common metrics are calculated from available engine outputs whenever the workflow has the required structure and confidence files.")
        metric_cols = st.columns(4)
        run_common_interface_metrics = metric_cols[0].checkbox(
            "ipSAE / iPAE",
            value=True,
            help="Runs the repo batch ipSAE/LIS/pDockQ script for AF3, ColabFold, and Boltz-style outputs when present.",
        )
        metric_cols[1].checkbox("LIS", value=True, disabled=True)
        metric_cols[2].checkbox("pDockQ / pDockQ2", value=True, disabled=True)
        metric_cols[3].checkbox("Native confidence", value=True, disabled=True)
        extra_cols = st.columns(4)
        run_pyrosetta_input = extra_cols[0].checkbox(
            "Rosetta input metrics",
            value=True,
            disabled=dataset_source in {"Simple ESMFold2 CSV", "Metric table only"},
            help="Runs PyRosetta interface metrics for the original input structures.",
        )
        run_predicted_rosetta_metrics = extra_cols[1].checkbox(
            "Rosetta predicted metrics",
            value=True,
            disabled=dataset_source in {"Simple ESMFold2 CSV", "Metric table only"},
            help="Stages available predicted structures as PDBs, then relaxes and scores them with PyRosetta.",
        )
        pyrosetta_nprocs = extra_cols[2].number_input(
            "PyRosetta processes",
            min_value=1,
            max_value=64,
            value=default_pyrosetta_nprocs,
            step=1,
            key="benchmark_pyrosetta_nprocs",
            disabled=not (run_pyrosetta_input or run_predicted_rosetta_metrics),
            help="Used for all enabled Rosetta feature calculations.",
        )
        run_pymol_metrics = extra_cols[3].checkbox(
            "PyMOL metrics",
            value=True,
            disabled=dataset_source in {"Simple ESMFold2 CSV", "Metric table only"},
            help="Calculates SASA, interface residue, hydrogen-bond, and secondary-structure metrics for input and predicted PDB folders.",
        )

with tabs[3]:
    st.subheader("Run")
    selected_engines = []
    if run_alphafast_af3:
        selected_engines.append("AlphaFast AF3")
    if run_colabfold:
        selected_engines.append("ColabFold")
    if run_repo_af2ig:
        selected_engines.append("AF2 initial guess")
    if run_boltz2_ig:
        selected_engines.append("Boltz-2")
    if run_repo_esm or dataset_source == "Simple ESMFold2 CSV":
        selected_engines.append("ESMFold2")
    if run_rf3:
        selected_engines.append("RF3")
    if run_protenix:
        selected_engines.append("Protenix")
    if run_boltzgen_fold:
        selected_engines.append("BoltzGen fold")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Dataset", str(dataset_source))
    summary_cols[1].metric("Engines", len(selected_engines))
    summary_cols[2].metric("Model inputs", "yes" if generate_inputs and dataset_source not in {"Simple ESMFold2 CSV", "Metric table only"} else "n/a")
    summary_cols[3].metric("Rosetta", "yes" if run_pyrosetta_input else "no")
    st.write(", ".join(selected_engines) if selected_engines else "No prediction engine selected.")
    if dataset_source == "Installed Overath 2025" and selected_targets:
        selected_count = int(runtime_size.get("candidate_count") or 0)
        if selected_count:
            st.success(f"This run will process exactly the {selected_count:,} checked records.")
        else:
            st.warning("No records are checked. Select at least one record in the Dataset tab.")
    estimate_engine_keys: list[str] = []
    if run_alphafast_af3:
        estimate_engine_keys.append("alphafast_af3")
    if run_colabfold:
        estimate_engine_keys.append("colabfold")
    if run_repo_af2ig:
        estimate_engine_keys.append("af2_initial_guess")
    if run_boltz2_ig:
        estimate_engine_keys.append("boltz2_initial_guess")
    if run_repo_esm or dataset_source == "Simple ESMFold2 CSV":
        estimate_engine_keys.append("esmfold2")
    if run_rf3:
        estimate_engine_keys.append("rf3")
    if run_protenix:
        estimate_engine_keys.append("protenix")
    if run_boltzgen_fold:
        estimate_engine_keys.append("boltzgen_fold")
    if run_common_interface_metrics or run_predicted_rosetta_metrics or run_pymol_metrics:
        estimate_engine_keys.append("postprocessing")

    estimate_count = int(runtime_size.get("candidate_count") or 0)
    estimate_total_residues = runtime_size.get("total_residues")
    if estimate_engine_keys and estimate_count:
        estimate = estimate_engines(
            engines=estimate_engine_keys,
            candidate_count=estimate_count,
            total_residues=int(estimate_total_residues) if estimate_total_residues else None,
        )
        with st.expander("Runtime estimate", expanded=True):
            runtime_cols = st.columns(4)
            runtime_cols[0].metric("Estimated total", estimate["total_time"])
            runtime_cols[1].metric("Candidates", f"{estimate_count:,}")
            runtime_cols[2].metric("Residues", f"{int(estimate_total_residues):,}" if estimate_total_residues else "n/a")
            runtime_cols[3].metric("Engines/steps", len(estimate_engine_keys))
            estimate_df = _estimate_rows_dataframe(estimate)
            if not estimate_df.empty:
                st.dataframe(
                    estimate_df,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "seconds_per_candidate": st.column_config.NumberColumn("sec / candidate", format="%.1f"),
                    },
                )
            st.caption(
                "Estimates use previous completed jobs when available and fallback rates otherwise. "
                "They improve as more benchmark runs finish."
            )
    elif estimate_engine_keys:
        st.caption("Runtime estimate needs a candidate count. Select a target or provide a previewable CSV.")

    if dataset_source == "Metric table only":
        metric_disabled = not use_published_metrics and not metric_text and not metric_path.strip()
        if st.button("Run metric benchmark", type="primary", disabled=metric_disabled):
            try:
                with st.spinner("Benchmarking metric table..."):
                    run_dir = run_precomputed_metric_benchmark(
                        input_csv=Path(metric_path).expanduser() if (not use_published_metrics and not metric_text and metric_path.strip()) else None,
                        input_csv_text=metric_text,
                        use_published_dataset=bool(use_published_metrics),
                        label_column=str(label_column or "binder"),
                        max_rows=int(metric_max_rows),
                        max_columns=int(metric_max_cols),
                    )
                result = read_json(run_dir / "result.json")
                metrics = result.get("metrics") or {}
                st.success(
                    f"Metric benchmark finished | top: {metrics.get('top_feature')} | "
                    f"AP: {metrics.get('top_feature_average_precision')}"
                )
                show_pipeline_links(run_dir, [run_dir])
            except Exception as exc:
                for row in collect_jobs("benchmark"):
                    run_dir = Path(str(row.get("run_dir") or ""))
                    input_payload = read_json(run_dir / "input.json")
                    if (
                        input_payload.get("job_type") == "benchmark_collection"
                        and str(row.get("status")) in ACTIVE_STATUSES
                        and str((input_payload.get("params") or {}).get("collection_name") or "") == str(collection_name)
                    ):
                        finish_job(
                            run_dir,
                            False,
                            {
                                "metrics": {"error": str(exc)},
                                "outputs": {},
                            },
                        )
                        break
                st.error(str(exc))

    elif dataset_source == "Simple ESMFold2 CSV":
        simple_disabled = not simple_modes or (not simple_csv_text and not simple_csv_path.strip())
        if st.button("Run ESMFold2 benchmark", type="primary", disabled=simple_disabled):
            try:
                with st.spinner("Running ESMFold2 binder benchmark..."):
                    run_dir = run_esmfold2_binder_benchmark(
                        input_csv=Path(simple_csv_path).expanduser() if not simple_csv_text and simple_csv_path.strip() else None,
                        input_csv_text=simple_csv_text,
                        modes=list(simple_modes),
                        max_records=int(simple_max_records),
                        num_loops=int(simple_recycling_loops),
                        num_sampling_steps=int(simple_sampling_steps),
                        seed=int(simple_seed),
                        device=str(simple_device or "cuda"),
                        contact_cutoff=float(simple_contact_cutoff),
                        use_docker=True,
                    )
                result = read_json(run_dir / "result.json")
                metrics = result.get("metrics") or {}
                st.success(f"Benchmark finished | records: {metrics.get('record_count')} | AUROC: {metrics.get('auroc')}")
                show_pipeline_links(run_dir, [run_dir])
            except Exception as exc:
                st.error(str(exc))

    else:
        repo_disabled = (
            not known_csv_text
            and repo_zip is None
            and not repo_zip_path.strip()
            and repo_input_csv is None
            and not repo_input_csv_path.strip()
            and not repo_pdb_dir.strip()
        )
        if st.button("Run benchmark", type="primary", disabled=repo_disabled):
            try:
                run_dir = enqueue_de_novo_binder_scoring_dataset(
                    input_zip=Path(repo_zip_path).expanduser() if repo_zip is None and repo_zip_path.strip() else None,
                    input_zip_bytes=repo_zip_bytes,
                    input_csv=Path(repo_input_csv_path).expanduser() if repo_input_csv is None and repo_input_csv_path.strip() else None,
                    input_csv_text=known_csv_text or repo_csv_text,
                    input_pdb_dir=known_pdb_dir or (Path(repo_pdb_dir).expanduser() if repo_pdb_dir.strip() else None),
                    mode=str(repo_mode or "pdb_only"),
                    generate_inputs=bool(generate_inputs),
                    models=list(repo_models),
                    run_pyrosetta_input_metrics=bool(run_pyrosetta_input),
                    pyrosetta_nprocs=int(pyrosetta_nprocs),
                    run_common_interface_metrics=bool(run_common_interface_metrics),
                    run_predicted_rosetta_metrics=bool(run_predicted_rosetta_metrics),
                    run_pymol_metrics=bool(run_pymol_metrics),
                    run_esmfold2=bool(run_repo_esm),
                    esmfold2_modes=list(repo_esm_modes),
                    esmfold2_use_target_msa=bool(esmfold2_use_target_msa),
                    run_af2_initial_guess=bool(run_repo_af2ig),
                    af2_num_recycles=int(af2_recycles),
                    af2_multimer=bool(af2_multimer),
                    af2_use_binder_template=bool(af2_binder_template),
                    af2_use_interface_template=bool(af2_interface_template and af2_binder_template),
                    run_boltz2_initial_guess=bool(run_boltz2_ig),
                    boltz2_use_target_template=bool(boltz2_target_template),
                    boltz2_use_target_msa=bool(boltz2_use_target_msa),
                    boltz2_recycling_steps=int(boltz2_recycling_steps),
                    boltz2_sampling_steps=int(boltz2_sampling_steps),
                    boltz2_diffusion_samples=int(boltz2_diffusion_samples),
                    boltz2_write_full_pae=bool(boltz2_write_full_pae),
                    run_rf3=bool(run_rf3),
                    rf3_checkpoint_path=Path(rf3_checkpoint_path).expanduser(),
                    rf3_use_target_msa=bool(rf3_use_target_msa),
                    rf3_recycles=int(rf3_recycles),
                    rf3_num_steps=int(rf3_num_steps),
                    rf3_diffusion_batch_size=int(rf3_diffusion_batch_size),
                    rf3_seed=int(rf3_seed),
                    run_protenix=bool(run_protenix),
                    protenix_use_msa=bool(protenix_use_msa),
                    protenix_cycle=int(protenix_cycle),
                    protenix_diffusion_steps=int(protenix_diffusion_steps),
                    protenix_samples=int(protenix_samples),
                    run_boltzgen_fold=bool(run_boltzgen_fold),
                    boltzgen_recycling_steps=int(boltzgen_recycling_steps),
                    boltzgen_sampling_steps=int(boltzgen_sampling_steps),
                    boltzgen_diffusion_samples=int(boltzgen_diffusion_samples),
                    run_colabfold=bool(run_colabfold),
                    colabfold_cache_dir=Path(colabfold_cache_dir).expanduser(),
                    colabfold_msa_source=str(colabfold_msa_source or "msa_repository_then_alphafast_mmseqs_gpu"),
                    msa_repository_dir=Path(msa_repository_dir).expanduser(),
                    colabfold_num_recycles=int(colabfold_recycles),
                    colabfold_num_models=int(colabfold_models),
                    colabfold_use_target_templates=bool(colabfold_use_target_templates),
                    colabfold_max_template_hits=int(colabfold_max_template_hits),
                    colabfold_gpu_device=int(gpu_device),
                    run_alphafast_af3=bool(run_alphafast_af3),
                    alphafast_db_dir=Path(alphafast_db_dir).expanduser(),
                    alphafast_weights_dir=Path(alphafast_weights_dir).expanduser(),
                    alphafast_batch_size=int(alphafast_batch_size),
                    alphafast_num_recycles=int(alphafast_recycles),
                    alphafast_gpu_device=int(gpu_device),
                    max_records=int(repo_max_records),
                    num_loops=int(repo_loops),
                    num_sampling_steps=int(repo_sampling),
                    seed=int(repo_seed),
                    device="cuda",
                )
                spawn_worker_for_run(run_dir)
                st.success("Benchmark queued. The local worker will keep running even if you navigate away from this page.")
                show_pipeline_links(run_dir, [run_dir])
            except Exception as exc:
                st.error(str(exc))

with tabs[4]:
    st.subheader("Results")
    rows = collect_jobs("benchmark")
    if not rows:
        st.info("No benchmark jobs yet.")
    else:
        df = _benchmark_job_table(rows)
        display_cols = [
            "result",
            "kind",
            "description",
            "status",
            "records",
            "top_feature",
            "top_ap",
            "plots",
            "created_at",
            "job_code",
            "run_id",
        ]
        display_df = df[[col for col in display_cols if col in df.columns]].copy()
        table_key = "benchmark_results"
        event = st.dataframe(
            display_df,
            width="stretch",
            hide_index=True,
            key=f"{table_key}_jobs_table",
            on_select="rerun",
            selection_mode="multi-row",
            column_config={
                "result": st.column_config.LinkColumn("result", display_text="Open"),
                "top_ap": st.column_config.NumberColumn("top AP", format="%.3f"),
            },
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
        selected_rows = display_df.iloc[selected_indices].copy() if selected_indices else display_df.iloc[0:0].copy()
        active_selected = selected_rows[selected_rows["status"].isin(ACTIVE_STATUSES)]
        if not active_selected.empty:
            st.warning("Running, queued, or preparing jobs cannot be deleted.")
        selected_rows = selected_rows[~selected_rows["status"].isin(ACTIVE_STATUSES)]
        selected_refs = [
            ("benchmark", str(row["run_id"]))
            for row in selected_rows.to_dict(orient="records")
        ]
        selected_refs_key = f"{table_key}_selected_delete_refs"
        if selected_refs:
            st.session_state[selected_refs_key] = selected_refs
        cached_selected_refs = st.session_state.get(selected_refs_key) or []
        delete_clicked = st.button(
            "Delete selected benchmark jobs",
            type="primary",
            disabled=not cached_selected_refs,
            key=f"{table_key}_request_delete_jobs",
        )
        if delete_clicked and cached_selected_refs:
            show_delete_jobs_dialog(
                table_key=table_key,
                pending_refs=cached_selected_refs,
                selected_refs_key=selected_refs_key,
                label="benchmark job",
            )
        else:
            st.caption("Select finished benchmark rows in the table to enable deletion.")

with tabs[5]:
    st.subheader("Benchmark Collections")
    st.caption(
        "Combine engine outputs from multiple completed benchmark jobs without modifying the original runs. "
        "This is useful for rerunning one engine and comparing it with earlier engines on the same records."
    )
    rows = collect_jobs("benchmark")
    source_rows: list[dict] = []
    unavailable_rows: list[dict] = []
    for row in rows:
        run_dir = Path(str(row.get("run_dir") or ""))
        input_payload = read_json(run_dir / "input.json")
        if input_payload.get("job_type") == "benchmark_collection":
            continue
        metrics_path = run_dir / "artifacts" / "benchmark" / "merged_benchmark_metrics.csv"
        unavailable_reason = ""
        if str(row.get("status")) != "completed":
            unavailable_reason = f"status: {row.get('status')}"
        elif not metrics_path.exists():
            unavailable_reason = "missing merged_benchmark_metrics.csv"
        if unavailable_reason:
            benchmark_dir = run_dir / "artifacts" / "benchmark"
            partial_metrics = sorted(path.name for path in benchmark_dir.glob("*_metrics.csv")) if benchmark_dir.exists() else []
            unavailable_rows.append(
                {
                    "job_code": str(row.get("job_code")),
                    "description": " | ".join(
                        part
                        for part in [
                            _job_dataset_label(input_payload),
                            _enabled_engines(input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}),
                        ]
                        if part
                    ),
                    "status": row.get("status"),
                    "current_phase": row.get("current_phase"),
                    "current_engine": row.get("current_engine"),
                    "reason": unavailable_reason,
                    "partial_metric_tables": ", ".join(partial_metrics[:5]),
                    "created_at": str(row.get("created_at") or ""),
                    "run_id": str(row.get("run_id")),
                }
            )
            continue
        engines_available = benchmark_engines_in_metrics(metrics_path)
        if not engines_available:
            unavailable_rows.append(
                {
                    "job_code": str(row.get("job_code")),
                    "description": " | ".join(
                        part
                        for part in [
                            _job_dataset_label(input_payload),
                            _enabled_engines(input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}),
                        ]
                        if part
                    ),
                    "status": row.get("status"),
                    "current_phase": row.get("current_phase"),
                    "current_engine": row.get("current_engine"),
                    "reason": "merged metrics exist, but no recognized engine columns",
                    "partial_metric_tables": "",
                    "created_at": str(row.get("created_at") or ""),
                    "run_id": str(row.get("run_id")),
                }
            )
            continue
        result = read_json(run_dir / "result.json")
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        source_rows.append(
            {
                "run_id": str(row.get("run_id")),
                "job_code": str(row.get("job_code")),
                "created_at": str(row.get("created_at") or ""),
                "description": " | ".join(
                    part
                    for part in [
                        _job_dataset_label(input_payload),
                        _enabled_engines(input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}),
                        f"{metrics.get('record_count')} rows" if metrics.get("record_count") else "",
                    ]
                    if part
                ),
                "engines_available": engines_available,
                "engine_text": ", ".join(engines_available),
                "metrics_path": metrics_path,
            }
        )
    if not source_rows:
        st.info("No completed benchmark runs with merged metrics are available yet.")
        if unavailable_rows:
            with st.expander("Recent runs not ready for collections", expanded=True):
                st.caption("A run becomes selectable once it is completed and has `artifacts/benchmark/merged_benchmark_metrics.csv`.")
                st.dataframe(pd.DataFrame(unavailable_rows).head(20), hide_index=True, width="stretch")
    else:
        source_df = pd.DataFrame(
            [
                {
                    "job_code": row["job_code"],
                    "description": row["description"],
                    "engines": row["engine_text"],
                    "created_at": row["created_at"],
                    "run_id": row["run_id"],
                }
                for row in source_rows
            ]
        )
        with st.expander("Available source benchmark runs", expanded=True):
            st.dataframe(source_df, hide_index=True, width="stretch")
        if unavailable_rows:
            with st.expander("Recent runs not ready for collections", expanded=False):
                st.caption("A run becomes selectable once it is completed and has `artifacts/benchmark/merged_benchmark_metrics.csv`.")
                st.dataframe(pd.DataFrame(unavailable_rows).head(20), hide_index=True, width="stretch")

        run_label_by_id = {
            row["run_id"]: f"{row['job_code']} | {row['description']} | {row['run_id']}"
            for row in source_rows
        }
        selected_source_ids = st.multiselect(
            "Source runs, in replacement order",
            [row["run_id"] for row in source_rows],
            default=[],
            format_func=run_label_by_id.get,
            help="If the same feature column is selected from more than one run, the later selected run replaces the earlier value.",
        )
        selected_source_rows = [row for run_id in selected_source_ids for row in source_rows if row["run_id"] == run_id]
        target_options: list[str] = []
        for row in selected_source_rows:
            try:
                df_targets = pd.read_csv(row["metrics_path"], usecols=lambda col: col == "target_id")
            except Exception:
                continue
            if "target_id" in df_targets.columns:
                target_options.extend(str(value) for value in df_targets["target_id"].dropna().unique())
        target_options = sorted({target for target in target_options if target.strip() and target.lower() != "nan"})
        collection_targets = st.multiselect(
            "Targets to include",
            target_options,
            default=[],
            help="Leave empty to include all targets present in the selected source runs.",
            disabled=not bool(selected_source_rows),
        )
        include_input_columns = st.checkbox(
            "Allow input/reference metrics as a selectable source",
            value=False,
            help="Usually off. Enable only when you want input/reference metrics included in the collection ranking.",
            disabled=not bool(selected_source_rows),
        )
        collection_name = st.text_input(
            "Collection name",
            value="Benchmark collection",
            disabled=not bool(selected_source_rows),
        )

        collection_selections: list[dict[str, object]] = []
        if selected_source_rows:
            st.markdown("**Engine Sources**")
        for index, row in enumerate(selected_source_rows, start=1):
            engine_options = [
                engine
                for engine in row["engines_available"]
                if include_input_columns or engine != "Input"
            ]
            default_engines = [engine for engine in engine_options if engine != "Input"]
            selected_engines = st.multiselect(
                f"{index}. {row['job_code']} engines",
                engine_options,
                default=default_engines,
                key=f"benchmark_collection_engines_{row['run_id']}",
                help=f"Source run: {row['run_id']}",
            )
            if selected_engines:
                collection_selections.append(
                    {
                        "run_id": row["run_id"],
                        "engines": selected_engines,
                    }
                )

        if collection_selections:
            st.info(
                f"Collection will merge {len(collection_selections):,} source runs. "
                "Duplicate feature columns are replaced by later sources in the selected order."
            )
        if st.button(
            "Create benchmark collection",
            type="primary",
            disabled=not bool(collection_selections),
        ):
            try:
                run_dir = create_benchmark_collection(
                    name=collection_name,
                    selections=collection_selections,
                    target_ids=collection_targets,
                    include_input_columns=bool(include_input_columns),
                )
                st.success("Benchmark collection created.")
                show_pipeline_links(run_dir, [run_dir])
            except Exception as exc:
                st.error(str(exc))

        collection_jobs = [
            row for row in rows
            if read_json(Path(str(row.get("run_dir") or "")) / "input.json").get("job_type") == "benchmark_collection"
        ]
        if collection_jobs:
            st.markdown("**Existing Collections**")
            collection_table = _benchmark_job_table(collection_jobs)
            st.dataframe(
                collection_table[
                    [col for col in ["result", "description", "status", "records", "top_feature", "top_ap", "created_at", "job_code", "run_id"] if col in collection_table.columns]
                ],
                width="stretch",
                hide_index=True,
                column_config={
                    "result": st.column_config.LinkColumn("result", display_text="Open"),
                    "top_ap": st.column_config.NumberColumn("top AP", format="%.3f"),
                },
            )
