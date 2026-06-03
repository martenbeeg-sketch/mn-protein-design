from __future__ import annotations

import os
from io import StringIO
from pathlib import Path

import pandas as pd
import streamlit as st

from mn_protein_design.app.pages.common import result_link, show_pipeline_links
from mn_protein_design.core.jobs import collect_jobs, read_json
from mn_protein_design.core.runtime_estimator import estimate_engines
from mn_protein_design.workflows.benchmark import (
    ALPHAFAST_DB_DIR,
    ALPHAFAST_WEIGHTS_DIR,
    COLABFOLD_CACHE_DIR,
    MSA_REPOSITORY_DIR,
    PUBLISHED_DATASET,
    run_de_novo_binder_scoring_dataset,
    run_esmfold2_binder_benchmark,
    run_precomputed_metric_benchmark,
)
from mn_protein_design.workflows.esm_binder import ESMFOLD2_MODEL_DIR
from mn_protein_design.workflows.refolding import BOLTZ_MODELS_DIR


KNOWN_BENCHMARK_ROOT = Path("/mnt/db/reference_files/de_novo_binder_scoring_overath_2025")
KNOWN_BENCHMARK_CSV = KNOWN_BENCHMARK_ROOT / "final_dataset.csv"
KNOWN_BENCHMARK_PDB_DIR = KNOWN_BENCHMARK_ROOT / "input_pdbs"


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
    if not engines and params.get("modes"):
        engines.append("ESMFold2")
    return ", ".join(engines) or "metrics only"


def _job_dataset_label(input_payload: dict) -> str:
    job_type = str(input_payload.get("job_type") or "")
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
                "result": result_link("benchmark", str(row.get("run_id")), "Open result"),
                "description": " | ".join(description_parts),
                "dataset": _job_dataset_label(input_payload),
                "engines": _enabled_engines(params),
                "records": records,
                "top_feature": top_feature,
                "top_ap": top_ap,
                "plots": _has_plot_outputs(run_dir),
                "status": row.get("status"),
                "created_at": row.get("created_at"),
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
generate_inputs = True
repo_models: list[str] = ["af3", "boltz", "colabfold"]
run_pyrosetta_input = True
default_pyrosetta_nprocs = max(1, min(64, os.cpu_count() or 1))
pyrosetta_nprocs = default_pyrosetta_nprocs
run_common_interface_metrics = True
run_predicted_rosetta_metrics = True
run_pymol_metrics = True
repo_mode = "pdb_only"
repo_max_records = 20
repo_esm_modes = ["initial_guess"]
repo_sampling = 32
repo_loops = 3
repo_seed = 0
af2_recycles = 3
af2_multimer = True
af2_binder_template = False
af2_interface_template = False
boltz2_target_template = True
boltz2_recycling_steps = 10
boltz2_sampling_steps = 200
boltz2_diffusion_samples = 3
boltz2_write_full_pae = True
colabfold_cache_dir = str(COLABFOLD_CACHE_DIR)
colabfold_recycles = 3
colabfold_models = 3
colabfold_msa_source = "msa_repository_then_alphafast_mmseqs_gpu"
msa_repository_dir = str(MSA_REPOSITORY_DIR)
alphafast_db_dir = str(ALPHAFAST_DB_DIR)
alphafast_weights_dir = str(ALPHAFAST_WEIGHTS_DIR)
alphafast_batch_size = 0
alphafast_recycles = 10
gpu_device = 0
simple_modes = ["sequence", "initial_guess"]
simple_max_records = 20
simple_sampling_steps = 32
simple_recycling_loops = 3
simple_seed = 0
simple_contact_cutoff = 8.0
simple_device = "cuda"
label_column = "binder"
metric_max_rows = 0
metric_max_cols = 500
runtime_size = {"candidate_count": 0, "total_residues": None}

tabs = st.tabs(["Dataset", "Engines", "Metrics", "Run", "Results"])

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

            selected_targets = st.multiselect(
                "Targets to benchmark",
                summary["target_id"].astype(str).tolist(),
                default=[],
                help="Select one or more targets from the installed benchmark dataset.",
            )
            max_known_per_target = st.number_input(
                "Max rows per selected target",
                min_value=0,
                max_value=10000,
                value=20,
                step=10,
                help="0 means all rows for selected targets.",
            )
            subset = _known_benchmark_subset(str(KNOWN_BENCHMARK_CSV), tuple(selected_targets), int(max_known_per_target))
            if not selected_targets:
                st.warning("Select at least one target to enable a rerun from the installed dataset.")
                subset = subset.head(int(max_known_per_target or 20))
            elif int(max_known_per_target) == 0:
                st.success(f"All selected target rows will be benchmarked: {len(subset):,} records.")
            else:
                st.info(
                    f"Benchmarking up to {int(max_known_per_target):,} rows per selected target: "
                    f"{len(subset):,} records selected."
                )
            preview_cols = ["binder_id", "target_id", "binder", "source", "binder_chain", "target_chains", "A_length", "B_length"]
            st.dataframe(subset[[col for col in preview_cols if col in subset.columns]].head(200), width="stretch", hide_index=True)
            if selected_targets:
                known_csv_text = subset.to_csv(index=False)
                known_pdb_dir = KNOWN_BENCHMARK_PDB_DIR
                runtime_size = _dataset_runtime_size(subset)

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

    engine_cols = st.columns(5)
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

    if dataset_source not in {"Simple ESMFold2 CSV", "Metric table only"}:
        repo_mode = st.segmented_control("Input mode", ["pdb_only", "hybrid", "seq_only_csv"], selection_mode="single", default="pdb_only")
        generate_inputs = True

        with st.expander("Compute", expanded=True):
            gpu_device = st.number_input(
                "GPU device",
                min_value=0,
                max_value=15,
                value=0,
                step=1,
                help="Used by GPU-backed benchmark engines that expose device selection, currently AlphaFast AF3 and ColabFold.",
            )

        with st.expander("MSA Reference Data", expanded=True):
            msa_cols = st.columns(4)
            colabfold_msa_source = msa_cols[0].segmented_control(
                "MSA source",
                ["msa_repository_then_alphafast_mmseqs_gpu", "msa_repository", "alphafast_mmseqs_gpu", "repo_run_csv"],
                selection_mode="single",
                default="msa_repository_then_alphafast_mmseqs_gpu",
                disabled=not run_colabfold,
                format_func={
                    "msa_repository_then_alphafast_mmseqs_gpu": "Repository, then AlphaFast/MMseqs",
                    "msa_repository": "Repository only",
                    "alphafast_mmseqs_gpu": "AlphaFast/MMseqs GPU",
                    "repo_run_csv": "Existing run.csv paths",
                }.get,
                help="Used for target-chain MSAs. Binder chains stay no-MSA in this benchmark setup.",
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
                disabled=not (run_alphafast_af3 or run_colabfold),
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
            alphafast_recycles = af3_cols[1].number_input("AF3 recycles", min_value=1, max_value=48, value=10, step=1, disabled=not run_alphafast_af3)

        with st.expander("ColabFold Settings", expanded=run_colabfold):
            colab_cols = st.columns(3)
            colabfold_cache_dir = colab_cols[0].text_input("ColabFold / AF2 model cache", value=str(COLABFOLD_CACHE_DIR), disabled=not run_colabfold)
            colabfold_recycles = colab_cols[1].number_input("ColabFold recycles", min_value=1, max_value=48, value=3, step=1, disabled=not run_colabfold)
            colabfold_models = colab_cols[2].number_input("ColabFold models", min_value=1, max_value=5, value=3, step=1, disabled=not run_colabfold)

        with st.expander("AF2 Initial Guess Settings", expanded=run_repo_af2ig):
            af2_cols = st.columns(4)
            af2_recycles = af2_cols[0].number_input("AF2-IG recycles", min_value=1, max_value=24, value=3, step=1, disabled=not run_repo_af2ig)
            af2_multimer = af2_cols[1].checkbox("AF2 multimer", value=True, disabled=not run_repo_af2ig)
            af2_binder_template = af2_cols[2].checkbox("Binder template", value=False, disabled=not run_repo_af2ig)
            af2_interface_template = af2_cols[3].checkbox("Interface template", value=False, disabled=not run_repo_af2ig or not af2_binder_template)

        with st.expander("Boltz-2 Settings", expanded=run_boltz2_ig):
            boltz_cols = st.columns(5)
            boltz_cols[0].text_input("Boltz-2 model cache", value=str(BOLTZ_MODELS_DIR), disabled=True)
            boltz2_target_template = boltz_cols[1].checkbox("Use target template", value=True, disabled=not run_boltz2_ig)
            boltz2_recycling_steps = boltz_cols[2].number_input("Recycling steps", min_value=1, max_value=48, value=10, step=1, disabled=not run_boltz2_ig)
            boltz2_sampling_steps = boltz_cols[3].number_input("Sampling steps", min_value=1, max_value=1000, value=200, step=1, disabled=not run_boltz2_ig)
            boltz2_diffusion_samples = boltz_cols[4].number_input("Diffusion samples", min_value=1, max_value=20, value=3, step=1, disabled=not run_boltz2_ig)
            boltz2_write_full_pae = st.checkbox("Write full PAE", value=True, disabled=not run_boltz2_ig)

    with st.expander("ESMFold2 Settings", expanded=run_repo_esm or dataset_source == "Simple ESMFold2 CSV"):
        if dataset_source == "Simple ESMFold2 CSV":
            simple_modes = st.multiselect(
                "Modes",
                ["sequence", "initial_guess"],
                default=["sequence", "initial_guess"],
                format_func={"sequence": "Sequence only", "initial_guess": "Initial guess"}.get,
            )
            simple_cols = st.columns(6)
            simple_max_records = simple_cols[0].number_input("Max records", min_value=1, max_value=10000, value=20, step=1)
            simple_sampling_steps = simple_cols[1].number_input("Sampling steps", min_value=1, max_value=256, value=32, step=1)
            simple_recycling_loops = simple_cols[2].number_input("Recycling loops", min_value=1, max_value=16, value=3, step=1)
            simple_seed = simple_cols[3].number_input("Seed", min_value=0, max_value=999999, value=0, step=1)
            simple_contact_cutoff = simple_cols[4].number_input("Contact cutoff", min_value=2.0, max_value=20.0, value=8.0, step=0.5)
            simple_device = simple_cols[5].selectbox("Device", ["cuda", "auto", "cpu"], index=0)
        else:
            repo_esm_modes = st.multiselect(
                "Modes",
                ["sequence", "initial_guess"],
                default=["initial_guess"],
                disabled=not run_repo_esm,
                format_func={"sequence": "Sequence only", "initial_guess": "Initial guess"}.get,
            )
            esm_cols = st.columns(5)
            esm_cols[0].text_input("ESMFold2 model dir", value=str(ESMFOLD2_MODEL_DIR), disabled=True)
            repo_max_records = esm_cols[1].number_input(
                "Engine max records",
                min_value=0,
                max_value=10000,
                value=20,
                step=10,
                disabled=not (run_repo_esm or run_repo_af2ig or run_boltz2_ig or run_colabfold or run_alphafast_af3),
                help="0 means all selected dataset rows. This limit applies to prediction/refolding engines.",
            )
            repo_sampling = esm_cols[2].number_input("ESMFold2 sampling steps", min_value=1, max_value=256, value=32, step=1, disabled=not run_repo_esm)
            repo_loops = esm_cols[3].number_input("ESMFold2 recycling loops", min_value=1, max_value=16, value=3, step=1, disabled=not run_repo_esm)
            repo_seed = esm_cols[4].number_input("ESMFold2 seed", min_value=0, max_value=999999, value=0, step=1, disabled=not run_repo_esm)

with tabs[2]:
    st.subheader("Metrics")
    if dataset_source == "Metric table only":
        metric_cols = st.columns(3)
        label_column = metric_cols[0].text_input("Label column", value="binder")
        metric_max_rows = metric_cols[1].number_input("Max rows", min_value=0, max_value=1000000, value=0, step=100, help="0 means all rows.")
        metric_max_cols = metric_cols[2].number_input("Max numeric columns", min_value=1, max_value=10000, value=500, step=50)
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
    summary_cols = st.columns(4)
    summary_cols[0].metric("Dataset", str(dataset_source))
    summary_cols[1].metric("Engines", len(selected_engines))
    summary_cols[2].metric("Model inputs", "yes" if generate_inputs and dataset_source not in {"Simple ESMFold2 CSV", "Metric table only"} else "n/a")
    summary_cols[3].metric("Rosetta", "yes" if run_pyrosetta_input else "no")
    st.write(", ".join(selected_engines) if selected_engines else "No prediction engine selected.")
    if dataset_source == "Installed Overath 2025" and selected_targets:
        selected_count = int(runtime_size.get("candidate_count") or 0)
        if int(max_known_per_target) == 0:
            st.success(f"This run will use all selected rows: {selected_count:,} records.")
        else:
            st.info(f"This run is limited to {int(max_known_per_target):,} rows per target: {selected_count:,} records.")
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
    if run_common_interface_metrics or run_predicted_rosetta_metrics or run_pymol_metrics:
        estimate_engine_keys.append("postprocessing")

    estimate_count = int(runtime_size.get("candidate_count") or 0)
    if dataset_source not in {"Metric table only", "Simple ESMFold2 CSV"} and repo_max_records:
        estimate_count = min(estimate_count or int(repo_max_records), int(repo_max_records))
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
                progress_box = st.empty()

                def _benchmark_progress(payload: dict) -> None:
                    label = str(payload.get("progress_label") or "Running benchmark workflow...")
                    step = payload.get("progress_step")
                    total = payload.get("progress_total")
                    if step and total:
                        label = f"Step {step}/{total}: {label}"
                    progress_box.info(label)

                with st.spinner("Running benchmark workflow..."):
                    run_dir = run_de_novo_binder_scoring_dataset(
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
                        run_af2_initial_guess=bool(run_repo_af2ig),
                        af2_num_recycles=int(af2_recycles),
                        af2_multimer=bool(af2_multimer),
                        af2_use_binder_template=bool(af2_binder_template),
                        af2_use_interface_template=bool(af2_interface_template and af2_binder_template),
                        run_boltz2_initial_guess=bool(run_boltz2_ig),
                        boltz2_use_target_template=bool(boltz2_target_template),
                        boltz2_recycling_steps=int(boltz2_recycling_steps),
                        boltz2_sampling_steps=int(boltz2_sampling_steps),
                        boltz2_diffusion_samples=int(boltz2_diffusion_samples),
                        boltz2_write_full_pae=bool(boltz2_write_full_pae),
                        run_colabfold=bool(run_colabfold),
                        colabfold_cache_dir=Path(colabfold_cache_dir).expanduser(),
                        colabfold_msa_source=str(colabfold_msa_source or "msa_repository_then_alphafast_mmseqs_gpu"),
                        msa_repository_dir=Path(msa_repository_dir).expanduser(),
                        colabfold_num_recycles=int(colabfold_recycles),
                        colabfold_num_models=int(colabfold_models),
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
                        progress_callback=_benchmark_progress,
                    )
                progress_box.success("Benchmark workflow finished.")
                result = read_json(run_dir / "result.json")
                metrics = result.get("metrics") or {}
                st.success(f"Benchmark finished | records: {metrics.get('record_count')}")
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
        st.dataframe(
            df[[col for col in display_cols if col in df.columns]],
            width="stretch",
            hide_index=True,
            column_config={
                "result": st.column_config.LinkColumn("result", display_text="Open"),
                "top_ap": st.column_config.NumberColumn("top AP", format="%.3f"),
            },
        )
