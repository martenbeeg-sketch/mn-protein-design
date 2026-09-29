from __future__ import annotations

import ast
import os
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from mn_protein_design.app.pages.common import (
    esmfold2_preset_label,
    esmfold2_preset_selector,
    gpu_run_panel,
    refresh_results_button,
    result_link,
    selected_dataframe_rows,
    show_delete_jobs_dialog,
    show_pipeline_links,
)
from mn_protein_design.core.jobs import ACTIVE_STATUSES, collect_jobs, finish_job, read_json
from mn_protein_design.core.local_worker import spawn_worker_for_run
from mn_protein_design.core.runtime_estimator import estimate_engines, format_duration
from mn_protein_design.runtime import runs_root
from mn_protein_design.workflows.benchmark import (
    ALPHAFAST_DB_DIR,
    ALPHAFAST_WEIGHTS_DIR,
    COLABFOLD_CACHE_DIR,
    MSA_REPOSITORY_DIR,
    PUBLISHED_DATASET,
    benchmark_engines_in_metrics,
    benchmark_missing_pyrosetta_rows,
    create_benchmark_collection,
    create_benchmark_matrix_workspace,
    enqueue_de_novo_binder_scoring_dataset,
    enqueue_missing_pyrosetta_benchmark_metrics,
    inspect_benchmark_run,
    prepare_benchmark_run_resume,
    run_esmfold2_binder_benchmark,
    run_precomputed_metric_benchmark,
)
from mn_protein_design.workflows.capacity_benchmark import capacity_warnings
from mn_protein_design.workflows.esm_binder import ESMFOLD2_MODEL_DIR
from mn_protein_design.workflows.refolding import (
    BOLTZ_MODELS_DIR,
    OPENFOLD3_CHECKPOINT,
    PROTENIX_V1_20250630_MODEL,
    PROTENIX_V1_MODEL,
    PROTENIX_V2_MODEL,
    RF3_CHECKPOINT,
)


KNOWN_BENCHMARK_ROOT = Path("/mnt/db/reference_files/de_novo_binder_scoring_overath_2025")
KNOWN_BENCHMARK_CSV = KNOWN_BENCHMARK_ROOT / "final_dataset.csv"
KNOWN_BENCHMARK_PDB_DIR = KNOWN_BENCHMARK_ROOT / "input_pdbs"
BENCHMARK_MATRIX_ENGINE_ORDER = [
    "AF3",
    "AF2-IG",
    "Boltz-2",
    "BoltzGen Fold",
    "ColabFold",
    "ESMFold2",
    "OpenFold-3",
    "Protenix v0.5",
    "Protenix v1",
    "Protenix v2",
    "RF3",
]
BENCHMARK_ENGINE_METRIC_FILES = {
    "AF3": "alphafast_af3_metrics.csv",
    "AF2-IG": "af2_initial_guess_metrics.csv",
    "Boltz-2": "boltz2_initial_guess_metrics.csv",
    "BoltzGen Fold": "boltzgen_fold_metrics.csv",
    "ColabFold": "colabfold_metrics.csv",
    "ESMFold2": "esmfold2_metrics.csv",
    "OpenFold-3": "openfold3_metrics.csv",
    "Protenix v0.5": "protenix_metrics.csv",
    "Protenix v1": "protenix_v1_metrics.csv",
    "Protenix v2": "protenix_v2_metrics.csv",
    "RF3": "rf3_metrics.csv",
    "Input": "input_rosetta_metrics.csv",
}
BENCHMARK_ENGINE_ALIASES = {
    "Protenix": "Protenix v0.5",
}


def _canonical_benchmark_engine_label(engine: object) -> str:
    text = str(engine or "").strip()
    return BENCHMARK_ENGINE_ALIASES.get(text, text)


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
        return {"candidate_count": int(fallback_count or 0), "total_residues": None, "max_system_residues": None}
    candidate_count = int(len(df))
    length_columns = [column for column in ["A_length", "B_length"] if column in df.columns]
    total_residues = None
    max_system_residues = None
    if length_columns:
        total_series = pd.Series([0] * len(df), index=df.index, dtype="float64")
        for column in length_columns:
            values = pd.to_numeric(df[column], errors="coerce").fillna(0)
            total_series = total_series + values
        total = int(total_series.sum())
        total_residues = total or None
        max_system_residues = int(total_series.max()) if not total_series.empty and total_series.max() > 0 else None
    return {"candidate_count": candidate_count, "total_residues": total_residues, "max_system_residues": max_system_residues}


def _estimate_rows_dataframe(estimate: dict) -> pd.DataFrame:
    rows = []
    for row in estimate.get("rows") or []:
        rows.append(
            {
                "engine": row.get("label"),
                "estimated_time": row.get("estimated_time"),
                "seconds_per_candidate": row.get("seconds_per_candidate"),
                "seconds_per_residue": row.get("seconds_per_residue"),
                "basis": row.get("basis"),
                "history_runs": row.get("history_runs"),
            }
        )
    return pd.DataFrame(rows)


def _show_capacity_warnings(warnings: list[dict[str, object]]) -> None:
    relevant = [row for row in warnings if row.get("severity") in {"error", "warning"}]
    unknown = [row for row in warnings if row.get("severity") == "info"]
    if relevant:
        st.warning("Capacity benchmark warning: at least one selected engine is outside known successful limits.")
        st.dataframe(
            pd.DataFrame(relevant),
            hide_index=True,
            width="stretch",
            column_order=[
                "engine",
                "severity",
                "evidence_scope",
                "message",
                "max_success_total_length",
                "min_failed_total_length",
                "min_oom_total_length",
                "preset",
                "gpu_device",
            ],
        )
    elif warnings and all(row.get("severity") == "ok" for row in warnings):
        st.success("Largest selected system is within the matching capacity benchmark range.")
    if unknown:
        with st.expander("Capacity evidence not available for some engines", expanded=False):
            st.dataframe(pd.DataFrame(unknown), hide_index=True, width="stretch")


def _enabled_engines(params: dict) -> str:
    if isinstance(params.get("selections"), list):
        collection_engines: list[str] = []
        for selection in params.get("selections") or []:
            if isinstance(selection, dict):
                collection_engines.extend(str(engine) for engine in (selection.get("engines") or []))
        if collection_engines:
            return ", ".join(dict.fromkeys(collection_engines))
    engines = _benchmark_engine_labels_from_params(params)
    return ", ".join(engines) or "metrics only"


def _benchmark_engine_labels_from_params(params: dict) -> list[str]:
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
    if params.get("run_openfold3"):
        engines.append("OpenFold-3")
    if params.get("run_protenix"):
        engines.append("Protenix v0.5")
    if params.get("run_protenix_v1"):
        engines.append("Protenix v1")
    if params.get("run_protenix_v2"):
        engines.append("Protenix v2")
    if params.get("run_boltzgen_fold"):
        engines.append("BoltzGen Fold")
    if not engines and params.get("modes"):
        engines.append("ESMFold2")
    return engines


def _bool_param(params: dict, worker_kwargs: dict, key: str, default: bool = False) -> bool:
    value = params.get(key, worker_kwargs.get(key, default))
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _list_param(params: dict, worker_kwargs: dict, key: str) -> list[str]:
    value = params.get(key, worker_kwargs.get(key, []))
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [part.strip() for part in text.split(",") if part.strip()]
    return []


def _benchmark_evidence_label(*, msa: bool = False, template: bool = False) -> str:
    if msa and template:
        return "MSA+template"
    if msa:
        return "MSA"
    if template:
        return "template"
    return "none"


def _benchmark_evidence_summary(params: dict, worker_kwargs: dict) -> tuple[str, str]:
    """Summarize whether selected benchmark engines used target MSA and/or template evidence."""
    if isinstance(params.get("selections"), list):
        engines: list[str] = []
        for selection in params.get("selections") or []:
            if isinstance(selection, dict):
                engines.extend(str(engine) for engine in selection.get("engines") or [])
        return "mixed", "; ".join(dict.fromkeys(engines))

    uses_msa = False
    uses_template = False
    engine_modes: list[tuple[str, str]] = []

    def add(engine: str, *, msa: bool = False, template: bool = False) -> None:
        nonlocal uses_msa, uses_template
        uses_msa = uses_msa or bool(msa)
        uses_template = uses_template or bool(template)
        engine_modes.append((engine, _benchmark_evidence_label(msa=msa, template=template)))

    if _bool_param(params, worker_kwargs, "run_alphafast_af3"):
        add(
            "AF3",
            msa=not _bool_param(params, worker_kwargs, "alphafast_query_only_msa", False),
            template=_bool_param(params, worker_kwargs, "alphafast_use_target_templates", True),
        )
    if _bool_param(params, worker_kwargs, "run_colabfold"):
        add(
            "ColabFold",
            msa=_bool_param(params, worker_kwargs, "colabfold_use_target_msa", True),
            template=_bool_param(params, worker_kwargs, "colabfold_use_target_templates", True),
        )
    if _bool_param(params, worker_kwargs, "run_af2_initial_guess"):
        add("AF2-IG", template=True)
    if _bool_param(params, worker_kwargs, "run_boltz2_initial_guess"):
        add(
            "Boltz-2",
            msa=_bool_param(params, worker_kwargs, "boltz2_use_target_msa", True),
            template=_bool_param(params, worker_kwargs, "boltz2_use_target_template", True),
        )
    if _bool_param(params, worker_kwargs, "run_esmfold2") or params.get("modes"):
        add(
            "ESMFold2",
            msa=_bool_param(params, worker_kwargs, "esmfold2_use_target_msa", False),
            template="initial_guess" in set(
                _list_param(params, worker_kwargs, "esmfold2_modes") or _list_param(params, worker_kwargs, "modes")
            ),
        )
    if _bool_param(params, worker_kwargs, "run_rf3"):
        add(
            "RF3",
            msa=_bool_param(params, worker_kwargs, "rf3_use_target_msa", True),
            template=_bool_param(params, worker_kwargs, "rf3_use_target_template", True),
        )
    if _bool_param(params, worker_kwargs, "run_openfold3"):
        add("OpenFold-3", msa=_bool_param(params, worker_kwargs, "openfold3_use_target_msa", True))
    if _bool_param(params, worker_kwargs, "run_protenix"):
        add("Protenix v0.5", msa=_bool_param(params, worker_kwargs, "protenix_use_msa", True))
    if _bool_param(params, worker_kwargs, "run_protenix_v1"):
        add(
            "Protenix v1",
            msa=_bool_param(params, worker_kwargs, "protenix_v1_use_msa", True),
            template=_bool_param(params, worker_kwargs, "protenix_v1_use_template", False),
        )
    if _bool_param(params, worker_kwargs, "run_protenix_v2"):
        add(
            "Protenix v2",
            msa=_bool_param(params, worker_kwargs, "protenix_v2_use_msa", True),
            template=_bool_param(params, worker_kwargs, "protenix_v2_use_template", False),
        )
    if _bool_param(params, worker_kwargs, "run_boltzgen_fold"):
        add("BoltzGen Fold", template=True)

    return _benchmark_evidence_label(msa=uses_msa, template=uses_template), "; ".join(
        f"{engine}: {mode}" for engine, mode in engine_modes
    )


def _benchmark_engines_available_for_run(run_dir: Path, input_payload: dict, metrics_path: Path) -> list[str]:
    detected = [_canonical_benchmark_engine_label(engine) for engine in benchmark_engines_in_metrics(metrics_path)]
    job_type = str(input_payload.get("job_type") or "")
    if job_type == "benchmark_collection":
        return detected

    params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
    selected = {_canonical_benchmark_engine_label(engine) for engine in _benchmark_engine_labels_from_params(params)}
    if not selected:
        return detected

    benchmark_dir = run_dir / "artifacts" / "benchmark"
    available: list[str] = []
    for engine in detected:
        if engine not in selected and engine != "Input":
            continue
        metric_name = BENCHMARK_ENGINE_METRIC_FILES.get(engine)
        if metric_name and not (benchmark_dir / metric_name).exists():
            continue
        available.append(engine)
    return available


def _short_bool(value: object) -> str:
    return "yes" if bool(value) else "no"


def _benchmark_engine_settings(input_payload: dict, worker_kwargs: dict, engine: str) -> str:
    params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}

    def get_value(*names: str, default: object = "") -> object:
        for name in names:
            if name in params and params.get(name) not in (None, ""):
                return params.get(name)
            if name in worker_kwargs and worker_kwargs.get(name) not in (None, ""):
                return worker_kwargs.get(name)
        return default

    def add(parts: list[str], label: str, value: object) -> None:
        if value not in (None, "", []):
            parts.append(f"{label}={value}")

    parts: list[str] = []
    if engine == "ESMFold2":
        loops = get_value("num_loops", default="")
        steps = get_value("num_sampling_steps", default="")
        add(parts, "preset", esmfold2_preset_label(loops, steps))
        add(parts, "modes", ",".join(str(item) for item in get_value("esmfold2_modes", "modes", default=[]) or []))
        add(parts, "target_msa", _short_bool(get_value("esmfold2_use_target_msa", default=False)))
        add(parts, "seed", get_value("seed", default=""))
    elif engine == "AF2-IG":
        add(parts, "recycles", get_value("af2_num_recycles", default=""))
        add(parts, "multimer", _short_bool(get_value("af2_multimer", default=True)))
        add(parts, "target_ig", "yes")
        add(parts, "complex_ig", _short_bool(get_value("af2_use_initial_guess", default=False)))
        add(parts, "binder_template", _short_bool(get_value("af2_use_binder_template", default=False)))
        add(parts, "interface_template", _short_bool(get_value("af2_use_interface_template", default=False)))
    elif engine == "Boltz-2":
        add(parts, "target_template", _short_bool(get_value("boltz2_use_target_template", default=True)))
        add(parts, "target_msa", _short_bool(get_value("boltz2_use_target_msa", default=True)))
        add(parts, "recycles", get_value("boltz2_recycling_steps", default=""))
        add(parts, "sampling", get_value("boltz2_sampling_steps", default=""))
        add(parts, "samples", get_value("boltz2_diffusion_samples", default=""))
    elif engine == "BoltzGen Fold":
        add(parts, "recycles", get_value("boltzgen_recycling_steps", default=""))
        add(parts, "sampling", get_value("boltzgen_sampling_steps", default=""))
        add(parts, "samples", get_value("boltzgen_diffusion_samples", default=""))
    elif engine == "ColabFold":
        add(parts, "msa", get_value("colabfold_msa_source", default=""))
        add(parts, "recycles", get_value("colabfold_num_recycles", default=""))
        add(parts, "models", get_value("colabfold_num_models", default=""))
        add(parts, "templates", _short_bool(get_value("colabfold_use_target_templates", default=False)))
        add(parts, "template_hits", get_value("colabfold_max_template_hits", default=""))
    elif engine == "AF3":
        add(parts, "templates", _short_bool(get_value("alphafast_use_target_templates", default=True)))
        add(parts, "target_msa", _short_bool(not bool(get_value("alphafast_query_only_msa", default=False))))
        add(parts, "recycles", get_value("alphafast_num_recycles", default=""))
        add(parts, "batch", get_value("alphafast_batch_size", default=""))
        add(parts, "gpu", get_value("alphafast_gpu_device", "gpu_device", default=""))
    elif engine == "RF3":
        add(parts, "templates", _short_bool(get_value("rf3_use_target_template", default=True)))
        add(parts, "target_msa", _short_bool(get_value("rf3_use_target_msa", default=True)))
        add(parts, "recycles", get_value("rf3_recycles", default=""))
        add(parts, "steps", get_value("rf3_num_steps", default=""))
        add(parts, "batch", get_value("rf3_diffusion_batch_size", default=""))
        checkpoint = str(get_value("rf3_checkpoint_path", default=""))
        add(parts, "checkpoint", Path(checkpoint).name if checkpoint else "")
    elif engine == "OpenFold-3":
        add(parts, "target_msa", _short_bool(get_value("openfold3_use_target_msa", default=True)))
        add(parts, "samples", get_value("openfold3_num_diffusion_samples", default=""))
        add(parts, "seeds", get_value("openfold3_num_model_seeds", default=""))
        add(parts, "recycles", get_value("openfold3_num_recycles", default=""))
        add(parts, "msa_server", _short_bool(get_value("openfold3_use_msa_server", default=False)))
        checkpoint = str(get_value("openfold3_checkpoint_path", default=""))
        add(parts, "checkpoint", Path(checkpoint).name if checkpoint else "")
    elif engine in {"Protenix", "Protenix v0.5"}:
        add(parts, "msa", _short_bool(get_value("protenix_use_msa", default=True)))
        add(parts, "cycles", get_value("protenix_cycle", default=""))
        add(parts, "steps", get_value("protenix_diffusion_steps", default=""))
        add(parts, "samples", get_value("protenix_samples", default=""))
    elif engine == "Protenix v1":
        add(parts, "model", get_value("protenix_v1_model_name", default=""))
        add(parts, "default_params", _short_bool(get_value("protenix_v1_use_default_params", default=True)))
        add(parts, "msa", _short_bool(get_value("protenix_v1_use_msa", default=True)))
        add(parts, "template", _short_bool(get_value("protenix_v1_use_template", default=False)))
        add(parts, "cycles", get_value("protenix_v1_cycle", default=""))
        add(parts, "steps", get_value("protenix_v1_diffusion_steps", default=""))
        add(parts, "samples", get_value("protenix_v1_samples", default=""))
    elif engine == "Protenix v2":
        add(parts, "model", get_value("protenix_v2_model_name", default=""))
        add(parts, "default_params", _short_bool(get_value("protenix_v2_use_default_params", default=True)))
        add(parts, "msa", _short_bool(get_value("protenix_v2_use_msa", default=True)))
        add(parts, "template", _short_bool(get_value("protenix_v2_use_template", default=False)))
        add(parts, "cycles", get_value("protenix_v2_cycle", default=""))
        add(parts, "steps", get_value("protenix_v2_diffusion_steps", default=""))
        add(parts, "samples", get_value("protenix_v2_samples", default=""))
    return "; ".join(parts)


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
        if job_type == "refolding_evaluation":
            continue
        result = read_json(run_dir / "result.json")
        feature_summary = _first_feature_summary(run_dir)
        params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
        worker_payload = read_json(run_dir / "worker_request.json")
        worker_kwargs = worker_payload.get("kwargs") if isinstance(worker_payload.get("kwargs"), dict) else {}
        esm_loops = params.get("num_loops", worker_kwargs.get("num_loops"))
        esm_steps = params.get("num_sampling_steps", worker_kwargs.get("num_sampling_steps"))
        esm_preset = esmfold2_preset_label(esm_loops, esm_steps)
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
        evidence_mode, evidence_detail = _benchmark_evidence_summary(params, worker_kwargs)
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
        recovery = ""
        if str(row.get("status") or "") in {"failed", "paused"} and (run_dir / "worker_request.json").exists():
            try:
                recovery_report = inspect_benchmark_run(run_dir)
                if recovery_report.get("can_resume"):
                    recovery = f"resumable: {recovery_report.get('first_incomplete_stage') or 'next incomplete stage'}"
            except Exception:
                recovery = ""
        enriched.append(
            {
                "delete": False,
                "result": result_link("benchmark", str(row.get("run_id")), "Open result"),
                "job_code_link": f"{result_link('benchmark', str(row.get('run_id')))}&job_code={row.get('job_code')}",
                "kind": "collection" if job_type == "benchmark_collection" else "benchmark",
                "description": " | ".join(description_parts),
                "dataset": _job_dataset_label(input_payload),
                "engines": _enabled_engines(params),
                "evidence_mode": evidence_mode,
                "evidence_detail": evidence_detail,
                "esmfold2_preset": esm_preset,
                "esmfold2_loops": esm_loops,
                "esmfold2_steps": esm_steps,
                "records": records,
                "top_feature": top_feature,
                "top_ap": top_ap,
                "plots": _has_plot_outputs(run_dir),
                "status": row.get("status"),
                "recovery": recovery,
                "created_at": row.get("created_at"),
                "task_group": "benchmark",
                "run_id": row.get("run_id"),
                "job_code": row.get("job_code"),
            }
        )
    return pd.DataFrame(enriched)


def _available_benchmark_sources(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    source_rows: list[dict] = []
    unavailable_rows: list[dict] = []
    for row in rows:
        run_dir = Path(str(row.get("run_dir") or ""))
        input_payload = read_json(run_dir / "input.json")
        if input_payload.get("job_type") == "benchmark_matrix_workspace":
            continue
        if input_payload.get("job_type") == "refolding_evaluation":
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
        engines_available = _benchmark_engines_available_for_run(run_dir, input_payload, metrics_path)
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
        worker_payload = read_json(run_dir / "worker_request.json")
        worker_kwargs = worker_payload.get("kwargs") if isinstance(worker_payload.get("kwargs"), dict) else {}
        result = read_json(run_dir / "result.json")
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        job_type = str(input_payload.get("job_type") or "")
        source_rows.append(
            {
                "run_id": str(row.get("run_id")),
                "job_code": str(row.get("job_code")),
                "kind": "collection" if job_type == "benchmark_collection" else "benchmark",
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
                "dataset": _job_dataset_label(input_payload),
                "engines_available": engines_available,
                "engine_settings": {
                    engine: _benchmark_engine_settings(input_payload, worker_kwargs, engine)
                    for engine in engines_available
                },
                "engine_text": ", ".join(engines_available),
                "metrics_path": metrics_path,
                "record_count": metrics.get("record_count"),
            }
        )
    return source_rows, unavailable_rows


def _benchmark_target_counts(metrics_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(metrics_path, usecols=lambda col: col in {"target_id", "binder_id"})
    except Exception:
        return pd.DataFrame(columns=["target_id", "records"])
    if "target_id" not in df.columns:
        df["target_id"] = "all targets"
    grouped = df.groupby("target_id", dropna=False).size().reset_index(name="records")
    grouped["target_id"] = grouped["target_id"].astype(str)
    return grouped


def _benchmark_matrix_cells(source_rows: list[dict]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    engine_rank = {engine: index for index, engine in enumerate(BENCHMARK_MATRIX_ENGINE_ORDER)}
    for source in source_rows:
        target_counts = _benchmark_target_counts(Path(source["metrics_path"]))
        if target_counts.empty:
            continue
        for _, target_row in target_counts.iterrows():
            target_id = str(target_row.get("target_id") or "all targets")
            for engine in source.get("engines_available") or []:
                if engine == "Input":
                    continue
                rows.append(
                    {
                        "select": False,
                        "engine": engine,
                        "engine_rank": engine_rank.get(engine, len(engine_rank)),
                        "target_id": target_id,
                        "status": "completed",
                        "records": int(target_row.get("records") or 0),
                        "run_id": source["run_id"],
                        "job_code": source["job_code"],
                        "kind": source.get("kind") or "benchmark",
                        "dataset": source.get("dataset") or "",
                        "settings": (source.get("engine_settings") or {}).get(engine, ""),
                        "description": source.get("description") or "",
                        "created_at": source.get("created_at") or "",
                        "result": result_link("benchmark", str(source["run_id"]), "Open"),
                    }
                )
    if not rows:
        return pd.DataFrame()
    cells = pd.DataFrame(rows)
    cells["created_sort"] = cells["created_at"].astype(str)
    cells["kind_rank"] = cells["kind"].map({"benchmark": 0, "collection": 1}).fillna(2)
    cells = cells.sort_values(["engine_rank", "target_id", "kind_rank", "created_sort"], ascending=[True, True, True, False])
    return cells


def _input_target_counts_for_benchmark(run_dir: Path, input_payload: dict) -> pd.DataFrame:
    records = _input_records_for_benchmark(run_dir, input_payload)
    if records.empty:
        return pd.DataFrame(columns=["target_id", "records"])
    grouped = records.groupby("target_id", dropna=False).size().reset_index(name="records")
    grouped["target_id"] = grouped["target_id"].astype(str)
    return grouped


def _input_records_for_benchmark(run_dir: Path, input_payload: dict) -> pd.DataFrame:
    inputs = input_payload.get("inputs") if isinstance(input_payload.get("inputs"), dict) else {}
    candidate_paths = [
        run_dir / "artifacts" / "raw" / "de_novo_binder_scoring" / "dataset" / "input.csv",
        run_dir / "artifacts" / "queued_inputs" / "input.csv",
    ]
    input_csv = str(inputs.get("input_csv") or "").strip()
    if input_csv:
        candidate_paths.append(Path(input_csv).expanduser())
    worker_payload = read_json(run_dir / "worker_request.json")
    worker_kwargs = worker_payload.get("kwargs") if isinstance(worker_payload.get("kwargs"), dict) else {}
    worker_csv = str(worker_kwargs.get("input_csv") or "").strip()
    if worker_csv:
        candidate_paths.append(Path(worker_csv).expanduser())
    for path in candidate_paths:
        if path.exists():
            try:
                df = pd.read_csv(path)
            except Exception:
                continue
            if df.empty or "binder_id" not in df.columns:
                continue
            if "target_id" not in df.columns:
                df["target_id"] = "all targets"
            df["binder_id"] = df["binder_id"].astype(str)
            df["target_id"] = df["target_id"].astype(str)
            df["total_residues"] = df.apply(_input_row_total_residues, axis=1)
            return df[["binder_id", "target_id", "total_residues"]].copy()
    return pd.DataFrame(columns=["binder_id", "target_id", "total_residues"])


def _numeric_or_zero(value: object) -> int:
    try:
        if value is None or pd.isna(value):
            return 0
        return max(0, int(float(str(value))))
    except (TypeError, ValueError):
        return 0


def _input_row_total_residues(row: pd.Series) -> int:
    binder_chain = str(row.get("binder_chain") or "A").strip() or "A"
    total = _numeric_or_zero(row.get(f"{binder_chain}_length") or row.get("A_length"))
    target_chains = _parse_text_list(row.get("target_chains"))
    if target_chains:
        for chain in target_chains:
            if chain == binder_chain:
                continue
            length = (
                _numeric_or_zero(row.get(f"{chain}_length"))
                or _numeric_or_zero(row.get(f"target_subchain_{chain}_len"))
                or _sequence_length_value(row.get(f"target_subchain_{chain}_seq"))
            )
            total += int(length)
    else:
        total += _numeric_or_zero(row.get("B_length"))
    return int(total)


def _sequence_length_value(value: object) -> int:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return 0
    return len(text)


def _parse_text_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return []
    try:
        import ast

        parsed = ast.literal_eval(text)
    except Exception:
        parsed = None
    if isinstance(parsed, (list, tuple, set)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [part.strip().strip("'\"") for part in text.replace(";", ",").split(",") if part.strip().strip("'\"")]


def _child_tool_for_matrix_engine(engine: str) -> str:
    engine = _canonical_benchmark_engine_label(engine)
    return {
        "Protenix v0.5": "protenix",
        "Protenix v1": "protenix_v1",
        "Protenix v2": "protenix_v2",
        "RF3": "rf3",
        "OpenFold-3": "openfold3",
        "BoltzGen Fold": "boltzgen_fold",
    }.get(engine, "")


def _latest_internal_child_run(parent_run_dir: Path, tool: str) -> Path | None:
    if not tool:
        return None
    matches: list[Path] = []
    root = runs_root()
    if not root.exists():
        return None
    parent_text = str(parent_run_dir)
    for metadata_path in root.glob("*/*/metadata.json"):
        child_dir = metadata_path.parent
        if child_dir == parent_run_dir:
            continue
        metadata = read_json(metadata_path)
        if str(metadata.get("parent_run_dir") or metadata.get("internal_parent_run_dir") or "") != parent_text:
            continue
        if str(metadata.get("tool") or metadata.get("engine") or "") != tool:
            continue
        matches.append(child_dir)
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def _active_child_candidate_ids(child_run_dir: Path) -> set[str]:
    metadata = read_json(child_run_dir / "metadata.json")
    for key in ("current_candidate_ids", "current_step_candidate_ids", "active_candidate_ids"):
        values = metadata.get(key)
        if isinstance(values, list):
            return {str(value) for value in values if str(value)}

    command_payload = read_json(child_run_dir / "command.json")
    steps = command_payload.get("steps") if isinstance(command_payload.get("steps"), list) else []
    if not steps:
        return set()
    stdout_path = child_run_dir / "stdout.log"
    if not stdout_path.exists():
        return set()
    try:
        command_lines = [line.strip()[2:] for line in stdout_path.read_text(errors="ignore").splitlines() if line.startswith("$ ")]
    except Exception:
        return set()
    if not command_lines:
        return set()
    active_command = command_lines[-1]
    for step in steps:
        command = step.get("command")
        if isinstance(command, list) and " ".join(str(part) for part in command) == active_command:
            return {str(value) for value in step.get("candidate_ids") or [] if str(value)}
    return set()


def _active_targets_for_child_engine(parent_run_dir: Path, engine: str, input_records: pd.DataFrame) -> set[str]:
    child_run_dir = _latest_internal_child_run(parent_run_dir, _child_tool_for_matrix_engine(engine))
    if child_run_dir is None or input_records.empty:
        return set()
    metadata = read_json(child_run_dir / "metadata.json")
    target_values = metadata.get("current_target_ids")
    if isinstance(target_values, list):
        targets = {str(value) for value in target_values if str(value)}
        if targets:
            return targets
    active_ids = _active_child_candidate_ids(child_run_dir)
    if not active_ids:
        return set()
    binder_to_target = {
        str(row["binder_id"]): str(row["target_id"])
        for _, row in input_records.iterrows()
    }
    binder_to_target.update({str(key).lower(): value for key, value in list(binder_to_target.items())})
    return {binder_to_target[candidate_id] for candidate_id in active_ids if candidate_id in binder_to_target}


def _child_engine_step_index(parent_run_dir: Path, engine: str) -> tuple[int | None, list[dict]]:
    child_run_dir = _latest_internal_child_run(parent_run_dir, _child_tool_for_matrix_engine(engine))
    if child_run_dir is None:
        return None, []
    metadata = read_json(child_run_dir / "metadata.json")
    step_index = metadata.get("current_step_index")
    command_payload = read_json(child_run_dir / "command.json")
    steps = command_payload.get("steps") if isinstance(command_payload.get("steps"), list) else []
    if step_index:
        return int(step_index), steps
    stdout_path = child_run_dir / "stdout.log"
    if not steps or not stdout_path.exists():
        return None, steps
    try:
        command_lines = [line.strip()[2:] for line in stdout_path.read_text(errors="ignore").splitlines() if line.startswith("$ ")]
    except Exception:
        return None, steps
    if not command_lines:
        return None, steps
    active_command = command_lines[-1]
    for step_index, step in enumerate(steps, start=1):
        command = step.get("command")
        if isinstance(command, list) and " ".join(str(part) for part in command) == active_command:
            return step_index, steps
    return None, steps


def _child_engine_state(parent_run_dir: Path, engine: str) -> tuple[Path | None, int | None, list[dict]]:
    child_run_dir = _latest_internal_child_run(parent_run_dir, _child_tool_for_matrix_engine(engine))
    if child_run_dir is None:
        return None, None, []
    step_index, steps = _child_engine_step_index(parent_run_dir, engine)
    return child_run_dir, step_index, steps


def _candidate_residue_map(input_records: pd.DataFrame) -> dict[str, int]:
    if input_records.empty or "binder_id" not in input_records.columns:
        return {}
    values: dict[str, int] = {}
    for _, row in input_records.iterrows():
        binder_id = str(row.get("binder_id") or "")
        residues = _numeric_or_zero(row.get("total_residues"))
        if binder_id and residues:
            values[binder_id] = residues
            values[binder_id.lower()] = residues
    return values


def _step_candidate_ids(step: dict) -> set[str]:
    values = {str(value) for value in step.get("candidate_ids") or [] if str(value)}
    values.update({value.lower() for value in list(values)})
    return values


def _step_residue_total(step: dict, residue_by_candidate: dict[str, int]) -> int:
    explicit = _numeric_or_zero(step.get("total_residues"))
    if explicit:
        return explicit
    total = 0
    for candidate_id in [str(value) for value in step.get("candidate_ids") or [] if str(value)]:
        total += int(residue_by_candidate.get(candidate_id) or residue_by_candidate.get(candidate_id.lower()) or 0)
    return total


def _format_matrix_seconds(seconds: float | None) -> str:
    if seconds is None:
        return ""
    return format_duration(float(seconds))


def _parse_timestamp(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _current_step_elapsed_seconds(metadata: dict, step_index: int | None, completed_indices: set[int]) -> float:
    if step_index is None or step_index in completed_indices:
        return 0.0
    started_at = _parse_timestamp(metadata.get("updated_at"))
    if started_at is None:
        return 0.0
    return max(0.0, (datetime.now(timezone.utc) - started_at).total_seconds())


def _engine_estimate_seconds(engine: str, candidate_count: int, total_residues: int) -> float | None:
    if candidate_count <= 0:
        return None
    estimate = estimate_engines(
        engines=[_child_tool_for_matrix_engine(engine) or engine],
        candidate_count=int(candidate_count),
        total_residues=int(total_residues) if total_residues else None,
    )
    rows = estimate.get("rows") or []
    if not rows:
        return None
    value = rows[0].get("estimated_seconds")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _child_runtime_progress(
    parent_run_dir: Path,
    engine: str,
    target_id: str,
    input_records: pd.DataFrame,
) -> dict[str, str]:
    child_run_dir, step_index, steps = _child_engine_state(parent_run_dir, engine)
    if child_run_dir is None or step_index is None or not steps or input_records.empty:
        return {}

    metadata = read_json(child_run_dir / "metadata.json")
    residue_by_candidate = _candidate_residue_map(input_records)
    target_candidate_ids = set(
        input_records.loc[input_records["target_id"].astype(str).eq(str(target_id)), "binder_id"].astype(str)
    )
    target_ids = set(target_candidate_ids)
    target_ids.update({value.lower() for value in list(target_ids)})
    target_step_indices = [
        index
        for index, step in enumerate(steps, start=1)
        if target_ids & _step_candidate_ids(step)
    ]
    if not target_step_indices:
        return {}

    timing_payload = read_json(child_run_dir / "artifacts" / "runtime_step_timings.json")
    timing_rows = timing_payload.get("timings") if isinstance(timing_payload.get("timings"), list) else []
    timings_by_index: dict[int, dict] = {}
    for timing in timing_rows:
        try:
            timings_by_index[int(timing.get("step_index"))] = timing
        except (TypeError, ValueError):
            continue
    completed_indices = set(timings_by_index)
    current_elapsed = _current_step_elapsed_seconds(metadata, step_index, completed_indices)

    def step_seconds(index: int) -> float:
        timing = timings_by_index.get(index) or {}
        try:
            return float(timing.get("seconds") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    completed_seconds = sum(step_seconds(index) for index in completed_indices)
    overall_elapsed = completed_seconds + current_elapsed
    target_completed_seconds = sum(step_seconds(index) for index in target_step_indices if index in completed_indices)
    target_elapsed = target_completed_seconds + (current_elapsed if step_index in target_step_indices else 0.0)

    overall_residues = sum(_step_residue_total(step, residue_by_candidate) for step in steps)
    target_residues = sum(_step_residue_total(steps[index - 1], residue_by_candidate) for index in target_step_indices)
    completed_residues = sum(
        _step_residue_total(steps[index - 1], residue_by_candidate)
        for index in completed_indices
        if 0 < index <= len(steps)
    )
    timed_seconds = sum(step_seconds(index) for index in completed_indices)
    estimated_overall_by_batch = None
    estimated_target_by_batch = None
    if completed_indices and timed_seconds:
        seconds_per_step = timed_seconds / float(max(1, len(completed_indices)))
        estimated_overall_by_batch = seconds_per_step * float(len(steps))
        estimated_target_by_batch = seconds_per_step * float(len(target_step_indices))
    if completed_residues and timed_seconds:
        seconds_per_residue = timed_seconds / float(completed_residues)
        estimated_overall_by_residue = seconds_per_residue * float(overall_residues or 0)
        estimated_target_by_residue = seconds_per_residue * float(target_residues or 0)
        if estimated_overall_by_batch is not None:
            estimated_overall = max(estimated_overall_by_residue, estimated_overall_by_batch)
            estimated_target = max(estimated_target_by_residue, estimated_target_by_batch or 0.0)
            estimate_basis = "current/conservative"
        else:
            estimated_overall = estimated_overall_by_residue
            estimated_target = estimated_target_by_residue
            estimate_basis = "current/residue"
    elif completed_indices:
        estimated_overall = estimated_overall_by_batch
        estimated_target = estimated_target_by_batch
        estimate_basis = "current/batch"
    else:
        estimated_overall = _engine_estimate_seconds(engine, len(input_records), overall_residues)
        estimated_target = _engine_estimate_seconds(engine, len(target_candidate_ids), target_residues)
        estimate_basis = "history/fallback"

    target_done = sum(1 for index in target_step_indices if index < step_index)
    target_current = target_done + (1 if step_index in target_step_indices else 0)
    target_current = max(0, min(target_current, len(target_step_indices)))
    overall_progress = f"{int(step_index)}/{len(steps)}"
    target_progress = f"{target_current}/{len(target_step_indices)}"
    remaining_overall = max(0.0, float(estimated_overall or 0.0) - overall_elapsed) if estimated_overall else None
    remaining_target = max(0.0, float(estimated_target or 0.0) - target_elapsed) if estimated_target else None
    return {
        "batch_progress": target_progress,
        "overall_batch_progress": overall_progress,
        "target_batch_progress": target_progress,
        "elapsed_overall": _format_matrix_seconds(overall_elapsed),
        "elapsed_target": _format_matrix_seconds(target_elapsed),
        "estimated_overall": _format_matrix_seconds(estimated_overall),
        "estimated_target": _format_matrix_seconds(estimated_target),
        "estimated_overall_by_batch": _format_matrix_seconds(estimated_overall_by_batch),
        "estimated_target_by_batch": _format_matrix_seconds(estimated_target_by_batch),
        "estimated_overall_by_residue": _format_matrix_seconds(locals().get("estimated_overall_by_residue")),
        "estimated_target_by_residue": _format_matrix_seconds(locals().get("estimated_target_by_residue")),
        "remaining_overall": _format_matrix_seconds(remaining_overall),
        "remaining_target": _format_matrix_seconds(remaining_target),
        "estimate_basis": estimate_basis,
    }


def _child_target_batch_progress(
    parent_run_dir: Path,
    engine: str,
    target_id: str,
    input_records: pd.DataFrame,
) -> str:
    if input_records.empty:
        return ""
    step_index, steps = _child_engine_step_index(parent_run_dir, engine)
    if step_index is None or not steps:
        return ""
    target_ids = set(
        input_records.loc[input_records["target_id"].astype(str).eq(str(target_id)), "binder_id"].astype(str)
    )
    target_ids.update({value.lower() for value in target_ids})
    if not target_ids:
        return ""
    target_step_indices: list[int] = []
    for index, step in enumerate(steps, start=1):
        candidate_ids = {str(value) for value in step.get("candidate_ids") or [] if str(value)}
        candidate_ids.update({value.lower() for value in list(candidate_ids)})
        if target_ids & candidate_ids:
            target_step_indices.append(index)
    if not target_step_indices:
        return ""
    completed_for_target = sum(1 for index in target_step_indices if index < step_index)
    if step_index in target_step_indices:
        current_for_target = completed_for_target + 1
    else:
        current_for_target = completed_for_target
    total_for_target = len(target_step_indices)
    if current_for_target <= 0:
        return f"0/{total_for_target}"
    return f"{current_for_target}/{total_for_target}"


def _live_benchmark_matrix_cells(rows: list[dict]) -> pd.DataFrame:
    live_rows: list[dict[str, object]] = []
    engine_rank = {engine: index for index, engine in enumerate(BENCHMARK_MATRIX_ENGINE_ORDER)}
    engine_metric_files = {
        "AF3": "alphafast_af3_metrics.csv",
        "AF2-IG": "af2_initial_guess_metrics.csv",
        "Boltz-2": "boltz2_initial_guess_metrics.csv",
        "BoltzGen Fold": "boltzgen_fold_metrics.csv",
        "ColabFold": "colabfold_metrics.csv",
        "ESMFold2": "esmfold2_metrics.csv",
        "OpenFold-3": "openfold3_metrics.csv",
        "Protenix v0.5": "protenix_metrics.csv",
        "Protenix v1": "protenix_v1_metrics.csv",
        "Protenix v2": "protenix_v2_metrics.csv",
        "RF3": "rf3_metrics.csv",
    }
    live_statuses = set(ACTIVE_STATUSES) | {"failed"}
    for row in rows:
        status = str(row.get("status") or "")
        if status not in live_statuses:
            continue
        run_dir = Path(str(row.get("run_dir") or ""))
        input_payload = read_json(run_dir / "input.json")
        job_type = str(input_payload.get("job_type") or "")
        if job_type in {"benchmark_collection", "benchmark_matrix_workspace", "refolding_evaluation"}:
            continue
        params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
        engines = _benchmark_engine_labels_from_params(params)
        if not engines:
            continue
        input_records = _input_records_for_benchmark(run_dir, input_payload)
        target_counts = (
            input_records.groupby("target_id", dropna=False).size().reset_index(name="records")
            if not input_records.empty
            else pd.DataFrame(columns=["target_id", "records"])
        )
        if not target_counts.empty:
            target_counts["target_id"] = target_counts["target_id"].astype(str)
        if target_counts.empty:
            continue
        worker_payload = read_json(run_dir / "worker_request.json")
        worker_kwargs = worker_payload.get("kwargs") if isinstance(worker_payload.get("kwargs"), dict) else {}
        current_phase = str(row.get("current_phase") or "")
        current_engine = str(row.get("current_engine") or "")
        active_targets_by_engine: dict[str, set[str]] = {}

        def matrix_status_for_engine(engine: str, target_id: str) -> str:
            if status == "failed":
                return "failed"
            if status == "queued":
                return "queued"
            metric_name = engine_metric_files.get(engine)
            metric_path = run_dir / "artifacts" / "benchmark" / metric_name if metric_name else None
            if metric_path is not None and metric_path.exists():
                return "completed"
            if current_phase.startswith("Predicting"):
                if current_engine == engine:
                    active_targets = active_targets_by_engine.setdefault(
                        engine,
                        _active_targets_for_child_engine(run_dir, engine, input_records),
                    )
                    if active_targets:
                        return "running" if target_id in active_targets else "queued"
                    return "queued"
                if current_engine in engines and engine in engines:
                    return "completed" if engines.index(engine) < engines.index(current_engine) else "queued"
                return "queued"
            if current_phase.startswith("Calculating benchmark metrics"):
                return "completed"
            return "queued"

        for _, target_row in target_counts.iterrows():
            target_id = str(target_row.get("target_id") or "all targets")
            for engine in engines:
                cell_status = matrix_status_for_engine(engine, target_id)
                runtime_progress: dict[str, str] = {}
                if cell_status == "running":
                    runtime_progress = _child_runtime_progress(
                        run_dir,
                        engine,
                        target_id,
                        input_records,
                    )
                batch_progress = runtime_progress.get("batch_progress", "")
                live_rows.append(
                    {
                        "select": False,
                        "engine": engine,
                        "engine_rank": engine_rank.get(engine, len(engine_rank)),
                        "target_id": target_id,
                        "status": cell_status,
                        "batch_progress": batch_progress,
                        "overall_batch_progress": runtime_progress.get("overall_batch_progress", ""),
                        "target_batch_progress": runtime_progress.get("target_batch_progress", ""),
                        "elapsed_overall": runtime_progress.get("elapsed_overall", ""),
                        "elapsed_target": runtime_progress.get("elapsed_target", ""),
                        "estimated_overall": runtime_progress.get("estimated_overall", ""),
                        "estimated_target": runtime_progress.get("estimated_target", ""),
                        "estimated_overall_by_batch": runtime_progress.get("estimated_overall_by_batch", ""),
                        "estimated_target_by_batch": runtime_progress.get("estimated_target_by_batch", ""),
                        "estimated_overall_by_residue": runtime_progress.get("estimated_overall_by_residue", ""),
                        "estimated_target_by_residue": runtime_progress.get("estimated_target_by_residue", ""),
                        "remaining_overall": runtime_progress.get("remaining_overall", ""),
                        "remaining_target": runtime_progress.get("remaining_target", ""),
                        "estimate_basis": runtime_progress.get("estimate_basis", ""),
                        "records": int(target_row.get("records") or 0),
                        "run_id": str(row.get("run_id") or ""),
                        "job_code": str(row.get("job_code") or ""),
                        "kind": "live benchmark",
                        "dataset": _job_dataset_label(input_payload),
                        "settings": _benchmark_engine_settings(input_payload, worker_kwargs, engine),
                        "description": " | ".join(
                            part
                            for part in [
                                _job_dataset_label(input_payload),
                                _enabled_engines(params),
                                str(row.get("current_phase") or ""),
                                str(row.get("current_engine") or ""),
                                f"target batches {runtime_progress.get('target_batch_progress')}" if runtime_progress.get("target_batch_progress") else "",
                                f"overall batches {runtime_progress.get('overall_batch_progress')}" if runtime_progress.get("overall_batch_progress") else "",
                                f"elapsed target {runtime_progress.get('elapsed_target')}" if runtime_progress.get("elapsed_target") else "",
                                f"ETA target {runtime_progress.get('remaining_target')}" if runtime_progress.get("remaining_target") else "",
                            ]
                            if part
                        ),
                        "created_at": str(row.get("created_at") or ""),
                        "result": result_link("benchmark", str(row.get("run_id")), "Open"),
                    }
                )
    if not live_rows:
        return pd.DataFrame()
    live = pd.DataFrame(live_rows)
    live["created_sort"] = live["created_at"].astype(str)
    live["kind_rank"] = 3
    return live.sort_values(["engine_rank", "target_id", "created_sort"], ascending=[True, True, False]).copy()


def _canonical_benchmark_cells(cells: pd.DataFrame) -> pd.DataFrame:
    if cells.empty:
        return cells.copy()
    ranked = cells.copy()
    status_rank = {"completed": 0, "running": 1, "queued": 2, "failed": 3, "missing": 4}
    ranked["_status_rank"] = ranked["status"].astype(str).map(status_rank).fillna(5).astype(int)
    if "kind_rank" not in ranked.columns:
        ranked["kind_rank"] = ranked["kind"].map({"benchmark": 0, "collection": 1, "live benchmark": 3}).fillna(2)
    if "created_sort" not in ranked.columns:
        ranked["created_sort"] = ranked["created_at"].astype(str)
    ranked = ranked.sort_values(
        ["engine_rank", "target_id", "_status_rank", "kind_rank", "created_sort"],
        ascending=[True, True, True, True, False],
    )
    return ranked.drop_duplicates(["engine", "target_id"], keep="first").drop(columns=["_status_rank"]).copy()


def _clean_checkbox_series(values: pd.Series) -> pd.Series:
    def coerce(value: object) -> bool:
        if value is None:
            return False
        try:
            if pd.isna(value):
                return False
        except (TypeError, ValueError):
            pass
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    return values.map(coerce).astype(bool)


def _installed_overath_targets() -> list[str]:
    if not KNOWN_BENCHMARK_CSV.exists():
        return []
    try:
        summary = _known_benchmark_summary(str(KNOWN_BENCHMARK_CSV))
    except Exception:
        return []
    return sorted(
        target
        for target in summary["target_id"].dropna().astype(str).tolist()
        if target.strip() and target.strip().lower() != "nan"
    )


def _benchmark_matrix_with_missing(
    cells: pd.DataFrame,
    *,
    target_options: list[str],
    engine_options: list[str],
) -> pd.DataFrame:
    engine_rank = {engine: index for index, engine in enumerate(BENCHMARK_MATRIX_ENGINE_ORDER)}
    base = cells.copy()
    existing = set()
    if not base.empty:
        existing = set(zip(base["engine"].astype(str), base["target_id"].astype(str), strict=False))
    missing_rows: list[dict[str, object]] = []
    for engine in engine_options:
        for target_id in target_options:
            if (engine, target_id) in existing:
                continue
            missing_rows.append(
                {
                    "select": False,
                    "engine": engine,
                    "engine_rank": engine_rank.get(engine, len(engine_rank)),
                    "target_id": target_id,
                    "status": "missing",
                    "batch_progress": "",
                    "overall_batch_progress": "",
                    "target_batch_progress": "",
                    "elapsed_overall": "",
                    "elapsed_target": "",
                    "estimated_overall": "",
                    "estimated_target": "",
                    "estimated_overall_by_batch": "",
                    "estimated_target_by_batch": "",
                    "estimated_overall_by_residue": "",
                    "estimated_target_by_residue": "",
                    "remaining_overall": "",
                    "remaining_target": "",
                    "estimate_basis": "",
                    "records": 0,
                    "run_id": "",
                    "job_code": "",
                    "kind": "",
                    "dataset": "",
                    "settings": "",
                    "description": "No completed benchmark result for this target and engine.",
                    "created_at": "",
                    "result": "",
                    "available_runs": 0,
                }
            )
    if missing_rows:
        base = pd.concat([base, pd.DataFrame(missing_rows)], ignore_index=True)
    if base.empty:
        return base
    base["engine_rank"] = base["engine"].map(engine_rank).fillna(len(engine_rank)).astype(int)
    return base.sort_values(["engine_rank", "target_id", "status"], ascending=[True, True, True]).copy()


def _matrix_collection_selections(selected_cells: pd.DataFrame) -> tuple[list[dict[str, object]], list[str]]:
    selections_by_run: dict[str, dict[str, object]] = {}
    all_targets: set[str] = set()
    for row in selected_cells.to_dict(orient="records"):
        run_id = str(row.get("run_id") or "")
        engine = str(row.get("engine") or "")
        target_id = str(row.get("target_id") or "")
        if not run_id or not engine or not target_id:
            continue
        all_targets.add(target_id)
        selection = selections_by_run.setdefault(run_id, {"run_id": run_id, "engines": [], "engine_targets": {}})
        engines = selection["engines"]
        if isinstance(engines, list) and engine not in engines:
            engines.append(engine)
        engine_targets = selection["engine_targets"]
        if isinstance(engine_targets, dict):
            engine_targets.setdefault(engine, [])
            if target_id not in engine_targets[engine]:
                engine_targets[engine].append(target_id)
    selections = list(selections_by_run.values())
    for selection in selections:
        if isinstance(selection.get("engine_targets"), dict):
            selection["engine_targets"] = {
                engine: sorted(targets)
                for engine, targets in selection["engine_targets"].items()
            }
    return selections, sorted(all_targets)


def _benchmark_matrix_workspace_jobs(rows: list[dict]) -> list[dict[str, object]]:
    workspaces: list[dict[str, object]] = []
    for row in rows:
        run_dir = Path(str(row.get("run_dir") or ""))
        input_payload = read_json(run_dir / "input.json")
        if input_payload.get("job_type") != "benchmark_matrix_workspace":
            continue
        result = read_json(run_dir / "result.json")
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        workspace_payload = read_json(run_dir / "artifacts" / "benchmark" / "matrix_workspace.json")
        name = str(workspace_payload.get("name") or metrics.get("workspace_name") or row.get("campaign_name") or "Benchmark matrix workspace")
        workspaces.append(
            {
                "run_id": str(row.get("run_id")),
                "run_dir": run_dir,
                "name": name,
                "status": row.get("status"),
                "cell_count": metrics.get("cell_count"),
                "engine_count": metrics.get("engine_count"),
                "target_count": metrics.get("target_count"),
                "created_at": str(row.get("created_at") or ""),
                "job_code": str(row.get("job_code") or ""),
            }
        )
    return sorted(workspaces, key=lambda item: str(item.get("created_at") or ""), reverse=True)


def _benchmark_matrix_workspace_cells(workspace: dict[str, object] | None, cells: pd.DataFrame) -> pd.DataFrame:
    if workspace is None or cells.empty:
        return cells.iloc[0:0].copy() if not cells.empty else pd.DataFrame()
    sources_path = Path(str(workspace.get("run_dir") or "")) / "artifacts" / "benchmark" / "benchmark_matrix_workspace_sources.csv"
    if not sources_path.exists():
        return cells.iloc[0:0].copy()
    try:
        selected = pd.read_csv(sources_path)
    except Exception:
        return cells.iloc[0:0].copy()
    if selected.empty or not {"run_id", "engine", "target_id"}.issubset(selected.columns):
        return cells.iloc[0:0].copy()
    selected_keys = set(
        zip(
            selected["run_id"].fillna("").astype(str),
            selected["engine"].fillna("").astype(str),
            selected["target_id"].fillna("").astype(str),
            strict=False,
        )
    )
    selected_cells = cells[
        cells.apply(
            lambda row: (str(row.get("run_id") or ""), str(row.get("engine") or ""), str(row.get("target_id") or "")) in selected_keys,
            axis=1,
        )
    ].copy()
    return selected_cells


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
run_openfold3 = False
run_protenix = False
run_protenix_v1 = False
run_protenix_v2 = False
run_boltzgen_fold = False
generate_inputs = True
repo_models: list[str] = ["af3", "boltz", "colabfold"]
run_pyrosetta_input = True
default_pyrosetta_nprocs = max(1, min(64, os.cpu_count() or 1))
pyrosetta_nprocs = default_pyrosetta_nprocs
run_common_interface_metrics = True
run_predicted_rosetta_metrics = True
run_pymol_metrics = True
repo_mode = "hybrid"
repo_max_records = 0
repo_esm_modes = ["initial_guess"]
repo_sampling = 32
repo_loops = 3
repo_seed = 0
af2_recycles = 3
af2_multimer = True
af2_legacy_initial_guess = False
af2_binder_template = bool(st.session_state.get("benchmark_af2ig_binder_template", False))
af2_interface_template = bool(st.session_state.get("benchmark_af2ig_interface_template", False))
boltz2_target_template = bool(st.session_state.get("benchmark_boltz2_template", True))
boltz2_use_target_msa = bool(st.session_state.get("benchmark_boltz2_use_target_msa", True))
boltz2_recycling_steps = 10
boltz2_sampling_steps = 200
boltz2_diffusion_samples = 3
boltz2_write_full_pae = True
rf3_checkpoint_path = str(RF3_CHECKPOINT)
rf3_use_target_msa = bool(st.session_state.get("benchmark_rf3_use_target_msa", True))
rf3_use_target_template = bool(st.session_state.get("benchmark_rf3_use_target_template", True))
rf3_recycles = 10
rf3_num_steps = 50
rf3_diffusion_batch_size = 5
rf3_seed = 0
openfold3_checkpoint_path = str(OPENFOLD3_CHECKPOINT)
openfold3_use_target_msa = bool(st.session_state.get("benchmark_openfold3_use_target_msa", True))
openfold3_num_diffusion_samples = 5
openfold3_num_model_seeds = 1
openfold3_num_recycles = 3
openfold3_use_msa_server = False
protenix_use_msa = bool(st.session_state.get("benchmark_protenix_use_msa", True))
protenix_cycle = 3
protenix_diffusion_steps = 50
protenix_samples = 5
protenix_v1_model_name = PROTENIX_V1_MODEL
protenix_v1_use_msa = bool(st.session_state.get("benchmark_protenix_v1_use_msa", True))
protenix_v1_use_template = bool(st.session_state.get("benchmark_protenix_v1_template", False))
protenix_v1_cycle = 10
protenix_v1_diffusion_steps = 200
protenix_v1_samples = 5
protenix_v2_model_name = PROTENIX_V2_MODEL
protenix_v2_use_msa = bool(st.session_state.get("benchmark_protenix_v2_use_msa", True))
protenix_v2_use_template = bool(st.session_state.get("benchmark_protenix_v2_template", False))
protenix_v2_cycle = 10
protenix_v2_diffusion_steps = 200
protenix_v2_samples = 5
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
require_real_target_msa = bool(st.session_state.get("benchmark_require_real_target_msa", True))
alphafast_db_dir = str(ALPHAFAST_DB_DIR)
alphafast_weights_dir = str(ALPHAFAST_WEIGHTS_DIR)
alphafast_batch_size = 0
alphafast_recycles = 10
alphafast_use_target_templates = bool(st.session_state.get("benchmark_alphafast_use_target_templates", True))
alphafast_use_target_msa = bool(st.session_state.get("benchmark_alphafast_use_target_msa", True))
gpu_device = "0"
simple_modes = ["sequence", "initial_guess"]
simple_max_records = 0
simple_sampling_steps = 68
simple_recycling_loops = 3
simple_seed = 0
simple_contact_cutoff = 8.0
simple_device = "cuda"
label_column = "binder"
metric_max_rows = 0
metric_max_cols = 500
runtime_size = {"candidate_count": 0, "total_residues": None, "max_system_residues": None}
selected_targets: list[str] = []
selected_known_count = 0

tabs = st.tabs(["Dataset", "Engines", "Metrics", "Run", "Results", "Matrix", "Collections"])

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
                if st.session_state.pop("known_benchmark_clear_select_all", False):
                    st.session_state["known_benchmark_select_all"] = False
                    st.session_state["known_benchmark_select_all_previous"] = False
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
                first_per_target_max = max(1, int(subset.groupby("target_id").size().max()))
                first_per_target_key = "known_benchmark_first_per_target_count"
                if int(st.session_state.get(first_per_target_key, 1)) > first_per_target_max:
                    st.session_state[first_per_target_key] = first_per_target_max
                first_per_target_cols = st.columns([1, 1, 4])
                first_per_target_count = first_per_target_cols[0].number_input(
                    "First records per target",
                    min_value=1,
                    max_value=first_per_target_max,
                    value=min(10, first_per_target_max),
                    step=1,
                    key=first_per_target_key,
                    help="Select the first N records for each selected target using the current record table order.",
                )
                if first_per_target_cols[1].button(
                    "Select first N per target",
                    key="known_benchmark_select_first_per_target",
                    help="Uses the current record table order and replaces the existing row selection.",
                ):
                    selected_ids = set(
                        ordered_subset.groupby("target_id", group_keys=False)
                        .head(int(first_per_target_count))["binder_id"]
                        .astype(str)
                    )
                    st.session_state[selected_ids_key] = selected_ids
                    st.session_state["known_benchmark_clear_select_all"] = True
                    st.session_state["known_benchmark_select_all_previous"] = False
                    st.session_state["known_benchmark_editor_version"] = (
                        int(st.session_state.get("known_benchmark_editor_version", 0)) + 1
                    )
                    st.rerun()
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
                runtime_size = {"candidate_count": int(simple_max_records), "total_residues": None, "max_system_residues": None}
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

    engine_disabled = dataset_source in {"Simple ESMFold2 CSV", "Metric table only"}
    benchmark_engine_state = {
        "benchmark_run_alphafast_af3": engine_disabled,
        "benchmark_run_colabfold": engine_disabled,
        "benchmark_run_af2ig": engine_disabled,
        "benchmark_run_esmfold2": dataset_source == "Metric table only",
        "benchmark_run_boltz2": engine_disabled,
        "benchmark_run_rf3": engine_disabled,
        "benchmark_run_openfold3": engine_disabled,
        "benchmark_run_protenix": engine_disabled,
        "benchmark_run_protenix_v1": engine_disabled,
        "benchmark_run_protenix_v2": engine_disabled,
        "benchmark_run_boltzgen_fold": engine_disabled,
    }
    if "benchmark_engine_defaults_all_selected_v1" not in st.session_state:
        for engine_key, disabled in benchmark_engine_state.items():
            st.session_state[engine_key] = not disabled
        st.session_state["benchmark_engine_defaults_all_selected_v1"] = True
    for engine_key, disabled in benchmark_engine_state.items():
        if engine_key not in st.session_state:
            st.session_state[engine_key] = not disabled
        if disabled:
            st.session_state[engine_key] = False

    template_msa_engine_keys = {
        "benchmark_run_alphafast_af3",
        "benchmark_run_colabfold",
        "benchmark_run_esmfold2",
        "benchmark_run_boltz2",
        "benchmark_run_rf3",
        "benchmark_run_protenix_v1",
        "benchmark_run_protenix_v2",
    }
    template_only_engine_keys = set(template_msa_engine_keys)
    template_only_engine_keys.add("benchmark_run_af2ig")
    msa_engine_keys = {
        "benchmark_run_alphafast_af3",
        "benchmark_run_colabfold",
        "benchmark_run_esmfold2",
        "benchmark_run_boltz2",
        "benchmark_run_rf3",
        "benchmark_run_openfold3",
        "benchmark_run_protenix",
        "benchmark_run_protenix_v1",
        "benchmark_run_protenix_v2",
    }
    bulk_cols = st.columns([1, 1, 1.5, 1.5, 1.35, 2.65])
    if bulk_cols[0].button("Select all engines", key="benchmark_select_all_engines"):
        for engine_key, disabled in benchmark_engine_state.items():
            if not disabled:
                st.session_state[engine_key] = True
        st.rerun()
    if bulk_cols[1].button("Deselect all engines", key="benchmark_deselect_all_engines"):
        for engine_key, disabled in benchmark_engine_state.items():
            if not disabled:
                st.session_state[engine_key] = False
        st.rerun()
    if bulk_cols[2].button("Template + MSA engines", key="benchmark_select_template_msa_engines"):
        for engine_key, disabled in benchmark_engine_state.items():
            if not disabled:
                st.session_state[engine_key] = engine_key in template_msa_engine_keys
        st.session_state["benchmark_alphafast_use_target_templates"] = True
        st.session_state["benchmark_alphafast_use_target_msa"] = True
        st.session_state["benchmark_colabfold_use_target_templates"] = True
        st.session_state["benchmark_colabfold_use_target_msa"] = True
        st.session_state["benchmark_af2ig_binder_template"] = True
        st.session_state["benchmark_af2ig_interface_template"] = True
        st.session_state["benchmark_esmfold2_modes"] = ["initial_guess"]
        st.session_state["benchmark_esmfold2_use_target_msa"] = True
        st.session_state["benchmark_boltz2_template"] = True
        st.session_state["benchmark_boltz2_use_target_msa"] = True
        st.session_state["benchmark_rf3_use_target_template"] = True
        st.session_state["benchmark_rf3_use_target_msa"] = True
        st.session_state["benchmark_protenix_v1_template"] = True
        st.session_state["benchmark_protenix_v1_use_msa"] = True
        st.session_state["benchmark_protenix_v2_template"] = True
        st.session_state["benchmark_protenix_v2_use_msa"] = True
        st.session_state["benchmark_openfold3_use_target_msa"] = False
        st.session_state["benchmark_protenix_use_msa"] = False
        st.session_state["benchmark_require_real_target_msa"] = True
        st.rerun()
    if bulk_cols[3].button("Template-only engines", key="benchmark_select_template_only_engines"):
        for engine_key, disabled in benchmark_engine_state.items():
            if not disabled:
                st.session_state[engine_key] = engine_key in template_only_engine_keys
        st.session_state["benchmark_alphafast_use_target_templates"] = True
        st.session_state["benchmark_alphafast_use_target_msa"] = False
        st.session_state["benchmark_colabfold_use_target_templates"] = True
        st.session_state["benchmark_colabfold_use_target_msa"] = False
        st.session_state["benchmark_af2ig_binder_template"] = True
        st.session_state["benchmark_af2ig_interface_template"] = True
        st.session_state["benchmark_esmfold2_modes"] = ["initial_guess"]
        st.session_state["benchmark_esmfold2_use_target_msa"] = False
        st.session_state["benchmark_boltz2_template"] = True
        st.session_state["benchmark_boltz2_use_target_msa"] = False
        st.session_state["benchmark_rf3_use_target_template"] = True
        st.session_state["benchmark_rf3_use_target_msa"] = False
        st.session_state["benchmark_protenix_v1_template"] = True
        st.session_state["benchmark_protenix_v1_use_msa"] = False
        st.session_state["benchmark_protenix_v2_template"] = True
        st.session_state["benchmark_protenix_v2_use_msa"] = False
        st.session_state["benchmark_openfold3_use_target_msa"] = False
        st.session_state["benchmark_protenix_use_msa"] = False
        st.session_state["benchmark_require_real_target_msa"] = False
        st.rerun()
    if bulk_cols[4].button("MSA-only engines", key="benchmark_select_msa_only_engines"):
        for engine_key, disabled in benchmark_engine_state.items():
            if not disabled:
                st.session_state[engine_key] = engine_key in msa_engine_keys
        st.session_state["benchmark_alphafast_use_target_templates"] = False
        st.session_state["benchmark_alphafast_use_target_msa"] = True
        st.session_state["benchmark_colabfold_use_target_templates"] = False
        st.session_state["benchmark_colabfold_use_target_msa"] = True
        st.session_state["benchmark_boltz2_template"] = False
        st.session_state["benchmark_boltz2_use_target_msa"] = True
        st.session_state["benchmark_rf3_use_target_template"] = False
        st.session_state["benchmark_rf3_use_target_msa"] = True
        st.session_state["benchmark_protenix_v1_template"] = False
        st.session_state["benchmark_protenix_v1_use_msa"] = True
        st.session_state["benchmark_protenix_v2_template"] = False
        st.session_state["benchmark_protenix_v2_use_msa"] = True
        st.session_state["benchmark_esmfold2_modes"] = ["sequence"]
        st.session_state["benchmark_esmfold2_use_target_msa"] = True
        st.session_state["benchmark_openfold3_use_target_msa"] = True
        st.session_state["benchmark_protenix_use_msa"] = True
        st.session_state["benchmark_require_real_target_msa"] = True
        st.rerun()
    if bulk_cols[5].button("No MSA + no template", key="benchmark_disable_msa_template"):
        st.session_state["benchmark_run_af2ig"] = False
        st.session_state["benchmark_alphafast_use_target_templates"] = False
        st.session_state["benchmark_alphafast_use_target_msa"] = False
        st.session_state["benchmark_colabfold_use_target_templates"] = False
        st.session_state["benchmark_colabfold_use_target_msa"] = False
        st.session_state["benchmark_af2ig_legacy_initial_guess"] = False
        st.session_state["benchmark_af2ig_binder_template"] = False
        st.session_state["benchmark_af2ig_interface_template"] = False
        st.session_state["benchmark_esmfold2_modes"] = ["sequence"]
        st.session_state["benchmark_esmfold2_use_target_msa"] = False
        st.session_state["benchmark_boltz2_template"] = False
        st.session_state["benchmark_boltz2_use_target_msa"] = False
        st.session_state["benchmark_rf3_use_target_template"] = False
        st.session_state["benchmark_rf3_use_target_msa"] = False
        st.session_state["benchmark_openfold3_use_target_msa"] = False
        st.session_state["benchmark_protenix_use_msa"] = False
        st.session_state["benchmark_protenix_v1_template"] = False
        st.session_state["benchmark_protenix_v1_use_msa"] = False
        st.session_state["benchmark_protenix_v2_template"] = False
        st.session_state["benchmark_protenix_v2_use_msa"] = False
        st.session_state["benchmark_require_real_target_msa"] = False
        st.rerun()

    engine_cols = st.columns(11)
    with engine_cols[0]:
        run_alphafast_af3 = st.checkbox("AlphaFast AF3", value=True, disabled=engine_disabled, key="benchmark_run_alphafast_af3")
    with engine_cols[1]:
        run_colabfold = st.checkbox("ColabFold", value=True, disabled=engine_disabled, key="benchmark_run_colabfold")
    with engine_cols[2]:
        run_repo_af2ig = st.checkbox("AF2 initial guess", value=True, disabled=engine_disabled, key="benchmark_run_af2ig")
    with engine_cols[3]:
        run_repo_esm = st.checkbox("ESMFold2", value=True, disabled=dataset_source == "Metric table only", key="benchmark_run_esmfold2")
    with engine_cols[4]:
        run_boltz2_ig = st.checkbox("Boltz-2", value=True, disabled=engine_disabled, key="benchmark_run_boltz2")
    with engine_cols[5]:
        run_rf3 = st.checkbox("RF3", value=True, disabled=engine_disabled, key="benchmark_run_rf3")
    with engine_cols[6]:
        run_openfold3 = st.checkbox("OpenFold-3", value=True, disabled=engine_disabled, key="benchmark_run_openfold3")
    with engine_cols[7]:
        run_protenix = st.checkbox("Protenix v0.5", value=True, disabled=engine_disabled, key="benchmark_run_protenix")
    with engine_cols[8]:
        run_protenix_v1 = st.checkbox("Protenix v1", value=True, disabled=engine_disabled, key="benchmark_run_protenix_v1")
    with engine_cols[9]:
        run_protenix_v2 = st.checkbox("Protenix v2", value=True, disabled=engine_disabled, key="benchmark_run_protenix_v2")
    with engine_cols[10]:
        run_boltzgen_fold = st.checkbox(
            "BoltzGen fold",
            value=True,
            disabled=engine_disabled,
            key="benchmark_run_boltzgen_fold",
            help="Runs BoltzGen's target-template-conditioned folding stage. This is not MSA-based sequence-only refolding.",
        )

    if dataset_source not in {"Simple ESMFold2 CSV", "Metric table only"}:
        has_repo_csv = bool(
            known_csv_text
            or repo_csv_text
            or repo_input_csv_path.strip()
            or repo_input_csv is not None
            or repo_zip is not None
            or repo_zip_path.strip()
        )
        has_repo_pdbs = bool(
            known_pdb_dir
            or repo_pdb_dir.strip()
            or repo_zip is not None
            or repo_zip_path.strip()
        )
        if has_repo_csv and has_repo_pdbs:
            repo_mode = "hybrid"
            st.caption("Input mode: CSV rows define the benchmark records; PDBs provide the structures.")
        elif has_repo_csv:
            repo_mode = "seq_only_csv"
            st.caption("Input mode: sequence CSV only.")
        else:
            repo_mode = "pdb_only"
            st.caption("Input mode: PDB folder only.")
        selected_record_count = int(runtime_size.get("candidate_count") or 0)
        if selected_record_count:
            st.caption(f"Prediction engines will process all {selected_record_count:,} checked records.")
        generate_inputs = True

        with st.expander("MSA Reference Data", expanded=True):
            msa_consuming_engine_selected = bool(
                (run_alphafast_af3 and alphafast_use_target_msa)
                or (run_colabfold and bool(st.session_state.get("benchmark_colabfold_use_target_msa", True)))
                or (run_boltz2_ig and boltz2_use_target_msa)
                or (run_repo_esm and esmfold2_use_target_msa)
                or (run_rf3 and rf3_use_target_msa)
                or (run_openfold3 and openfold3_use_target_msa)
                or (run_protenix and protenix_use_msa)
                or (run_protenix_v1 and protenix_v1_use_msa)
                or (run_protenix_v2 and protenix_v2_use_msa)
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
            require_real_target_msa = st.checkbox(
                "Require real target MSAs before prediction",
                value=True,
                key="benchmark_require_real_target_msa",
                disabled=not msa_consuming_engine_selected,
                help=(
                    "When enabled, every selected target chain must have an A3M with more than the query sequence. "
                    "Repository hits that are query-only are regenerated with AlphaFast/MMseqs when that source is enabled; "
                    "if a real MSA still cannot be produced, the job stops before GPU prediction."
                ),
            )
        default_models = []
        if run_alphafast_af3 or run_colabfold:
            default_models.append("af3")
        if run_boltz2_ig:
            default_models.append("boltz")
        if run_colabfold:
            default_models.append("colabfold")
        if run_protenix:
            default_models.append("protenix")
        if run_protenix_v1:
            default_models.append("protenix_v1")
        if run_protenix_v2:
            default_models.append("protenix_v2")
        repo_models = default_models or ["af3", "boltz", "colabfold"]

        with st.expander("AlphaFast AF3 Settings", expanded=run_alphafast_af3):
            af3_cols = st.columns(4)
            alphafast_weights_dir = af3_cols[0].text_input("AF3 weights dir", value=str(ALPHAFAST_WEIGHTS_DIR), disabled=not run_alphafast_af3)
            alphafast_recycles = af3_cols[1].number_input("AF3 recycles", min_value=1, max_value=48, value=10, step=1, key="benchmark_alphafast_recycles", disabled=not run_alphafast_af3)
            alphafast_use_target_templates = af3_cols[2].checkbox(
                "Use templates",
                value=alphafast_use_target_templates,
                key="benchmark_alphafast_use_target_templates",
                disabled=not run_alphafast_af3,
                help="Embeds the staged input target chains as AF3 templates. Binder chains and binder-interface geometry are not templated.",
            )
            alphafast_use_target_msa = af3_cols[3].checkbox(
                "Use target MSAs",
                value=alphafast_use_target_msa,
                key="benchmark_alphafast_use_target_msa",
                disabled=not run_alphafast_af3,
                help="When off, AlphaFast AF3 runs with query-only/no target MSA input.",
            )

        with st.expander("ColabFold Settings", expanded=run_colabfold):
            colab_cols = st.columns(5)
            colabfold_cache_dir = colab_cols[0].text_input("ColabFold / AF2 model cache", value=str(COLABFOLD_CACHE_DIR), disabled=not run_colabfold)
            colabfold_recycles = colab_cols[1].number_input("ColabFold recycles", min_value=1, max_value=48, value=3, step=1, key="benchmark_colabfold_recycles", disabled=not run_colabfold)
            colabfold_models = colab_cols[2].number_input("ColabFold models", min_value=1, max_value=5, value=3, step=1, key="benchmark_colabfold_models", disabled=not run_colabfold)
            colabfold_use_target_templates = colab_cols[3].checkbox(
                "Use templates",
                value=colabfold_use_target_templates,
                key="benchmark_colabfold_use_target_templates",
                disabled=not run_colabfold,
                help="Uses the staged input target as the template. Binder chains and binder-interface geometry are not templated.",
            )
            colabfold_use_target_msa = colab_cols[4].checkbox(
                "Use target MSAs",
                value=bool(st.session_state.get("benchmark_colabfold_use_target_msa", True)),
                key="benchmark_colabfold_use_target_msa",
                disabled=not run_colabfold,
                help="When off, ColabFold does not request or inject real target MSAs.",
            )
            colabfold_max_template_hits = 4

        with st.expander("AF2 Target-Only Initial Guess Settings", expanded=run_repo_af2ig):
            af2_cols = st.columns(4)
            af2_recycles = af2_cols[0].number_input("AF2-IG recycles", min_value=1, max_value=24, value=3, step=1, key="benchmark_af2ig_recycles", disabled=not run_repo_af2ig)
            af2_multimer = af2_cols[1].checkbox("AF2 multimer", value=True, disabled=not run_repo_af2ig)
            af2_legacy_initial_guess = af2_cols[2].checkbox(
                "Whole-complex initial guess (legacy)",
                value=False,
                key="benchmark_af2ig_legacy_initial_guess",
                disabled=not run_repo_af2ig,
                help="Opt-in compatibility mode. This consumes the staged binder-target coordinates. Leave off for binder sequence plus selected target template only.",
            )
            af2_binder_template = af2_cols[3].checkbox(
                "Binder template (legacy)",
                value=False,
                key="benchmark_af2ig_binder_template",
                disabled=not run_repo_af2ig or not af2_legacy_initial_guess,
            )
            af2_interface_template = st.checkbox(
                "Preserve template interface geometry (legacy)",
                value=False,
                key="benchmark_af2ig_interface_template",
                disabled=not run_repo_af2ig or not af2_legacy_initial_guess or not af2_binder_template,
            )
            if not af2_legacy_initial_guess:
                st.caption("Default: the selected target PDB is the target-only initial guess. Binder coordinates and the original input complex interface are not supplied to prediction.")

        with st.expander("ESMFold2 Settings", expanded=run_repo_esm):
            repo_esm_modes = st.multiselect(
                "Modes",
                ["sequence", "initial_guess"],
                default=["initial_guess"],
                key="benchmark_esmfold2_modes",
                disabled=not run_repo_esm,
                format_func={"sequence": "Sequence only", "initial_guess": "Initial guess"}.get,
            )
            esmfold2_preset_selector(
                key="benchmark_repo",
                steps_key="benchmark_esmfold2_sampling_steps",
                loops_key="benchmark_esmfold2_recycling_loops",
                disabled=not run_repo_esm,
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
            repo_sampling = esm_cols[2].number_input("ESMFold2 sampling steps", min_value=1, max_value=256, value=68, step=1, key="benchmark_esmfold2_sampling_steps", disabled=not run_repo_esm)
            repo_loops = esm_cols[3].number_input("ESMFold2 recycling loops", min_value=1, max_value=64, value=10, step=1, key="benchmark_esmfold2_recycling_loops", disabled=not run_repo_esm)
            repo_seed = esm_cols[4].number_input("ESMFold2 seed", min_value=0, max_value=999999, value=0, step=1, key="benchmark_esmfold2_seed", disabled=not run_repo_esm)

        with st.expander("Boltz-2 Settings", expanded=run_boltz2_ig):
            boltz_cols = st.columns(6)
            boltz_cols[0].text_input("Boltz-2 model cache", value=str(BOLTZ_MODELS_DIR), disabled=True)
            boltz2_target_template = boltz_cols[1].checkbox(
                "Use templates",
                value=boltz2_target_template,
                key="benchmark_boltz2_template",
                disabled=not run_boltz2_ig,
                help="Uses the staged input target as the template.",
            )
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
            rf3_cols = st.columns(7)
            rf3_checkpoint_path = rf3_cols[0].text_input("RF3 checkpoint", value=str(RF3_CHECKPOINT), disabled=not run_rf3)
            rf3_use_target_template = rf3_cols[1].checkbox(
                "Use templates",
                value=rf3_use_target_template,
                key="benchmark_rf3_use_target_template",
                disabled=not run_rf3,
                help="Uses the staged input target chains as RF3 template coordinates; binder chains remain untemplated.",
            )
            rf3_use_target_msa = rf3_cols[2].checkbox(
                "Use target MSAs",
                value=True,
                key="benchmark_rf3_use_target_msa",
                disabled=not run_rf3,
                help="Default on. Each declared target chain receives its prepared A3M; binder chains remain MSA-free.",
            )
            rf3_recycles = rf3_cols[3].number_input("RF3 recycles", min_value=1, max_value=48, value=10, step=1, key="benchmark_rf3_recycles", disabled=not run_rf3)
            rf3_num_steps = rf3_cols[4].number_input("RF3 diffusion steps", min_value=1, max_value=1000, value=50, step=1, key="benchmark_rf3_num_steps", disabled=not run_rf3)
            rf3_diffusion_batch_size = rf3_cols[5].number_input("RF3 samples", min_value=1, max_value=20, value=5, step=1, key="benchmark_rf3_diffusion_batch_size", disabled=not run_rf3)
            rf3_seed = rf3_cols[6].number_input("RF3 seed", min_value=0, max_value=999999, value=0, step=1, key="benchmark_rf3_seed", disabled=not run_rf3)
            st.caption("RF3 folds the complex from sequences plus optional target-chain templates and per-chain target MSAs.")

        with st.expander("OpenFold-3 Settings", expanded=run_openfold3):
            openfold_cols = st.columns(6)
            openfold3_checkpoint_path = openfold_cols[0].text_input(
                "OpenFold-3 checkpoint",
                value=str(OPENFOLD3_CHECKPOINT),
                key="benchmark_openfold3_checkpoint",
                disabled=not run_openfold3,
                help="Default shared path: /mnt/db/reference_files/openfold3/of3-p2-155k.pt.",
            )
            openfold3_use_target_msa = openfold_cols[1].checkbox(
                "Use target MSAs",
                value=True,
                key="benchmark_openfold3_use_target_msa",
                disabled=not run_openfold3,
                help="Default on. Binder chains remain MSA-free; declared target chains use prepared A3M files when available.",
            )
            openfold3_num_diffusion_samples = openfold_cols[2].number_input(
                "Diffusion samples",
                min_value=1,
                max_value=20,
                value=5,
                step=1,
                key="benchmark_openfold3_samples",
                disabled=not run_openfold3,
            )
            openfold3_num_model_seeds = openfold_cols[3].number_input(
                "Model seeds",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
                key="benchmark_openfold3_seeds",
                disabled=not run_openfold3,
            )
            openfold3_num_recycles = openfold_cols[4].number_input(
                "Recycles",
                min_value=1,
                max_value=48,
                value=3,
                step=1,
                key="benchmark_openfold3_recycles",
                disabled=not run_openfold3,
                help="OpenFold-3 default architecture.shared.num_recycles is 3.",
            )
            openfold3_use_msa_server = openfold_cols[5].checkbox(
                "Use MSA server",
                value=False,
                key="benchmark_openfold3_msa_server",
                disabled=not run_openfold3,
                help="Default off for local/high-throughput runs. Prefer the app's shared target-MSA repository.",
            )
            st.caption("OpenFold-3 runs from chain sequences. Target MSAs are attached as precomputed main MSAs; binder chains are supplied without MSAs.")

        with st.expander("Protenix v0.5 Settings", expanded=run_protenix):
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
            st.caption("Legacy PXDesign-backed Protenix v0.5 adapter.")

        with st.expander("Protenix v1 Settings", expanded=run_protenix_v1):
            protenix_v1_cols = st.columns(6)
            protenix_v1_model_name = protenix_v1_cols[0].selectbox(
                "Model",
                [PROTENIX_V1_MODEL, PROTENIX_V1_20250630_MODEL],
                index=0,
                disabled=not run_protenix_v1,
                key="benchmark_protenix_v1_model",
            )
            protenix_v1_use_msa = protenix_v1_cols[1].checkbox("Use target MSAs", value=True, key="benchmark_protenix_v1_use_msa", disabled=not run_protenix_v1)
            protenix_v1_use_template = protenix_v1_cols[2].checkbox("Use templates", value=protenix_v1_use_template, key="benchmark_protenix_v1_template", disabled=not run_protenix_v1)
            protenix_v1_cycle = protenix_v1_cols[3].number_input("Pairformer cycles", min_value=1, max_value=48, value=10, step=1, key="benchmark_protenix_v1_cycle", disabled=not run_protenix_v1)
            protenix_v1_diffusion_steps = protenix_v1_cols[4].number_input("Diffusion steps", min_value=1, max_value=1000, value=200, step=1, key="benchmark_protenix_v1_steps", disabled=not run_protenix_v1)
            protenix_v1_samples = protenix_v1_cols[5].number_input("Samples", min_value=1, max_value=20, value=5, step=1, key="benchmark_protenix_v1_samples", disabled=not run_protenix_v1)
            st.caption("Standalone Protenix CLI using /mnt/db/reference_files/protenix for checkpoints and data.")

        with st.expander("Protenix v2 Settings", expanded=run_protenix_v2):
            protenix_v2_cols = st.columns(6)
            protenix_v2_model_name = protenix_v2_cols[0].text_input("Model", value=PROTENIX_V2_MODEL, disabled=not run_protenix_v2, key="benchmark_protenix_v2_model")
            protenix_v2_use_msa = protenix_v2_cols[1].checkbox("Use target MSAs", value=True, key="benchmark_protenix_v2_use_msa", disabled=not run_protenix_v2)
            protenix_v2_use_template = protenix_v2_cols[2].checkbox("Use templates", value=protenix_v2_use_template, key="benchmark_protenix_v2_template", disabled=not run_protenix_v2)
            protenix_v2_cycle = protenix_v2_cols[3].number_input("Pairformer cycles", min_value=1, max_value=48, value=10, step=1, key="benchmark_protenix_v2_cycle", disabled=not run_protenix_v2)
            protenix_v2_diffusion_steps = protenix_v2_cols[4].number_input("Diffusion steps", min_value=1, max_value=1000, value=200, step=1, key="benchmark_protenix_v2_steps", disabled=not run_protenix_v2)
            protenix_v2_samples = protenix_v2_cols[5].number_input("Samples", min_value=1, max_value=20, value=5, step=1, key="benchmark_protenix_v2_samples", disabled=not run_protenix_v2)
            st.caption("Standalone Protenix v2 CLI using /mnt/db/reference_files/protenix for checkpoints and data.")

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
            esmfold2_preset_selector(
                key="benchmark_simple",
                steps_key="benchmark_simple_sampling_steps",
                loops_key="benchmark_simple_recycling_loops",
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
            simple_sampling_steps = simple_cols[1].number_input("Sampling steps", min_value=1, max_value=256, value=68, step=1, key="benchmark_simple_sampling_steps")
            simple_recycling_loops = simple_cols[2].number_input("Recycling loops", min_value=1, max_value=64, value=10, step=1, key="benchmark_simple_recycling_loops")
            simple_seed = simple_cols[3].number_input("Seed", min_value=0, max_value=999999, value=0, step=1, key="benchmark_simple_seed")
            simple_contact_cutoff = simple_cols[4].number_input("Contact cutoff", min_value=2.0, max_value=20.0, value=8.0, step=0.5, key="benchmark_simple_contact_cutoff")
            simple_device = simple_cols[5].selectbox("Device", ["cuda", "auto"], index=0)
            st.caption("Biohub ESMFold2 requires CUDA and a scheduled GPU allocation.")

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
    with st.expander("Compute", expanded=True):
        gpu_device = gpu_run_panel(key="benchmark", default="0")

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
    if run_openfold3:
        selected_engines.append("OpenFold-3")
    if run_protenix:
        selected_engines.append("Protenix v0.5")
    if run_protenix_v1:
        selected_engines.append("Protenix v1")
    if run_protenix_v2:
        selected_engines.append("Protenix v2")
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
            st.caption(
                f"Run benchmark will queue one combined benchmark job for "
                f"{len(selected_targets):,} selected target(s), {selected_count:,} record(s), "
                f"and {len(selected_engines):,} selected engine(s)."
            )
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
    if run_openfold3:
        estimate_engine_keys.append("openfold3")
    if run_protenix:
        estimate_engine_keys.append("protenix")
    if run_protenix_v1:
        estimate_engine_keys.append("protenix_v1")
    if run_protenix_v2:
        estimate_engine_keys.append("protenix_v2")
    if run_boltzgen_fold:
        estimate_engine_keys.append("boltzgen_fold")
    if run_common_interface_metrics or run_predicted_rosetta_metrics or run_pymol_metrics:
        estimate_engine_keys.append("postprocessing")

    estimate_count = int(runtime_size.get("candidate_count") or 0)
    estimate_total_residues = runtime_size.get("total_residues")
    estimate_engine_params: dict[str, dict[str, int]] = {}
    if "esmfold2" in estimate_engine_keys:
        estimate_engine_params["esmfold2"] = {
            "num_loops": int(simple_recycling_loops if dataset_source == "Simple ESMFold2 CSV" else repo_loops),
            "num_sampling_steps": int(simple_sampling_steps if dataset_source == "Simple ESMFold2 CSV" else repo_sampling),
        }
    if estimate_engine_keys and estimate_count:
        estimate = estimate_engines(
            engines=estimate_engine_keys,
            candidate_count=estimate_count,
            total_residues=int(estimate_total_residues) if estimate_total_residues else None,
            engine_params=estimate_engine_params,
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
                        "seconds_per_residue": st.column_config.NumberColumn("sec / residue", format="%.3f"),
                    },
                )
            st.caption(
                "Estimates use previous completed jobs when available and fallback rates otherwise. "
                "They improve as more benchmark runs finish."
            )
    elif estimate_engine_keys:
        st.caption("Runtime estimate needs a candidate count. Select a target or provide a previewable CSV.")
    capacity_engine_keys = [engine for engine in estimate_engine_keys if engine != "postprocessing"]
    estimate_max_system_residues = runtime_size.get("max_system_residues")
    capacity_preset = (
        "full_workflow"
        if (run_common_interface_metrics or run_pyrosetta_input or run_predicted_rosetta_metrics or run_pymol_metrics)
        else "practical"
    )
    if capacity_engine_keys and estimate_max_system_residues:
        with st.expander("Capacity warning", expanded=True):
            st.caption(
                f"Largest selected system: {int(estimate_max_system_residues):,} residues. "
                "Warnings use completed Capacity Benchmark runs with the same GPU and comparable run depth."
            )
            _show_capacity_warnings(
                capacity_warnings(
                    engine_keys=capacity_engine_keys,
                    total_length=int(estimate_max_system_residues),
                    matrix_mode="sequence_copy_multimer",
                    preset=capacity_preset,
                    gpu_device=str(gpu_device),
                )
            )
    elif capacity_engine_keys and estimate_count:
        st.caption("Capacity warning needs per-record lengths. Installed benchmark records provide this automatically.")

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
                        gpu_device=gpu_device,
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
                    af2_use_initial_guess=bool(af2_legacy_initial_guess),
                    af2_use_binder_template=bool(af2_legacy_initial_guess and af2_binder_template),
                    af2_use_interface_template=bool(
                        af2_legacy_initial_guess
                        and af2_binder_template
                        and af2_interface_template
                    ),
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
                    rf3_use_target_template=bool(rf3_use_target_template),
                    rf3_recycles=int(rf3_recycles),
                    rf3_num_steps=int(rf3_num_steps),
                    rf3_diffusion_batch_size=int(rf3_diffusion_batch_size),
                    rf3_seed=int(rf3_seed),
                    run_openfold3=bool(run_openfold3),
                    openfold3_checkpoint_path=Path(openfold3_checkpoint_path).expanduser(),
                    openfold3_use_target_msa=bool(openfold3_use_target_msa),
                    openfold3_num_diffusion_samples=int(openfold3_num_diffusion_samples),
                    openfold3_num_model_seeds=int(openfold3_num_model_seeds),
                    openfold3_num_recycles=int(openfold3_num_recycles),
                    openfold3_use_msa_server=bool(openfold3_use_msa_server),
                    run_protenix=bool(run_protenix),
                    protenix_use_msa=bool(protenix_use_msa),
                    protenix_cycle=int(protenix_cycle),
                    protenix_diffusion_steps=int(protenix_diffusion_steps),
                    protenix_samples=int(protenix_samples),
                    run_protenix_v1=bool(run_protenix_v1),
                    protenix_v1_model_name=str(protenix_v1_model_name),
                    protenix_v1_use_msa=bool(protenix_v1_use_msa),
                    protenix_v1_use_template=bool(protenix_v1_use_template),
                    protenix_v1_use_default_params=True,
                    protenix_v1_cycle=int(protenix_v1_cycle),
                    protenix_v1_diffusion_steps=int(protenix_v1_diffusion_steps),
                    protenix_v1_samples=int(protenix_v1_samples),
                    run_protenix_v2=bool(run_protenix_v2),
                    protenix_v2_model_name=str(protenix_v2_model_name),
                    protenix_v2_use_msa=bool(protenix_v2_use_msa),
                    protenix_v2_use_template=bool(protenix_v2_use_template),
                    protenix_v2_use_default_params=True,
                    protenix_v2_cycle=int(protenix_v2_cycle),
                    protenix_v2_diffusion_steps=int(protenix_v2_diffusion_steps),
                    protenix_v2_samples=int(protenix_v2_samples),
                    run_boltzgen_fold=bool(run_boltzgen_fold),
                    boltzgen_recycling_steps=int(boltzgen_recycling_steps),
                    boltzgen_sampling_steps=int(boltzgen_sampling_steps),
                    boltzgen_diffusion_samples=int(boltzgen_diffusion_samples),
                    run_colabfold=bool(run_colabfold),
                    colabfold_cache_dir=Path(colabfold_cache_dir).expanduser(),
                    colabfold_msa_source=str(colabfold_msa_source if colabfold_use_target_msa else "repo_run_csv"),
                    msa_repository_dir=Path(msa_repository_dir).expanduser(),
                    require_real_target_msa=bool(require_real_target_msa),
                    colabfold_num_recycles=int(colabfold_recycles),
                    colabfold_num_models=int(colabfold_models),
                    colabfold_use_target_templates=bool(colabfold_use_target_templates),
                    colabfold_use_target_msa=bool(colabfold_use_target_msa),
                    colabfold_max_template_hits=int(colabfold_max_template_hits),
                    colabfold_gpu_device=gpu_device,
                    run_alphafast_af3=bool(run_alphafast_af3),
                    alphafast_db_dir=Path(alphafast_db_dir).expanduser(),
                    alphafast_weights_dir=Path(alphafast_weights_dir).expanduser(),
                    alphafast_batch_size=int(alphafast_batch_size),
                    alphafast_num_recycles=int(alphafast_recycles),
                    alphafast_use_target_templates=bool(alphafast_use_target_templates),
                    alphafast_query_only_msa=not bool(alphafast_use_target_msa),
                    alphafast_gpu_device=gpu_device,
                    gpu_device=gpu_device,
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
    refresh_results_button("benchmark_refresh_results")
    rows = collect_jobs("benchmark")
    if not rows:
        st.info("No benchmark jobs yet.")
    else:
        df = _benchmark_job_table(rows)
        if df.empty:
            st.info("No benchmark jobs yet. Refolding evaluation runs are listed under Refolding / Validation.")
        else:
            display_cols = [
                "job_code_link",
                "kind",
                "description",
                "evidence_mode",
                "evidence_detail",
                "status",
                "recovery",
                "esmfold2_preset",
                "esmfold2_loops",
                "esmfold2_steps",
                "records",
                "top_feature",
                "top_ap",
                "plots",
                "created_at",
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
                    "job_code_link": st.column_config.LinkColumn("job_code", display_text=r"job_code=([^&]+)"),
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
            run_dirs_by_id = {str(row.get("run_id")): Path(str(row.get("run_dir"))) for row in rows}
            recovery_key = f"{table_key}_recovery_report"
            if len(selected_rows) == 1:
                selected_run_id = str(selected_rows.iloc[0]["run_id"])
                selected_run_dir = run_dirs_by_id.get(selected_run_id)
                if selected_run_dir is not None:
                    if st.button("Check selected job completion", key=f"{table_key}_inspect_job"):
                        try:
                            st.session_state[recovery_key] = inspect_benchmark_run(selected_run_dir)
                        except Exception as exc:
                            st.error(f"Could not inspect the selected benchmark: {exc}")
            elif len(selected_rows) > 1:
                st.caption("Select one benchmark row to inspect or resume it.")

            recovery_report = st.session_state.get(recovery_key)
            selected_run_ids = set(selected_rows["run_id"].astype(str)) if not selected_rows.empty else set()
            if recovery_report and str(recovery_report.get("run_id")) in selected_run_ids:
                completed = int(recovery_report.get("completed_stage_count") or 0)
                requested = int(recovery_report.get("requested_stage_count") or 0)
                first_incomplete = recovery_report.get("first_incomplete_stage")
                if recovery_report.get("finished_correctly"):
                    st.success(f"Job {recovery_report.get('job_code')} finished correctly ({completed}/{requested} stages complete).")
                else:
                    state = "stopped/stale" if recovery_report.get("stale") else str(recovery_report.get("status") or "incomplete")
                    st.warning(
                        f"Job {recovery_report.get('job_code')} is {state}: {completed}/{requested} stages are complete. "
                        f"Resume point: {first_incomplete or 'unknown'}."
                    )
                stage_rows = []
                for stage in recovery_report.get("stages") or []:
                    if not stage.get("requested"):
                        continue
                    stage_rows.append(
                        {
                            "stage": stage.get("stage"),
                            "status": "complete" if stage.get("complete") else "missing/incomplete",
                            "records": stage.get("records"),
                            "expected": stage.get("expected_records"),
                        }
                    )
                st.dataframe(pd.DataFrame(stage_rows), width="stretch", hide_index=True)
                free_gib = float(recovery_report.get("free_bytes") or 0) / (1024**3)
                required_gib = float(recovery_report.get("required_free_bytes") or 0) / (1024**3)
                st.caption(
                    f"Free disk space: {free_gib:.1f} GiB; conservative resume requirement: {required_gib:.1f} GiB. "
                    "Existing completed engine outputs will be reused."
                )
                currently_active = str(recovery_report.get("status")) in ACTIVE_STATUSES and not recovery_report.get("stale")
                resume_disabled = (
                    not recovery_report.get("can_resume")
                    or currently_active
                    or free_gib < required_gib
                )
                if free_gib < required_gib:
                    st.error(f"Free at least {required_gib:.1f} GiB before resuming this benchmark.")
                resume_run_dir = run_dirs_by_id.get(str(recovery_report.get("run_id")))
                resume_input = read_json(resume_run_dir / "input.json") if resume_run_dir is not None else {}
                resume_params = resume_input.get("params") if isinstance(resume_input.get("params"), dict) else {}
                resume_worker = read_json(resume_run_dir / "worker_request.json") if resume_run_dir is not None else {}
                resume_kwargs = resume_worker.get("kwargs") if isinstance(resume_worker.get("kwargs"), dict) else {}
                current_resume_gpu = str(
                    resume_params.get("gpu_device")
                    or resume_kwargs.get("gpu_device")
                    or resume_kwargs.get("alphafast_gpu_device")
                    or "0"
                )
                with st.expander("Resume compute", expanded=True):
                    resume_gpu_device = gpu_run_panel(
                        key=f"benchmark_resume_{recovery_report.get('run_id')}",
                        default=current_resume_gpu,
                    )
                if st.button(
                    f"Resume from {first_incomplete or 'incomplete stage'}",
                    type="primary",
                    disabled=resume_disabled,
                    key=f"{table_key}_resume_job",
                ):
                    try:
                        prepare_benchmark_run_resume(resume_run_dir, gpu_device=resume_gpu_device)
                        spawn_worker_for_run(resume_run_dir)
                        st.session_state.pop(recovery_key, None)
                        st.success(f"Benchmark resume queued on GPU {resume_gpu_device}. Completed engine tables will be reused.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Could not resume the benchmark: {exc}")

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
    st.subheader("Benchmark Matrix")
    st.caption(
        "Engines are rows and targets are columns. The auto current master updates from all benchmark runs; "
        "saved umbrellas are frozen snapshots used to build reproducible collections."
    )
    rows = collect_jobs("benchmark")
    overath_targets = _installed_overath_targets()
    overath_target_set = set(overath_targets)
    source_rows, unavailable_rows = _available_benchmark_sources(rows)
    cells = _benchmark_matrix_cells(source_rows)
    live_cells = _live_benchmark_matrix_cells(rows)
    if overath_target_set:
        if not cells.empty:
            cells = cells[cells["target_id"].astype(str).isin(overath_target_set)].copy()
        if not live_cells.empty:
            live_cells = live_cells[live_cells["target_id"].astype(str).isin(overath_target_set)].copy()
    if not live_cells.empty:
        cells = pd.concat([cells, live_cells], ignore_index=True, sort=False) if not cells.empty else live_cells.copy()
        cells["created_sort"] = cells["created_at"].astype(str)
        cells["kind_rank"] = cells["kind"].map({"benchmark": 0, "collection": 1, "live benchmark": 3}).fillna(2)
        cells = cells.sort_values(
            ["engine_rank", "target_id", "created_sort", "kind_rank"],
            ascending=[True, True, False, True],
        ).copy()
    workspaces = _benchmark_matrix_workspace_jobs(rows)
    workspace_options = ["__auto__"] + [str(row["run_id"]) for row in workspaces]
    workspace_by_id = {str(row["run_id"]): row for row in workspaces}
    selected_workspace_id = st.selectbox(
        "Matrix umbrella",
        workspace_options,
        index=0,
        format_func=lambda value: (
            "Auto current master (live coverage)"
            if value == "__auto__"
            else f"Saved snapshot: {workspace_by_id.get(str(value), {}).get('name', 'Benchmark matrix workspace')} | "
            f"{workspace_by_id.get(str(value), {}).get('job_code', '')} | {value}"
        ),
        key="benchmark_matrix_workspace_v2",
    )
    active_workspace = workspace_by_id.get(str(selected_workspace_id)) if selected_workspace_id != "__auto__" else None
    if active_workspace is None:
        st.caption(
            "Auto current master mode: the matrix uses the best available completed source per engine-target cell, "
            "plus queued/running cells where no completed source exists."
        )
    else:
        st.caption(
            f"Saved umbrella snapshot: {active_workspace.get('name')} | "
            f"{active_workspace.get('cell_count') or 0} curated cells | {active_workspace.get('created_at')}. "
            "This view is frozen; new benchmark runs appear in Auto current master until this snapshot is saved again."
        )
    if cells.empty and not overath_targets:
        st.info("No completed benchmark cells with merged metrics are available yet.")
        if unavailable_rows:
            with st.expander("Recent runs not ready for the matrix", expanded=True):
                st.dataframe(pd.DataFrame(unavailable_rows).head(30), hide_index=True, width="stretch")
    else:
        cell_targets = [] if cells.empty else cells["target_id"].dropna().astype(str).unique().tolist()
        engine_options = list(BENCHMARK_MATRIX_ENGINE_ORDER)
        if active_workspace is None:
            target_options = sorted(set(overath_targets) | set(cell_targets))
            visible_source_cells = cells.copy()
            if not visible_source_cells.empty:
                visible_source_cells = visible_source_cells[visible_source_cells["kind"].astype(str) != "collection"].copy()
            canonical_cells = _canonical_benchmark_cells(visible_source_cells)
        else:
            canonical_cells = _benchmark_matrix_workspace_cells(active_workspace, cells)
            workspace_targets: list[str] = []
            workspace_path = Path(str(active_workspace.get("run_dir") or "")) / "artifacts" / "benchmark" / "matrix_workspace.json"
            if workspace_path.exists():
                workspace_payload = read_json(workspace_path)
                workspace_targets = [
                    str(target)
                    for target in (workspace_payload.get("targets") or [])
                    if str(target).strip()
                ]
            canonical_targets = [] if canonical_cells.empty else canonical_cells["target_id"].dropna().astype(str).unique().tolist()
            target_options = sorted(set(overath_targets) | set(workspace_targets) | set(canonical_targets))
        plot_df = _benchmark_matrix_with_missing(
            canonical_cells,
            target_options=target_options,
            engine_options=engine_options,
        )
        if plot_df.empty:
            st.info("No matrix cells are available yet.")
        else:
            if cells.empty:
                plot_df["available_runs"] = 0
            else:
                duplicate_counts = (
                    cells.groupby(["engine", "target_id"], dropna=False)
                    .size()
                    .reset_index(name="available_runs")
                )
                plot_df = (
                    plot_df.drop(columns=["available_runs"], errors="ignore")
                    .merge(duplicate_counts, on=["engine", "target_id"], how="left")
                )
                plot_df["available_runs"] = plot_df["available_runs"].fillna(0).astype(int)
            matrix_height = max(320, min(900, 72 * max(1, plot_df["engine"].nunique())))
            matrix_width = max(520, min(1400, 90 * max(1, plot_df["target_id"].nunique())))
            base_chart = alt.Chart(plot_df)
            heatmap = (
                base_chart
                .mark_rect()
                .encode(
                    x=alt.X("target_id:N", title="target", sort=target_options),
                    y=alt.Y("engine:N", title="engine", sort=BENCHMARK_MATRIX_ENGINE_ORDER),
                    color=alt.Color(
                        "status:N",
                        title="status",
                        scale=alt.Scale(
                            domain=["completed", "running", "queued", "failed", "missing"],
                            range=["#0B74C9", "#F5A3A6", "#FF2E34", "#7DBDEA", "#E5E7EB"],
                        ),
                    ),
                    tooltip=[
                        "engine",
                        "target_id",
                        "status",
                        "batch_progress",
                        "overall_batch_progress",
                        "target_batch_progress",
                        "elapsed_overall",
                        "elapsed_target",
                        "estimated_overall",
                        "estimated_target",
                        "estimated_overall_by_batch",
                        "estimated_target_by_batch",
                        "estimated_overall_by_residue",
                        "estimated_target_by_residue",
                        "remaining_overall",
                        "remaining_target",
                        "estimate_basis",
                        "records",
                        "kind",
                        "settings",
                        "job_code",
                        "run_id",
                        "available_runs",
                        "description",
                    ],
                )
            )
            progress_text = (
                base_chart
                .transform_filter("datum.batch_progress != null && datum.batch_progress != ''")
                .mark_text(color="#172033", fontSize=13, fontWeight="bold")
                .encode(
                    x=alt.X("target_id:N", sort=target_options),
                    y=alt.Y("engine:N", sort=BENCHMARK_MATRIX_ENGINE_ORDER),
                    text="batch_progress:N",
                )
            )
            chart = (heatmap + progress_text).properties(width=matrix_width, height=matrix_height)
            st.altair_chart(chart, width="content")
            running_progress = plot_df[plot_df["status"].astype(str).eq("running")].copy()
            if not running_progress.empty:
                st.markdown("**Live Progress**")
                progress_columns = [
                    "engine",
                    "target_id",
                    "overall_batch_progress",
                    "target_batch_progress",
                    "elapsed_overall",
                    "elapsed_target",
                    "estimated_overall",
                    "estimated_target",
                    "estimated_overall_by_batch",
                    "estimated_overall_by_residue",
                    "remaining_overall",
                    "remaining_target",
                    "estimate_basis",
                ]
                st.dataframe(
                    running_progress[[column for column in progress_columns if column in running_progress.columns]],
                    hide_index=True,
                    width="stretch",
                )

            st.markdown("**Curate The Matrix Umbrella**")
            st.caption(
                "Each engine-target cell must have at most one checked source row. "
                "Canonical rows are what the current matrix uses; uncheck that row when choosing a replacement."
            )
            if cells.empty:
                selectable = pd.DataFrame()
            elif active_workspace is not None:
                selectable = canonical_cells[
                    canonical_cells["status"].astype(str).eq("completed")
                    & canonical_cells["kind"].astype(str).eq("benchmark")
                ].copy()
                st.caption("Loaded umbrella mode: the editor below shows only the curated source row for each matrix cell.")
            else:
                selectable = cells[
                    cells["status"].astype(str).eq("completed")
                    & cells["kind"].astype(str).eq("benchmark")
                ].copy()
            if selectable.empty:
                st.caption("No completed source benchmark runs are available for the matrix yet.")
                selected_cells = selectable.copy()
            else:
                canonical_keys = {
                    f"{row.get('run_id')}::{row.get('engine')}::{row.get('target_id')}"
                    for row in plot_df[plot_df["status"].astype(str).eq("completed")].to_dict(orient="records")
                }
                selectable["canonical"] = selectable.apply(
                    lambda row: f"{row['run_id']}::{row['engine']}::{row['target_id']}" in canonical_keys,
                    axis=1,
                )
                selectable["canonical"] = _clean_checkbox_series(selectable["canonical"])
                selectable = selectable.sort_values(
                    ["engine_rank", "target_id", "canonical", "created_at"],
                    ascending=[True, True, False, False],
                ).copy()
                selected_state_key = f"benchmark_matrix_selected_cells_{selected_workspace_id}"
                selected_state = st.session_state.get(selected_state_key)
                selectable["select"] = selectable.apply(
                    lambda row: bool(
                        selected_state.get(f"{row['run_id']}::{row['engine']}::{row['target_id']}", False)
                        if isinstance(selected_state, dict)
                        else f"{row['run_id']}::{row['engine']}::{row['target_id']}" in canonical_keys
                    ),
                    axis=1,
                )
                selectable["select"] = _clean_checkbox_series(selectable["select"])
                selectable["engine_target"] = selectable["engine"].astype(str) + " | " + selectable["target_id"].astype(str)
            display_columns = [
                "select",
                "canonical",
                "engine_target",
                "settings",
                "result",
                "engine",
                "target_id",
                "records",
                "kind",
                "dataset",
                "job_code",
                "created_at",
                "run_id",
                "description",
            ]
            if not selectable.empty:
                displayed_columns = [col for col in display_columns if col in selectable.columns]
                edited = st.data_editor(
                    selectable[displayed_columns],
                    hide_index=True,
                    width="stretch",
                    key="benchmark_matrix_cell_editor",
                    column_config={
                        "select": st.column_config.CheckboxColumn("select"),
                        "canonical": st.column_config.CheckboxColumn("canonical"),
                        "result": st.column_config.LinkColumn("result", display_text="Open"),
                    },
                    disabled=[col for col in displayed_columns if col != "select"],
                )
                if "select" in edited.columns:
                    edited["select"] = _clean_checkbox_series(edited["select"])
                if "canonical" in edited.columns:
                    edited["canonical"] = _clean_checkbox_series(edited["canonical"])
                selected_cells = edited[edited["select"]].copy() if "select" in edited.columns else edited.iloc[0:0].copy()
                st.session_state[selected_state_key] = {
                    f"{row['run_id']}::{row['engine']}::{row['target_id']}": True
                    for row in selected_cells.to_dict(orient="records")
                }
            if not selected_cells.empty:
                duplicate_selection_rows = pd.DataFrame()
                if {"engine", "target_id"}.issubset(selected_cells.columns):
                    duplicate_mask = selected_cells.duplicated(["engine", "target_id"], keep=False)
                    duplicate_selection_rows = selected_cells[duplicate_mask].copy()
                duplicate_selection = not duplicate_selection_rows.empty
                if duplicate_selection:
                    st.error(
                        "The umbrella has duplicate selected source runs for the same engine-target cell. "
                        "Uncheck all but one row for each duplicate before saving."
                    )
                    duplicate_display = duplicate_selection_rows.sort_values(
                        ["engine", "target_id", "created_at"],
                        ascending=[True, True, False],
                    )
                    st.dataframe(
                        duplicate_display[
                            [
                                col
                                for col in [
                                    "engine_target",
                                    "settings",
                                    "result",
                                    "records",
                                    "job_code",
                                    "created_at",
                                    "run_id",
                                    "description",
                                ]
                                if col in duplicate_display.columns
                            ]
                        ],
                        hide_index=True,
                        width="stretch",
                        column_config={"result": st.column_config.LinkColumn("result", display_text="Open")},
                    )
                selections, selected_targets_for_collection = _matrix_collection_selections(selected_cells)
                st.info(
                    f"Selected {len(selected_cells):,} umbrella cells across "
                    f"{len(selected_targets_for_collection):,} target(s) and {len(selections):,} source run(s)."
                )
                workspace_name = st.text_input(
                    "Umbrella name",
                    value=str(active_workspace.get("name") if active_workspace else "Overath benchmark matrix"),
                    key="benchmark_matrix_workspace_name",
                )
                with st.expander("Selected source mapping", expanded=False):
                    st.json(selections)
                if st.button(
                    "Save matrix umbrella",
                    type="primary",
                    key="benchmark_matrix_save_workspace",
                    disabled=duplicate_selection,
                ):
                    try:
                        run_dir = create_benchmark_matrix_workspace(
                            name=workspace_name,
                            selections=selections,
                        )
                        st.success("Benchmark matrix umbrella saved.")
                        show_pipeline_links(run_dir, [run_dir])
                    except Exception as exc:
                        st.error(str(exc))
                with st.expander("Optional: create a merged benchmark collection from this umbrella", expanded=False):
                    include_input_columns = st.checkbox(
                        "Input metrics",
                        value=False,
                        key="benchmark_matrix_include_input_metrics",
                    )
                    matrix_collection_name = st.text_input(
                        "Collection name",
                        value=f"{workspace_name} collection",
                        key="benchmark_matrix_collection_name",
                    )
                    if st.button(
                        "Create collection from selected umbrella cells",
                        key="benchmark_matrix_create_collection",
                        disabled=duplicate_selection,
                    ):
                        try:
                            run_dir = create_benchmark_collection(
                                name=matrix_collection_name,
                                selections=selections,
                                target_ids=selected_targets_for_collection,
                                include_input_columns=bool(include_input_columns),
                            )
                            st.success("Benchmark collection created from selected umbrella cells.")
                            show_pipeline_links(run_dir, [run_dir])
                        except Exception as exc:
                            st.error(str(exc))
                with st.expander("Maintenance: recalculate missing PyRosetta metrics", expanded=False):
                    st.caption(
                        "Checks the selected umbrella/source rows for missing predicted Rosetta columns, "
                        "then queues a CPU-only backfill job against the original source benchmark runs."
                    )
                    try:
                        missing_pyrosetta = pd.DataFrame(
                            benchmark_missing_pyrosetta_rows(selected_cells.to_dict(orient="records"))
                        )
                    except Exception as exc:
                        missing_pyrosetta = pd.DataFrame()
                        st.error(f"Could not inspect selected rows: {exc}")
                    if missing_pyrosetta.empty:
                        st.caption("No selected source rows could be inspected.")
                    else:
                        missing_display = missing_pyrosetta.copy()
                        if "status" in missing_display.columns:
                            missing_display = missing_display.sort_values(
                                ["status", "engine", "targets", "job_code"],
                                ascending=[False, True, True, True],
                            )
                        st.dataframe(
                            missing_display[
                                [
                                    col
                                    for col in [
                                        "status",
                                        "engine",
                                        "targets",
                                        "records",
                                        "existing_rosetta_records",
                                        "missing_rosetta_records",
                                        "targets_missing",
                                        "job_code",
                                        "run_id",
                                        "metric_column",
                                    ]
                                    if col in missing_display.columns
                                ]
                            ],
                            hide_index=True,
                            width="stretch",
                        )
                        missing_only = missing_pyrosetta[
                            pd.to_numeric(
                                missing_pyrosetta.get("missing_rosetta_records", pd.Series(dtype=int)),
                                errors="coerce",
                            ).fillna(0)
                            > 0
                        ].copy()
                        total_missing = int(
                            pd.to_numeric(
                                missing_only.get("missing_rosetta_records", pd.Series(dtype=int)),
                                errors="coerce",
                            )
                            .fillna(0)
                            .sum()
                        )
                        st.info(
                            f"{len(missing_only):,} selected engine-target cell(s) have "
                            f"{total_missing:,} missing PyRosetta record(s)."
                        )
                        pyrosetta_nprocs_backfill = st.number_input(
                            "PyRosetta CPU workers",
                            min_value=1,
                            max_value=max(1, int(os.cpu_count() or 1)),
                            value=min(32, max(1, int(os.cpu_count() or 1))),
                            step=1,
                            key="benchmark_matrix_pyrosetta_backfill_nprocs",
                        )
                        if st.button(
                            "Recalculate missing PyRosetta metrics for selected source runs",
                            key="benchmark_matrix_pyrosetta_backfill",
                            disabled=duplicate_selection or missing_only.empty,
                        ):
                            try:
                                run_dir = enqueue_missing_pyrosetta_benchmark_metrics(
                                    missing_rows=missing_only.to_dict(orient="records"),
                                    pyrosetta_nprocs=int(pyrosetta_nprocs_backfill),
                                )
                                spawn_worker_for_run(run_dir)
                                st.success("PyRosetta backfill job queued.")
                                show_pipeline_links(run_dir, [run_dir])
                            except Exception as exc:
                                st.error(str(exc))
            else:
                st.caption("Tick one or more exact source rows to save a matrix umbrella.")

        if unavailable_rows:
            with st.expander("Runs not currently represented in the matrix", expanded=False):
                st.dataframe(pd.DataFrame(unavailable_rows).head(40), hide_index=True, width="stretch")

with tabs[6]:
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
        if input_payload.get("job_type") in {"benchmark_collection", "benchmark_matrix_workspace"}:
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
        engines_available = _benchmark_engines_available_for_run(run_dir, input_payload, metrics_path)
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
