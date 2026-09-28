from __future__ import annotations

import ast
import gzip
import hashlib
import json
import re
import shutil
import shlex
import subprocess
import sys
from io import StringIO
from pathlib import Path
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from mn_protein_design.app.components.molstar_viewer import (
    ChainVisualization,
    StructureVisualization,
    molstar_custom_component,
)
from mn_protein_design.app.pages.common import refresh_results_button, result_link, show_contract_files
from mn_protein_design.core.artifacts import build_run_zip
from mn_protein_design.core.benchmark_presets import load_feature_presets, save_feature_preset
from mn_protein_design.core.candidates import STAGE_COMPLEX_REFOLDING, candidate_stage_counts, read_candidates, write_candidates
from mn_protein_design.core.jobs import collect_jobs, create_job, finish_job, get_run_dir, read_json, write_json
from mn_protein_design.core.runtime_estimator import ENGINE_LABELS, format_duration
from mn_protein_design.core.structures import filter_pdb_text


DESIGN_CAMPAIGN_ENGINE_LABELS = {
    "template_redesign": "Template redesign",
    "rfdiffusion_classic": "RFdiffusion classic",
    "bindcraft": "BindCraft",
    "rfdiffusion3_foundry": "RFdiffusion3 / Foundry",
    "boltzgen": "BoltzGen",
    "pxdesign": "PXDesign",
    "genie3": "Genie3",
    "esmfold2_binder_design": "ESMFold2 binder design",
    "protpardelle_1c": "Protpardelle-1c",
    "proteina_complexa": "Proteina-Complexa",
}


FEATURE_ENGINE_PREFIXES = [
    "alphafast_af3_",
    "af3_",
    "colabfold_",
    "colab_",
    "af2_",
    "esmfold2_",
    "boltz1_",
    "boltz2_",
    "rf3_",
    "openfold3_",
    "protenix_v1_",
    "protenix_v2_",
    "protenix_",
    "boltzgen_fold_",
    "input_",
]
CHAIN_INDEXED_CONFIDENCE_FEATURE_RE = re.compile(
    r"(?:^|_)(?:"
    r"chain_\d+_(?:iptm|ptm|pae(?:_min)?|plddt)|"
    r"chain_pair_\d+_\d+_(?:iptm|ptm|pae(?:_min)?)"
    r")$",
    re.IGNORECASE,
)


@st.cache_data(show_spinner=False)
def _read_csv_cached(path_text: str, mtime_ns: int, size: int) -> pd.DataFrame:
    _ = (mtime_ns, size)
    return pd.read_csv(path_text)


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        stat = path.stat()
        return _read_csv_cached(str(path), int(stat.st_mtime_ns), int(stat.st_size))
    except Exception as exc:
        st.warning(f"Could not read {path.name}: {exc}")
        return None


def _display_metric(label: str, value: object) -> None:
    if value is None or value == "":
        st.metric(label, "n/a")
        return
    if isinstance(value, float):
        st.metric(label, f"{value:.3g}")
        return
    st.metric(label, str(value))


def _truthy_benchmark_label(value: object) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "binder", "binds", "positive"}:
        return 1
    if text in {"0", "false", "no", "n", "nonbinder", "non-binder", "negative"}:
        return 0
    try:
        number = float(text)
    except ValueError:
        return None
    if number == 1:
        return 1
    if number == 0:
        return 0
    return None


def _truthy_bool_value(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _unique_chain_list(*chain_groups: list[str]) -> list[str]:
    chains: list[str] = []
    seen: set[str] = set()
    for group in chain_groups:
        for chain in group:
            text = str(chain).strip()
            if not text or text in seen:
                continue
            chains.append(text)
            seen.add(text)
    return chains


def _is_target_refolding_input_builder(value: object) -> bool:
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value or "").strip() == "target_refolding_input_builder"


def _manual_auroc(labels: list[int], scores: list[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _manual_average_precision(labels: list[int], scores: list[float]) -> float | None:
    if not any(label == 1 for label in labels) or not any(label == 0 for label in labels):
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    positives_seen = 0
    precision_sum = 0.0
    total_positives = sum(1 for label in labels if label == 1)
    for rank, (_score, label) in enumerate(ordered, start=1):
        if label == 1:
            positives_seen += 1
            precision_sum += positives_seen / rank
    return precision_sum / total_positives if total_positives else None


def _optimal_threshold_summary(
    values: pd.Series,
    labels: pd.Series,
    *,
    direction: str,
) -> dict[str, Any] | None:
    work = pd.DataFrame(
        {
            "value": pd.to_numeric(values, errors="coerce"),
            "label": labels.map(_truthy_benchmark_label),
        }
    ).dropna(subset=["value", "label"])
    if work.empty:
        return None
    work["label"] = work["label"].astype(int)
    positives = int((work["label"] == 1).sum())
    negatives = int((work["label"] == 0).sum())
    if positives == 0 or negatives == 0:
        return None
    lower_is_better = str(direction or "higher").lower() == "lower"
    rows: list[dict[str, Any]] = []
    for threshold in sorted(work["value"].dropna().unique()):
        selected = work["value"] <= threshold if lower_is_better else work["value"] >= threshold
        selected_count = int(selected.sum())
        if selected_count == 0:
            continue
        tp = int(((work["label"] == 1) & selected).sum())
        fp = selected_count - tp
        precision = tp / selected_count if selected_count else 0.0
        recall = tp / positives if positives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "selected": selected_count,
                "true_binders": tp,
                "false_positives": fp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "success_rate": precision,
                "selected_fraction": selected_count / len(work),
                "condition": "<=" if lower_is_better else ">=",
            }
        )
    if not rows:
        return None
    return max(rows, key=lambda row: (row["f1"], row["precision"], row["recall"], -row["selected"]))


def _threshold_precision_recall_curve(
    values: pd.Series,
    labels: pd.Series,
    *,
    direction: str,
) -> pd.DataFrame:
    work = pd.DataFrame(
        {
            "threshold": pd.to_numeric(values, errors="coerce"),
            "label": labels.map(_truthy_benchmark_label),
        }
    ).dropna(subset=["threshold", "label"])
    if work.empty:
        return pd.DataFrame()
    work["label"] = work["label"].astype(int)
    positives = int((work["label"] == 1).sum())
    if positives == 0:
        return pd.DataFrame()
    lower_is_better = str(direction or "higher").lower() == "lower"
    rows: list[dict[str, Any]] = []
    for threshold in sorted(work["threshold"].dropna().unique()):
        selected = work["threshold"] <= threshold if lower_is_better else work["threshold"] >= threshold
        selected_count = int(selected.sum())
        if selected_count == 0:
            continue
        tp = int(((work["label"] == 1) & selected).sum())
        precision = tp / selected_count if selected_count else 0.0
        recall = tp / positives if positives else 0.0
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision,
                "recall": recall,
                "selected": selected_count,
                "condition": "<=" if lower_is_better else ">=",
            }
        )
    return pd.DataFrame(rows)


def _row_number(row: pd.Series, column: str) -> float | None:
    if column not in row.index:
        return None
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    return float(value) if pd.notna(value) else None


def _effective_feature_direction(ranking_row: pd.Series | None, fallback: str = "higher") -> str:
    if ranking_row is None or ranking_row.empty:
        return fallback
    direction = str(ranking_row.get("direction") or fallback or "higher").lower()
    ap = _row_number(ranking_row, "average_precision")
    inverse_ap = _row_number(ranking_row, "inverse_average_precision")
    if ap is not None and inverse_ap is not None and abs(inverse_ap - ap) > 1e-12:
        return "lower" if inverse_ap > ap else "higher"
    auroc = _row_number(ranking_row, "auroc")
    inverse_auroc = _row_number(ranking_row, "inverse_auroc")
    if auroc is not None and inverse_auroc is not None and abs(inverse_auroc - auroc) > 1e-12:
        return "lower" if inverse_auroc > auroc else "higher"
    best_auroc_direction = str(ranking_row.get("best_auroc_direction") or "").lower()
    if best_auroc_direction in {"higher", "lower"}:
        return best_auroc_direction
    return "lower" if direction == "lower" else "higher"


def _label_column(df: pd.DataFrame) -> str | None:
    for column in ["binder", "label", "is_binder", "binds"]:
        if column in df.columns:
            return column
    return None


def _show_csv_table(path: Path, columns: list[str] | None = None, rows: int = 200) -> bool:
    df = _read_csv(path)
    if df is None:
        return False
    if df.empty:
        st.info(f"{path.name} is empty.")
        return True
    shown = df
    if columns:
        keep = [col for col in columns if col in df.columns]
        if keep:
            shown = df[keep]
    st.dataframe(shown.head(rows), hide_index=True, width="stretch")
    return True


def _show_csv_preview(path: Path, title: str, columns: list[str] | None = None, rows: int = 200) -> None:
    if not path.exists():
        return
    st.subheader(title)
    _show_csv_table(path, columns=columns, rows=rows)


def _show_metric_expander(
    path: Path,
    title: str,
    description: str,
    columns: list[str] | None = None,
    *,
    expanded: bool = False,
    rows: int = 200,
) -> None:
    if not path.exists():
        return
    with st.expander(title, expanded=expanded):
        st.caption(description)
        _show_csv_table(path, columns=columns, rows=rows)


def _show_combined_interface_metrics(benchmark_dir: Path) -> None:
    source_paths = [
        benchmark_dir / "common_interface_metrics.csv",
        benchmark_dir / "af2_common_interface_metrics.csv",
        benchmark_dir / "esmfold2_common_interface_metrics.csv",
        benchmark_dir / "rf3_common_interface_metrics.csv",
        benchmark_dir / "protenix_common_interface_metrics.csv",
        benchmark_dir / "boltzgen_fold_common_interface_metrics.csv",
    ]
    tables = [table for path in source_paths if (table := _read_csv(path)) is not None and not table.empty]
    if not tables:
        return
    combined = tables[0]
    for table in tables[1:]:
        new_columns = ["binder_id", *[col for col in table.columns if col != "binder_id" and col not in combined.columns]]
        combined = combined.merge(table[new_columns], on="binder_id", how="outer")
    preferred = ["binder_id"]
    for engine in ["af3", "af2", "boltz2", "colab", "esmfold2", "rf3", "protenix", "boltzgen_fold"]:
        preferred.extend(
            [
                f"{engine}_ipSAE_min",
                f"{engine}_LIS",
                f"{engine}_pDockQ_min",
                f"{engine}_pDockQ2_min",
                f"{engine}_ipae",
            ]
        )
    keep = [col for col in preferred if col in combined.columns]
    st.dataframe(combined[keep] if keep else combined, hide_index=True, width="stretch")


def _show_rosetta_metrics_by_engine(path: Path) -> None:
    df = _read_csv(path)
    if df is None or df.empty:
        return
    engine_labels = [
        ("af3", "AlphaFast AF3"),
        ("af2", "AF2 initial guess"),
        ("boltz2", "Boltz-2"),
        ("colab", "ColabFold"),
        ("esmfold2", "ESMFold2"),
        ("rf3", "RF3"),
        ("protenix", "Protenix v0.5"),
        ("protenix_v1", "Protenix v1"),
        ("protenix_v2", "Protenix v2"),
        ("boltzgen_fold", "BoltzGen Fold"),
    ]
    shown = False
    for engine, label in engine_labels:
        prefix = f"{engine}_rosetta_"
        metric_columns = [col for col in df.columns if col.startswith(prefix)]
        if not metric_columns:
            continue
        shown = True
        st.markdown(f"**{label}**")
        engine_df = df[["binder_id", *metric_columns]].copy()
        engine_df = engine_df.rename(columns={col: col[len(prefix) :] for col in metric_columns})
        st.dataframe(engine_df, hide_index=True, width="stretch")
    if not shown:
        st.info("No per-engine Rosetta metrics are available.")


def _show_runtime_timings(metrics: dict) -> None:
    timings = metrics.get("runtime_engine_timings")
    if not isinstance(timings, dict) or not timings:
        return
    rows = []
    for engine, timing in timings.items():
        if not isinstance(timing, dict):
            continue
        seconds = pd.to_numeric(pd.Series([timing.get("seconds")]), errors="coerce").iloc[0]
        if pd.isna(seconds):
            continue
        candidate_count = timing.get("candidate_count")
        seconds_per_candidate = None
        try:
            if candidate_count:
                seconds_per_candidate = float(seconds) / float(candidate_count)
        except (TypeError, ValueError, ZeroDivisionError):
            seconds_per_candidate = None
        rows.append(
            {
                "engine": ENGINE_LABELS.get(str(engine), str(engine)),
                "time": format_duration(float(seconds)),
                "seconds": float(seconds),
                "candidates": candidate_count,
                "residues": timing.get("total_residues"),
                "sec_per_candidate": seconds_per_candidate,
            }
        )
    if not rows:
        return
    rows.sort(key=lambda row: row["seconds"], reverse=True)
    st.subheader("Runtime By Engine")
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "seconds": st.column_config.NumberColumn("seconds", format="%.1f"),
            "sec_per_candidate": st.column_config.NumberColumn("sec / candidate", format="%.1f"),
        },
    )


def _benchmark_result_engine_labels(benchmark_dir: Path, result: dict) -> list[str]:
    labels: list[str] = []
    outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
    engine_artifacts = outputs.get("engine_artifacts") if isinstance(outputs.get("engine_artifacts"), dict) else {}
    for engine_key in engine_artifacts:
        label = next(
            (
                engine_label
                for engine_label, table_engine_key, _csv_name in STRUCTURE_ENGINE_TABLES
                if table_engine_key == str(engine_key)
            ),
            str(engine_key).replace("_", " "),
        )
        if label not in labels:
            labels.append(label)
    for engine_label, _engine_key, csv_name in STRUCTURE_ENGINE_TABLES:
        if engine_label in labels:
            continue
        table = _read_csv(benchmark_dir / csv_name)
        if table is not None and not table.empty:
            labels.append(engine_label)
    return labels


def _benchmark_result_source_table(run_dir: Path, benchmark_dir: Path, result: dict) -> pd.DataFrame | None:
    sources = _read_csv(benchmark_dir / "benchmark_collection_sources.csv")
    if sources is not None and not sources.empty:
        return sources.copy()

    input_json = read_json(run_dir / "input.json")
    if str(input_json.get("job_type") or "") == "refolding_evaluation":
        return None
    queued = _read_csv(run_dir / "artifacts" / "queued_inputs" / "input.csv")
    merged = _read_csv(benchmark_dir / "merged_benchmark_metrics.csv")
    raw_run = _read_csv(run_dir / "artifacts" / "raw" / "de_novo_binder_scoring" / "output" / "run.csv")
    record_count = 0
    selected_ids = input_json.get("inputs", {}).get("selected_candidate_ids")
    if isinstance(selected_ids, list) and selected_ids:
        record_count = len(selected_ids)
    for table in [queued, merged, raw_run]:
        if record_count:
            break
        if table is not None and not table.empty:
            record_count = int(len(table))
            break
    target_count = None
    if merged is not None and not merged.empty and "target_id" in merged.columns:
        target_count = int(merged["target_id"].dropna().astype(str).nunique())
    elif queued is not None and not queued.empty and "target_id" in queued.columns:
        target_count = int(queued["target_id"].dropna().astype(str).nunique())
    metadata = read_json(run_dir / "metadata.json")
    job_code = str(metadata.get("job_code") or "").strip()
    engine_labels = _benchmark_result_engine_labels(benchmark_dir, result)
    return pd.DataFrame(
        [
            {
                "order": 1,
                "run_id": run_dir.name,
                "job_code": job_code,
                "engines": ", ".join(engine_labels),
                "engine_variants": "",
                "records": record_count or None,
                "coverage_records": record_count or None,
                "targets": target_count,
                "is_collection_backbone": True,
                "source_kind": "single benchmark run",
            }
        ]
    )


def _show_benchmark_collection_notice(run_dir: Path, benchmark_dir: Path, result: dict) -> None:
    collection = read_json(benchmark_dir / "collection.json")
    sources = _benchmark_result_source_table(run_dir, benchmark_dir, result)
    if not collection and (sources is None or sources.empty):
        return
    provenance = _read_csv(benchmark_dir / "benchmark_collection_provenance.csv")
    job_code_by_run: dict[str, str] = {}
    if provenance is not None and not provenance.empty:
        for _, row in provenance.iterrows():
            run_id = str(row.get("source_run_id") or "").strip()
            job_code = str(row.get("source_job_code") or "").strip()
            if run_id and job_code and job_code.lower() != "nan":
                job_code_by_run.setdefault(run_id, job_code)
    source_lines: list[str] = []
    if sources is not None and not sources.empty:
        for _, row in sources.sort_values("order").iterrows():
            run_id = str(row.get("run_id") or "").strip()
            if not run_id or run_id.lower() == "nan":
                continue
            row_job_code = str(row.get("job_code") or "").strip()
            job_code = row_job_code if row_job_code and row_job_code.lower() != "nan" else job_code_by_run.get(run_id, run_id[:10])
            engines = str(row.get("engines") or "").strip()
            engine_variants = str(row.get("engine_variants") or "").strip()
            records = row.get("records")
            coverage = row.get("coverage_records")
            backbone = str(row.get("is_collection_backbone") or "").lower() == "true"
            bits = [job_code]
            if engines and engines.lower() != "nan":
                bits.append(engines)
            if engine_variants and engine_variants.lower() != "nan":
                bits.append(engine_variants)
            try:
                if pd.notna(records):
                    bits.append(f"{int(records)} source rows")
                if pd.notna(coverage) and pd.notna(records) and int(coverage) != int(records):
                    bits.append(f"{int(coverage)} rows used for selected features")
            except Exception:
                pass
            if backbone:
                bits.append("backbone")
            bits.append(run_id)
            source_lines.append(" | ".join(bits))
    if collection:
        name = collection.get("name") or "Benchmark collection"
        st.info(
            f"This result is a benchmark collection: **{name}**. "
            "It combines selected engine outputs from existing benchmark jobs. "
            "Deleting this collection removes only this collection run; the source benchmark jobs and their prediction artifacts are kept."
        )
    if source_lines:
        label = "Collection source jobs" if collection else "Benchmark source job"
        with st.expander(label, expanded=bool(collection)):
            st.write("\n".join(f"- {line}" for line in source_lines))


CLASS_COLOR_SCALE = alt.Scale(domain=["binder", "nonbinder", "candidate"], range=["#D55E00", "#0072B2", "#6B7280"])
BENCHMARK_CLASS_COLOR_SCALE = alt.Scale(domain=["binder", "nonbinder"], range=["#D55E00", "#0072B2"])
LINE_COLOR_SCALE = alt.Scale(domain=["precision", "recall"], range=["#009E73", "#CC79A7"])
ENGINE_COLOR_DOMAIN = [
    "AF3",
    "AF2-IG",
    "Boltz-2",
    "BoltzGen Fold",
    "ColabFold",
    "ESMFold2",
    "ESMFold2 Fast",
    "ESMFold2 Standard",
    "ESMFold2 Careful",
    "ESMFold2 High diffusion",
    "ESMFold2 Design rank",
    "ESMFold2 Custom 3/32",
    "ESMFold2 Custom 3/50",
    "ESMFold2 Custom 10/68",
    "ESMFold2 Custom 20/68",
    "ESMFold2 Custom 10/200",
    "ESMFold2 Custom 3/200",
    "OpenFold-3",
    "Protenix v0.5",
    "Protenix v1",
    "Protenix v2",
    "RF3",
    "Input PyRosetta",
    "Input",
    "Published metrics",
    "Other",
    "Manual combination",
]
ENGINE_LABEL_ALIASES = {
    "Protenix": "Protenix v0.5",
}
ENGINE_COLOR_RANGE = [
    "#56B4E9",
    "#0072B2",
    "#E41A1C",
    "#F4A3A3",
    "#009E73",
    "#7AD99F",
    "#22C55E",
    "#16A34A",
    "#86EFAC",
    "#15803D",
    "#4ADE80",
    "#2F9E44",
    "#22C55E",
    "#16A34A",
    "#86EFAC",
    "#15803D",
    "#4ADE80",
    "#8B5CF6",
    "#D55E00",
    "#CC79A7",
    "#A6761D",
    "#111827",
    "#8B5CF6",
    "#9CA3AF",
    "#111827",
]
ENGINE_COLOR_SCALE = alt.Scale(domain=ENGINE_COLOR_DOMAIN, range=ENGINE_COLOR_RANGE)
ENGINE_COLOR_BY_NAME = dict(zip(ENGINE_COLOR_DOMAIN, ENGINE_COLOR_RANGE))
PAE_ROSETTA_COMBO_CACHE_VERSION = 9
MANUAL_FEATURE_COMBO_CATEGORY = "Manual combinations"
MANUAL_FEATURE_COMBO_SOURCE_CATEGORIES = {
    "Engine Confidence / PAE",
    "Recalculated PAE / Interface",
    "Interface Energy / Rosetta",
    "Interface Geometry / PyMOL",
}
ESMFOLD2_VARIANT_LABELS = {
    "fast": "Fast",
    "standard": "Standard",
    "careful": "Careful",
    "high_diffusion": "High diffusion",
    "design_rank": "Design rank",
}


def _canonical_engine_label(engine: object) -> str:
    text = str(engine or "").strip()
    return ENGINE_LABEL_ALIASES.get(text, text)


def _display_feature_name(feature: object) -> str:
    text = str(feature or "")
    if text.startswith("boltz1_"):
        return f"boltz2_{text.removeprefix('boltz1_')}"
    return text


def _engine_labels_by_specificity() -> list[str]:
    labels = list(ENGINE_COLOR_DOMAIN) + [label for label in ENGINE_LABEL_ALIASES if label not in ENGINE_COLOR_DOMAIN]
    return sorted(labels, key=len, reverse=True)


def _engine_color_scale(engines: list[object] | pd.Series | pd.Index) -> alt.Scale:
    present = {str(engine) for engine in engines if str(engine)}
    domain = [engine for engine in ENGINE_COLOR_DOMAIN if engine in present]
    domain.extend(sorted(engine for engine in present if engine not in set(domain)))
    fallback = "#7a869a"
    color_range = [_engine_color(engine, fallback=fallback) for engine in domain]
    return alt.Scale(domain=domain, range=color_range)


def _engine_color(engine: object, *, fallback: str = "#7a869a") -> str:
    name = str(engine or "")
    if name in ENGINE_COLOR_BY_NAME:
        return ENGINE_COLOR_BY_NAME[name]
    if name.startswith("ESMFold2 Custom"):
        return "#2F9E44"
    if name.startswith("ESMFold2"):
        return "#16A34A"
    return fallback


def _esmfold2_variant_engine_label(token: object) -> str:
    text = str(token or "").strip()
    if not text:
        return "ESMFold2"
    if text in ESMFOLD2_VARIANT_LABELS:
        return f"ESMFold2 {ESMFOLD2_VARIANT_LABELS[text]}"
    custom_match = re.match(r"custom_(\d+)_(\d+)$", text)
    if custom_match:
        return f"ESMFold2 Custom {custom_match.group(1)}/{custom_match.group(2)}"
    return f"ESMFold2 {text.replace('_', ' ').title()}"


def _feature_engine(feature: object, fallback: str = "Other") -> str:
    text = str(feature or "")
    if text.startswith("input_rosetta_"):
        return "Input PyRosetta"
    if " × " in text:
        for engine in _engine_labels_by_specificity():
            if text.startswith(f"{engine} "):
                return _canonical_engine_label(engine)
    if text.startswith("esmfold2_"):
        suffix = text.removeprefix("esmfold2_")
        for token in ESMFOLD2_VARIANT_LABELS:
            if suffix.startswith(f"{token}_"):
                return _esmfold2_variant_engine_label(token)
        custom_match = re.match(r"custom_(\d+)_(\d+)_", suffix)
        if custom_match:
            return _esmfold2_variant_engine_label(f"custom_{custom_match.group(1)}_{custom_match.group(2)}")
    known_prefixes = [
        ("alphafast_af3", "AF3"),
        ("af3_", "AF3"),
        ("colabfold", "ColabFold"),
        ("colab_", "ColabFold"),
        ("af2_", "AF2-IG"),
        ("esmfold2", "ESMFold2"),
        ("boltz1", "Boltz-2"),
        ("boltz2", "Boltz-2"),
        ("rf3_", "RF3"),
        ("openfold3_", "OpenFold-3"),
        ("protenix_v1_", "Protenix v1"),
        ("protenix_v2_", "Protenix v2"),
        ("protenix_", "Protenix v0.5"),
        ("boltzgen_fold_", "BoltzGen Fold"),
        ("input_", "Input"),
    ]
    for prefix, label in known_prefixes:
        if text.startswith(prefix):
            return label
    raw = text.lower()
    esmfold2_unprefixed = {
        "ipae_binder_to_target",
        "min_interaction_pae",
        "interaction_pae",
        "pair_chains_iptm",
        "ipae_target_to_binder",
        "pae",
        "confidence",
        "plddt_mean",
        "binder_interface_contacts",
        "ipae_contact_pairs",
    }
    if raw in esmfold2_unprefixed:
        return "ESMFold2"
    if "_rosetta_" in text:
        return str(text.split("_rosetta_", 1)[0]).upper()
    if "_pymol_" in text:
        return str(text.split("_pymol_", 1)[0]).upper()
    return fallback


def _ranking_engine_fallback(path: Path) -> str:
    name = path.name
    if name.startswith("esmfold2_"):
        return "ESMFold2"
    if name.startswith("alphafast_af3_") or name.startswith("common_interface_"):
        return "AF3/Shared"
    if name.startswith("colabfold_"):
        return "ColabFold"
    if name.startswith("af2_"):
        return "AF2-IG"
    if name.startswith("boltz2_"):
        return "Boltz-2"
    if name.startswith("openfold3_"):
        return "OpenFold-3"
    if name.startswith("metric_"):
        return "Published metrics"
    return "Other"


def _feature_alias_group(feature: object, fallback: str = "Other") -> str:
    text = str(feature or "")
    engine = _feature_engine(text, fallback=fallback)
    raw = text
    for prefix in FEATURE_ENGINE_PREFIXES:
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    for marker in ["_rosetta_", "_pymol_"]:
        if marker in raw:
            raw = raw.split(marker, 1)[1]
            break
    raw = raw.lower()
    groups = {
        "ipae": "contact_interface_pae_mean",
        "interaction_pae": "contact_interface_pae_mean",
        "contact_ipae_mean": "contact_interface_pae_mean",
        "contact_interface_pae_mean": "contact_interface_pae_mean",
        "min_interaction_pae": "contact_interface_pae_min",
        "contact_ipae_min": "contact_interface_pae_min",
        "contact_interface_pae_min": "contact_interface_pae_min",
        "ipae_contact_pairs": "contact_interface_pair_count",
        "contact_ipae_pairs": "contact_interface_pair_count",
        "contact_interface_pair_count": "contact_interface_pair_count",
        "ipae_mean": "all_interface_pae_mean",
        "ipae_min": "all_interface_pae_min",
        "ipae_binder_to_target": "directional_pae_binder_to_target",
        "ipae_target_to_binder": "directional_pae_target_to_binder",
        "ipsae_min": "ipsae_min",
        "ipsae_max": "ipsae_max",
        "ipsae_avg": "ipsae_avg",
        "ipsae_min_in_calculation": "ipsae_min_in_calculation",
        "ipsae_d0chn": "ipsae_d0chn",
        "ipsae_d0dom": "ipsae_d0dom",
        "lis": "lis",
        "pdockq_min": "pdockq_min",
        "pdockq_max": "pdockq_max",
        "pdockq2_min": "pdockq2_min",
        "pdockq2_max": "pdockq2_max",
        "interface_packstat": "rosetta_interface_packstat",
        "interface_sc": "rosetta_interface_sc",
        "interface_dg": "rosetta_interface_dg",
        "interface_dg_dsasa_ratio": "rosetta_interface_dg_dsasa_ratio",
    }
    return f"{engine}:{groups.get(raw, raw)}"


def _feature_alias_priority(feature: object) -> int:
    text = str(feature or "")
    raw = text
    for prefix in FEATURE_ENGINE_PREFIXES:
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    raw = raw.lower()
    priorities = {
        "esmfold2_contact_ipae_mean": 0,
        "esmfold2_contact_ipae_min": 0,
        "esmfold2_contact_ipae_pairs": 0,
        "contact_ipae_mean": 0,
        "contact_ipae_min": 0,
        "contact_ipae_pairs": 0,
        "contact_interface_pae_mean": 0,
        "contact_interface_pae_min": 0,
        "interaction_pae": 1,
        "min_interaction_pae": 1,
        "ipae": 2,
        "ipae_contact_pairs": 2,
    }
    return priorities.get(text, priorities.get(raw, 10))


def _raw_feature_name(feature: object) -> str:
    raw = str(feature or "")
    for prefix in FEATURE_ENGINE_PREFIXES:
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break
    for marker in ["_rosetta_", "_pymol_"]:
        if marker in raw:
            raw = raw.split(marker, 1)[1]
            break
    return raw.lower()


def _is_chain_indexed_confidence_feature(feature: object) -> bool:
    return bool(CHAIN_INDEXED_CONFIDENCE_FEATURE_RE.search(str(feature or "")))


def _is_amino_acid_composition_feature(feature: object) -> bool:
    text = str(feature or "").lower()
    raw = _raw_feature_name(feature)
    amino_acid_tokens = [
        "interface_aa_",
        "rosetta_interface_aa_",
        "aa_ala",
        "aa_arg",
        "aa_asn",
        "aa_asp",
        "aa_cys",
        "aa_gln",
        "aa_glu",
        "aa_gly",
        "aa_his",
        "aa_ile",
        "aa_leu",
        "aa_lys",
        "aa_met",
        "aa_phe",
        "aa_pro",
        "aa_ser",
        "aa_thr",
        "aa_trp",
        "aa_tyr",
        "aa_val",
    ]
    sequence_composition_tokens = [
        "alanine_fraction",
        "glycine_fraction",
        "proline_fraction",
        "cysteine_fraction",
        "charged_fraction",
        "hydrophobic_fraction",
        "net_charge_per_residue",
        "aa_entropy",
        "low_complexity_fraction",
        "max_same_aa_run",
        "aaa_runs",
        "gly_runs",
        "pro_runs",
    ]
    return any(token in text or token in raw for token in amino_acid_tokens + sequence_composition_tokens)


def _is_size_or_count_proxy_feature(feature: object) -> bool:
    text = str(feature or "").lower()
    raw = _raw_feature_name(feature)
    size_or_count_tokens = [
        "interface_nres",
        "nres_binder",
        "nres_target",
        "interface_residue",
        "interface_residues",
        "interface_contacts",
        "binder_interface_contacts",
        "contact_interface_pair_count",
        "ipae_contact_pairs",
        "contact_ipae_pairs",
        "interface_hbond",
        "interface_hbonds",
        "unsat_hbond",
        "unsat_hbonds",
        "hbond_count",
        "hbonds_count",
    ]
    return any(token in text or token in raw for token in size_or_count_tokens)


def _is_allowed_pae_rosetta_partner_feature(feature: object) -> bool:
    text = str(feature or "").lower()
    raw = _raw_feature_name(feature)
    if _is_amino_acid_composition_feature(feature) or _is_size_or_count_proxy_feature(feature):
        return False
    if any(token in text or token in raw for token in ["sap_delta", "delta_sap"]):
        return True
    if any(token in text or token in raw for token in ["pymol", "packstat", "sap"]):
        return False
    allowed_tokens = [
        "interface_dg_dsasa_ratio",
        "interface_dg_sasa_ratio",
        "interface_dg",
        "interface_sc",
        "rosetta_interface_dg_dsasa_ratio",
        "rosetta_interface_dg_sasa_ratio",
        "rosetta_interface_dg",
        "rosetta_interface_sc",
        "interface_δg/δsasa",
        "interface_δg",
        "rosetta_interface_δg/δsasa",
        "rosetta_interface_δg",
    ]
    return any(token in text or token in raw for token in allowed_tokens)


def _is_unsupported_auto_pae_rosetta_combo_feature(feature: object) -> bool:
    text = str(feature or "")
    if " × input " in text.lower() or " × " not in text:
        return False
    right = text.split(" × ", 1)[1]
    return not _is_allowed_pae_rosetta_partner_feature(right)


def _drop_chain_indexed_confidence_features(ranking: pd.DataFrame) -> pd.DataFrame:
    if ranking is None or ranking.empty or "feature" not in ranking.columns:
        return ranking
    hidden_features = ranking["feature"].map(
        lambda feature: _is_chain_indexed_confidence_feature(feature)
        or _is_amino_acid_composition_feature(feature)
        or _is_size_or_count_proxy_feature(feature)
        or _is_unsupported_auto_pae_rosetta_combo_feature(feature)
    )
    return ranking[~hidden_features].copy()


def _is_informative_feature(feature: object) -> bool:
    if _is_chain_indexed_confidence_feature(feature):
        return False
    if _is_amino_acid_composition_feature(feature):
        return False
    if _is_size_or_count_proxy_feature(feature):
        return False
    raw = _raw_feature_name(feature)
    if raw.startswith("has_"):
        return False
    if raw.endswith(("_ready", "_used", "_cutoff", "_count", "_counts", "_pairs", "_rank", "_score")):
        return False
    if raw in {
        "binder_ca_count",
        "binder_length",
        "hotspot_count",
        "hotspots_contacted",
        "initial_guess_note",
        "initial_guess_used",
        "ipsae_ready",
        "benchmark_rank",
        "benchmark_score",
    }:
        return False
    return True


def _is_overath_key_feature(feature: object) -> bool:
    text = str(feature or "").lower()
    if text.startswith("input_rosetta_"):
        return False
    raw = _raw_feature_name(feature)
    if not _is_informative_feature(feature):
        return False
    key_tokens = [
        "ipsae_min",
        "lis",
        "ipae",
        "pae_interaction",
        "interaction_pae",
        "iptm",
        "actifptm",
        "interface_dg_sasa_ratio",
        "dg_dsasa",
        "interface_sc",
        "shape_complement",
        "sap_delta",
        "delta_sap",
        "dockq",
        "rmsd_binder",
        "binder_rmsd",
        "target_aligned_binder_rmsd",
    ]
    if any(token in raw for token in key_tokens) or any(token in text for token in key_tokens):
        return True
    if "_x_" in text:
        has_confidence = any(token in text for token in ["ipsae_min", "lis", "ipae", "iptm", "dockq"])
        has_structure = any(token in text for token in ["dg_sasa", "dg_dsasa", "shape_complement", "sap_delta", "delta_sap"])
        return has_confidence and has_structure
    return False


def _is_recalculated_pae_interface_feature(feature: object) -> bool:
    text = str(feature or "").lower()
    if text.startswith("input_rosetta_"):
        return False
    raw = _raw_feature_name(feature)
    # Keep native engine/parser PAE summaries in "Engine Confidence / PAE".
    # This category is reserved for the standardized ipSAE/interface
    # post-processing layer, not for every metric whose name contains "PAE".
    recalculated_tokens = [
        "ipsae",
        "lis",
        "pdockq",
        "pdockq2",
    ]
    return any(token in raw for token in recalculated_tokens) or any(token in text for token in recalculated_tokens)


def _is_engine_confidence_pae_feature(feature: object) -> bool:
    if str(feature or "").lower().startswith("input_rosetta_"):
        return False
    raw = _raw_feature_name(feature)
    if _is_recalculated_pae_interface_feature(feature):
        return False
    native_tokens = [
        "actifptm",
        "confidence",
        "ipde",
        "iplddt",
        "iptm",
        "pae",
        "plddt",
        "ptm",
        "ranking_confidence",
        "ranking_score",
    ]
    return any(token in raw for token in native_tokens)


def _feature_category(feature: object) -> str:
    text = str(feature or "").lower()
    raw = _raw_feature_name(feature)
    if " × input " in text or " × input rosetta " in text:
        return "PAE × Input Rosetta combinations"
    if " × " in str(feature or ""):
        return "PAE × Rosetta combinations"
    if text.startswith("input_rosetta_"):
        return "Input PyRosetta"
    if _is_engine_confidence_pae_feature(feature):
        return "Engine Confidence / PAE"
    if _is_recalculated_pae_interface_feature(feature):
        return "Recalculated PAE / Interface"
    if "rosetta_" in text or any(
        token in raw
        for token in [
            "interface_dg",
            "interface_dsasa",
            "dg_dsasa",
            "interface_sc",
            "packstat",
            "sap",
            "hbond",
            "unsat",
            "binder_hydrophobicity",
        ]
    ):
        return "Interface Energy / Rosetta"
    if "pymol_" in text or any(
        token in raw
        for token in [
            "shape_complement",
            "sasa",
            "interface_residue",
            "interface_residues",
            "nonpolar",
            "paratope",
            "epitope",
            "helix",
            "sheet",
            "loop",
        ]
    ):
        return "Interface Geometry / PyMOL"
    if not _is_informative_feature(feature):
        return "Setup / Status"
    return "Other"


def _combo_feature_term(feature: object) -> str:
    text = str(feature or "")
    raw = _raw_feature_name(text)
    replacements = {
        "ipsae": "ipSAE",
        "ipae": "iPAE",
        "pdockq": "pDockQ",
        "pdockq2": "pDockQ2",
        "dg_dsasa": "ΔG/ΔSASA",
        "interface_dg_dsasa_ratio": "interface_ΔG/ΔSASA",
        "interface_dg_sasa_ratio": "interface_ΔG/ΔSASA",
        "interface_dg": "interface_ΔG",
        "interface_dsasa": "interface_ΔSASA",
        "interface_sc": "interface_SC",
        "sap_delta": "SAP_delta",
        "delta_sap": "delta_SAP",
        "packstat": "packstat",
    }
    for needle, replacement in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        raw = raw.replace(needle, replacement)
    return raw


def _metric_search_blob(feature: object) -> str:
    """Search text for metric selectors, including display and punctuation-free forms."""
    text = str(feature or "")
    display = _combo_feature_term(text)
    pieces = [
        text,
        display,
        text.replace("×", "x"),
        display.replace("×", "x"),
    ]
    expanded = " ".join(piece.lower() for piece in pieces)
    normalized = " ".join(re.sub(r"[^a-z0-9]+", " ", piece.lower()) for piece in pieces)
    compact = " ".join(re.sub(r"[^a-z0-9]+", "", piece.lower()) for piece in pieces)
    return f"{expanded} {normalized} {compact}"


def _metric_query_matches(feature: object, terms: list[str]) -> bool:
    blob = _metric_search_blob(feature)
    text = str(feature or "").lower()
    display = _combo_feature_term(feature).lower()
    for term in terms:
        raw = term.lower()
        normalized = re.sub(r"[^a-z0-9]+", " ", raw).strip()
        compact = re.sub(r"[^a-z0-9]+", "", raw)
        if "_" in raw:
            candidates = [raw]
            if raw.startswith("interface_"):
                candidates.append(f"rosetta_{raw}")
            if raw.startswith("dg_"):
                candidates.append(f"interface_{raw}")
                candidates.append(f"rosetta_interface_{raw}")
            if raw.startswith("delta_"):
                candidates.append(f"sap_{raw.removeprefix('delta_')}")
            patterns = [
                re.compile(rf"(?<![a-z0-9_]){re.escape(candidate)}(?![a-z0-9_])")
                for candidate in dict.fromkeys(candidates)
            ]
            if any(pattern.search(text) or pattern.search(display) for pattern in patterns):
                continue
            return False
        if raw in blob or (normalized and normalized in blob) or (compact and compact in blob):
            continue
        return False
    return True


def _metric_family_key(feature: object) -> str:
    text = str(feature or "")
    if " × " in text:
        for engine in _engine_labels_by_specificity():
            prefix = f"{engine} "
            if text.startswith(prefix):
                return text.removeprefix(prefix)
        return text
    return _raw_feature_name(text)


def _combo_feature_name(engine: str, left_feature: object, right_feature: object) -> str:
    return f"{engine} {_combo_feature_term(left_feature)} × {_combo_feature_term(right_feature)}"


def _input_combo_feature_name(engine: str, left_feature: object, right_feature: object) -> str:
    return f"{engine} {_combo_feature_term(left_feature)} × input {_combo_feature_term(right_feature)}"


def _combo_product_direction(left_direction: object, right_direction: object) -> str:
    left = str(left_direction or "higher").lower()
    right = str(right_direction or "higher").lower()
    if left == "higher" and right == "higher":
        return "higher"
    return "lower"


def _clear_pae_rosetta_combo_caches() -> None:
    for cached_func in [
        _cached_pae_rosetta_combo_rankings,
        _cached_pae_rosetta_combo_metric_columns,
        _cached_pae_input_rosetta_combo_rankings,
        _cached_pae_input_rosetta_combo_metric_columns,
    ]:
        clear_func = getattr(cached_func, "clear", None)
        if callable(clear_func):
            clear_func()


def _score_feature_vector(
    labels_raw: list[int | None],
    values: pd.Series,
) -> dict[str, Any] | None:
    valid_labels: list[int] = []
    valid_scores: list[float] = []
    for label, score in zip(labels_raw, values.tolist()):
        if label in {0, 1} and pd.notna(score):
            valid_labels.append(int(label))
            valid_scores.append(float(score))
    if not valid_scores:
        return None
    positives = [score for label, score in zip(valid_labels, valid_scores) if label == 1]
    negatives = [score for label, score in zip(valid_labels, valid_scores) if label == 0]
    if not positives or not negatives:
        return None
    auroc = _manual_auroc(valid_labels, valid_scores)
    ap = _manual_average_precision(valid_labels, valid_scores)
    inverse_scores = [-score for score in valid_scores]
    inverse_auroc = _manual_auroc(valid_labels, inverse_scores)
    inverse_ap = _manual_average_precision(valid_labels, inverse_scores)
    best_auroc_direction = "higher"
    best_auroc = auroc
    if inverse_auroc is not None and (best_auroc is None or inverse_auroc > best_auroc):
        best_auroc_direction = "lower"
        best_auroc = inverse_auroc
    best_direction = "higher"
    best_ap = ap
    if inverse_ap is not None and (best_ap is None or inverse_ap > best_ap):
        best_direction = "lower"
        best_ap = inverse_ap
    return {
        "direction": best_direction,
        "count": len(valid_scores),
        "positive_count": len(positives),
        "negative_count": len(negatives),
        "auroc": auroc,
        "average_precision": ap,
        "inverse_auroc": inverse_auroc,
        "inverse_average_precision": inverse_ap,
        "best_auroc": best_auroc,
        "best_auroc_direction": best_auroc_direction,
        "best_average_precision": best_ap,
        "positive_mean": float(sum(positives) / len(positives)) if positives else None,
        "negative_mean": float(sum(negatives) / len(negatives)) if negatives else None,
    }


def _manual_combo_feature_label(feature: object) -> str:
    text = str(feature or "")
    engine = _feature_engine(text)
    category = _feature_category(text)
    return f"{engine} | {category} | {_combo_feature_term(text)}"


def _manual_combo_values(
    selected_values: pd.DataFrame,
    operation: str,
    numerator_feature: str | None = None,
) -> pd.Series:
    operation = operation.lower()
    values = selected_values.apply(pd.to_numeric, errors="coerce")
    if operation == "average":
        return values.mean(axis=1, skipna=False)
    if operation == "ratio":
        numerator = numerator_feature or str(values.columns[0])
        denominator_columns = [column for column in values.columns if column != numerator]
        denominator = values[denominator_columns].prod(axis=1, skipna=False)
        denominator = denominator.mask(denominator == 0)
        return values[numerator] / denominator
    return values.prod(axis=1, skipna=False)


def _manual_combo_feature_name(selected_features: list[str], operation: str, numerator_feature: str | None = None) -> str:
    terms = [_combo_feature_term(feature) for feature in selected_features]
    if operation == "average":
        return f"Manual average: {'; '.join(terms)}"
    if operation == "ratio":
        numerator = numerator_feature or selected_features[0]
        denominator_terms = [_combo_feature_term(feature) for feature in selected_features if feature != numerator]
        return f"Manual ratio: {_combo_feature_term(numerator)} / {'; '.join(denominator_terms)}"
    return f"Manual product: {'; '.join(terms)}"


def _show_manual_feature_composer(
    *,
    run_key: str,
    ranking_key: str,
    ranking: pd.DataFrame,
    metrics: pd.DataFrame | None,
    selected_engines: list[str],
    target_filter_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if metrics is None or metrics.empty:
        st.info("Manual feature combinations require a metrics table.")
        return pd.DataFrame(), metrics
    label_col = _label_column(metrics)
    if not label_col:
        st.info("Manual feature combinations require binder/nonbinder labels.")
        return pd.DataFrame(), metrics

    source = ranking.copy()
    if "feature" not in source.columns:
        st.info("No source features are available for manual combinations.")
        return pd.DataFrame(), metrics
    if "engine" not in source.columns:
        source["engine"] = source["feature"].map(_feature_engine)
    if "category" not in source.columns:
        source["category"] = source["feature"].map(_feature_category)
    for column in ["best_average_precision", "best_auroc"]:
        if column in source.columns:
            source[column] = pd.to_numeric(source[column], errors="coerce")

    source = source[
        source["category"].astype(str).isin(MANUAL_FEATURE_COMBO_SOURCE_CATEGORIES)
        & source["engine"].astype(str).isin(selected_engines)
        & source["feature"].astype(str).isin(set(str(column) for column in metrics.columns))
        & source["feature"].map(_is_informative_feature)
        & ~source["feature"].map(_is_chain_indexed_confidence_feature)
    ].copy()
    if source.empty:
        st.info("No engine-derived source features are available for the selected engines.")
        return pd.DataFrame(), metrics

    st.caption(
        "Build one temporary feature from existing engine-derived features. "
        "Source features are limited to engine confidence/PAE, recalculated PAE/interface, Rosetta, and PyMOL categories."
    )
    query = st.text_input(
        "Filter source features",
        value="",
        key=f"{run_key}_{ranking_key}_{target_filter_label}_manual_combo_filter_v1",
        placeholder="e.g. iptm ipase pdockq rosetta pymol",
    ).strip().lower()
    if query:
        terms = [term for term in re.split(r"\s+", query) if term]
        source = source[source["feature"].map(lambda feature: _metric_query_matches(feature, terms))].copy()
        if source.empty:
            st.info("No source features match the manual-combination filter.")
            return pd.DataFrame(), metrics

    source = source.sort_values(
        ["engine", "category", "best_average_precision", "best_auroc", "feature"],
        ascending=[True, True, False, False, True],
    )
    features = source["feature"].astype(str).drop_duplicates().tolist()
    label_by_feature = {feature: _manual_combo_feature_label(feature) for feature in features}
    selected_features = st.multiselect(
        "Source features",
        features,
        default=[],
        format_func=lambda feature: label_by_feature.get(feature, str(feature)),
        key=f"{run_key}_{ranking_key}_{target_filter_label}_manual_combo_features_v1",
        help="Select at least two numeric source features. They may come from different engines.",
    )
    operation = st.segmented_control(
        "Combination",
        ["product", "average", "ratio"],
        selection_mode="single",
        default="product",
        key=f"{run_key}_{ranking_key}_{target_filter_label}_manual_combo_operation_v1",
    )
    operation = str(operation or "product")
    numerator_feature = None
    if operation == "ratio" and selected_features:
        numerator_feature = st.selectbox(
            "Ratio numerator",
            selected_features,
            format_func=lambda feature: label_by_feature.get(feature, str(feature)),
            key=f"{run_key}_{ranking_key}_{target_filter_label}_manual_combo_numerator_v1",
            help="The selected numerator is divided by the product of the remaining selected features.",
        )

    state_key = f"{run_key}_{ranking_key}_{target_filter_label}_manual_combo_result_v1"
    calculate = st.button(
        "Calculate manual combination",
        key=f"{run_key}_{ranking_key}_{target_filter_label}_manual_combo_calculate_v1",
    )
    if calculate:
        if len(selected_features) < 2:
            st.warning("Select at least two source features.")
        else:
            values = _manual_combo_values(metrics[selected_features], operation, numerator_feature)
            labels_raw = [_truthy_benchmark_label(value) for value in metrics[label_col].tolist()]
            score_row = _score_feature_vector(labels_raw, values)
            if score_row is None:
                st.warning("The selected features do not have enough shared labeled records to score.")
            else:
                feature_name = _manual_combo_feature_name(selected_features, operation, numerator_feature)
                row = {
                    "feature": feature_name,
                    "engine": "Manual combination",
                    "category": MANUAL_FEATURE_COMBO_CATEGORY,
                    "aliases": feature_name,
                    "alias_count": 1,
                    "component_features": "; ".join(selected_features),
                    "operation": operation,
                    **score_row,
                }
                st.session_state[state_key] = {
                    "row": row,
                    "values": values.tolist(),
                    "selected_features": selected_features,
                }
                st.success("Manual combination calculated for the current view.")

    result = st.session_state.get(state_key)
    if not result:
        st.info("Choose source features and click calculate to add a temporary feature to this view.")
        return pd.DataFrame(), metrics

    row = dict(result.get("row") or {})
    values = pd.Series(result.get("values") or [], index=metrics.index, dtype="float64")
    feature = str(row.get("feature") or "Manual combination")
    combo_metrics = metrics.copy()
    combo_metrics[feature] = values
    combo_ranking = pd.DataFrame([row])
    st.dataframe(
        combo_ranking[
            [
                "feature",
                "operation",
                "component_features",
                "direction",
                "best_average_precision",
                "best_auroc",
                "count",
                "positive_count",
                "negative_count",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    return combo_ranking, combo_metrics


def _with_pae_rosetta_combo_rankings(
    ranking: pd.DataFrame,
    metrics: pd.DataFrame | None,
    *,
    max_features_per_side: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if metrics is None or metrics.empty or ranking.empty or "feature" not in ranking.columns:
        return ranking, metrics
    label_col = _label_column(metrics)
    if not label_col:
        return ranking, metrics
    work = ranking.copy()
    if "engine" not in work.columns:
        work["engine"] = work["feature"].map(_feature_engine)
    if "category" not in work.columns:
        work["category"] = work["feature"].map(_feature_category)
    for column in ["best_average_precision", "best_auroc"]:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    labels_raw = [_truthy_benchmark_label(value) for value in metrics[label_col].tolist()]
    combo_metrics = metrics.copy()
    combo_columns: dict[str, pd.Series] = {}
    combo_rows: list[dict[str, Any]] = []
    direction_by_feature = {
        str(row.get("feature") or ""): str(row.get("direction") or "higher")
        for row in work.to_dict(orient="records")
    }
    engines = [
        engine
        for engine in ENGINE_COLOR_DOMAIN
        if engine in set(work["engine"].dropna().astype(str))
        and engine not in {"Input", "Input PyRosetta", "Published metrics", "Other"}
    ]
    for engine in engines:
        engine_rows = work[work["engine"].astype(str) == engine].copy()
        pae_rows = (
            engine_rows[engine_rows["category"] == "Recalculated PAE / Interface"]
            .dropna(subset=["best_average_precision"])
            .sort_values(["best_average_precision", "best_auroc"], ascending=[False, False])
        )
        rosetta_rows = (
            engine_rows[engine_rows["category"] == "Interface Energy / Rosetta"]
            .dropna(subset=["best_average_precision"])
            .sort_values(["best_average_precision", "best_auroc"], ascending=[False, False])
        )
        rosetta_rows = rosetta_rows[rosetta_rows["feature"].map(_is_allowed_pae_rosetta_partner_feature)].copy()
        if pae_rows.empty or rosetta_rows.empty:
            continue
        raw_values: dict[str, pd.Series] = {}
        for feature in [*pae_rows["feature"].astype(str).tolist(), *rosetta_rows["feature"].astype(str).tolist()]:
            if feature in raw_values or feature not in combo_metrics.columns:
                continue
            raw_values[feature] = pd.to_numeric(combo_metrics[feature], errors="coerce")
        for left_feature in pae_rows["feature"].astype(str).tolist():
            left_values = raw_values.get(left_feature)
            if left_values is None:
                continue
            left_direction = direction_by_feature.get(left_feature) or _feature_sort_direction(left_feature)
            for right_feature in rosetta_rows["feature"].astype(str).tolist():
                right_values = raw_values.get(right_feature)
                if right_values is None:
                    continue
                right_direction = direction_by_feature.get(right_feature) or _feature_sort_direction(right_feature)
                combo_name = _combo_feature_name(engine, left_feature, right_feature)
                if combo_name in combo_columns:
                    continue
                combo_values = left_values * right_values
                score_row = _score_feature_vector(labels_raw, combo_values)
                if score_row is None:
                    continue
                combo_columns[combo_name] = combo_values
                combo_rows.append(
                    {
                        "feature": combo_name,
                        "engine": engine,
                        "category": "PAE × Rosetta combinations",
                        "aliases": combo_name,
                        "alias_count": 1,
                        **score_row,
                        "component_direction": _combo_product_direction(left_direction, right_direction),
                    }
                )
    if not combo_rows:
        return ranking, metrics
    combo_metrics = combo_metrics.drop(columns=list(combo_columns), errors="ignore")
    combo_metrics = pd.concat([combo_metrics, pd.DataFrame(combo_columns, index=combo_metrics.index)], axis=1)
    combined = pd.concat([ranking, pd.DataFrame(combo_rows)], ignore_index=True, sort=False)
    return combined, combo_metrics


@st.cache_data(show_spinner=False)
def _cached_pae_rosetta_combo_rankings(
    ranking: pd.DataFrame,
    metrics: pd.DataFrame | None,
    max_features_per_side: int,
    cache_version: int = PAE_ROSETTA_COMBO_CACHE_VERSION,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    _ = (max_features_per_side, cache_version)
    return _with_pae_rosetta_combo_rankings(
        ranking,
        metrics,
        max_features_per_side=0,
    )


def _with_pae_input_rosetta_combo_rankings(
    ranking: pd.DataFrame,
    metrics: pd.DataFrame | None,
    *,
    max_features_per_side: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if metrics is None or metrics.empty or ranking.empty or "feature" not in ranking.columns:
        return ranking, metrics
    label_col = _label_column(metrics)
    if not label_col:
        return ranking, metrics
    work = ranking.copy()
    if "engine" not in work.columns:
        work["engine"] = work["feature"].map(_feature_engine)
    if "category" not in work.columns:
        work["category"] = work["feature"].map(_feature_category)
    for column in ["best_average_precision", "best_auroc"]:
        if column in work.columns:
            work[column] = pd.to_numeric(work[column], errors="coerce")
    input_rows = (
        work[work["category"] == "Input PyRosetta"]
        .dropna(subset=["best_average_precision"])
        .sort_values(["best_average_precision", "best_auroc"], ascending=[False, False])
    )
    input_rows = input_rows[input_rows["feature"].map(_is_allowed_pae_rosetta_partner_feature)].copy()
    if input_rows.empty:
        return ranking, metrics
    labels_raw = [_truthy_benchmark_label(value) for value in metrics[label_col].tolist()]
    combo_metrics = metrics.copy()
    combo_columns: dict[str, pd.Series] = {}
    combo_rows: list[dict[str, Any]] = []
    direction_by_feature = {
        str(row.get("feature") or ""): str(row.get("direction") or "higher")
        for row in work.to_dict(orient="records")
    }
    engines = [
        engine
        for engine in ENGINE_COLOR_DOMAIN
        if engine in set(work["engine"].dropna().astype(str))
        and engine not in {"Input", "Input PyRosetta", "Published metrics", "Other"}
    ]
    for engine in engines:
        engine_rows = work[work["engine"].astype(str) == engine].copy()
        pae_rows = (
            engine_rows[engine_rows["category"] == "Recalculated PAE / Interface"]
            .dropna(subset=["best_average_precision"])
            .sort_values(["best_average_precision", "best_auroc"], ascending=[False, False])
        )
        if pae_rows.empty:
            continue
        raw_values: dict[str, pd.Series] = {}
        for feature in [*pae_rows["feature"].astype(str).tolist(), *input_rows["feature"].astype(str).tolist()]:
            if feature in raw_values or feature not in combo_metrics.columns:
                continue
            raw_values[feature] = pd.to_numeric(combo_metrics[feature], errors="coerce")
        for left_feature in pae_rows["feature"].astype(str).tolist():
            left_values = raw_values.get(left_feature)
            if left_values is None:
                continue
            left_direction = direction_by_feature.get(left_feature) or _feature_sort_direction(left_feature)
            for right_feature in input_rows["feature"].astype(str).tolist():
                right_values = raw_values.get(right_feature)
                if right_values is None:
                    continue
                right_direction = direction_by_feature.get(right_feature) or _feature_sort_direction(right_feature)
                combo_name = _input_combo_feature_name(engine, left_feature, right_feature)
                if combo_name in combo_columns:
                    continue
                combo_values = left_values * right_values
                score_row = _score_feature_vector(labels_raw, combo_values)
                if score_row is None:
                    continue
                combo_columns[combo_name] = combo_values
                combo_rows.append(
                    {
                        "feature": combo_name,
                        "engine": engine,
                        "category": "PAE × Input Rosetta combinations",
                        "aliases": combo_name,
                        "alias_count": 1,
                        **score_row,
                        "component_direction": _combo_product_direction(left_direction, right_direction),
                    }
                )
    if not combo_rows:
        return ranking, metrics
    combo_metrics = combo_metrics.drop(columns=list(combo_columns), errors="ignore")
    combo_metrics = pd.concat([combo_metrics, pd.DataFrame(combo_columns, index=combo_metrics.index)], axis=1)
    combined = pd.concat([ranking, pd.DataFrame(combo_rows)], ignore_index=True, sort=False)
    return combined, combo_metrics


@st.cache_data(show_spinner=False)
def _cached_pae_input_rosetta_combo_rankings(
    ranking: pd.DataFrame,
    metrics: pd.DataFrame | None,
    max_features_per_side: int,
    cache_version: int = PAE_ROSETTA_COMBO_CACHE_VERSION,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    _ = (max_features_per_side, cache_version)
    return _with_pae_input_rosetta_combo_rankings(
        ranking,
        metrics,
        max_features_per_side=0,
    )


def _feature_sort_direction(feature: object) -> str:
    raw = _raw_feature_name(feature)
    text = str(feature or "").lower()
    lower_tokens = [
        "pae",
        "ipae",
        "rmsd",
        "interface_dg",
        "interface_Δg",
        "dg_dsasa",
        "dg_sasa",
        "energy",
        "unsat",
        "clash",
        "error",
        "loss",
    ]
    higher_tokens = [
        "ipsae",
        "lis",
        "pdockq",
        "dockq",
        "iptm",
        "ptm",
        "plddt",
        "confidence",
        "packstat",
        "interface_sc",
        "shape_complement",
    ]
    if " × " in text:
        if any(token in raw or token in text for token in lower_tokens):
            return "lower"
        return "higher"
    if any(token in raw or token in text for token in higher_tokens):
        return "higher"
    if any(token in raw or token in text for token in lower_tokens):
        return "lower"
    return "higher"


def _numeric_metric_feature_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics is None or metrics.empty:
        return pd.DataFrame()
    excluded = {
        "binder_id",
        "candidate_id",
        "target_id",
        "target_chains",
        "binder_chains",
        "binder",
        "label",
        "class",
        "source",
    }
    rows: list[dict[str, Any]] = []
    for column in metrics.columns:
        if column in excluded or column.startswith("_"):
            continue
        values = pd.to_numeric(metrics[column], errors="coerce")
        valid = values.dropna()
        if valid.empty:
            continue
        rows.append(
            {
                "feature": column,
                "engine": _feature_engine(column),
                "category": _feature_category(column),
                "direction": _feature_sort_direction(column),
                "valid_records": int(valid.size),
                "mean": float(valid.mean()),
                "min": float(valid.min()),
                "max": float(valid.max()),
            }
        )
    if not rows:
        return pd.DataFrame()
    order = {name: idx for idx, name in enumerate(ENGINE_COLOR_DOMAIN)}
    result = pd.DataFrame(rows)
    result["_engine_order"] = result["engine"].map(lambda value: order.get(str(value), 999))
    return (
        result.sort_values(["_engine_order", "category", "feature"])
        .drop(columns=["_engine_order"])
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def _cached_numeric_metric_feature_table(metrics: pd.DataFrame) -> pd.DataFrame:
    return _numeric_metric_feature_table(metrics)


def _with_pae_rosetta_combo_metric_columns(
    metrics: pd.DataFrame | None,
    *,
    max_features_per_side: int = 8,
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    if metrics is None or metrics.empty:
        return metrics, pd.DataFrame()
    feature_table = _cached_numeric_metric_feature_table(metrics)
    if feature_table.empty:
        return metrics, pd.DataFrame()
    combo_metrics = metrics.copy()
    combo_columns: dict[str, pd.Series] = {}
    combo_rows: list[dict[str, Any]] = []
    engines = [
        engine
        for engine in ENGINE_COLOR_DOMAIN
        if engine in set(feature_table["engine"].dropna().astype(str))
        and engine not in {"Input", "Input PyRosetta", "Published metrics", "Other"}
    ]
    for engine in engines:
        engine_rows = feature_table[feature_table["engine"].astype(str) == engine].copy()
        pae_rows = (
            engine_rows[engine_rows["category"] == "Recalculated PAE / Interface"]
            .sort_values(["valid_records", "feature"], ascending=[False, True])
        )
        rosetta_rows = (
            engine_rows[engine_rows["category"] == "Interface Energy / Rosetta"]
            .sort_values(["valid_records", "feature"], ascending=[False, True])
        )
        rosetta_rows = rosetta_rows[rosetta_rows["feature"].map(_is_allowed_pae_rosetta_partner_feature)].copy()
        if pae_rows.empty or rosetta_rows.empty:
            continue
        raw_values: dict[str, pd.Series] = {}
        direction_by_feature: dict[str, str] = {}
        for feature, direction in [
            *zip(pae_rows["feature"].astype(str), pae_rows["direction"].astype(str)),
            *zip(rosetta_rows["feature"].astype(str), rosetta_rows["direction"].astype(str)),
        ]:
            if feature in raw_values or feature not in combo_metrics.columns:
                continue
            raw_values[feature] = pd.to_numeric(combo_metrics[feature], errors="coerce")
            direction_by_feature[feature] = direction
        for left_feature in pae_rows["feature"].astype(str).tolist():
            left_values = raw_values.get(left_feature)
            if left_values is None:
                continue
            left_direction = direction_by_feature.get(left_feature) or _feature_sort_direction(left_feature)
            for right_feature in rosetta_rows["feature"].astype(str).tolist():
                right_values = raw_values.get(right_feature)
                if right_values is None:
                    continue
                right_direction = direction_by_feature.get(right_feature) or _feature_sort_direction(right_feature)
                combo_name = _combo_feature_name(engine, left_feature, right_feature)
                if combo_name in combo_columns:
                    continue
                combo_values = left_values * right_values
                valid = pd.to_numeric(combo_values, errors="coerce").dropna()
                if valid.empty:
                    continue
                combo_columns[combo_name] = combo_values
                combo_rows.append(
                    {
                        "feature": combo_name,
                        "engine": engine,
                        "category": "PAE × Rosetta combinations",
                        "direction": _combo_product_direction(left_direction, right_direction),
                        "valid_records": int(valid.size),
                        "mean": float(valid.mean()),
                        "min": float(valid.min()),
                        "max": float(valid.max()),
                    }
                )
    if combo_columns:
        combo_metrics = combo_metrics.drop(columns=list(combo_columns), errors="ignore")
        combo_metrics = pd.concat([combo_metrics, pd.DataFrame(combo_columns, index=combo_metrics.index)], axis=1)
    return combo_metrics, pd.DataFrame(combo_rows)


@st.cache_data(show_spinner=False)
def _cached_pae_rosetta_combo_metric_columns(
    metrics: pd.DataFrame | None,
    max_features_per_side: int,
    cache_version: int = PAE_ROSETTA_COMBO_CACHE_VERSION,
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    _ = (max_features_per_side, cache_version)
    return _with_pae_rosetta_combo_metric_columns(
        metrics,
        max_features_per_side=0,
    )


def _with_pae_input_rosetta_combo_metric_columns(
    metrics: pd.DataFrame | None,
    *,
    max_features_per_side: int = 8,
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    if metrics is None or metrics.empty:
        return metrics, pd.DataFrame()
    feature_table = _cached_numeric_metric_feature_table(metrics)
    if feature_table.empty:
        return metrics, pd.DataFrame()
    input_rows = (
        feature_table[feature_table["category"] == "Input PyRosetta"]
        .sort_values(["valid_records", "feature"], ascending=[False, True])
    )
    input_rows = input_rows[input_rows["feature"].map(_is_allowed_pae_rosetta_partner_feature)].copy()
    if input_rows.empty:
        return metrics, pd.DataFrame()
    combo_metrics = metrics.copy()
    combo_columns: dict[str, pd.Series] = {}
    combo_rows: list[dict[str, Any]] = []
    engines = [
        engine
        for engine in ENGINE_COLOR_DOMAIN
        if engine in set(feature_table["engine"].dropna().astype(str))
        and engine not in {"Input", "Input PyRosetta", "Published metrics", "Other"}
    ]
    for engine in engines:
        engine_rows = feature_table[feature_table["engine"].astype(str) == engine].copy()
        pae_rows = (
            engine_rows[engine_rows["category"] == "Recalculated PAE / Interface"]
            .sort_values(["valid_records", "feature"], ascending=[False, True])
        )
        if pae_rows.empty:
            continue
        raw_values: dict[str, pd.Series] = {}
        direction_by_feature: dict[str, str] = {}
        for feature, direction in [
            *zip(pae_rows["feature"].astype(str), pae_rows["direction"].astype(str)),
            *zip(input_rows["feature"].astype(str), input_rows["direction"].astype(str)),
        ]:
            if feature in raw_values or feature not in combo_metrics.columns:
                continue
            raw_values[feature] = pd.to_numeric(combo_metrics[feature], errors="coerce")
            direction_by_feature[feature] = direction
        for left_feature in pae_rows["feature"].astype(str).tolist():
            left_values = raw_values.get(left_feature)
            if left_values is None:
                continue
            left_direction = direction_by_feature.get(left_feature) or _feature_sort_direction(left_feature)
            for right_feature in input_rows["feature"].astype(str).tolist():
                right_values = raw_values.get(right_feature)
                if right_values is None:
                    continue
                right_direction = direction_by_feature.get(right_feature) or _feature_sort_direction(right_feature)
                combo_name = _input_combo_feature_name(engine, left_feature, right_feature)
                if combo_name in combo_columns:
                    continue
                combo_values = left_values * right_values
                valid = pd.to_numeric(combo_values, errors="coerce").dropna()
                if valid.empty:
                    continue
                combo_columns[combo_name] = combo_values
                combo_rows.append(
                    {
                        "feature": combo_name,
                        "engine": engine,
                        "category": "PAE × Input Rosetta combinations",
                        "direction": _combo_product_direction(left_direction, right_direction),
                        "valid_records": int(valid.size),
                        "mean": float(valid.mean()),
                        "min": float(valid.min()),
                        "max": float(valid.max()),
                    }
                )
    if combo_columns:
        combo_metrics = combo_metrics.drop(columns=list(combo_columns), errors="ignore")
        combo_metrics = pd.concat([combo_metrics, pd.DataFrame(combo_columns, index=combo_metrics.index)], axis=1)
    return combo_metrics, pd.DataFrame(combo_rows)


@st.cache_data(show_spinner=False)
def _cached_pae_input_rosetta_combo_metric_columns(
    metrics: pd.DataFrame | None,
    max_features_per_side: int,
    cache_version: int = PAE_ROSETTA_COMBO_CACHE_VERSION,
) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    _ = (max_features_per_side, cache_version)
    return _with_pae_input_rosetta_combo_metric_columns(
        metrics,
        max_features_per_side=0,
    )


COMBO_CACHE_KEY_COLUMNS = [
    "binder_id",
    "candidate_id",
    "target_id",
    "source",
    "label",
    "binder",
]


def _combo_cache_paths(cache_dir: Path, kind: str) -> tuple[Path, Path]:
    cache_dir = cache_dir / "derived_metric_cache"
    stem = f"{kind}_v{PAE_ROSETTA_COMBO_CACHE_VERSION}"
    return cache_dir / f"{stem}_ranking.csv", cache_dir / f"{stem}_metrics.csv"


def _combo_cache_kind(*parts: object) -> str:
    text = "_".join(str(part or "").strip() for part in parts if str(part or "").strip())
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._-")
    return text or "combos"


def _combo_cache_key_columns(metrics: pd.DataFrame | None, cached_metrics: pd.DataFrame | None = None) -> list[str]:
    if metrics is None or metrics.empty:
        return []
    keys = [column for column in COMBO_CACHE_KEY_COLUMNS if column in metrics.columns]
    if cached_metrics is not None and not cached_metrics.empty:
        keys = [column for column in keys if column in cached_metrics.columns]
    return keys


def _combo_metric_columns_for_rows(metrics: pd.DataFrame | None, combo_rows: pd.DataFrame, category: str) -> list[str]:
    if metrics is None or metrics.empty or combo_rows.empty or "feature" not in combo_rows.columns:
        return []
    features = combo_rows.loc[
        combo_rows.get("category", pd.Series(index=combo_rows.index, dtype=str)).astype(str) == category,
        "feature",
    ]
    return [feature for feature in features.dropna().astype(str).drop_duplicates().tolist() if feature in metrics.columns]


def _save_combo_cache(
    cache_dir: Path | None,
    kind: str,
    ranking: pd.DataFrame,
    metrics: pd.DataFrame | None,
    category: str,
) -> None:
    if cache_dir is None or ranking.empty or metrics is None or metrics.empty:
        return
    combo_rows = ranking[ranking.get("category", pd.Series(index=ranking.index, dtype=str)).astype(str) == category].copy()
    combo_columns = _combo_metric_columns_for_rows(metrics, combo_rows, category)
    if combo_rows.empty or not combo_columns:
        return
    ranking_path, metrics_path = _combo_cache_paths(cache_dir, kind)
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    combo_rows = combo_rows.drop_duplicates(subset=["feature"], keep="last")
    key_columns = _combo_cache_key_columns(metrics)
    metrics_cache = metrics[[*key_columns, *combo_columns]].copy()
    combo_rows.to_csv(ranking_path, index=False)
    metrics_cache.to_csv(metrics_path, index=False)


def _load_combo_cache(
    cache_dir: Path | None,
    kind: str,
    ranking: pd.DataFrame,
    metrics: pd.DataFrame | None,
    category: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None, bool]:
    if cache_dir is None or metrics is None or metrics.empty:
        return ranking, metrics, False
    ranking_path, metrics_path = _combo_cache_paths(cache_dir, kind)
    cached_rows = _read_csv(ranking_path)
    cached_metrics = _read_csv(metrics_path)
    if cached_rows is None or cached_rows.empty or cached_metrics is None or cached_metrics.empty:
        return ranking, metrics, False
    if "feature" not in cached_rows.columns:
        return ranking, metrics, False
    if "category" not in cached_rows.columns:
        cached_rows["category"] = category
    combo_columns = _combo_metric_columns_for_rows(cached_metrics, cached_rows, category)
    if not combo_columns:
        return ranking, metrics, False

    merged_metrics = metrics.drop(columns=combo_columns, errors="ignore").copy()
    key_columns = _combo_cache_key_columns(merged_metrics, cached_metrics)
    if key_columns:
        cached_subset = cached_metrics[[*key_columns, *combo_columns]].copy()
        cached_subset = cached_subset.drop_duplicates(subset=key_columns, keep="last")
        merged_metrics = merged_metrics.merge(cached_subset, on=key_columns, how="left")
    elif len(cached_metrics) == len(merged_metrics):
        for column in combo_columns:
            merged_metrics[column] = cached_metrics[column].to_numpy()
    else:
        return ranking, metrics, False

    existing_features = set(ranking.get("feature", pd.Series(dtype=str)).dropna().astype(str))
    new_rows = cached_rows[~cached_rows["feature"].astype(str).isin(existing_features)].copy()
    merged_ranking = ranking.copy()
    if not new_rows.empty:
        merged_ranking = pd.concat([merged_ranking, new_rows], ignore_index=True, sort=False)
    return merged_ranking, merged_metrics, True


def _with_saved_combo_cache_metrics(benchmark_dir: Path, metrics: pd.DataFrame | None) -> pd.DataFrame | None:
    if metrics is None or metrics.empty:
        return metrics
    cache_dir = benchmark_dir / "derived_metric_cache"
    if not cache_dir.exists():
        return metrics
    merged = metrics.copy()
    version_marker = f"_v{PAE_ROSETTA_COMBO_CACHE_VERSION}_"
    for ranking_path in sorted(cache_dir.glob(f"*{version_marker}ranking.csv"), key=lambda path: path.stat().st_mtime):
        metrics_path = ranking_path.with_name(ranking_path.name.replace("_ranking.csv", "_metrics.csv"))
        cached_rows = _read_csv(ranking_path)
        cached_metrics = _read_csv(metrics_path)
        if cached_rows is None or cached_rows.empty or cached_metrics is None or cached_metrics.empty:
            continue
        category_series = cached_rows.get("category", pd.Series(index=cached_rows.index, dtype=str)).dropna().astype(str)
        categories = category_series.drop_duplicates().tolist() or ["PAE × Rosetta combinations"]
        combo_columns: list[str] = []
        for category in categories:
            combo_columns.extend(_combo_metric_columns_for_rows(cached_metrics, cached_rows, category))
        combo_columns = [column for column in dict.fromkeys(combo_columns) if column not in merged.columns]
        if not combo_columns:
            continue
        key_columns = _combo_cache_key_columns(merged, cached_metrics)
        if key_columns:
            cached_subset = cached_metrics[[*key_columns, *combo_columns]].copy()
            cached_subset = cached_subset.drop_duplicates(subset=key_columns, keep="last")
            merged = merged.merge(cached_subset, on=key_columns, how="left")
        elif len(cached_metrics) == len(merged):
            for column in combo_columns:
                merged[column] = cached_metrics[column].to_numpy()
    return merged


@st.cache_data(show_spinner=False)
def _source_candidate_rank_table(run_dir_text: str) -> pd.DataFrame:
    run_dir = Path(run_dir_text)
    input_payload = read_json(run_dir / "input.json")
    input_section = input_payload.get("inputs") if isinstance(input_payload.get("inputs"), dict) else {}
    source_run_dir = _resolve_run_relative_path(
        run_dir,
        input_payload.get("source_run_dir") or input_section.get("source_run_dir"),
    )
    candidate_paths: list[Path] = []
    candidates_jsonl = _resolve_run_relative_path(
        run_dir,
        input_payload.get("candidates_jsonl") or input_section.get("candidates_jsonl"),
    )
    if candidates_jsonl is not None:
        candidate_paths.append(candidates_jsonl)
    if source_run_dir is not None:
        candidate_paths.append(source_run_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl")
        candidate_paths.append(source_run_dir / "artifacts" / "normalized_candidates" / "imported_candidate_metrics.csv")
    candidate_paths.append(run_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl")
    candidate_paths.append(run_dir / "artifacts" / "normalized_candidates" / "imported_candidate_metrics.csv")
    rank_fields = [
        "derived_binding_confidence_rank",
        "boltzbio_rank",
        "source_rank",
        "source_metric_boltzbio_rank",
        "import_rank",
        "import_index",
        "import_table_row",
        "bindcraft_final_rank",
        "bindcraft_Rank",
        "bindcraft_ranked_pdb_rank",
        "input_rank",
        "original_rank",
        "rank",
    ]
    sequence_fields = [
        "binder_sequence",
        "bindcraft_Sequence",
        "sequence",
        "seq",
        "designed_sequence",
        "amino_acid_sequence",
    ]
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    rows_by_key: dict[str, dict[str, Any]] = {}

    def _with_derived_binding_confidence_rank(table: pd.DataFrame) -> pd.DataFrame:
        if table.empty or "derived_binding_confidence_rank" in table.columns:
            return table
        score_column = next(
            (
                col
                for col in table.columns
                if "binding confidence" in str(col).lower()
            ),
            None,
        )
        if score_column is None:
            return table
        result = table.copy()
        scores = pd.to_numeric(result[score_column], errors="coerce")
        if scores.notna().any():
            result["derived_binding_confidence_rank"] = scores.rank(method="min", ascending=False)
            result["derived_binding_confidence_score"] = scores
        return result

    def _candidate_key_variants(candidate_text: str) -> list[str]:
        base = candidate_text.strip().lower()
        underscored = re.sub(r"[^a-z0-9]+", "_", base).strip("_")
        compact = re.sub(r"[^a-z0-9]+", "", base)
        return [key for key in dict.fromkeys([base, underscored, compact]) if key]

    def _add_candidate_metadata(candidate_id: object, payload: dict[str, Any]) -> None:
        candidate_text = str(candidate_id or "").strip()
        if not candidate_text:
            return
        for key in _candidate_key_variants(candidate_text):
            row = rows_by_key.setdefault(key, {"_candidate_key": key})
            if "sequence" not in row:
                for field in sequence_fields:
                    value = payload.get(field)
                    if value in {None, ""}:
                        continue
                    sequence = str(value).strip()
                    if sequence:
                        row["sequence"] = sequence
                        row["sequence_length"] = len(sequence)
                        row["sequence_source"] = field
                        break
            current_rank_source = str(row.get("original_rank_source") or "")
            can_replace_rank = "original_rank" not in row or current_rank_source in {"import_index", "import_table_row"}
            if can_replace_rank:
                for field in rank_fields:
                    value = payload.get(field)
                    if value in {None, ""}:
                        continue
                    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                    if pd.isna(numeric):
                        continue
                    if field in {"import_index", "import_table_row"} and "original_rank" in row:
                        continue
                    row["original_rank"] = int(numeric) if float(numeric).is_integer() else float(numeric)
                    row["original_rank_source"] = field
                    break
            if len(row) > 1 and key not in seen:
                rows.append(row)
                seen.add(key)

    for path in dict.fromkeys(candidate_paths):
        if path is None or not path.exists():
            continue
        try:
            if path.suffix.lower() == ".jsonl":
                for candidate in read_candidates(path):
                    metrics = candidate.get("metrics") if isinstance(candidate, dict) else {}
                    payload: dict[str, Any] = {}
                    if isinstance(metrics, dict):
                        payload.update(metrics)
                    if isinstance(candidate, dict):
                        payload.update(
                            {
                                key: candidate.get(key)
                                for key in [*rank_fields, *sequence_fields]
                                if key in candidate
                            }
                        )
                        _add_candidate_metadata(candidate.get("candidate_id"), payload)
            elif path.suffix.lower() == ".csv":
                table = pd.read_csv(path)
                table = _with_derived_binding_confidence_rank(table)
                id_column = next((col for col in ["candidate_id", "binder_id", "design"] if col in table.columns), None)
                if id_column:
                    for _, row in table.iterrows():
                        _add_candidate_metadata(row.get(id_column), row.to_dict())
        except Exception:
            continue
    return pd.DataFrame(rows)


def _label_free_metric_design_selector(
    records: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    run_key: str,
    run_dir: Path | None = None,
) -> str | None:
    if metrics is None or metrics.empty or "binder_id" not in metrics.columns:
        return None
    allowed_designs = set(records["binder_id"].dropna().astype(str))
    if not allowed_designs:
        return None
    metrics_with_combos = metrics
    if metrics_with_combos is None or metrics_with_combos.empty:
        return None
    feature_table = _cached_numeric_metric_feature_table(metrics_with_combos)
    if feature_table.empty:
        return None
    feature_table = feature_table[
        feature_table["feature"].astype(str).isin(set(metrics_with_combos.columns))
    ].copy()
    all_metric_feature_table = feature_table.copy()
    engine_order = [
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
        "Input PyRosetta",
    ]
    engines = [engine for engine in engine_order if engine in set(feature_table["engine"].astype(str))]
    engines.extend(sorted(engine for engine in set(feature_table["engine"].astype(str)) if engine not in set(engines)))
    selected_engines = st.multiselect(
        "Metric engines",
        engines,
        default=engines,
        key=f"{run_key}_refolding_metric_engines_v1",
        help="Filter raw refolding metrics by engine/evaluation source. Input PyRosetta is the original input complex.",
    )
    if selected_engines:
        feature_table = feature_table[feature_table["engine"].astype(str).isin(selected_engines)].copy()
    else:
        st.info("Select at least one metric engine.")
        return None
    categories = [
        "All",
        "Input PyRosetta",
        "Engine Confidence / PAE",
        "Recalculated PAE / Interface",
        "Interface Energy / Rosetta",
        "Interface Geometry / PyMOL",
        "PAE × Rosetta combinations",
        "PAE × Input Rosetta combinations",
        MANUAL_FEATURE_COMBO_CATEGORY,
    ]
    base_categories = set(feature_table["category"].astype(str))
    available_categories = [category for category in categories if category == "All" or category in base_categories]
    if (
        "PAE × Rosetta combinations" not in available_categories
        and "Recalculated PAE / Interface" in base_categories
        and "Interface Energy / Rosetta" in base_categories
    ):
        available_categories.append("PAE × Rosetta combinations")
    if (
        "PAE × Input Rosetta combinations" not in available_categories
        and "Recalculated PAE / Interface" in base_categories
        and "Input PyRosetta" in base_categories
    ):
        available_categories.append("PAE × Input Rosetta combinations")
    selected_category = st.segmented_control(
        "Metric category",
        available_categories,
        default="All",
        key=f"{run_key}_refolding_metric_category_v1",
    )
    selected_category = selected_category or "All"
    if selected_category == "PAE × Rosetta combinations":
        metrics_with_combos = metrics
        combo_ready_key = f"{run_key}_refolding_pae_rosetta_combos_ready_v1"
        rebuild_combos = st.button(
            "Rebuild PAE × Rosetta combinations",
            key=f"{run_key}_refolding_rebuild_pae_rosetta_combos_v1",
            help="Clear the cached combination table and rebuild raw PAE/interface × Rosetta products from the current metrics.",
        )
        if rebuild_combos:
            _clear_pae_rosetta_combo_caches()
            st.success("Combination cache cleared. Rebuilding from the current metrics.")
            with st.spinner("Building PAE × Rosetta combination metrics..."):
                metrics_with_combos, combo_features = _cached_pae_rosetta_combo_metric_columns(metrics, 8)
            if combo_features is not None and not combo_features.empty:
                st.session_state[combo_ready_key] = True
        elif st.session_state.get(combo_ready_key):
            metrics_with_combos, combo_features = _cached_pae_rosetta_combo_metric_columns(metrics, 8)
        else:
            combo_features = all_metric_feature_table[
                all_metric_feature_table["category"].astype(str) == "PAE × Rosetta combinations"
            ].copy()
        if metrics_with_combos is None or metrics_with_combos.empty or combo_features.empty:
            st.info("No PAE × Rosetta combination metrics are available yet. Click rebuild to compute them.")
            return None
        if not combo_features.empty:
            all_metric_feature_table = pd.concat([all_metric_feature_table, combo_features], ignore_index=True, sort=False)
        feature_table = combo_features.copy()
    if selected_category == "PAE × Input Rosetta combinations":
        metrics_with_combos = metrics
        combo_ready_key = f"{run_key}_refolding_pae_input_rosetta_combos_ready_v1"
        rebuild_combos = st.button(
            "Rebuild PAE × Input Rosetta combinations",
            key=f"{run_key}_refolding_rebuild_pae_input_rosetta_combos_v1",
            help="Clear the cached combination table and rebuild engine PAE/interface × shared input PyRosetta products.",
        )
        if rebuild_combos:
            _clear_pae_rosetta_combo_caches()
            st.success("Combination cache cleared. Rebuilding from the current metrics.")
            with st.spinner("Building PAE × Input Rosetta combination metrics..."):
                metrics_with_combos, combo_features = _cached_pae_input_rosetta_combo_metric_columns(metrics, 8)
            if combo_features is not None and not combo_features.empty:
                st.session_state[combo_ready_key] = True
        elif st.session_state.get(combo_ready_key):
            metrics_with_combos, combo_features = _cached_pae_input_rosetta_combo_metric_columns(metrics, 8)
        else:
            combo_features = all_metric_feature_table[
                all_metric_feature_table["category"].astype(str) == "PAE × Input Rosetta combinations"
            ].copy()
        if metrics_with_combos is None or metrics_with_combos.empty or combo_features.empty:
            st.info("No PAE × Input Rosetta combination metrics are available yet. Click rebuild to compute them.")
            return None
        if not combo_features.empty:
            all_metric_feature_table = pd.concat([all_metric_feature_table, combo_features], ignore_index=True, sort=False)
        feature_table = combo_features.copy()
    if selected_category != "All":
        feature_table = feature_table[feature_table["category"].astype(str) == selected_category].copy()
    query = st.text_input(
        "Filter metrics",
        value="",
        key=f"{run_key}_refolding_metric_search_v1",
        placeholder="e.g. ipSAE, interface_dG, input_rosetta, packstat",
    ).strip().lower()
    if query:
        terms = [term for term in re.split(r"\s+", query) if term]
        feature_table = feature_table[
            feature_table["feature"].map(lambda value: _metric_query_matches(value, terms))
        ].copy()
    if feature_table.empty:
        st.info("No metrics match the current engine/category/filter selection.")
        return None
    feature_table = feature_table.sort_values(["engine", "category", "feature"]).reset_index(drop=True)
    direction_lookup = {
        str(row.get("feature") or ""): str(row.get("direction") or _feature_sort_direction(row.get("feature")))
        for row in feature_table.to_dict(orient="records")
    }

    base_ranked = metrics_with_combos[metrics_with_combos["binder_id"].astype(str).isin(allowed_designs)].copy()
    st.subheader("Filtered Metric Ranking")
    st.caption(
        f"{feature_table['feature'].nunique()} filtered metric column(s) across "
        f"{feature_table['engine'].nunique()} engine/source group(s). "
        "The detailed ranked plot below drives the selected structure."
    )
    matrix_engines = [
        engine
        for engine in engine_order
        if engine in set(feature_table["engine"].dropna().astype(str))
    ]
    matrix_engines.extend(
        sorted(engine for engine in set(feature_table["engine"].dropna().astype(str)) if engine not in set(matrix_engines))
    )
    feature_table["_metric_family"] = feature_table["feature"].map(_metric_family_key)
    family_summary = (
        feature_table.groupby("_metric_family", dropna=False)
        .agg(
            engines=("engine", lambda values: len(set(str(value) for value in values if str(value)))),
            valid_records=("valid_records", "max"),
            category=("category", "first"),
        )
        .reset_index()
        .sort_values(["engines", "valid_records", "_metric_family"], ascending=[False, False, True])
    )
    family_options = [str(value) for value in family_summary["_metric_family"].tolist() if str(value)]
    selected_family = None
    if family_options:
        previous_family_key = f"{run_key}_refolding_metric_matrix_family_v1"
        previous_family = st.session_state.get(previous_family_key)
        family_index = family_options.index(previous_family) if previous_family in family_options else 0
        family_labels = {
            str(row.get("_metric_family") or ""): (
                f"{_combo_feature_term(row.get('_metric_family'))} | "
                f"{int(row.get('engines') or 0)} engine(s) | "
                f"{row.get('category') or 'metric'}"
            )
            for row in family_summary.to_dict(orient="records")
        }
        selected_family = st.selectbox(
            "Metric family for all engines",
            family_options,
            index=family_index,
            key=previous_family_key,
            help=(
                "Choose one feature family once. With an empty filter this selector shows all families in the "
                "current category; the text filter above is the only narrowing control."
            ),
            format_func=lambda family: family_labels.get(str(family), str(family)),
        )
    matrix_feature_choices: list[str] = []
    preview_rows: list[dict[str, Any]] = []
    for engine in matrix_engines:
        engine_features = feature_table[feature_table["engine"].astype(str) == engine].copy()
        if engine_features.empty:
            continue
        if selected_family:
            family_features = engine_features[engine_features["_metric_family"].astype(str) == str(selected_family)].copy()
            if not family_features.empty:
                engine_features = family_features
        engine_features = engine_features.sort_values(["valid_records", "feature"], ascending=[False, True]).reset_index(drop=True)
        feature_options_for_engine = engine_features["feature"].astype(str).tolist()
        default_feature = feature_options_for_engine[0]
        matrix_feature_choices.append(default_feature)
        preview_rows.append(
            {
                "engine": engine,
                "matrix_feature": default_feature,
                "valid_records": int(engine_features.iloc[0].get("valid_records") or 0),
            }
        )
    if preview_rows:
        with st.expander("Per-engine preview features", expanded=False):
            st.dataframe(pd.DataFrame(preview_rows), hide_index=True, use_container_width=True)
    show_metric_matrix = st.checkbox(
        "Show per-engine ranking matrix",
        value=False,
        key=f"{run_key}_refolding_metric_show_matrix_v1",
        help="Rendering many Altair charts can be slow for large refolding runs. Leave this off for faster feature and structure selection.",
    )
    if show_metric_matrix:
        for row_start in range(0, len(matrix_engines), 3):
            cols = st.columns(3)
            for col, engine in zip(cols, matrix_engines[row_start : row_start + 3]):
                engine_features = feature_table[feature_table["engine"].astype(str) == engine].copy()
                if engine_features.empty:
                    continue
                if selected_family:
                    engine_features = engine_features[
                        engine_features["_metric_family"].astype(str) == str(selected_family)
                    ].copy()
                    if engine_features.empty:
                        with col:
                            st.markdown(f"**{engine}**")
                            st.info(f"No `{selected_family}` metric for this engine.")
                        continue
                engine_features = engine_features.sort_values(["valid_records", "feature"], ascending=[False, True]).reset_index(drop=True)
                matrix_feature = str(engine_features.iloc[0]["feature"])
                with col:
                    st.markdown(f"**{engine}**")
                    st.caption(_combo_feature_term(matrix_feature))
                    if str(matrix_feature) not in matrix_feature_choices:
                        matrix_feature_choices.append(str(matrix_feature))
                    values = pd.to_numeric(base_ranked[matrix_feature], errors="coerce")
                    engine_df = base_ranked.loc[
                        values.notna(),
                        [col_name for col_name in ["binder_id", "target_id"] if col_name in base_ranked.columns],
                    ].copy()
                    if engine_df.empty:
                        st.info("No valid values for this feature.")
                        continue
                    feature_direction = direction_lookup.get(str(matrix_feature), _feature_sort_direction(matrix_feature))
                    engine_df["feature_value"] = values.loc[values.notna()].astype(float).to_numpy()
                    oriented = -engine_df["feature_value"] if feature_direction == "lower" else engine_df["feature_value"]
                    engine_df["rank_score"] = oriented
                    engine_df["feature"] = matrix_feature
                    engine_df["direction"] = feature_direction
                    engine_df["engine"] = engine
                    engine_df = engine_df.sort_values("rank_score", ascending=False).reset_index(drop=True)
                    engine_df["rank"] = engine_df.index + 1
                    chart = (
                        alt.Chart(engine_df)
                        .mark_circle(size=58, opacity=0.85)
                        .encode(
                            x=alt.X("rank:Q", title="rank"),
                            y=alt.Y("rank_score:Q", title="rank score"),
                            color=alt.value("#4F8DF7"),
                            tooltip=[
                                "engine:N",
                                "feature:N",
                                "binder_id:N",
                                "direction:N",
                                "rank:Q",
                                alt.Tooltip("feature_value:Q", title="feature value", format=".4f"),
                                alt.Tooltip("rank_score:Q", title="rank score", format=".4f"),
                            ],
                        )
                        .properties(height=220)
                        .interactive()
                    )
                    st.altair_chart(chart, width="stretch", key=f"{run_key}_refolding_metric_matrix_{engine}_v2")

    if selected_family and preview_rows:
        single_engine_feature = {
            str(row["engine"]): str(row["matrix_feature"])
            for row in preview_rows
            if row.get("engine") and row.get("matrix_feature")
        }
        single_engine_options = [engine for engine in matrix_engines if engine in single_engine_feature]
        previous_engine_key = f"{run_key}_refolding_metric_single_engine_v1"
        previous_engine = st.session_state.get(previous_engine_key)
        engine_index = single_engine_options.index(previous_engine) if previous_engine in single_engine_options else 0
        selected_single_engine = st.selectbox(
            "Engine for single ranked plot",
            single_engine_options,
            index=engine_index,
            key=previous_engine_key,
            help=(
                "The metric family above is applied across engines. This selector only chooses which engine's "
                "version drives the detailed scatter and structure selection below."
            ),
        )
        selected_feature = single_engine_feature[str(selected_single_engine)]
        selected_feature_rows = feature_table[feature_table["feature"].astype(str) == str(selected_feature)]
        selected_record_count = (
            int(selected_feature_rows.iloc[0].get("valid_records") or 0)
            if not selected_feature_rows.empty
            else int(pd.to_numeric(base_ranked.get(selected_feature), errors="coerce").notna().sum())
        )
        st.caption(
            "Single-plot metric: "
            f"`{_combo_feature_term(selected_feature)}` | {selected_single_engine} | "
            f"{selected_record_count} records"
        )
    else:
        detail_feature_options = feature_table["feature"].astype(str).tolist()
        detail_feature_table = feature_table.copy()
        previous_feature_key = f"{run_key}_refolding_metric_feature_v5"
        previous_feature = st.session_state.get(previous_feature_key)
        feature_index = 0
        if previous_feature in detail_feature_options:
            feature_index = detail_feature_options.index(previous_feature)
        feature_labels = {
            str(row.get("feature") or ""): (
                f"{_combo_feature_term(row.get('feature'))} | "
                f"{row.get('engine') or _feature_engine(row.get('feature'))} | "
                f"{row.get('category') or _feature_category(row.get('feature'))} | "
                f"{int(row.get('valid_records') or 0)} records"
            )
            for row in detail_feature_table.to_dict(orient="records")
        }
        selected_feature = st.selectbox(
            "Metric to rank structures",
            detail_feature_options,
            index=feature_index,
            key=previous_feature_key,
            help=(
                "Choose one of the metrics available after the engine/category/filter controls above. "
                "With an empty filter this list shows all matching metric columns."
            ),
            format_func=lambda feature: feature_labels.get(str(feature), str(feature)),
        )
    default_direction = direction_lookup.get(str(selected_feature), _feature_sort_direction(selected_feature))
    direction = st.segmented_control(
        "Ranking direction",
        ["higher", "lower"],
        default=default_direction if default_direction in {"higher", "lower"} else "higher",
        key=f"{run_key}_refolding_metric_direction_v3",
        help="Use lower for PAE/RMSD/energy-like values; use higher for confidence/ipSAE/LIS/pDockQ-like values.",
    )
    direction = direction or default_direction
    selected_feature_rows = feature_table[feature_table["feature"].astype(str) == str(selected_feature)]
    selected_metric_engine = (
        str(selected_feature_rows.iloc[0].get("engine") or _feature_engine(selected_feature))
        if not selected_feature_rows.empty
        else _feature_engine(selected_feature)
    )
    prefilter_feature_table = all_metric_feature_table[
        all_metric_feature_table["feature"].astype(str).isin(set(metrics_with_combos.columns))
    ].copy()
    selected_engine_prefilter_table = prefilter_feature_table[
        prefilter_feature_table["engine"].astype(str) == str(selected_metric_engine)
    ].copy()
    custom_prefilter_feature_table = (
        selected_engine_prefilter_table.copy() if not selected_engine_prefilter_table.empty else prefilter_feature_table.copy()
    )
    prefilter_feature_table = prefilter_feature_table.sort_values(["engine", "category", "feature"]).reset_index(drop=True)
    custom_prefilter_feature_table = custom_prefilter_feature_table.sort_values(
        ["engine", "category", "feature"]
    ).reset_index(drop=True)
    configured_rules: list[dict[str, Any]] = []
    if not prefilter_feature_table.empty:
        with st.expander("Prefilter structures before ranking", expanded=False):
            st.caption(
                "Optional upstream filters applied to this ranked scatter and design list. "
                f"Defaults prefer `{selected_metric_engine}` metrics, then fall back to input/reference metrics when needed."
            )
            prefilter_options = prefilter_feature_table["feature"].astype(str).tolist()
            prefilter_labels = {
                str(row.get("feature") or ""): (
                    f"{_combo_feature_term(row.get('feature'))} | "
                    f"{row.get('engine') or _feature_engine(row.get('feature'))} | "
                    f"{row.get('category') or _feature_category(row.get('feature'))} | "
                    f"{int(row.get('valid_records') or 0)} records"
                )
                for row in prefilter_feature_table.to_dict(orient="records")
            }
            custom_prefilter_options = custom_prefilter_feature_table["feature"].astype(str).tolist()
            custom_prefilter_labels = {
                str(row.get("feature") or ""): (
                    f"{_combo_feature_term(row.get('feature'))} | "
                    f"{row.get('engine') or _feature_engine(row.get('feature'))} | "
                    f"{row.get('category') or _feature_category(row.get('feature'))} | "
                    f"{int(row.get('valid_records') or 0)} records"
                )
                for row in custom_prefilter_feature_table.to_dict(orient="records")
            }

            def _find_prefilter_option(tokens: list[str], preferred_engine: str | None = None) -> str | None:
                candidates = prefilter_feature_table.copy()
                if preferred_engine:
                    preferred = candidates[candidates["engine"].astype(str) == preferred_engine].copy()
                    if not preferred.empty:
                        candidates = preferred
                for token in tokens:
                    token_lower = token.lower()
                    matches = candidates[
                        candidates["feature"].astype(str).str.lower().str.contains(token_lower, regex=False)
                        | candidates["feature"].map(_raw_feature_name).astype(str).str.contains(token_lower, regex=False)
                    ].copy()
                    if not matches.empty:
                        matches = matches.sort_values(["valid_records", "feature"], ascending=[False, True])
                        return str(matches.iloc[0]["feature"])
                return None

            def _find_prefilter_for_selected_engine(
                tokens: list[str],
                *,
                fallback_engine: str | None = None,
            ) -> str | None:
                selected_engine_feature = _find_prefilter_option(tokens, preferred_engine=selected_metric_engine)
                if selected_engine_feature:
                    return selected_engine_feature
                if fallback_engine and fallback_engine != selected_metric_engine:
                    fallback_feature = _find_prefilter_option(tokens, preferred_engine=fallback_engine)
                    if fallback_feature:
                        return fallback_feature
                return _find_prefilter_option(tokens)

            default_rules: list[dict[str, Any]] = []
            shape_feature = _find_prefilter_for_selected_engine(
                ["input_interface_shape_complementary", "input_interface_sc", "shape_complement"],
                fallback_engine="Input PyRosetta",
            )
            if shape_feature:
                default_rules.append(
                    {
                        "label": "Input shape complementarity",
                        "feature": shape_feature,
                        "operator": ">=",
                        "threshold": 0.62,
                    }
                )
            rmsd_feature = _find_prefilter_for_selected_engine(
                ["rmsd_binder", "target_aligned_binder_rmsd", "binder_rmsd"],
            )
            if rmsd_feature:
                default_rules.append(
                    {
                        "label": "Binder/backbone RMSD",
                        "feature": rmsd_feature,
                        "operator": "<=",
                        "threshold": 3.73,
                    }
                )
            additional_filters = int(
                st.number_input(
                    "Additional custom metric filters",
                    min_value=0,
                    max_value=5,
                    value=0,
                    step=1,
                    key=f"{run_key}_refolding_prefilter_extra_count_v1",
                    help="Add arbitrary metric thresholds beyond the default Overath-style prefilters.",
                )
            )
            for rule_idx, rule in enumerate(default_rules + [{"label": f"Custom filter {i + 1}"} for i in range(additional_filters)]):
                is_custom_rule = "feature" not in rule
                active_prefilter_options = custom_prefilter_options if is_custom_rule else prefilter_options
                active_prefilter_labels = custom_prefilter_labels if is_custom_rule else prefilter_labels
                if not active_prefilter_options:
                    continue
                default_feature = str(rule.get("feature") or active_prefilter_options[0])
                if default_feature not in active_prefilter_options:
                    default_feature = active_prefilter_options[0]
                values = pd.to_numeric(metrics_with_combos[default_feature], errors="coerce").dropna() if default_feature else pd.Series(dtype=float)
                if values.empty:
                    continue
                raw_min = float(values.min())
                raw_max = float(values.max())
                default_threshold = float(rule.get("threshold", values.median()))
                default_threshold = min(max(default_threshold, raw_min), raw_max)
                cols = st.columns([0.9, 3.4, 0.9, 1.3])
                with cols[0]:
                    enabled = st.checkbox(
                        rule["label"],
                        value=bool(rule.get("feature")),
                        key=f"{run_key}_refolding_prefilter_enabled_{rule_idx}_v1",
                    )
                with cols[1]:
                    feature_choice = st.selectbox(
                        "Metric",
                        active_prefilter_options,
                        index=active_prefilter_options.index(default_feature),
                        key=f"{run_key}_refolding_prefilter_feature_{rule_idx}_v1",
                        label_visibility="collapsed",
                        format_func=lambda feature: active_prefilter_labels.get(str(feature), str(feature)),
                    )
                with cols[2]:
                    operator = st.segmented_control(
                        "Operator",
                        [">=", "<="],
                        default=str(rule.get("operator") or ">="),
                        key=f"{run_key}_refolding_prefilter_operator_{rule_idx}_v1",
                        label_visibility="collapsed",
                    ) or str(rule.get("operator") or ">=")
                selected_values = pd.to_numeric(metrics_with_combos[feature_choice], errors="coerce").dropna()
                if selected_values.empty:
                    continue
                selected_min = float(selected_values.min())
                selected_max = float(selected_values.max())
                selected_default = min(max(default_threshold, selected_min), selected_max)
                with cols[3]:
                    threshold = float(
                        st.number_input(
                            "Threshold",
                            min_value=selected_min,
                            max_value=selected_max,
                            value=selected_default,
                            step=max((selected_max - selected_min) / 200.0, 0.001),
                            format="%.6f",
                            key=f"{run_key}_refolding_prefilter_threshold_{rule_idx}_v1",
                            label_visibility="collapsed",
                        )
                    )
                if enabled:
                    configured_rules.append(
                        {
                            "feature": str(feature_choice),
                            "operator": str(operator),
                            "threshold": threshold,
                        }
                    )
            if configured_rules:
                prefilter_mask = pd.Series(True, index=base_ranked.index)
                for rule in configured_rules:
                    values = pd.to_numeric(base_ranked[rule["feature"]], errors="coerce")
                    if rule["operator"] == "<=":
                        prefilter_mask &= values.le(float(rule["threshold"]))
                    else:
                        prefilter_mask &= values.ge(float(rule["threshold"]))
                filtered_ranked = base_ranked[prefilter_mask].copy()
                st.caption(
                    f"Prefilter retained {len(filtered_ranked):,} of {len(base_ranked):,} structures "
                    f"using {len(configured_rules)} active rule(s)."
                )
                if filtered_ranked.empty:
                    st.warning("No structures pass the active prefilters; ranking still shows the unfiltered list.")
                else:
                    base_ranked = filtered_ranked
    threshold_enabled = st.checkbox(
        "Use manual threshold",
        value=False,
        key=f"{run_key}_refolding_metric_manual_threshold_enabled_v1",
        help="Draw a cutoff for the selected metric and optionally restrict the ranked design list to passing structures.",
    )
    threshold_value: float | None = None
    threshold_condition = "<=" if str(direction).lower() == "lower" else ">="
    show_only_passing = False
    selected_raw_values = pd.to_numeric(base_ranked[selected_feature], errors="coerce").dropna()
    if threshold_enabled:
        if selected_raw_values.empty:
            st.info("No numeric values are available for this metric threshold.")
        else:
            raw_min = float(selected_raw_values.min())
            raw_max = float(selected_raw_values.max())
            raw_median = float(selected_raw_values.median())
            span = raw_max - raw_min
            step = max(span / 200.0, 0.001)
            threshold_cols = st.columns([2, 1])
            with threshold_cols[0]:
                threshold_value = float(
                    st.number_input(
                        f"Manual threshold ({selected_feature} {threshold_condition})",
                        min_value=raw_min,
                        max_value=raw_max,
                        value=raw_median,
                        step=step,
                        format="%.6f",
                        key=f"{run_key}_refolding_metric_manual_threshold_value_v1",
                    )
                )
            with threshold_cols[1]:
                show_only_passing = st.checkbox(
                    "Only passing designs",
                    value=False,
                    key=f"{run_key}_refolding_metric_manual_threshold_filter_v1",
                    help="Limit the ranked-design dropdown to structures passing this threshold.",
                )
    ranked = base_ranked[[col for col in ["binder_id", "target_id"] if col in base_ranked.columns]].copy()
    ranked[selected_feature] = pd.to_numeric(base_ranked[selected_feature], errors="coerce")
    ranked = ranked.dropna(subset=[selected_feature]).reset_index(drop=True)
    if ranked.empty:
        st.info("No structures have valid values for this exact metric.")
        return None
    ranked["feature_value"] = ranked[selected_feature]
    ranked["rank_score"] = -ranked[selected_feature] if direction == "lower" else ranked[selected_feature]
    selected_feature_row = feature_table[feature_table["feature"].astype(str) == str(selected_feature)].iloc[0]
    ranked["feature"] = selected_feature
    ranked["engine"] = str(selected_feature_row.get("engine") or _feature_engine(selected_feature))
    ranked["direction"] = direction
    ranked = ranked.sort_values("rank_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    if threshold_value is not None:
        ranked["passes_threshold"] = (
            ranked[selected_feature] <= threshold_value
            if str(direction).lower() == "lower"
            else ranked[selected_feature] >= threshold_value
        )
        passing_count = int(ranked["passes_threshold"].sum())
        st.caption(
            f"Manual threshold: `{selected_feature} {threshold_condition} {threshold_value:.6g}`. "
            f"{passing_count:,} of {len(ranked):,} ranked structures pass."
        )
    else:
        ranked["passes_threshold"] = True
    ranked["threshold_status"] = ranked["passes_threshold"].map(lambda value: "pass" if bool(value) else "fail")
    selector_ranked = ranked[ranked["passes_threshold"]].copy() if show_only_passing else ranked
    if selector_ranked.empty:
        st.warning("No structures pass the current threshold; showing all ranked structures instead.")
        selector_ranked = ranked
    ranked_options = selector_ranked["binder_id"].astype(str).tolist()
    ranked_design_key = f"{run_key}_benchmark_structure_ranked_design_v1"
    viewer_design_key = f"{run_key}_benchmark_structure_design_v1"
    highlighted_design = (
        st.session_state.get(ranked_design_key)
        or st.session_state.get(viewer_design_key)
        or (ranked_options[0] if ranked_options else None)
    )
    if highlighted_design not in ranked_options and ranked_options:
        highlighted_design = ranked_options[0]
    ranked["viewer_selection"] = ranked["binder_id"].astype(str) == str(highlighted_design)
    tooltip = [
        "binder_id:N",
        *(
            ["target_id:N"]
            if "target_id" in ranked.columns
            else []
        ),
        "engine:N",
        "feature:N",
        "direction:N",
        "rank:Q",
        alt.Tooltip("feature_value:Q", title="feature value", format=".4f"),
        alt.Tooltip("rank_score:Q", title="rank score", format=".4f"),
        "threshold_status:N",
    ]
    threshold_rank_score = None
    if threshold_value is not None:
        threshold_rank_score = -threshold_value if str(direction).lower() == "lower" else threshold_value
    y_for_domain = pd.to_numeric(ranked["rank_score"], errors="coerce").dropna().tolist()
    if threshold_rank_score is not None:
        y_for_domain.append(float(threshold_rank_score))
    y_domain = None
    if y_for_domain:
        y_min = min(y_for_domain)
        y_max = max(y_for_domain)
        span = y_max - y_min
        padding = span * 0.08 if span > 0 else max(abs(y_max) * 0.08, 0.05)
        y_domain = [y_min - padding, y_max + padding]
    chart = (
        alt.Chart(ranked)
        .mark_circle(opacity=0.9)
        .encode(
            x=alt.X("rank:Q", title="rank"),
            y=alt.Y(
                "rank_score:Q",
                title="rank score",
                scale=alt.Scale(domain=y_domain, zero=False) if y_domain else alt.Undefined,
            ),
            color=alt.condition(
                "datum.viewer_selection",
                alt.value("#111827"),
                alt.Color("engine:N", scale=_engine_color_scale(ranked["engine"])),
            ),
            size=alt.condition("datum.viewer_selection", alt.value(180), alt.value(75)),
            stroke=alt.condition("datum.viewer_selection", alt.value("#111827"), alt.value(None)),
            strokeWidth=alt.condition("datum.viewer_selection", alt.value(2.5), alt.value(0)),
            tooltip=tooltip,
        )
        .add_params(alt.selection_point(name="design_pick", fields=["binder_id"], empty=False, on="click"))
        .properties(height=280)
        .interactive()
    )
    if threshold_value is not None and threshold_rank_score is not None:
        threshold_rule_df = pd.DataFrame(
            [
                {
                    "threshold_rank_score": threshold_rank_score,
                    "label": f"{selected_feature} {threshold_condition} {threshold_value:.6g}",
                }
            ]
        )
        threshold_rule = (
            alt.Chart(threshold_rule_df)
            .mark_rule(color="#E11D48", strokeDash=[5, 3], strokeWidth=2)
            .encode(
                y=alt.Y("threshold_rank_score:Q"),
                tooltip=[
                    alt.Tooltip("label:N", title="threshold"),
                    alt.Tooltip("threshold_rank_score:Q", title="rank score cutoff", format=".4g"),
                ],
            )
        )
        chart = chart + threshold_rule
    event = st.altair_chart(
        chart,
        width="stretch",
        key=f"{run_key}_refolding_metric_rank_chart_v1",
        on_select="rerun",
        selection_mode="design_pick",
    )
    clicked_design = _selected_design_from_altair_event(event)
    chart_selection_key = f"{run_key}_refolding_metric_rank_chart_last_design_v1"
    if clicked_design in ranked_options and st.session_state.get(chart_selection_key) != str(clicked_design):
        st.session_state[chart_selection_key] = str(clicked_design)
        st.session_state[ranked_design_key] = str(clicked_design)
        st.session_state[viewer_design_key] = str(clicked_design)
        st.rerun()
    passing_table = ranked[ranked["passes_threshold"]].copy()
    filter_config = {
        "selected_feature": str(selected_feature),
        "direction": str(direction),
        "prefilters": [
            {
                "feature": str(rule.get("feature") or ""),
                "operator": str(rule.get("operator") or ""),
                "threshold": _jsonable_value(rule.get("threshold")),
            }
            for rule in configured_rules
        ],
        "manual_threshold": {
            "enabled": bool(threshold_enabled),
            "condition": threshold_condition,
            "value": _jsonable_value(threshold_value),
            "only_passing_designs": bool(show_only_passing),
        },
    }
    if run_dir is not None:
        st.session_state[f"{run_key}_benchmark_structure_ranked_overlay_context_v1"] = {
            "engine": str(ranked["engine"].iloc[0]),
            "feature": str(selected_feature),
            "direction": str(direction),
            "filter_config": filter_config,
            "designs": passing_table["binder_id"].astype(str).tolist(),
            "rank_by_design": {
                str(row["binder_id"]): int(row["rank"])
                for row in passing_table[["binder_id", "rank"]].to_dict(orient="records")
            },
            "value_by_design": {
                str(row["binder_id"]): float(row["feature_value"])
                for row in passing_table[["binder_id", "feature_value"]].to_dict(orient="records")
                if pd.notna(row.get("feature_value"))
            },
        }
    if not passing_table.empty:
        key_columns = [col for col in ["binder_id", "target_id"] if col in passing_table.columns and col in base_ranked.columns]
        prefilter_columns = [
            str(rule.get("feature"))
            for rule in configured_rules
            if str(rule.get("feature") or "") in base_ranked.columns and str(rule.get("feature")) != str(selected_feature)
        ]
        original_rank_table = _source_candidate_rank_table(str(run_dir)) if run_dir is not None else pd.DataFrame()
        table_metric_feature_table = pd.concat(
            [all_metric_feature_table, feature_table],
            ignore_index=True,
            sort=False,
        )
        table_metric_feature_table = table_metric_feature_table[
            table_metric_feature_table["feature"].astype(str).isin(set(base_ranked.columns))
        ].copy()
        table_metric_feature_table = table_metric_feature_table.drop_duplicates(subset=["feature"], keep="first")
        if "_metric_family" not in table_metric_feature_table.columns:
            table_metric_feature_table["_metric_family"] = table_metric_feature_table["feature"].map(_metric_family_key)
        existing_table_features = set(table_metric_feature_table["feature"].dropna().astype(str))
        metadata_table_rows: list[dict[str, Any]] = []
        metadata_excluded_columns = {
            "binder_id",
            "candidate_id",
            "target_id",
            "rank",
            str(selected_feature),
            "rank_score",
            "passes_threshold",
        }
        for column in base_ranked.columns:
            column_name = str(column)
            if column_name in existing_table_features or column_name in metadata_excluded_columns or column_name.startswith("_"):
                continue
            if not (
                column_name.startswith("import_")
                or column_name.startswith("source_")
                or column_name in {"original_binder_id", "binder_sequence", "A_seq", "A_length"}
            ):
                continue
            values = base_ranked[column].dropna()
            if values.empty:
                continue
            metadata_table_rows.append(
                {
                    "feature": column_name,
                    "engine": "Input",
                    "category": "Input / imported columns",
                    "direction": _feature_sort_direction(column_name),
                    "valid_records": int(values.size),
                    "_metric_family": _metric_family_key(column_name),
                }
            )
        if metadata_table_rows:
            table_metric_feature_table = pd.concat(
                [table_metric_feature_table, pd.DataFrame(metadata_table_rows)],
                ignore_index=True,
                sort=False,
            )
        selected_family_for_table = str(selected_family or _metric_family_key(selected_feature))
        same_family_metric_columns = (
            table_metric_feature_table[
                table_metric_feature_table["_metric_family"].astype(str) == selected_family_for_table
            ]["feature"]
            .astype(str)
            .tolist()
        )
        table_metric_options = [
            str(row.get("feature") or "")
            for row in table_metric_feature_table.sort_values(["engine", "category", "feature"]).to_dict(orient="records")
            if str(row.get("feature") or "") and str(row.get("feature") or "") not in set(key_columns)
        ]
        table_metric_options = list(dict.fromkeys(table_metric_options))
        default_table_metric_columns = [
            col
            for col in dict.fromkeys([*prefilter_columns, *same_family_metric_columns])
            if col in table_metric_options and col != str(selected_feature)
        ]
        selected_table_metric_columns = default_table_metric_columns
        if table_metric_options:
            table_metric_labels = {
                str(row.get("feature") or ""): (
                    f"{_combo_feature_term(row.get('feature'))} | "
                    f"{row.get('engine') or _feature_engine(row.get('feature'))} | "
                    f"{row.get('category') or _feature_category(row.get('feature'))} | "
                    f"{int(row.get('valid_records') or 0)} records"
                )
                for row in table_metric_feature_table.to_dict(orient="records")
            }
            with st.expander("Extra metric columns in ranked table", expanded=False):
                st.caption(
                    "These values are appended to the passing-design table and exported into candidate sets. "
                    "By default this includes the same metric family across engines plus active prefilter metrics."
                )
                table_metric_query = st.text_input(
                    "Filter selectable columns",
                    value="",
                    key=f"{run_key}_refolding_ranked_table_extra_metric_filter_v1",
                    placeholder="e.g. import, binding, kim1, pae, triage, rosetta",
                ).strip().lower()
                visible_table_metric_options = table_metric_options
                if table_metric_query:
                    table_metric_terms = [term for term in re.split(r"\s+", table_metric_query) if term]
                    visible_table_metric_options = [
                        option
                        for option in table_metric_options
                        if _metric_query_matches(option, table_metric_terms)
                        or _metric_query_matches(table_metric_labels.get(str(option), str(option)), table_metric_terms)
                    ]
                selected_defaults = [
                    option
                    for option in dict.fromkeys([*default_table_metric_columns, *selected_table_metric_columns])
                    if option in visible_table_metric_options
                ]
                if table_metric_query and not visible_table_metric_options:
                    st.info("No extra columns match that filter.")
                selected_table_metric_columns = st.multiselect(
                    "Metric columns",
                    visible_table_metric_options,
                    default=selected_defaults,
                    key=f"{run_key}_refolding_ranked_table_extra_metrics_v1",
                    format_func=lambda feature: table_metric_labels.get(str(feature), str(feature)),
                )
        extra_columns = [
            col
            for col in dict.fromkeys([*prefilter_columns, *selected_table_metric_columns])
            if col in base_ranked.columns
        ]
        table_columns = [
            col
            for col in ["binder_id", "target_id", "rank", selected_feature, "rank_score", "passes_threshold"]
            if col in passing_table.columns
        ]
        shown_table = passing_table[table_columns].copy()
        if key_columns and extra_columns:
            extras = base_ranked[key_columns + extra_columns].copy()
            extras = extras.drop_duplicates(subset=key_columns, keep="first")
            shown_table = shown_table.merge(extras, on=key_columns, how="left")
        if not original_rank_table.empty and "binder_id" in passing_table.columns:
            shown_table["_candidate_key"] = shown_table["binder_id"].astype(str).str.lower()
            shown_table = shown_table.merge(original_rank_table, on="_candidate_key", how="left")
            shown_table = shown_table.drop(columns=["_candidate_key"], errors="ignore")
        if "original_rank" not in shown_table.columns or shown_table["original_rank"].isna().any():
            if "original_rank" not in shown_table.columns:
                shown_table["original_rank"] = pd.NA
            if "original_rank_source" not in shown_table.columns:
                shown_table["original_rank_source"] = pd.NA
            binding_confidence_column = next(
                (
                    col
                    for col in base_ranked.columns
                    if "binding confidence" in str(col).lower()
                ),
                None,
            )
            if binding_confidence_column is not None and key_columns:
                confidence_rank_extras = base_ranked[key_columns + [binding_confidence_column]].copy()
                confidence_scores = pd.to_numeric(confidence_rank_extras[binding_confidence_column], errors="coerce")
                if confidence_scores.notna().any():
                    confidence_rank_extras["derived_binding_confidence_rank"] = confidence_scores.rank(
                        method="min",
                        ascending=False,
                    )
                    confidence_rank_extras = confidence_rank_extras[
                        [*key_columns, "derived_binding_confidence_rank"]
                    ].copy()
                    confidence_rank_extras = confidence_rank_extras.drop_duplicates(subset=key_columns, keep="first")
                    shown_table = shown_table.merge(confidence_rank_extras, on=key_columns, how="left")
                    current_source = shown_table["original_rank_source"].astype(str) if "original_rank_source" in shown_table.columns else ""
                    weak_rank_mask = current_source.isin({"import_index", "import_table_row"})
                    missing_rank_mask = shown_table["original_rank"].isna() | weak_rank_mask
                    numeric_rank = pd.to_numeric(
                        shown_table.get("derived_binding_confidence_rank"),
                        errors="coerce",
                    )
                    fill_mask = missing_rank_mask & numeric_rank.notna()
                    if fill_mask.any():
                        shown_table.loc[fill_mask, "original_rank"] = numeric_rank.loc[fill_mask]
                        shown_table.loc[fill_mask, "original_rank_source"] = (
                            f"{binding_confidence_column} rank"
                        )
            source_rank_columns = [
                col
                for col in [
                    "derived_binding_confidence_rank",
                    "boltzbio_rank",
                    "source_rank",
                    "source_metric_boltzbio_rank",
                    "import_rank",
                    "import_index",
                    "import_table_row",
                    "bindcraft_final_rank",
                    "bindcraft_Rank",
                    "bindcraft_ranked_pdb_rank",
                    "input_rank",
                    "original_rank",
                    "rank",
                ]
                if col in base_ranked.columns and col not in shown_table.columns
            ]
            if key_columns and source_rank_columns:
                rank_extras = base_ranked[key_columns + source_rank_columns].copy()
                rank_extras = rank_extras.drop_duplicates(subset=key_columns, keep="first")
                shown_table = shown_table.merge(rank_extras, on=key_columns, how="left")
            missing_rank_mask = shown_table["original_rank"].isna()
            for rank_column in source_rank_columns:
                if rank_column not in shown_table.columns:
                    continue
                numeric_rank = pd.to_numeric(shown_table[rank_column], errors="coerce")
                fill_mask = missing_rank_mask & numeric_rank.notna()
                if fill_mask.any():
                    shown_table.loc[fill_mask, "original_rank"] = numeric_rank.loc[fill_mask]
                    shown_table.loc[fill_mask, "original_rank_source"] = rank_column
                    missing_rank_mask = shown_table["original_rank"].isna()
                if not missing_rank_mask.any():
                    break
        shown_table = shown_table.rename(
            columns={
                "binder_id": "design",
                "target_id": "target",
                "rank": "current_rank",
                selected_feature: "selected_metric",
                "rank_score": "rank_score",
                "passes_threshold": "passes_threshold",
            }
        )
        preferred_table_order = [
            "design",
            "sequence",
            "sequence_length",
            "target",
            "current_rank",
            "original_rank",
            "original_rank_source",
            "selected_metric",
            "rank_score",
            "passes_threshold",
        ]
        shown_table = shown_table[
            [col for col in preferred_table_order if col in shown_table.columns]
            + [col for col in shown_table.columns if col not in set(preferred_table_order)]
        ]
        st.markdown("**Passing Ranked Designs**")
        st.dataframe(
            shown_table.head(500),
            hide_index=True,
            width="stretch",
            column_config={
                "current_rank": st.column_config.NumberColumn("current rank", format="%d"),
                "original_rank": st.column_config.NumberColumn("original rank", format="%d"),
                "selected_metric": st.column_config.NumberColumn(_combo_feature_term(selected_feature), format="%.4f"),
                "rank_score": st.column_config.NumberColumn("rank score", format="%.4f"),
            },
        )
        with st.expander("Create Candidate Set From This Filtered Ranking", expanded=False):
            st.caption(
                "Creates a normal candidate set from the filtered ranked table using the selected plot engine's "
                "predicted complexes. This is the handoff point for later sequence redesign/optimization."
            )
            export_cols = st.columns([2.5, 1])
            with export_cols[0]:
                export_name = st.text_input(
                    "Candidate set name",
                    value=f"{selected_feature} filtered {ranked['engine'].iloc[0]}",
                    key=f"{run_key}_ranked_result_export_name_v1",
                )
            with export_cols[1]:
                export_max = int(
                    st.number_input(
                        "Max designs",
                        min_value=0,
                        max_value=int(len(passing_table)),
                        value=min(int(len(passing_table)), 100),
                        step=1,
                        key=f"{run_key}_ranked_result_export_max_v1",
                        help="0 exports the complete filtered table.",
                    )
                )
            st.caption(
                "Exported candidates are marked as `complex_refolding` and keep the predicted complex, target, "
                "sequence, original rank, and all merged result metrics."
            )
            if st.button(
                "Create candidate set",
                type="primary",
                key=f"{run_key}_ranked_result_export_button_v1",
            ):
                try:
                    export_run = _create_candidate_set_from_ranked_results(
                        parent_run_dir=Path(str(run_dir)),
                        records=records,
                        metrics=metrics,
                        ranked_table=passing_table,
                        selected_engine=str(ranked["engine"].iloc[0]),
                        selected_feature=str(selected_feature),
                        direction=str(direction),
                        filter_config=filter_config,
                        export_name=export_name.strip() or f"{selected_feature} filtered",
                        max_candidates=export_max,
                    )
                    st.success(f"Candidate set created: {export_run.name}")
                    st.link_button(
                        "Open candidate set",
                        result_link("candidate-import", export_run.name),
                    )
                except Exception as exc:
                    st.error(str(exc))
    default_design = str(st.session_state.get(ranked_design_key) or highlighted_design)
    if default_design not in ranked_options and ranked_options:
        default_design = ranked_options[0]
    selected_design = st.selectbox(
        "Ranked design",
        ranked_options,
        index=ranked_options.index(default_design),
        key=ranked_design_key,
        format_func=lambda design: (
            f"#{int(ranked.loc[ranked['binder_id'].astype(str) == str(design), 'rank'].iloc[0])} | {design}"
        ),
    )
    valid_count = int(pd.to_numeric(metrics_with_combos[selected_feature], errors="coerce").notna().sum())
    st.caption(
        f"Structure-driving metric: `{selected_feature}` ({direction}; {valid_count} valid records). "
        "This is a sortable refolding metric, not an AP/AUROC benchmark preset."
    )
    return str(selected_design)


def _deduplicate_feature_aliases(ranking: pd.DataFrame) -> pd.DataFrame:
    if "feature" not in ranking.columns:
        return ranking
    work = ranking.copy()
    if "engine" in work.columns:
        work["alias_group"] = work.apply(
            lambda row: _feature_alias_group(row["feature"], fallback=str(row.get("engine") or "Other")),
            axis=1,
        )
    else:
        work["alias_group"] = work["feature"].map(_feature_alias_group)
    work["alias_priority"] = work["feature"].map(_feature_alias_priority)
    rows: list[pd.Series] = []
    for _group_name, group in work.groupby("alias_group", dropna=False):
        ordered = group.sort_values(
            ["alias_priority", "best_average_precision", "best_auroc"],
            ascending=[True, False, False],
        )
        row = ordered.iloc[0].copy()
        aliases = [str(value) for value in group["feature"].tolist()]
        row["alias_count"] = len(aliases)
        row["aliases"] = ", ".join(aliases)
        rows.append(row)
    if not rows:
        return ranking
    deduped = pd.DataFrame(rows)
    return deduped.sort_values(
        ["best_average_precision", "best_auroc"],
        ascending=[False, False],
    ).drop(columns=["alias_priority"], errors="ignore")


def _ranking_label(path: Path) -> str:
    labels = {
        "merged_benchmark_feature_ranking.csv": "Merged benchmark feature ranking",
        "metric_feature_benchmark.csv": "Metric feature benchmark",
        "esmfold2_feature_benchmark.csv": "ESMFold2 feature benchmark",
        "alphafast_af3_feature_benchmark.csv": "AlphaFast AF3 feature benchmark",
        "colabfold_feature_benchmark.csv": "ColabFold feature benchmark",
        "boltz2_initial_guess_feature_benchmark.csv": "Boltz-2 feature benchmark",
        "af2_initial_guess_feature_benchmark.csv": "AF2 initial guess feature benchmark",
        "common_interface_feature_benchmark.csv": "Common interface feature benchmark",
        "esmfold2_common_interface_feature_benchmark.csv": "ESMFold2 common interface feature benchmark",
        "af2_common_interface_feature_benchmark.csv": "AF2 common interface feature benchmark",
        "rf3_common_interface_feature_benchmark.csv": "RF3 common interface feature benchmark",
        "protenix_common_interface_feature_benchmark.csv": "Protenix common interface feature benchmark",
        "boltzgen_fold_common_interface_feature_benchmark.csv": "BoltzGen fold common interface feature benchmark",
        "esmfold2_balanced_candidate_feature_benchmark.csv": "ESMFold2 balanced candidate feature benchmark",
    }
    return labels.get(path.name, path.stem.replace("_", " ").title())


def _benchmark_ranking_files(benchmark_dir: Path) -> list[Path]:
    preferred = [
        "merged_benchmark_feature_ranking.csv",
        "metric_feature_benchmark.csv",
        "esmfold2_balanced_candidate_feature_benchmark.csv",
        "esmfold2_feature_benchmark.csv",
        "alphafast_af3_feature_benchmark.csv",
        "colabfold_feature_benchmark.csv",
        "af2_initial_guess_feature_benchmark.csv",
        "boltz2_initial_guess_feature_benchmark.csv",
        "common_interface_feature_benchmark.csv",
        "esmfold2_common_interface_feature_benchmark.csv",
        "af2_common_interface_feature_benchmark.csv",
        "rf3_common_interface_feature_benchmark.csv",
        "protenix_common_interface_feature_benchmark.csv",
        "boltzgen_fold_common_interface_feature_benchmark.csv",
    ]
    files: list[Path] = []
    seen: set[Path] = set()
    for name in preferred:
        path = benchmark_dir / name
        if path.exists():
            files.append(path)
            seen.add(path)
    for path in sorted(benchmark_dir.glob("*feature*benchmark*.csv")):
        if path not in seen:
            files.append(path)
            seen.add(path)
    return files


def _combined_feature_ranking_from_files(ranking_files: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in ranking_files:
        if path.name == "merged_benchmark_feature_ranking.csv":
            continue
        table = _read_csv(path)
        if table is None or table.empty or "feature" not in table.columns:
            continue
        table = table.copy()
        table["source_table"] = path.name
        table["engine"] = table["feature"].map(
            lambda feature, fallback=_ranking_engine_fallback(path): _feature_engine(feature, fallback=fallback)
        )
        frames.append(table)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _benchmark_summary_files(benchmark_dir: Path) -> list[Path]:
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
        "rf3_common_interface_feature_summary.json",
        "protenix_common_interface_feature_summary.json",
        "boltzgen_fold_common_interface_feature_summary.json",
    ]
    files: list[Path] = []
    seen: set[Path] = set()
    for name in preferred:
        path = benchmark_dir / name
        if path.exists():
            files.append(path)
            seen.add(path)
    for path in sorted(benchmark_dir.glob("*feature_summary.json")):
        if path not in seen:
            files.append(path)
    return files


def _ranking_preview(
    path: Path,
    *,
    rows: int = 100,
    feature_prefix: str | None = None,
    target_metrics: pd.DataFrame | None = None,
) -> bool:
    df = _read_csv(path)
    if df is None or df.empty:
        return False
    if target_metrics is not None:
        df = _recompute_feature_ranking_for_metrics(df, target_metrics)
        if df.empty:
            return False
    if feature_prefix and "feature" in df.columns:
        df = df[df["feature"].astype(str).str.startswith(feature_prefix)].copy()
        if df.empty:
            return False
    preferred = [
        "feature",
        "direction",
        "best_average_precision",
        "best_auroc",
        "average_precision",
        "auroc",
        "positive_mean",
        "negative_mean",
        "labeled_count",
    ]
    keep = [col for col in preferred if col in df.columns]
    st.dataframe((df[keep] if keep else df).head(rows), hide_index=True, width="stretch")
    return True


def _show_feature_rank_graphs(
    metrics: pd.DataFrame,
    ranking_row: pd.Series,
    *,
    chart_key: str,
) -> None:
    feature = str(ranking_row.get("feature") or "")
    label_col = _label_column(metrics)
    if not feature or feature not in metrics.columns or not label_col:
        st.info("This feature cannot be plotted because the merged metrics table lacks compatible labels or values.")
        return
    direction = _effective_feature_direction(ranking_row)
    plot_df = metrics[[label_col, feature]].copy()
    plot_df[feature] = pd.to_numeric(plot_df[feature], errors="coerce")
    plot_df = plot_df.dropna(subset=[feature]).reset_index(drop=True)
    if plot_df.empty:
        st.info("No valid per-record values are available for this feature.")
        return
    plot_df["record"] = plot_df.index + 1
    plot_df["class"] = plot_df[label_col].map(
        lambda value: "binder" if _truthy_benchmark_label(value) == 1 else "nonbinder"
    )
    plot_df["rank_score"] = -plot_df[feature] if direction == "lower" else plot_df[feature]
    ranked_df = plot_df.sort_values("rank_score", ascending=False).reset_index(drop=True)
    ranked_df["rank"] = ranked_df.index + 1
    ranked_df["is_binder"] = (ranked_df["class"] == "binder").astype(int)
    ranked_df["true_positives"] = ranked_df["is_binder"].cumsum()
    ranked_df["precision"] = ranked_df["true_positives"] / ranked_df["rank"]
    total_binders = int(ranked_df["is_binder"].sum())
    total_nonbinders = int(len(ranked_df) - total_binders)
    ranked_df["recall"] = ranked_df["true_positives"] / total_binders if total_binders else 0.0
    ranked_df["false_positives"] = (1 - ranked_df["is_binder"]).cumsum()
    ranked_df["tpr"] = ranked_df["recall"]
    ranked_df["fpr"] = ranked_df["false_positives"] / total_nonbinders if total_nonbinders else 0.0
    threshold_summary = _optimal_threshold_summary(
        plot_df[feature],
        plot_df[label_col],
        direction=str(direction),
    )
    if threshold_summary:
        per_target_threshold = pd.DataFrame()
        target_median_success_rate = None
        if "target_id" in plot_df.columns:
            lower_is_better = str(direction).lower() == "lower"
            threshold_value = float(threshold_summary["threshold"])
            target_summary_df = plot_df.copy()
            target_summary_df["_passes_threshold"] = (
                target_summary_df[selected_feature] <= threshold_value
                if lower_is_better
                else target_summary_df[selected_feature] >= threshold_value
            )
            per_target_rows: list[dict[str, Any]] = []
            for target_id, target_group in target_summary_df.groupby("target_id", dropna=False, sort=False):
                passed = target_group[target_group["_passes_threshold"]]
                target_binders = int((target_group["class"] == "binder").sum())
                selected_count = int(len(passed))
                selected_binders = int((passed["class"] == "binder").sum()) if selected_count else 0
                per_target_rows.append(
                    {
                        "target": str(target_id) if pd.notna(target_id) else "unknown",
                        "records": int(len(target_group)),
                        "binders": target_binders,
                        "selected": selected_count,
                        "true_binders_selected": selected_binders,
                        "false_positives": selected_count - selected_binders,
                        "success_rate_percent": (selected_binders / selected_count * 100.0) if selected_count else None,
                        "recall_percent": (selected_binders / target_binders * 100.0) if target_binders else None,
                    }
                )
            per_target_threshold = pd.DataFrame(per_target_rows)
            if not per_target_threshold.empty and "success_rate_percent" in per_target_threshold.columns:
                target_success_values = pd.to_numeric(
                    per_target_threshold["success_rate_percent"],
                    errors="coerce",
                ).dropna()
                if not target_success_values.empty:
                    target_median_success_rate = float(target_success_values.median())
        threshold_cols = st.columns(7 if target_median_success_rate is not None else 6)
        threshold_cols[0].metric(
            "Optimal threshold",
            f"{threshold_summary['condition']} {threshold_summary['threshold']:.4g}",
            help="Threshold that maximizes F1 on the selected benchmark feature.",
        )
        threshold_cols[1].metric(
            "Success rate",
            f"{threshold_summary['success_rate'] * 100:.1f}%",
            help="Precision at the threshold: true binders among selected designs.",
        )
        metric_offset = 0
        if target_median_success_rate is not None:
            threshold_cols[2].metric(
                "Target median success",
                f"{target_median_success_rate:.1f}%",
                help="Median per-target precision at this threshold, considering only targets with at least one selected design.",
            )
            metric_offset = 1
        threshold_cols[2 + metric_offset].metric(
            "Recall",
            f"{threshold_summary['recall'] * 100:.1f}%",
            help="Fraction of all labeled binders recovered at the F1-optimal threshold: true binders selected divided by all binders in this feature slice.",
        )
        threshold_cols[3 + metric_offset].metric(
            "F1",
            f"{threshold_summary['f1']:.3f}",
            help="Maximum F1 found while scanning possible cutoffs for this feature. F1 balances success rate/precision and recall: 2 * precision * recall / (precision + recall).",
        )
        threshold_cols[4 + metric_offset].metric(
            "Selected",
            f"{int(threshold_summary['selected']):,}",
            help=f"Number of records passing the F1-optimal cutoff; {threshold_summary['selected_fraction'] * 100:.1f}% of records with this feature.",
        )
        threshold_cols[5 + metric_offset].metric(
            "True binders",
            f"{int(threshold_summary['true_binders']):,}",
            help=f"Number of selected records that are labeled binders; {int(threshold_summary['false_positives']):,} nonbinders would also pass.",
        )
        st.caption(
            f"Using `{feature}` with cutoff `{feature} {threshold_summary['condition']} "
            f"{threshold_summary['threshold']:.4g}` selects {int(threshold_summary['selected']):,} designs; "
            f"{threshold_summary['success_rate'] * 100:.1f}% are labeled binders in this benchmark slice."
        )
    roc_df = pd.concat(
        [
            pd.DataFrame([{"rank": 0, "fpr": 0.0, "tpr": 0.0}]),
            ranked_df[["rank", "fpr", "tpr"]],
        ],
        ignore_index=True,
    )
    auroc_value = pd.to_numeric(pd.Series([ranking_row.get("best_auroc")]), errors="coerce").iloc[0]
    cols = st.columns(3)
    with cols[0]:
        st.subheader("Ranked By Feature")
        ranked_chart = (
            alt.Chart(ranked_df)
            .mark_circle(size=75, opacity=0.9)
            .encode(
                x=alt.X("rank:Q", title="rank"),
                y=alt.Y("rank_score:Q", title="rank score"),
                color=alt.Color("class:N", scale=CLASS_COLOR_SCALE),
                tooltip=["rank:Q", "class:N", alt.Tooltip("rank_score:Q", format=".4f")],
            )
            .interactive()
        )
        st.altair_chart(ranked_chart, width="stretch", key=f"{chart_key}_ranked")
    with cols[1]:
        st.subheader("Precision / Recall By Threshold")
        threshold_curve = _threshold_precision_recall_curve(
            plot_df[feature],
            plot_df[label_col],
            direction=str(direction),
        )
        if threshold_curve.empty:
            st.info("No threshold curve is available for this feature.")
        else:
            pr_df = threshold_curve[["threshold", "precision", "recall", "selected"]].melt(
                id_vars=["threshold", "selected"],
                var_name="metric",
                value_name="value",
            )
            pr_chart = (
                alt.Chart(pr_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("threshold:Q", title=f"{_combo_feature_term(feature)} threshold"),
                    y=alt.Y("value:Q", title="value", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("metric:N", scale=LINE_COLOR_SCALE),
                    tooltip=[
                        alt.Tooltip("threshold:Q", title="threshold", format=".4g"),
                        alt.Tooltip("selected:Q", title="selected"),
                        "metric:N",
                        alt.Tooltip("value:Q", format=".3f"),
                    ],
                )
                .interactive()
            )
            if threshold_summary:
                pr_threshold_df = pd.DataFrame([{"threshold": float(threshold_summary["threshold"])}])
                pr_chart = pr_chart + (
                    alt.Chart(pr_threshold_df)
                    .mark_rule(color="#E11D48", strokeDash=[5, 3], strokeWidth=2)
                    .encode(x=alt.X("threshold:Q"))
                )
            st.altair_chart(pr_chart, width="stretch", key=f"{chart_key}_precision_recall_threshold_v1")
    with cols[2]:
        roc_title = "ROC Curve"
        if pd.notna(auroc_value):
            roc_title = f"ROC Curve (AUROC {float(auroc_value):.3f})"
        st.subheader(roc_title)
        diagonal = pd.DataFrame([{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}])
        diagonal_chart = (
            alt.Chart(diagonal)
            .mark_line(strokeDash=[4, 4], color="#9CA3AF")
            .encode(
                x=alt.X("fpr:Q", title="false positive rate", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("tpr:Q", title="true positive rate", scale=alt.Scale(domain=[0, 1])),
            )
        )
        roc_chart = (
            alt.Chart(roc_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("fpr:Q", title="false positive rate", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("tpr:Q", title="true positive rate", scale=alt.Scale(domain=[0, 1])),
                tooltip=[
                    "rank:Q",
                    alt.Tooltip("fpr:Q", title="FPR", format=".3f"),
                    alt.Tooltip("tpr:Q", title="TPR", format=".3f"),
                ],
            )
            .interactive()
        )
        st.altair_chart(diagonal_chart + roc_chart, width="stretch", key=f"{chart_key}_roc")
    st.caption(f"Ranking direction for `{feature}`: {direction}. `rank_score` is higher-is-better after applying that direction.")


def _show_best_engine_feature_rankings(benchmark_dir: Path) -> None:
    ranking_path = benchmark_dir / "merged_benchmark_feature_ranking.csv"
    metrics_path = benchmark_dir / "merged_benchmark_metrics.csv"
    ranking = _read_csv(ranking_path)
    metrics = _read_csv(metrics_path)
    if ranking is None or ranking.empty or metrics is None or metrics.empty or "feature" not in ranking.columns:
        return
    ranking = _with_legacy_af2_ranking_rows(benchmark_dir, ranking_path, ranking)
    ranking = ranking.copy()
    ranking["engine"] = ranking["feature"].map(_feature_engine)
    ranking["category"] = ranking["feature"].map(_feature_category)
    for column in ["best_average_precision", "best_auroc"]:
        if column in ranking.columns:
            ranking[column] = pd.to_numeric(ranking[column], errors="coerce")
    ranking = ranking[
        ranking["feature"].astype(str).isin(set(metrics.columns))
        & ranking["best_average_precision"].notna()
        & ranking["engine"].astype(str).isin([engine for engine in ENGINE_COLOR_DOMAIN if engine not in {"Input", "Other", "Published metrics"}])
        & ranking["feature"].map(_is_informative_feature)
    ].copy()
    if ranking.empty:
        return
    rows: list[pd.Series] = []
    for _engine, group in ranking.groupby("engine", dropna=False):
        rows.append(group.sort_values(["best_average_precision", "best_auroc"], ascending=[False, False]).iloc[0])
    best = pd.DataFrame(rows).sort_values(["best_average_precision", "best_auroc"], ascending=[False, False]).reset_index(drop=True)
    st.subheader("Best AP Feature Per Engine")
    st.caption("For each engine, this selects the single highest-AP feature from the merged benchmark ranking and plots binder/nonbinder separation by rank.")
    show_cols = [
        "engine",
        "category",
        "feature",
        "direction",
        "best_average_precision",
        "best_auroc",
        "positive_mean",
        "negative_mean",
    ]
    st.dataframe(best[[col for col in show_cols if col in best.columns]], hide_index=True, width="stretch")
    preset_cols = st.columns([3, 1])
    with preset_cols[0]:
        preset_name = st.text_input(
            "Preset name",
            value=f"{benchmark_dir.parents[1].name}_best_ap_per_engine",
            key=f"{benchmark_dir.parents[1].name}_save_feature_preset_name",
        )
    with preset_cols[1]:
        if st.button(
            "Save ranking preset",
            key=f"{benchmark_dir.parents[1].name}_save_feature_preset",
            help="Store these best-AP engine features for ranking future refolding/evaluation runs.",
        ):
            preset_rows = [
                {
                    key: (None if pd.isna(value) else value)
                    for key, value in row.items()
                    if key in show_cols
                }
                for row in best.to_dict(orient="records")
            ]
            path = save_feature_preset(
                name=preset_name,
                source_run_dir=benchmark_dir.parents[1],
                rows=preset_rows,
                description="Best average-precision feature per engine from benchmark results.",
            )
            st.success(f"Saved preset: {path.name}")
    feature_choices = [f"{row.engine}: {row.feature}" for row in best.itertuples(index=False)]
    selected_choice = st.selectbox(
        "Rank graph feature",
        feature_choices,
        index=0,
        key=f"{benchmark_dir.parents[1].name}_feature_ranking_best_engine_feature_v1",
    )
    selected_idx = feature_choices.index(selected_choice)
    selected_row = best.iloc[selected_idx]
    _show_feature_rank_graphs(
        metrics,
        selected_row,
        chart_key=f"{benchmark_dir.parents[1].name}_feature_ranking_{selected_row['engine']}_{selected_row['feature']}",
    )


def _preset_rows_from_ranking(rows: pd.DataFrame, *, extra: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    keep_cols = [
        "engine",
        "category",
        "feature",
        "direction",
        "best_average_precision",
        "best_auroc",
        "positive_mean",
        "negative_mean",
        "source_table",
        "target_filter",
        "selection_category",
    ]
    payload_rows: list[dict[str, Any]] = []
    for row in rows.to_dict(orient="records"):
        payload = {
            key: (None if pd.isna(value) else value)
            for key, value in row.items()
            if key in keep_cols
        }
        if extra:
            payload.update(extra)
        payload_rows.append(payload)
    return payload_rows


def _save_preset_controls(
    *,
    run_key: str,
    key_suffix: str,
    benchmark_dir: Path,
    rows: pd.DataFrame,
    default_name: str,
    description: str,
    selection_rule: str,
    extra: dict[str, Any] | None = None,
) -> None:
    if rows.empty:
        return
    cols = st.columns([3, 1])
    with cols[0]:
        preset_name = st.text_input(
            "Preset name",
            value=default_name,
            key=f"{run_key}_{key_suffix}_preset_name",
        )
    with cols[1]:
        if st.button(
            "Save preset",
            key=f"{run_key}_{key_suffix}_save_preset",
            help="Store these features for ranking future refolding/evaluation runs.",
        ):
            path = save_feature_preset(
                name=preset_name,
                source_run_dir=benchmark_dir.parents[1],
                rows=_preset_rows_from_ranking(rows, extra=extra),
                description=description,
                selection_rule=selection_rule,
            )
            st.success(f"Saved preset: {path.name}")


def _show_ranking_preset_builder(
    *,
    run_key: str,
    ranking_key: str,
    benchmark_dir: Path,
    ranking: pd.DataFrame,
    engine_summary: pd.DataFrame,
    selected_category: str,
    selected_targets: list[str] | None,
    selected_ranking_label: str,
) -> None:
    if ranking.empty:
        return
    target_label = "all_targets"
    if selected_targets:
        target_label = "_".join(str(target) for target in selected_targets[:4])
        if len(selected_targets) > 4:
            target_label += f"_plus_{len(selected_targets) - 4}"
    source_extra = {
        "source_table": selected_ranking_label,
        "target_filter": target_label,
        "selection_category": selected_category,
    }
    safe_category = selected_category.lower().replace(" ", "_").replace("/", "_")
    safe_target = target_label.replace(" ", "_").replace("/", "_")
    if True:
        st.caption("Manually choose one feature per engine from the current result filters, then save that selection as a preset.")
        manual_pool = ranking.copy()
        if "feature" not in manual_pool.columns or "engine" not in manual_pool.columns:
            return
        categories = ["All", *sorted(str(value) for value in manual_pool["category"].dropna().astype(str).unique())]
        default_categories = [selected_category] if selected_category in categories and selected_category != "All" else ["All"]
        manual_categories = st.multiselect(
            "Manual categories",
            categories,
            default=default_categories,
            key=f"{run_key}_{ranking_key}_{safe_category}_{safe_target}_manual_categories",
        )
        if manual_categories and "All" not in manual_categories:
            manual_pool = manual_pool[manual_pool["category"].astype(str).isin(set(manual_categories))].copy()
        search_text = st.text_input(
            "Feature search",
            value="",
            key=f"{run_key}_{ranking_key}_{safe_category}_{safe_target}_manual_search",
        ).strip().lower()
        if search_text:
            manual_pool = manual_pool[
                manual_pool["feature"].astype(str).str.lower().str.contains(search_text, regex=False)
            ].copy()
        if manual_pool.empty:
            st.info("No features match the manual filters.")
            return
        manual_pool = manual_pool.sort_values(
            ["engine", "best_average_precision", "best_auroc"],
            ascending=[True, False, False],
        )
        selected_rows: list[pd.Series] = []
        manual_engines = [
            engine
            for engine in ENGINE_COLOR_DOMAIN
            if engine not in {"Input", "Published metrics", "Other"}
            and engine in set(manual_pool["engine"].dropna().astype(str))
        ]
        manual_engines.extend(
            sorted(
                engine
                for engine in set(manual_pool["engine"].dropna().astype(str))
                if engine not in set(manual_engines) and engine not in {"Input", "Published metrics", "Other"}
            )
        )
        for engine in manual_engines:
            engine_rows = manual_pool[manual_pool["engine"].astype(str) == engine].copy()
            if engine_rows.empty:
                continue
            options = ["Skip"]
            option_rows: list[pd.Series | None] = [None]
            for _, row in engine_rows.head(80).iterrows():
                ap = pd.to_numeric(pd.Series([row.get("best_average_precision")]), errors="coerce").iloc[0]
                auroc = pd.to_numeric(pd.Series([row.get("best_auroc")]), errors="coerce").iloc[0]
                ap_text = f"{float(ap):.3f}" if pd.notna(ap) else "n/a"
                auroc_text = f"{float(auroc):.3f}" if pd.notna(auroc) else "n/a"
                options.append(f"{row.get('feature')} | AP {ap_text} | AUROC {auroc_text} | {row.get('category')}")
                option_rows.append(row)
            selected = st.selectbox(
                engine,
                options,
                index=1 if len(options) > 1 else 0,
                key=f"{run_key}_{ranking_key}_{safe_category}_{safe_target}_manual_{engine}",
            )
            selected_row = option_rows[options.index(selected)]
            if selected_row is not None:
                selected_rows.append(selected_row)
        manual_df = pd.DataFrame(selected_rows) if selected_rows else pd.DataFrame()
        if not manual_df.empty:
            st.dataframe(
                manual_df[[col for col in ["engine", "category", "feature", "direction", "best_average_precision", "best_auroc"] if col in manual_df.columns]],
                hide_index=True,
                width="stretch",
            )
            _save_preset_controls(
                run_key=run_key,
                key_suffix=f"{ranking_key}_{safe_category}_{safe_target}_manual",
                benchmark_dir=benchmark_dir,
                rows=manual_df,
                default_name=f"{benchmark_dir.parents[1].name}_{safe_category}_{safe_target}_manual",
                description=(
                    f"Manual per-engine feature preset from {selected_ranking_label}; "
                    f"category={selected_category}; targets={target_label}."
                ),
                selection_rule="manual_feature_per_engine",
                extra=source_extra,
            )


def _best_rankable_features_by_engine(benchmark_dir: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    ranking_path = benchmark_dir / "merged_benchmark_feature_ranking.csv"
    ranking = _read_csv(ranking_path)
    if ranking is None or ranking.empty or "feature" not in ranking.columns:
        return pd.DataFrame()
    ranking = _with_legacy_af2_ranking_rows(benchmark_dir, ranking_path, ranking).copy()
    ranking["engine"] = ranking["feature"].map(_feature_engine)
    ranking["category"] = ranking["feature"].map(_feature_category)
    for column in ["best_average_precision", "best_auroc"]:
        if column in ranking.columns:
            ranking[column] = pd.to_numeric(ranking[column], errors="coerce")
    valid_engines = {engine for engine in ENGINE_COLOR_DOMAIN if engine not in {"Input", "Other", "Published metrics"}}
    ranking = ranking[
        ranking["feature"].astype(str).isin(set(metrics.columns))
        & ranking["best_average_precision"].notna()
        & ranking["engine"].astype(str).isin(valid_engines)
        & ranking["feature"].map(_is_informative_feature)
    ].copy()
    ranking["_valid_records"] = ranking["feature"].map(
        lambda feature: int(pd.to_numeric(metrics[str(feature)], errors="coerce").notna().sum())
        if str(feature) in metrics.columns
        else 0
    )
    rows: list[pd.Series] = []
    for _engine, group in ranking.groupby("engine", dropna=False):
        max_valid = int(group["_valid_records"].max()) if not group.empty else 0
        if max_valid > 0:
            group = group[group["_valid_records"] >= max(1, int(max_valid * 0.8))].copy()
        rows.append(
            group.sort_values(
                ["best_average_precision", "best_auroc", "_valid_records"],
                ascending=[False, False, False],
            ).iloc[0]
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .drop(columns=["_valid_records"], errors="ignore")
        .sort_values(["best_average_precision", "best_auroc"], ascending=[False, False])
        .reset_index(drop=True)
    )


def _preset_rankable_features(preset: dict[str, Any], metrics: pd.DataFrame) -> pd.DataFrame:
    rows = list(preset.get("features") or [])
    if not rows:
        return pd.DataFrame()
    ranking = pd.DataFrame(rows).copy()
    if "feature" not in ranking.columns:
        return pd.DataFrame()
    if "engine" not in ranking.columns:
        ranking["engine"] = ranking["feature"].map(_feature_engine)
    if "category" not in ranking.columns:
        ranking["category"] = ranking["feature"].map(_feature_category)
    if "direction" not in ranking.columns:
        ranking["direction"] = "higher"
    for column in ["best_average_precision", "best_auroc"]:
        if column in ranking.columns:
            ranking[column] = pd.to_numeric(ranking[column], errors="coerce")
        else:
            ranking[column] = None
    valid_engines = {engine for engine in ENGINE_COLOR_DOMAIN if engine not in {"Input", "Other", "Published metrics"}}
    ranking = ranking[
        ranking["feature"].astype(str).isin(set(metrics.columns))
        & ranking["engine"].astype(str).isin(valid_engines)
        & ranking["feature"].map(_is_informative_feature)
    ].copy()
    if ranking.empty:
        return pd.DataFrame()
    ranking["_sort_ap"] = pd.to_numeric(ranking["best_average_precision"], errors="coerce").fillna(-1.0)
    ranking["_sort_auroc"] = pd.to_numeric(ranking["best_auroc"], errors="coerce").fillna(-1.0)
    return (
        ranking.sort_values(["_sort_ap", "_sort_auroc"], ascending=[False, False])
        .drop(columns=["_sort_ap", "_sort_auroc"])
        .drop_duplicates("engine")
        .reset_index(drop=True)
    )


def _selected_design_from_altair_event(event: object) -> str | None:
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    if not isinstance(selection, dict):
        return None
    point_selection = selection.get("design_pick") or next(iter(selection.values()), None)
    if not point_selection:
        return None
    if isinstance(point_selection, dict):
        binder_id = point_selection.get("binder_id")
        if isinstance(binder_id, list) and binder_id:
            return str(binder_id[0])
        if binder_id:
            return str(binder_id)
        for key in ["selection", "points", "vlPoint"]:
            selected_rows = point_selection.get(key)
            if isinstance(selected_rows, list) and selected_rows:
                first = selected_rows[0]
                if isinstance(first, dict) and first.get("binder_id"):
                    return str(first["binder_id"])
            if isinstance(selected_rows, dict) and selected_rows.get("binder_id"):
                return str(selected_rows["binder_id"])
    if isinstance(point_selection, list) and point_selection:
        first = point_selection[0]
        if isinstance(first, dict) and first.get("binder_id"):
            return str(first["binder_id"])
    return None


def _selected_metric_from_altair_event(event: object) -> str | None:
    selection = getattr(event, "selection", None)
    if selection is None and isinstance(event, dict):
        selection = event.get("selection")
    if not isinstance(selection, dict):
        return None
    point_selection = selection.get("metric_pick") or next(iter(selection.values()), None)
    if not point_selection:
        return None
    if isinstance(point_selection, dict):
        feature = point_selection.get("feature")
        if isinstance(feature, list) and feature:
            return str(feature[0])
        if feature:
            return str(feature)
        for key in ["selection", "points", "vlPoint"]:
            selected_rows = point_selection.get(key)
            if isinstance(selected_rows, list) and selected_rows:
                first = selected_rows[0]
                if isinstance(first, dict) and first.get("feature"):
                    return str(first["feature"])
            if isinstance(selected_rows, dict) and selected_rows.get("feature"):
                return str(selected_rows["feature"])
    if isinstance(point_selection, list) and point_selection:
        first = point_selection[0]
        if isinstance(first, dict) and first.get("feature"):
            return str(first["feature"])
    return None


def _ranked_design_selector(
    benchmark_dir: Path,
    records: pd.DataFrame,
    metrics: pd.DataFrame | None,
    *,
    run_key: str,
    allow_current_best: bool = True,
    preset_only: bool = False,
    run_dir: Path | None = None,
) -> str | None:
    if metrics is None or metrics.empty or "binder_id" not in metrics.columns:
        return None
    metrics = _with_saved_combo_cache_metrics(benchmark_dir, metrics)
    st.subheader("Design Ranking")
    if preset_only:
        st.caption(
            "Pick a saved benchmark-derived ranking preset, then click a point or choose a ranked design. "
            "AP/AUROC come from the benchmark that created the preset; this refolding run is ranked only by the selected feature values."
        )
    else:
        st.caption(
            "Pick an engine-specific best-AP metric, then click a point or choose a ranked design. "
            "The structure matrix below updates to that design."
        )
    current_best = _best_rankable_features_by_engine(benchmark_dir, metrics)
    preset_options = load_feature_presets()
    source_options: list[tuple[str, pd.DataFrame]] = []
    if preset_only:
        mode_options = ["Metric filter"]
        if preset_options:
            mode_options.append("Saved preset")
        rank_mode = st.segmented_control(
            "Ranking mode",
            mode_options,
            default="Metric filter",
            key=f"{run_key}_benchmark_structure_rank_mode_v1",
            help=(
                "Metric filter ranks this refolding/evaluation run directly by any available metric column. "
                "Saved preset uses a benchmark-derived feature preset when one matches these columns."
            ),
        )
        if (rank_mode or "Metric filter") == "Metric filter":
            return _label_free_metric_design_selector(records, metrics, run_key=run_key, run_dir=run_dir)
    if allow_current_best and not current_best.empty:
        source_options.append(("Current run best AP features", current_best))
    matching_preset_labels: list[str] = []
    for preset in preset_options:
        preset_best = _preset_rankable_features(preset, metrics)
        if not preset_best.empty:
            name = str(preset.get("name") or Path(str(preset.get("_path") or "preset")).stem)
            label = f"Preset: {name}"
            matching_preset_labels.append(label)
            source_options.append((label, preset_best))
    if not source_options:
        if preset_only:
            st.info("No saved ranking preset matches feature columns in this refolding/evaluation run.")
        return None
    if matching_preset_labels:
        with st.expander(f"Matching saved ranking presets ({len(matching_preset_labels)})", expanded=False):
            for label in matching_preset_labels:
                st.write(f"- {label.removeprefix('Preset: ')}")
    if len(source_options) > 1:
        default_source_index = 0
        previous_source = st.session_state.get(f"{run_key}_benchmark_structure_rank_preset_v1")
        if previous_source not in {label for label, _features in source_options} and matching_preset_labels:
            default_source_index = [label for label, _features in source_options].index(matching_preset_labels[0])
        source_label = st.selectbox(
            "Ranking source",
            [label for label, _features in source_options],
            index=default_source_index,
            key=f"{run_key}_benchmark_structure_rank_preset_v1",
            help=(
                "Choose which saved preset or current best-AP feature set defines the ranking matrix. "
                "The detailed plot below can then focus on one feature from this source."
            ),
        )
        best_features = dict(source_options)[source_label]
    else:
        source_label, best_features = source_options[0]
        st.caption(f"Ranking source: {source_label}")
    ranked_design_key = f"{run_key}_benchmark_structure_ranked_design_v1"
    viewer_design_key = f"{run_key}_benchmark_structure_design_v1"
    show_selection_highlight = st.checkbox(
        "Highlight selected design",
        value=True,
        key=f"{run_key}_benchmark_structure_rank_highlight_v1",
        help="Turn this off for cleaner exported/presentation plots.",
    )
    highlighted_design = st.session_state.get(ranked_design_key) or st.session_state.get(viewer_design_key)
    structure_engine_count_by_design = (
        records.assign(_binder_id=records["binder_id"].astype(str))
        .groupby("_binder_id")["engine"]
        .nunique()
        .to_dict()
    )
    max_structure_engines = max(structure_engine_count_by_design.values()) if structure_engine_count_by_design else 0
    coverage_mode = "Any structure"
    if max_structure_engines > 1:
        coverage_mode = st.segmented_control(
            "Ranked design structure coverage",
            ["Best coverage", "Any structure"],
            default="Best coverage",
            key=f"{run_key}_benchmark_structure_rank_coverage_v1",
            help=(
                "Best coverage restricts ranked designs to those with the most engine structures available, "
                "which is usually what you want for multi-engine overlays. Any structure keeps Protenix-only "
                "or otherwise partial rows in the ranking."
            ),
        ) or "Best coverage"
    allowed_designs = set(records["binder_id"].dropna().astype(str))
    if coverage_mode == "Best coverage" and max_structure_engines > 1:
        before_coverage = len(allowed_designs)
        allowed_designs = {
            design
            for design in allowed_designs
            if int(structure_engine_count_by_design.get(str(design), 0)) == int(max_structure_engines)
        }
        if allowed_designs:
            st.caption(
                f"Ranking limited to {len(allowed_designs)} design(s) with the best available structure coverage "
                f"({max_structure_engines} engines); {before_coverage - len(allowed_designs)} partial-coverage design(s) hidden."
            )
        else:
            allowed_designs = set(records["binder_id"].dropna().astype(str))
    label_col = _label_column(metrics)
    show_benchmark_classes = bool(label_col) and not preset_only
    show_engine_matrix = st.toggle(
        "Show preset engine ranking matrix",
        value=False,
        key=f"{run_key}_benchmark_structure_show_rank_matrix_v1",
        help=(
            "Draws one ranking panel per engine. This is useful for diagnostics, but it can be slow "
            "to redraw when changing the selected design."
        ),
    )
    matrix_rows: list[pd.DataFrame] = []
    if show_engine_matrix:
        for _, feature_row in best_features.iterrows():
            matrix_feature = str(feature_row.get("feature") or "")
            if not matrix_feature or matrix_feature not in metrics.columns:
                continue
            matrix_engine = str(feature_row.get("engine") or _feature_engine(matrix_feature))
            matrix_direction = str(feature_row.get("direction") or "higher")
            matrix_ranked = metrics[metrics["binder_id"].astype(str).isin(allowed_designs)].copy()
            matrix_ranked[matrix_feature] = pd.to_numeric(matrix_ranked[matrix_feature], errors="coerce")
            matrix_ranked = matrix_ranked.dropna(subset=[matrix_feature]).reset_index(drop=True)
            if matrix_ranked.empty:
                continue
            matrix_ranked["rank_score"] = (
                -matrix_ranked[matrix_feature] if matrix_direction == "lower" else matrix_ranked[matrix_feature]
            )
            matrix_ranked = matrix_ranked.sort_values("rank_score", ascending=False).reset_index(drop=True)
            matrix_ranked["rank"] = matrix_ranked.index + 1
            matrix_ranked["engine"] = matrix_engine
            matrix_ranked["feature"] = matrix_feature
            matrix_ranked["direction"] = matrix_direction
            matrix_ranked["feature_value"] = matrix_ranked[matrix_feature]
            if label_col:
                matrix_ranked["class"] = matrix_ranked[label_col].map(
                    lambda value: "binder" if _truthy_benchmark_label(value) == 1 else "nonbinder"
                )
            else:
                matrix_ranked["class"] = "candidate"
            matrix_rows.append(matrix_ranked)
    if matrix_rows:
        matrix_df = pd.concat(matrix_rows, ignore_index=True)
        if not highlighted_design and not matrix_df.empty:
            highlighted_design = str(matrix_df.sort_values(["engine", "rank"]).iloc[0]["binder_id"])
        st.subheader("Preset Engine Ranking Matrix")
        if show_benchmark_classes:
            st.caption(
                "Each panel ranks the same designs by that engine's preset score. Binder/nonbinder colors use the benchmark labels. "
                "The matrix only highlights the selected design; use the single ranking plot below to choose which structure is shown."
            )
        else:
            st.caption(
                "Each panel ranks the same designs by that engine's preset score. "
                "The matrix only highlights the selected design; use the single ranking plot below to choose which structure is shown."
            )
        matrix_engines = [
            engine
            for engine in ENGINE_COLOR_DOMAIN
            if engine in set(matrix_df["engine"].dropna().astype(str))
        ]
        matrix_engines.extend(
            sorted(
                engine
                for engine in set(matrix_df["engine"].dropna().astype(str))
                if engine not in set(matrix_engines)
            )
        )
        for row_start in range(0, len(matrix_engines), 3):
            cols = st.columns(3)
            for col, engine in zip(cols, matrix_engines[row_start : row_start + 3]):
                engine_df = matrix_df[matrix_df["engine"].astype(str) == engine].copy()
                if engine_df.empty:
                    continue
                engine_df["viewer_selection"] = (
                    engine_df["binder_id"].astype(str) == str(highlighted_design)
                    if show_selection_highlight
                    else False
                )
                selected_engine_df = engine_df[engine_df["viewer_selection"]].copy()
                selection_missing = bool(show_selection_highlight and highlighted_design and selected_engine_df.empty)
                feature_name = str(engine_df["feature"].dropna().iloc[0])
                matrix_color = (
                    alt.Color("class:N", scale=BENCHMARK_CLASS_COLOR_SCALE)
                    if show_benchmark_classes
                    else alt.value("#3B82F6")
                )
                matrix_tooltip = [
                    "engine:N",
                    "binder_id:N",
                    "feature:N",
                    "direction:N",
                    "rank:Q",
                    alt.Tooltip("feature_value:Q", title="feature value", format=".4f"),
                    alt.Tooltip("rank_score:Q", title="rank score", format=".4f"),
                ]
                if show_benchmark_classes:
                    matrix_tooltip.insert(2, "class:N")
                base_matrix_chart = (
                    alt.Chart(engine_df)
                    .mark_circle(opacity=0.75)
                    .encode(
                        x=alt.X("rank:Q", title="rank"),
                        y=alt.Y("rank_score:Q", title=feature_name),
                        color=matrix_color,
                        size=alt.value(70),
                        stroke=alt.value("#1F2937"),
                        strokeWidth=alt.value(0.35),
                        tooltip=matrix_tooltip,
                    )
                    .properties(height=320)
                )
                if not selected_engine_df.empty:
                    selected_layer = (
                        alt.Chart(selected_engine_df)
                        .mark_circle(opacity=1.0, size=230, color="#111827", stroke="#FFFFFF", strokeWidth=1.8)
                        .encode(
                            x=alt.X("rank:Q", title="rank"),
                            y=alt.Y("rank_score:Q", title=feature_name),
                            tooltip=matrix_tooltip,
                        )
                    )
                    matrix_chart = (base_matrix_chart + selected_layer).interactive()
                else:
                    matrix_chart = base_matrix_chart.interactive()
                with col:
                    st.markdown(f"**{engine}**")
                    if selection_missing:
                        st.caption(f"Selected design has no value for `{feature_name}`.")
                    st.altair_chart(
                        matrix_chart,
                        width="stretch",
                        key=f"{run_key}_benchmark_structure_rank_matrix_{engine}_v2",
                    )
    choices = [f"{row.engine}: {row.feature}" for row in best_features.itertuples(index=False)]
    selected_choice = st.selectbox(
        "Detailed plot metric",
        choices,
        index=0,
        key=f"{run_key}_benchmark_structure_rank_feature_v1",
        help="Choose the single metric used by the clickable ranking plot and ranked-design dropdown below.",
    )
    selected_row = best_features.iloc[choices.index(selected_choice)]
    feature = str(selected_row["feature"])
    direction = str(selected_row.get("direction") or "higher")
    if feature not in metrics.columns:
        return None
    ranked = metrics[metrics["binder_id"].astype(str).isin(allowed_designs)].copy()
    ranked[feature] = pd.to_numeric(ranked[feature], errors="coerce")
    ranked = ranked.dropna(subset=[feature]).reset_index(drop=True)
    if ranked.empty:
        st.info("No designs with valid values are available for this ranking metric.")
        return None
    if label_col:
        ranked["class"] = ranked[label_col].map(lambda value: "binder" if _truthy_benchmark_label(value) == 1 else "nonbinder")
    else:
        ranked["class"] = "candidate"
    ranked["rank_score"] = -ranked[feature] if direction == "lower" else ranked[feature]
    score_display_column = f"-{feature}" if direction == "lower" else feature
    ranked[score_display_column] = ranked["rank_score"]
    ranked = ranked.sort_values("rank_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = ranked.index + 1
    ranked_options = ranked["binder_id"].astype(str).tolist()
    highlighted_design = (
        st.session_state.get(ranked_design_key)
        or st.session_state.get(viewer_design_key)
        or (ranked_options[0] if ranked_options else None)
    )
    if highlighted_design not in ranked_options and ranked_options:
        highlighted_design = ranked_options[0]
    ranked["viewer_selection"] = (
        ranked["binder_id"].astype(str) == str(highlighted_design)
        if show_selection_highlight
        else False
    )
    hover_columns = [
        "binder_id",
        "target_id",
        "rank",
        "rank_score",
        feature,
    ]
    if not preset_only:
        hover_columns.insert(2, "class")
    hover_columns = [column for column in hover_columns if column in ranked.columns]
    tooltip_fields = []
    for column in hover_columns:
        if column in {"rank_score", feature}:
            tooltip_fields.append(alt.Tooltip(column, title=column, format=".4f"))
        else:
            tooltip_fields.append(alt.Tooltip(column, title=column))
    selector = alt.selection_point(name="design_pick", fields=["binder_id"], empty=False, on="click")
    chart = (
        alt.Chart(ranked)
        .mark_circle(opacity=0.9)
        .encode(
            x=alt.X("rank:Q", title="rank"),
            y=alt.Y(f"{score_display_column}:Q", title=score_display_column),
            color=(
                alt.condition("datum.viewer_selection", alt.value("#111827"), alt.value("#3B82F6"))
                if preset_only
                else alt.Color("class:N", scale=BENCHMARK_CLASS_COLOR_SCALE)
            ),
            size=alt.condition("datum.viewer_selection", alt.value(180), alt.value(75)),
            stroke=alt.condition("datum.viewer_selection", alt.value("#111827"), alt.value(None)),
            strokeWidth=alt.condition("datum.viewer_selection", alt.value(2.5), alt.value(0)),
            tooltip=tooltip_fields,
        )
        .add_params(selector)
        .properties(height=420)
    )
    event = st.altair_chart(
        chart,
        width="stretch",
        key=f"{run_key}_benchmark_structure_rank_chart_v1",
        on_select="rerun",
        selection_mode="design_pick",
    )
    clicked_design = _selected_design_from_altair_event(event)
    chart_selection_key = f"{run_key}_benchmark_structure_rank_chart_last_design_v1"
    if clicked_design in ranked_options and st.session_state.get(chart_selection_key) != str(clicked_design):
        st.session_state[chart_selection_key] = str(clicked_design)
        st.session_state[ranked_design_key] = str(clicked_design)
        st.session_state[viewer_design_key] = str(clicked_design)
        st.rerun()
    default_design = str(st.session_state.get(ranked_design_key) or highlighted_design)
    if default_design not in ranked_options and ranked_options:
        default_design = ranked_options[0]
    if clicked_design in ranked_options:
        st.caption(f"Selected from plot: `{clicked_design}`")
    selected_design = st.selectbox(
        "Ranked design",
        ranked_options,
        index=ranked_options.index(default_design),
        key=ranked_design_key,
        format_func=lambda design: (
            f"#{int(ranked.loc[ranked['binder_id'].astype(str) == str(design), 'rank'].iloc[0])} | {design}"
        ),
    )
    ap = selected_row.get("best_average_precision")
    auroc = selected_row.get("best_auroc")
    st.caption(
        f"Ranking feature: `{feature}` ({direction}; AP {float(ap):.3f}, AUROC {float(auroc):.3f})."
        if pd.notna(ap) and pd.notna(auroc)
        else f"Ranking feature: `{feature}` ({direction})."
    )
    return str(selected_design)


def _show_grouped_feature_rankings(
    benchmark_dir: Path,
    *,
    target_metrics: pd.DataFrame | None = None,
) -> None:
    st.caption(
        "Feature rankings ask how well each numeric feature separates known binders from nonbinders. "
        "The groups below separate where the feature came from: native predictor output, shared PAE/interface "
        "recalculation, Rosetta/PyRosetta structure scoring, or PyMOL geometry analysis."
    )
    _show_best_engine_feature_rankings(benchmark_dir)

    merged = benchmark_dir / "merged_benchmark_feature_ranking.csv"
    if merged.exists():
        with st.expander("Merged benchmark feature ranking", expanded=True):
            st.caption(
                "Overview across all available features after merging engine-native confidence, shared interface scores, "
                "Rosetta metrics, PyMOL metrics, and input metadata."
            )
            _ranking_preview(merged, target_metrics=target_metrics)

    groups: list[tuple[str, str, list[tuple[str, Path, str | None]]]] = [
        (
            "Engine-Native Prediction Features",
            "These features are emitted by the prediction/refolding engine itself or by its native parser. "
            "They are best interpreted within the same engine because similarly named metrics may not be identical across models.",
            [
                ("AlphaFast AF3", benchmark_dir / "alphafast_af3_feature_benchmark.csv", None),
                ("AF2 initial guess", benchmark_dir / "af2_initial_guess_feature_benchmark.csv", None),
                ("Boltz-2", benchmark_dir / "boltz2_initial_guess_feature_benchmark.csv", None),
                ("ColabFold", benchmark_dir / "colabfold_feature_benchmark.csv", None),
                ("ESMFold2", benchmark_dir / "esmfold2_feature_benchmark.csv", None),
                ("RF3", benchmark_dir / "rf3_feature_benchmark.csv", None),
                ("Protenix v0.5", benchmark_dir / "protenix_feature_benchmark.csv", None),
                ("Protenix v1", benchmark_dir / "protenix_v1_feature_benchmark.csv", None),
                ("Protenix v2", benchmark_dir / "protenix_v2_feature_benchmark.csv", None),
                ("BoltzGen Fold", benchmark_dir / "boltzgen_fold_feature_benchmark.csv", None),
            ],
        ),
        (
            "PAE / Interface Recalculation",
            "These features are recalculated after prediction from each engine's predicted structure and PAE-like confidence matrix "
            "using the same interface formulas, including ipSAE, LIS, pDockQ, pDockQ2, and interface PAE.",
            [
                ("AlphaFast AF3", benchmark_dir / "common_interface_feature_benchmark.csv", "af3_"),
                ("Boltz-2", benchmark_dir / "common_interface_feature_benchmark.csv", "boltz2_"),
                ("ColabFold", benchmark_dir / "common_interface_feature_benchmark.csv", "colab_"),
                ("AF2 initial guess", benchmark_dir / "af2_common_interface_feature_benchmark.csv", None),
                ("ESMFold2", benchmark_dir / "esmfold2_common_interface_feature_benchmark.csv", None),
                ("RF3", benchmark_dir / "rf3_common_interface_feature_benchmark.csv", None),
                ("Protenix v0.5", benchmark_dir / "protenix_common_interface_feature_benchmark.csv", None),
                ("Protenix v1", benchmark_dir / "protenix_v1_common_interface_feature_benchmark.csv", None),
                ("Protenix v2", benchmark_dir / "protenix_v2_common_interface_feature_benchmark.csv", None),
                ("BoltzGen Fold", benchmark_dir / "boltzgen_fold_common_interface_feature_benchmark.csv", None),
            ],
        ),
        (
            "Rosetta / PyRosetta Structure Evaluation",
            "These features are recalculated from each predicted 3D complex with the Rosetta/PyRosetta interface and SAP scoring scripts. "
            "They describe interface energy, packing, solvent-accessible surface area, hydrogen bonds, and related structure scores.",
            [
                ("AlphaFast AF3", benchmark_dir / "predicted_rosetta_feature_benchmark.csv", "af3_rosetta_"),
                ("AF2 initial guess", benchmark_dir / "predicted_rosetta_feature_benchmark.csv", "af2_rosetta_"),
                ("Boltz-2", benchmark_dir / "predicted_rosetta_feature_benchmark.csv", "boltz2_rosetta_"),
                ("ColabFold", benchmark_dir / "predicted_rosetta_feature_benchmark.csv", "colab_rosetta_"),
                ("ESMFold2", benchmark_dir / "predicted_rosetta_feature_benchmark.csv", "esmfold2_rosetta_"),
                ("RF3", benchmark_dir / "predicted_rosetta_feature_benchmark.csv", "rf3_rosetta_"),
                ("Protenix v0.5", benchmark_dir / "predicted_rosetta_feature_benchmark.csv", "protenix_rosetta_"),
                ("Protenix v1", benchmark_dir / "predicted_rosetta_feature_benchmark.csv", "protenix_v1_rosetta_"),
                ("Protenix v2", benchmark_dir / "predicted_rosetta_feature_benchmark.csv", "protenix_v2_rosetta_"),
                ("BoltzGen Fold", benchmark_dir / "predicted_rosetta_feature_benchmark.csv", "boltzgen_fold_rosetta_"),
                ("Input structures", benchmark_dir / "predicted_rosetta_feature_benchmark.csv", "input_rosetta_"),
            ],
        ),
        (
            "PyMOL Geometry Evaluation",
            "These features are recalculated from each predicted 3D complex with the PyMOL geometry script. "
            "They describe contacts, geometry, epitope/secondary-structure features, and other shape-derived interface terms.",
            [
                ("AlphaFast AF3", benchmark_dir / "pymol_files" / "pymol_metrics_af3_feature_benchmark.csv", None),
                ("AF2 initial guess", benchmark_dir / "pymol_files" / "pymol_metrics_af2_feature_benchmark.csv", None),
                ("Boltz-2", benchmark_dir / "pymol_files" / "pymol_metrics_boltz2_feature_benchmark.csv", None),
                ("ColabFold", benchmark_dir / "pymol_files" / "pymol_metrics_colab_feature_benchmark.csv", None),
                ("ESMFold2", benchmark_dir / "pymol_files" / "pymol_metrics_esmfold2_feature_benchmark.csv", None),
                ("RF3", benchmark_dir / "pymol_files" / "pymol_metrics_rf3_feature_benchmark.csv", None),
                ("Protenix v0.5", benchmark_dir / "pymol_files" / "pymol_metrics_protenix_feature_benchmark.csv", None),
                ("Protenix v1", benchmark_dir / "pymol_files" / "pymol_metrics_protenix_v1_feature_benchmark.csv", None),
                ("Protenix v2", benchmark_dir / "pymol_files" / "pymol_metrics_protenix_v2_feature_benchmark.csv", None),
                ("BoltzGen Fold", benchmark_dir / "pymol_files" / "pymol_metrics_boltzgen_fold_feature_benchmark.csv", None),
                ("Input structures", benchmark_dir / "pymol_files" / "pymol_metrics_input_feature_benchmark.csv", None),
            ],
        ),
    ]

    for title, description, tables in groups:
        st.subheader(title)
        st.caption(description)
        found = False
        for engine, path, feature_prefix in tables:
            if not path.exists():
                continue
            with st.expander(engine, expanded=not found):
                shown = _ranking_preview(
                    path,
                    feature_prefix=feature_prefix,
                    target_metrics=target_metrics,
                )
                if not shown:
                    st.info("No ranked features are available for this engine in this group.")
                found = found or shown
        if not found:
            st.info("No feature ranking tables are available for this group.")


def _first_benchmark_feature_summary(benchmark_dir: Path) -> dict:
    for path in _benchmark_summary_files(benchmark_dir):
        summary = read_json(path)
        if summary:
            if summary.get("top_feature_average_precision") is None:
                ranking_name = path.name.replace("_feature_summary.json", "_feature_benchmark.csv")
                if path.name == "merged_benchmark_feature_summary.json":
                    ranking_name = "merged_benchmark_feature_ranking.csv"
                ranking_df = _read_csv(benchmark_dir / ranking_name)
                if ranking_df is not None and not ranking_df.empty and "best_average_precision" in ranking_df.columns:
                    best_ap = pd.to_numeric(ranking_df["best_average_precision"], errors="coerce")
                    if best_ap.notna().any():
                        summary["top_feature_average_precision"] = float(best_ap.dropna().iloc[0])
            return summary
    return {}


def _metrics_file_for_ranking(benchmark_dir: Path, ranking_path: Path) -> Path | None:
    if ranking_path.name == "merged_benchmark_feature_ranking.csv":
        path = benchmark_dir / "merged_benchmark_metrics.csv"
        return path if path.exists() else None
    candidates = [
        benchmark_dir / ranking_path.name.replace("_feature_benchmark.csv", "_metrics.csv"),
        benchmark_dir / ranking_path.name.replace("_feature_ranking.csv", "_metrics.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    merged = benchmark_dir / "merged_benchmark_metrics.csv"
    return merged if merged.exists() else None


def _with_legacy_af2_ranking_rows(benchmark_dir: Path, ranking_path: Path, ranking: pd.DataFrame) -> pd.DataFrame:
    if ranking_path.name != "merged_benchmark_feature_ranking.csv" or "feature" not in ranking.columns:
        return ranking
    features = set(ranking["feature"].dropna().astype(str))
    native_af2_features = {
        "af2_iptm",
        "af2_ptm",
        "af2_binder_plddt",
        "af2_binder_pae",
        "af2_i_con_loss",
        "af2_con_loss",
        "af2_target_aligned_binder_rmsd",
        "af2_monomer_refolding_rmsd",
        "af2_time",
    }
    if features.intersection(native_af2_features):
        return ranking
    af2_ranking = _read_csv(benchmark_dir / "af2_initial_guess_feature_benchmark.csv")
    if af2_ranking is None or af2_ranking.empty or "feature" not in af2_ranking.columns:
        return ranking
    af2_ranking = af2_ranking.copy()
    af2_ranking["feature"] = af2_ranking["feature"].astype(str).map(
        lambda feature: feature if feature.startswith("af2_") else f"af2_{feature}"
    )
    return pd.concat([ranking, af2_ranking], ignore_index=True)


def _target_summary(metrics: pd.DataFrame | None) -> dict[str, Any]:
    if metrics is None or metrics.empty or "target_id" not in metrics.columns:
        return {}
    target_series = metrics["target_id"].dropna().astype(str)
    targets = sorted(target_series.unique().tolist())
    label_col = _label_column(metrics)
    summary: dict[str, Any] = {
        "target_count": len(targets),
        "targets": targets,
        "records": int(len(metrics)),
    }
    if label_col:
        labels = [_truthy_benchmark_label(value) for value in metrics[label_col].tolist()]
        summary["positive_count"] = sum(1 for label in labels if label == 1)
        summary["negative_count"] = sum(1 for label in labels if label == 0)
    return summary


def _benchmark_target_metadata_table(run_dir: Path, benchmark_dir: Path) -> pd.DataFrame | None:
    for path in [
        benchmark_dir / "merged_benchmark_metrics.csv",
        run_dir / "artifacts" / "queued_inputs" / "input.csv",
        run_dir / "artifacts" / "raw" / "de_novo_binder_scoring" / "dataset" / "input.csv",
        run_dir / "artifacts" / "raw" / "de_novo_binder_scoring" / "output" / "run.csv",
    ]:
        table = _read_csv(path)
        if table is not None and not table.empty and "target_id" in table.columns:
            return table
    return None


def _target_summary_text(summary: dict[str, Any]) -> str | None:
    targets = summary.get("targets") or []
    if not targets:
        return None
    if len(targets) == 1:
        label = str(targets[0])
    else:
        shown = ", ".join(str(target) for target in targets[:8])
        suffix = f", +{len(targets) - 8} more" if len(targets) > 8 else ""
        label = f"{len(targets)} targets: {shown}{suffix}"
    details = []
    if summary.get("records") is not None:
        details.append(f"{summary['records']} records")
    if summary.get("positive_count") is not None and summary.get("negative_count") is not None:
        details.append(f"{summary['positive_count']} binders / {summary['negative_count']} nonbinders")
    return f"Targets: {label}" + (f" ({'; '.join(details)})" if details else "")


def _recompute_feature_ranking_for_metrics(ranking: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    label_col = _label_column(metrics)
    if not label_col or "feature" not in ranking.columns:
        return pd.DataFrame()
    labels_raw = [_truthy_benchmark_label(value) for value in metrics[label_col].tolist()]
    rows: list[dict[str, Any]] = []
    for _, source_row in ranking.iterrows():
        feature = str(source_row.get("feature") or "")
        if not feature or feature not in metrics.columns:
            continue
        if _is_chain_indexed_confidence_feature(feature):
            continue
        values = pd.to_numeric(metrics[feature], errors="coerce")
        valid_labels: list[int] = []
        valid_scores: list[float] = []
        for label, score in zip(labels_raw, values.tolist()):
            if label in {0, 1} and pd.notna(score):
                valid_labels.append(int(label))
                valid_scores.append(float(score))
        if not valid_scores:
            continue
        positives = [score for label, score in zip(valid_labels, valid_scores) if label == 1]
        negatives = [score for label, score in zip(valid_labels, valid_scores) if label == 0]
        if not positives or not negatives:
            continue
        auroc = _manual_auroc(valid_labels, valid_scores)
        ap = _manual_average_precision(valid_labels, valid_scores)
        inverse_scores = [-score for score in valid_scores]
        inverse_auroc = _manual_auroc(valid_labels, inverse_scores)
        inverse_ap = _manual_average_precision(valid_labels, inverse_scores)
        best_auroc_direction = "higher"
        best_auroc = auroc
        if inverse_auroc is not None and (best_auroc is None or inverse_auroc > best_auroc):
            best_auroc_direction = "lower"
            best_auroc = inverse_auroc
        best_direction = "higher"
        best_ap = ap
        if inverse_ap is not None and (best_ap is None or inverse_ap > best_ap):
            best_direction = "lower"
            best_ap = inverse_ap
        row = source_row.to_dict()
        row.update(
            {
                "direction": best_direction,
                "count": len(valid_scores),
                "positive_count": len(positives),
                "negative_count": len(negatives),
                "auroc": auroc,
                "average_precision": ap,
                "inverse_auroc": inverse_auroc,
                "inverse_average_precision": inverse_ap,
                "best_auroc": best_auroc,
                "best_auroc_direction": best_auroc_direction,
                "best_average_precision": best_ap,
                "positive_mean": float(sum(positives) / len(positives)) if positives else None,
                "negative_mean": float(sum(negatives) / len(negatives)) if negatives else None,
            }
        )
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["best_average_precision", "best_auroc"],
        ascending=[False, False],
    ).reset_index(drop=True)


def _benchmark_threshold_report_ranking(
    ranking: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    ranking_basis: str,
    label_col: str,
) -> pd.DataFrame:
    if ranking.empty or metrics.empty or not label_col:
        return pd.DataFrame()
    scored = _recompute_feature_ranking_for_metrics(ranking, metrics)
    if scored.empty:
        return pd.DataFrame()
    if ranking_basis == "overall":
        return scored
    if "target_id" not in metrics.columns:
        return pd.DataFrame()
    direction_by_feature = {
        str(row.get("feature")): _effective_feature_direction(pd.Series(row))
        for row in scored.to_dict(orient="records")
        if str(row.get("feature") or "") in metrics.columns
    }
    target_features = [
        str(feature)
        for feature in scored["feature"].astype(str).tolist()
        if str(feature) in metrics.columns
    ]
    per_target, aggregate = _per_target_feature_performance(
        metrics,
        features=target_features,
        directions=direction_by_feature,
        label_col=label_col,
    )
    if per_target.empty or aggregate.empty:
        return pd.DataFrame()
    eligible_targets = {
        str(target_id) if pd.notna(target_id) else "unknown"
        for target_id, target_df in metrics.groupby("target_id", dropna=False, sort=False)
        if {
            label
            for label in (_truthy_benchmark_label(value) for value in target_df[label_col].tolist())
            if label in {0, 1}
        }
        == {0, 1}
    }
    if not eligible_targets:
        return pd.DataFrame()
    complete_features = set(
        per_target.groupby("feature")["target_id"].nunique().loc[
            lambda counts: counts == len(eligible_targets)
        ].index.astype(str)
    )
    aggregate = aggregate[aggregate["feature"].astype(str).isin(complete_features)].copy()
    if aggregate.empty:
        return pd.DataFrame()
    ap_col = "median_average_precision" if ranking_basis == "target median" else "mean_average_precision"
    auroc_col = "median_auroc" if ranking_basis == "target median" else "mean_auroc"
    basis_score_columns = [
        column
        for column in dict.fromkeys(
            [
                "feature",
                "direction",
                "proteins",
                "records",
                ap_col,
                auroc_col,
                "mean_average_precision",
                "median_average_precision",
                "sd_average_precision",
                "mean_auroc",
                "median_auroc",
                "sd_auroc",
            ]
        )
        if column in aggregate.columns
    ]
    basis_scores = aggregate[basis_score_columns].rename(
        columns={
            ap_col: "best_average_precision",
            auroc_col: "best_auroc",
            "proteins": "target_count",
            "records": "target_ranked_records",
        }
    )
    merge_drop = [
        "direction",
        "best_average_precision",
        "best_auroc",
        "mean_average_precision",
        "median_average_precision",
        "sd_average_precision",
        "mean_auroc",
        "median_auroc",
        "sd_auroc",
        "target_count",
        "target_ranked_records",
    ]
    merged = (
        scored.loc[:, ~scored.columns.duplicated()]
        .drop(columns=[column for column in merge_drop if column in scored.columns], errors="ignore")
        .merge(basis_scores.loc[:, ~basis_scores.columns.duplicated()], on="feature", how="inner")
    )
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged.sort_values(["best_average_precision", "best_auroc"], ascending=[False, False]).reset_index(drop=True)


def _target_threshold_success_summary(
    metrics: pd.DataFrame,
    *,
    feature: str,
    label_col: str,
    direction: str,
    threshold: float,
) -> dict[str, Any]:
    if metrics.empty or "target_id" not in metrics.columns or feature not in metrics.columns:
        return {}
    lower_is_better = str(direction).lower() == "lower"
    work = metrics[["target_id", label_col, feature]].copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work = work.dropna(subset=[feature]).copy()
    if work.empty:
        return {}
    work["_class"] = work[label_col].map(_truthy_benchmark_label)
    work = work[work["_class"].isin({0, 1})].copy()
    if work.empty:
        return {}
    work["_passes_threshold"] = work[feature] <= threshold if lower_is_better else work[feature] >= threshold
    rows: list[dict[str, Any]] = []
    for target_id, target_group in work.groupby("target_id", dropna=False, sort=False):
        passed = target_group[target_group["_passes_threshold"]]
        target_binders = int((target_group["_class"] == 1).sum())
        selected_count = int(len(passed))
        selected_binders = int((passed["_class"] == 1).sum()) if selected_count else 0
        rows.append(
            {
                "target": str(target_id) if pd.notna(target_id) else "unknown",
                "records": int(len(target_group)),
                "binders": target_binders,
                "selected": selected_count,
                "true_binders_selected": selected_binders,
                "false_positives": selected_count - selected_binders,
                "success_rate_percent": (selected_binders / selected_count * 100.0) if selected_count else None,
                "recall_percent": (selected_binders / target_binders * 100.0) if target_binders else None,
            }
        )
    per_target = pd.DataFrame(rows)
    if per_target.empty:
        return {}
    success_values = pd.to_numeric(per_target["success_rate_percent"], errors="coerce").dropna()
    recall_values = pd.to_numeric(per_target["recall_percent"], errors="coerce").dropna()
    targets_total = int(len(per_target))
    targets_with_selection = int((per_target["selected"] > 0).sum())
    targets_with_true_binder = int((per_target["true_binders_selected"] > 0).sum())
    targets_only_false = int(
        ((per_target["selected"] > 0) & (per_target["true_binders_selected"] == 0)).sum()
    )
    summary: dict[str, Any] = {
        "target_count": targets_total,
        "targets_with_selected_designs": targets_with_selection,
        "targets_with_true_binders_selected": targets_with_true_binder,
        "targets_with_no_selected_designs": targets_total - targets_with_selection,
        "targets_selected_only_false_positives": targets_only_false,
    }
    if not success_values.empty:
        summary.update(
            {
                "target_median_success_rate_percent": float(success_values.median()),
                "target_mean_success_rate_percent": float(success_values.mean()),
                "target_q25_success_rate_percent": float(success_values.quantile(0.25)),
                "target_q75_success_rate_percent": float(success_values.quantile(0.75)),
            }
        )
    if not recall_values.empty:
        summary.update(
            {
                "target_median_recall_percent": float(recall_values.median()),
                "target_mean_recall_percent": float(recall_values.mean()),
            }
        )
    return summary


def _show_category_threshold_report_tool(
    *,
    benchmark_dir: Path,
    run_key: str,
    ranking_key: str,
    ranking: pd.DataFrame,
    metrics: pd.DataFrame | None,
    selected_engines: list[str],
    target_filter_label: str,
) -> None:
    if metrics is None or metrics.empty or ranking.empty:
        return
    label_col = _label_column(metrics)
    if not label_col:
        return
    with st.expander("Category threshold report", expanded=False):
        st.caption(
            "Calculate the top feature(s) per engine and feature category for selected targets, then export the "
            "F1-optimal threshold, success rate, recall, F1, selected count, and binder counts."
        )
        report_categories = [
            "Engine Confidence / PAE",
            "Recalculated PAE / Interface",
            "Interface Energy / Rosetta",
            "Interface Geometry / PyMOL",
            "PAE × Rosetta combinations",
            "PAE × Input Rosetta combinations",
        ]
        selected_categories = st.multiselect(
            "Feature categories",
            report_categories,
            default=[
                category
                for category in [
                    "Engine Confidence / PAE",
                    "Recalculated PAE / Interface",
                    "PAE × Rosetta combinations",
                ]
                if category in report_categories
            ],
            key=f"{run_key}_{ranking_key}_{target_filter_label}_threshold_report_categories_v1",
        )
        report_targets: list[str] = []
        if "target_id" in metrics.columns:
            available_targets = sorted(metrics["target_id"].dropna().astype(str).unique().tolist())
            default_targets = [target for target in available_targets if "rbd" in target.lower()]
            report_targets = st.multiselect(
                "Targets for report",
                available_targets,
                default=default_targets or available_targets,
                key=f"{run_key}_{ranking_key}_{target_filter_label}_threshold_report_targets_v1",
            )
        report_basis = st.segmented_control(
            "Best-feature criterion",
            ["overall", "target median", "target mean"],
            default="target median",
            key=f"{run_key}_{ranking_key}_{target_filter_label}_threshold_report_basis_v1",
        )
        report_basis = str(report_basis or "target median")
        top_features_per_engine = int(
            st.number_input(
                "Top features per engine/category",
                min_value=1,
                max_value=50,
                value=1,
                step=1,
                key=f"{run_key}_{ranking_key}_{target_filter_label}_threshold_report_top_features_v1",
                help="Keep this many ranked parameter/feature winners for every selected engine within every selected feature category.",
            )
        )
        load_combo_caches = st.checkbox(
            "Include saved combination-feature caches",
            value=True,
            key=f"{run_key}_{ranking_key}_{target_filter_label}_threshold_report_combo_cache_v1",
            help="Uses saved PAE × Rosetta and PAE × Input Rosetta tables when they exist for the all-target collection.",
        )
        calculate = st.button(
            "Calculate category threshold report",
            key=f"{run_key}_{ranking_key}_{target_filter_label}_threshold_report_calculate_v1",
        )
        if not calculate:
            return
        if not selected_categories:
            st.warning("Select at least one feature category.")
            return
        progress = st.empty()
        progress.info("Preparing category threshold report...")
        report_ranking = ranking.copy()
        report_metrics = metrics.copy()
        if load_combo_caches:
            progress.info("Loading saved combination-feature caches...")
            for category, suffix in [
                ("PAE × Rosetta combinations", "pae_rosetta"),
                ("PAE × Input Rosetta combinations", "pae_input_rosetta"),
            ]:
                report_ranking, report_metrics, _loaded = _load_combo_cache(
                    benchmark_dir,
                    _combo_cache_kind(ranking_key, "all targets", suffix),
                    report_ranking,
                    report_metrics,
                    category,
                )
        report_ranking = report_ranking.copy()
        report_ranking["engine"] = report_ranking["feature"].map(_feature_engine)
        report_ranking["category"] = report_ranking["feature"].map(_feature_category)
        report_ranking = _drop_chain_indexed_confidence_features(report_ranking)
        progress.info("Filtering target records and candidate features...")
        if report_targets and "target_id" in report_metrics.columns:
            report_metrics = report_metrics[
                report_metrics["target_id"].astype(str).isin([str(target) for target in report_targets])
            ].copy()
        if report_metrics.empty:
            progress.empty()
            st.warning("No records remain after the selected target filter.")
            return
        source_ranking = report_ranking[
            report_ranking["category"].astype(str).isin(selected_categories)
            & report_ranking["engine"].astype(str).isin(selected_engines)
            & report_ranking["feature"].astype(str).isin(set(str(column) for column in report_metrics.columns))
            & report_ranking["feature"].map(_is_informative_feature)
            & ~report_ranking["engine"].astype(str).isin({"Input", "Other", "Published metrics"})
        ].copy()
        source_ranking = source_ranking.loc[:, ~source_ranking.columns.duplicated()]
        if source_ranking.empty:
            progress.empty()
            st.warning("No features match the selected categories, engines, and target records.")
            return
        progress.info(f"Scoring {len(source_ranking):,} features for {len(report_metrics):,} records...")
        scored = _benchmark_threshold_report_ranking(
            source_ranking,
            report_metrics,
            ranking_basis=report_basis,
            label_col=label_col,
        )
        if scored.empty:
            progress.empty()
            st.warning("No selected features could be scored for this target/category selection.")
            return
        progress.info("Selecting top feature(s) per category and engine, then calculating optimal thresholds...")
        scored = scored.sort_values(
            ["category", "engine", "best_average_precision", "best_auroc"],
            ascending=[True, True, False, False],
        )
        scored["_feature_rank_within_engine_category"] = (
            scored.groupby(["category", "engine"], sort=False).cumcount() + 1
        )
        winners = (
            scored.groupby(["category", "engine"], sort=False, as_index=False)
            .head(top_features_per_engine)
            .copy()
        )
        labels = report_metrics[label_col]
        total_labels = labels.map(_truthy_benchmark_label)
        total_binders = int((total_labels == 1).sum())
        total_nonbinders = int((total_labels == 0).sum())
        report_rows: list[dict[str, Any]] = []
        for _, row in winners.iterrows():
            feature = str(row.get("feature") or "")
            if feature not in report_metrics.columns:
                continue
            direction = _effective_feature_direction(row)
            threshold = _optimal_threshold_summary(report_metrics[feature], labels, direction=direction)
            if not threshold:
                continue
            target_success = _target_threshold_success_summary(
                report_metrics,
                feature=feature,
                label_col=label_col,
                direction=direction,
                threshold=float(threshold.get("threshold")),
            )
            report_rows.append(
                {
                    "category": row.get("category"),
                    "engine": row.get("engine"),
                    "feature_rank_within_engine_category": row.get("_feature_rank_within_engine_category"),
                    "feature": feature,
                    "criterion": report_basis,
                    "direction": direction,
                    "best_average_precision": row.get("best_average_precision"),
                    "best_auroc": row.get("best_auroc"),
                    "threshold_condition": threshold.get("condition"),
                    "optimal_threshold": threshold.get("threshold"),
                    "success_rate_percent": 100.0 * float(threshold.get("success_rate") or 0.0),
                    "recall_percent": 100.0 * float(threshold.get("recall") or 0.0),
                    "f1": threshold.get("f1"),
                    "selected": threshold.get("selected"),
                    "true_binders_selected": threshold.get("true_binders"),
                    "false_positives": threshold.get("false_positives"),
                    "total_binders": total_binders,
                    "total_nonbinders": total_nonbinders,
                    "records": int(len(report_metrics)),
                    "targets": ",".join(report_targets) if report_targets else "all",
                    "target_count": report_metrics["target_id"].nunique(dropna=True) if "target_id" in report_metrics.columns else None,
                    **target_success,
                }
            )
        report = pd.DataFrame(report_rows)
        if report.empty:
            progress.empty()
            st.warning("Thresholds could not be calculated for the selected winner features.")
            return
        report = report.sort_values(
            [
                "category",
                "engine",
                "feature_rank_within_engine_category",
                "success_rate_percent",
                "recall_percent",
                "best_average_precision",
            ],
            ascending=[True, True, True, False, False, False],
        ).reset_index(drop=True)
        st.dataframe(
            report,
            hide_index=True,
            use_container_width=True,
            column_config={
                "success_rate_percent": st.column_config.NumberColumn("success rate (%)", format="%.1f"),
                "target_median_success_rate_percent": st.column_config.NumberColumn("target median success (%)", format="%.1f"),
                "target_mean_success_rate_percent": st.column_config.NumberColumn("target mean success (%)", format="%.1f"),
                "target_q25_success_rate_percent": st.column_config.NumberColumn("target success q25 (%)", format="%.1f"),
                "target_q75_success_rate_percent": st.column_config.NumberColumn("target success q75 (%)", format="%.1f"),
                "recall_percent": st.column_config.NumberColumn("recall (%)", format="%.1f"),
                "target_median_recall_percent": st.column_config.NumberColumn("target median recall (%)", format="%.1f"),
                "target_mean_recall_percent": st.column_config.NumberColumn("target mean recall (%)", format="%.1f"),
                "best_average_precision": st.column_config.NumberColumn("AP", format="%.3f"),
                "best_auroc": st.column_config.NumberColumn("AUROC", format="%.3f"),
                "optimal_threshold": st.column_config.NumberColumn("threshold", format="%.5g"),
                "f1": st.column_config.NumberColumn("F1", format="%.3f"),
            },
        )
        progress.success(f"Calculated {len(report):,} category × engine threshold rows.")
        st.download_button(
            "Download threshold report CSV",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name=f"{run_key}_category_threshold_report.csv",
            mime="text/csv",
            key=f"{run_key}_{ranking_key}_{target_filter_label}_threshold_report_download_v1",
        )


def _common_interface_metric_paths(benchmark_dir: Path) -> list[Path]:
    names = [
        "common_interface_metrics.csv",
        "af2_common_interface_metrics.csv",
        "esmfold2_common_interface_metrics.csv",
        "rf3_common_interface_metrics.csv",
        "protenix_common_interface_metrics.csv",
        "boltzgen_fold_common_interface_metrics.csv",
    ]
    return [benchmark_dir / name for name in names if (benchmark_dir / name).exists()]


def _common_interface_ranking_paths(benchmark_dir: Path) -> list[Path]:
    names = [
        "common_interface_feature_benchmark.csv",
        "af2_common_interface_feature_benchmark.csv",
        "esmfold2_common_interface_feature_benchmark.csv",
        "rf3_common_interface_feature_benchmark.csv",
        "protenix_common_interface_feature_benchmark.csv",
        "boltzgen_fold_common_interface_feature_benchmark.csv",
    ]
    return [benchmark_dir / name for name in names if (benchmark_dir / name).exists()]


def _with_common_interface_metrics(benchmark_dir: Path, metrics: pd.DataFrame | None) -> pd.DataFrame | None:
    if metrics is None or metrics.empty or "binder_id" not in metrics.columns:
        return metrics
    enriched = metrics.copy()
    for path in _common_interface_metric_paths(benchmark_dir):
        table = _read_csv(path)
        if table is None or table.empty or "binder_id" not in table.columns:
            continue
        add_cols = [col for col in table.columns if col == "binder_id" or col not in enriched.columns]
        if len(add_cols) <= 1:
            continue
        enriched = enriched.merge(table[add_cols], on="binder_id", how="left")
    return enriched


def _with_common_interface_rankings(benchmark_dir: Path, ranking: pd.DataFrame) -> pd.DataFrame:
    if ranking.empty or "feature" not in ranking.columns:
        return ranking
    existing = set(ranking["feature"].dropna().astype(str))
    supplemental: list[pd.DataFrame] = []
    for path in _common_interface_ranking_paths(benchmark_dir):
        table = _read_csv(path)
        if table is None or table.empty or "feature" not in table.columns:
            continue
        table = table[~table["feature"].astype(str).isin(existing)].copy()
        if table.empty:
            continue
        supplemental.append(table)
        existing.update(table["feature"].dropna().astype(str))
    if not supplemental:
        return ranking
    return pd.concat([ranking, *supplemental], ignore_index=True)


def _per_target_feature_performance(
    metrics: pd.DataFrame,
    *,
    features: list[str],
    directions: dict[str, str],
    label_col: str,
    target_col: str = "target_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if target_col not in metrics.columns:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for target_id, target_df in metrics.groupby(target_col, dropna=False, sort=False):
        labels_raw = [_truthy_benchmark_label(value) for value in target_df[label_col].tolist()]
        for feature in features:
            if feature not in target_df.columns:
                continue
            values = pd.to_numeric(target_df[feature], errors="coerce")
            valid_labels: list[int] = []
            valid_scores: list[float] = []
            for label, score in zip(labels_raw, values.tolist()):
                if label in {0, 1} and pd.notna(score):
                    valid_labels.append(int(label))
                    valid_scores.append(float(score))
            positives = sum(1 for label in valid_labels if label == 1)
            negatives = sum(1 for label in valid_labels if label == 0)
            if not valid_scores or not positives or not negatives:
                continue
            direction = directions.get(feature, "higher")
            rank_scores = [-score for score in valid_scores] if direction == "lower" else valid_scores
            rows.append(
                {
                    "target_id": str(target_id) if pd.notna(target_id) else "unknown",
                    "feature": feature,
                    "direction": direction,
                    "records": len(valid_scores),
                    "positive_count": positives,
                    "negative_count": negatives,
                    "average_precision": _manual_average_precision(valid_labels, rank_scores),
                    "auroc": _manual_auroc(valid_labels, rank_scores),
                }
            )
    per_target = pd.DataFrame(rows)
    if per_target.empty:
        return per_target, pd.DataFrame()
    per_target["engine"] = per_target["feature"].map(_feature_engine)
    aggregate = (
        per_target.groupby(["feature", "direction"], as_index=False)
        .agg(
            proteins=("target_id", "nunique"),
            records=("records", "sum"),
            mean_average_precision=("average_precision", "mean"),
            median_average_precision=("average_precision", "median"),
            q25_average_precision=("average_precision", lambda values: values.quantile(0.25)),
            q75_average_precision=("average_precision", lambda values: values.quantile(0.75)),
            sd_average_precision=("average_precision", "std"),
            mean_auroc=("auroc", "mean"),
            median_auroc=("auroc", "median"),
            q25_auroc=("auroc", lambda values: values.quantile(0.25)),
            q75_auroc=("auroc", lambda values: values.quantile(0.75)),
            sd_auroc=("auroc", "std"),
        )
        .fillna({"sd_average_precision": 0.0, "sd_auroc": 0.0})
    )
    aggregate["engine"] = aggregate["feature"].map(_feature_engine)
    aggregate["ap_low"] = (aggregate["mean_average_precision"] - aggregate["sd_average_precision"]).clip(lower=0, upper=1)
    aggregate["ap_high"] = (aggregate["mean_average_precision"] + aggregate["sd_average_precision"]).clip(lower=0, upper=1)
    aggregate["auroc_low"] = (aggregate["mean_auroc"] - aggregate["sd_auroc"]).clip(lower=0, upper=1)
    aggregate["auroc_high"] = (aggregate["mean_auroc"] + aggregate["sd_auroc"]).clip(lower=0, upper=1)
    return per_target, aggregate.sort_values(["mean_average_precision", "mean_auroc"], ascending=[False, False])


def _show_benchmark_plots(
    benchmark_dir: Path,
    *,
    target_metrics: pd.DataFrame | None = None,
    selected_targets: list[str] | None = None,
) -> None:
    run_key = benchmark_dir.parents[1].name if len(benchmark_dir.parents) > 1 else benchmark_dir.name
    ranking_files = _benchmark_ranking_files(benchmark_dir)
    if not ranking_files:
        st.info("No feature ranking CSV is available for plots.")
        return
    has_merged_ranking = any(path.name == "merged_benchmark_feature_ranking.csv" for path in ranking_files)
    combined_ranking = _combined_feature_ranking_from_files(ranking_files) if not has_merged_ranking and len(ranking_files) > 1 else pd.DataFrame()
    ranking_options: list[object] = []
    if not combined_ranking.empty:
        ranking_options.append("available_engine_feature_benchmarks")
    ranking_options.extend(ranking_files)
    selected_ranking_option: object = ranking_options[0]
    if len(ranking_options) > 1:
        selected_ranking_option = st.selectbox(
            "Ranking table",
            ranking_options,
            index=0,
            format_func=lambda option: (
                "Available engine feature benchmarks"
                if option == "available_engine_feature_benchmarks"
                else _ranking_label(Path(option))
            ),
            key=f"{run_key}_benchmark_ranking_table",
        )
    selected_ranking_path = selected_ranking_option if isinstance(selected_ranking_option, Path) else None
    ranking_key = selected_ranking_path.stem if selected_ranking_path is not None else str(selected_ranking_option)
    selected_ranking_label = selected_ranking_path.name if selected_ranking_path is not None else "available engine feature benchmarks"
    ranking = combined_ranking.copy() if selected_ranking_path is None else _read_csv(selected_ranking_path)
    metrics_path = _metrics_file_for_ranking(benchmark_dir, selected_ranking_path) if selected_ranking_path is not None else None
    metrics = _read_csv(metrics_path) if metrics_path else None
    if ranking is None or ranking.empty:
        st.info("No merged feature ranking is available for plots.")
        return
    if selected_ranking_path is not None:
        ranking = _with_legacy_af2_ranking_rows(benchmark_dir, selected_ranking_path, ranking)
    if selected_ranking_path is not None and selected_ranking_path.name == "merged_benchmark_feature_ranking.csv":
        ranking = _with_common_interface_rankings(benchmark_dir, ranking)
        metrics = _with_common_interface_metrics(benchmark_dir, metrics)
    merged_metrics = _read_csv(benchmark_dir / "merged_benchmark_metrics.csv")
    merged_metrics = _with_common_interface_metrics(benchmark_dir, merged_metrics)
    target_filter_label = "all targets"
    if target_metrics is not None:
        selected_target_metrics = target_metrics.copy()
        if selected_targets:
            target_filter_label = "_".join(str(target) for target in selected_targets[:4])
            if len(selected_targets) > 4:
                target_filter_label += f"_plus_{len(selected_targets) - 4}"
        if (
            merged_metrics is not None
            and not merged_metrics.empty
            and len(selected_target_metrics) != len(merged_metrics)
        ):
            recomputed = _recompute_feature_ranking_for_metrics(ranking, selected_target_metrics)
            if recomputed.empty:
                st.info("Target-filtered feature ranking requires at least one binder and one nonbinder with valid scores.")
                return
            ranking = recomputed
            metrics = selected_target_metrics
    elif merged_metrics is not None and not merged_metrics.empty and "target_id" in merged_metrics.columns:
        target_summary = _target_summary(merged_metrics)
        available_targets = target_summary.get("targets") or []
        if available_targets:
            selected_targets = st.multiselect(
                "Targets",
                available_targets,
                default=available_targets,
                key=f"{run_key}_{selected_ranking_path.stem}_target_filter_v1",
                help="Filter plots and recompute AP/AUROC for the selected target rows.",
            )
            if not selected_targets:
                st.info("Select at least one target to show benchmark plots.")
                return
            target_metrics = merged_metrics[
                merged_metrics["target_id"].astype(str).isin([str(target) for target in selected_targets])
            ].copy()
            selected_summary = _target_summary(target_metrics)
            target_text = _target_summary_text(selected_summary)
            if target_text:
                st.caption(target_text)
            if len(selected_targets) != len(available_targets):
                recomputed = _recompute_feature_ranking_for_metrics(ranking, target_metrics)
                if recomputed.empty:
                    st.info("Target-filtered feature ranking requires at least one binder and one nonbinder with valid scores.")
                    return
                ranking = recomputed
                metrics = target_metrics
                target_filter_label = "_".join(str(target) for target in selected_targets[:4])
                if len(selected_targets) > 4:
                    target_filter_label += f"_plus_{len(selected_targets) - 4}"

    ranking = ranking.copy()
    fallback_engine = _ranking_engine_fallback(selected_ranking_path) if selected_ranking_path is not None else "Other"
    if "engine" in ranking.columns:
        ranking["engine"] = ranking.apply(
            lambda row: _feature_engine(row["feature"], fallback=str(row.get("engine") or fallback_engine)),
            axis=1,
        )
    else:
        ranking["engine"] = ranking["feature"].map(lambda feature: _feature_engine(feature, fallback=fallback_engine))
    for column in ["best_average_precision", "best_auroc", "average_precision", "auroc"]:
        if column in ranking.columns:
            ranking[column] = pd.to_numeric(ranking[column], errors="coerce")
    ranking["alias_count"] = 1
    ranking["aliases"] = ranking["feature"].astype(str)
    ranking["category"] = ranking["feature"].map(_feature_category)
    ranking = _drop_chain_indexed_confidence_features(ranking)

    show_aliases = st.checkbox("Show repeated alias metrics", value=False, key=f"{run_key}_{ranking_key}_show_aliases_v2")
    show_setup_features = st.checkbox("Show setup/status features", value=False, key=f"{run_key}_{ranking_key}_show_setup_v2")
    plot_ranking = ranking if show_aliases else _deduplicate_feature_aliases(ranking)
    plot_ranking = plot_ranking.copy()
    plot_ranking["display_feature"] = plot_ranking["feature"].map(_display_feature_name)
    if "category" not in plot_ranking.columns:
        plot_ranking["category"] = plot_ranking["feature"].map(_feature_category)
    if not show_aliases and len(plot_ranking) < len(ranking):
        st.caption(f"Collapsed {len(ranking) - len(plot_ranking)} repeated alias metrics in the plots.")
    if not show_setup_features:
        before_filter = len(plot_ranking)
        plot_ranking = plot_ranking[plot_ranking["feature"].map(_is_informative_feature)].copy()
        hidden_count = before_filter - len(plot_ranking)
        if hidden_count:
            st.caption(f"Hidden {hidden_count} setup/status features in the plots.")
    show_context_features = st.checkbox(
        "Show input/reference and uncategorized features",
        value=False,
        key=f"{run_key}_{ranking_key}_show_context_features_v1",
        help=(
            "Input/reference features come from the starting structure or metadata, not from a prediction engine. "
            "Uncategorized features have no reliable engine provenance in the merged table."
        ),
    )
    if not show_context_features:
        before_context_filter = len(plot_ranking)
        plot_ranking = plot_ranking[~plot_ranking["engine"].astype(str).isin({"Input", "Other", "Published metrics"})].copy()
        hidden_context_count = before_context_filter - len(plot_ranking)
        if hidden_context_count:
            st.caption(f"Hidden {hidden_context_count} input/reference or uncategorized features in the plots.")
    engine_order = [
        "AF3",
        "AF2-IG",
        "Boltz-2",
        "BoltzGen Fold",
        "ColabFold",
        "ESMFold2",
        "Protenix v0.5",
        "Protenix v1",
        "Protenix v2",
        "RF3",
        "Input PyRosetta",
    ]
    if show_context_features:
        engine_order.extend(["Input", "Published metrics", "Other"])
    available_engines = [
        engine
        for engine in engine_order
        if engine in set(plot_ranking["engine"].dropna().astype(str))
    ]
    available_engines.extend(
        sorted(
            engine
            for engine in set(plot_ranking["engine"].dropna().astype(str))
            if engine not in set(available_engines)
        )
    )
    selected_engines = st.multiselect(
        "Engines",
        available_engines,
        default=available_engines,
        key=f"{run_key}_{ranking_key}_engine_filter_v3",
        help="Filter all benchmark plots and feature choices to one or more prediction/evaluation engines.",
    )
    if selected_engines:
        plot_ranking = plot_ranking[plot_ranking["engine"].astype(str).isin(selected_engines)].copy()
    else:
        st.info("Select at least one engine to show benchmark plots.")
        return
    category_counts = plot_ranking["category"].value_counts().to_dict() if "category" in plot_ranking.columns else {}
    if category_counts:
        counts_text = ", ".join(f"{name}: {count}" for name, count in sorted(category_counts.items()))
        overath_count = int(plot_ranking["feature"].map(_is_overath_key_feature).sum())
        st.caption(f"Categories in selected ranking table: {counts_text}; Overath key metrics: {overath_count}")
    category_options = [
        "All",
        "Input PyRosetta",
        "Overath key metrics",
        "Engine Confidence / PAE",
        "Recalculated PAE / Interface",
        "Interface Energy / Rosetta",
        "Interface Geometry / PyMOL",
        "PAE × Rosetta combinations",
        "PAE × Input Rosetta combinations",
        MANUAL_FEATURE_COMBO_CATEGORY,
    ]
    default_category = "Overath key metrics"
    if "Input PyRosetta" in set(plot_ranking["category"].dropna().astype(str)):
        default_category = "Input PyRosetta"
    selected_category = st.segmented_control(
        "Feature category",
        category_options,
        selection_mode="single",
        default=default_category,
        key=f"{run_key}_{ranking_key}_feature_category_v4",
    )
    selected_category = selected_category or default_category
    if selected_category == "Overath key metrics":
        st.caption(
            "Paper-prioritized view: interface confidence scores such as ipSAE/LIS/interface PAE/ipTM, "
            "plus orthogonal descriptors highlighted by Overath et al., including interface dG/dSASA, "
            "shape complementarity, SAP, and binder/input RMSD filters."
        )
    elif selected_category == "Input PyRosetta":
        st.caption(
            "Input-complex PyRosetta scores from the original benchmark structures before prediction/refolding. "
            "Use these as a reference baseline for binder/nonbinder separability."
        )
    elif selected_category == "Engine Confidence / PAE":
        st.caption(
            "Native prediction outputs reported by each engine or its parser, such as pTM, ipTM, pLDDT, "
            "ranking/confidence scores, and engine-level PAE summaries."
        )
    elif selected_category == "Recalculated PAE / Interface":
        st.caption(
            "Post-processing scores recalculated from predicted PAE/contact information, such as ipSAE, LIS, "
            "pDockQ/pDockQ2, and directional interface PAE."
        )
    elif selected_category == "PAE × Rosetta combinations":
        st.caption(
            "Raw products computed within each engine: a recalculated PAE/interface score is multiplied by a "
            "Rosetta/interface-energy score. These values stay in their physical/raw scale, so thresholds learned "
            "from benchmarks can be reused on refolding or design candidates. Example: ipSAE_min × interface_ΔG/ΔSASA."
        )
        combo_category = "PAE × Rosetta combinations"
        combo_kind = _combo_cache_kind(ranking_key, target_filter_label, "pae_rosetta")
        combo_loaded = False
        rebuild_combos = st.button(
            "Rebuild PAE × Rosetta combinations",
            key=f"{run_key}_{ranking_key}_rebuild_pae_rosetta_combos_v1",
            help="Rebuild raw PAE/interface × Rosetta products from the current benchmark metrics and overwrite the saved derived table.",
        )
        if rebuild_combos:
            _clear_pae_rosetta_combo_caches()
            with st.spinner("Building PAE × Rosetta combination metrics..."):
                ranking, metrics = _cached_pae_rosetta_combo_rankings(ranking, metrics, 15)
            _save_combo_cache(benchmark_dir, combo_kind, ranking, metrics, combo_category)
            st.success("Saved rebuilt PAE × Rosetta combination metrics for this benchmark run.")
            combo_loaded = True
        else:
            ranking, metrics, combo_loaded = _load_combo_cache(
                benchmark_dir,
                combo_kind,
                ranking,
                metrics,
                combo_category,
            )
        if combo_loaded:
            ranking = ranking.copy()
            ranking["engine"] = ranking["feature"].map(lambda feature: _feature_engine(feature, fallback=fallback_engine))
            for column in ["best_average_precision", "best_auroc", "average_precision", "auroc"]:
                if column in ranking.columns:
                    ranking[column] = pd.to_numeric(ranking[column], errors="coerce")
            ranking["alias_count"] = 1
            ranking["aliases"] = ranking["feature"].astype(str)
            ranking["category"] = ranking["feature"].map(_feature_category)
            ranking = _drop_chain_indexed_confidence_features(ranking)
            plot_ranking = ranking if show_aliases else _deduplicate_feature_aliases(ranking)
            if "category" not in plot_ranking.columns:
                plot_ranking["category"] = plot_ranking["feature"].map(_feature_category)
            if not show_setup_features:
                plot_ranking = plot_ranking[plot_ranking["feature"].map(_is_informative_feature)].copy()
            if not show_context_features:
                plot_ranking = plot_ranking[~plot_ranking["engine"].astype(str).isin({"Input", "Other", "Published metrics"})].copy()
            if selected_engines:
                plot_ranking = plot_ranking[plot_ranking["engine"].astype(str).isin(selected_engines)].copy()
        elif not (plot_ranking["category"].astype(str) == combo_category).any():
            st.info("No PAE × Rosetta combination metrics are available yet. Click rebuild to compute them.")
    elif selected_category == "PAE × Input Rosetta combinations":
        st.caption(
            "Raw products computed across each engine and the shared input/reference complex: a recalculated "
            "PAE/interface score, such as ipSAE or LIS, is multiplied by an input PyRosetta score. "
            "Input PyRosetta is shared across engines for the same benchmark record."
        )
        combo_category = "PAE × Input Rosetta combinations"
        combo_kind = _combo_cache_kind(ranking_key, target_filter_label, "pae_input_rosetta")
        combo_loaded = False
        rebuild_combos = st.button(
            "Rebuild PAE × Input Rosetta combinations",
            key=f"{run_key}_{ranking_key}_rebuild_pae_input_rosetta_combos_v1",
            help="Rebuild engine PAE/interface × shared input PyRosetta products and overwrite the saved derived table.",
        )
        if rebuild_combos:
            _clear_pae_rosetta_combo_caches()
            with st.spinner("Building PAE × Input Rosetta combination metrics..."):
                ranking, metrics = _cached_pae_input_rosetta_combo_rankings(ranking, metrics, 15)
            _save_combo_cache(benchmark_dir, combo_kind, ranking, metrics, combo_category)
            st.success("Saved rebuilt PAE × Input Rosetta combination metrics for this benchmark run.")
            combo_loaded = True
        else:
            ranking, metrics, combo_loaded = _load_combo_cache(
                benchmark_dir,
                combo_kind,
                ranking,
                metrics,
                combo_category,
            )
        if combo_loaded:
            ranking = ranking.copy()
            ranking["engine"] = ranking["feature"].map(lambda feature: _feature_engine(feature, fallback=fallback_engine))
            for column in ["best_average_precision", "best_auroc", "average_precision", "auroc"]:
                if column in ranking.columns:
                    ranking[column] = pd.to_numeric(ranking[column], errors="coerce")
            ranking["alias_count"] = 1
            ranking["aliases"] = ranking["feature"].astype(str)
            ranking["category"] = ranking["feature"].map(_feature_category)
            ranking = _drop_chain_indexed_confidence_features(ranking)
            plot_ranking = ranking if show_aliases else _deduplicate_feature_aliases(ranking)
            if "category" not in plot_ranking.columns:
                plot_ranking["category"] = plot_ranking["feature"].map(_feature_category)
            if not show_setup_features:
                plot_ranking = plot_ranking[plot_ranking["feature"].map(_is_informative_feature)].copy()
            if not show_context_features:
                plot_ranking = plot_ranking[~plot_ranking["engine"].astype(str).isin({"Input", "Other", "Published metrics"})].copy()
            if selected_engines:
                plot_ranking = plot_ranking[plot_ranking["engine"].astype(str).isin(selected_engines)].copy()
        elif not (plot_ranking["category"].astype(str) == combo_category).any():
            st.info("No PAE × Input Rosetta combination metrics are available yet. Click rebuild to compute them.")
    manual_combo_view = selected_category == MANUAL_FEATURE_COMBO_CATEGORY
    if manual_combo_view:
        manual_ranking, metrics = _show_manual_feature_composer(
            run_key=run_key,
            ranking_key=ranking_key,
            ranking=ranking,
            metrics=metrics,
            selected_engines=selected_engines,
            target_filter_label=target_filter_label,
        )
        if manual_ranking.empty:
            return
        ranking = pd.concat([ranking, manual_ranking], ignore_index=True, sort=False)
        plot_ranking = manual_ranking.copy()
    elif selected_category == "Overath key metrics":
        plot_ranking = plot_ranking[plot_ranking["feature"].map(_is_overath_key_feature)].copy()
    elif selected_category != "All":
        plot_ranking = plot_ranking[plot_ranking["category"] == selected_category].copy()
    if plot_ranking.empty:
        all_category_rows = ranking[ranking["category"] == selected_category].copy() if selected_category != "All" else ranking.copy()
        missing_engines = sorted(
            engine
            for engine in all_category_rows.get("engine", pd.Series(dtype=str)).dropna().astype(str).unique()
            if engine not in set(selected_engines)
        )
        if missing_engines:
            st.info(
                "No features are available for this category after the current engine filter. "
                f"Add engine(s): {', '.join(missing_engines)}."
            )
        else:
            st.info("No features are available for this category after the current filters.")
        return
    category_feature_query = st.text_input(
        "Filter category features",
        value="",
        key=f"{run_key}_{ranking_key}_{safe_category if 'safe_category' in locals() else selected_category}_feature_filter_v1",
        placeholder="e.g. interface_dG, sap, packstat, ipsae, LIS, pDockQ",
        help="Searches all features in the selected category before the top-feature display limit is applied.",
    ).strip().lower()
    if category_feature_query:
        query_terms = [term for term in re.split(r"\s+", category_feature_query) if term]
        plot_ranking = plot_ranking[
            plot_ranking["feature"].map(lambda feature: _metric_query_matches(feature, query_terms))
        ].copy()
        if plot_ranking.empty:
            st.info("No features in this category match the current filter.")
            return
        st.caption(f"{len(plot_ranking)} category features match the filter.")
    label_col = "binder" if metrics is not None and "binder" in metrics.columns else "label" if metrics is not None and "label" in metrics.columns else None
    ranking_basis = st.segmented_control(
        "Ranking statistic",
        ["overall", "target median", "target mean"],
        default="overall",
        key=f"{run_key}_{ranking_key}_{selected_category}_ranking_basis_v1".replace(" ", "_").replace("/", "_"),
        help=(
            "Overall uses the ranking table AP/AUROC across all selected records. "
            "Target median/mean first calculate AP/AUROC per target, then aggregate those target scores."
        ),
    )
    ranking_basis = str(ranking_basis or "overall")
    ap_axis_title = "average precision"
    auroc_axis_title = "AUROC"
    ranking_basis_label = "overall"
    if ranking_basis != "overall":
        if metrics is None or metrics.empty or not label_col or "target_id" not in metrics.columns:
            st.info("Target mean/median ranking needs a metrics table with labels and target IDs. Showing overall ranking instead.")
            ranking_basis = "overall"
        else:
            direction_by_basis_feature = {
                str(row.get("feature")): _effective_feature_direction(pd.Series(row))
                for row in plot_ranking.to_dict(orient="records")
                if str(row.get("feature") or "") in metrics.columns
            }
            target_basis_features = [
                str(feature)
                for feature in plot_ranking["feature"].astype(str).tolist()
                if str(feature) in metrics.columns
            ]
            if not target_basis_features:
                st.info("Target mean/median ranking has no matching per-record metric columns. Showing overall ranking instead.")
                ranking_basis = "overall"
            else:
                _per_target_basis, aggregate_basis = _per_target_feature_performance(
                    metrics,
                    features=target_basis_features,
                    directions=direction_by_basis_feature,
                    label_col=label_col,
                )
                if aggregate_basis.empty:
                    st.info("Target mean/median ranking needs at least one binder and one nonbinder per target. Showing overall ranking instead.")
                    ranking_basis = "overall"
                else:
                    eligible_targets = {
                        str(target_id) if pd.notna(target_id) else "unknown"
                        for target_id, target_df in metrics.groupby("target_id", dropna=False, sort=False)
                        if {
                            label
                            for label in (_truthy_benchmark_label(value) for value in target_df[label_col].tolist())
                            if label in {0, 1}
                        }
                        == {0, 1}
                    }
                    complete_features = set(
                        _per_target_basis.groupby("feature")["target_id"].nunique().loc[
                            lambda counts: counts == len(eligible_targets)
                        ].index.astype(str)
                    )
                    excluded_partial_count = int(len(aggregate_basis) - len(complete_features))
                    aggregate_basis = aggregate_basis[
                        aggregate_basis["feature"].astype(str).isin(complete_features)
                    ].copy()
                    if excluded_partial_count:
                        st.caption(
                            f"Excluded {excluded_partial_count} partial-coverage feature(s) from target-level ranking; "
                            "only features scored for every selected target are eligible."
                        )
                    if aggregate_basis.empty:
                        st.info("No features have complete target coverage after the current filters.")
                    ap_col = "median_average_precision" if ranking_basis == "target median" else "mean_average_precision"
                    auroc_col = "median_auroc" if ranking_basis == "target median" else "mean_auroc"
                    basis_score_columns = [
                        "feature",
                        "direction",
                        "proteins",
                        "records",
                        ap_col,
                        auroc_col,
                        "mean_average_precision",
                        "median_average_precision",
                        "sd_average_precision",
                        "mean_auroc",
                        "median_auroc",
                        "sd_auroc",
                    ]
                    basis_score_columns = [
                        column for column in dict.fromkeys(basis_score_columns) if column in aggregate_basis.columns
                    ]
                    basis_scores = aggregate_basis[basis_score_columns].rename(
                        columns={
                            ap_col: "best_average_precision",
                            auroc_col: "best_auroc",
                            "proteins": "target_count",
                            "records": "target_ranked_records",
                        }
                    )
                    merge_drop = [
                        "direction",
                        "best_average_precision",
                        "best_auroc",
                        "mean_average_precision",
                        "median_average_precision",
                        "sd_average_precision",
                        "mean_auroc",
                        "median_auroc",
                        "sd_auroc",
                        "target_count",
                        "target_ranked_records",
                    ]
                    plot_ranking = (
                        plot_ranking.drop(columns=[column for column in merge_drop if column in plot_ranking.columns])
                        .merge(basis_scores, on="feature", how="inner")
                        .copy()
                    )
                    ranking_basis_label = "target median" if ranking_basis == "target median" else "target mean"
                    ap_axis_title = f"{ranking_basis_label} AP"
                    auroc_axis_title = f"{ranking_basis_label} AUROC"
                    if plot_ranking.empty:
                        st.info("No features have target-level scores after the current filters.")
                        return
    _show_category_threshold_report_tool(
        benchmark_dir=benchmark_dir,
        run_key=run_key,
        ranking_key=ranking_key,
        ranking=ranking,
        metrics=metrics,
        selected_engines=selected_engines,
        target_filter_label=target_filter_label,
    )
    bar_selection_mode = st.segmented_control(
        "Bar plot feature selection",
        ["combined top", "best per engine", "first per engine"],
        default="combined top",
        key=f"{run_key}_{ranking_key}_bar_selection_mode_v1",
        help=(
            "Combined top shows the highest-ranked features overall. Best per engine shows one winner per engine. "
            "First per engine walks the overall ranking and keeps the first X features for each engine."
        ),
    )
    bar_selection_mode = str(bar_selection_mode or "combined top")
    show_best_per_engine = bar_selection_mode == "best per engine"
    show_first_per_engine = bar_selection_mode == "first per engine"

    top_n = 20
    per_engine_top_n = 3
    if show_first_per_engine:
        per_engine_top_n = st.number_input(
            "Features per engine",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            key=f"{run_key}_{ranking_key}_first_per_engine_n_v1",
            help="Keep this many features per engine while preserving the overall AP ranking order.",
        )
    elif not show_best_per_engine:
        top_n = st.slider("Top features shown", min_value=5, max_value=50, value=20, step=5, key=f"{run_key}_{ranking_key}_top_n_v2")
    plot_ranking = plot_ranking.copy()
    plot_ranking["display_feature"] = plot_ranking["feature"].map(_display_feature_name)
    engine_summary = pd.DataFrame()
    if "best_average_precision" in plot_ranking.columns:
        engine_summary = (
            plot_ranking.dropna(subset=["best_average_precision"])
            .sort_values(["engine", "best_average_precision", "best_auroc"], ascending=[True, False, False])
            .groupby("engine", as_index=False)
            .head(1)
            .sort_values("best_average_precision", ascending=False)
            .copy()
        )
    target_label = "all_targets"
    if selected_targets:
        target_label = "_".join(str(target) for target in selected_targets[:4])
        if len(selected_targets) > 4:
            target_label += f"_plus_{len(selected_targets) - 4}"
    safe_category = selected_category.lower().replace(" ", "_").replace("/", "_")
    safe_target = target_label.replace(" ", "_").replace("/", "_")
    chart_state_key = (
        f"{run_key}_{ranking_key}_{target_filter_label}_{selected_category}_"
        f"{ranking_basis}_{int(bool(show_aliases))}_{int(bool(show_setup_features))}_{bar_selection_mode}_{int(top_n)}_{int(per_engine_top_n)}"
    ).replace(" ", "_").replace("/", "_")
    ranked_plot = (
        plot_ranking.dropna(subset=["best_average_precision"])
        .sort_values(["best_average_precision", "best_auroc"], ascending=[False, False])
        .copy()
    )
    if show_best_per_engine and not engine_summary.empty:
        top = engine_summary.copy()
        bar_metric_title = "AP" if ranking_basis == "overall" else ap_axis_title.title()
        bar_title = f"Best Feature Per Engine By {bar_metric_title}"
        bar_height = max(260, len(top) * 36)
    elif show_first_per_engine:
        if "engine" in ranked_plot.columns:
            top = (
                ranked_plot.assign(_engine_rank=ranked_plot.groupby("engine", sort=False).cumcount() + 1)
                .loc[lambda df: df["_engine_rank"] <= int(per_engine_top_n)]
                .drop(columns=["_engine_rank"])
                .copy()
            )
        else:
            top = ranked_plot.head(int(per_engine_top_n)).copy()
        bar_metric_title = "AP" if ranking_basis == "overall" else ap_axis_title.title()
        bar_title = f"First {int(per_engine_top_n)} Features Per Engine By Overall {bar_metric_title} Order"
        bar_height = max(260, len(top) * 26)
    else:
        top = ranked_plot.head(int(top_n)).copy()
        bar_metric_title = "AP" if ranking_basis == "overall" else ap_axis_title.title()
        bar_title = f"Top Features By {bar_metric_title}"
        bar_height = max(260, int(top_n) * 26)
    top_engine_color_scale = _engine_color_scale(top["engine"] if "engine" in top.columns else [])
    plot_engine_color_scale = _engine_color_scale(plot_ranking["engine"] if "engine" in plot_ranking.columns else [])
    engine_summary_color_scale = _engine_color_scale(
        engine_summary["engine"] if "engine" in engine_summary.columns else []
    )

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.subheader(bar_title)
        if "best_average_precision" in top.columns:
            top_feature_order = (
                top.sort_values("best_average_precision", ascending=False)["display_feature"].astype(str).tolist()
            )
            top_bar = (
                alt.Chart(top)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "best_average_precision:Q",
                        title=ap_axis_title,
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    y=alt.Y(
                        "display_feature:N",
                        title=None,
                        sort=top_feature_order,
                        axis=alt.Axis(labelLimit=260),
                    ),
                    color=alt.Color("engine:N", scale=top_engine_color_scale),
                    tooltip=[
                        alt.Tooltip("display_feature:N", title="feature"),
                        "engine:N",
                        "category:N",
                        "direction:N",
                        "aliases:N",
                        alt.Tooltip("best_average_precision:Q", title=ap_axis_title, format=".3f"),
                        alt.Tooltip("best_auroc:Q", title=auroc_axis_title, format=".3f"),
                        alt.Tooltip("positive_mean:Q", title="binder mean", format=".3f"),
                        alt.Tooltip("negative_mean:Q", title="nonbinder mean", format=".3f"),
                    ],
                )
            )
            top_labels = (
                alt.Chart(top)
                .mark_text(align="left", baseline="middle", dx=5, color="#1f2937", fontSize=11)
                .encode(
                    x=alt.X("best_average_precision:Q", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y(
                        "display_feature:N",
                        sort=top_feature_order,
                    ),
                    text=alt.Text("best_average_precision:Q", format=".3f"),
                )
            )
            top_chart = alt.layer(top_bar, top_labels).properties(height=bar_height).interactive()
            st.altair_chart(top_chart, width="stretch", key=f"{chart_state_key}_top_ap")
        else:
            st.info("Average precision columns are missing.")
    with chart_cols[1]:
        st.subheader("AP vs AUROC")
        if {"best_average_precision", "best_auroc"}.issubset(plot_ranking.columns):
            scatter_data = plot_ranking.dropna(subset=["best_average_precision", "best_auroc"]).copy()
            base_scatter = (
                alt.Chart(scatter_data)
                .mark_circle(size=55, opacity=0.45 if (show_best_per_engine or show_first_per_engine) else 0.85)
                .encode(
                    x=alt.X("best_auroc:Q", title=auroc_axis_title, scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("best_average_precision:Q", title=ap_axis_title, scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("engine:N", scale=plot_engine_color_scale),
                    tooltip=[
                        alt.Tooltip("display_feature:N", title="feature"),
                        "engine:N",
                        "category:N",
                        "direction:N",
                        "aliases:N",
                        alt.Tooltip("best_average_precision:Q", title=ap_axis_title, format=".3f"),
                        alt.Tooltip("best_auroc:Q", title=auroc_axis_title, format=".3f"),
                    ],
                )
                .properties(height=max(260, int(top_n) * 26))
            )
            highlight_rows = top if (show_best_per_engine or show_first_per_engine) else pd.DataFrame()
            if not highlight_rows.empty:
                highlight_data = highlight_rows.dropna(subset=["best_average_precision", "best_auroc"]).copy()
                highlight_scatter = (
                    alt.Chart(highlight_data)
                    .mark_point(size=180, filled=True, stroke="black", strokeWidth=2, opacity=1.0)
                    .encode(
                        x=alt.X("best_auroc:Q", title=auroc_axis_title, scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y("best_average_precision:Q", title=ap_axis_title, scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("engine:N", scale=plot_engine_color_scale),
                        tooltip=[
                            alt.Tooltip("display_feature:N", title="feature"),
                            "engine:N",
                            "category:N",
                            "direction:N",
                            "aliases:N",
                            alt.Tooltip("best_average_precision:Q", title=ap_axis_title, format=".3f"),
                            alt.Tooltip("best_auroc:Q", title=auroc_axis_title, format=".3f"),
                        ],
                    )
                )
                scatter_chart = (base_scatter + highlight_scatter).interactive()
            else:
                scatter_chart = base_scatter.interactive()
            st.altair_chart(scatter_chart, width="stretch", key=f"{chart_state_key}_ap_auroc")
        else:
            st.info("AP/AUROC columns are missing.")

    preset_auto_tab, preset_manual_tab = st.tabs(["Save best AP preset", "Manual selection"])
    with preset_auto_tab:
        if not engine_summary.empty:
            st.caption(
                "Save the best feature per engine from the current ranking statistic, target filter, category, and engine selection."
            )
            _save_preset_controls(
                run_key=run_key,
                key_suffix=f"{ranking_key}_{safe_category}_{safe_target}_{ranking_basis}_visible_auto",
                benchmark_dir=benchmark_dir,
                rows=engine_summary,
                default_name=f"{benchmark_dir.parents[1].name}_{safe_category}_{safe_target}_{ranking_basis.replace(' ', '_')}_best_ap",
                description=(
                    f"Best feature per engine by {ranking_basis_label} AP from {selected_ranking_label}; "
                    f"category={selected_category}; targets={target_label}."
                ),
                selection_rule=f"best_{ranking_basis.replace(' ', '_')}_ap_per_engine_filtered",
                extra={
                    "source_table": selected_ranking_label,
                    "target_filter": target_label,
                    "selection_category": selected_category,
                    "ranking_basis": ranking_basis,
                },
            )
        else:
            st.info("No best-AP engine features are available for the current filters.")
    with preset_manual_tab:
        _show_ranking_preset_builder(
            run_key=run_key,
            ranking_key=ranking_key,
            benchmark_dir=benchmark_dir,
            ranking=plot_ranking,
            engine_summary=engine_summary,
            selected_category=selected_category,
            selected_targets=[str(target) for target in selected_targets] if selected_targets else None,
            selected_ranking_label=selected_ranking_label,
        )

    if "best_average_precision" in plot_ranking.columns:
        st.subheader("Best Feature Per Engine")
        show_cols = [
            "engine",
            "category",
            "feature",
            "direction",
            "best_average_precision",
            "best_auroc",
            "positive_mean",
            "negative_mean",
        ]
        st.dataframe(engine_summary[[col for col in show_cols if col in engine_summary.columns]], hide_index=True, width="stretch")
        engine_chart = (
            alt.Chart(engine_summary)
            .mark_bar()
            .encode(
                x=alt.X("engine:N", title="engine"),
                y=alt.Y("best_average_precision:Q", title="best AP", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("engine:N", scale=engine_summary_color_scale),
                tooltip=[
                    "engine:N",
                    alt.Tooltip("display_feature:N", title="feature"),
                    alt.Tooltip("best_average_precision:Q", title="AP", format=".3f"),
                    alt.Tooltip("best_auroc:Q", title="AUROC", format=".3f"),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(engine_chart, width="stretch", key=f"{chart_state_key}_best_engine")

    if metrics is None or metrics.empty:
        st.info("No matching per-record metrics table is available for binder/nonbinder separation.")
        return
    st.subheader("Binder / Nonbinder Separation")
    label_col = "binder" if "binder" in metrics.columns else "label" if "label" in metrics.columns else None
    selector_ranking = top if (show_best_per_engine or show_first_per_engine) and not top.empty else ranked_plot
    if not selector_ranking.empty and {"best_average_precision", "best_auroc"}.issubset(selector_ranking.columns):
        selector_ranking = selector_ranking.sort_values(
            ["best_average_precision", "best_auroc"],
            ascending=[False, False],
            na_position="last",
        ).copy()
    feature_options = [feature for feature in selector_ranking["feature"].tolist() if feature in metrics.columns]
    if not label_col or not feature_options:
        st.info("Merged metrics do not contain compatible labels and ranked feature columns.")
        return
    selected_feature = st.selectbox("Feature", feature_options, index=0, key=f"{chart_state_key}_feature_v2")
    feature_chart_key = f"{chart_state_key}_{str(selected_feature).replace(' ', '_').replace('/', '_')}"
    selected_row = selector_ranking[selector_ranking["feature"] == selected_feature].head(1)
    direction = _effective_feature_direction(selected_row.iloc[0] if not selected_row.empty else None)
    plot_columns = [label_col, selected_feature]
    for column in ["binder_id", "target_id", "source", "original_binder_id"]:
        if column in metrics.columns and column not in plot_columns:
            plot_columns.append(column)
    plot_df = metrics[plot_columns].copy()
    plot_df[selected_feature] = pd.to_numeric(plot_df[selected_feature], errors="coerce")
    plot_df["selected_value"] = plot_df[selected_feature]
    plot_df["_input_order"] = range(len(plot_df))
    plot_df = plot_df.dropna(subset=[selected_feature]).copy()
    sort_columns = [column for column in ["target_id", "source", "binder_id", "_input_order"] if column in plot_df.columns]
    if sort_columns:
        plot_df = plot_df.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    else:
        plot_df = plot_df.reset_index(drop=True)
    plot_df["record"] = plot_df.index + 1
    plot_df["class"] = plot_df[label_col].map(lambda value: "binder" if str(value).lower() in {"1", "true", "yes", "y", "binder"} else "nonbinder")
    if "target_id" in plot_df.columns:
        plot_df["protein"] = plot_df["target_id"].fillna("").astype(str).replace("", "unknown")
    coverage_parts = [f"{len(plot_df):,} of {len(metrics):,} records have values for `{selected_feature}`"]
    coverage_parts.append(
        f"{int((plot_df['class'] == 'binder').sum()):,} binders and {int((plot_df['class'] == 'nonbinder').sum()):,} nonbinders in this feature slice"
    )
    if "target_id" in metrics.columns:
        total_targets = int(metrics["target_id"].nunique(dropna=True))
        valid_targets = int(plot_df["target_id"].nunique(dropna=True)) if "target_id" in plot_df.columns else 0
        coverage_parts.append(f"{valid_targets} of {total_targets} targets")
    st.caption("Selected feature coverage: " + "; ".join(coverage_parts) + ". Feature-specific plots use only those records.")
    plot_df["rank_score"] = -plot_df[selected_feature] if direction == "lower" else plot_df[selected_feature]
    ranked_df = plot_df.sort_values("rank_score", ascending=False).reset_index(drop=True)
    ranked_df["rank"] = ranked_df.index + 1
    ranked_df["is_binder"] = (ranked_df["class"] == "binder").astype(int)
    ranked_df["true_positives"] = ranked_df["is_binder"].cumsum()
    ranked_df["precision"] = ranked_df["true_positives"] / ranked_df["rank"]
    total_binders = int(ranked_df["is_binder"].sum())
    total_nonbinders = int(len(ranked_df) - total_binders)
    ranked_df["recall"] = ranked_df["true_positives"] / total_binders if total_binders else 0.0
    ranked_df["false_positives"] = (1 - ranked_df["is_binder"]).cumsum()
    ranked_df["tpr"] = ranked_df["recall"]
    ranked_df["fpr"] = ranked_df["false_positives"] / total_nonbinders if total_nonbinders else 0.0
    threshold_summary = _optimal_threshold_summary(
        plot_df[selected_feature],
        plot_df[label_col],
        direction=str(direction),
    )
    if threshold_summary:
        per_target_threshold = pd.DataFrame()
        target_median_success_rate = None
        if "target_id" in plot_df.columns:
            lower_is_better = str(direction).lower() == "lower"
            threshold_value = float(threshold_summary["threshold"])
            target_summary_df = plot_df.copy()
            target_summary_df["_passes_threshold"] = (
                target_summary_df[selected_feature] <= threshold_value
                if lower_is_better
                else target_summary_df[selected_feature] >= threshold_value
            )
            per_target_rows: list[dict[str, Any]] = []
            for target_id, target_group in target_summary_df.groupby("target_id", dropna=False, sort=False):
                passed = target_group[target_group["_passes_threshold"]]
                target_binders = int((target_group["class"] == "binder").sum())
                selected_count = int(len(passed))
                selected_binders = int((passed["class"] == "binder").sum()) if selected_count else 0
                per_target_rows.append(
                    {
                        "target": str(target_id) if pd.notna(target_id) else "unknown",
                        "records": int(len(target_group)),
                        "binders": target_binders,
                        "selected": selected_count,
                        "true_binders_selected": selected_binders,
                        "false_positives": selected_count - selected_binders,
                        "success_rate_percent": (selected_binders / selected_count * 100.0) if selected_count else None,
                        "recall_percent": (selected_binders / target_binders * 100.0) if target_binders else None,
                    }
                )
            per_target_threshold = pd.DataFrame(per_target_rows)
            if not per_target_threshold.empty:
                target_success_values = pd.to_numeric(
                    per_target_threshold["success_rate_percent"],
                    errors="coerce",
                ).dropna()
                if not target_success_values.empty:
                    target_median_success_rate = float(target_success_values.median())
        threshold_cols = st.columns(7 if target_median_success_rate is not None else 6)
        threshold_cols[0].metric(
            "Optimal threshold",
            f"{threshold_summary['condition']} {threshold_summary['threshold']:.4g}",
            help="Threshold that maximizes F1 on the selected benchmark feature.",
        )
        threshold_cols[1].metric(
            "Success rate",
            f"{threshold_summary['success_rate'] * 100:.1f}%",
            help="Precision at the threshold: true binders among selected designs.",
        )
        metric_offset = 0
        if target_median_success_rate is not None:
            threshold_cols[2].metric(
                "Target median success",
                f"{target_median_success_rate:.1f}%",
                help="Median per-target precision at this threshold, considering only targets with at least one selected design.",
            )
            metric_offset = 1
        threshold_cols[2 + metric_offset].metric("Recall", f"{threshold_summary['recall'] * 100:.1f}%")
        threshold_cols[3 + metric_offset].metric("F1", f"{threshold_summary['f1']:.3f}")
        threshold_cols[4 + metric_offset].metric(
            "Selected",
            f"{int(threshold_summary['selected']):,}",
            help=f"{threshold_summary['selected_fraction'] * 100:.1f}% of records with this feature.",
        )
        threshold_cols[5 + metric_offset].metric(
            "True binders",
            f"{int(threshold_summary['true_binders']):,}",
            help=f"{int(threshold_summary['false_positives']):,} nonbinders would also pass.",
        )
        st.caption(
            f"Using `{selected_feature}` with cutoff `{selected_feature} {threshold_summary['condition']} "
            f"{threshold_summary['threshold']:.4g}` selects {int(threshold_summary['selected']):,} designs; "
            f"{threshold_summary['success_rate'] * 100:.1f}% are labeled binders in this benchmark slice."
        )
        if not per_target_threshold.empty:
            targets_total = int(len(per_target_threshold))
            targets_with_any_selection = int((per_target_threshold["selected"] > 0).sum())
            targets_with_true_binder_selection = int((per_target_threshold["true_binders_selected"] > 0).sum())
            targets_with_only_false_positive_selection = int(
                (
                    (per_target_threshold["selected"] > 0)
                    & (per_target_threshold["true_binders_selected"] == 0)
                ).sum()
            )
            targets_without_selection = targets_total - targets_with_any_selection
            target_selection_cols = st.columns(4)
            target_selection_cols[0].metric("Targets with selected designs", f"{targets_with_any_selection}/{targets_total}")
            target_selection_cols[1].metric(
                "Targets with true binders selected",
                f"{targets_with_true_binder_selection}/{targets_total}",
            )
            target_selection_cols[2].metric("Targets with no selected designs", f"{targets_without_selection}/{targets_total}")
            target_selection_cols[3].metric(
                "Targets selected only false positives",
                f"{targets_with_only_false_positive_selection}/{targets_total}",
            )
            st.caption(
                "Target coverage at this cutoff: "
                f"{targets_with_any_selection} target(s) would pass at least one design; "
                f"{targets_with_true_binder_selection} target(s) would pass at least one labeled binder."
            )
            per_target_threshold = per_target_threshold.sort_values(
                ["success_rate_percent", "recall_percent", "target"],
                ascending=[False, False, True],
                na_position="last",
            ).reset_index(drop=True)
            with st.expander("Success rate per target at this threshold", expanded=True):
                st.dataframe(
                    per_target_threshold,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "success_rate_percent": st.column_config.NumberColumn("success rate (%)", format="%.1f"),
                        "recall_percent": st.column_config.NumberColumn("recall (%)", format="%.1f"),
                    },
                )
    threshold_tab, selection_tab = st.tabs(["Threshold summary", "Selection success"])
    with threshold_tab:
        st.caption("The F1-optimal threshold summary and its per-target table are shown above.")
    with selection_tab:
        selection_mode = st.segmented_control(
            "Selection rule",
            ["Top N", "Score threshold"],
            default="Top N",
            key=f"{feature_chart_key}_selection_success_mode_v1",
        )
        selection_mode = str(selection_mode or "Top N")
        selected_rows = ranked_df.iloc[0:0].copy()
        selection_description = ""
        if selection_mode == "Top N":
            top_count = int(
                st.number_input(
                    "Take the top designs",
                    min_value=1,
                    max_value=max(1, len(ranked_df)),
                    value=min(20, max(1, len(ranked_df))),
                    step=1,
                    key=f"{feature_chart_key}_selection_success_top_n_v1",
                )
            )
            selected_rows = ranked_df.head(top_count).copy()
            selection_description = f"Top {top_count:,} designs ranked by `{selected_feature}`"
        else:
            feature_min = float(ranked_df[selected_feature].min())
            feature_max = float(ranked_df[selected_feature].max())
            default_threshold = float(threshold_summary["threshold"]) if threshold_summary else float(ranked_df[selected_feature].median())
            threshold_step = max((feature_max - feature_min) / 200.0, 1e-6)
            chosen_threshold = float(
                st.number_input(
                    f"Score threshold ({'below or equal' if direction == 'lower' else 'above or equal'} is selected)",
                    min_value=feature_min,
                    max_value=feature_max,
                    value=min(max(default_threshold, feature_min), feature_max),
                    step=threshold_step,
                    format="%.5g",
                    key=f"{feature_chart_key}_selection_success_threshold_v1",
                )
            )
            passes = ranked_df[selected_feature] <= chosen_threshold if direction == "lower" else ranked_df[selected_feature] >= chosen_threshold
            selected_rows = ranked_df[passes].copy()
            selection_description = (
                f"`{selected_feature} {'<=' if direction == 'lower' else '>='} {chosen_threshold:.5g}`"
            )
        selected_count = int(len(selected_rows))
        selected_binders = int(selected_rows["is_binder"].sum()) if selected_count else 0
        false_positives = selected_count - selected_binders
        success_rate = selected_binders / selected_count if selected_count else 0.0
        recall = selected_binders / total_binders if total_binders else 0.0
        f1_value = 2 * success_rate * recall / (success_rate + recall) if success_rate + recall else 0.0
        selection_metrics = st.columns(5)
        selection_metrics[0].metric("Selected", f"{selected_count:,}")
        selection_metrics[1].metric("Success rate", f"{success_rate * 100:.1f}%")
        selection_metrics[2].metric("Recall", f"{recall * 100:.1f}%")
        selection_metrics[3].metric("True binders", f"{selected_binders:,}")
        selection_metrics[4].metric("False positives", f"{false_positives:,}")
        st.caption(f"{selection_description}; F1 = {f1_value:.3f}.")
        if "target_id" in ranked_df.columns:
            selected_index = set(selected_rows.index.tolist())
            per_target_selection_rows: list[dict[str, Any]] = []
            for target_id, target_group in ranked_df.groupby("target_id", dropna=False, sort=False):
                target_selected = target_group[target_group.index.isin(selected_index)]
                target_binders = int(target_group["is_binder"].sum())
                target_selected_count = int(len(target_selected))
                target_selected_binders = int(target_selected["is_binder"].sum()) if target_selected_count else 0
                per_target_selection_rows.append(
                    {
                        "target": str(target_id) if pd.notna(target_id) else "unknown",
                        "records": int(len(target_group)),
                        "binders": target_binders,
                        "selected": target_selected_count,
                        "true_binders_selected": target_selected_binders,
                        "false_positives": target_selected_count - target_selected_binders,
                        "success_rate_percent": (
                            target_selected_binders / target_selected_count * 100.0 if target_selected_count else None
                        ),
                        "recall_percent": target_selected_binders / target_binders * 100.0 if target_binders else None,
                    }
                )
            per_target_selection = pd.DataFrame(per_target_selection_rows).sort_values(
                ["success_rate_percent", "recall_percent", "target"],
                ascending=[False, False, True],
                na_position="last",
            )
            if not per_target_selection.empty:
                targets_total = int(len(per_target_selection))
                targets_with_any_selection = int((per_target_selection["selected"] > 0).sum())
                targets_with_true_binder_selection = int((per_target_selection["true_binders_selected"] > 0).sum())
                targets_without_selection = targets_total - targets_with_any_selection
                targets_with_only_false_positive_selection = int(
                    (
                        (per_target_selection["selected"] > 0)
                        & (per_target_selection["true_binders_selected"] == 0)
                    ).sum()
                )
                target_selection_cols = st.columns(4)
                target_selection_cols[0].metric("Targets with selected designs", f"{targets_with_any_selection}/{targets_total}")
                target_selection_cols[1].metric(
                    "Targets with true binders selected",
                    f"{targets_with_true_binder_selection}/{targets_total}",
                )
                target_selection_cols[2].metric("Targets with no selected designs", f"{targets_without_selection}/{targets_total}")
                target_selection_cols[3].metric(
                    "Targets selected only false positives",
                    f"{targets_with_only_false_positive_selection}/{targets_total}",
                )
            st.dataframe(
                per_target_selection,
                hide_index=True,
                width="stretch",
                column_config={
                    "success_rate_percent": st.column_config.NumberColumn("success rate (%)", format="%.1f"),
                    "recall_percent": st.column_config.NumberColumn("recall (%)", format="%.1f"),
                },
            )
    roc_df = pd.concat(
        [
            pd.DataFrame([{"rank": 0, "fpr": 0.0, "tpr": 0.0}]),
            ranked_df[["rank", "fpr", "tpr"]],
        ],
        ignore_index=True,
    )
    auroc_value = None
    if "best_auroc" in selected_row.columns and not selected_row.empty:
        auroc_value = pd.to_numeric(selected_row["best_auroc"], errors="coerce").iloc[0]
    ranked_cols = st.columns(3)
    with ranked_cols[0]:
        st.subheader("Ranked By Feature")
        threshold_rank_score = None
        threshold_rule_df = pd.DataFrame()
        if threshold_summary:
            threshold_rank_score = (
                -threshold_summary["threshold"]
                if str(direction).lower() == "lower"
                else threshold_summary["threshold"]
            )
            threshold_rule_df = pd.DataFrame([{"threshold_rank_score": threshold_rank_score}])
        ranked_chart = (
            alt.Chart(ranked_df)
            .mark_circle(size=75, opacity=0.9)
            .encode(
                x=alt.X("rank:Q", title="rank"),
                y=alt.Y("rank_score:Q", title="rank score"),
                color=alt.Color("class:N", scale=CLASS_COLOR_SCALE),
                tooltip=["rank:Q", "class:N", alt.Tooltip("rank_score:Q", format=".4f")],
            )
            .interactive()
        )
        if not threshold_rule_df.empty:
            ranked_threshold_rule = (
                alt.Chart(threshold_rule_df)
                .mark_rule(color="#E11D48", strokeDash=[5, 3], strokeWidth=2)
                .encode(y=alt.Y("threshold_rank_score:Q"))
            )
            ranked_chart = ranked_chart + ranked_threshold_rule
        st.altair_chart(ranked_chart, width="stretch", key=f"{feature_chart_key}_ranked")
    with ranked_cols[1]:
        st.subheader("Precision / Recall By Threshold")
        threshold_curve = _threshold_precision_recall_curve(
            plot_df[selected_feature],
            plot_df[label_col],
            direction=str(direction),
        )
        if threshold_curve.empty:
            st.info("No threshold curve is available for this feature.")
        else:
            pr_df = threshold_curve[["threshold", "precision", "recall", "selected"]].melt(
                id_vars=["threshold", "selected"],
                var_name="metric",
                value_name="value",
            )
            pr_chart = (
                alt.Chart(pr_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("threshold:Q", title=f"{_combo_feature_term(selected_feature)} threshold"),
                    y=alt.Y("value:Q", title="value", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("metric:N", scale=LINE_COLOR_SCALE),
                    tooltip=[
                        alt.Tooltip("threshold:Q", title="threshold", format=".4g"),
                        alt.Tooltip("selected:Q", title="selected"),
                        "metric:N",
                        alt.Tooltip("value:Q", format=".3f"),
                    ],
                )
                .interactive()
            )
            if threshold_summary:
                pr_threshold_df = pd.DataFrame([{"threshold": float(threshold_summary["threshold"])}])
                pr_threshold_rule = (
                    alt.Chart(pr_threshold_df)
                    .mark_rule(color="#E11D48", strokeDash=[5, 3], strokeWidth=2)
                    .encode(x=alt.X("threshold:Q"))
                )
                pr_chart = pr_chart + pr_threshold_rule
            st.altair_chart(pr_chart, width="stretch", key=f"{feature_chart_key}_precision_recall_threshold_v1")
    with ranked_cols[2]:
        roc_title = "ROC Curve"
        if pd.notna(auroc_value):
            roc_title = f"ROC Curve (AUROC {float(auroc_value):.3f})"
        st.subheader(roc_title)
        diagonal = pd.DataFrame([{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}])
        diagonal_chart = (
            alt.Chart(diagonal)
            .mark_line(strokeDash=[4, 4], color="#9CA3AF")
            .encode(
                x=alt.X("fpr:Q", title="false positive rate", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("tpr:Q", title="true positive rate", scale=alt.Scale(domain=[0, 1])),
            )
        )
        roc_chart = (
            alt.Chart(roc_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("fpr:Q", title="false positive rate", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("tpr:Q", title="true positive rate", scale=alt.Scale(domain=[0, 1])),
                tooltip=[
                    "rank:Q",
                    alt.Tooltip("fpr:Q", title="FPR", format=".3f"),
                    alt.Tooltip("tpr:Q", title="TPR", format=".3f"),
                ],
            )
            .interactive()
        )
        st.altair_chart(diagonal_chart + roc_chart, width="stretch", key=f"{feature_chart_key}_roc")
    with st.expander("Input-order scatter with target separators", expanded=True):
        input_tooltip: list[Any] = [
            "record:Q",
            "class:N",
            alt.Tooltip("selected_value:Q", title=str(selected_feature), format=".4f"),
        ]
        for column in ["binder_id", "target_id", "source", "original_binder_id"]:
            if column in plot_df.columns:
                input_tooltip.insert(-1, alt.Tooltip(f"{column}:N", title=column))
        y_values = pd.to_numeric(plot_df["selected_value"], errors="coerce")
        y_min = y_values.min()
        y_max = y_values.max()
        if threshold_summary:
            threshold_value = float(threshold_summary["threshold"])
            y_min = min(float(y_min), threshold_value) if pd.notna(y_min) else threshold_value
            y_max = max(float(y_max), threshold_value) if pd.notna(y_max) else threshold_value
        y_domain = None
        label_y = None
        if pd.notna(y_min) and pd.notna(y_max):
            y_min_float = float(y_min)
            y_max_float = float(y_max)
            span = y_max_float - y_min_float
            padding = span * 0.08 if span > 0 else max(abs(y_max_float) * 0.08, 1.0)
            y_domain = [y_min_float - padding, y_max_float + padding]
            label_y = y_max_float + padding * 0.45
        max_record = int(pd.to_numeric(plot_df["record"], errors="coerce").max()) if "record" in plot_df.columns else 0
        x_domain = [0, max(max_record + 1, 1)]
        input_chart = (
            alt.Chart(plot_df)
            .mark_circle(size=75, opacity=0.9)
            .encode(
                x=alt.X("record:Q", title="valid input record", scale=alt.Scale(domain=x_domain)),
                y=alt.Y("selected_value:Q", title=str(selected_feature), scale=alt.Scale(domain=y_domain) if y_domain else alt.Undefined),
                color=alt.Color("class:N", scale=CLASS_COLOR_SCALE),
                tooltip=input_tooltip,
            )
            .interactive()
        )
        if threshold_summary:
            threshold_line_df = pd.DataFrame(
                [
                    {
                        "threshold": float(threshold_summary["threshold"]),
                        "label": f"{selected_feature} {threshold_summary['condition']} {threshold_summary['threshold']:.4g}",
                    }
                ]
            )
            threshold_line = (
                alt.Chart(threshold_line_df)
                .mark_rule(color="#E11D48", strokeDash=[5, 3], strokeWidth=2)
                .encode(
                    y=alt.Y("threshold:Q"),
                    tooltip=[
                        alt.Tooltip("label:N", title="threshold"),
                        alt.Tooltip("threshold:Q", title="value", format=".4g"),
                    ],
                )
            )
            threshold_label = (
                alt.Chart(threshold_line_df)
                .mark_text(align="left", baseline="bottom", dx=6, dy=-3, color="#E11D48", fontSize=11)
                .encode(
                    x=alt.value(8),
                    y=alt.Y("threshold:Q"),
                    text=alt.Text("label:N"),
                )
            )
            input_chart = input_chart + threshold_line + threshold_label
        if "protein" in plot_df.columns and plot_df["protein"].nunique(dropna=True) > 1:
            protein_labels = (
                plot_df.dropna(subset=["protein"])
                .groupby("protein", sort=False, as_index=False)
                .agg(
                    record=("record", "min"),
                    record_end=("record", "max"),
                    count=("record", "count"),
                    binders=("class", lambda values: int((values == "binder").sum())),
                    value_min=("selected_value", "min"),
                    value_max=("selected_value", "max"),
                )
            )
            if len(protein_labels) <= 40 and label_y is not None:
                protein_labels["label_y"] = label_y
                target_tooltip = [
                    alt.Tooltip("protein:N", title="target_id"),
                    alt.Tooltip("record:Q", title="first valid record"),
                    alt.Tooltip("record_end:Q", title="last valid record"),
                    alt.Tooltip("count:Q", title="valid records"),
                    alt.Tooltip("binders:Q", title="binders"),
                    alt.Tooltip("value_min:Q", title="target min", format=".4f"),
                    alt.Tooltip("value_max:Q", title="target max", format=".4f"),
                ]
                target_rules = (
                    alt.Chart(protein_labels)
                    .mark_rule(color="#374151", opacity=0.55, strokeDash=[4, 3], strokeWidth=1.5)
                    .encode(
                        x=alt.X("record:Q", title="valid input record", scale=alt.Scale(domain=x_domain)),
                        tooltip=target_tooltip,
                    )
                )
                target_labels = (
                    alt.Chart(protein_labels)
                    .mark_text(align="left", baseline="top", angle=325, dx=3, dy=0, color="#4B5563", fontSize=11)
                    .encode(
                        x=alt.X("record:Q", title="valid input record", scale=alt.Scale(domain=x_domain)),
                        y=alt.Y("label_y:Q"),
                        text=alt.Text("protein:N"),
                        tooltip=target_tooltip,
                    )
                )
                input_chart = target_rules + input_chart + target_labels
            st.caption(
                "Hover points for binder and target details; target labels mark the first valid record for each protein "
                "after the current feature, target, and prefilter selections."
            )
        st.altair_chart(input_chart, width="stretch", key=f"{feature_chart_key}_input_order")
    if "target_id" in metrics.columns and metrics["target_id"].nunique(dropna=True) > 1:
        st.subheader("Per-Protein Feature Performance")
        direction_by_feature = {
            str(row.get("feature")): _effective_feature_direction(pd.Series(row))
            for row in selector_ranking.to_dict(orient="records")
            if str(row.get("feature") or "") in feature_options
        }
        default_features = []
        if "feature" in engine_summary.columns:
            default_features = (
                engine_summary[engine_summary["feature"].isin(feature_options)]
                .drop_duplicates("feature")["feature"]
                .astype(str)
                .tolist()
            )
        if not default_features:
            default_features = [str(selected_feature)]
            ranked_default_features = (
                selector_ranking[selector_ranking["feature"].isin(feature_options)]
                .drop_duplicates("feature")["feature"]
                .astype(str)
                .head(8)
                .tolist()
            )
            for feature in ranked_default_features:
                if feature not in default_features:
                    default_features.append(feature)
        feature_query = st.text_input(
            "Filter features",
            value="",
            key=f"{chart_state_key}_per_target_feature_filter_v1",
            placeholder="e.g. ipsae, ipae, plddt, rosetta",
        ).strip().lower()
        filtered_feature_options = feature_options
        if feature_query:
            query_terms = [term for term in re.split(r"\s+", feature_query) if term]
            filtered_feature_options = [
                feature
                for feature in feature_options
                if all(term in str(feature).lower() for term in query_terms)
            ]
            if not filtered_feature_options:
                st.info("No features match the current filter.")
        feature_selection_key = f"{chart_state_key}_per_target_features_v2"
        feature_query_tracker_key = f"{chart_state_key}_per_target_feature_filter_applied_v2"
        previous_feature_query = st.session_state.get(feature_query_tracker_key)
        if previous_feature_query != feature_query:
            st.session_state[feature_selection_key] = (
                list(filtered_feature_options)
                if feature_query
                else [feature for feature in default_features if feature in filtered_feature_options]
            )
            st.session_state[feature_query_tracker_key] = feature_query
        selected_per_target_features = st.multiselect(
            "Features for per-protein AP/AUROC",
            filtered_feature_options,
            default=[feature for feature in default_features if feature in filtered_feature_options],
            key=feature_selection_key,
            help=(
                "Typing a filter selects all matching features across engines. "
                "You can then remove individual features from the selection."
            ),
        )
        per_target_scope = st.segmented_control(
            "Feature scope for per-target plots",
            ["all matching features", "selected features"],
            default="selected features",
            key=f"{chart_state_key}_per_target_feature_scope_v2",
            help=(
                "Use all features that match the current category/search/engine filters, or restrict to the multiselect above."
            ),
        )
        per_target_scope = str(per_target_scope or "all matching features")
        per_target_features_for_plots = (
            list(filtered_feature_options)
            if per_target_scope == "all matching features"
            else [str(feature) for feature in selected_per_target_features]
        )
        if per_target_features_for_plots:
            per_target_perf, aggregate_perf = _per_target_feature_performance(
                metrics,
                features=[str(feature) for feature in per_target_features_for_plots],
                directions=direction_by_feature,
                label_col=label_col,
            )
            if aggregate_perf.empty:
                st.info("Per-protein AP/AUROC needs at least one binder and one nonbinder with valid scores for each target.")
            else:
                shown_cols = [
                    "engine",
                    "feature",
                    "direction",
                    "proteins",
                    "median_average_precision",
                    "mean_average_precision",
                    "sd_average_precision",
                    "median_auroc",
                    "mean_auroc",
                    "sd_auroc",
                    "records",
                ]
                st.dataframe(
                    aggregate_perf[[col for col in shown_cols if col in aggregate_perf.columns]],
                    hide_index=True,
                    width="stretch",
                )
                aggregate_choice = st.segmented_control(
                    "Per-target aggregation",
                    ["median", "mean"],
                    default="median",
                    key=f"{chart_state_key}_per_target_aggregate_stat_v1",
                    help=(
                        "Median is the default because per-target AP/AUROC can be skewed by target class balance "
                        "and unusually easy or hard targets. Mean is useful when you want the arithmetic average."
                    ),
                )
                aggregate_choice = str(aggregate_choice or "median")
                ap_metric_col = "median_average_precision" if aggregate_choice == "median" else "mean_average_precision"
                auroc_metric_col = "median_auroc" if aggregate_choice == "median" else "mean_auroc"
                engine_perf = aggregate_perf.dropna(subset=[ap_metric_col, auroc_metric_col]).copy()
                engine_perf["engine_feature"] = engine_perf["engine"].astype(str) + " | " + engine_perf["feature"].astype(str)
                if not engine_perf.empty:
                    if aggregate_choice == "median":
                        engine_perf["ap_error_low"] = engine_perf["q25_average_precision"]
                        engine_perf["ap_error_high"] = engine_perf["q75_average_precision"]
                        engine_perf["auroc_error_low"] = engine_perf["q25_auroc"]
                        engine_perf["auroc_error_high"] = engine_perf["q75_auroc"]
                        error_label = "IQR"
                    else:
                        engine_perf["ap_error_low"] = engine_perf["ap_low"]
                        engine_perf["ap_error_high"] = engine_perf["ap_high"]
                        engine_perf["auroc_error_low"] = engine_perf["auroc_low"]
                        engine_perf["auroc_error_high"] = engine_perf["auroc_high"]
                        error_label = "SD"
                    engine_color_scale = _engine_color_scale(engine_perf["engine"])
                    engine_sort = (
                        engine_perf.sort_values([ap_metric_col, auroc_metric_col], ascending=[False, False])["engine_feature"].tolist()
                    )
                    engine_y = alt.Y(
                        "engine_feature:N",
                        title=None,
                        sort=engine_sort,
                        axis=alt.Axis(labelLimit=320),
                        scale=alt.Scale(paddingOuter=0.45),
                    )
                    engine_chart_height = max(320, 44 * len(engine_perf) + 36)
                    engine_ap_base = alt.Chart(engine_perf).encode(
                        y=engine_y,
                        color=alt.Color("engine:N", scale=engine_color_scale),
                        tooltip=[
                            "engine:N",
                            "feature:N",
                            alt.Tooltip("proteins:Q", title="targets"),
                            alt.Tooltip("records:Q", title="records"),
                            alt.Tooltip(f"{ap_metric_col}:Q", title=f"{aggregate_choice} AP", format=".3f"),
                            alt.Tooltip("mean_average_precision:Q", title="mean AP", format=".3f"),
                            alt.Tooltip("sd_average_precision:Q", title="AP SD", format=".3f"),
                            alt.Tooltip("q25_average_precision:Q", title="AP q25", format=".3f"),
                            alt.Tooltip("q75_average_precision:Q", title="AP q75", format=".3f"),
                        ],
                    )
                    engine_ap_chart = (
                        engine_ap_base.mark_bar(opacity=0.86)
                        .encode(x=alt.X(f"{ap_metric_col}:Q", title=f"{aggregate_choice} AP", scale=alt.Scale(domain=[0, 1])))
                        + engine_ap_base.mark_rule(color="#111827", strokeWidth=2)
                        .encode(
                            x=alt.X("ap_error_low:Q", title=f"{aggregate_choice} AP", scale=alt.Scale(domain=[0, 1])),
                            x2="ap_error_high:Q",
                            color=alt.value("#111827"),
                        )
                        + engine_ap_base.mark_tick(color="#111827", size=14, thickness=2)
                        .encode(
                            x=alt.X(f"{ap_metric_col}:Q", scale=alt.Scale(domain=[0, 1])),
                            color=alt.value("#111827"),
                        )
                        + engine_ap_base.mark_text(
                            align="left",
                            baseline="bottom",
                            dx=5,
                            dy=-7,
                            color="#1f2937",
                            fontSize=11,
                            fontWeight="bold",
                        )
                        .encode(
                            x=alt.X(f"{ap_metric_col}:Q", scale=alt.Scale(domain=[0, 1])),
                            text=alt.Text(f"{ap_metric_col}:Q", format=".3f"),
                            color=alt.value("#1f2937"),
                        )
                    ).properties(height=engine_chart_height).interactive()
                    engine_auroc_base = alt.Chart(engine_perf).encode(
                        y=engine_y,
                        color=alt.Color("engine:N", scale=engine_color_scale),
                        tooltip=[
                            "engine:N",
                            "feature:N",
                            alt.Tooltip("proteins:Q", title="targets"),
                            alt.Tooltip("records:Q", title="records"),
                            alt.Tooltip(f"{auroc_metric_col}:Q", title=f"{aggregate_choice} AUROC", format=".3f"),
                            alt.Tooltip("mean_auroc:Q", title="mean AUROC", format=".3f"),
                            alt.Tooltip("sd_auroc:Q", title="AUROC SD", format=".3f"),
                            alt.Tooltip("q25_auroc:Q", title="AUROC q25", format=".3f"),
                            alt.Tooltip("q75_auroc:Q", title="AUROC q75", format=".3f"),
                        ],
                    )
                    engine_auroc_chart = (
                        engine_auroc_base.mark_bar(opacity=0.86)
                        .encode(x=alt.X(f"{auroc_metric_col}:Q", title=f"{aggregate_choice} AUROC", scale=alt.Scale(domain=[0, 1])))
                        + engine_auroc_base.mark_rule(color="#111827", strokeWidth=2)
                        .encode(
                            x=alt.X("auroc_error_low:Q", title=f"{aggregate_choice} AUROC", scale=alt.Scale(domain=[0, 1])),
                            x2="auroc_error_high:Q",
                            color=alt.value("#111827"),
                        )
                        + engine_auroc_base.mark_tick(color="#111827", size=14, thickness=2)
                        .encode(
                            x=alt.X(f"{auroc_metric_col}:Q", scale=alt.Scale(domain=[0, 1])),
                            color=alt.value("#111827"),
                        )
                        + engine_auroc_base.mark_text(
                            align="left",
                            baseline="bottom",
                            dx=5,
                            dy=-7,
                            color="#1f2937",
                            fontSize=11,
                            fontWeight="bold",
                        )
                        .encode(
                            x=alt.X(f"{auroc_metric_col}:Q", scale=alt.Scale(domain=[0, 1])),
                            text=alt.Text(f"{auroc_metric_col}:Q", format=".3f"),
                            color=alt.value("#1f2937"),
                        )
                    ).properties(height=engine_chart_height).interactive()
                    st.caption(f"Error bars show per-target {error_label}: IQR for median, SD for mean.")
                    st.subheader(f"Per-Target {aggregate_choice.title()} Performance By Engine")
                    engine_cols = st.columns(2)
                    with engine_cols[0]:
                        st.altair_chart(engine_ap_chart, width="stretch", key=f"{feature_chart_key}_per_target_engine_ap")
                    with engine_cols[1]:
                        st.altair_chart(engine_auroc_chart, width="stretch", key=f"{feature_chart_key}_per_target_engine_auroc")
                    target_grid = per_target_perf.dropna(subset=["average_precision", "auroc"]).copy()
                    if not target_grid.empty:
                        target_grid["engine_feature"] = (
                            target_grid["engine"].astype(str) + " | " + target_grid["feature"].astype(str)
                        )
                        target_order = sorted(target_grid["target_id"].astype(str).unique().tolist())
                        row_order = (
                            target_grid.groupby("engine_feature", as_index=False)
                            .agg(sort_ap=("average_precision", "median"), sort_auroc=("auroc", "median"))
                            .sort_values(["sort_ap", "sort_auroc"], ascending=[False, False])["engine_feature"]
                            .tolist()
                        )
                        heatmap_height = max(420, 42 * max(1, len(row_order)))
                        heatmap_width = max(640, 72 * max(1, len(target_order)))
                        ap_heatmap = (
                            alt.Chart(target_grid)
                            .mark_rect()
                            .encode(
                                x=alt.X(
                                    "target_id:N",
                                    title="target",
                                    sort=target_order,
                                    axis=alt.Axis(labelAngle=-60, labelOverlap=False),
                                ),
                                y=alt.Y(
                                    "engine_feature:N",
                                    title=None,
                                    sort=row_order,
                                    axis=alt.Axis(labelLimit=420, labelOverlap=False),
                                ),
                                color=alt.Color(
                                    "average_precision:Q",
                                    title="AP",
                                    scale=alt.Scale(domain=[0, 1], scheme="blues"),
                                ),
                                tooltip=[
                                    "target_id:N",
                                    "engine:N",
                                    "feature:N",
                                    alt.Tooltip("average_precision:Q", title="AP", format=".3f"),
                                    alt.Tooltip("auroc:Q", title="AUROC", format=".3f"),
                                    alt.Tooltip("records:Q", title="records"),
                                    alt.Tooltip("positive_count:Q", title="binders"),
                                    alt.Tooltip("negative_count:Q", title="nonbinders"),
                                ],
                            )
                            .properties(width=heatmap_width, height=heatmap_height)
                            .interactive()
                        )
                        auroc_heatmap = (
                            alt.Chart(target_grid)
                            .mark_rect()
                            .encode(
                                x=alt.X(
                                    "target_id:N",
                                    title="target",
                                    sort=target_order,
                                    axis=alt.Axis(labelAngle=-60, labelOverlap=False),
                                ),
                                y=alt.Y(
                                    "engine_feature:N",
                                    title=None,
                                    sort=row_order,
                                    axis=alt.Axis(labelLimit=420, labelOverlap=False),
                                ),
                                color=alt.Color(
                                    "auroc:Q",
                                    title="AUROC",
                                    scale=alt.Scale(domain=[0, 1], scheme="greens"),
                                ),
                                tooltip=[
                                    "target_id:N",
                                    "engine:N",
                                    "feature:N",
                                    alt.Tooltip("average_precision:Q", title="AP", format=".3f"),
                                    alt.Tooltip("auroc:Q", title="AUROC", format=".3f"),
                                    alt.Tooltip("records:Q", title="records"),
                                    alt.Tooltip("positive_count:Q", title="binders"),
                                    alt.Tooltip("negative_count:Q", title="nonbinders"),
                                ],
                            )
                            .properties(width=heatmap_width, height=heatmap_height)
                            .interactive()
                        )
                        st.subheader("Per-Target Performance Matrix")
                        matrix_cols = st.columns(2)
                        with matrix_cols[0]:
                            st.altair_chart(ap_heatmap, width="stretch", key=f"{feature_chart_key}_per_target_engine_ap_matrix")
                        with matrix_cols[1]:
                            st.altair_chart(
                                auroc_heatmap,
                                width="stretch",
                                key=f"{feature_chart_key}_per_target_engine_auroc_matrix",
                            )
                bar_height = st.slider(
                    "Per-protein plot height",
                    min_value=220,
                    max_value=800,
                    value=360,
                    step=20,
                    key=f"{chart_state_key}_per_target_plot_height_v1",
                )
                ap_base = alt.Chart(aggregate_perf).encode(
                    x=alt.X("feature:N", title="feature", sort="-y", axis=alt.Axis(labelAngle=-35)),
                    tooltip=[
                        "feature:N",
                        "direction:N",
                        alt.Tooltip("proteins:Q", title="proteins"),
                        alt.Tooltip("records:Q", title="records"),
                        alt.Tooltip("mean_average_precision:Q", title="mean AP", format=".3f"),
                        alt.Tooltip("sd_average_precision:Q", title="AP SD", format=".3f"),
                    ],
                )
                auroc_base = alt.Chart(aggregate_perf).encode(
                    x=alt.X("feature:N", title="feature", sort="-y", axis=alt.Axis(labelAngle=-35)),
                    tooltip=[
                        "feature:N",
                        "direction:N",
                        alt.Tooltip("proteins:Q", title="proteins"),
                        alt.Tooltip("records:Q", title="records"),
                        alt.Tooltip("mean_auroc:Q", title="mean AUROC", format=".3f"),
                        alt.Tooltip("sd_auroc:Q", title="AUROC SD", format=".3f"),
                    ],
                )
                ap_chart = (
                    ap_base.mark_bar(color="#0072B2", opacity=0.82)
                    .encode(y=alt.Y("mean_average_precision:Q", title="mean AP", scale=alt.Scale(domain=[0, 1])))
                    + ap_base.mark_rule(color="#111827", strokeWidth=2)
                    .encode(
                        y=alt.Y("ap_low:Q", title="mean AP", scale=alt.Scale(domain=[0, 1])),
                        y2="ap_high:Q",
                    )
                    + ap_base.mark_tick(color="#111827", size=16, thickness=2)
                    .encode(y=alt.Y("mean_average_precision:Q", scale=alt.Scale(domain=[0, 1])))
                ).properties(height=bar_height).interactive()
                auroc_chart = (
                    auroc_base.mark_bar(color="#009E73", opacity=0.82)
                    .encode(y=alt.Y("mean_auroc:Q", title="mean AUROC", scale=alt.Scale(domain=[0, 1])))
                    + auroc_base.mark_rule(color="#111827", strokeWidth=2)
                    .encode(
                        y=alt.Y("auroc_low:Q", title="mean AUROC", scale=alt.Scale(domain=[0, 1])),
                        y2="auroc_high:Q",
                    )
                    + auroc_base.mark_tick(color="#111827", size=16, thickness=2)
                    .encode(y=alt.Y("mean_auroc:Q", scale=alt.Scale(domain=[0, 1])))
                ).properties(height=bar_height).interactive()
                perf_cols = st.columns(2)
                with perf_cols[0]:
                    st.altair_chart(ap_chart, width="stretch", key=f"{feature_chart_key}_per_target_ap")
                with perf_cols[1]:
                    st.altair_chart(auroc_chart, width="stretch", key=f"{feature_chart_key}_per_target_auroc")
                with st.expander("Per-protein values", expanded=False):
                    st.dataframe(per_target_perf, hide_index=True, width="stretch")
        else:
            st.info("Select one or more features to summarize per-protein performance.")
    st.caption(f"Ranking direction for this feature: {direction}. `rank_score` is higher-is-better after applying that direction.")


STRUCTURE_ENGINE_TABLES = [
    ("AF3", "alphafast_af3", "alphafast_af3_metrics.csv"),
    ("AF2-IG", "af2_initial_guess", "af2_initial_guess_metrics.csv"),
    ("Boltz-2", "boltz2", "boltz2_initial_guess_metrics.csv"),
    ("BoltzGen Fold", "boltzgen_fold", "boltzgen_fold_metrics.csv"),
    ("ColabFold", "colabfold", "colabfold_metrics.csv"),
    ("ESMFold2", "esmfold2", "esmfold2_metrics.csv"),
    ("OpenFold-3", "openfold3", "openfold3_metrics.csv"),
    ("Protenix v0.5", "protenix", "protenix_metrics.csv"),
    ("Protenix v1", "protenix_v1", "protenix_v1_metrics.csv"),
    ("Protenix v2", "protenix_v2", "protenix_v2_metrics.csv"),
    ("RF3", "rf3", "rf3_metrics.csv"),
]

def _rank_overlay_color(index: int, total: int) -> str:
    anchors = ["#0072B2", "#009E73", "#E69F00", "#D55E00", "#CC79A7"]
    if total <= 1:
        return anchors[0]
    position = max(0.0, min(1.0, index / max(total - 1, 1)))
    scaled = position * (len(anchors) - 1)
    left = int(scaled)
    right = min(left + 1, len(anchors) - 1)
    fraction = scaled - left

    def _hex_to_rgb(value: str) -> tuple[int, int, int]:
        value = value.lstrip("#")
        return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)

    left_rgb = _hex_to_rgb(anchors[left])
    right_rgb = _hex_to_rgb(anchors[right])
    rgb = tuple(
        int(round(left_part + (right_part - left_part) * fraction))
        for left_part, right_part in zip(left_rgb, right_rgb)
    )
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"

STRUCTURE_ENGINE_METRIC_PDB_KEYS = {
    "alphafast_af3": "af3",
    "af2_initial_guess": "af2",
    "boltz2": "boltz2",
    "boltzgen_fold": "boltzgen_fold",
    "colabfold": "colab",
    "esmfold2": "esmfold2",
    "openfold3": "openfold3",
    "protenix": "protenix",
    "protenix_v1": "protenix_v1",
    "protenix_v2": "protenix_v2",
    "rf3": "rf3",
}


def _molstar_color(value: str) -> str:
    return "0x" + value.lstrip("#")


def _remap_moved_run_path(run_dir: Path, path: Path) -> Path | None:
    if not path.is_absolute():
        return None
    parts = path.parts
    if run_dir.name in parts:
        index = parts.index(run_dir.name)
        suffix = Path(*parts[index + 1 :])
        candidate = run_dir / suffix
        if candidate.exists():
            return candidate
    if "artifacts" in parts:
        index = parts.index("artifacts")
        suffix = Path(*parts[index + 1 :])
        candidate = run_dir / "artifacts" / suffix
        if candidate.exists():
            return candidate
    return None


def _metric_structure_path(benchmark_dir: Path, engine_key: str, design_id: str) -> Path | None:
    metric_key = STRUCTURE_ENGINE_METRIC_PDB_KEYS.get(engine_key)
    if not metric_key:
        return None
    for suffix in [".pdb", ".cif", ".mmcif"]:
        path = benchmark_dir / "predicted_metric_pdbs" / metric_key / f"{design_id}{suffix}"
        if path.exists():
            return path
    return None


def _viewer_structure_path(benchmark_dir: Path, engine_key: str, design_id: str) -> Path | None:
    metric_key = STRUCTURE_ENGINE_METRIC_PDB_KEYS.get(engine_key)
    if not metric_key:
        return None
    for suffix in [".pdb", ".cif", ".mmcif"]:
        path = benchmark_dir / "predicted_viewer_structures" / metric_key / f"{design_id}{suffix}"
        if path.exists():
            return path
    return None


def _preferred_structure_path(benchmark_dir: Path, engine_key: str, design_id: str) -> Path | None:
    return _viewer_structure_path(benchmark_dir, engine_key, design_id) or _metric_structure_path(
        benchmark_dir,
        engine_key,
        design_id,
    )


def _native_engine_prediction_path(
    run_dir: Path,
    engine_key: str,
    design_id: str,
    row: pd.Series | None = None,
) -> Path | None:
    """Find a native engine output when the staged viewer file is missing or a broken symlink."""
    safe_id = _candidate_structure_id(design_id)
    if engine_key != "af2_initial_guess":
        return None
    output_dir = (
        run_dir
        / "artifacts"
        / "engines"
        / "af2_initial_guess"
        / "artifacts"
        / "raw"
        / "af2_initial_guess"
        / "output"
        / "af2_initial_guess"
    )
    if not output_dir.exists():
        output_dir = run_dir / "artifacts" / "raw" / "af2_initial_guess" / "output" / "af2_initial_guess"
    if not output_dir.exists():
        return None
    pattern_groups = [
        f"{safe_id}_af2_initial_guess.pdb",
        f"{safe_id}_af2_initial_guess_model*.pdb",
        f"{safe_id}_af2ig_mt_tt.pdb",
        f"{safe_id}_af2_initial_guess_binder_model*.pdb",
    ]
    for pattern in pattern_groups:
        matches = sorted(path for path in output_dir.glob(pattern) if path.exists())
        if matches:
            return matches[0]
    return None


def _resolve_engine_structure_path(run_dir: Path, engine_key: str, value: object) -> Path | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    candidates = [path] if path.is_absolute() else [run_dir / path]
    remapped = _remap_moved_run_path(run_dir, path)
    if remapped is not None:
        candidates.insert(0, remapped)
    if not path.is_absolute():
        candidates.append(run_dir / "artifacts" / "engines" / engine_key / path)
    for candidate in candidates:
        if candidate.exists() and candidate.suffix.lower() in {".pdb", ".cif", ".mmcif"}:
            return candidate
    return None


def _child_engine_run_dirs(parent_run_dir: Path, engine_key: str) -> list[Path]:
    tool_names = {
        "af2_initial_guess": {"af2_initial_guess"},
        "boltz2": {"boltz2_initial_guess", "boltz2"},
        "boltzgen_fold": {"boltzgen_fold"},
        "rf3": {"rf3"},
        "openfold3": {"openfold3"},
        "protenix": {"protenix"},
        "protenix_v1": {"protenix_v1"},
        "protenix_v2": {"protenix_v2"},
    }.get(engine_key, {engine_key})
    child_dirs: list[Path] = []
    seen: set[Path] = set()

    metrics = read_json(parent_run_dir / "result.json").get("metrics")
    if isinstance(metrics, dict):
        for key in [
            f"{engine_key}_child_run",
            f"{engine_key}_initial_guess_child_run",
            "af2_initial_guess_child_run",
            "boltz2_initial_guess_child_run",
        ]:
            value = metrics.get(key)
            if not value:
                continue
            path = Path(str(value)).expanduser()
            if path.exists() and path not in seen:
                child_dirs.append(path)
                seen.add(path)

    for job in collect_jobs("refolding-validation", include_hidden=True):
        child_dir = Path(str(job.get("run_dir") or "")).expanduser()
        if not child_dir.exists() or child_dir in seen:
            continue
        metadata = read_json(child_dir / "metadata.json")
        if str(metadata.get("parent_run_id") or "") != parent_run_dir.name:
            continue
        if str(metadata.get("parent_run_dir") or "") and Path(str(metadata.get("parent_run_dir"))).name != parent_run_dir.name:
            continue
        input_json = read_json(child_dir / "input.json")
        tool = str(input_json.get("tool") or metadata.get("tool") or "")
        if tool not in tool_names:
            continue
        child_dirs.append(child_dir)
        seen.add(child_dir)
    return child_dirs


def _linked_prediction_run_dirs(run_dir: Path, engine_key: str | None = None) -> list[Path]:
    linked: list[Path] = []
    seen: set[Path] = set()
    payloads = [
        read_json(run_dir / "metadata.json"),
        read_json(run_dir / "result.json"),
        read_json(run_dir / "input.json"),
    ]
    for payload in payloads:
        candidates: list[dict[str, Any]] = []
        for key in ("linked_prediction_runs", "linked_refolding_runs"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates.extend(item for item in value if isinstance(item, dict))
        for container_key in ("metrics", "outputs", "downstream_artifacts", "inputs", "params"):
            container = payload.get(container_key)
            if not isinstance(container, dict):
                continue
            for key in ("linked_prediction_runs", "linked_refolding_runs"):
                value = container.get(key)
                if isinstance(value, list):
                    candidates.extend(item for item in value if isinstance(item, dict))
        for item in candidates:
            item_engine = str(item.get("engine_key") or item.get("engine") or item.get("tool") or "").strip()
            if engine_key and item_engine and item_engine != engine_key:
                continue
            path_text = str(item.get("run_dir") or item.get("path") or "").strip()
            if not path_text:
                continue
            path = Path(path_text).expanduser()
            if not path.is_absolute():
                path = (run_dir / path).resolve()
            if path.exists() and path not in seen:
                linked.append(path)
                seen.add(path)
    return linked


def _collection_source_run_dir(run_dir: Path, source_run_id: object) -> Path | None:
    source_id = str(source_run_id or "").strip()
    if not source_id:
        return None
    sibling = run_dir.parent / source_id
    if sibling.exists():
        return sibling
    for job in collect_jobs("benchmark", include_hidden=True):
        if str(job.get("run_id") or "") != source_id:
            continue
        path = Path(str(job.get("run_dir") or "")).expanduser()
        if path.exists():
            return path
    return None


def _resolve_child_engine_structure_path(parent_run_dir: Path, engine_key: str, value: object) -> Path | None:
    for child_dir in [*_linked_prediction_run_dirs(parent_run_dir, engine_key), *_child_engine_run_dirs(parent_run_dir, engine_key)]:
        path = _resolve_engine_structure_path(child_dir, engine_key, value)
        if path is not None:
            return path
    return None


def _resolve_collection_engine_structure_path(
    run_dir: Path,
    *,
    engine_label: str,
    engine_key: str,
    value: object,
) -> Path | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    sources = _read_csv(run_dir / "artifacts" / "benchmark" / "benchmark_collection_sources.csv")
    if sources is None or sources.empty or "run_id" not in sources.columns or "engines" not in sources.columns:
        return None
    path = Path(text)
    for _, source_row in sources.iterrows():
        engines = {
            item.strip()
            for item in str(source_row.get("engines") or "").split(",")
            if item.strip()
        }
        if engine_label not in engines:
            continue
        source_run_id = str(source_row.get("run_id") or "").strip()
        if not source_run_id:
            continue
        source_run_dir = _collection_source_run_dir(run_dir, source_run_id)
        if source_run_dir is None:
            continue
        candidates = [path] if path.is_absolute() else [source_run_dir / path]
        if not path.is_absolute():
            candidates.append(source_run_dir / "artifacts" / "engines" / engine_key / path)
        for candidate in candidates:
            if candidate.exists() and candidate.suffix.lower() in {".pdb", ".cif", ".mmcif"}:
                return candidate
    return None


def _candidate_structure_id(value: object) -> str:
    text = str(value or "")
    for suffix in [
        "_esmfold2bm_ig",
        "_af2ig_mt_tt",
        "_boltz2ig_tt",
        "_complex_protenix",
        "_complex_rf3",
        "_complex_boltzgen_fold",
    ]:
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _alphafast_model_from_summary(run_dir: Path, value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    summary_path = Path(text)
    if not summary_path.is_absolute():
        summary_path = run_dir / summary_path
    if not summary_path.exists():
        return None
    model_path = summary_path.with_name(summary_path.name.replace("_summary_confidences.json", "_model.cif"))
    return model_path if model_path.exists() else None


def _colabfold_model_for_design(run_dir: Path, design_id: str) -> Path | None:
    root = run_dir / "artifacts" / "engines" / "colabfold"
    if not root.exists():
        root = run_dir / "artifacts" / "raw" / "colabfold"
    if not root.exists():
        return None
    matches = sorted(root.glob(f"**/{design_id}*.pdb")) + sorted(root.glob(f"**/{design_id}*.cif"))
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda path: (
            0 if "rank_001" in path.name or "rank_1" in path.name else 1,
            path.name,
        ),
    )[0]


def _chain_map_for_engine_design(run_dir: Path, engine_key: str, design_id: str) -> dict[str, Any]:
    design = str(design_id or "").strip()
    if not design:
        return {}
    metric_key = STRUCTURE_ENGINE_METRIC_PDB_KEYS.get(engine_key)
    benchmark_root = run_dir / "artifacts" / "benchmark"
    engine_root = run_dir / "artifacts" / "engines" / engine_key
    candidates = [
        benchmark_root / "predicted_metric_pdbs" / str(metric_key or "") / f"{design}.chain_map.json",
        engine_root / f"{design}.chain_map.json",
        engine_root / "artifacts" / "raw" / engine_key / "inputs" / f"{design}.chain_map.json",
        engine_root / "artifacts" / "raw" / "boltz2_initial_guess" / "input_pdbs" / f"{design}.chain_map.json",
        engine_root / "artifacts" / "raw" / "af2_initial_guess" / "inputs" / f"{design}.chain_map.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return read_json(candidate)
    if not engine_root.exists():
        return {}
    matches = sorted(engine_root.glob(f"**/{design}.chain_map.json"))
    return read_json(matches[0]) if matches else {}


def _collection_esmfold2_sources(run_dir: Path, benchmark_dir: Path) -> list[tuple[str, Path, str, str]]:
    sources = _read_csv(benchmark_dir / "benchmark_collection_sources.csv")
    if sources is None or sources.empty or "engine_variants" not in sources.columns or "run_id" not in sources.columns:
        return []
    rows: list[tuple[str, Path, str, str]] = []
    for _, row in sources.iterrows():
        variants = str(row.get("engine_variants") or "")
        if "ESMFold2:" not in variants:
            continue
        token = ""
        for item in variants.split(","):
            item = item.strip()
            if item.startswith("ESMFold2:"):
                token = item.split(":", 1)[1].strip()
                break
        if not token:
            continue
        source_run_id = str(row.get("run_id") or "").strip()
        if not source_run_id:
            continue
        source_run_dir = _collection_source_run_dir(run_dir, source_run_id)
        if source_run_dir is None:
            continue
        rows.append((source_run_id, source_run_dir, token, _esmfold2_variant_engine_label(token)))
    return rows if len(rows) > 1 else []


def _collection_engine_metric_sources(
    run_dir: Path,
    benchmark_dir: Path,
    *,
    engine_label: str,
    csv_name: str,
) -> list[tuple[str, Path, Path, pd.DataFrame]]:
    sources = _read_csv(benchmark_dir / "benchmark_collection_sources.csv")
    if sources is None or sources.empty or "run_id" not in sources.columns or "engines" not in sources.columns:
        return []
    rows: list[tuple[str, Path, Path, pd.DataFrame]] = []
    for _, source_row in sources.iterrows():
        engines = {
            item.strip()
            for item in str(source_row.get("engines") or "").split(",")
            if item.strip()
        }
        if engine_label not in engines:
            continue
        source_run_id = str(source_row.get("run_id") or "").strip()
        if not source_run_id:
            continue
        source_run_dir = _collection_source_run_dir(run_dir, source_run_id)
        if source_run_dir is None:
            continue
        source_benchmark_dir = source_run_dir / "artifacts" / "benchmark"
        source_table = _read_csv(source_benchmark_dir / csv_name)
        if source_table is None or source_table.empty:
            continue
        rows.append((source_run_id, source_run_dir, source_benchmark_dir, source_table))
    return rows


def _engine_target_chains_from_map(chain_map: dict[str, Any], original_target_chains: list[str]) -> list[str]:
    original_set = {str(chain) for chain in original_target_chains if str(chain)}
    fragment_rows = chain_map.get("target_fragments") if isinstance(chain_map.get("target_fragments"), list) else []
    fragment_mapped = [
        str(row.get("engine_chain"))
        for row in fragment_rows
        if isinstance(row, dict)
        and row.get("engine_chain")
        and (
            not original_set
            or str(row.get("source_chain") or "") in original_set
            or str(row.get("declared_chain") or "") in original_set
        )
    ]
    if fragment_mapped:
        return list(dict.fromkeys(fragment_mapped))
    target_rows = chain_map.get("targets") if isinstance(chain_map.get("targets"), list) else []
    by_original = {
        str(row.get("original_chain")): str(row.get("engine_chain"))
        for row in target_rows
        if isinstance(row, dict) and row.get("original_chain") and row.get("engine_chain")
    }
    mapped = [by_original[str(chain)] for chain in original_target_chains if str(chain) in by_original]
    if mapped:
        return mapped
    return [
        str(row.get("engine_chain"))
        for row in target_rows
        if isinstance(row, dict) and row.get("engine_chain")
    ]


def _benchmark_structure_records(
    run_dir: Path,
    benchmark_dir: Path,
    *,
    allow_input_fallback: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    esmfold2_collection_sources = _collection_esmfold2_sources(run_dir, benchmark_dir)
    esmfold2_variant_rows: list[dict[str, object]] = []
    for source_run_id, source_run_dir, variant_token, variant_label in esmfold2_collection_sources:
        source_benchmark_dir = source_run_dir / "artifacts" / "benchmark"
        table = _read_csv(source_benchmark_dir / "esmfold2_metrics.csv")
        if table is None or table.empty:
            continue
        for _, row in table.iterrows():
            design_id = str(row.get("binder_id") or _candidate_structure_id(row.get("candidate_id")) or "").strip()
            if not design_id:
                continue
            path = _preferred_structure_path(source_benchmark_dir, "esmfold2", design_id)
            if path is None and allow_input_fallback and "complex_pdb" in table.columns:
                path = path or _resolve_engine_structure_path(source_run_dir, "esmfold2", row.get("complex_pdb"))
            if path is None:
                continue
            chain_map = _chain_map_for_engine_design(source_run_dir, "esmfold2", design_id)
            esmfold2_variant_rows.append(
                {
                    "binder_id": design_id,
                    "candidate_id": row.get("candidate_id") or design_id,
                    "engine": variant_label,
                    "engine_key": "esmfold2",
                    "engine_variant": variant_token,
                    "source_run_id": source_run_id,
                    "path": str(path),
                    "chain_map": chain_map,
                    "target_id": row.get("target_id"),
                    "label": row.get("label") if "label" in table.columns else row.get("binder"),
                }
            )
    rows.extend(esmfold2_variant_rows)
    seen_structure_keys = {
        (str(record.get("engine") or ""), str(record.get("binder_id") or ""))
        for record in rows
    }
    linked_engine_sources: list[tuple[str, str, str, Path, pd.DataFrame]] = []
    for label, _prefix, engine_key in SEQUENCE_VALIDATION_ENGINES:
        for linked_run_dir in _linked_prediction_run_dirs(run_dir, engine_key):
            candidates_path = linked_run_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl"
            if not candidates_path.exists():
                continue
            candidate_rows: list[dict[str, object]] = []
            try:
                with candidates_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        try:
                            candidate = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        candidate_rows.append(candidate)
            except OSError:
                continue
            if candidate_rows:
                linked_engine_sources.append(
                    (label, engine_key, linked_run_dir.name, linked_run_dir, pd.DataFrame(candidate_rows))
                )
    for engine_label, engine_key, csv_name in STRUCTURE_ENGINE_TABLES:
        if engine_key == "esmfold2" and esmfold2_variant_rows:
            continue
        table = _read_csv(benchmark_dir / csv_name)
        table_sources: list[tuple[str, Path, Path, pd.DataFrame]] = []
        if table is not None and not table.empty:
            table_sources.append(("", run_dir, benchmark_dir, table))
        for _linked_label, linked_engine_key, linked_run_id, linked_run_dir, linked_table in linked_engine_sources:
            if linked_engine_key == engine_key:
                table_sources.append((linked_run_id, linked_run_dir, linked_run_dir / "artifacts" / "benchmark", linked_table))
        if True:
            table_sources.extend(
                _collection_engine_metric_sources(
                    run_dir,
                    benchmark_dir,
                    engine_label=engine_label,
                    csv_name=csv_name,
                )
            )
        for source_run_id, source_run_dir, source_benchmark_dir, source_table in table_sources:
            for _, row in source_table.iterrows():
                design_id = str(row.get("binder_id") or _candidate_structure_id(row.get("candidate_id")) or "").strip()
                if not design_id:
                    continue
                structure_key = (engine_label, design_id)
                if structure_key in seen_structure_keys:
                    continue
                path = _preferred_structure_path(source_benchmark_dir, engine_key, design_id)
                if path is None:
                    path = _native_engine_prediction_path(source_run_dir, engine_key, design_id, row)
                if path is None and allow_input_fallback and "complex_pdb" in source_table.columns:
                    path = path or _resolve_engine_structure_path(source_run_dir, engine_key, row.get("complex_pdb"))
                if path is None and allow_input_fallback and source_run_dir == run_dir and "complex_pdb" in source_table.columns:
                    path = _resolve_child_engine_structure_path(run_dir, engine_key, row.get("complex_pdb"))
                if path is None and source_run_dir == run_dir and "complex_pdb" in source_table.columns:
                    path = _resolve_collection_engine_structure_path(
                        run_dir,
                        engine_label=engine_label,
                        engine_key=engine_key,
                        value=row.get("complex_pdb"),
                    )
                if path is None and engine_key == "alphafast_af3" and "summary_confidences" in source_table.columns:
                    path = _alphafast_model_from_summary(source_run_dir, row.get("summary_confidences"))
                if path is None and engine_key == "colabfold":
                    path = _colabfold_model_for_design(source_run_dir, design_id)
                if path is None:
                    continue
                chain_map = _chain_map_for_engine_design(source_run_dir, engine_key, design_id)
                rows.append(
                    {
                        "binder_id": design_id,
                        "candidate_id": row.get("candidate_id") or design_id,
                        "engine": engine_label,
                        "engine_key": engine_key,
                        "source_run_id": source_run_id,
                        "path": str(path),
                        "chain_map": chain_map,
                        "target_id": row.get("target_id"),
                        "label": row.get("label") if "label" in source_table.columns else row.get("binder"),
                    }
                )
                seen_structure_keys.add(structure_key)
    return pd.DataFrame(rows)


def _candidate_source_keys_for_design(
    run_dir: Path,
    design_id: str,
    benchmark_dir: Path | None = None,
) -> set[str]:
    keys = {str(design_id or "").strip().lower()}
    csv_paths = [
        run_dir / "artifacts" / "queued_inputs" / "candidate_repo_dataset" / "input.csv",
        run_dir / "artifacts" / "queued_inputs" / "input.csv",
        run_dir / "artifacts" / "raw" / "de_novo_binder_scoring" / "output" / "run.csv",
        run_dir / "artifacts" / "raw" / "de_novo_binder_scoring" / "dataset" / "input.csv",
    ]
    if benchmark_dir is not None:
        csv_paths.append(benchmark_dir / "merged_benchmark_metrics.csv")
    for csv_path in csv_paths:
        table = _read_csv(csv_path)
        if table is None or table.empty or "binder_id" not in table.columns:
            continue
        subset = table[table["binder_id"].astype(str).str.strip().str.lower() == str(design_id or "").strip().lower()]
        if subset.empty:
            continue
        for column in ["binder_id", "candidate_id", "original_binder_id"]:
            if column not in subset.columns:
                continue
            for value in subset[column].dropna().tolist():
                text = str(value).strip().lower()
                if text:
                    keys.add(text)
    return keys


def _benchmark_dataset_row_for_design(
    run_dir: Path,
    design_id: str,
    benchmark_dir: Path | None = None,
) -> pd.Series | None:
    design_keys = _candidate_source_keys_for_design(run_dir, design_id, benchmark_dir)
    csv_paths = [
        run_dir / "artifacts" / "queued_inputs" / "candidate_repo_dataset" / "input.csv",
        run_dir / "artifacts" / "queued_inputs" / "input.csv",
        run_dir / "artifacts" / "raw" / "de_novo_binder_scoring" / "output" / "run.csv",
        run_dir / "artifacts" / "raw" / "de_novo_binder_scoring" / "dataset" / "input.csv",
    ]
    if benchmark_dir is not None:
        csv_paths.append(benchmark_dir / "merged_benchmark_metrics.csv")
    for csv_path in csv_paths:
        table = _read_csv(csv_path)
        if table is None or table.empty:
            continue
        id_columns = [
            column
            for column in ["binder_id", "candidate_id", "original_binder_id"]
            if column in table.columns
        ]
        if not id_columns:
            continue
        mask = pd.Series(False, index=table.index)
        for column in id_columns:
            mask |= table[column].astype(str).str.strip().str.lower().isin(design_keys)
        subset = table[mask]
        if not subset.empty:
            return subset.iloc[0]
    return None


def _target_chains_from_benchmark_row(row: pd.Series | None) -> list[str]:
    if row is None:
        return []
    target_chains: list[str] = []
    for column in ["target_chains", "engine_target_chains", "refolding_target_source_chains", "source_input_target_chains"]:
        if column not in row.index:
            continue
        target_chains = _parse_chain_list(row.get(column))
        if target_chains:
            break
    is_target_only_refolding = (
        ("capacity_target_only" in row.index and _truthy_bool_value(row.get("capacity_target_only")))
        or ("source" in row.index and _is_target_refolding_input_builder(row.get("source")))
    )
    if is_target_only_refolding:
        for column in ["binder_chains", "source_input_binder_chains", "source_chain", "binder_chain"]:
            if column not in row.index:
                continue
            binder_chains = _parse_chain_list(row.get(column))
            if binder_chains:
                return _unique_chain_list(binder_chains, target_chains)
    return target_chains


def _case_insensitive_structure_match(directory: Path | None, design_keys: set[str]) -> Path | None:
    if directory is None or not directory.is_dir():
        return None
    normalized_keys = {str(key).strip().lower() for key in design_keys if str(key).strip()}
    if not normalized_keys:
        return None
    try:
        candidates = sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".pdb", ".cif", ".mmcif"}
        )
    except OSError:
        return None
    for path in candidates:
        if path.stem.strip().lower() in normalized_keys:
            return path
    return None


def _benchmark_input_complex_for_design(
    run_dir: Path,
    design_id: str,
    benchmark_dir: Path | None = None,
) -> Path | None:
    design_keys = _candidate_source_keys_for_design(run_dir, design_id, benchmark_dir)
    input_json = read_json(run_dir / "input.json")
    inputs = input_json.get("inputs") if isinstance(input_json.get("inputs"), dict) else {}
    external_input_dir = inputs.get("input_pdb_dir")
    directories = [
        run_dir / "artifacts" / "raw" / "de_novo_binder_scoring" / "dataset" / "input_pdbs",
        run_dir / "artifacts" / "raw" / "de_novo_binder_scoring" / "output" / "input_pdbs",
        Path(str(external_input_dir)) if external_input_dir else None,
    ]
    for directory in directories:
        match = _case_insensitive_structure_match(directory, design_keys)
        if match is not None:
            return match
    return None


def _source_input_reference_for_design(
    run_dir: Path,
    design_id: str,
    benchmark_dir: Path | None = None,
) -> tuple[Path | None, list[str], str | None]:
    design_keys = _candidate_source_keys_for_design(run_dir, design_id, benchmark_dir)
    staged_csv = run_dir / "artifacts" / "queued_inputs" / "candidate_repo_dataset" / "input.csv"
    staged_table = _read_csv(staged_csv)
    if staged_table is not None and not staged_table.empty and "source_input_complex_pdb" in staged_table.columns:
        id_columns = [
            column
            for column in ["binder_id", "candidate_id", "original_binder_id"]
            if column in staged_table.columns
        ]
        staged_mask = pd.Series(False, index=staged_table.index)
        for column in id_columns:
            staged_mask |= staged_table[column].astype(str).str.strip().str.lower().isin(design_keys)
        staged_rows = staged_table[staged_mask]
        if not staged_rows.empty:
            staged_row = staged_rows.iloc[0]
            source_path = _resolve_run_relative_path(run_dir, staged_row.get("source_input_complex_pdb"))
            source_chains = _parse_chain_list(staged_row.get("source_input_target_chains"))
            if _truthy_bool_value(staged_row.get("capacity_target_only")) or _is_target_refolding_input_builder(
                staged_row.get("source")
            ):
                source_binder_chains: list[str] = []
                for column in ["source_input_binder_chains", "binder_chains", "source_chain", "binder_chain"]:
                    if column not in staged_row.index:
                        continue
                    source_binder_chains = _parse_chain_list(staged_row.get(column))
                    if source_binder_chains:
                        break
                source_chains = _unique_chain_list(source_binder_chains, source_chains)
            if source_path is not None:
                return source_path, source_chains, str(staged_csv)

    input_json = read_json(run_dir / "input.json")
    inputs = input_json.get("inputs") if isinstance(input_json.get("inputs"), dict) else {}
    source_run_dir = _resolve_run_relative_path(run_dir, inputs.get("source_run_dir"))
    if source_run_dir is None and inputs.get("source_run_dir"):
        source_candidate = Path(str(inputs.get("source_run_dir")))
        if source_candidate.exists():
            source_run_dir = source_candidate

    candidates_jsonl = _resolve_run_relative_path(run_dir, inputs.get("candidates_jsonl"))
    if candidates_jsonl is None and source_run_dir is not None:
        candidates_jsonl = source_run_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl"
    if candidates_jsonl is None or not candidates_jsonl.exists():
        benchmark_input = _benchmark_input_complex_for_design(run_dir, design_id, benchmark_dir)
        benchmark_chains = _target_chains_from_benchmark_row(
            _benchmark_dataset_row_for_design(run_dir, design_id, benchmark_dir)
        )
        if benchmark_input is not None:
            return benchmark_input, benchmark_chains, str(run_dir / "input.json")
        return None, [], None

    try:
        with candidates_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                candidate_id = str(row.get("candidate_id") or "").strip().lower()
                original_id = str(row.get("original_binder_id") or "").strip().lower()
                row_keys = {candidate_id, original_id}
                raw_metadata = row.get("raw_metadata") if isinstance(row.get("raw_metadata"), dict) else {}
                original_path_name = Path(str(raw_metadata.get("original_path") or "")).stem.lower()
                if original_path_name:
                    row_keys.add(original_path_name)
                if not (design_keys & row_keys):
                    continue
                target_chains = _parse_chain_list(row.get("target_chains"))
                raw_metadata = row.get("raw_metadata") if isinstance(row.get("raw_metadata"), dict) else {}
                if _truthy_bool_value(raw_metadata.get("capacity_target_only")) or _truthy_bool_value(
                    row.get("capacity_target_only")
                ) or _is_target_refolding_input_builder(row.get("source")) or _is_target_refolding_input_builder(
                    raw_metadata.get("source")
                ):
                    binder_chains: list[str] = []
                    for column in ["binder_chains", "source_input_binder_chains", "source_chain", "binder_chain"]:
                        binder_chains = _parse_chain_list(row.get(column)) or _parse_chain_list(raw_metadata.get(column))
                        if binder_chains:
                            break
                    target_chains = _unique_chain_list(binder_chains, target_chains)
                for column in ["complex_pdb"]:
                    if source_run_dir is not None:
                        source_path = _resolve_run_relative_path(source_run_dir, row.get(column))
                        if source_path is not None:
                            return source_path, target_chains, str(candidates_jsonl)
                    source_path = _resolve_run_relative_path(run_dir, row.get(column))
                    if source_path is not None:
                        return source_path, target_chains, str(candidates_jsonl)
    except OSError:
        return None, [], None
    benchmark_input = _benchmark_input_complex_for_design(run_dir, design_id, benchmark_dir)
    if benchmark_input is not None:
        benchmark_chains = _target_chains_from_benchmark_row(
            _benchmark_dataset_row_for_design(run_dir, design_id, benchmark_dir)
        )
        return benchmark_input, benchmark_chains, str(run_dir / "input.json")
    return None, [], str(candidates_jsonl)


def _reference_structure_for_design(run_dir: Path, benchmark_dir: Path, design_id: str) -> Path | None:
    source_reference, _source_target_chains, _source_metadata = _source_input_reference_for_design(
        run_dir, design_id, benchmark_dir
    )
    if source_reference is not None:
        return source_reference
    reference_sources = [
        (
            benchmark_dir / "merged_benchmark_metrics.csv",
            ["input_pdb", "complex_pdb", "monomer_refolding_reference", "reference_pdb"],
        ),
        (
            run_dir / "artifacts" / "queued_inputs" / "candidate_repo_dataset" / "input.csv",
            ["input_pdb", "complex_pdb"],
        ),
        (
            run_dir / "artifacts" / "raw" / "de_novo_binder_scoring" / "dataset" / "input.csv",
            ["input_pdb", "complex_pdb"],
        ),
        (
            benchmark_dir / "boltz2_initial_guess_metrics.csv",
            ["monomer_refolding_reference", "reference_pdb"],
        ),
        (
            benchmark_dir / "af2_initial_guess_metrics.csv",
            ["monomer_refolding_reference", "reference_pdb"],
        ),
    ]
    design_key = str(design_id).strip().lower()
    for csv_path, columns in reference_sources:
        table = _read_csv(csv_path)
        if table is None or table.empty or "binder_id" not in table.columns:
            continue
        subset = table[table["binder_id"].astype(str).str.strip().str.lower() == design_key]
        if subset.empty:
            continue
        for column in columns:
            if column not in subset.columns:
                continue
            path = _resolve_engine_structure_path(run_dir, "input", subset.iloc[0].get(column))
            if path is not None:
                return path
    for candidate in [
        run_dir / "artifacts" / "raw" / "de_novo_binder_scoring" / "output" / "input_pdbs" / f"{design_id}.pdb",
        run_dir / "artifacts" / "benchmark" / "input_pdbs" / f"{design_id}.pdb",
    ]:
        if candidate.exists():
            return candidate
    return None


def _resolve_run_relative_path(run_dir: Path, value: object) -> Path | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    candidates = [path] if path.is_absolute() else [run_dir / path]
    remapped = _remap_moved_run_path(run_dir, path)
    if remapped is not None:
        candidates.insert(0, remapped)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _path_label(path: Path | None, run_dir: Path) -> str:
    if path is None:
        return "n/a"
    runs_root = run_dir.parents[1] if len(run_dir.parents) >= 2 else None
    for root in [run_dir, runs_root]:
        if root is None:
            continue
        try:
            return str(path.relative_to(root))
        except ValueError:
            continue
    parts = path.parts
    if "runs" in parts:
        index = parts.index("runs")
        return str(Path(*parts[index:]))
    return str(path)


def _target_pdb_reference_for_design(run_dir: Path, design_id: str) -> Path | None:
    benchmark_dir = run_dir / "artifacts" / "benchmark"
    for csv_name in ["merged_benchmark_metrics.csv", "af2_initial_guess_metrics.csv", "boltz2_initial_guess_metrics.csv"]:
        table = _read_csv(benchmark_dir / csv_name)
        if table is None or table.empty or "binder_id" not in table.columns:
            continue
        subset = table[table["binder_id"].astype(str) == str(design_id)]
        if subset.empty:
            continue
        for column in ["target_pdb"]:
            if column not in subset.columns:
                continue
            target = _resolve_run_relative_path(run_dir, subset.iloc[0].get(column))
            if target is not None:
                return target
    for candidate in [
        run_dir / "artifacts" / "queued_inputs" / "candidate_repo_dataset" / "target_pdbs" / f"{design_id}.pdb",
    ]:
        if candidate.exists():
            return candidate

    input_json = read_json(run_dir / "input.json")
    inputs = input_json.get("inputs") if isinstance(input_json.get("inputs"), dict) else {}
    direct_target = _resolve_run_relative_path(run_dir, inputs.get("target_pdb"))
    if direct_target is not None:
        return direct_target

    source_input, source_target_chains, _source_metadata = _source_input_reference_for_design(
        run_dir,
        design_id,
        benchmark_dir,
    )
    if source_input is not None and source_target_chains:
        # Some benchmark datasets keep target and binder in one source PDB.
        # Target PDB mode renders only the declared target chain(s).
        return source_input

    predicted_input = _case_insensitive_structure_match(
        benchmark_dir / "predicted_metric_pdbs" / "input",
        _candidate_source_keys_for_design(run_dir, design_id, benchmark_dir),
    )
    if predicted_input is not None:
        return predicted_input

    source_run_dir = _resolve_run_relative_path(run_dir, inputs.get("source_run_dir"))
    if source_run_dir is None and inputs.get("source_run_dir"):
        source_candidate = Path(str(inputs.get("source_run_dir")))
        if source_candidate.exists():
            source_run_dir = source_candidate

    candidates_jsonl = _resolve_run_relative_path(run_dir, inputs.get("candidates_jsonl"))
    if candidates_jsonl is None and source_run_dir is not None:
        candidates_jsonl = source_run_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl"
    if candidates_jsonl is None or not candidates_jsonl.exists():
        return None

    design_text = str(design_id or "").strip()
    design_keys = {design_text, design_text.lower()}
    try:
        with candidates_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                candidate_id = str(row.get("candidate_id") or "").strip()
                original_id = str(row.get("original_binder_id") or "").strip()
                if (
                    candidate_id not in design_keys
                    and candidate_id.lower() not in design_keys
                    and original_id not in design_keys
                    and original_id.lower() not in design_keys
                ):
                    continue
                if source_run_dir is not None:
                    target = _resolve_run_relative_path(source_run_dir, row.get("target_pdb"))
                    if target is not None:
                        return target
                target = _resolve_run_relative_path(run_dir, row.get("target_pdb"))
                if target is not None:
                    return target
    except OSError:
        return None
    return None


def _parse_chain_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, (list, tuple, set)):
            return [str(item).strip() for item in parsed if str(item).strip()]
        if isinstance(parsed, str) and parsed.strip():
            return [parsed.strip()]
    cleaned = text.strip("[](){}")
    return [part.strip().strip("'\"") for part in cleaned.split(",") if part.strip().strip("'\"")]


def _candidate_export_rel_path(run_dir: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return str(path)


def _jsonable_value(value: object) -> object:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return [_jsonable_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    return str(value)


def _create_candidate_set_from_ranked_results(
    *,
    parent_run_dir: Path,
    records: pd.DataFrame,
    metrics: pd.DataFrame,
    ranked_table: pd.DataFrame,
    selected_engine: str,
    selected_feature: str,
    direction: str,
    filter_config: dict[str, Any] | None,
    export_name: str,
    max_candidates: int,
) -> Path:
    if ranked_table.empty:
        raise ValueError("No filtered ranked designs are available to export.")
    if metrics is None or metrics.empty or "binder_id" not in metrics.columns:
        raise ValueError("Merged benchmark metrics are required to export a candidate set.")
    ranked = ranked_table.copy()
    ranked["binder_id"] = ranked["binder_id"].astype(str)
    if max_candidates > 0:
        ranked = ranked.head(int(max_candidates))
    if ranked.empty:
        raise ValueError("No ranked designs remain after the max-candidate limit.")
    selected_ids = ranked["binder_id"].astype(str).tolist()
    engine_records = records[
        records["engine"].astype(str).eq(str(selected_engine))
        & records["binder_id"].astype(str).isin(selected_ids)
    ].copy()
    if engine_records.empty:
        raise ValueError(f"No predicted structures were found for {selected_engine} in the filtered ranked designs.")
    order = {binder_id: index for index, binder_id in enumerate(selected_ids)}
    engine_records["_export_order"] = engine_records["binder_id"].astype(str).map(order)
    engine_records = engine_records.sort_values("_export_order").drop_duplicates("binder_id", keep="first")

    metrics_by_binder = (
        metrics.assign(_binder_id=metrics["binder_id"].astype(str))
        .drop_duplicates("_binder_id", keep="first")
        .set_index("_binder_id")
    )
    ranked_by_binder = (
        ranked.assign(_binder_id=ranked["binder_id"].astype(str))
        .drop_duplicates("_binder_id", keep="first")
        .set_index("_binder_id")
    )
    job = create_job(
        "candidate-import",
        job_type="result_selection_export",
        tool="ranked_result_selection",
        inputs={
            "source_run_dir": str(parent_run_dir),
            "selected_engine": selected_engine,
            "selected_feature": selected_feature,
        },
        params={
            "export_name": export_name,
            "max_candidates": int(max_candidates or 0),
            "direction": direction,
            "filter_config": _jsonable_value(filter_config or {}),
        },
    )
    structure_dir = job.run_dir / "artifacts" / "selected_structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    summary_rows: list[dict[str, object]] = []
    for _, record in engine_records.iterrows():
        binder_id = str(record.get("binder_id") or "").strip()
        if not binder_id:
            continue
        metric_row = metrics_by_binder.loc[binder_id].to_dict() if binder_id in metrics_by_binder.index else {}
        rank_row = ranked_by_binder.loc[binder_id].to_dict() if binder_id in ranked_by_binder.index else {}
        source_complex = Path(str(record.get("path") or ""))
        if not source_complex.exists():
            continue
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", binder_id).strip("_") or f"candidate_{len(candidates) + 1}"
        complex_path = structure_dir / f"{safe_id}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', selected_engine).strip('_').lower()}_complex{source_complex.suffix or '.pdb'}"
        shutil.copy2(source_complex, complex_path)
        target_source_text = str(metric_row.get("source_target_pdb") or metric_row.get("target_pdb") or "").strip()
        target_path = None
        if target_source_text:
            source_target = Path(target_source_text)
            if source_target.exists():
                target_path = structure_dir / f"{safe_id}_target{source_target.suffix or '.pdb'}"
                if not target_path.exists():
                    shutil.copy2(source_target, target_path)
        metric_payload = {
            str(key): _jsonable_value(value)
            for key, value in metric_row.items()
            if str(key) not in {"_binder_id"}
        }
        metric_payload.update(
            {
                "result_selection_rank": _jsonable_value(rank_row.get("rank")),
                "result_selection_feature": selected_feature,
                "result_selection_feature_value": _jsonable_value(rank_row.get("feature_value") or rank_row.get(selected_feature)),
                "result_selection_rank_score": _jsonable_value(rank_row.get("rank_score")),
                "result_selection_engine": selected_engine,
                "result_selection_direction": direction,
            }
        )
        target_chains = _parse_chain_list(metric_row.get("target_chains"))
        binder_sequence = str(
            metric_row.get("binder_sequence")
            or metric_row.get("A_seq")
            or metric_row.get("sequence")
            or ""
        ).strip()
        original_id = str(metric_row.get("original_binder_id") or binder_id).strip()
        candidates.append(
            {
                "candidate_id": binder_id,
                "stage": STAGE_COMPLEX_REFOLDING,
                "source_tool": "ranked_result_selection",
                "target_pdb": _candidate_export_rel_path(job.run_dir, target_path),
                "complex_pdb": _candidate_export_rel_path(job.run_dir, complex_path),
                "binder_pdb": None,
                "binder_sequence": binder_sequence,
                "target_chains": target_chains,
                "binder_chains": ["A"],
                "binder_length": str(len(binder_sequence)) if binder_sequence else None,
                "metrics": metric_payload,
                "parents": [original_id] if original_id else [],
                "raw_metadata": {
                    "export_name": export_name,
                    "source_run_dir": str(parent_run_dir),
                    "source_binder_id": binder_id,
                    "source_original_binder_id": original_id,
                    "source_prediction_path": str(source_complex),
                    "selected_engine": selected_engine,
                    "selected_feature": selected_feature,
                    "filter_config": _jsonable_value(filter_config or {}),
                    "sequence_redesign_ready": True,
                    "interface_fixing_note": (
                        "Use this predicted complex as the redesign backbone; fixed interface residues can be derived "
                        "from binder-target contacts on chains A versus the target chains."
                    ),
                },
            }
        )
        summary_rows.append(
            {
                "candidate_id": binder_id,
                "original_binder_id": original_id,
                "engine": selected_engine,
                "rank": rank_row.get("rank"),
                "feature": selected_feature,
                "feature_value": rank_row.get("feature_value") or rank_row.get(selected_feature),
                "complex_pdb": _candidate_export_rel_path(job.run_dir, complex_path),
                "target_pdb": _candidate_export_rel_path(job.run_dir, target_path),
            }
        )
    if not candidates:
        finish_job(job.run_dir, False, {"metrics": {"error": "No predicted structures could be exported."}})
        raise ValueError("No predicted structures could be exported.")
    normalized = write_candidates(job.run_dir, "ranked_result_selection", candidates)
    normalized_dir = job.run_dir / "artifacts" / "normalized_candidates"
    summary_path = normalized_dir / "result_selection_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    metrics_path = normalized_dir / "imported_candidate_metrics.csv"
    pd.DataFrame(
        [{"candidate_id": candidate.get("candidate_id"), **(candidate.get("metrics") or {})} for candidate in candidates]
    ).to_csv(metrics_path, index=False)
    write_json(
        normalized_dir / "result_selection_manifest.json",
        {
            "export_name": export_name,
            "source_run_dir": str(parent_run_dir),
            "selected_engine": selected_engine,
            "selected_feature": selected_feature,
            "direction": direction,
            "filter_config": _jsonable_value(filter_config or {}),
            "candidate_count": len(normalized),
            "stage": STAGE_COMPLEX_REFOLDING,
            "sequence_redesign_ready": True,
        },
    )
    finish_job(
        job.run_dir,
        True,
        {
            "outputs": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
                "summary": str(summary_path.relative_to(job.run_dir)),
                "imported_candidate_metrics": str(metrics_path.relative_to(job.run_dir)),
            },
            "metrics": {
                "candidate_count": len(normalized),
                "source_ranked_count": int(len(ranked_table)),
                "exported_engine": selected_engine,
                "exported_feature": selected_feature,
            },
            "downstream_artifacts": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "summary": str(summary_path.relative_to(job.run_dir)),
            },
        },
    )
    return job.run_dir


def _create_candidate_set_from_sequence_redesign_selection(
    *,
    parent_run_dir: Path,
    benchmark_dir: Path,
    merged: pd.DataFrame,
    ranked_table: pd.DataFrame,
    selected_engine: tuple[str, str, str],
    selected_feature: str,
    direction: str,
    filter_config: dict[str, Any] | None,
    export_name: str,
    max_candidates: int,
) -> Path:
    if ranked_table.empty:
        raise ValueError("No filtered redesigned sequences are available to export.")
    id_column = _sequence_validation_id_column(ranked_table)
    ranked = ranked_table.copy()
    ranked[id_column] = ranked[id_column].astype(str)
    if max_candidates > 0:
        ranked = ranked.head(int(max_candidates))
    if ranked.empty:
        raise ValueError("No redesigned sequences remain after the max-candidate limit.")

    selected_engine_label, selected_engine_prefix, selected_engine_key = selected_engine
    job = create_job(
        "candidate-import",
        job_type="result_selection_export",
        tool="sequence_redesign_result_selection",
        inputs={
            "source_run_dir": str(parent_run_dir),
            "selected_engine": selected_engine_label,
            "selected_feature": selected_feature,
        },
        params={
            "export_name": export_name,
            "max_candidates": int(max_candidates or 0),
            "direction": direction,
            "filter_config": _jsonable_value(filter_config or {}),
        },
    )
    structure_dir = job.run_dir / "artifacts" / "selected_structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    summary_rows: list[dict[str, object]] = []
    merged_by_id = (
        merged.assign(_candidate_id=merged[_sequence_validation_id_column(merged)].astype(str))
        .drop_duplicates("_candidate_id", keep="first")
        .set_index("_candidate_id")
        if merged is not None and not merged.empty
        else pd.DataFrame()
    )
    for _, rank_row in ranked.iterrows():
        candidate_id = str(rank_row.get(id_column) or "").strip()
        if not candidate_id:
            continue
        design_key = _candidate_structure_id(candidate_id)
        prediction_path = _preferred_structure_path(benchmark_dir, selected_engine_key, design_key)
        if prediction_path is None:
            prediction_path = benchmark_dir / "predicted_metric_pdbs" / selected_engine_prefix / f"{design_key}.pdb"
        if prediction_path is None or not prediction_path.exists():
            continue
        metric_row = merged_by_id.loc[candidate_id].to_dict() if candidate_id in merged_by_id.index else {}
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate_id).strip("_") or f"candidate_{len(candidates) + 1}"
        engine_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", selected_engine_label).strip("_").lower()
        complex_path = structure_dir / f"{safe_id}_{engine_slug}_complex{prediction_path.suffix or '.pdb'}"
        shutil.copy2(prediction_path, complex_path)

        target_path = None
        for target_key in ["source_target_pdb", "target_pdb"]:
            target_text = str(metric_row.get(target_key) or "").strip()
            if not target_text:
                continue
            source_target = _resolve_run_relative_path(parent_run_dir, target_text)
            if source_target is not None and source_target.exists():
                target_path = structure_dir / f"{safe_id}_target{source_target.suffix or '.pdb'}"
                if not target_path.exists():
                    shutil.copy2(source_target, target_path)
                break

        metric_payload = {
            str(key): _jsonable_value(value)
            for key, value in metric_row.items()
            if str(key) not in {"_candidate_id"}
        }
        for key, value in rank_row.to_dict().items():
            if str(key).startswith("_parent_metrics"):
                continue
            metric_payload[f"selection_{key}"] = _jsonable_value(value)
        metric_payload.update(
            {
                "result_selection_rank": _jsonable_value(rank_row.get("viewer_rank")),
                "result_selection_feature": selected_feature,
                "result_selection_feature_value": _jsonable_value(rank_row.get("feature_value") or rank_row.get(selected_feature)),
                "result_selection_rank_score": _jsonable_value(rank_row.get("rank_score")),
                "result_selection_parent_rank_score": _jsonable_value(rank_row.get("parent_rank_score")),
                "result_selection_rank_score_delta_vs_parent": _jsonable_value(rank_row.get("rank_score_delta_vs_parent")),
                "result_selection_parent_score_status": _jsonable_value(rank_row.get("parent_score_status")),
                "result_selection_parent_score_components": _jsonable_value(rank_row.get("parent_score_components")),
                "result_selection_engine": selected_engine_label,
                "result_selection_direction": direction,
            }
        )
        binder_sequence = str(rank_row.get("binder_sequence") or metric_row.get("binder_sequence") or metric_row.get("sequence") or "").strip()
        parent_sequence = str(rank_row.get("parent_sequence") or "").strip()
        parent_id = str(rank_row.get("parent_id") or metric_row.get("original_binder_id") or "").strip()
        target_chains = _parse_chain_list(metric_row.get("target_chains")) or ["B"]
        binder_chains = _parse_chain_list(metric_row.get("binder_chains")) or ["A"]
        candidates.append(
            {
                "candidate_id": candidate_id,
                "stage": STAGE_COMPLEX_REFOLDING,
                "source_tool": "sequence_redesign_result_selection",
                "target_pdb": _candidate_export_rel_path(job.run_dir, target_path),
                "complex_pdb": _candidate_export_rel_path(job.run_dir, complex_path),
                "binder_pdb": None,
                "binder_sequence": binder_sequence,
                "target_chains": target_chains,
                "binder_chains": binder_chains,
                "binder_length": str(len(binder_sequence)) if binder_sequence else None,
                "metrics": metric_payload,
                "parents": [parent_id] if parent_id else [],
                "raw_metadata": {
                    "export_name": export_name,
                    "source_run_dir": str(parent_run_dir),
                    "source_sequence_validation_run_id": parent_run_dir.name,
                    "source_candidate_id": candidate_id,
                    "source_parent_id": parent_id,
                    "source_parent_sequence": parent_sequence,
                    "source_prediction_path": str(prediction_path),
                    "selected_engine": selected_engine_label,
                    "selected_feature": selected_feature,
                    "direction": direction,
                    "filter_config": _jsonable_value(filter_config or {}),
                    "selection_kind": "sequence_redesign_filtered_viewer",
                },
            }
        )
        summary_rows.append(
            {
                "candidate_id": candidate_id,
                "parent_id": parent_id,
                "engine": selected_engine_label,
                "rank": rank_row.get("viewer_rank"),
                "feature": selected_feature,
                "feature_value": rank_row.get("feature_value") or rank_row.get(selected_feature),
                "rank_score": rank_row.get("rank_score"),
                "parent_rank_score": rank_row.get("parent_rank_score"),
                "rank_score_delta_vs_parent": rank_row.get("rank_score_delta_vs_parent"),
                "parent_score_status": rank_row.get("parent_score_status"),
                "parent_score_components": rank_row.get("parent_score_components"),
                "binder_sequence": binder_sequence,
                "parent_sequence": parent_sequence,
                "complex_pdb": _candidate_export_rel_path(job.run_dir, complex_path),
                "target_pdb": _candidate_export_rel_path(job.run_dir, target_path),
            }
        )
    if not candidates:
        finish_job(job.run_dir, False, {"metrics": {"error": "No sequence-redesign prediction structures could be exported."}})
        raise ValueError("No sequence-redesign prediction structures could be exported.")

    normalized = write_candidates(job.run_dir, "sequence_redesign_result_selection", candidates)
    normalized_dir = job.run_dir / "artifacts" / "normalized_candidates"
    summary_path = normalized_dir / "result_selection_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    metrics_path = normalized_dir / "imported_candidate_metrics.csv"
    pd.DataFrame(
        [{"candidate_id": candidate.get("candidate_id"), **(candidate.get("metrics") or {})} for candidate in candidates]
    ).to_csv(metrics_path, index=False)
    write_json(
        normalized_dir / "result_selection_manifest.json",
        {
            "export_name": export_name,
            "source_run_dir": str(parent_run_dir),
            "selected_engine": selected_engine_label,
            "selected_feature": selected_feature,
            "direction": direction,
            "filter_config": _jsonable_value(filter_config or {}),
            "candidate_count": len(normalized),
            "stage": STAGE_COMPLEX_REFOLDING,
            "selection_kind": "sequence_redesign_filtered_viewer",
        },
    )
    finish_job(
        job.run_dir,
        True,
        {
            "outputs": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "campaign_result": "artifacts/normalized_candidates/campaign_result.json",
                "summary": str(summary_path.relative_to(job.run_dir)),
                "imported_candidate_metrics": str(metrics_path.relative_to(job.run_dir)),
            },
            "metrics": {
                "candidate_count": len(normalized),
                "source_ranked_count": int(len(ranked_table)),
                "exported_engine": selected_engine_label,
                "exported_feature": selected_feature,
            },
            "downstream_artifacts": {
                "candidates_jsonl": "artifacts/normalized_candidates/candidates.jsonl",
                "summary": str(summary_path.relative_to(job.run_dir)),
            },
        },
    )
    return job.run_dir


def _target_chains_for_design(merged_metrics: pd.DataFrame | None, design_id: str) -> list[str]:
    if merged_metrics is None or merged_metrics.empty or "binder_id" not in merged_metrics.columns:
        return []
    subset = merged_metrics[merged_metrics["binder_id"].astype(str) == str(design_id)]
    if subset.empty:
        return []
    target_chains: list[str] = []
    if "target_chains" in subset.columns:
        for value in subset["target_chains"].dropna().tolist():
            chains = _parse_chain_list(value)
            if chains:
                target_chains = chains
                break
    is_target_only_refolding = (
        ("capacity_target_only" in subset.columns and subset["capacity_target_only"].map(_truthy_bool_value).any())
        or ("source" in subset.columns and subset["source"].map(_is_target_refolding_input_builder).any())
    )
    if is_target_only_refolding:
        for column in ["binder_chains", "source_input_binder_chains", "source_chain", "binder_chain"]:
            if column not in subset.columns:
                continue
            for value in subset[column].dropna().tolist():
                chains = _parse_chain_list(value)
                if chains:
                    return _unique_chain_list(chains, target_chains)
    return target_chains


def _structure_path_is_cif(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".cif", ".mmcif", ".cif.gz", ".mmcif.gz"))


def _read_structure_text(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            magic = handle.read(2)
    except OSError:
        magic = b""
    if path.name.lower().endswith(".gz") or magic == b"\x1f\x8b":
        with gzip.open(path, "rt", errors="ignore") as handle:
            return handle.read()
    return path.read_text(errors="ignore")


def _parse_structure_file(path: Path):
    from Bio.PDB import MMCIFParser, PDBParser

    text = _read_structure_text(path)
    if _structure_path_is_cif(path):
        parser = MMCIFParser(QUIET=True)
        try:
            return parser.get_structure(path.stem, StringIO(text))
        except KeyError as exc:
            if str(exc).strip("'\"") != "_atom_site.occupancy":
                raise
            patched_text = _mmcif_text_with_default_occupancy(text)
            return parser.get_structure(path.stem, StringIO(patched_text))
    return PDBParser(QUIET=True).get_structure(path.stem, StringIO(text))


def _chain_ids_for_structure_path(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    try:
        structure = _parse_structure_file(path)
        model = next(structure.get_models(), None)
    except Exception:
        return []
    if model is None:
        return []
    return [str(chain.id) for chain in model]


def _mmcif_text_with_default_occupancy(text: str) -> str:
    lines = text.splitlines()
    output: list[str] = []
    atom_site_headers: list[str] = []
    in_atom_site_loop = False
    inserted_header = False
    for line in lines:
        stripped = line.strip()
        if stripped == "loop_":
            in_atom_site_loop = True
            atom_site_headers = []
            inserted_header = False
            output.append(line)
            continue
        if in_atom_site_loop and stripped.startswith("_atom_site."):
            if stripped not in atom_site_headers:
                atom_site_headers.append(stripped)
            output.append(line)
            if stripped == "_atom_site.B_iso_or_equiv" and "_atom_site.occupancy" not in atom_site_headers:
                output.append("_atom_site.occupancy")
                atom_site_headers.append("_atom_site.occupancy")
                inserted_header = True
            continue
        if in_atom_site_loop and stripped.startswith(("ATOM ", "HETATM ")):
            tokens = shlex.split(stripped, posix=False)
            if inserted_header and len(tokens) == len(atom_site_headers) - 1:
                insert_at = atom_site_headers.index("_atom_site.occupancy")
                tokens.insert(insert_at, "1.00")
                output.append(" ".join(tokens))
            else:
                output.append(line)
            continue
        if in_atom_site_loop and stripped.startswith("#"):
            in_atom_site_loop = False
            atom_site_headers = []
            inserted_header = False
        output.append(line)
    return "\n".join(output) + "\n"


def _target_ca_atoms(structure: object, target_chains: list[str]) -> list[object]:
    chain_set = {str(chain) for chain in target_chains}
    atoms: list[object] = []
    model = next(structure.get_models(), None)
    if model is None:
        return atoms
    for chain_id in target_chains:
        chain = model[chain_id] if chain_id in model else None
        if chain is None:
            continue
        for residue in chain:
            if "CA" in residue:
                atoms.append(residue["CA"])
    if atoms:
        return atoms
    for chain in model:
        if str(chain.id) not in chain_set:
            continue
        for residue in chain:
            if "CA" in residue:
                atoms.append(residue["CA"])
    return atoms


def _chain_sequence(chain: object) -> str:
    from Bio.SeqUtils import seq1

    sequence = ""
    for residue in chain:
        if getattr(residue, "id", ("",))[0] != " ":
            continue
        try:
            sequence += seq1(residue.resname)
        except Exception:
            sequence += "X"
    return sequence


def _chain_ca_sequence_atoms(chain: object) -> tuple[str, list[object]]:
    from Bio.SeqUtils import seq1

    sequence = ""
    atoms: list[object] = []
    for residue in chain:
        if getattr(residue, "id", ("",))[0] != " " or "CA" not in residue:
            continue
        try:
            sequence += seq1(residue.resname)
        except Exception:
            sequence += "X"
        atoms.append(residue["CA"])
    return sequence, atoms


def _chain_sequences_by_id(structure: object) -> dict[str, str]:
    model = next(structure.get_models(), None)
    if model is None:
        return {}
    return {str(chain.id): _chain_sequence(chain) for chain in model}


def _infer_matching_target_chains(
    moving_structure: object,
    fixed_structure: object,
    fixed_target_chains: list[str],
) -> list[str]:
    fixed_sequences = _chain_sequences_by_id(fixed_structure)
    moving_sequences = _chain_sequences_by_id(moving_structure)
    used_moving: set[str] = set()
    inferred: list[str] = []
    for fixed_chain in fixed_target_chains:
        fixed_sequence = fixed_sequences.get(str(fixed_chain), "")
        if not fixed_sequence:
            continue
        exact_match = next(
            (
                moving_chain
                for moving_chain, moving_sequence in moving_sequences.items()
                if moving_chain not in used_moving and moving_sequence == fixed_sequence
            ),
            None,
        )
        if exact_match is not None:
            inferred.append(exact_match)
            used_moving.add(exact_match)
            continue
        best_match = None
        best_identity = 0.0
        for moving_chain, moving_sequence in moving_sequences.items():
            if moving_chain in used_moving or not moving_sequence:
                continue
            identity = _sequence_identity(fixed_sequence, moving_sequence)
            if identity > best_identity:
                best_match = moving_chain
                best_identity = identity
        if best_match is not None and best_identity >= 0.8:
            inferred.append(best_match)
            used_moving.add(best_match)
    return inferred


def _sequence_identity(a: str, b: str) -> float:
    length = min(len(a), len(b))
    if length <= 0:
        return 0.0
    return sum(1 for left, right in zip(a[:length], b[:length]) if left == right) / float(length)


def _chain_has_ca_atoms(structure: object, chain_id: str) -> bool:
    model = next(structure.get_models(), None)
    if model is None or chain_id not in model:
        return False
    return any("CA" in residue for residue in model[chain_id])


def _resolve_moving_target_chains_for_alignment(
    path: Path,
    fixed_structure: object,
    fixed_target_chains: list[str],
    moving_target_chains: list[str],
) -> list[str]:
    try:
        moving_structure = _parse_structure_file(path)
    except Exception:
        return moving_target_chains
    requested = [str(chain) for chain in moving_target_chains if str(chain)]
    inferred = _infer_matching_target_chains(
        moving_structure,
        fixed_structure,
        fixed_target_chains,
    )
    if requested and all(_chain_has_ca_atoms(moving_structure, chain) for chain in requested):
        fixed_sequences = _chain_sequences_by_id(fixed_structure)
        moving_sequences = _chain_sequences_by_id(moving_structure)
        fixed_sequence = "".join(fixed_sequences.get(str(chain), "") for chain in fixed_target_chains)
        requested_sequence = "".join(moving_sequences.get(str(chain), "") for chain in requested)
        if fixed_sequence and requested_sequence and _sequence_identity(fixed_sequence, requested_sequence) >= 0.8:
            return requested
    return inferred or requested


def _pdb_safe_chain_id_map(structure: object) -> dict[str, str]:
    model = next(structure.get_models(), None)
    if model is None:
        return {}
    chain_ids = [str(chain.id) for chain in model]
    used = {chain_id for chain_id in chain_ids if len(chain_id) == 1}
    available = (
        chain_id
        for chain_id in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        if chain_id not in used
    )
    chain_map: dict[str, str] = {}
    for chain_id in chain_ids:
        if len(chain_id) == 1:
            chain_map[chain_id] = chain_id
            continue
        try:
            chain_map[chain_id] = next(available)
        except StopIteration as exc:
            raise ValueError("Structure has too many chains for temporary PDB serialization.") from exc
    return chain_map


def _structure_to_pdb_text(structure: object) -> str:
    from copy import deepcopy
    from Bio.PDB import PDBIO

    output_structure = deepcopy(structure)
    chain_map = _pdb_safe_chain_id_map(output_structure)
    model = next(output_structure.get_models(), None)
    if model is not None:
        for chain in model:
            chain.id = chain_map[str(chain.id)]
    output = StringIO()
    writer = PDBIO()
    writer.set_structure(output_structure)
    writer.save(output)
    return output.getvalue()


def _structure_chains_to_pdb_text(path: Path, chains: list[str]) -> str | None:
    from Bio.PDB import PDBIO, Select

    selected = {str(chain) for chain in chains if str(chain)}
    if not selected:
        return None

    class _ChainSelect(Select):
        def accept_chain(self, chain: object) -> bool:
            return str(getattr(chain, "id", "")) in selected

    try:
        structure = _parse_structure_file(path)
        output = StringIO()
        writer = PDBIO()
        writer.set_structure(structure)
        writer.save(output, _ChainSelect())
        text = output.getvalue()
    except Exception:
        return None
    return text if any(line.startswith(("ATOM", "HETATM")) for line in text.splitlines()) else None


def _sequence_aligned_target_ca_atom_pairs(
    fixed_structure: object,
    moving_structure: object,
    fixed_target_chains: list[str],
    moving_target_chains: list[str],
) -> tuple[list[object], list[object], str]:
    from Bio.Align import PairwiseAligner

    fixed_model = next(fixed_structure.get_models(), None)
    moving_model = next(moving_structure.get_models(), None)
    if fixed_model is None or moving_model is None:
        return [], [], "no model"
    fixed_atoms: list[object] = []
    moving_atoms: list[object] = []
    chain_notes: list[str] = []
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -0.5
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    for fixed_chain_id, moving_chain_id in zip(fixed_target_chains, moving_target_chains):
        if fixed_chain_id not in fixed_model or moving_chain_id not in moving_model:
            continue
        fixed_sequence, fixed_chain_atoms = _chain_ca_sequence_atoms(fixed_model[fixed_chain_id])
        moving_sequence, moving_chain_atoms = _chain_ca_sequence_atoms(moving_model[moving_chain_id])
        if not fixed_sequence or not moving_sequence:
            continue
        try:
            alignment = aligner.align(fixed_sequence, moving_sequence)[0]
        except Exception:
            continue
        before_count = len(fixed_atoms)
        for fixed_block, moving_block in zip(alignment.aligned[0], alignment.aligned[1]):
            fixed_start, fixed_end = int(fixed_block[0]), int(fixed_block[1])
            moving_start, moving_end = int(moving_block[0]), int(moving_block[1])
            block_count = min(fixed_end - fixed_start, moving_end - moving_start)
            for offset in range(block_count):
                fixed_atoms.append(fixed_chain_atoms[fixed_start + offset])
                moving_atoms.append(moving_chain_atoms[moving_start + offset])
        chain_notes.append(
            f"{moving_chain_id}->{fixed_chain_id}: {len(fixed_atoms) - before_count}/"
            f"{min(len(fixed_chain_atoms), len(moving_chain_atoms))} paired CA"
        )
    return fixed_atoms, moving_atoms, "; ".join(chain_notes)


def _concatenated_sequence_aligned_target_ca_atom_pairs(
    fixed_structure: object,
    moving_structure: object,
    fixed_target_chains: list[str],
    moving_target_chains: list[str],
) -> tuple[list[object], list[object], str]:
    from Bio.Align import PairwiseAligner

    fixed_model = next(fixed_structure.get_models(), None)
    moving_model = next(moving_structure.get_models(), None)
    if fixed_model is None or moving_model is None:
        return [], [], "no model"
    fixed_sequence = ""
    moving_sequence = ""
    fixed_atoms_by_index: list[object] = []
    moving_atoms_by_index: list[object] = []
    fixed_present_chains: list[str] = []
    moving_present_chains: list[str] = []
    for chain_id in fixed_target_chains:
        if chain_id not in fixed_model:
            continue
        chain_sequence, chain_atoms = _chain_ca_sequence_atoms(fixed_model[chain_id])
        if not chain_sequence:
            continue
        fixed_present_chains.append(str(chain_id))
        fixed_sequence += chain_sequence
        fixed_atoms_by_index.extend(chain_atoms)
    for chain_id in moving_target_chains:
        if chain_id not in moving_model:
            continue
        chain_sequence, chain_atoms = _chain_ca_sequence_atoms(moving_model[chain_id])
        if not chain_sequence:
            continue
        moving_present_chains.append(str(chain_id))
        moving_sequence += chain_sequence
        moving_atoms_by_index.extend(chain_atoms)
    if not fixed_sequence or not moving_sequence:
        return [], [], "no sequence"

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -0.5
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    try:
        alignment = aligner.align(fixed_sequence, moving_sequence)[0]
    except Exception:
        return [], [], "sequence alignment failed"

    fixed_atoms: list[object] = []
    moving_atoms: list[object] = []
    for fixed_block, moving_block in zip(alignment.aligned[0], alignment.aligned[1]):
        fixed_start, fixed_end = int(fixed_block[0]), int(fixed_block[1])
        moving_start, moving_end = int(moving_block[0]), int(moving_block[1])
        block_count = min(fixed_end - fixed_start, moving_end - moving_start)
        for offset in range(block_count):
            fixed_atoms.append(fixed_atoms_by_index[fixed_start + offset])
            moving_atoms.append(moving_atoms_by_index[moving_start + offset])
    note = (
        f"concatenated sequence fallback {','.join(moving_present_chains)} -> "
        f"{','.join(fixed_present_chains)}: {len(fixed_atoms)}/"
        f"{min(len(fixed_atoms_by_index), len(moving_atoms_by_index))} paired CA"
    )
    return fixed_atoms, moving_atoms, note


def _residue_number_target_ca_atom_pairs(
    fixed_structure: object,
    moving_structure: object,
    fixed_target_chains: list[str],
    moving_target_chains: list[str],
) -> tuple[list[object], list[object], str]:
    fixed_model = next(fixed_structure.get_models(), None)
    moving_model = next(moving_structure.get_models(), None)
    if fixed_model is None or moving_model is None:
        return [], [], "no model"
    fixed_atoms: list[object] = []
    moving_atoms: list[object] = []
    chain_notes: list[str] = []
    for fixed_chain_id, moving_chain_id in zip(fixed_target_chains, moving_target_chains):
        if fixed_chain_id not in fixed_model or moving_chain_id not in moving_model:
            chain_notes.append(f"{moving_chain_id}->{fixed_chain_id}: missing chain")
            continue
        moving_by_residue: dict[tuple[int, str], object] = {}
        for residue in moving_model[moving_chain_id]:
            residue_id = getattr(residue, "id", ("", None, ""))
            if residue_id[0] != " " or "CA" not in residue:
                continue
            moving_by_residue[(int(residue_id[1]), str(residue_id[2]).strip())] = residue["CA"]
        before_count = len(fixed_atoms)
        fixed_ca_count = 0
        for residue in fixed_model[fixed_chain_id]:
            residue_id = getattr(residue, "id", ("", None, ""))
            if residue_id[0] != " " or "CA" not in residue:
                continue
            fixed_ca_count += 1
            moving_atom = moving_by_residue.get((int(residue_id[1]), str(residue_id[2]).strip()))
            if moving_atom is None:
                continue
            fixed_atoms.append(residue["CA"])
            moving_atoms.append(moving_atom)
        chain_notes.append(
            f"{moving_chain_id}->{fixed_chain_id}: {len(fixed_atoms) - before_count}/"
            f"{fixed_ca_count} residue-number matched CA"
        )
    return fixed_atoms, moving_atoms, "; ".join(chain_notes)


def _aligned_structure_text(
    path: Path,
    fixed_structure: object,
    fixed_target_chains: list[str],
    moving_target_chains: list[str],
) -> tuple[str | None, float | None, str]:
    from Bio.PDB import Superimposer

    def _pairing_candidates(moving_structure: object) -> list[tuple[str, list[object], list[object], str]]:
        candidates: list[tuple[str, list[object], list[object], str]] = []
        fixed_atoms, moving_atoms, note = _residue_number_target_ca_atom_pairs(
            fixed_structure,
            moving_structure,
            fixed_target_chains,
            moving_target_chains,
        )
        candidates.append(("residue-number", fixed_atoms, moving_atoms, note))
        fixed_atoms, moving_atoms, note = _sequence_aligned_target_ca_atom_pairs(
            fixed_structure,
            moving_structure,
            fixed_target_chains,
            moving_target_chains,
        )
        candidates.append(("sequence", fixed_atoms, moving_atoms, note))
        fixed_atoms, moving_atoms, note = _concatenated_sequence_aligned_target_ca_atom_pairs(
            fixed_structure,
            moving_structure,
            fixed_target_chains,
            moving_target_chains,
        )
        candidates.append(("concatenated sequence", fixed_atoms, moving_atoms, note))
        fixed_atoms = _target_ca_atoms(fixed_structure, fixed_target_chains)
        moving_atoms = _target_ca_atoms(moving_structure, moving_target_chains)
        candidates.append(("positional", fixed_atoms, moving_atoms, "positional CA pairing"))
        return candidates

    try:
        fixed_total_atoms = _target_ca_atoms(fixed_structure, fixed_target_chains)
        fixed_total_count = len(fixed_total_atoms)
        best: tuple[float, int, float, str, object] | None = None
        best_note = ""
        for method, fixed_atoms, moving_atoms, pairing_note in _pairing_candidates(_parse_structure_file(path)):
            atom_count = min(len(fixed_atoms), len(moving_atoms))
            if atom_count < 3:
                continue
            moving_structure = _parse_structure_file(path)
            for candidate_method, candidate_fixed_atoms, candidate_moving_atoms, candidate_pairing_note in _pairing_candidates(
                moving_structure
            ):
                if candidate_method != method:
                    continue
                atom_count = min(len(candidate_fixed_atoms), len(candidate_moving_atoms))
                if atom_count < 3:
                    continue
                superimposer = Superimposer()
                superimposer.set_atoms(candidate_fixed_atoms[:atom_count], candidate_moving_atoms[:atom_count])
                superimposer.apply(moving_structure.get_atoms())
                rms = float(superimposer.rms)
                coverage = atom_count / fixed_total_count if fixed_total_count else 0.0
                best_coverage = best[2] if best is not None else 0.0
                full_coverage = coverage >= 0.95
                best_full_coverage = best_coverage >= 0.95
                if (
                    best is None
                    or (full_coverage and not best_full_coverage)
                    or (
                        full_coverage == best_full_coverage
                        and (
                            rms < best[0]
                            or (abs(rms - best[0]) < 1e-6 and atom_count > best[1])
                        )
                    )
                ):
                    best = (rms, atom_count, coverage, method, moving_structure)
                    best_note = candidate_pairing_note
                break
    except Exception as exc:
        return None, None, f"alignment failed: {exc}"
    if best is None:
        return None, None, "alignment skipped: fewer than 3 shared target CA atoms"
    rms, atom_count, _coverage, method, moving_structure = best
    return (
        _structure_to_pdb_text(moving_structure),
        rms,
        f"aligned on {atom_count}/{fixed_total_count} target CA atoms "
        f"({','.join(moving_target_chains)} -> {','.join(fixed_target_chains)}; {method}; {best_note})",
    )


def _alignment_ca_count(alignment_note: object) -> int | None:
    match = re.search(r"aligned on\s+(\d+)(?:/\d+)?\s+target CA atoms", str(alignment_note or ""))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _alignment_reference_ca_count(alignment_note: object) -> int | None:
    match = re.search(r"aligned on\s+\d+/(\d+)\s+target CA atoms", str(alignment_note or ""))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _py3dmol_grid_html(
    entries: list[dict[str, object]],
    *,
    reference_pdb: str | None,
    include_reference: bool,
    height: int,
) -> str:
    panels = entries[:9]
    rows = max(1, (len(panels) + 2) // 3)
    cols = 3
    js_url = "https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js"
    panels_json = json.dumps(
        [
            {
                "engine": str(entry.get("engine") or ""),
                "pdb": str(entry.get("pdb") or ""),
                "color": str(entry.get("color") or "#7a869a"),
                "format": str(entry.get("format") or ""),
            }
            for entry in panels
        ]
    )
    reference_json = json.dumps(reference_pdb or "")
    return f"""
<div id="benchmark-3dmol-grid" style="width:100%; height:{height}px; position:relative; border:1px solid #d8dee9; border-radius:6px; overflow:hidden;"></div>
<div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:6px; margin:8px 0 0 0; font:12px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
{''.join(
    f'<div style="display:flex; align-items:center; gap:6px; min-width:0;"><span style="width:10px; height:10px; border-radius:50%; background:{str(entry.get("color") or "#7a869a")}; flex:0 0 auto;"></span><span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{str(entry.get("engine") or "")}</span></div>'
    for entry in panels
)}
</div>
<script>
(function() {{
  const panels = {panels_json};
  const referencePdb = {reference_json};
  const includeReference = {str(bool(include_reference)).lower()};
  const rows = {rows};
  const cols = {cols};
  const jsUrl = {json.dumps(js_url)};
  function modelFormat(text, explicitFormat) {{
    const fmt = (explicitFormat || '').toLowerCase();
    if (fmt === 'cif' || fmt === 'mmcif' || fmt === 'pdb') return fmt === 'mmcif' ? 'cif' : fmt;
    const trimmed = (text || '').trimStart();
    if (trimmed.startsWith('data_') || trimmed.includes('_atom_site.')) return 'cif';
    return 'pdb';
  }}
  function loadScriptAsync(uri) {{
    return new Promise((resolve, reject) => {{
      if (window.$3Dmol) {{
        resolve();
        return;
      }}
      const tag = document.createElement('script');
      tag.src = uri;
      tag.async = true;
      tag.onload = resolve;
      tag.onerror = reject;
      document.head.appendChild(tag);
    }});
  }}
  loadScriptAsync(jsUrl).then(function() {{
    const container = document.getElementById('benchmark-3dmol-grid');
    if (!container) return;
    container.innerHTML = '';
    const grid = $3Dmol.createViewerGrid(container, {{rows: rows, cols: cols, control_all: true}}, {{backgroundColor: 'white'}});
    for (let idx = 0; idx < panels.length; idx += 1) {{
      const row = Math.floor(idx / cols);
      const col = idx % cols;
      const viewer = grid[row][col];
      if (includeReference && referencePdb) {{
        const referenceModel = viewer.addModel(referencePdb, modelFormat(referencePdb, ''));
        referenceModel.setStyle({{}}, {{cartoon: {{color: '#8a8f98', opacity: 0.45}}}});
      }}
      const engineModel = viewer.addModel(panels[idx].pdb, modelFormat(panels[idx].pdb, panels[idx].format));
      engineModel.setStyle({{}}, {{cartoon: {{color: panels[idx].color}}}});
      viewer.zoomTo();
      viewer.render();
    }}
  }}).catch(function(error) {{
    const container = document.getElementById('benchmark-3dmol-grid');
    if (container) {{
      container.innerHTML = '<p style="padding:12px; background:#fff3cd; color:#664d03;">3Dmol.js could not be loaded. Check browser network access to jsDelivr.</p>';
    }}
  }});
}})();
</script>
"""


@st.cache_data(show_spinner=False)
def _read_masif_surface_cached(path_text: str, mtime_ns: int, size: int) -> pd.DataFrame:
    _ = (mtime_ns, size)
    path = Path(path_text)
    with path.open("r", errors="ignore") as handle:
        first = handle.readline().strip()
        if first != "ply":
            raise ValueError("Only ASCII PLY MaSIF surfaces are supported")
        vertex_count = 0
        properties: list[str] = []
        in_vertex = False
        for line in handle:
            stripped = line.strip()
            if stripped.startswith("element vertex"):
                vertex_count = int(stripped.split()[-1])
                in_vertex = True
                continue
            if stripped.startswith("element ") and not stripped.startswith("element vertex"):
                in_vertex = False
            if in_vertex and stripped.startswith("property "):
                properties.append(stripped.split()[-1])
            if stripped == "end_header":
                break
        rows: list[list[float]] = []
        for _idx in range(vertex_count):
            values = handle.readline().split()
            if len(values) < len(properties):
                break
            rows.append([float(value) for value in values[: len(properties)]])
    return pd.DataFrame(rows, columns=properties)


def _read_masif_surface(path: Path) -> pd.DataFrame | None:
    try:
        stat = path.stat()
        return _read_masif_surface_cached(str(path), int(stat.st_mtime_ns), int(stat.st_size))
    except Exception as exc:
        st.warning(f"Could not read MaSIF surface {path.name}: {exc}")
        return None


def _read_masif_scores(prediction_path: Path | None, surface_df: pd.DataFrame) -> np.ndarray:
    if prediction_path and prediction_path.exists():
        try:
            scores = np.asarray(np.load(prediction_path), dtype=float).reshape(-1)
            if len(scores) == len(surface_df):
                return scores
        except Exception:
            pass
    if "iface" in surface_df.columns:
        return surface_df["iface"].astype(float).to_numpy()
    return np.zeros(len(surface_df), dtype=float)


def _parse_pdb_atoms(pdb_text: str) -> list[dict[str, object]]:
    atoms: list[dict[str, object]] = []
    for line_index, line in enumerate(pdb_text.splitlines()):
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 54:
            continue
        try:
            coord = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float)
        except ValueError:
            continue
        atoms.append(
            {
                "line_index": line_index,
                "coord": coord,
                "residue_key": (line[21].strip() or "_", line[22:26].strip(), line[26].strip(), line[17:20].strip()),
            }
        )
    return atoms


def _pdb_with_bfactors(pdb_text: str, atoms: list[dict[str, object]], values: np.ndarray) -> str:
    lines = pdb_text.splitlines()
    atom_value_by_line = {int(atom["line_index"]): float(values[idx]) for idx, atom in enumerate(atoms)}
    for line_index, value in atom_value_by_line.items():
        line = lines[line_index]
        if len(line) < 66:
            line = line.ljust(66)
        lines[line_index] = f"{line[:60]}{max(0.0, min(100.0, value * 100.0)):6.2f}{line[66:]}"
    return "\n".join(lines) + "\n"


def _residue_bfactor_table(pdb_text: str, threshold: float) -> pd.DataFrame:
    rows: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 66:
            continue
        try:
            bfactor = float(line[60:66])
        except ValueError:
            continue
        chain = line[21].strip() or "_"
        residue_number = line[22:26].strip()
        insertion_code = line[26].strip()
        residue_name = line[17:20].strip()
        key = (chain, residue_number, insertion_code, residue_name)
        current = rows.setdefault(
            key,
            {
                "chain": chain,
                "residue": residue_number,
                "insertion": insertion_code,
                "amino_acid": residue_name,
                "binding_bfactor": 0.0,
                "atom_count": 0,
            },
        )
        current["binding_bfactor"] = max(float(current["binding_bfactor"]), bfactor)
        current["atom_count"] = int(current["atom_count"]) + 1
    table = pd.DataFrame(rows.values())
    if table.empty:
        return table
    table["binding_score"] = table["binding_bfactor"] / 100.0
    table["above_threshold"] = table["binding_score"] >= threshold
    table = table.sort_values(["binding_bfactor", "chain", "residue"], ascending=[False, True, True])
    return table.reset_index(drop=True)


def _masif_pointcloud_pdb(surface_df: pd.DataFrame, scores: np.ndarray) -> str:
    lines = []
    for idx, row in surface_df.reset_index(drop=True).iterrows():
        serial = (idx % 99999) + 1
        resseq = (idx % 9999) + 1
        score = max(0.0, min(100.0, float(scores[idx]) * 100.0))
        lines.append(
            f"HETATM{serial:5d}  H   PNT A{resseq:4d}    "
            f"{float(row['x']):8.3f}{float(row['y']):8.3f}{float(row['z']):8.3f}"
            f"  1.00{score:6.2f}           H"
        )
    return "\n".join(lines) + "\nEND\n"


def _write_masif_binding_pdbs(
    *,
    output_dir: Path,
    target_pdb: Path,
    surface_df: pd.DataFrame,
    scores: np.ndarray,
    atom_distance_cutoff: float,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    vertices = surface_df[["x", "y", "z"]].astype(float).to_numpy()
    pdb_text = target_pdb.read_text(errors="ignore")
    atoms = _parse_pdb_atoms(pdb_text)
    if not atoms or len(vertices) == 0:
        return {}

    atom_coords = np.vstack([np.asarray(atom["coord"], dtype=float) for atom in atoms])
    atom_scores = np.zeros(len(atoms), dtype=float)
    best_dist2 = np.full(len(atoms), np.inf, dtype=float)
    chunk = 256
    for start in range(0, len(atom_coords), chunk):
        stop = min(start + chunk, len(atom_coords))
        diff = atom_coords[start:stop, None, :] - vertices[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        nearest = np.argmin(dist2, axis=1)
        best_dist2[start:stop] = dist2[np.arange(stop - start), nearest]
        atom_scores[start:stop] = scores[nearest]
    atom_scores[best_dist2 > atom_distance_cutoff * atom_distance_cutoff] = 0.0

    residue_scores: dict[tuple[str, str, str, str], float] = {}
    for atom, score in zip(atoms, atom_scores, strict=False):
        key = atom["residue_key"]
        residue_scores[key] = max(residue_scores.get(key, 0.0), float(score))
    residue_atom_scores = np.array([residue_scores.get(atom["residue_key"], 0.0) for atom in atoms], dtype=float)

    paths = {
        "pointcloud": output_dir / "masif_pointcloud_binding.pdb",
        "atom": output_dir / "masif_per_atom_binding.pdb",
        "residue": output_dir / "masif_per_residue_binding.pdb",
    }
    paths["pointcloud"].write_text(_masif_pointcloud_pdb(surface_df, scores))
    paths["atom"].write_text(_pdb_with_bfactors(pdb_text, atoms, atom_scores))
    paths["residue"].write_text(_pdb_with_bfactors(pdb_text, atoms, residue_atom_scores))
    return paths


def _pymol_string(value: Path | str) -> str:
    return str(value).replace("\\", "/").replace('"', '\\"')


def _masif_pymol_script(
    *,
    residue_pdb: Path,
    atom_pdb: Path,
    pointcloud_pdb: Path,
    pse_path: Path,
    threshold: float,
) -> str:
    threshold_b = max(0.0, min(100.0, threshold * 100.0))
    return "\n".join(
        [
            "reinitialize",
            f'load "{_pymol_string(residue_pdb)}", masif_residue_binding',
            f'load "{_pymol_string(atom_pdb)}", masif_atom_binding',
            f'load "{_pymol_string(pointcloud_pdb)}", masif_surface_points',
            "disable masif_atom_binding",
            "hide everything",
            "show cartoon, masif_residue_binding",
            "show surface, masif_residue_binding",
            "set transparency, 0.45, masif_residue_binding",
            "show spheres, masif_surface_points",
            "set sphere_scale, 0.25, masif_surface_points",
            "spectrum b, blue_white_red, minimum=0, maximum=100, masif_residue_binding",
            "spectrum b, blue_white_red, minimum=0, maximum=100, masif_surface_points",
            f"select masif_hotspot_residues, masif_residue_binding and b > {threshold_b:.2f}",
            "show sticks, masif_hotspot_residues",
            "color red, masif_hotspot_residues",
            "set surface_quality, 1",
            "set ray_opaque_background, off",
            "bg_color white",
            "orient masif_residue_binding",
            f'save "{_pymol_string(pse_path)}"',
            "quit",
            "",
        ]
    )


def _ensure_masif_pymol_session(
    *,
    output_dir: Path,
    residue_pdb: Path,
    atom_pdb: Path,
    pointcloud_pdb: Path,
    threshold: float,
) -> tuple[Path | None, Path, str | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pml_path = output_dir / "masif_binding_session.pml"
    pse_path = output_dir / "masif_binding_session.pse"
    script = _masif_pymol_script(
        residue_pdb=residue_pdb,
        atom_pdb=atom_pdb,
        pointcloud_pdb=pointcloud_pdb,
        pse_path=pse_path,
        threshold=threshold,
    )
    pml_path.write_text(script)
    if pse_path.exists() and pse_path.stat().st_mtime_ns >= pml_path.stat().st_mtime_ns:
        return pse_path, pml_path, None
    pymol = shutil.which("pymol")
    if not pymol:
        env_pymol = Path(sys.executable).with_name("pymol")
        if env_pymol.exists():
            pymol = str(env_pymol)
    if not pymol:
        return None, pml_path, "PyMOL executable was not found on the app PATH."
    completed = subprocess.run(
        [pymol, "-cq", str(pml_path)],
        cwd=output_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
        check=False,
    )
    log_path = output_dir / "masif_binding_session_pymol.log"
    log_path.write_text((completed.stdout or "") + ("\n" if completed.stdout and completed.stderr else "") + (completed.stderr or ""))
    if completed.returncode != 0:
        return None, pml_path, f"PyMOL returned code {completed.returncode}. See {log_path.name}."
    if not pse_path.exists():
        return None, pml_path, "PyMOL finished but did not create the session file."
    return pse_path, pml_path, None


def _py3dmol_masif_html(
    target_pdb: str,
    threshold: float,
    *,
    background_color: str,
    show_surface: bool,
    selected_residue: dict[str, object] | None,
    height: int = 540,
) -> str:
    js_url = "https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js"
    background = "#000000" if background_color == "black" else "#ffffff"
    cartoon_color = "#d4d7dd" if background_color == "black" else "#a1a8b3"
    return f"""
<div id="masif-3dmol-view" style="width:100%; height:{height}px; position:relative; border:1px solid #d8dee9; border-radius:6px; overflow:hidden;"></div>
<div id="masif-3dmol-picked-residue" style="margin:8px 0 0 0; padding:8px 10px; border:1px solid #d8dee9; border-radius:6px; font:13px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color:#111827; background:#f9fafb;">
  Hover or click a red hotspot residue to identify it.
</div>
<script>
(function() {{
  const targetPdb = {json.dumps(target_pdb)};
  const threshold = {float(threshold) * 100.0:.3f};
  const showSurface = {str(bool(show_surface)).lower()};
  const selectedResidue = {json.dumps(selected_residue or {})};
  const jsUrl = {json.dumps(js_url)};
  function colorByBinding(atom) {{
    const b = Math.max(0, Math.min(100, atom.b || 0));
    const t = b / 100.0;
    let r, g, bl;
    if (t <= 0.5) {{
      const u = t / 0.5;
      r = Math.round(25 + (245 - 25) * u);
      g = Math.round(82 + (245 - 82) * u);
      bl = Math.round(230 + (245 - 230) * u);
    }} else {{
      const u = (t - 0.5) / 0.5;
      r = Math.round(245 + (215 - 245) * u);
      g = Math.round(245 + (35 - 245) * u);
      bl = Math.round(245 + (35 - 245) * u);
    }}
    return 'rgb(' + r + ',' + g + ',' + bl + ')';
  }}
  function residueText(atom) {{
    const chain = atom.chain || '';
    const resn = atom.resn || '';
    const resi = atom.resi || '';
    const atomName = atom.atom || '';
    const score = ((atom.b || 0) / 100.0).toFixed(3);
    return chain + ':' + resn + resi + ' ' + atomName + '  score ' + score;
  }}
  function updatePickedResidue(atom, prefix) {{
    const box = document.getElementById('masif-3dmol-picked-residue');
    if (!box || !atom) return;
    box.textContent = prefix + ': ' + residueText(atom);
  }}
  function residueSelectionFromRow(row) {{
    if (!row || !row.chain || !row.residue) return null;
    return {{
      predicate: function(atom) {{
        const atomChain = atom.chain || '';
        const atomResi = String(atom.resi || '');
        const rowResi = String(row.residue || '');
        const atomResn = String(atom.resn || '').trim();
        const rowResn = String(row.amino_acid || '').trim();
        return atomChain === String(row.chain) && atomResi === rowResi && (!rowResn || atomResn === rowResn);
      }}
    }};
  }}
  function meanPoint(atoms) {{
    const center = {{x: 0, y: 0, z: 0}};
    if (!atoms || atoms.length === 0) return center;
    atoms.forEach(function(atom) {{
      center.x += atom.x || 0;
      center.y += atom.y || 0;
      center.z += atom.z || 0;
    }});
    center.x /= atoms.length;
    center.y /= atoms.length;
    center.z /= atoms.length;
    return center;
  }}
  function orientResidueFromOutside(viewerRef, model, selection, selectedAtoms) {{
    if (!selectedAtoms || selectedAtoms.length === 0) {{
      viewerRef.zoomTo(selection);
      return;
    }}
    const allAtoms = model.selectedAtoms({{}});
    const proteinCenter = meanPoint(allAtoms);
    const residueCenter = meanPoint(selectedAtoms);
    let vx = residueCenter.x - proteinCenter.x;
    let vy = residueCenter.y - proteinCenter.y;
    let vz = residueCenter.z - proteinCenter.z;
    const norm = Math.sqrt(vx * vx + vy * vy + vz * vz) || 1.0;
    vx /= norm;
    vy /= norm;
    vz /= norm;
    const cameraDistance = 42;
    viewerRef.zoomTo(selection);
    viewerRef.setCameraParameters({{
      position: {{
        x: residueCenter.x + vx * cameraDistance,
        y: residueCenter.y + vy * cameraDistance,
        z: residueCenter.z + vz * cameraDistance
      }},
      target: residueCenter,
      up: {{x: 0, y: 1, z: 0}},
      fov: 20,
      z: cameraDistance
    }});
  }}
  function loadScriptAsync(uri) {{
    return new Promise((resolve, reject) => {{
      if (window.$3Dmol) {{
        resolve();
        return;
      }}
      const tag = document.createElement('script');
      tag.src = uri;
      tag.async = true;
      tag.onload = resolve;
      tag.onerror = reject;
      document.head.appendChild(tag);
    }});
  }}
  loadScriptAsync(jsUrl).then(function() {{
    const container = document.getElementById('masif-3dmol-view');
    if (!container) return;
    const viewer = $3Dmol.createViewer(container, {{backgroundColor: {json.dumps(background)}}});
    const target = viewer.addModel(targetPdb, 'pdb');
    const selectedByScore = {{predicate: function(atom) {{ return (atom.b || 0) >= threshold; }}}};
    target.setStyle({{}}, {{cartoon: {{color: {json.dumps(cartoon_color)}, opacity: 0.2}}}});
    target.setStyle(selectedByScore, {{stick: {{colorfunc: colorByBinding, radius: 0.16}}, sphere: {{radius: 0.2, colorfunc: colorByBinding, opacity: 0.65}}}});
    let hoverLabel = null;
    function showResidueLabel(atom, viewerRef, prefix) {{
      if (!atom) return;
      if (hoverLabel) {{
        viewerRef.removeLabel(hoverLabel);
        hoverLabel = null;
      }}
      hoverLabel = viewerRef.addLabel(
        residueText(atom),
        {{
          position: atom,
          backgroundColor: 'rgba(17,24,39,0.92)',
          fontColor: 'white',
          fontSize: 13,
          borderThickness: 1,
          borderColor: 'white',
          inFront: true
        }}
      );
      updatePickedResidue(atom, prefix);
      viewerRef.render();
    }}
    viewer.setHoverable(
      selectedByScore,
      true,
      function(atom, viewerRef) {{
        showResidueLabel(atom, viewerRef, 'Hover');
      }},
      function(atom, viewerRef) {{
        if (hoverLabel) {{
          viewerRef.removeLabel(hoverLabel);
          hoverLabel = null;
          viewerRef.render();
        }}
      }}
    );
    viewer.setClickable(selectedByScore, true, function(atom, viewerRef) {{
      showResidueLabel(atom, viewerRef, 'Clicked');
    }});
    if (showSurface) {{
      viewer.addSurface(
        $3Dmol.SurfaceType.VDW,
        {{opacity: 0.92, colorfunc: colorByBinding}},
        {{model: target}}
      );
    }}
    const selectedRowSelection = residueSelectionFromRow(selectedResidue);
    if (selectedRowSelection) {{
      target.setStyle(selectedRowSelection, {{stick: {{colorfunc: colorByBinding, radius: 0.3}}, sphere: {{radius: 0.36, colorfunc: colorByBinding, opacity: 0.95}}}});
      const atoms = target.selectedAtoms(selectedRowSelection);
      if (atoms && atoms.length > 0) {{
        showResidueLabel(atoms[0], viewer, 'Selected');
        orientResidueFromOutside(viewer, target, selectedRowSelection, atoms);
      }} else {{
        viewer.zoomTo();
      }}
    }} else {{
      viewer.zoomTo();
    }}
    viewer.render();
  }}).catch(function() {{
    const container = document.getElementById('masif-3dmol-view');
    if (container) {{
      container.innerHTML = '<p style="padding:12px; background:#fff3cd; color:#664d03;">3Dmol.js could not be loaded. Check browser network access to jsDelivr.</p>';
    }}
  }});
}})();
</script>
"""


def _molstar_binding_color_params() -> dict[str, object]:
    # Mol*'s uncertainty theme maps the color list in the opposite direction
    # from the MaSIF score semantics we show in the legend.
    return {
        "domain": [0, 100],
        "list": {
            "colors": [
                0xD72323,
                0xF5F5F5,
                0x1952E6,
            ]
        },
    }


def _masif_selection_string(residue: dict[str, object] | None) -> str | None:
    if not isinstance(residue, dict):
        return None
    chain = str(residue.get("chain") or "").strip()
    number = str(residue.get("residue") or "").strip()
    if not chain or not number:
        return None
    return f"{chain}{number}"


def _residue_names_from_pdb(pdb_text: str) -> dict[tuple[str, str], str]:
    names: dict[tuple[str, str], str] = {}
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
            continue
        chain = line[21].strip() or "_"
        residue_number = line[22:26].strip()
        residue_name = line[17:20].strip()
        names.setdefault((chain, residue_number), residue_name)
    return names


def _pdb_with_residue_bfactors(pdb_text: str, residue_scores: dict[tuple[str, str], float]) -> str:
    lines: list[str] = []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            lines.append(line)
            continue
        chain = line[21].strip() or "_"
        residue_number = line[22:26].strip()
        score = max(0.0, min(100.0, float(residue_scores.get((chain, residue_number), 0.0)) * 100.0))
        padded = line if len(line) >= 66 else line.ljust(66)
        lines.append(f"{padded[:60]}{score:6.2f}{padded[66:]}")
    return "\n".join(lines) + "\n"


def _show_surf2spot_results(run_dir: Path) -> None:
    root = run_dir / "artifacts" / "surf2spot"
    if not root.exists():
        return
    predict_dir = root / "predict"
    preprocess_dir = root / "preprocess"
    hotspot_tables = sorted(predict_dir.glob("*.csv"))
    pse_files = sorted(predict_dir.glob("*.pse"))
    surface_files = sorted(predict_dir.glob("*_pred.ply"))
    source_pdbs = sorted(preprocess_dir.glob("*.pdb")) or sorted((root / "input").glob("*.pdb"))
    if not hotspot_tables and not pse_files and not surface_files:
        return

    st.subheader("Surf2Spot Hotspot Prediction")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Hotspot tables", len(hotspot_tables))
    metric_cols[1].metric("PyMOL sessions", len(pse_files))
    metric_cols[2].metric("Surface meshes", len(surface_files))
    metric_cols[3].metric("Prepared PDBs", len(source_pdbs))

    selected_table = None
    if hotspot_tables:
        selected_table = st.selectbox(
            "Hotspot table",
            hotspot_tables,
            format_func=lambda path: path.name,
            key=f"{run_dir.name}_surf2spot_table",
        )
    source_pdb = source_pdbs[0] if source_pdbs else None
    if source_pdb and selected_table:
        try:
            table = pd.read_csv(selected_table)
        except Exception as exc:
            st.warning(f"Could not read Surf2Spot table {selected_table.name}: {exc}")
            table = pd.DataFrame()
        if not table.empty and {"aa_id", "score"}.issubset(table.columns):
            pdb_text = source_pdb.read_text(errors="ignore")
            residue_names = _residue_names_from_pdb(pdb_text)
            chain_ids = sorted({chain for chain, _residue in residue_names}) or ["_"]
            selected_chain = chain_ids[0]
            table = table.copy()
            table["chain"] = selected_chain
            table["residue"] = table["aa_id"].astype(str)
            table["amino_acid"] = table["residue"].map(lambda residue: residue_names.get((selected_chain, str(residue)), ""))
            table["score"] = pd.to_numeric(table["score"], errors="coerce").fillna(0.0)
            residue_scores = {
                (selected_chain, str(row["aa_id"])): float(row["score"])
                for row in table.to_dict(orient="records")
            }
            visualization_dir = root / "visualization"
            visualization_dir.mkdir(parents=True, exist_ok=True)
            binding_pdb = visualization_dir / f"{selected_table.stem}_surf2spot_hotspots_bfactor.pdb"
            binding_pdb.write_text(_pdb_with_residue_bfactors(pdb_text, residue_scores))

            show_surface = st.checkbox(
                "Show surface",
                value=True,
                key=f"{run_dir.name}_surf2spot_surface",
                help="Toggle surface rendering. Residues are colored by Surf2Spot score in the PDB B-factor column.",
            )
            representation = "molecular-surface+ball-and-stick" if show_surface else "ball-and-stick"
            molstar_custom_component(
                [
                    StructureVisualization(
                        pdb=binding_pdb.read_text(errors="ignore"),
                        color="uncertainty",  # type: ignore[arg-type]
                        color_params=_molstar_binding_color_params(),
                        representation_type=representation,  # type: ignore[arg-type]
                    )
                ],
                key=f"{run_dir.name}_surf2spot_molstar_{selected_table.stem}_{bool(show_surface)}",
                height=720,
                show_controls=True,
                selection_mode=True,
                html_filename=f"{run_dir.name}_surf2spot_hotspots",
            )
            st.caption(
                "Mol* uses Surf2Spot scores written into the PDB B-factor column. Red marks stronger predicted hotspot residues."
            )
            st.markdown(
                """
<div style="display:flex; align-items:center; gap:10px; margin:4px 0 14px 0; max-width:520px;">
  <span style="font-size:12px; color:#4b5563;">weak</span>
  <div style="height:12px; flex:1; border-radius:6px; border:1px solid #d1d5db; background:linear-gradient(90deg, rgb(25,82,230), rgb(245,245,245), rgb(215,35,35));"></div>
  <span style="font-size:12px; color:#4b5563;">strong hotspot</span>
</div>
""",
                unsafe_allow_html=True,
            )
            st.subheader("Predicted Hotspot Residues")
            show_only_hotspots = st.checkbox(
                "Show only nonzero hotspot scores",
                value=True,
                key=f"{run_dir.name}_surf2spot_nonzero",
            )
            display_table = table[table["score"] > 0].copy() if show_only_hotspots else table.copy()
            display_table = display_table.sort_values(["score", "aa_id"], ascending=[False, True])
            st.dataframe(
                display_table[["chain", "aa_id", "amino_acid", "score"]],
                hide_index=True,
                width="stretch",
                column_config={"score": st.column_config.NumberColumn("score", format="%.3f")},
            )

    if pse_files or surface_files:
        with st.expander("Open in PyMOL or download raw Surf2Spot outputs", expanded=False):
            for pse_file in pse_files:
                st.download_button(
                    f"Download {pse_file.name}",
                    data=pse_file.read_bytes(),
                    file_name=pse_file.name,
                    mime="application/octet-stream",
                    key=f"{run_dir.name}_surf2spot_download_{pse_file.name}",
                )
            for surface_file in surface_files:
                st.write(str(surface_file.relative_to(run_dir)))


def _show_scannet_results(run_dir: Path) -> None:
    root = run_dir / "artifacts" / "scannet"
    if not root.exists():
        return
    prediction_tables = sorted(root.glob("**/predictions_*.csv"))
    annotated_pdbs = sorted(root.glob("**/annotated_*.pdb"))
    script_files = sorted(root.glob("**/*.cxc")) + sorted(root.glob("**/*.py"))
    if not prediction_tables and not annotated_pdbs:
        return

    st.subheader("ScanNet PPI Prediction")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Prediction tables", len(prediction_tables))
    metric_cols[1].metric("Annotated PDBs", len(annotated_pdbs))
    metric_cols[2].metric("Scripts", len(script_files))

    selected_table = None
    if prediction_tables:
        selected_table = st.selectbox(
            "Prediction table",
            prediction_tables,
            format_func=lambda path: path.name,
            key=f"{run_dir.name}_scannet_table",
        )
    source_pdb = annotated_pdbs[0] if annotated_pdbs else run_dir / "artifacts" / "input" / "target.pdb"
    if selected_table and source_pdb.exists():
        try:
            table = pd.read_csv(selected_table)
        except Exception as exc:
            st.warning(f"Could not read ScanNet table {selected_table.name}: {exc}")
            table = pd.DataFrame()
        required = {"Chain", "Residue Index", "Sequence", "Binding site probability"}
        if not table.empty and required.issubset(table.columns):
            table = table.copy()
            table["score"] = pd.to_numeric(table["Binding site probability"], errors="coerce").fillna(0.0)
            table["chain"] = table["Chain"].astype(str)
            table["residue"] = table["Residue Index"].astype(str)
            table["amino_acid"] = table["Sequence"].astype(str)
            residue_scores = {
                (str(row["Chain"]), str(row["Residue Index"])): float(row["score"])
                for row in table.to_dict(orient="records")
            }
            visualization_dir = root / "visualization"
            visualization_dir.mkdir(parents=True, exist_ok=True)
            binding_pdb = visualization_dir / f"{selected_table.stem}_scannet_bfactor.pdb"
            binding_pdb.write_text(_pdb_with_residue_bfactors(source_pdb.read_text(errors="ignore"), residue_scores))

            show_surface = st.checkbox(
                "Show surface",
                value=True,
                key=f"{run_dir.name}_scannet_surface",
                help="Toggle surface rendering. Residues are colored by ScanNet probability in the PDB B-factor column.",
            )
            representation = "molecular-surface+ball-and-stick" if show_surface else "ball-and-stick"
            molstar_custom_component(
                [
                    StructureVisualization(
                        pdb=binding_pdb.read_text(errors="ignore"),
                        color="uncertainty",  # type: ignore[arg-type]
                        color_params=_molstar_binding_color_params(),
                        representation_type=representation,  # type: ignore[arg-type]
                    )
                ],
                key=f"{run_dir.name}_scannet_molstar_{selected_table.stem}_{bool(show_surface)}",
                height=720,
                show_controls=True,
                selection_mode=True,
                html_filename=f"{run_dir.name}_scannet_ppi",
            )
            st.caption(
                "Mol* uses ScanNet binding-site probabilities written into the PDB B-factor column. Red marks stronger predicted PPI residues."
            )
            st.markdown(
                """
<div style="display:flex; align-items:center; gap:10px; margin:4px 0 14px 0; max-width:520px;">
  <span style="font-size:12px; color:#4b5563;">weak</span>
  <div style="height:12px; flex:1; border-radius:6px; border:1px solid #d1d5db; background:linear-gradient(90deg, rgb(25,82,230), rgb(245,245,245), rgb(215,35,35));"></div>
  <span style="font-size:12px; color:#4b5563;">strong PPI</span>
</div>
""",
                unsafe_allow_html=True,
            )
            st.subheader("Predicted PPI Residues")
            threshold = st.slider(
                "Minimum binding-site probability",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key=f"{run_dir.name}_scannet_threshold",
            )
            display_table = table[table["score"] >= threshold].copy()
            display_table = display_table.sort_values(["score", "chain", "residue"], ascending=[False, True, True])
            st.dataframe(
                display_table[["chain", "residue", "amino_acid", "score"]],
                hide_index=True,
                width="stretch",
                column_config={"score": st.column_config.NumberColumn("binding-site probability", format="%.3f")},
            )

    if script_files:
        with st.expander("Open in ChimeraX / helper scripts", expanded=False):
            for path in script_files:
                st.write(str(path.relative_to(run_dir)))
                with st.expander(f"Preview {path.name}", expanded=False):
                    st.code(path.read_text(errors="ignore")[:10000], language="text")


def _show_pesto_results(run_dir: Path) -> None:
    root = run_dir / "artifacts" / "pesto"
    if not root.exists():
        return
    score_tables = sorted((root / "output").glob("*.csv"))
    scored_pdbs = sorted((root / "output").glob("*.pdb"))
    source_pdb = root / "input" / "target.pdb"
    if not score_tables:
        return

    st.subheader("PeSTo PPI Prediction")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Prediction tables", len(score_tables))
    metric_cols[1].metric("Scored PDBs", len(scored_pdbs))
    metric_cols[2].metric("Model", "i_v4_1")

    selected_table = st.selectbox(
        "PeSTo prediction table",
        score_tables,
        format_func=lambda path: path.name,
        key=f"{run_dir.name}_pesto_table",
    )
    try:
        table = pd.read_csv(selected_table)
    except Exception as exc:
        st.warning(f"Could not read PeSTo table {selected_table.name}: {exc}")
        return
    required = {"chain", "residue", "amino_acid", "score"}
    if table.empty or not required.issubset(table.columns):
        st.warning("The PeSTo output table does not contain the expected normalized residue columns.")
        return

    table = table.copy()
    table["chain"] = table["chain"].astype(str)
    table["residue"] = table["residue"].astype(str)
    table["score"] = pd.to_numeric(table["score"], errors="coerce").fillna(0.0)
    threshold = st.slider(
        "Minimum PeSTo interface probability",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        key=f"{run_dir.name}_pesto_threshold",
        help="Filter the predicted-residue table. The default cutoff is 0.50.",
    )
    pdb_path = source_pdb if source_pdb.exists() else (scored_pdbs[0] if scored_pdbs else None)
    if pdb_path is not None:
        residue_scores = {
            (str(row["chain"]), str(row["residue"])): float(row["score"])
            for row in table.to_dict(orient="records")
        }
        visualization_dir = root / "visualization"
        visualization_dir.mkdir(parents=True, exist_ok=True)
        binding_pdb = visualization_dir / "pesto_ppi_bfactor.pdb"
        binding_pdb.write_text(_pdb_with_residue_bfactors(pdb_path.read_text(errors="ignore"), residue_scores))
        show_surface = st.checkbox(
            "Show surface",
            value=True,
            key=f"{run_dir.name}_pesto_surface",
            help="Toggle molecular-surface rendering for the PeSTo residue probabilities.",
        )
        representation = "molecular-surface+ball-and-stick" if show_surface else "ball-and-stick"
        molstar_custom_component(
            [
                StructureVisualization(
                    pdb=binding_pdb.read_text(errors="ignore"),
                    color="uncertainty",  # type: ignore[arg-type]
                    color_params=_molstar_binding_color_params(),
                    representation_type=representation,  # type: ignore[arg-type]
                )
            ],
            key=f"{run_dir.name}_pesto_molstar_{bool(show_surface)}",
            height=720,
            show_controls=True,
            selection_mode=True,
            html_filename=f"{run_dir.name}_pesto_ppi",
        )
        st.caption("Mol* colors original-numbered residues by PeSTo protein-interface probability; red marks stronger predictions.")

    display_table = table[table["score"] >= threshold].copy()
    display_table = display_table.sort_values(["score", "chain", "residue"], ascending=[False, True, True])
    filtered_csv = root / "output" / f"pesto_possible_interacting_residues_threshold_{threshold:.2f}.csv"
    if st.button(
        "Create possible-interacting-residues CSV",
        key=f"{run_dir.name}_pesto_write_threshold_csv",
        help="Writes only residues at or above the selected threshold; the raw PeSTo score CSV is retained.",
    ):
        display_table.to_csv(filtered_csv, index=False)
        st.success(f"Created {filtered_csv.name} with {len(display_table)} possible interacting residues.")
    st.dataframe(
        display_table[["chain", "residue", "amino_acid", "residue_name", "score"]],
        hide_index=True,
        width="stretch",
        column_config={"score": st.column_config.NumberColumn("interface probability", format="%.3f")},
    )


def _show_masif_seed_visualization(run_dir: Path) -> None:
    root = run_dir / "artifacts" / "masif_seed"
    if not root.exists():
        return
    surface_files = sorted((root / "pred_surfaces").glob("*.ply"))
    prediction_files = sorted((root / "pred_data").glob("pred_*.npy"))
    target_pdb = root / "input" / "target.pdb"
    if not surface_files or not target_pdb.exists():
        return

    st.subheader("MaSIF Surface Prediction")
    surface_path = surface_files[0]
    surface_df = _read_masif_surface(surface_path)
    if surface_df is None or surface_df.empty or not {"x", "y", "z"}.issubset(surface_df.columns):
        return
    prediction_path = prediction_files[0] if prediction_files else None
    scores = _read_masif_scores(prediction_path, surface_df)
    surface_df = surface_df.copy()
    surface_df["binding_score"] = scores

    threshold = st.slider(
        "Interface score threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Scores above this value are highlighted in the 3D viewer and counted as predicted interface surface.",
    )
    atom_cutoff = st.slider(
        "Surface-to-atom mapping cutoff (A)",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.25,
        help="Nearest surface points farther than this distance from an atom are ignored when writing B-factor PDBs.",
    )

    high_count = int(np.sum(scores >= threshold))
    cols = st.columns(4)
    cols[0].metric("Surface vertices", f"{len(surface_df):,}")
    cols[1].metric("Highlighted", f"{high_count:,}")
    cols[2].metric("Max score", f"{float(np.max(scores)):.3f}" if len(scores) else "n/a")
    cols[3].metric("Mean score", f"{float(np.mean(scores)):.3f}" if len(scores) else "n/a")

    output_dir = root / "visualization"
    try:
        pdb_paths = _write_masif_binding_pdbs(
            output_dir=output_dir,
            target_pdb=target_pdb,
            surface_df=surface_df,
            scores=scores,
            atom_distance_cutoff=atom_cutoff,
        )
    except Exception as exc:
        st.warning(f"Could not create MaSIF B-factor PDBs: {exc}")
        pdb_paths = {}

    pointcloud_path = pdb_paths.get("pointcloud")
    residue_path = pdb_paths.get("residue")
    atom_path = pdb_paths.get("atom")
    if pointcloud_path and residue_path and pointcloud_path.exists() and residue_path.exists():
        selected_residue_key = f"{run_dir.name}_masif_selected_residue"
        selected_residue = st.session_state.get(selected_residue_key)
        viewer_cols = st.columns([1, 3])
        with viewer_cols[0]:
            show_surface = st.checkbox(
                "Show surface",
                value=True,
                key=f"{run_dir.name}_masif_molstar_surface",
                help="Toggle molecular-surface rendering. Ball-and-stick residues keep the same B-factor score coloring.",
            )
        representation = "molecular-surface+ball-and-stick" if show_surface else "ball-and-stick"
        selected_selection = _masif_selection_string(selected_residue if isinstance(selected_residue, dict) else None)
        molstar_custom_component(
            [
                StructureVisualization(
                    pdb=residue_path.read_text(errors="ignore"),
                    color="uncertainty",  # type: ignore[arg-type]
                    color_params=_molstar_binding_color_params(),
                    representation_type=representation,  # type: ignore[arg-type]
                    highlighted_selections=[selected_selection] if selected_selection else None,
                )
            ],
            key=f"{run_dir.name}_masif_molstar_{bool(show_surface)}_{selected_selection or 'none'}",
            height=720,
            show_controls=True,
            selection_mode=True,
            html_filename=f"{run_dir.name}_masif_molstar",
        )
        st.caption(
            "Mol* uses the MaSIF score stored in the PDB B-factor column. Surface and residues use the same blue-white-red scale; use the Mol* sequence/selection panel or the residue table below to inspect specific amino acids."
        )
        st.markdown(
            """
<div style="display:flex; align-items:center; gap:10px; margin:4px 0 14px 0; max-width:520px;">
  <span style="font-size:12px; color:#4b5563;">weak</span>
  <div style="height:12px; flex:1; border-radius:6px; border:1px solid #d1d5db; background:linear-gradient(90deg, rgb(25,82,230), rgb(245,245,245), rgb(215,35,35));"></div>
  <span style="font-size:12px; color:#4b5563;">strong binding</span>
</div>
""",
            unsafe_allow_html=True,
        )
        residue_table = _residue_bfactor_table(residue_path.read_text(errors="ignore"), threshold)
        if not residue_table.empty:
            show_only_hotspots = st.checkbox(
                "Show only residues above threshold",
                value=True,
                key=f"{run_dir.name}_masif_residue_table_threshold",
            )
            display_table = residue_table[residue_table["above_threshold"]].copy() if show_only_hotspots else residue_table.copy()
            st.subheader("Predicted Interacting Amino Acids")
            residue_event = st.dataframe(
                display_table[
                    [
                        "chain",
                        "residue",
                        "insertion",
                        "amino_acid",
                        "binding_score",
                        "binding_bfactor",
                        "atom_count",
                    ]
                ],
                hide_index=True,
                width="stretch",
                key=f"{run_dir.name}_masif_residue_table",
                on_select="rerun",
                selection_mode="single-row",
                column_config={
                    "binding_score": st.column_config.NumberColumn("binding score", format="%.3f"),
                    "binding_bfactor": st.column_config.NumberColumn("B-factor", format="%.2f"),
                },
            )
            selected_rows = getattr(getattr(residue_event, "selection", None), "rows", []) or []
            if selected_rows:
                selected_index = int(selected_rows[0])
                if 0 <= selected_index < len(display_table):
                    row = display_table.iloc[selected_index]
                    selected_payload = {
                        "chain": str(row.get("chain") or ""),
                        "residue": str(row.get("residue") or ""),
                        "insertion": str(row.get("insertion") or ""),
                        "amino_acid": str(row.get("amino_acid") or ""),
                        "binding_score": float(row.get("binding_score") or 0.0),
                        "binding_bfactor": float(row.get("binding_bfactor") or 0.0),
                    }
                    if st.session_state.get(selected_residue_key) != selected_payload:
                        st.session_state[selected_residue_key] = selected_payload
                        st.rerun()
            if isinstance(st.session_state.get(selected_residue_key), dict):
                selected = st.session_state[selected_residue_key]
                st.caption(
                    f"Selected residue: {selected.get('chain')}:{selected.get('amino_acid')}{selected.get('residue')} "
                    f"(score {float(selected.get('binding_score') or 0.0):.3f})."
                )

    with st.expander("Open in PyMOL or ChimeraX", expanded=False):
        if residue_path and atom_path and pointcloud_path:
            st.write(str(residue_path.relative_to(run_dir)))
            st.write(str(atom_path.relative_to(run_dir)))
            st.write(str(pointcloud_path.relative_to(run_dir)))
            existing_pse = output_dir / "masif_binding_session.pse"
            if st.button(
                "Create PyMOL session",
                key=f"{run_dir.name}_masif_create_pse",
                help="Runs PyMOL in quiet command-line mode and saves a .pse session next to the MaSIF visualization PDBs.",
            ):
                pse_path, pml_path, error = _ensure_masif_pymol_session(
                    output_dir=output_dir,
                    residue_pdb=residue_path,
                    atom_pdb=atom_path,
                    pointcloud_pdb=pointcloud_path,
                    threshold=threshold,
                )
                if error:
                    st.warning(error)
                elif pse_path:
                    st.success(f"Created {pse_path.relative_to(run_dir)}")
                st.caption(f"PyMOL script: {pml_path.relative_to(run_dir)}")
            if existing_pse.exists():
                st.download_button(
                    "Download PyMOL session",
                    data=existing_pse.read_bytes(),
                    file_name=existing_pse.name,
                    mime="application/octet-stream",
                    key=f"{run_dir.name}_masif_download_pse",
                )
            st.code(
                "\n".join(
                    [
                        f"load {residue_path}",
                        f"load {pointcloud_path}",
                        "spectrum b, blue_white_red, minimum=0, maximum=100",
                        "show cartoon, masif_per_residue_binding",
                        "show spheres, masif_pointcloud_binding",
                        "set sphere_scale, 0.25, masif_pointcloud_binding",
                    ]
                ),
                language="text",
            )


def _show_benchmark_structure_viewer(
    run_dir: Path,
    benchmark_dir: Path,
    merged_metrics: pd.DataFrame | None,
    *,
    is_refolding_evaluation: bool = False,
) -> None:
    run_metadata = read_json(run_dir / "metadata.json")
    input_payload = read_json(run_dir / "input.json")
    is_capacity_view = bool(
        run_metadata.get("capacity_parent_run_id")
        or run_metadata.get("capacity_matrix_mode")
        or input_payload.get("job_type") == "refolding_capacity_benchmark"
        or (run_dir / "artifacts" / "capacity_matrix.csv").exists()
    )
    st.subheader("Predicted Structure Overlay")
    if is_capacity_view:
        st.caption(
            "Select one capacity system and inspect the engine prediction. "
            "Capacity views show native predicted coordinates without the synthetic input/reference complex."
        )
    else:
        st.caption(
            "Select one design and overlay the available engine predictions in the same coordinate frame. "
            "If an input/reference structure is available it is shown in gray."
        )
    records = _benchmark_structure_records(
        run_dir,
        benchmark_dir,
        allow_input_fallback=not is_capacity_view,
    )
    if records.empty:
        st.info("No predicted PDB/mmCIF structures were found for this benchmark run.")
        return
    metadata = pd.DataFrame()
    if merged_metrics is not None and not merged_metrics.empty and "binder_id" in merged_metrics.columns:
        allowed_binder_ids = set(merged_metrics["binder_id"].dropna().astype(str))
        records = records[records["binder_id"].astype(str).isin(allowed_binder_ids)].copy()
        columns = [
            col
            for col in ["binder_id", "target_id", "target_chains", "binder", "label", "source"]
            if col in merged_metrics.columns
        ]
        metadata = merged_metrics[columns].drop_duplicates(subset=["binder_id"]).copy()
    if not metadata.empty:
        records = records.merge(metadata, on="binder_id", how="left", suffixes=("", "_input"))
        if "target_id_input" in records.columns:
            records["target_id"] = records["target_id"].fillna(records["target_id_input"])
    label_source_column = next((column for column in ["binder", "label", "binder_input", "label_input"] if column in records.columns), None)
    class_filter = "All"
    has_known_labels = False
    if label_source_column:
        known_labels = records[label_source_column].map(_truthy_benchmark_label)
        has_known_labels = bool(known_labels.notna().any())
    if not is_refolding_evaluation and has_known_labels:
        class_filter = st.segmented_control(
            "Design class",
            ["All", "Binders", "Nonbinders"],
            default="All",
            key=f"{run_dir.name}_benchmark_structure_class_filter_v1",
            help="Filter the design selector by the benchmark binder/nonbinder label.",
        )
    if label_source_column and has_known_labels:
        records["_structure_label"] = records[label_source_column].map(_truthy_benchmark_label)
        if class_filter == "Binders":
            records = records[records["_structure_label"] == 1].copy()
        elif class_filter == "Nonbinders":
            records = records[records["_structure_label"] == 0].copy()
        label_by_design = records.drop_duplicates("binder_id").set_index("binder_id")["_structure_label"]
        binder_count = int((label_by_design == 1).sum())
        nonbinder_count = int((label_by_design == 0).sum())
        unknown_count = int(label_by_design.isna().sum())
        st.caption(
            f"Designs after class filter: {len(label_by_design)} "
            f"({binder_count} binders / {nonbinder_count} nonbinders"
            + (f" / {unknown_count} unknown" if unknown_count else "")
            + ")"
        )
    elif not is_refolding_evaluation and class_filter != "All":
        st.info("This benchmark run does not expose binder/nonbinder labels for structure filtering.")
    engine_count_by_design = (
        records.assign(_binder_id=records["binder_id"].astype(str))
        .groupby("_binder_id")["engine"]
        .nunique()
        .to_dict()
    )
    max_engine_count = max(engine_count_by_design.values()) if engine_count_by_design else 0
    design_options = sorted(
        records["binder_id"].dropna().astype(str).unique(),
        key=lambda design: (-int(engine_count_by_design.get(str(design), 0)), str(design)),
    )
    if not design_options:
        st.info("No designs match the selected class filter.")
        return
    ranked_design = None
    if not is_capacity_view:
        ranked_design = _ranked_design_selector(
            benchmark_dir,
            records,
            merged_metrics,
            run_key=run_dir.name,
            allow_current_best=not is_refolding_evaluation,
            preset_only=is_refolding_evaluation,
            run_dir=run_dir,
        )
    if is_refolding_evaluation and ranked_design is None and not is_capacity_view:
        st.subheader("Design Selection")
        st.caption(
            "No saved benchmark preset matched this refolding/evaluation run. "
            "Choose a design directly, then compare the available engine structures and per-record metrics."
        )
    design_key = f"{run_dir.name}_benchmark_structure_design_v1"
    if ranked_design in design_options:
        selected_design = str(ranked_design)
        st.session_state[design_key] = selected_design
    else:
        current_design = st.session_state.get(design_key)
        if current_design not in design_options:
            current_design = design_options[0]
            st.session_state[design_key] = current_design
        current_engine_count = int(engine_count_by_design.get(str(current_design), 0))
        if max_engine_count and current_engine_count < max_engine_count:
            st.caption(
                f"The current design has structures for {current_engine_count} engine(s); "
                f"the best-covered designs have {max_engine_count} engine(s)."
            )
        selected_design = st.selectbox(
            "Design",
            design_options,
            index=design_options.index(str(current_design)),
            key=design_key,
            format_func=lambda design: f"{design} ({int(engine_count_by_design.get(str(design), 0))} engines)",
        )
    ranked_overlay_context = st.session_state.get(f"{run_dir.name}_benchmark_structure_ranked_overlay_context_v1")
    selected_visual_designs = [str(selected_design)]
    if isinstance(ranked_overlay_context, dict) and ranked_overlay_context.get("designs"):
        ranked_design_order = [
            str(design)
            for design in ranked_overlay_context.get("designs", [])
            if str(design) in design_options
        ]
        if selected_design not in ranked_design_order:
            ranked_design_order = [str(selected_design), *ranked_design_order]
        rank_by_design = {
            str(design): value
            for design, value in (ranked_overlay_context.get("rank_by_design") or {}).items()
        }
        value_by_design = {
            str(design): value
            for design, value in (ranked_overlay_context.get("value_by_design") or {}).items()
        }

        def _ranked_design_label(design: str) -> str:
            rank_value = rank_by_design.get(str(design))
            score_value = value_by_design.get(str(design))
            bits = []
            if rank_value is not None:
                try:
                    bits.append(f"#{int(rank_value)}")
                except (TypeError, ValueError):
                    bits.append(f"#{rank_value}")
            bits.append(str(design))
            if score_value is not None:
                try:
                    bits.append(f"{float(score_value):.4g}")
                except (TypeError, ValueError):
                    bits.append(str(score_value))
            return " | ".join(bits)

        selected_visual_designs = st.multiselect(
            "Ranked designs to display",
            ranked_design_order,
            default=[str(selected_design)],
            format_func=_ranked_design_label,
            key=f"{run_dir.name}_benchmark_structure_ranked_designs_v1",
            help=(
                "Select multiple designs from the currently filtered ranked table. "
                "They are aligned on the same target reference and shown together in the structure viewer."
            ),
        )
        if not selected_visual_designs:
            selected_visual_designs = [str(selected_design)]
    selected_visual_designs = [str(design) for design in selected_visual_designs if str(design) in design_options]
    if not selected_visual_designs:
        selected_visual_designs = [str(selected_design)]
    visual_design_order = {design: idx for idx, design in enumerate(selected_visual_designs)}
    design_records = records[records["binder_id"].astype(str).isin(visual_design_order)].copy()
    if len(selected_visual_designs) > 1 and not design_records.empty:
        design_records["_overlay_index"] = design_records["binder_id"].astype(str).map(visual_design_order)
        design_records = design_records.sort_values(["_overlay_index", "engine"])
    observed_engines = set(design_records["engine"].dropna().astype(str))
    engine_order = [
        *ENGINE_COLOR_DOMAIN,
        *[label for label, _key, _csv in STRUCTURE_ENGINE_TABLES if label not in ENGINE_COLOR_DOMAIN],
    ]
    available_engines = [engine for engine in engine_order if engine in observed_engines]
    available_engines.extend(sorted(engine for engine in observed_engines if engine not in set(available_engines)))
    default_structure_engines = available_engines
    ranked_context_engine = ""
    if isinstance(ranked_overlay_context, dict):
        ranked_context_engine = str(ranked_overlay_context.get("engine") or "")
    if len(selected_visual_designs) > 1 and ranked_context_engine in available_engines:
        default_structure_engines = [ranked_context_engine]
    selected_engines = st.multiselect(
        "Prediction engines",
        available_engines,
        default=default_structure_engines,
        key=f"{run_dir.name}_benchmark_structure_engines_v1",
    )
    if is_capacity_view:
        show_reference = False
    else:
        show_reference = st.checkbox(
            "Show input/reference structure",
            value=True,
            key=f"{run_dir.name}_benchmark_structure_reference_v1",
        )
    design_records = design_records[design_records["engine"].astype(str).isin(selected_engines)].copy()
    target_chains = _target_chains_for_design(merged_metrics, selected_design)
    if not target_chains:
        target_chains = _target_chains_from_benchmark_row(
            _benchmark_dataset_row_for_design(run_dir, selected_design, benchmark_dir)
        )
    (
        _source_input_reference_path,
        source_input_target_chains,
        source_input_metadata_path,
    ) = _source_input_reference_for_design(run_dir, selected_design, benchmark_dir)
    if not target_chains and is_refolding_evaluation and source_input_target_chains:
        target_chains = list(source_input_target_chains)
    input_reference_path = _source_input_reference_path
    if input_reference_path is None and source_input_metadata_path is None:
        input_reference_path = _reference_structure_for_design(run_dir, benchmark_dir, selected_design)
    if is_capacity_view:
        input_reference_path = None
    if _source_input_reference_path is not None and input_reference_path != _source_input_reference_path and not is_capacity_view:
        if is_refolding_evaluation:
            st.warning(
                "A source/import target reference is available, but the viewer selected a derived target reference. "
                "Using the source/import target reference instead."
            )
        else:
            st.warning(
                "A source/import input complex is available, but the viewer selected a derived input reference. "
                "Using the source/import complex instead."
            )
        input_reference_path = _source_input_reference_path
    target_reference_path = None if is_capacity_view else _target_pdb_reference_for_design(run_dir, selected_design)
    if is_capacity_view:
        align_on_target = False
    else:
        align_on_target = st.checkbox(
            "Align predictions on target chains",
            value=True,
            disabled=not target_chains,
            key=f"{run_dir.name}_benchmark_structure_align_target_v1",
            help="Superpose each selected prediction on the declared target chain(s) before display. Binder coordinates move with the target alignment.",
        )
    target_reference_label = "Prepared target" if is_refolding_evaluation else "Target PDB"
    input_reference_label = "Input target/reference" if is_refolding_evaluation else "Input complex"
    predicted_reference_label = "Predicted target" if is_refolding_evaluation else "Predicted structure"
    alignment_reference_options: list[str] = []
    if target_reference_path is not None:
        alignment_reference_options.append(target_reference_label)
    if input_reference_path is not None:
        alignment_reference_options.append(input_reference_label)
    alignment_reference = alignment_reference_options[0] if alignment_reference_options else predicted_reference_label
    if align_on_target and alignment_reference_options:
        alignment_reference = st.segmented_control(
            "Alignment reference",
            alignment_reference_options,
            default=input_reference_label if input_reference_label in alignment_reference_options else alignment_reference_options[0],
            key=f"{run_dir.name}_benchmark_structure_alignment_reference_v2",
            help=(
                "Prepared target aligns every prediction to the staged target file. "
                "Input target/reference keeps the original imported target coordinates fixed when available."
                if is_refolding_evaluation
                else (
                    "Target PDB aligns every prediction to the standalone target-only structure. "
                    "Input complex keeps the original input complex fixed and aligns predictions onto its target chain."
                )
            ),
        )
    input_reference_alignment_chains = target_chains
    if align_on_target and alignment_reference == input_reference_label and input_reference_path is not None:
        input_reference_chain_options = _chain_ids_for_structure_path(input_reference_path)
        source_default_chains = [chain for chain in source_input_target_chains if chain in input_reference_chain_options]
        if source_default_chains:
            input_reference_alignment_chains = source_default_chains
            st.text_input(
                "Input target chain(s)" if is_refolding_evaluation else "Input-complex target chain(s)",
                value=", ".join(source_default_chains),
                disabled=True,
                key=f"{run_dir.name}_{selected_design}_benchmark_structure_source_input_alignment_chains_v1",
                help=(
                    "Taken from source/import metadata so the original target reference is aligned on its real chain."
                    if is_refolding_evaluation
                    else "Taken from the source/import candidate metadata so the original complex is aligned on its real target chain."
                ),
            )
            if source_input_metadata_path:
                st.caption(
                    "Input/source metadata: "
                    f"`{_path_label(Path(source_input_metadata_path), run_dir)}`"
                )
        else:
            default_input_chains = [chain for chain in target_chains if chain in input_reference_chain_options] or input_reference_chain_options
            input_reference_alignment_chains = st.multiselect(
                "Input target chains to align against" if is_refolding_evaluation else "Input-complex chains to align against",
                input_reference_chain_options,
                default=default_input_chains,
                key=f"{run_dir.name}_{selected_design}_benchmark_structure_input_alignment_chains_v4",
                help=(
                    "Choose the chain(s) in the input/reference target that define the target reference frame."
                    if is_refolding_evaluation
                    else "Choose the chain(s) in the input complex that represent the target reference frame."
                ),
            )
            if source_input_metadata_path:
                st.error(
                    "The source/import candidate metadata was found, but its target chain(s) were not present "
                    + (
                        "in the input target/reference. Refusing to use staged refolding chain IDs as source target chains."
                        if is_refolding_evaluation
                        else "in the input complex. Refusing to use staged refolding chain IDs as original-complex chains."
                    )
                )
        st.caption(f"Input/reference PDB: `{_path_label(input_reference_path, run_dir)}`")
        if source_default_chains and target_chains:
            st.caption(
                ("Input target alignment map: " if is_refolding_evaluation else "Input-complex alignment map: ")
                +
                f"prediction/staged target {', '.join(target_chains)} -> "
                f"{'source input target' if is_refolding_evaluation else 'original input-complex target'} "
                f"{', '.join(source_default_chains)}."
            )
    if align_on_target and alignment_reference == target_reference_label and target_reference_path is not None:
        st.caption(
            f"{'Prepared target PDB' if is_refolding_evaluation else 'Target/reference PDB'}: "
            f"`{_path_label(target_reference_path, run_dir)}`"
        )
    if align_on_target and target_chains:
        if alignment_reference == input_reference_label:
            if source_input_target_chains:
                st.caption(
                    f"Prediction/staged target chains: {', '.join(target_chains)}. "
                    f"{'Source input target chains' if is_refolding_evaluation else 'Original input-complex target chains'}: "
                    f"{', '.join(input_reference_alignment_chains) or 'none'}."
                )
            else:
                st.caption(
                    f"Prediction/staged target chains: {', '.join(target_chains)}. "
                    f"{'Input target alignment chains' if is_refolding_evaluation else 'Input-complex alignment chains'}: "
                    f"{', '.join(input_reference_alignment_chains) or 'none'}."
                )
        elif alignment_reference == target_reference_label:
            st.caption(
                f"Declared target chains: {', '.join(target_chains)}. "
                +
                (
                    "Predictions are superposed on the prepared target reference."
                    if is_refolding_evaluation
                    else "Predictions are superposed on the standalone target PDB by default."
                )
            )
        else:
            st.caption(
                f"Declared target chains: {', '.join(target_chains)}. "
                "Per-engine mapped target chains are used for superposition when chain maps are available."
            )
    elif not target_chains:
        st.caption("Target-chain IDs were not found for this design; showing native coordinates.")
    viewer_layout = st.segmented_control(
        "Structure layout",
        ["Linked 3Dmol grid", "Mol* overlay"],
        default="Linked 3Dmol grid",
        key=f"{run_dir.name}_benchmark_structure_layout_v1",
        help="Linked 3Dmol grid shows one aligned prediction per panel with synchronized camera controls. Mol* overlay stacks selected predictions in one scene.",
    )
    overlay_scope = "Selected ranked designs" if len(selected_visual_designs) > 1 else "Selected design across engines"
    if len(selected_visual_designs) > 1:
        if viewer_layout == "Mol* overlay" and len(design_records) > 150:
            st.warning("Large Mol* overlays can be slow to render; reduce the selected designs if the viewer becomes sluggish.")
        st.caption(
            f"Displaying {len(selected_visual_designs):,} ranked designs"
            + (f" for {', '.join(selected_engines)}" if selected_engines else "")
            + "."
        )
    show_reference_in_grid = False
    if viewer_layout == "Linked 3Dmol grid" and not is_capacity_view:
        show_reference_in_grid = st.checkbox(
            "Show input/reference in each grid panel",
            value=True,
            key=f"{run_dir.name}_benchmark_structure_grid_reference_v1",
            help="Add the input/reference structure in gray to each per-engine panel when available.",
        )
    structures: list[StructureVisualization] = []
    viewer_entries: list[dict[str, object]] = []
    reference_path = None
    if show_reference:
        reference_path = target_reference_path if alignment_reference == target_reference_label else input_reference_path
        if reference_path is None:
            reference_path = input_reference_path or target_reference_path
    structure_paths = [Path(str(row["path"])) for _, row in design_records.iterrows()]
    fixed_path = None
    if alignment_reference == target_reference_label:
        fixed_path = target_reference_path
    elif alignment_reference == input_reference_label:
        fixed_path = input_reference_path
    if fixed_path is None:
        fixed_path = reference_path or (structure_paths[0] if structure_paths else None)
    fixed_target_chains = target_chains
    if fixed_path is not None and alignment_reference == input_reference_label:
        fixed_target_chains = input_reference_alignment_chains
    elif fixed_path is not None and reference_path is None and not design_records.empty:
        fixed_record = design_records[design_records["path"].astype(str) == str(fixed_path)].head(1)
        if not fixed_record.empty:
            mapped_fixed_chains = _engine_target_chains_from_map(
                fixed_record.iloc[0].get("chain_map") if isinstance(fixed_record.iloc[0].get("chain_map"), dict) else {},
                target_chains,
            )
            if mapped_fixed_chains:
                fixed_target_chains = mapped_fixed_chains
    fixed_structure = None
    alignment_rows: list[dict[str, object]] = []
    if align_on_target and target_chains and fixed_path is not None:
        try:
            fixed_structure = _parse_structure_file(fixed_path)
        except Exception as exc:
            st.warning(f"Could not parse the target-alignment reference structure: {exc}")
    if reference_path is not None:
        reference_label = (
            alignment_reference
            if alignment_reference in {target_reference_label, input_reference_label}
            else "Input/reference"
        )
        reference_pdb_text = reference_path.read_text(errors="ignore")
        reference_rmsd = None
        reference_alignment_note = "native coordinates"
        if fixed_structure is not None and align_on_target and target_chains:
            if fixed_path is not None and reference_path == fixed_path:
                reference_alignment_note = f"reference frame ({','.join(fixed_target_chains)})"
            else:
                reference_target_chains = (
                    input_reference_alignment_chains
                    if reference_path == input_reference_path
                    else target_chains
                )
                aligned_reference_text, reference_rmsd, reference_alignment_note = _aligned_structure_text(
                    reference_path,
                    fixed_structure,
                    fixed_target_chains,
                    reference_target_chains,
                )
                if aligned_reference_text:
                    reference_pdb_text = aligned_reference_text
        if reference_label == target_reference_label:
            target_only_text = _structure_chains_to_pdb_text(reference_path, fixed_target_chains)
            if target_only_text:
                reference_pdb_text = target_only_text
                reference_alignment_note += "; target chains only"
        reference_structure = StructureVisualization(
            pdb=reference_pdb_text,
            color="uniform",
            color_params={"value": "0x8a8f98"},
            representation_type="cartoon",
        )
        structures.append(reference_structure)
        viewer_entries.append(
            {
                "engine": reference_label,
                "structure": reference_structure,
                "pdb": reference_structure.pdb,
                "color": "#8a8f98",
                "is_reference": True,
            }
        )
        alignment_rows.append(
            {
                "engine": reference_label,
                "path": str(reference_path),
                "target_alignment_rmsd": reference_rmsd,
                "template_ca_rmsd": reference_rmsd,
                "template_ca_atoms": _alignment_ca_count(reference_alignment_note),
                "template_ca_reference_atoms": _alignment_reference_ca_count(reference_alignment_note),
                "alignment": reference_alignment_note,
            }
        )
    ranked_table_overlay = len(selected_visual_designs) > 1 and "_overlay_index" in design_records.columns
    for _, row in design_records.iterrows():
        path = Path(str(row["path"]))
        overlay_rank = (int(row.get("_overlay_index") or 0) + 1) if ranked_table_overlay else None
        design_label = str(row.get("binder_id") or row.get("candidate_id") or "")
        entry_label = str(row["engine"])
        if overlay_rank is not None:
            entry_label = f"#{overlay_rank} | {design_label} | {row['engine']}"
        color = (
            _rank_overlay_color(int(row.get("_overlay_index") or 0), max(len(selected_visual_designs), 1))
            if ranked_table_overlay
            else _engine_color(row["engine"])
        )
        pdb_text = path.read_text(errors="ignore")
        rmsd = None
        alignment_note = "native coordinates"
        if alignment_reference == input_reference_label:
            # Override refolding has two target namespaces: the prediction/staged target
            # chain(s), usually B, and the original candidate-complex target chain(s),
            # often A.  The prediction must move by its staged target chain(s), not by
            # the source/import chain IDs.  When one original input chain was split
            # into staged fragments, the chain map carries those moving chains in
            # target_fragments.
            moving_target_chains = _engine_target_chains_from_map(
                row.get("chain_map") if isinstance(row.get("chain_map"), dict) else {},
                input_reference_alignment_chains,
            ) or list(target_chains)
        else:
            moving_target_chains = _engine_target_chains_from_map(
                row.get("chain_map") if isinstance(row.get("chain_map"), dict) else {},
                target_chains,
            )
        if not moving_target_chains and fixed_structure is not None and align_on_target and target_chains:
            try:
                moving_target_chains = _infer_matching_target_chains(
                    _parse_structure_file(path),
                    fixed_structure,
                    fixed_target_chains,
                )
            except Exception:
                moving_target_chains = []
        moving_target_chains = moving_target_chains or target_chains
        if fixed_structure is not None and align_on_target and target_chains:
            moving_target_chains = _resolve_moving_target_chains_for_alignment(
                path,
                fixed_structure,
                fixed_target_chains,
                moving_target_chains,
            )
            if fixed_path is not None and path == fixed_path:
                alignment_note = f"reference frame ({','.join(fixed_target_chains)})"
            else:
                aligned_text, rmsd, alignment_note = _aligned_structure_text(
                    path,
                    fixed_structure,
                    fixed_target_chains,
                    moving_target_chains,
                )
                if aligned_text:
                    pdb_text = aligned_text
        structures.append(
            StructureVisualization(
                pdb=pdb_text,
                color="uniform",
                color_params={"value": _molstar_color(color)},
                representation_type="cartoon",
            )
        )
        engine_structure = structures[-1]
        viewer_entries.append(
            {
                "engine": entry_label,
                "candidate_id": row["candidate_id"],
                "binder_id": row.get("binder_id"),
                "ranked_overlay_rank": overlay_rank,
                "structure": engine_structure,
                "pdb": pdb_text,
                "color": color,
                "is_reference": False,
            }
        )
        alignment_rows.append(
            {
                "engine": row["engine"],
                "binder_id": row.get("binder_id"),
                "candidate_id": row["candidate_id"],
                "ranked_overlay_rank": overlay_rank,
                "path": row["path"],
                "target_chains_used": ",".join(moving_target_chains),
                "target_alignment_rmsd": rmsd,
                "template_ca_rmsd": rmsd,
                "template_ca_atoms": _alignment_ca_count(alignment_note),
                "template_ca_reference_atoms": _alignment_reference_ca_count(alignment_note),
                "alignment": alignment_note,
            }
        )
    if not structures:
        st.info("Select at least one available engine, or enable the input/reference structure.")
        return
    target_id = ""
    if "target_id" in design_records.columns and not design_records["target_id"].dropna().empty:
        target_id = str(design_records["target_id"].dropna().iloc[0])
    label_value = ""
    for column in ["binder", "label"]:
        if column in design_records.columns and not design_records[column].dropna().empty:
            parsed = _truthy_benchmark_label(design_records[column].dropna().iloc[0])
            if parsed is not None:
                label_value = "known binder" if parsed else "nonbinder"
                break
    if ranked_table_overlay:
        summary_bits = [
            f"{len(selected_visual_designs)} ranked designs",
            ", ".join(selected_engines),
        ]
    else:
        summary_bits = [selected_design]
    if target_id:
        summary_bits.append(f"target {target_id}")
    if label_value:
        summary_bits.append(label_value)
    st.caption(" | ".join(summary_bits))
    rmsd_rows = [
        {
            "engine": row.get("engine"),
            "candidate_id": row.get("candidate_id"),
            "Template CA RMSD (A)": row.get("template_ca_rmsd"),
            "Aligned CA atoms": row.get("template_ca_atoms"),
            "Reference CA atoms": row.get("template_ca_reference_atoms"),
            "Template coverage": (
                (
                    float(row.get("template_ca_atoms")) / float(row.get("template_ca_reference_atoms"))
                )
                if row.get("template_ca_atoms") is not None
                and row.get("template_ca_reference_atoms") not in (None, 0)
                else None
            ),
            "Target chains": row.get("target_chains_used"),
        }
        for row in alignment_rows
        if not pd.isna(row.get("template_ca_rmsd"))
    ]
    if rmsd_rows:
        rmsd_table = pd.DataFrame(rmsd_rows)
        if "Template CA RMSD (A)" in rmsd_table.columns:
            rmsd_table["Template CA RMSD (A)"] = pd.to_numeric(
                rmsd_table["Template CA RMSD (A)"], errors="coerce"
            ).round(3)
        coverage_values = pd.Series(dtype="float64")
        if "Template coverage" in rmsd_table.columns:
            coverage_values = pd.to_numeric(rmsd_table["Template coverage"], errors="coerce")
            rmsd_table["Template coverage"] = coverage_values.map(lambda value: f"{value:.0%}" if pd.notna(value) else "")
        st.dataframe(rmsd_table, hide_index=True, width="stretch")
        incomplete_rows = rmsd_table[coverage_values.notna() & (coverage_values < 0.8)] if not coverage_values.empty else pd.DataFrame()
        if not incomplete_rows.empty:
            engines = ", ".join(incomplete_rows["engine"].dropna().astype(str).tolist())
            st.warning(
                "Template RMSD is based on incomplete CA coverage"
                + (f" for {engines}" if engines else "")
                + "; use it as a partial-fragment alignment score, not as full target preservation."
            )
    if viewer_layout == "Mol* overlay":
        molstar_custom_component(
            structures=structures,
            key=(
                f"{run_dir.name}_benchmark_structure_viewer_"
                f"{selected_design}_{'_'.join(selected_engines)}_{bool(reference_path)}_"
                f"{bool(align_on_target)}_{alignment_reference}_{'-'.join(fixed_target_chains)}_"
                f"{overlay_scope}_{'-'.join(selected_visual_designs[:12])}_overlay"
            ),
            height=720,
            show_controls=True,
            html_filename=(
                f"{run_dir.name}_{selected_engines[0]}_ranked_overlay"
                if ranked_table_overlay and selected_engines
                else f"{run_dir.name}_{selected_design}_overlay"
            ),
        )
    else:
        reference_entry = next((entry for entry in viewer_entries if entry.get("is_reference")), None)
        engine_entries = [entry for entry in viewer_entries if not entry.get("is_reference")]
        if not engine_entries and reference_entry is not None:
            engine_entries = [reference_entry]
        if len(engine_entries) > 9:
            st.caption("Showing the first 9 selected structures in the linked grid.")
        grid_entries = engine_entries[:9]
        grid_rows = max(1, (len(grid_entries) + 2) // 3)
        grid_height = max(360, grid_rows * 320)
        components.html(
            _py3dmol_grid_html(
                grid_entries,
                reference_pdb=str(reference_entry.get("pdb")) if reference_entry else None,
                include_reference=show_reference_in_grid and reference_entry is not None,
                height=grid_height,
            ),
            height=grid_height + 70,
            scrolling=False,
        )
    with st.expander("Displayed structures", expanded=False):
        st.dataframe(pd.DataFrame(alignment_rows), hide_index=True, width="stretch")


BENCHMARK_MATRIX_ENGINE_ORDER = [
    "AF3",
    "AF2-IG",
    "Boltz-2",
    "BoltzGen Fold",
    "ColabFold",
    "ESMFold2",
    "Protenix v0.5",
    "Protenix v1",
    "Protenix v2",
    "RF3",
]


def _show_benchmark_matrix_workspace_results(run_dir: Path, result: dict) -> None:
    st.header("Benchmark Matrix Workspace")
    input_json = read_json(run_dir / "input.json")
    params = input_json.get("params") if isinstance(input_json.get("params"), dict) else {}
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    workspace_path = run_dir / "artifacts" / "benchmark" / "matrix_workspace.json"
    sources_path = run_dir / "artifacts" / "benchmark" / "benchmark_matrix_workspace_sources.csv"
    workspace = read_json(workspace_path) if workspace_path.exists() else {}

    workspace_name = str(
        metrics.get("workspace_name")
        or workspace.get("name")
        or params.get("workspace_name")
        or "Benchmark matrix workspace"
    )
    st.caption(
        f"{workspace_name}. This run stores which source benchmark contributes each engine-target cell. "
        "Merged plots and per-record tables are created by the optional collection step."
    )
    cols = st.columns(4)
    cols[0].metric("Cells", metrics.get("cell_count") or workspace.get("cell_count") or "n/a")
    cols[1].metric("Targets", metrics.get("target_count") or len(workspace.get("targets") or []) or "n/a")
    cols[2].metric("Engines", metrics.get("engine_count") or len(workspace.get("engines") or []) or "n/a")
    cols[3].metric("Source runs", metrics.get("source_run_count") or workspace.get("source_count") or "n/a")

    sources = _read_csv(sources_path)
    if sources is None or sources.empty:
        st.info("No workspace source mapping table was found.")
        return

    sources = sources.copy()
    sources["run_id"] = sources["run_id"].astype(str)
    sources["engine"] = sources["engine"].astype(str)
    sources["target_id"] = sources["target_id"].astype(str)
    sources["status"] = "completed"
    sources["result"] = sources["run_id"].map(lambda value: result_link("benchmark", value, "Open"))

    target_order = sorted(
        [str(target) for target in sources["target_id"].dropna().unique().tolist() if str(target)]
    )
    engine_values = set(sources["engine"].dropna().astype(str))
    engine_order = [
        engine for engine in BENCHMARK_MATRIX_ENGINE_ORDER if engine in engine_values
    ] + sorted(engine for engine in engine_values if engine not in BENCHMARK_MATRIX_ENGINE_ORDER)
    matrix_height = max(320, min(900, 72 * max(1, len(engine_order))))
    matrix_width = max(520, min(1400, 90 * max(1, len(target_order))))
    chart = (
        alt.Chart(sources)
        .mark_rect()
        .encode(
            x=alt.X("target_id:N", title="target", sort=target_order),
            y=alt.Y("engine:N", title="engine", sort=engine_order),
            color=alt.Color(
                "status:N",
                title="status",
                scale=alt.Scale(domain=["completed"], range=["#0B74C9"]),
            ),
            tooltip=["engine", "target_id", "run_id", "order"],
        )
        .properties(width=matrix_width, height=matrix_height)
    )
    st.altair_chart(chart, width="content")

    source_counts = (
        sources.groupby("run_id", dropna=False)
        .agg(cells=("target_id", "count"), engines=("engine", "nunique"), targets=("target_id", "nunique"))
        .reset_index()
    )
    source_counts["result"] = source_counts["run_id"].map(lambda value: result_link("benchmark", value, "Open"))
    with st.expander("Source runs", expanded=True):
        st.dataframe(
            source_counts[["result", "run_id", "cells", "engines", "targets"]],
            hide_index=True,
            width="stretch",
            column_config={"result": st.column_config.LinkColumn("result", display_text="Open")},
        )

    with st.expander("Engine-target source mapping", expanded=True):
        display = sources[["result", "order", "engine", "target_id", "run_id"]].sort_values(
            ["engine", "target_id", "order"]
        )
        st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            column_config={"result": st.column_config.LinkColumn("result", display_text="Open")},
        )

    st.info(
        "To get AP/AUROC plots and per-record tables, go back to Benchmark > Matrix, select this umbrella, "
        "then use 'Optional: create a merged benchmark collection from this umbrella'."
    )


def _show_capacity_benchmark_results(run_dir: Path, result: dict) -> None:
    from mn_protein_design.workflows.capacity_benchmark import (
        ENGINE_LABELS as CAPACITY_ENGINE_LABELS,
        capacity_child_rows,
        capacity_limit_rows,
        create_practical_capacity_benchmark_from_parent,
        extend_refolding_capacity_benchmark,
    )

    st.header("Capacity Benchmark Results")
    input_json = read_json(run_dir / "input.json")
    params = input_json.get("params") if isinstance(input_json.get("params"), dict) else {}
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    children = capacity_child_rows(run_dir)
    cols = st.columns(6)
    cols[0].metric("Systems", metrics.get("system_count", ""))
    cols[1].metric("Cells", metrics.get("cell_count", len(children)))
    cols[2].metric("Completed", sum(1 for row in children if row.get("status") == "completed"))
    cols[3].metric("Failed", sum(1 for row in children if row.get("status") == "failed"))
    cols[4].metric("Skipped", sum(1 for row in children if row.get("status") == "skipped"))
    cols[5].metric("Running/queued", sum(1 for row in children if row.get("status") in {"queued", "running", "preparing"}))
    st.caption(
        f"{params.get('benchmark_name') or run_dir.name} | preset: {params.get('preset') or 'n/a'} | "
        f"GPU: {params.get('capacity_device') or params.get('gpu_device') or 'n/a'} | "
        f"target: {params.get('target_name') or Path(str(params.get('target_pdb') or '')).stem or 'n/a'} | "
        f"chains: {', '.join(params.get('target_chains') or []) or 'n/a'}"
    )
    source_capacity_run_id = str(params.get("source_capacity_run_id") or "")
    if source_capacity_run_id:
        st.caption(f"Practical follow-up seeded from capacity umbrella `{source_capacity_run_id}`.")

    profile_rows = capacity_limit_rows(parent_run_dir=run_dir)
    if profile_rows:
        st.subheader("Capacity Profile")
        st.dataframe(
            pd.DataFrame(profile_rows),
            hide_index=True,
            width="stretch",
            column_order=[
                "engine",
                "preset",
                "gpu_device",
                "tested_cells",
                "completed_cells",
                "failed_cells",
                "max_success_total_length",
                "min_failed_total_length",
                "min_oom_total_length",
                "last_failure_kind",
            ],
        )

    with st.expander("Start practical benchmark from successful capacity limits", expanded=False):
        st.caption(
            "Creates a separate practical umbrella for the same target. Each engine starts at its own "
            "largest completed system from this capacity run."
        )
        successful_children = [
            row for row in children if str(row.get("status") or "") == "completed"
        ]
        practical_seeds: list[dict[str, object]] = []
        if successful_children:
            success_df = pd.DataFrame(successful_children)
            success_df["total_length_numeric"] = pd.to_numeric(
                success_df["total_length"], errors="coerce"
            )
            for _engine_key, group in success_df.groupby("engine_key", dropna=True):
                valid = group.dropna(subset=["total_length_numeric"])
                if valid.empty:
                    continue
                best = valid.sort_values("total_length_numeric").iloc[-1]
                practical_seeds.append(
                    {
                        "engine": best.get("engine"),
                        "copy_count": best.get("copy_count"),
                        "sequence_length": best.get("sequence_length"),
                        "total_sequence_length": int(best["total_length_numeric"]),
                    }
                )
        if practical_seeds:
            st.dataframe(pd.DataFrame(practical_seeds), hide_index=True, width="stretch")
            practical_name = st.text_input(
                "Practical umbrella name",
                value=f"{params.get('benchmark_name') or run_dir.name} - practical",
                key=f"capacity_practical_name_{run_dir.name}",
            )
            practical_gpu = st.text_input(
                "GPU device",
                value=str(params.get("capacity_device") or "0"),
                key=f"capacity_practical_gpu_{run_dir.name}",
            )
            practical_launch = st.checkbox(
                "Launch practical cells now",
                value=True,
                key=f"capacity_practical_launch_{run_dir.name}",
            )
            if st.button(
                "Create practical benchmark umbrella",
                type="primary",
                key=f"capacity_create_practical_{run_dir.name}",
            ):
                try:
                    practical_run = create_practical_capacity_benchmark_from_parent(
                        run_dir,
                        benchmark_name=practical_name,
                        gpu_device=practical_gpu,
                        launch=practical_launch,
                    )
                    st.success(f"Practical umbrella created: {practical_run.name}")
                    st.link_button(
                        "Open practical results",
                        f"/results?task_group=benchmark&run_id={practical_run.name}",
                    )
                except Exception as exc:
                    st.error(str(exc))
        else:
            st.info("No completed cells are available as practical starting points.")

    with st.expander("Extend or recalculate this benchmark", expanded=False):
        st.caption(
            "Adds systems or engines to this umbrella. Recalculation preserves the old cell for provenance "
            "and replaces it in the active matrix."
        )
        selected_engines = [str(engine) for engine in (params.get("engines") or []) if str(engine)]
        missing_engines = [
            engine for engine in CAPACITY_ENGINE_LABELS if engine not in selected_engines
        ]
        add_engines = st.multiselect(
            "Add folding engines",
            missing_engines,
            default=["boltzgen_fold"] if "boltzgen_fold" in missing_engines else [],
            format_func=lambda key: CAPACITY_ENGINE_LABELS.get(key, key),
            key=f"capacity_add_engines_{run_dir.name}",
        )
        capacity_matrix_mode = str(params.get("matrix_mode") or "")
        existing_copy_counts = sorted(
            {
                int(float(str(value)))
                for value in (params.get("copy_counts") or [])
                if str(value).strip()
            }
        )
        add_copy_text = ""
        if capacity_matrix_mode == "sequence_copy_multimer":
            add_copy_text = st.text_input(
                "Add multimer copy counts",
                value="",
                placeholder=str((max(existing_copy_counts) + 1) if existing_copy_counts else 2),
                key=f"capacity_add_copies_{run_dir.name}",
            )
        add_target_entries: list[dict[str, object]] = []
        if capacity_matrix_mode == "target_panel":
            try:
                from mn_protein_design.workflows.detection import ppi_target_jobs
                from mn_protein_design.workflows.refolding import _sequences_by_chain

                existing_target_keys = {
                    (
                        str(entry.get("target_pdb") or ""),
                        str(entry.get("chain") or ""),
                        str(entry.get("copy_count") or 1),
                    )
                    for entry in (params.get("target_panel_entries") or [])
                    if isinstance(entry, dict)
                }
                try:
                    matrix_df = pd.read_csv(run_dir / "artifacts" / "capacity_matrix.csv")
                    for row in matrix_df.to_dict(orient="records"):
                        existing_target_keys.add(
                            (
                                str(row.get("target_pdb") or ""),
                                str(row.get("source_chain") or ""),
                                str(row.get("copy_count") or 1),
                            )
                        )
                except Exception:
                    pass
                target_options: list[tuple[int, str, str, dict[str, object]]] = []
                used_labels: set[str] = set()
                for target in ppi_target_jobs():
                    target_pdb = Path(str(target.get("target_pdb") or "")).expanduser()
                    if not target_pdb.exists():
                        continue
                    try:
                        sequences = _sequences_by_chain(target_pdb)
                    except Exception:
                        sequences = {}
                    declared_chains = [
                        str(chain)
                        for chain in (target.get("chains") or sequences.keys())
                        if str(chain)
                    ]
                    for chain in declared_chains:
                        key = (str(target_pdb), chain, "1")
                        if key in existing_target_keys:
                            continue
                        sequence = str(sequences.get(chain) or "").replace("X", "")
                        length = len(sequence)
                        if length <= 0:
                            continue
                        target_name = str(target.get("target_name") or target_pdb.stem)
                        source = str(
                            target.get("source_label")
                            or target.get("source_category")
                            or target.get("tool")
                            or ""
                        )
                        label = f"{target_name} | chain {chain} | {length} aa"
                        if source:
                            label = f"{label} | {source}"
                        base_label = label
                        suffix = 2
                        while label in used_labels:
                            label = f"{base_label} ({suffix})"
                            suffix += 1
                        used_labels.add(label)
                        target_options.append(
                            (
                                length,
                                target_name.lower(),
                                label,
                                {
                                    "target_name": target_name,
                                    "target_pdb": str(target_pdb),
                                    "chain": chain,
                                    "copy_count": 1,
                                },
                            )
                        )
                target_options.sort(key=lambda item: (item[0], item[1], item[2]))
                target_labels = [label for _length, _name, label, _entry in target_options]
                target_by_label = {label: entry for _length, _name, label, entry in target_options}
                selected_target_labels = st.multiselect(
                    "Add target-panel targets",
                    target_labels,
                    default=[],
                    help="Adds selected target chains as monomer systems to this umbrella and queues them for the benchmark engines.",
                    key=f"capacity_add_targets_{run_dir.name}",
                )
                add_target_entries = [
                    dict(target_by_label[label])
                    for label in selected_target_labels
                    if label in target_by_label
                ]
                if not target_labels:
                    st.caption("No additional target-panel targets are available for this umbrella.")
            except Exception as exc:
                st.warning(f"Could not load additional target options: {exc}")
        parsed_copy_counts: list[int] = []
        for part in add_copy_text.replace(";", ",").split(","):
            try:
                value = int(part.strip())
            except ValueError:
                continue
            if value >= 2 and value not in parsed_copy_counts:
                parsed_copy_counts.append(value)
        terminal_children = [
            row
            for row in children
            if str(row.get("status") or "")
            in {"completed", "failed", "stopped", "skipped", "cancelled"}
        ]
        recalc_labels = {
            str(row.get("run_id") or ""): (
                f"{row.get('engine')} | {row.get('total_length')} residues | "
                f"{row.get('status')} | {row.get('job_code')}"
            )
            for row in terminal_children
            if str(row.get("run_id") or "")
        }
        recalculate_ids = st.multiselect(
            "Recalculate selected cells",
            list(recalc_labels),
            format_func=lambda child_run_id: recalc_labels.get(child_run_id, child_run_id),
            key=f"capacity_recalculate_{run_dir.name}",
        )
        launch_extension = st.checkbox(
            "Launch added cells now",
            value=True,
            key=f"capacity_launch_extension_{run_dir.name}",
        )
        if st.button(
            "Apply additions / recalculations",
            type="primary",
            disabled=not (add_engines or parsed_copy_counts or add_target_entries or recalculate_ids),
            key=f"capacity_extend_{run_dir.name}",
        ):
            try:
                extension = extend_refolding_capacity_benchmark(
                    run_dir,
                    engines_to_add=list(add_engines),
                    copy_counts_to_add=parsed_copy_counts,
                    target_panel_entries_to_add=add_target_entries,
                    recalculate_run_ids=list(recalculate_ids),
                    launch=launch_extension,
                )
                added_target_text = (
                    f" ({extension.get('added_targets', 0)} targets)"
                    if extension.get("added_targets")
                    else ""
                )
                st.success(
                    f"Added {extension['added_systems']} systems{added_target_text}, "
                    f"{extension['added_engines']} engines, "
                    f"{extension['added_cells']} cells "
                    f"({extension.get('inherited_failed_cells', 0)} inherited failures), and recalculated "
                    f"{extension['recalculated_cells']} cells."
                )
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    matrix_path = run_dir / "artifacts" / "capacity_matrix.csv"
    if matrix_path.exists():
        with st.expander("System matrix", expanded=False):
            try:
                st.dataframe(pd.read_csv(matrix_path), hide_index=True, width="stretch")
            except Exception as exc:
                st.warning(f"Could not read capacity matrix: {exc}")
    if not children:
        st.info("No child refolding cells were found for this capacity benchmark yet.")
        return
    df = pd.DataFrame(children)
    modes = set(df.get("matrix_mode", pd.Series(dtype=str)).dropna().astype(str))
    if modes and modes <= {"sequence_copy_multimer"}:
        display_cols = [
            "result",
            "engine",
            "status",
            "matrix_mode",
            "sequence_length",
            "copy_count",
            "total_length",
            "peak_gpu_memory_mib",
            "gpu_total_memory_mib",
            "current_phase",
            "failure_kind",
            "worker_error",
            "exception",
            "updated_at",
            "job_code",
            "run_id",
        ]
        display_df = df[[col for col in display_cols if col in df.columns]].copy()
        display_df = display_df.rename(
            columns={
                "total_length": "total_sequence_length",
            }
        )
    else:
        display_cols = [
            "result",
            "engine",
            "status",
            "matrix_mode",
            "copy_count",
            "sequence_length",
            "target_length",
            "binder_length",
            "total_length",
            "peak_gpu_memory_mib",
            "gpu_total_memory_mib",
            "current_phase",
            "failure_kind",
            "worker_error",
            "exception",
            "updated_at",
            "job_code",
            "run_id",
        ]
        display_df = df[[col for col in display_cols if col in df.columns]].copy()
    st.dataframe(
        display_df,
        hide_index=True,
        width="stretch",
        column_config={"result": st.column_config.LinkColumn("result", display_text="Open")},
    )
    plot_df = df.copy()
    plot_df["total_length"] = pd.to_numeric(plot_df["total_length"], errors="coerce")
    folding_only = set(plot_df.get("matrix_mode", pd.Series(dtype=str)).dropna().astype(str)) <= {
        "sequence_copy_multimer",
        "target_panel",
    }
    target_panel_only = set(plot_df.get("matrix_mode", pd.Series(dtype=str)).dropna().astype(str)) <= {"target_panel"}
    if folding_only:
        plot_df["total_sequence_length"] = plot_df["total_length"]
    plot_df["target_label"] = (
        plot_df["target_name"].fillna("").astype(str)
        if "target_name" in plot_df.columns
        else pd.Series([""] * len(plot_df), index=plot_df.index, dtype=str)
    )
    empty_target_label = plot_df["target_label"].str.strip() == ""
    if "candidate_id" in plot_df.columns:
        plot_df.loc[empty_target_label, "target_label"] = plot_df.loc[empty_target_label, "candidate_id"].astype(str)
    if target_panel_only:
        target_order = (
            plot_df[["target_label", "total_length"]]
            .dropna(subset=["target_label"])
            .sort_values(["total_length", "target_label"])
            .drop_duplicates("target_label")["target_label"]
            .astype(str)
            .tolist()
        )
        x_encoding = alt.X("target_label:N", title="target", sort=target_order)
    else:
        plot_df["x_label"] = plot_df["total_length"]
        x_encoding = alt.X("total_length:O", title="total residues")
    if not plot_df.empty and "engine" in plot_df and "total_length" in plot_df:
        status_order = ["completed", "failed", "queued", "running", "preparing", "skipped", "cancelled", "paused"]
        status_colors = ["#0072CE", "#7CC1F2", "#FF2D2D", "#F4A3A8", "#F59E0B", "#9CA3AF", "#6B7280", "#A78BFA"]
        tooltip = (
            ["engine", "target_label", "sequence_length", "copy_count", "total_sequence_length", "peak_gpu_memory_mib", "gpu_total_memory_mib", "status", "failure_kind", "worker_error", "job_code"]
            if folding_only
            else ["engine", "copy_count", "sequence_length", "target_length", "binder_length", "total_length", "peak_gpu_memory_mib", "gpu_total_memory_mib", "status", "failure_kind", "worker_error", "job_code"]
        )
        cell_selector = alt.selection_point(
            name="capacity_cell_pick",
            fields=["run_id"],
            empty=False,
            on="click",
        )
        chart = (
            alt.Chart(plot_df)
            .mark_rect()
            .encode(
                x=x_encoding,
                y=alt.Y("engine:N", title="engine"),
                color=alt.Color(
                    "status:N",
                    title="status",
                    scale=alt.Scale(domain=status_order, range=status_colors),
                ),
                stroke=alt.condition(cell_selector, alt.value("#111827"), alt.value("#FFFFFF")),
                strokeWidth=alt.condition(cell_selector, alt.value(4), alt.value(0.5)),
                tooltip=tooltip,
            )
            .add_params(cell_selector)
            .properties(width=720, height=720)
        )
        cell_event = st.altair_chart(
            chart,
            width="content",
            key=f"capacity_cell_matrix_{run_dir.name}",
            on_select="rerun",
            selection_mode="capacity_cell_pick",
        )
        selection = getattr(cell_event, "selection", None)
        if selection is None and isinstance(cell_event, dict):
            selection = cell_event.get("selection")
        selected_run_id = ""
        if isinstance(selection, dict):
            selected_value = selection.get("capacity_cell_pick")
            selected_records = selected_value if isinstance(selected_value, list) else [selected_value]
            for selected_record in selected_records:
                if not isinstance(selected_record, dict):
                    continue
                run_value = selected_record.get("run_id")
                if isinstance(run_value, list) and run_value:
                    selected_run_id = str(run_value[0])
                    break
                if run_value:
                    selected_run_id = str(run_value)
                    break
                for nested_key in ("selection", "points", "vlPoint"):
                    nested = selected_record.get(nested_key)
                    nested_records = nested if isinstance(nested, list) else [nested]
                    for nested_record in nested_records:
                        if not isinstance(nested_record, dict):
                            continue
                        nested_run_value = nested_record.get("run_id")
                        if isinstance(nested_run_value, list) and nested_run_value:
                            selected_run_id = str(nested_run_value[0])
                            break
                        if nested_run_value:
                            selected_run_id = str(nested_run_value)
                            break
                    if selected_run_id:
                        break
                if selected_run_id:
                    break
        if selected_run_id:
            selected_rows = plot_df[plot_df["run_id"].astype(str) == selected_run_id]
            if not selected_rows.empty:
                selected_cell = selected_rows.iloc[0]
                details = [
                    str(selected_cell.get("engine") or ""),
                    f"{int(selected_cell['total_length'])} residues"
                    if pd.notna(selected_cell.get("total_length"))
                    else "",
                    str(selected_cell.get("status") or ""),
                    str(selected_cell.get("job_code") or ""),
                ]
                st.caption("Selected prediction: " + " | ".join(value for value in details if value))
                result_url = str(selected_cell.get("result") or "")
                if result_url:
                    st.link_button("Open prediction result", result_url, type="primary")


def _show_manual_benchmark_feature_combinations(
    benchmark_dir: Path,
    *,
    target_metrics: pd.DataFrame | None = None,
    selected_targets: list[str] | None = None,
) -> None:
    run_key = benchmark_dir.parents[1].name if len(benchmark_dir.parents) > 1 else benchmark_dir.name
    ranking_path = benchmark_dir / "merged_benchmark_feature_ranking.csv"
    ranking = _read_csv(ranking_path)
    metrics = target_metrics.copy() if target_metrics is not None else _read_csv(benchmark_dir / "merged_benchmark_metrics.csv")
    if ranking is None or ranking.empty or metrics is None or metrics.empty:
        st.info("Manual feature combinations require merged benchmark metrics and feature ranking tables.")
        return

    ranking = _with_legacy_af2_ranking_rows(benchmark_dir, ranking_path, ranking)
    ranking = _with_common_interface_rankings(benchmark_dir, ranking)
    metrics = _with_common_interface_metrics(benchmark_dir, metrics)
    if target_metrics is not None:
        recomputed = _recompute_feature_ranking_for_metrics(ranking, metrics)
        if recomputed.empty:
            st.info("Manual combinations require at least one binder and one nonbinder with valid scores.")
            return
        ranking = recomputed

    ranking = ranking.copy()
    ranking["engine"] = ranking["feature"].map(_feature_engine)
    ranking["category"] = ranking["feature"].map(_feature_category)
    ranking = _drop_chain_indexed_confidence_features(ranking)
    for column in ["best_average_precision", "best_auroc", "average_precision", "auroc"]:
        if column in ranking.columns:
            ranking[column] = pd.to_numeric(ranking[column], errors="coerce")

    source_ranking = ranking[
        ranking["category"].astype(str).isin(MANUAL_FEATURE_COMBO_SOURCE_CATEGORIES)
        & ranking["feature"].astype(str).isin(set(str(column) for column in metrics.columns))
    ].copy()
    if source_ranking.empty:
        st.info("No engine-derived source features are available for manual combinations.")
        return

    engine_order = [
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
        "Input PyRosetta",
    ]
    available_engines = [
        engine
        for engine in engine_order
        if engine in set(source_ranking["engine"].dropna().astype(str))
    ]
    available_engines.extend(
        sorted(
            engine
            for engine in set(source_ranking["engine"].dropna().astype(str))
            if engine not in set(available_engines)
        )
    )
    selected_engines = st.multiselect(
        "Source engines",
        available_engines,
        default=available_engines,
        key=f"{run_key}_manual_combo_source_engines_v1",
        help="Manual source features may come from any selected engine.",
    )
    if not selected_engines:
        st.info("Select at least one source engine.")
        return

    target_filter_label = "all_targets"
    if selected_targets:
        target_filter_label = "_".join(str(target) for target in selected_targets[:4])
        if len(selected_targets) > 4:
            target_filter_label += f"_plus_{len(selected_targets) - 4}"
    manual_ranking, _ = _show_manual_feature_composer(
        run_key=run_key,
        ranking_key="manual_feature_composer",
        ranking=ranking,
        metrics=metrics,
        selected_engines=selected_engines,
        target_filter_label=target_filter_label,
    )
    if not manual_ranking.empty:
        st.caption(
            "The calculated feature is temporary for this browser session. "
            "Recalculate it after changing source engines, targets, or selected features."
        )


def _show_benchmark_results(run_dir: Path, result: dict) -> None:
    benchmark_dir = run_dir / "artifacts" / "benchmark"
    if not benchmark_dir.exists():
        return

    input_json = read_json(run_dir / "input.json")
    if str(input_json.get("job_type") or "") == "sequence_design_validation":
        _show_sequence_design_validation_results(run_dir, result)
        return
    is_refolding_evaluation = str(input_json.get("job_type") or "") == "refolding_evaluation"
    st.header("Refolding Evaluation Results" if is_refolding_evaluation else "Benchmark Results")
    metrics = result.get("metrics") or {}
    outputs = result.get("outputs") or {}
    colab_summary = outputs.get("colabfold_summary_metrics") or {}
    engine_artifacts = outputs.get("engine_artifacts") or {}
    feature_summary = _first_benchmark_feature_summary(benchmark_dir)
    merged_metrics = _read_csv(benchmark_dir / "merged_benchmark_metrics.csv")
    merged_metrics = _with_common_interface_metrics(benchmark_dir, merged_metrics)
    target_metadata = merged_metrics
    if target_metadata is None or target_metadata.empty or "target_id" not in target_metadata.columns:
        target_metadata = _benchmark_target_metadata_table(run_dir, benchmark_dir)
    all_target_summary = _target_summary(target_metadata)
    available_targets = all_target_summary.get("targets") or []
    selected_targets = available_targets
    if available_targets:
        selected_targets = st.multiselect(
            "Target proteins",
            available_targets,
            default=available_targets,
            key=f"{run_dir.name}_benchmark_result_target_filter_v1",
            help="Filter result counts, plots, rankings, per-record choices, and structures to the selected target proteins.",
        )
        if not selected_targets:
            st.info("Select at least one target protein to display benchmark results.")
            return
    filtered_merged_metrics = merged_metrics
    filtered_target_metadata = target_metadata
    if (
        merged_metrics is not None
        and not merged_metrics.empty
        and "target_id" in merged_metrics.columns
        and selected_targets
    ):
        selected_target_ids = {str(target) for target in selected_targets}
        filtered_merged_metrics = merged_metrics[
            merged_metrics["target_id"].astype(str).isin(selected_target_ids)
        ].copy()
    if (
        filtered_target_metadata is not None
        and not filtered_target_metadata.empty
        and "target_id" in filtered_target_metadata.columns
        and selected_targets
    ):
        selected_target_ids = {str(target) for target in selected_targets}
        filtered_target_metadata = filtered_target_metadata[
            filtered_target_metadata["target_id"].astype(str).isin(selected_target_ids)
        ].copy()
    target_filter_active = bool(available_targets) and len(selected_targets) != len(available_targets)
    target_summary = _target_summary(filtered_target_metadata)
    display_feature_summary = feature_summary
    if target_filter_active and filtered_merged_metrics is not None:
        merged_ranking = _read_csv(benchmark_dir / "merged_benchmark_feature_ranking.csv")
        if merged_ranking is not None and not merged_ranking.empty:
            filtered_ranking = _recompute_feature_ranking_for_metrics(
                merged_ranking,
                filtered_merged_metrics,
            )
            if not filtered_ranking.empty:
                top_row = filtered_ranking.iloc[0]
                display_feature_summary = {
                    **feature_summary,
                    "record_count": target_summary.get("records"),
                    "positive_count": target_summary.get("positive_count"),
                    "negative_count": target_summary.get("negative_count"),
                    "top_feature": top_row.get("feature"),
                    "top_feature_average_precision": top_row.get("best_average_precision"),
                }
    _show_benchmark_collection_notice(run_dir, benchmark_dir, result)

    if is_refolding_evaluation:
        cols = st.columns(5)
        with cols[0]:
            _display_metric("Candidates", metrics.get("record_count") or feature_summary.get("record_count"))
        with cols[1]:
            _display_metric("Targets", target_summary.get("target_count") or metrics.get("target_count"))
        with cols[2]:
            _display_metric("Engines", len(engine_artifacts) if isinstance(engine_artifacts, dict) else None)
        with cols[3]:
            _display_metric("MSA missing", metrics.get("msa_repository_missing_after_lookup"))
        with cols[4]:
            _display_metric("ColabFold jobs", metrics.get("colabfold_selected_count"))
    else:
        cols = st.columns(7)
        with cols[0]:
            _display_metric(
                "Records",
                target_summary.get("records")
                or display_feature_summary.get("record_count")
                or metrics.get("record_count")
                or colab_summary.get("record_count"),
            )
        with cols[1]:
            _display_metric("Targets", target_summary.get("target_count") or metrics.get("target_count"))
        with cols[2]:
            _display_metric(
                "Binders",
                target_summary.get("positive_count")
                if target_summary.get("positive_count") is not None
                else (
                    feature_summary.get("positive_count")
                    or metrics.get("positive_count")
                    or colab_summary.get("positive_count")
                ),
            )
        with cols[3]:
            _display_metric(
                "Nonbinders",
                target_summary.get("negative_count")
                if target_summary.get("negative_count") is not None
                else (
                    feature_summary.get("negative_count")
                    or metrics.get("negative_count")
                    or colab_summary.get("negative_count")
                ),
            )
        with cols[4]:
            _display_metric(
                "Top feature",
                display_feature_summary.get("top_feature")
                or metrics.get("merged_benchmark_top_feature")
                or metrics.get("top_feature")
                or colab_summary.get("top_feature"),
            )
        with cols[5]:
            _display_metric(
                "Top AP",
                display_feature_summary.get("top_feature_average_precision")
                or metrics.get("merged_benchmark_top_feature_average_precision")
                or metrics.get("top_feature_average_precision")
                or colab_summary.get("top_feature_average_precision"),
            )
        with cols[6]:
            _display_metric("ColabFold jobs", metrics.get("colabfold_selected_count"))
    target_text = _target_summary_text(target_summary)
    if target_text:
        st.caption(target_text)

    result_sources = _benchmark_result_source_table(run_dir, benchmark_dir, result)
    collection = read_json(benchmark_dir / "collection.json")
    if result_sources is not None and not result_sources.empty:
        if "is_collection_backbone" in result_sources.columns:
            backbone = result_sources[result_sources["is_collection_backbone"].astype(str).str.lower() == "true"]
        else:
            backbone = pd.DataFrame()
        if collection and not backbone.empty:
            row = backbone.iloc[0]
            st.caption(
                "Collection backbone: "
                f"{row.get('run_id')} ({row.get('records')} records). "
                "Partial engine reruns are merged onto this record set."
            )
        partial_sources = []
        for _, row in result_sources.iterrows():
            records = row.get("records")
            coverage = row.get("coverage_records")
            engines = row.get("engines")
            try:
                if pd.notna(records) and pd.notna(coverage) and int(coverage) < int(metrics.get("record_count") or feature_summary.get("record_count") or 0):
                    partial_sources.append(f"{engines}: {int(coverage)} rows")
            except Exception:
                continue
        if partial_sources:
            st.info("Partial collection coverage: " + "; ".join(partial_sources))

    if isinstance(engine_artifacts, dict) and engine_artifacts:
        engines = ", ".join(str(engine).replace("_", " ") for engine in engine_artifacts)
        st.caption(
            f"Engine outputs stored inside this benchmark run: {engines}. "
            "Use the parent tables, plots, and staged PDB outputs below for comparison."
        )

    _show_runtime_timings(metrics)

    if is_refolding_evaluation:
        view_options = ["Per Record Evaluation", "Structure Viewer", "Inputs and Outputs"]
        default_view = "Per Record Evaluation"
        view_label = "Refolding result view"
    else:
        view_options = [
            "Plots",
            "Feature Ranking",
            "Per Record Evaluation",
            "Structure Viewer",
            "Inputs and Outputs",
        ]
        default_view = "Plots"
        view_label = "Benchmark result view"
    view_key = f"{run_dir.name}_benchmark_result_view"
    if st.session_state.get(view_key) not in {None, *view_options}:
        st.session_state[view_key] = default_view
    view = st.segmented_control(
        view_label,
        view_options,
        default=default_view,
        key=view_key,
    )
    view = view or default_view
    engine_dirs = (
        {
            str(engine): run_dir / str(relative_path)
            for engine, relative_path in engine_artifacts.items()
            if relative_path
        }
        if isinstance(engine_artifacts, dict)
        else {}
    )

    if view == "Plots":
        _show_benchmark_plots(
            benchmark_dir,
            target_metrics=filtered_merged_metrics,
            selected_targets=[str(target) for target in selected_targets],
        )

    elif view == "Feature Ranking":
        _show_grouped_feature_rankings(
            benchmark_dir,
            target_metrics=filtered_merged_metrics if target_filter_active else None,
        )

    elif view == "Per Record Evaluation":
        st.subheader("Engine-Native Prediction Confidence")
        st.caption(
            "These values come directly from each prediction engine or its native output parser. "
            "They are useful within an engine, but similarly named values are not always defined identically across engines."
        )
        _show_metric_expander(
            benchmark_dir / "alphafast_af3_metrics.csv",
            "AlphaFast AF3 native metrics",
            "AF3 confidence outputs, including ranking score, ipTM, pTM, chain-pair ipTM, and minimum chain-pair PAE.",
            [
                "candidate_id",
                "label",
                "alphafast_af3_ranking_score",
                "alphafast_af3_iptm",
                "alphafast_af3_ptm",
                "alphafast_af3_chain_pair_1_2_pae_min",
                "alphafast_af3_chain_pair_2_1_pae_min",
            ],
            expanded=True,
        )
        _show_metric_expander(
            benchmark_dir / "af2_initial_guess_metrics.csv",
            "AF2 initial-guess native metrics",
            "AF2-Multimer prediction confidence and initial-guess validation outputs emitted by the AF2-IG adapter.",
            [
                "binder_id",
                "candidate_id",
                "label",
                "af2_iptm",
                "iptm",
                "af2_ptm",
                "ptm",
                "af2_ipae",
                "ipae",
                "af2_binder_plddt",
                "binder_plddt",
                "af2_target_aligned_binder_rmsd",
                "target_aligned_binder_rmsd",
                "af2_time",
                "time",
            ],
        )
        _show_metric_expander(
            benchmark_dir / "boltz2_initial_guess_metrics.csv",
            "Boltz-2 native metrics",
            "Boltz-2 confidence outputs, including confidence score, ipTM, pTM, complex pLDDT, ipLDDT, and ipDE.",
            [
                "candidate_id",
                "label",
                "boltz2_confidence_score",
                "boltz2_iptm",
                "boltz2_ptm",
                "boltz2_complex_plddt",
                "boltz2_complex_iplddt",
                "boltz2_complex_ipde",
            ],
        )
        _show_metric_expander(
            benchmark_dir / "colabfold_metrics.csv",
            "ColabFold native metrics",
            "ColabFold/AF2-Multimer confidence values aggregated across the selected models.",
            [
                "binder_id",
                "candidate_id",
                "label",
                "colab_actifptm_avg",
                "colab_ptm_avg",
                "colab_iptm_avg",
            ],
        )
        esmfold2_engine_table = (
            engine_dirs.get("esmfold2", benchmark_dir)
            / "artifacts"
            / "benchmark"
            / "benchmark_table.csv"
        )
        esmfold2_native = (
            esmfold2_engine_table
            if esmfold2_engine_table.exists()
            else benchmark_dir / "esmfold2_metrics.csv"
        )
        _show_metric_expander(
            esmfold2_native,
            "ESMFold2 native metrics",
            "ESMFold2 confidence and PAE-derived values emitted by the ESMFold2 benchmark adapter.",
            [
                "candidate_id",
                "label",
                "benchmark_mode",
                "esmfold2_benchmark_mode",
                "esmfold2_benchmark_score",
                "interface_contacts",
                "esmfold2_interface_contacts",
                "binder_plddt",
                "esmfold2_binder_plddt",
                "complex_ptm",
                "esmfold2_complex_ptm",
                "complex_iptm",
                "esmfold2_complex_iptm",
                "iptm",
                "esmfold2_iptm",
                "ptm",
                "esmfold2_ptm",
                "ipae",
                "esmfold2_ipae",
            ],
        )
        _show_metric_expander(
            benchmark_dir / "rf3_metrics.csv",
            "RF3 native metrics",
            "RF3 prediction confidence values parsed from the RF3 output, including ranking score, ipTM, pTM, pLDDT, and overall PAE/PDE.",
            [
                "binder_id",
                "candidate_id",
                "label",
                "rf3_ranking_score",
                "rf3_iptm",
                "rf3_ptm",
                "rf3_overall_plddt",
                "rf3_overall_pae",
                "rf3_overall_pde",
                "rf3_has_full_pae",
            ],
        )
        _show_metric_expander(
            benchmark_dir / "protenix_metrics.csv",
            "Protenix v0.5 native metrics",
            "Legacy PXDesign-backed Protenix v0.5 confidence values parsed from the prediction output, including ranking score, ipTM, pTM, pLDDT, gPDE, and recycle count.",
            [
                "binder_id",
                "candidate_id",
                "label",
                "protenix_ranking_score",
                "protenix_iptm",
                "protenix_ptm",
                "protenix_plddt",
                "protenix_gpde",
                "protenix_num_recycles",
                "protenix_has_full_pae",
            ],
        )
        _show_metric_expander(
            benchmark_dir / "protenix_v1_metrics.csv",
            "Protenix v1 native metrics",
            "Standalone Protenix v1 confidence values parsed from the prediction output, using /mnt/db/reference_files/protenix for model/cache data.",
            [
                "binder_id",
                "candidate_id",
                "label",
                "protenix_v1_ranking_score",
                "protenix_v1_iptm",
                "protenix_v1_ptm",
                "protenix_v1_plddt",
                "protenix_v1_gpde",
                "protenix_v1_num_recycles",
                "protenix_v1_has_full_pae",
            ],
        )
        _show_metric_expander(
            benchmark_dir / "protenix_v2_metrics.csv",
            "Protenix v2 native metrics",
            "Standalone Protenix v2 confidence values parsed from the prediction output, using /mnt/db/reference_files/protenix for model/cache data.",
            [
                "binder_id",
                "candidate_id",
                "label",
                "protenix_v2_ranking_score",
                "protenix_v2_iptm",
                "protenix_v2_ptm",
                "protenix_v2_plddt",
                "protenix_v2_gpde",
                "protenix_v2_num_recycles",
                "protenix_v2_has_full_pae",
            ],
        )
        _show_metric_expander(
            benchmark_dir / "boltzgen_fold_metrics.csv",
            "BoltzGen Fold native metrics",
            "BoltzGen target-template fold confidence values, including ipTM, pTM, pLDDT, design-target ipTM, and interaction PAE.",
            [
                "binder_id",
                "candidate_id",
                "label",
                "boltzgen_fold_iptm",
                "boltzgen_fold_ptm",
                "boltzgen_fold_plddt",
                "boltzgen_fold_design_iptm",
                "boltzgen_fold_design_to_target_iptm",
                "boltzgen_fold_interaction_pae",
                "boltzgen_fold_min_interaction_pae",
                "boltzgen_fold_has_full_pae",
            ],
        )

        st.subheader("Standardized Interface Scores")
        st.caption(
            "These are post-processing metrics calculated after prediction using the same interface-scoring formulas "
            "for every compatible engine. They are not additional prediction engines. "
            "`ipSAE`, `LIS`, `pDockQ`, `pDockQ2`, and interface PAE make cross-engine comparison more consistent."
        )
        st.caption(
            "Higher is better for ipSAE, LIS, pDockQ, and pDockQ2. Lower is better for interface PAE (`ipae`)."
        )
        with st.expander("Cross-engine interface score comparison", expanded=True):
            st.caption(
                "Combined display of the shared interface calculations for every compatible engine. "
                "The workflow stores several source CSVs because AF2-IG and ESMFold2 use separate output adapters, "
                "but the values belong to this single comparison layer."
            )
            _show_combined_interface_metrics(benchmark_dir)
        with st.expander("Interface score source tables", expanded=False):
            st.caption(
                "Technical source tables retained for traceability. Most users should use the combined comparison above."
            )
            for path, label in [
                (benchmark_dir / "common_interface_metrics.csv", "AF3, ColabFold, and Boltz-2 adapter"),
                (benchmark_dir / "af2_common_interface_metrics.csv", "AF2 initial-guess adapter"),
                (benchmark_dir / "esmfold2_common_interface_metrics.csv", "ESMFold2 adapter"),
                (benchmark_dir / "rf3_common_interface_metrics.csv", "RF3 adapter"),
                (benchmark_dir / "protenix_common_interface_metrics.csv", "Protenix v0.5 adapter"),
                (benchmark_dir / "boltzgen_fold_common_interface_metrics.csv", "BoltzGen Fold adapter"),
            ]:
                if path.exists():
                    st.markdown(f"**{label}**")
                    _show_csv_table(path, rows=50)

        st.subheader("Structure-Based Evaluation")
        st.caption(
            "These metrics are calculated from each predicted 3D complex after prediction. "
            "Rosetta evaluates interface energy and packing; PyMOL evaluates contacts, geometry, and secondary structure. "
            "For multi-chain targets, the declared binder chain group is evaluated against the complete declared target chain group."
        )
        rosetta_path = benchmark_dir / "predicted_rosetta_metrics.csv"
        if rosetta_path.exists():
            with st.expander("Rosetta interface metrics by engine", expanded=False):
                st.caption(
                    "Each table contains the same Rosetta energy, packing, solvent-accessible surface area, "
                    "hydrogen-bond, and SAP calculations applied to one engine's predicted complexes."
                )
                _show_rosetta_metrics_by_engine(rosetta_path)
        pymol_dir = benchmark_dir / "pymol_files"
        pymol_files = sorted(pymol_dir.glob("pymol_metrics_*.csv")) if pymol_dir.exists() else []
        if pymol_files:
            with st.expander("PyMOL geometry metrics", expanded=False):
                st.caption(
                    "Separate tables for each engine and the input structure, calculated with the shared PyMOL geometry script."
                )
                for path in pymol_files:
                    st.markdown(f"**{path.stem.replace('pymol_metrics_', '').replace('_', ' ').title()}**")
                    _show_csv_table(path, rows=50)

        st.subheader("Complete Analysis Export")
        st.caption(
            "The merged table joins input metadata, engine-native confidence, standardized interface scores, "
            "Rosetta metrics, and PyMOL metrics by binder ID. It is intended for download and detailed analysis."
        )
        _show_metric_expander(
            benchmark_dir / "merged_benchmark_metrics.csv",
            "Full merged benchmark metrics",
            "Wide table containing every available per-record metric and provenance-prefixed feature.",
            expanded=False,
            rows=200,
        )

    elif view == "Structure Viewer":
        _show_benchmark_structure_viewer(
            run_dir,
            benchmark_dir,
            filtered_merged_metrics,
            is_refolding_evaluation=is_refolding_evaluation,
        )

    elif view == "Inputs and Outputs":
        chain_msa_rel = outputs.get("chain_msa_map") or metrics.get("chain_msa_map")
        chain_msa_path = run_dir / str(chain_msa_rel) if chain_msa_rel else benchmark_dir / "chain_msa_map.json"
        if chain_msa_path.exists():
            payload = read_json(chain_msa_path)
            summary = payload.get("summary") or {}
            st.subheader("Chain and MSA Usage")
            msa_cols = st.columns(6)
            with msa_cols[0]:
                _display_metric("Records", summary.get("record_count"))
            with msa_cols[1]:
                _display_metric("Target chains", summary.get("target_chain_count"))
            with msa_cols[2]:
                _display_metric("Target MSAs", summary.get("target_msa_available_count"))
            with msa_cols[3]:
                _display_metric("Real MSAs", summary.get("target_msa_real_count"))
            with msa_cols[4]:
                _display_metric("Query-only MSAs", summary.get("target_msa_query_only_count"))
            with msa_cols[5]:
                _display_metric("Missing MSAs", summary.get("target_msa_missing_count"))
            rows = []
            for record in payload.get("records") or []:
                binder = record.get("binder") or {}
                targets = record.get("targets") or []
                rows.append(
                    {
                        "binder_id": record.get("binder_id"),
                        "role": "binder",
                        "chain": binder.get("chain"),
                        "msa_available": False,
                        "msa_status": "no_msa_expected",
                        "msa_sequence_count": 0,
                        "msa_path": None,
                    }
                )
                for target in targets:
                    rows.append(
                        {
                            "binder_id": record.get("binder_id"),
                            "role": "target",
                            "chain": target.get("chain"),
                            "msa_available": target.get("msa_available"),
                            "msa_status": target.get("msa_status"),
                            "msa_sequence_count": target.get("msa_sequence_count"),
                            "msa_path": target.get("msa_path"),
                        }
                    )
            if rows:
                with st.expander("Per-record chain and MSA map", expanded=False):
                    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        run_csv_rel = outputs.get("run_csv") or metrics.get("run_csv")
        run_csv = run_dir / str(run_csv_rel) if run_csv_rel else None
        if run_csv and run_csv.exists():
            _show_csv_preview(run_csv, "Generated run.csv", rows=50)
        for label, key in [
            ("Raw repo output", "output_dir"),
            ("ColabFold output", "colabfold_output"),
            ("AlphaFast AF3 output", "alphafast_af3_output"),
            ("Binder FASTA", "binder_fasta"),
        ]:
            rel = outputs.get(key) or metrics.get(key)
            if rel:
                path = run_dir / str(rel)
                st.caption(label)
                st.code(str(path), language="text")
        if engine_dirs:
            chain_map_rows: list[dict[str, Any]] = []
            for engine, engine_dir in engine_dirs.items():
                for path in sorted(engine_dir.glob("**/*.chain_map.json")):
                    chain_map_rows.append(
                        {
                            "engine": engine,
                            "chain_map": str(path.relative_to(run_dir)),
                        }
                    )
            if chain_map_rows:
                with st.expander("Engine staging chain maps", expanded=False):
                    st.dataframe(pd.DataFrame(chain_map_rows), hide_index=True, width="stretch")
            msa_manifest_rows: list[dict[str, Any]] = []
            for engine, engine_dir in engine_dirs.items():
                for path in sorted(engine_dir.glob("**/msa_manifest.json")):
                    summary = {}
                    try:
                        summary = (read_json(path).get("summary") or {}) if path.exists() else {}
                    except Exception:
                        summary = {}
                    msa_manifest_rows.append(
                        {
                            "engine": engine,
                            "manifest": str(path.relative_to(run_dir)),
                            "real_target_msas": summary.get(f"{engine}_target_msa_real_count"),
                            "query_only_target_msas": summary.get(f"{engine}_target_msa_query_only_count"),
                            "missing_target_msas": summary.get(f"{engine}_target_msa_missing_count"),
                        }
                    )
            if msa_manifest_rows:
                with st.expander("Engine target-MSA manifests", expanded=False):
                    st.dataframe(pd.DataFrame(msa_manifest_rows), hide_index=True, width="stretch")


def _design_campaign_number(metrics: dict[str, Any], *names: str) -> float | None:
    for name in names:
        value = metrics.get(name)
        if value in {None, ""}:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            return number
    return None


def _design_campaign_hotspot_metric(metrics: dict[str, Any], key: str) -> Any:
    atom_key = f"hotspot_atom_{key}"
    ca_key = f"hotspot_ca_{key}"
    value = metrics.get(atom_key)
    if value not in {None, ""}:
        return value
    return metrics.get(ca_key)


def _design_campaign_hotspot_contact_label(metrics: dict[str, Any]) -> str:
    contacted = _design_campaign_hotspot_metric(metrics, "contacted_count")
    total = _design_campaign_hotspot_metric(metrics, "count")
    if total in {None, ""}:
        return ""
    return f"{contacted or 0}/{total}"


def _design_campaign_hotspot_residues(value: object) -> set[tuple[str, int]]:
    residues: set[tuple[str, int]] = set()
    if isinstance(value, str):
        items: list[object] = re.split(r"[,;\s]+", value)
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = []
    for item in items:
        text = str(item or "").strip()
        if not text:
            continue
        match = re.match(r"^([A-Za-z0-9]+)\s*[:_-]?\s*(-?\d+)$", text)
        if not match:
            continue
        try:
            residues.add((match.group(1), int(match.group(2))))
        except ValueError:
            continue
    return residues


def _design_campaign_hotspot_tokens(residues: set[tuple[str, int]]) -> list[str]:
    return [f"{chain}{residue}" for chain, residue in sorted(residues, key=lambda item: (item[0], item[1]))]


def _design_campaign_hotspot_chains(
    residues: set[tuple[str, int]],
    *,
    color: str,
    label: str,
) -> list[ChainVisualization]:
    by_chain: dict[str, list[int]] = {}
    for chain, residue in sorted(residues):
        by_chain.setdefault(chain, []).append(residue)
    return [
        ChainVisualization(
            chain_id=chain,
            residues=values,
            color="uniform",
            color_params={"value": color},
            representation_type="cartoon+ball-and-stick",
            label=label,
        )
        for chain, values in by_chain.items()
    ]


def _atom_is_heavy(atom: object) -> bool:
    element = str(getattr(atom, "element", "") or "").strip().upper()
    name = str(getattr(atom, "name", "") or getattr(atom, "id", "") or "").strip().upper()
    return element != "H" and not name.startswith("H")


def _design_campaign_structure_geometry_stats(
    path_text: str,
    mtime_ns: int,
    size: int,
    binder_chains: tuple[str, ...],
    fixed_target_chains: tuple[str, ...],
    moving_target_chains: tuple[str, ...],
    hotspots: tuple[tuple[str, int], ...],
    contact_cutoff: float = 4.5,
) -> dict[str, Any]:
    _ = (mtime_ns, size)
    path = Path(path_text)
    try:
        structure = _parse_structure_file(path)
    except Exception as exc:
        return {"geometry_status": f"unreadable: {exc}"}
    model = next(structure.get_models(), None)
    if model is None:
        return {"geometry_status": "no model"}

    binder_coords: list[np.ndarray] = []
    binder_residues: set[tuple[str, int, str]] = set()
    for chain_id in binder_chains:
        if chain_id not in model:
            continue
        for residue in model[chain_id]:
            residue_id = getattr(residue, "id", ("", None, ""))
            if residue_id[0] != " ":
                continue
            residue_has_atom = False
            for atom in residue:
                if not _atom_is_heavy(atom):
                    continue
                binder_coords.append(np.asarray(atom.coord, dtype=float))
                residue_has_atom = True
            if residue_has_atom:
                binder_residues.add((str(chain_id), int(residue_id[1]), str(residue_id[2]).strip()))

    binder_array = np.asarray(binder_coords, dtype=float).reshape((-1, 3))
    radius_of_gyration = None
    if len(binder_array) > 0:
        center = binder_array.mean(axis=0)
        radius_of_gyration = float(np.sqrt(np.mean(np.sum((binder_array - center) ** 2, axis=1))))

    chain_map = {fixed: moving for fixed, moving in zip(fixed_target_chains, moving_target_chains)}
    contacted: set[tuple[str, int]] = set()
    min_distance = None
    if len(binder_array) > 0 and hotspots:
        backbone_names = {"N", "CA", "C", "O", "OXT"}
        for fixed_chain, residue_number in hotspots:
            moving_chain = chain_map.get(fixed_chain, fixed_chain)
            if moving_chain not in model:
                continue
            hotspot_coords: list[np.ndarray] = []
            for residue in model[moving_chain]:
                residue_id = getattr(residue, "id", ("", None, ""))
                if residue_id[0] != " " or int(residue_id[1]) != int(residue_number):
                    continue
                sidechain_coords = [
                    np.asarray(atom.coord, dtype=float)
                    for atom in residue
                    if _atom_is_heavy(atom) and str(getattr(atom, "name", "") or "").strip().upper() not in backbone_names
                ]
                hotspot_coords = sidechain_coords or [
                    np.asarray(atom.coord, dtype=float) for atom in residue if _atom_is_heavy(atom)
                ]
                break
            if not hotspot_coords:
                continue
            hotspot_array = np.asarray(hotspot_coords, dtype=float).reshape((-1, 3))
            distances = np.linalg.norm(hotspot_array[:, None, :] - binder_array[None, :, :], axis=2)
            residue_min = float(np.min(distances))
            min_distance = residue_min if min_distance is None else min(min_distance, residue_min)
            if residue_min <= float(contact_cutoff):
                contacted.add((fixed_chain, int(residue_number)))

    return {
        "binder_radius_of_gyration": radius_of_gyration,
        "binder_heavy_atoms": int(len(binder_array)),
        "binder_residues": int(len(binder_residues)),
        "computed_hotspot_contacted_count": int(len(contacted)),
        "computed_hotspot_count": int(len(hotspots)),
        "computed_hotspot_fraction": (float(len(contacted)) / float(len(hotspots))) if hotspots else None,
        "computed_hotspot_min_distance": min_distance,
        "computed_contacted_hotspots": ",".join(_design_campaign_hotspot_tokens(contacted)),
        "geometry_status": "ok",
        **_design_campaign_binder_secondary_structure_stats(model, binder_chains),
    }


def _design_campaign_binder_secondary_structure_stats(model: object, binder_chains: tuple[str, ...]) -> dict[str, Any]:
    try:
        import pydssp
    except Exception as exc:
        return {"secondary_structure_status": f"pydssp unavailable: {exc}"}

    assignments: list[str] = []
    for chain_id in binder_chains:
        if chain_id not in model:
            continue
        chain_coords: list[list[np.ndarray]] = []
        for residue in model[chain_id]:
            residue_id = getattr(residue, "id", ("", None, ""))
            if residue_id[0] != " ":
                continue
            backbone_atoms: list[np.ndarray] = []
            for atom_name in ("N", "CA", "C", "O"):
                if atom_name not in residue:
                    backbone_atoms = []
                    break
                backbone_atoms.append(np.asarray(residue[atom_name].coord, dtype=np.float32))
            if backbone_atoms:
                chain_coords.append(backbone_atoms)
        if len(chain_coords) < 4:
            continue
        try:
            chain_assignment = np.asarray(pydssp.assign(np.asarray(chain_coords, dtype=np.float32), out_type="c3")).reshape(-1)
        except Exception:
            continue
        assignments.extend(str(value) for value in chain_assignment)

    if not assignments:
        return {"secondary_structure_status": "no complete binder backbone"}

    helix_count = int(sum(value == "H" for value in assignments))
    sheet_count = int(sum(value == "E" for value in assignments))
    coil_count = int(sum(value not in {"H", "E"} for value in assignments))
    raw_segments: list[dict[str, int | str]] = []
    current_state = ""
    start = 0
    for index, state in enumerate(assignments):
        normalized_state = state if state in {"H", "E"} else "-"
        if index == 0:
            current_state = normalized_state
            start = 1
            continue
        if normalized_state == current_state:
            continue
        raw_segments.append({"state": current_state, "start": start, "end": index, "length": index - start + 1})
        current_state = normalized_state
        start = index + 1
    if assignments:
        raw_segments.append(
            {
                "state": current_state,
                "start": start,
                "end": len(assignments),
                "length": len(assignments) - start + 1,
            }
        )
    structured_segments: list[dict[str, int | str]] = []
    for segment in raw_segments:
        state = str(segment["state"])
        if state not in {"H", "E"}:
            continue
        min_length = 4 if state == "H" else 3
        if int(segment["length"]) < min_length:
            continue
        if (
            structured_segments
            and str(structured_segments[-1]["state"]) == state
            and int(segment["start"]) - int(structured_segments[-1]["end"]) - 1 < 3
        ):
            structured_segments[-1]["end"] = segment["end"]
            structured_segments[-1]["length"] = int(structured_segments[-1]["end"]) - int(structured_segments[-1]["start"]) + 1
        else:
            structured_segments.append(dict(segment))
    return {
        "binder_secondary_residues": int(len(assignments)),
        "binder_helix_residues": helix_count,
        "binder_sheet_residues": sheet_count,
        "binder_coil_residues": coil_count,
        "binder_structured_residues": helix_count + sheet_count,
        "binder_secondary_structure_elements": int(len(structured_segments)),
        "secondary_structure_status": "ok",
    }


@st.cache_data(show_spinner=False)
def _cached_design_campaign_structure_geometry_stats(
    path_text: str,
    mtime_ns: int,
    size: int,
    binder_chains: tuple[str, ...],
    fixed_target_chains: tuple[str, ...],
    moving_target_chains: tuple[str, ...],
    hotspots: tuple[tuple[str, int], ...],
    contact_cutoff: float,
) -> dict[str, Any]:
    return _design_campaign_structure_geometry_stats(
        path_text,
        mtime_ns,
        size,
        binder_chains,
        fixed_target_chains,
        moving_target_chains,
        hotspots,
        contact_cutoff,
    )


def _design_campaign_rank_metric_score(metrics: dict[str, Any], metric: object) -> float | None:
    key = str(metric or "").strip().lower()
    if not key:
        return None
    aliases = {
        "iptm": ("af2_iptm", "af2_average_iptm", "af2_model_1_iptm", "iptm", "ipTM", "i_ptm"),
        "ipae": ("af2_ipae", "af2_average_ipae", "af2_model_1_ipae", "ipae", "iPAE", "i_pae"),
        "binder_plddt": (
            "af2_binder_plddt",
            "af2_average_binder_plddt",
            "af2_model_1_binder_plddt",
            "binder_plddt",
            "average_binder_plddt",
            "plddt",
        ),
        "ptm": ("af2_ptm", "af2_average_ptm", "af2_model_1_ptm", "ptm", "pTM"),
        "target_aligned_binder_rmsd": (
            "target_aligned_binder_rmsd",
            "af2_target_aligned_binder_rmsd",
            "af2_average_target_aligned_binder_rmsd",
            "af2_model_1_target_aligned_binder_rmsd",
            "first_refolding_screen_screen_rmsd",
            "second_refolding_screen_screen_rmsd",
        ),
        "ranking_score": ("ranking_score", "af3_ranking_score", "boltz2_ranking_score"),
    }
    return _design_campaign_number(metrics, *(aliases.get(key) or (str(metric),)))


def _design_campaign_candidate_rows(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
        metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
        engine = str(metadata.get("design_campaign_engine") or candidate.get("source_tool") or "")
        failures = (
            metrics.get("common_filter_failures")
            or metrics.get("common_confidence_failures")
            or metrics.get("common_rosetta_failures")
            or ""
        )
        sequence = str(candidate.get("binder_sequence") or "")
        rows.append(
            {
                "rank": metrics.get("design_campaign_rank"),
                "campaign": metadata.get("design_campaign_source_campaign")
                or metadata.get("design_campaign_source_run_id")
                or "",
                "engine": engine,
                "engine label": DESIGN_CAMPAIGN_ENGINE_LABELS.get(engine, engine),
                "candidate": str(candidate.get("candidate_id") or ""),
                "stage": candidate.get("stage") or "",
                "selection": metrics.get("harmonized_selection_mode") or "",
                "complex AF2": metrics.get("complex_af2_parameter_family")
                or metrics.get("af2_validation_parameter_family")
                or "",
                "binder-fold AF2": metrics.get("binder_fold_af2_parameter_family") or "",
                "common pass": metrics.get("common_bindcraft_pass"),
                "ipSAE": _design_campaign_number(metrics, "ipsae", "ipSAE"),
                "iPAE": _design_campaign_number(metrics, "ipae", "iPAE", "i_pae"),
                "ipTM": _design_campaign_number(metrics, "iptm", "ipTM", "i_ptm"),
                "binder pLDDT": _design_campaign_number(metrics, "binder_plddt", "average_binder_plddt", "plddt"),
                "Rosetta dG": _design_campaign_number(metrics, "rosetta_interface_dG"),
                "interface sc": _design_campaign_number(metrics, "rosetta_interface_sc"),
                "binder fold RMSD": _design_campaign_number(
                    metrics,
                    "average_binder_rmsd",
                    "Binder_RMSD",
                ),
                "complex pose RMSD": _design_campaign_number(
                    metrics,
                    "average_hotspot_rmsd",
                    "Hotspot_RMSD",
                ),
                "hotspot contacts": _design_campaign_hotspot_contact_label(metrics),
                "hotspot fraction": _design_campaign_number(metrics, "hotspot_atom_contact_fraction", "hotspot_ca_contact_fraction"),
                "hotspot min distance": _design_campaign_number(metrics, "hotspot_atom_min_distance", "hotspot_ca_min_distance"),
                "target-aligned binder RMSD": _design_campaign_number(
                    metrics,
                    "target_aligned_binder_rmsd",
                    "af2_target_aligned_binder_rmsd",
                    "af2_average_target_aligned_binder_rmsd",
                    "af2_model_1_target_aligned_binder_rmsd",
                    "first_refolding_screen_screen_rmsd",
                    "second_refolding_screen_screen_rmsd",
                ),
                "binder length": candidate.get("binder_length") or len(sequence.replace(":", "")),
                "failures": failures,
                "_candidate_id": str(candidate.get("candidate_id") or ""),
            }
        )
    return rows


def _display_design_campaign_candidates(candidates: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    def with_unique_display_ids(display_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        id_counts: dict[str, int] = {}
        for candidate in display_candidates:
            candidate_id = str(candidate.get("candidate_id") or "").strip()
            if candidate_id:
                id_counts[candidate_id] = id_counts.get(candidate_id, 0) + 1
        if not any(count > 1 for count in id_counts.values()):
            return display_candidates

        seen: dict[str, int] = {}
        unique_candidates: list[dict[str, Any]] = []
        for candidate in display_candidates:
            candidate_id = str(candidate.get("candidate_id") or "").strip()
            if not candidate_id or id_counts.get(candidate_id, 0) <= 1:
                unique_candidates.append(candidate)
                continue
            seen[candidate_id] = seen.get(candidate_id, 0) + 1
            raw_metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
            source_run_dir = str(raw_metadata.get("source_run_dir") or "").strip()
            suffix = Path(source_run_dir).name if source_run_dir else ""
            if not suffix:
                identity = "|".join(
                    str(candidate.get(key) or "")
                    for key in ("complex_pdb", "binder_pdb", "backbone_pdb", "structure_path")
                )
                suffix = hashlib.sha1(identity.encode()).hexdigest()[:8] if identity else f"copy{seen[candidate_id]:03d}"
            display_id = f"{candidate_id}__{seen[candidate_id]:03d}_{suffix}"
            clone = dict(candidate)
            clone["candidate_id"] = display_id
            clone["display_candidate_id"] = display_id
            clone["original_candidate_id"] = candidate_id
            clone["raw_metadata"] = {
                **raw_metadata,
                "display_original_candidate_id": candidate_id,
                "display_candidate_id": display_id,
            }
            unique_candidates.append(clone)
        return unique_candidates

    if str(params.get("workflow_recipe") or "") != "staged":
        return with_unique_display_ids(candidates)
    has_proteina_generated = any(
        str(candidate.get("source_tool") or "") == "proteina_complexa"
        and str(
            (candidate.get("metrics") or {}).get("result_kind")
            or (candidate.get("raw_metadata") or {}).get("result_kind")
            or ""
        )
        == "generated_prefilter"
        for candidate in candidates
    )
    if not has_proteina_generated:
        return with_unique_display_ids(candidates)
    return with_unique_display_ids([
        candidate
        for candidate in candidates
        if not (
            str(candidate.get("source_tool") or "") == "proteina_complexa"
            and str(
                (candidate.get("metrics") or {}).get("result_kind")
                or (candidate.get("raw_metadata") or {}).get("result_kind")
                or ""
            )
            == "native_pipeline"
        )
    ])


def _design_campaign_candidate_path(run_dir: Path, candidate: dict[str, Any], *keys: str) -> Path | None:
    raw_metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
    source_candidate = raw_metadata.get("source_candidate") if isinstance(raw_metadata.get("source_candidate"), dict) else {}
    source_run_dir = _resolve_run_relative_path(run_dir, raw_metadata.get("source_run_dir"))
    if source_run_dir is None and raw_metadata.get("source_run_dir"):
        candidate_dir = Path(str(raw_metadata.get("source_run_dir"))).expanduser()
        if candidate_dir.exists():
            source_run_dir = candidate_dir
    candidate_roots = [run_dir]
    if source_run_dir is not None:
        candidate_roots.insert(0, source_run_dir)
    for key in keys:
        for value in (candidate.get(key), source_candidate.get(key)):
            for base_dir in candidate_roots:
                path = _resolve_run_relative_path(base_dir, value)
                if path is not None:
                    break
            else:
                path = None
            if path is None:
                continue
            if (
                str(candidate.get("source_tool") or "").lower() == "pxdesign"
                and path.name.lower() in {"target.cif", "target_binder.cif"}
                and "raw/pxdesign/output" in path.as_posix()
            ):
                fallback = _pxdesign_target_binder_prediction_fallback(source_run_dir or run_dir, path)
                if fallback is not None:
                    return fallback
            return path
    return None


def _pxdesign_target_binder_prediction_fallback(run_dir: Path, target_only_path: Path) -> Path | None:
    raw_output = run_dir / "artifacts" / "raw" / "pxdesign" / "output"
    try:
        target_only_path.relative_to(raw_output)
    except ValueError:
        return None
    matches = sorted(raw_output.glob("target_binder/*/predictions/*.cif"))
    if matches:
        return matches[0]
    matches = sorted(raw_output.glob("target_binder/**/target_binder*.cif"))
    return matches[0] if matches else None


def _design_campaign_resolve_metric_path(run_dir: Path, candidate: dict[str, Any], key: str) -> Path | None:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    value = metrics.get(key)
    if not value:
        return None
    return _resolve_run_relative_path(run_dir, value)


def _design_campaign_secondary_structure_source_path(run_dir: Path, candidate: dict[str, Any]) -> Path | None:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    reconstructed = metrics.get("binder_secondary_structure_reconstructed_pdb")
    if reconstructed:
        path = _resolve_run_relative_path(run_dir, reconstructed)
        if path is not None and path.exists():
            return path
    return None


def _design_campaign_unique_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("_candidate_id") or row.get("candidate") or "")
        if not key:
            key = "|".join(str(row.get(column) or "") for column in ("engine label", "structure", "rank"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _design_campaign_candidates_by_id(candidates: list[dict[str, Any]], candidate_id: str) -> list[dict[str, Any]]:
    matches = [candidate for candidate in candidates if str(candidate.get("candidate_id") or "") == str(candidate_id)]
    return matches or candidates[:1]


def _design_campaign_stage_structures(
    run_dir: Path,
    selected_candidates: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    if not selected_candidates:
        return []
    candidate = selected_candidates[-1]
    candidate_id = str(candidate.get("candidate_id") or "")
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    generator_reference: Path | None = None

    def add(
        label: str,
        path: Path | None,
        kind: str,
        reference: Path | None = None,
        *,
        candidate_id: str = "",
        sequence_level: object = "",
        role: object = "",
        rank_metric: object = "",
        rank_score: object = None,
        rank: object = "",
    ) -> None:
        if path is None:
            return
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen_paths:
            return
        seen_paths.add(key)
        records.append(
            {
                "label": label,
                "path": path,
                "kind": kind,
                "reference": reference,
                "candidate_id": candidate_id,
                "sequence_level": sequence_level,
                "role": role,
                "rank_metric": rank_metric,
                "rank_score": rank_score,
                "rank": rank,
            }
        )

    def screen_prediction_path(child_run_value: object, candidate_id_value: str, engine_dir: str = "af2") -> Path | None:
        if not child_run_value or not candidate_id_value:
            return None
        child_run = Path(str(child_run_value))
        for root in ("predicted_viewer_structures", "predicted_metric_pdbs"):
            for suffix in ("", "_af2ig_mt_tt", "_af2ig_mt_ct"):
                path = (
                    child_run
                    / "artifacts"
                    / "benchmark"
                    / root
                    / engine_dir
                    / f"{candidate_id_value}{suffix}.pdb"
                )
                if path.exists():
                    return path
        return None

    attempt_key = str(metrics.get("staged_attempt_key") or "")
    workflow_rows = summary.get("workflows") if isinstance(summary.get("workflows"), list) else []
    source_tool = str(candidate.get("source_tool") or "")
    should_lookup_rfdiffusion_parent = source_tool == "rfdiffusion_classic" or bool(attempt_key)
    if should_lookup_rfdiffusion_parent:
        for workflow in workflow_rows:
            if str(workflow.get("engine") or "") != "rfdiffusion_classic":
                continue
            workflow_run = Path(str(workflow.get("run_dir") or ""))
            candidates_path = workflow_run / "artifacts" / "normalized_candidates" / "candidates.jsonl"
            if not candidates_path.exists():
                continue
            for parent in read_candidates(candidates_path):
                parent_id = str(parent.get("candidate_id") or "")
                if attempt_key and parent_id not in attempt_key:
                    continue
                if not attempt_key and source_tool == "rfdiffusion_classic" and parent_id != candidate_id:
                    continue
                parent_path = _design_campaign_candidate_path(workflow_run, parent, "complex_pdb", "binder_pdb")
                if generator_reference is None:
                    generator_reference = parent_path
                add(
                    "RFdiffusion backbone/reference",
                    parent_path,
                    "generator",
                    candidate_id=parent_id,
                    role="generator backbone",
                )
                break

    add(
        "Sequence-designed input backbone",
        _design_campaign_resolve_metric_path(run_dir, candidate, "source_input_complex_pdb"),
        "sequence",
            reference=generator_reference,
            candidate_id=candidate_id,
            sequence_level=metrics.get("staged_sequence_level") or "",
            role="sequence-designed input",
            rank_metric=metrics.get("staged_attempt_selection_metric") or "",
            rank_score=_design_campaign_rank_metric_score(metrics, metrics.get("staged_attempt_selection_metric")),
            rank=metrics.get("staged_attempt_selection_rank") or "",
        )
    sequence_summary = summary.get("sequence_refinement") if isinstance(summary.get("sequence_refinement"), dict) else {}
    second_pass_summary = (
        sequence_summary.get("second_pass")
        if isinstance(sequence_summary.get("second_pass"), dict)
        else {}
    )
    second_level_available = (
        str(second_pass_summary.get("status") or "").lower() == "completed"
        and int(second_pass_summary.get("refined_candidate_count") or 0) > 0
    )
    for index, staged_candidate in enumerate(selected_candidates, start=1):
        staged_metrics = staged_candidate.get("metrics") if isinstance(staged_candidate.get("metrics"), dict) else {}
        staged_metadata = staged_candidate.get("raw_metadata") if isinstance(staged_candidate.get("raw_metadata"), dict) else {}
        result_kind = str(
            staged_metrics.get("result_kind") or staged_metadata.get("result_kind") or ""
        )
        candidate_stage = str(staged_candidate.get("stage") or "")
        is_generator_output = (
            result_kind
            in {
                "generation_only",
                "generator_trajectory",
                "generated_prefilter",
                "native_pipeline",
            }
            or candidate_stage.startswith("generation.")
            or str(staged_metadata.get("campaign_workflow_recipe") or "") == "engine_scout"
        )
        engine_label = DESIGN_CAMPAIGN_ENGINE_LABELS.get(
            str(staged_candidate.get("source_tool") or ""),
            str(staged_candidate.get("source_tool") or "Generator"),
        )
        label = f"{engine_label} generator output" if is_generator_output else "AF2-IG screen output"
        screen_child_run = None
        if not is_generator_output and staged_metrics.get("second_refolding_screen_screen_rmsd") not in {None, ""}:
            if not second_level_available:
                continue
            label = "Level 2 AF2-IG screen output"
            screen_child_run = sequence_summary.get("second_screen", {}).get("child_run")
            rank_metric = staged_metrics.get("staged_two_step_second_screen_selection_metric") or ""
            rank_value = staged_metrics.get("staged_two_step_second_screen_rank") or ""
        elif not is_generator_output and staged_metrics.get("first_refolding_screen_screen_rmsd") not in {None, ""}:
            label = "Level 1 AF2-IG screen output"
            screen_child_run = sequence_summary.get("first_screen", {}).get("child_run")
            rank_metric = staged_metrics.get("staged_two_step_first_screen_selection_metric") or staged_metrics.get("staged_attempt_selection_metric") or ""
            rank_value = staged_metrics.get("staged_two_step_first_screen_rank") or staged_metrics.get("staged_attempt_selection_rank") or ""
        elif len(selected_candidates) > 1:
            label = f"Screen output {index}"
            rank_metric = staged_metrics.get("staged_attempt_selection_metric") or ""
            rank_value = staged_metrics.get("staged_attempt_selection_rank") or ""
        else:
            rank_metric = staged_metrics.get("staged_attempt_selection_metric") or ""
            rank_value = staged_metrics.get("staged_attempt_selection_rank") or ""
        add(
            label,
            screen_prediction_path(screen_child_run, str(staged_candidate.get("candidate_id") or ""))
            or _design_campaign_candidate_path(run_dir, staged_candidate, "complex_pdb", "binder_pdb"),
            "generator" if is_generator_output else "screen",
            reference=generator_reference,
            candidate_id=str(staged_candidate.get("candidate_id") or ""),
            sequence_level=staged_metrics.get("staged_sequence_level") or "",
            role=staged_metrics.get("staged_output_role") or ("generator output" if is_generator_output else "screen output"),
            rank_metric=rank_metric,
            rank_score=_design_campaign_rank_metric_score(staged_metrics, rank_metric),
            rank=rank_value,
        )
        reconstructed_path = _design_campaign_secondary_structure_source_path(run_dir, staged_candidate)
        if reconstructed_path is not None:
            add(
                "Reconstructed poly-Gly backbone",
                reconstructed_path,
                "reconstructed",
                reference=generator_reference,
                candidate_id=str(staged_candidate.get("candidate_id") or ""),
                sequence_level=staged_metrics.get("staged_sequence_level") or "",
                role="CA-derived poly-glycine backbone for secondary-structure metrics",
                rank_metric=rank_metric,
                rank_score=_design_campaign_rank_metric_score(staged_metrics, rank_metric),
                rank=rank_value,
            )

    metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
    evaluation_child = metadata.get("campaign_evaluation_child_run")
    evaluation_summary = summary.get("evaluation") if isinstance(summary.get("evaluation"), dict) else {}
    evaluation_child = evaluation_child or evaluation_summary.get("child_run")
    if evaluation_child and candidate_id:
        child_run = Path(str(evaluation_child))
        for engine_dir in ("af3", "af2", "boltz2", "colabfold"):
            predicted = child_run / "artifacts" / "benchmark" / "predicted_metric_pdbs" / engine_dir / f"{candidate_id}.pdb"
            if predicted.exists():
                add(
                    f"Final evaluation refold ({engine_dir})",
                    predicted,
                    "evaluation",
                    reference=generator_reference,
                    candidate_id=candidate_id,
                    sequence_level=metrics.get("staged_sequence_level") or "",
                    role=f"final evaluation of {metrics.get('staged_output_role') or 'candidate'}",
                    rank_metric=metrics.get("staged_two_step_second_screen_selection_metric")
                    or metrics.get("staged_two_step_first_screen_selection_metric")
                    or metrics.get("staged_attempt_selection_metric")
                    or "",
                    rank_score=_design_campaign_rank_metric_score(
                        metrics,
                        metrics.get("staged_two_step_second_screen_selection_metric")
                        or metrics.get("staged_two_step_first_screen_selection_metric")
                        or metrics.get("staged_attempt_selection_metric"),
                    ),
                    rank=metrics.get("staged_two_step_second_screen_rank")
                    or metrics.get("staged_two_step_first_screen_rank")
                    or metrics.get("staged_attempt_selection_rank")
                    or "",
                )
    return records


def _design_campaign_reference_path(run_dir: Path, candidate: dict[str, Any]) -> Path | None:
    metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
    source_candidate = (
        metadata.get("source_candidate")
        if isinstance(metadata.get("source_candidate"), dict)
        else {}
    )
    source_run_dir = _resolve_run_relative_path(run_dir, metadata.get("source_run_dir"))
    if source_run_dir is None and metadata.get("source_run_dir"):
        candidate_dir = Path(str(metadata.get("source_run_dir"))).expanduser()
        if candidate_dir.exists():
            source_run_dir = candidate_dir
    candidate_roots = [run_dir]
    if source_run_dir is not None:
        candidate_roots.insert(0, source_run_dir)
    for value in (
        metadata.get("design_reference_pdb"),
        source_candidate.get("complex_pdb"),
        (candidate.get("metrics") or {}).get("monomer_refolding_reference"),
    ):
        for base_dir in candidate_roots:
            path = _resolve_run_relative_path(base_dir, value)
            if path is not None:
                return path
    return None


def _design_campaign_chain_ids(
    candidate: dict[str, Any],
    metadata_key: str,
    candidate_key: str,
) -> list[str]:
    metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
    source_candidate = (
        metadata.get("source_candidate")
        if isinstance(metadata.get("source_candidate"), dict)
        else {}
    )
    values = metadata.get(metadata_key) or source_candidate.get(candidate_key) or candidate.get(candidate_key) or []
    if isinstance(values, str):
        values = re.split(r"[\s,;]+", values)
    return [str(value).strip() for value in values if str(value).strip()]


def _structure_chain_ca_coordinates(structure: object, chains: list[str]) -> np.ndarray:
    model = next(structure.get_models(), None)
    if model is None:
        return np.empty((0, 3), dtype=float)
    coordinates: list[np.ndarray] = []
    for chain_id in chains:
        if chain_id not in model:
            continue
        for residue in model[chain_id]:
            if getattr(residue, "id", ("",))[0] == " " and "CA" in residue:
                coordinates.append(np.asarray(residue["CA"].coord, dtype=float))
    return np.asarray(coordinates, dtype=float).reshape((-1, 3))


def _design_campaign_pose_comparison(
    reference_path: Path,
    prediction_path: Path,
    reference_target_chains: list[str],
    prediction_target_chains: list[str],
    reference_binder_chains: list[str],
    prediction_binder_chains: list[str],
) -> tuple[str | None, dict[str, float | int | str | None], dict[str, str]]:
    try:
        reference_structure = _parse_structure_file(reference_path)
        prediction_chain_map = _pdb_safe_chain_id_map(_parse_structure_file(prediction_path))
        aligned_text, target_rmsd, alignment_note = _aligned_structure_text(
            prediction_path,
            reference_structure,
            reference_target_chains,
            prediction_target_chains,
        )
        if not aligned_text:
            return None, {"alignment": alignment_note, "target_rmsd": target_rmsd}, prediction_chain_map
        from Bio.PDB import PDBParser

        aligned_structure = PDBParser(QUIET=True).get_structure(
            prediction_path.stem,
            StringIO(aligned_text),
        )
        reference_binder = _structure_chain_ca_coordinates(
            reference_structure,
            reference_binder_chains,
        )
        reference_target = _structure_chain_ca_coordinates(
            reference_structure,
            reference_target_chains,
        )
        prediction_binder = _structure_chain_ca_coordinates(
            aligned_structure,
            [prediction_chain_map.get(chain, chain) for chain in prediction_binder_chains],
        )
        prediction_target = _structure_chain_ca_coordinates(
            aligned_structure,
            [prediction_chain_map.get(chain, chain) for chain in prediction_target_chains],
        )
        binder_count = min(len(reference_binder), len(prediction_binder))
        binder_rmsd = None
        if binder_count >= 3:
            delta = reference_binder[:binder_count] - prediction_binder[:binder_count]
            binder_rmsd = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))

        def contact_summary(binder: np.ndarray, target: np.ndarray) -> tuple[int | None, float | None]:
            if not len(binder) or not len(target):
                return None, None
            distances = np.linalg.norm(binder[:, None, :] - target[None, :, :], axis=2)
            return int(np.count_nonzero(distances < 8.0)), float(np.min(distances))

        reference_contacts, reference_min_distance = contact_summary(
            reference_binder,
            reference_target,
        )
        prediction_contacts, prediction_min_distance = contact_summary(
            prediction_binder,
            prediction_target,
        )
        return aligned_text, {
            "alignment": alignment_note,
            "target_rmsd": target_rmsd,
            "binder_rmsd": binder_rmsd,
            "reference_contacts": reference_contacts,
            "prediction_contacts": prediction_contacts,
            "reference_min_distance": reference_min_distance,
            "prediction_min_distance": prediction_min_distance,
            "binder_ca_count": binder_count,
        }, prediction_chain_map
    except Exception as exc:
        return None, {"alignment": f"comparison failed: {type(exc).__name__}: {exc}"}, {}


SEQUENCE_VALIDATION_ENGINES = [
    ("AF3", "af3", "alphafast_af3"),
    ("ColabFold", "colab", "colabfold"),
    ("AF2 initial guess", "af2", "af2_initial_guess"),
    ("ESMFold2", "esmfold2", "esmfold2"),
    ("Boltz-2", "boltz2", "boltz2"),
    ("RF3", "rf3", "rf3"),
    ("OpenFold-3", "openfold3", "openfold3"),
    ("Protenix v0.5", "protenix", "protenix"),
    ("Protenix v1", "protenix_v1", "protenix_v1"),
    ("Protenix v2", "protenix_v2", "protenix_v2"),
    ("BoltzGen Fold", "boltzgen_fold", "boltzgen_fold"),
]


def _sequence_validation_available_engines(benchmark_dir: Path, merged: pd.DataFrame | None) -> list[tuple[str, str, str]]:
    columns = set(merged.columns) if merged is not None else set()
    run_dir = benchmark_dir.parents[1]
    available: list[tuple[str, str, str]] = []
    for label, prefix, structure_key in SEQUENCE_VALIDATION_ENGINES:
        structure_dir = benchmark_dir / "predicted_metric_pdbs" / prefix
        has_structures = structure_dir.exists() and any(structure_dir.glob("*.pdb"))
        has_metrics = any(str(column).startswith(f"{prefix}_") for column in columns)
        has_linked_predictions = bool(_linked_prediction_run_dirs(run_dir, structure_key))
        if has_structures or has_metrics or has_linked_predictions:
            available.append((label, prefix, structure_key))
    return available


def _sequence_validation_id_column(table: pd.DataFrame) -> str:
    for column in ["candidate_id", "binder_id", "original_binder_id"]:
        if column in table.columns:
            return column
    return str(table.columns[0])


def _sequence_validation_row_for_id(table: pd.DataFrame | None, design_id: str) -> pd.Series | None:
    if table is None or table.empty:
        return None
    for column in ["candidate_id", "binder_id", "original_binder_id"]:
        if column not in table.columns:
            continue
        subset = table[table[column].astype(str) == str(design_id)]
        if not subset.empty:
            return subset.iloc[0]
    return None


def _sequence_validation_parent_reference(
    run_dir: Path,
    row: pd.Series | None,
    design_id: str,
) -> tuple[Path | None, list[str], list[str], str]:
    if row is None:
        return None, [], [], ""
    for column in ["source_parent_complex_pdb", "parent_complex_pdb"]:
        if column in row.index:
            path = _resolve_run_relative_path(run_dir, row.get(column))
            if path is not None:
                return path, _parse_chain_list(row.get("source_parent_binder_chains")), _parse_chain_list(row.get("source_parent_target_chains")), column

    design_run_dir = _resolve_run_relative_path(run_dir, row.get("source_input_run_dir") if "source_input_run_dir" in row.index else None)
    if design_run_dir is None:
        return None, [], [], ""
    design_input = read_json(design_run_dir / "input.json")
    inputs = design_input.get("inputs") if isinstance(design_input.get("inputs"), dict) else {}
    params = design_input.get("params") if isinstance(design_input.get("params"), dict) else {}
    parent_source_run_dir = (
        _resolve_run_relative_path(design_run_dir, inputs.get("source_run_dir"))
        or _resolve_run_relative_path(design_run_dir, params.get("source_run_dir"))
    )
    candidates_jsonl = design_run_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl"
    if not candidates_jsonl.exists():
        return None, [], [], ""
    try:
        with candidates_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(candidate.get("candidate_id") or "") != str(design_id):
                    continue
                raw_metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
                source_candidate = raw_metadata.get("source_candidate") if isinstance(raw_metadata.get("source_candidate"), dict) else {}
                source_complex = source_candidate.get("complex_pdb")
                source_path = None
                if parent_source_run_dir is not None:
                    source_path = _resolve_run_relative_path(parent_source_run_dir, source_complex)
                if source_path is None:
                    source_path = _resolve_run_relative_path(design_run_dir, source_complex)
                if source_path is not None:
                    return (
                        source_path,
                        _parse_chain_list(source_candidate.get("binder_chains")),
                        _parse_chain_list(source_candidate.get("target_chains")),
                        "parent source candidate",
                    )
                return None, [], [], ""
    except OSError:
        return None, [], [], ""
    return None, [], [], ""


def _sequence_validation_source_candidate(
    run_dir: Path,
    row: pd.Series | None,
    design_id: str,
) -> dict[str, Any]:
    if row is None:
        return {}
    design_run_dir = _resolve_run_relative_path(run_dir, row.get("source_input_run_dir") if "source_input_run_dir" in row.index else None)
    if design_run_dir is None:
        return {}
    candidates_jsonl = design_run_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl"
    if not candidates_jsonl.exists():
        return {}
    try:
        with candidates_jsonl.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    candidate = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(candidate.get("candidate_id") or "") != str(design_id):
                    continue
                raw_metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
                source_candidate = raw_metadata.get("source_candidate")
                return source_candidate if isinstance(source_candidate, dict) else {}
    except OSError:
        return {}
    return {}


def _sequence_validation_metric_number_from_metrics(metrics: dict[str, Any], key: str) -> float | None:
    aliases = {
        "ipsae": ["af3_ipSAE_min", "af3_ipsae_min", "ipSAE_min", "ipsae_min"],
        "dg_sasa": ["af3_rosetta_interface_dG_dSASA_ratio", "rosetta_interface_dG_dSASA_ratio"],
        "shape_complementarity": ["af3_rosetta_interface_sc", "rosetta_interface_sc"],
        "target_aligned_binder_rmsd": ["target_aligned_binder_rmsd"],
    }
    if key == "ipsae__x__dg_sasa":
        left = _sequence_validation_metric_number_from_metrics(metrics, "ipsae")
        right = _sequence_validation_metric_number_from_metrics(metrics, "dg_sasa")
        return None if left is None or right is None else left * right
    for alias in aliases.get(key, [key]):
        value = metrics.get(alias)
        try:
            if value is None or pd.isna(value):
                continue
            return float(value)
        except (TypeError, ValueError):
            continue
    if key == "target_aligned_binder_rmsd":
        return 0.0
    return None


def _sequence_validation_feature_value_from_metrics(metrics: dict[str, Any], feature: str) -> tuple[float | None, str, str]:
    if not isinstance(metrics, dict) or not feature:
        return None, "missing_parent_metric", ""

    def _as_float(value: object) -> float | None:
        try:
            if value is None or pd.isna(value):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    value = _as_float(metrics.get(feature))
    if value is not None:
        return value, "same_feature", str(feature)

    def _strip_engine_label(term: object) -> str:
        text = str(term or "").strip()
        for engine in _engine_labels_by_specificity():
            prefix = f"{engine} "
            if text.lower().startswith(prefix.lower()):
                return text[len(prefix):].strip()
        return text

    def _expected_engine_prefixes(text: object) -> tuple[str, ...]:
        lowered = str(text or "").lower()
        mapping = {
            "af3": ("af3_", "alphafast_af3_"),
            "colabfold": ("colab_", "colabfold_"),
            "af2-ig": ("af2_",),
            "af2 initial guess": ("af2_",),
            "esmfold2": ("esmfold2_",),
            "boltz-2": ("boltz2_",),
            "rf3": ("rf3_",),
            "openfold-3": ("openfold3_",),
            "protenix v1": ("protenix_v1_",),
            "protenix v2": ("protenix_v2_",),
            "protenix v0.5": ("protenix_",),
            "protenix": ("protenix_",),
            "boltzgen fold": ("boltzgen_fold_",),
        }
        for label, prefixes in mapping.items():
            if lowered.startswith(f"{label} "):
                return prefixes
        for prefix in FEATURE_ENGINE_PREFIXES:
            if lowered.startswith(prefix):
                return (prefix,)
        return ()

    feature_text = str(feature)
    expected_prefixes = _expected_engine_prefixes(feature_text)
    feature_term = _combo_feature_term(_strip_engine_label(feature_text)).lower()
    feature_raw = _raw_feature_name(_strip_engine_label(feature_text)).lower()
    metric_terms: dict[str, tuple[float, str]] = {}
    for key, metric_value in metrics.items():
        numeric = _as_float(metric_value)
        if numeric is None:
            continue
        key_text = str(key)
        aliases = {
            key_text.lower(),
            _combo_feature_term(key_text).lower(),
            _raw_feature_name(key_text).lower(),
            key_text.lower().replace("/", "_"),
            _combo_feature_term(key_text).lower().replace("/", "_"),
            _raw_feature_name(key_text).lower().replace("/", "_"),
        }
        for alias in aliases:
            metric_terms.setdefault(alias, (numeric, key_text))

    def _alias_value(alias: str, prefixes: tuple[str, ...] | None = None) -> tuple[float | None, str | None]:
        active_prefixes = expected_prefixes if prefixes is None else prefixes
        match = metric_terms.get(alias)
        if match is None:
            return None, None
        numeric, key_text = match
        if active_prefixes and not key_text.lower().startswith(active_prefixes):
            return None, key_text
        return numeric, key_text

    aliases_to_check = {
        feature_text.lower(),
        _strip_engine_label(feature_text).lower(),
        feature_term,
        feature_raw,
        feature_text.lower().replace("/", "_"),
        _strip_engine_label(feature_text).lower().replace("/", "_"),
        feature_term.replace("/", "_"),
    }
    engine_mismatch_key = ""
    for alias in aliases_to_check:
        numeric, key_text = _alias_value(alias)
        if numeric is not None:
            return numeric, "same_feature", key_text or alias
        if key_text:
            engine_mismatch_key = key_text

    if "×" in feature_text:
        terms = [part.strip() for part in feature_text.split("×", 1)]
    elif " x " in feature_text.lower():
        terms = [part.strip() for part in re.split(r"\s+x\s+", feature_text, maxsplit=1, flags=re.IGNORECASE)]
    else:
        return None, "missing_parent_metric", ""
    if len(terms) != 2:
        return None, "ambiguous_metric_match", ""
    left = _combo_feature_term(_strip_engine_label(terms[0])).lower()
    right = _combo_feature_term(_strip_engine_label(terms[1])).lower()
    left_prefixes = _expected_engine_prefixes(terms[0]) or expected_prefixes
    right_text = str(terms[1] or "").lower()
    if right_text.startswith("input ") or " input " in right_text:
        right_prefixes = ("input_",)
    else:
        right_prefixes = _expected_engine_prefixes(terms[1]) or expected_prefixes
    left_value, left_key = _alias_value(left, left_prefixes)
    if left_value is None:
        left_value, left_key = _alias_value(left.replace("/", "_"), left_prefixes)
    right_value, right_key = _alias_value(right, right_prefixes)
    if right_value is None:
        right_value, right_key = _alias_value(right.replace("/", "_"), right_prefixes)
    if left_key and left_value is None:
        engine_mismatch_key = left_key
    if right_key and right_value is None:
        engine_mismatch_key = right_key
    if engine_mismatch_key:
        return None, "engine_mismatch", engine_mismatch_key
    if left_value is None or right_value is None:
        return None, "missing_parent_metric", f"{left}; {right}"
    return left_value * right_value, "reconstructed_combo", f"{left_key or left}; {right_key or right}"


def _sequence_validation_filter_metric_keys(filter_config: dict[str, Any] | None) -> list[str]:
    keys: list[str] = []
    for metric in (filter_config or {}).get("scoring_weights", {}).keys():
        metric_text = str(metric or "")
        if metric_text and metric_text not in keys:
            keys.append(metric_text)
    for rule in (filter_config or {}).get("filters", []):
        if not isinstance(rule, dict) or not bool(rule.get("enabled", True)):
            continue
        metric_text = str(rule.get("metric") or "")
        if metric_text and metric_text not in keys:
            keys.append(metric_text)
    return keys


def _sequence_validation_add_parent_delta_columns(
    table: pd.DataFrame,
    *,
    run_dir: Path,
    merged: pd.DataFrame | None,
    filter_config: dict[str, Any] | None,
) -> pd.DataFrame:
    if table.empty:
        return table
    id_column = _sequence_validation_id_column(table)
    metric_keys = _sequence_validation_filter_metric_keys(filter_config)
    if not metric_keys:
        return table
    out = table.copy()
    for index, row in out.iterrows():
        design_id = str(row.get(id_column) or "")
        metric_row = _sequence_validation_row_for_id(merged, design_id)
        source_candidate = _sequence_validation_source_candidate(run_dir, metric_row, design_id)
        source_metrics = source_candidate.get("metrics") if isinstance(source_candidate.get("metrics"), dict) else {}
        for metric in metric_keys:
            original = _sequence_validation_metric_number_from_metrics(source_metrics, metric)
            if original is None:
                continue
            out.at[index, f"parent_{metric}"] = original
            if metric in out.columns:
                current = pd.to_numeric(pd.Series([out.at[index, metric]]), errors="coerce").iloc[0]
                if pd.notna(current):
                    out.at[index, f"delta_{metric}"] = float(current) - original
    return out


def _sequence_validation_metric_columns(columns: list[str], prefix: str) -> list[str]:
    preferred = [
        f"{prefix}_target_aligned_binder_rmsd",
        f"{prefix}_ipSAE_min",
        f"{prefix}_ipsae_min",
        f"{prefix}_rosetta_interface_dG_dSASA_ratio",
        f"{prefix}_rosetta_interface_sc",
        f"{prefix}_pDockQ_min",
        f"{prefix}_pose_rmsd_status",
    ]
    return [column for column in preferred if column in columns]


def _sequence_validation_metric_category(column: str) -> str:
    text = str(column)
    lowered = text.lower()
    if "× input" in text or " x input" in lowered:
        return "PAE × Input Rosetta combinations"
    if "×" in text or " x " in lowered:
        return "PAE × Rosetta combinations"
    if any(token in lowered for token in ["ipsae", "ipae", "pae", "pdockq", "lis"]):
        return "Recalculated PAE / Interface"
    if "rosetta" in lowered or "sap_" in lowered:
        return "Interface Energy / Rosetta"
    if "pymol" in lowered or "sasa" in lowered or "hbonds" in lowered:
        return "Interface Geometry / PyMOL"
    if "rmsd" in lowered:
        return "Pose / RMSD"
    if lowered.startswith("design_") or lowered in {
        "max_same_aa_run",
        "aaa_runs",
        "gly_runs",
        "pro_runs",
        "alanine_fraction",
        "glycine_fraction",
        "proline_fraction",
        "cysteine_fraction",
        "charged_fraction",
        "hydrophobic_fraction",
        "net_charge_per_residue",
        "aa_entropy",
        "low_complexity_fraction",
    }:
        return "Sequence Quality / MPNN"
    if lowered.startswith("input_"):
        return "Input / Reference"
    return "Other"


def _sequence_validation_metric_direction(column: str) -> str:
    lowered = str(column).lower()
    if "×" in str(column) or " x " in lowered:
        return _feature_sort_direction(column)
    lower_better_tokens = [
        "rmsd",
        "pae",
        "ipae",
        "dg",
        "sap",
        "unsat",
        "low_complexity",
        "max_same_aa_run",
        "aaa_runs",
        "gly_runs",
        "pro_runs",
        "cysteine_fraction",
    ]
    higher_better_tokens = ["ipsae", "pdockq", "lis", "iptm", "ptm", "plddt", "confidence", "sc", "packstat", "entropy"]
    if any(token in lowered for token in higher_better_tokens):
        return "higher"
    if any(token in lowered for token in lower_better_tokens):
        return "lower"
    return "higher"


def _sequence_validation_numeric_columns(table: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in table.columns:
        if column in {"rank", "candidate_id", "binder_id", "original_binder_id"}:
            continue
        values = pd.to_numeric(table[column], errors="coerce")
        if values.notna().any():
            columns.append(str(column))
    return columns


def _sequence_validation_rank_metric_options(table: pd.DataFrame, engine_prefix: str) -> dict[str, list[str]]:
    numeric_columns = _sequence_validation_numeric_columns(table)
    engine_label = _feature_engine(f"{engine_prefix}_placeholder")
    engine_columns = [
        column
        for column in numeric_columns
        if not _is_amino_acid_composition_feature(column)
        if not _is_size_or_count_proxy_feature(column)
        if str(column).startswith(f"{engine_prefix}_")
        or str(column).startswith(f"alphafast_{engine_prefix}_")
        or (engine_prefix == "af3" and str(column).startswith("alphafast_af3_"))
        or (
            _sequence_validation_metric_category(column)
            in {"PAE × Rosetta combinations", "PAE × Input Rosetta combinations"}
            and (
                str(column).startswith(f"{engine_label} ")
                or f"{engine_prefix}_" in str(column)
                or (engine_prefix == "af3" and "alphafast_af3_" in str(column))
            )
        )
    ]
    grouped: dict[str, list[str]] = {}
    for column in engine_columns:
        category = _sequence_validation_metric_category(column)
        if category in {"Sequence Quality / MPNN", "Pose / RMSD"}:
            continue
        grouped.setdefault(category, []).append(column)
    preferred_order = [
        "PAE × Rosetta combinations",
        "PAE × Input Rosetta combinations",
        "Recalculated PAE / Interface",
        "Interface Energy / Rosetta",
        "Interface Geometry / PyMOL",
        "Other",
    ]
    return {category: sorted(grouped[category]) for category in preferred_order if grouped.get(category)}


def _sequence_validation_metric_label(column: str) -> str:
    return str(column).replace("_", " ")


def _sequence_validation_rank_metric_priority(column: str) -> tuple[int, str]:
    lowered = str(column).lower()
    if "×" in str(column) or " x " in lowered:
        if ("ipsae_min" in lowered or "ipsae" in lowered) and ("dg_dsasa" in lowered or "dg/sasa" in lowered):
            return (0, lowered)
        if "pdockq2" in lowered and ("dg_dsasa" in lowered or "interface_dg" in lowered):
            return (1, lowered)
        if ("ipsae_min" in lowered or "ipsae" in lowered) and "interface_sc" in lowered:
            return (2, lowered)
        if "lis" in lowered and ("dg_dsasa" in lowered or "interface_dg" in lowered or "interface_sc" in lowered):
            return (3, lowered)
        if "interface_aa_" in lowered:
            return (20, lowered)
        return (10, lowered)
    if "ipsae_min" in lowered:
        return (0, lowered)
    if "pdockq2_min" in lowered:
        return (1, lowered)
    if "interface_dg_dsasa" in lowered or "interface_dg" in lowered:
        return (2, lowered)
    if "interface_sc" in lowered:
        return (3, lowered)
    return (10, lowered)


def _sequence_validation_apply_threshold(table: pd.DataFrame, column: str, operator: str, threshold: object) -> pd.DataFrame:
    if column not in table.columns:
        return table.iloc[0:0].copy()
    values = pd.to_numeric(table[column], errors="coerce")
    try:
        cutoff = float(threshold)
    except (TypeError, ValueError):
        return table
    if operator == "<=":
        mask = values <= cutoff
    elif operator == "<":
        mask = values < cutoff
    elif operator == ">=":
        mask = values >= cutoff
    elif operator == ">":
        mask = values > cutoff
    else:
        mask = values == cutoff
    return table[mask.fillna(False)].copy()


def _sequence_validation_metric_value(row: pd.Series | None, column: str, fmt: str, suffix: str = "") -> str:
    if row is None or column not in row.index or pd.isna(row.get(column)):
        return "n/a"
    try:
        return f"{float(row.get(column)):{fmt}}{suffix}"
    except Exception:
        return str(row.get(column))


def _sequence_validation_display_value(row: pd.Series | None, column: str) -> str:
    if row is None or column not in row.index or pd.isna(row.get(column)):
        return "n/a"
    value = row.get(column)
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if column.endswith("rmsd"):
        return f"{number:.2f} Å"
    return f"{number:.3f}"


def _parse_sequence_design_fasta_header(header: object) -> dict[str, Any]:
    text = str(header or "")
    parsed: dict[str, Any] = {}
    for part in text.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value_text = value.strip()
        if not key:
            continue
        try:
            value_obj: Any = float(value_text)
            if value_obj.is_integer() and key.lower() in {"id", "seed"}:
                value_obj = int(value_obj)
        except ValueError:
            value_obj = value_text
        parsed[key] = value_obj
    return parsed


def _sequence_quality_metrics(sequence: object) -> dict[str, Any]:
    seq = "".join(char for char in str(sequence or "").upper() if char.isalpha())
    length = len(seq)
    metrics: dict[str, Any] = {
        "sequence_length": length if length else None,
        "max_same_aa_run": None,
        "aa_entropy": None,
        "low_complexity_fraction": None,
        "alanine_fraction": None,
        "glycine_fraction": None,
        "proline_fraction": None,
        "cysteine_fraction": None,
        "charged_fraction": None,
        "positive_fraction": None,
        "negative_fraction": None,
        "hydrophobic_fraction": None,
        "polar_fraction": None,
        "aromatic_fraction": None,
        "net_charge_per_residue": None,
        "aaa_runs": 0,
        "gly_runs": 0,
        "pro_runs": 0,
        "has_4x_same_aa_run": False,
    }
    if not seq:
        return metrics
    counts = {aa: seq.count(aa) for aa in set(seq)}
    max_run = 1
    current_run = 1
    for index in range(1, length):
        if seq[index] == seq[index - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    hydrophobic = set("AILMFWVY")
    polar = set("STNQCY")
    aromatic = set("FWY")
    positive = set("KRH")
    negative = set("DE")
    entropy = 0.0
    for count in counts.values():
        p = count / length
        entropy -= p * np.log2(p)
    max_entropy = np.log2(20)
    metrics.update(
        {
            "max_same_aa_run": max_run,
            "aa_entropy": entropy,
            "low_complexity_fraction": 1.0 - (entropy / max_entropy),
            "alanine_fraction": seq.count("A") / length,
            "glycine_fraction": seq.count("G") / length,
            "proline_fraction": seq.count("P") / length,
            "cysteine_fraction": seq.count("C") / length,
            "charged_fraction": sum(seq.count(aa) for aa in positive | negative) / length,
            "positive_fraction": sum(seq.count(aa) for aa in positive) / length,
            "negative_fraction": sum(seq.count(aa) for aa in negative) / length,
            "hydrophobic_fraction": sum(seq.count(aa) for aa in hydrophobic) / length,
            "polar_fraction": sum(seq.count(aa) for aa in polar) / length,
            "aromatic_fraction": sum(seq.count(aa) for aa in aromatic) / length,
            "net_charge_per_residue": (sum(seq.count(aa) for aa in positive) - sum(seq.count(aa) for aa in negative)) / length,
            "aaa_runs": len(re.findall(r"A{3,}", seq)),
            "gly_runs": len(re.findall(r"G{3,}", seq)),
            "pro_runs": len(re.findall(r"P{3,}", seq)),
            "has_4x_same_aa_run": bool(re.search(r"([A-Z])\1{3,}", seq)),
        }
    )
    return metrics


@st.cache_data(show_spinner=False)
def _sequence_design_candidate_info_table(run_dir_text: str) -> pd.DataFrame:
    run_dir = Path(run_dir_text)
    input_payload = read_json(run_dir / "input.json")
    input_section = input_payload.get("inputs") if isinstance(input_payload.get("inputs"), dict) else {}
    candidates_jsonl = _resolve_run_relative_path(
        run_dir,
        input_payload.get("candidates_jsonl") or input_section.get("candidates_jsonl"),
    )
    if candidates_jsonl is None or not candidates_jsonl.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for candidate in read_candidates(candidates_jsonl):
        candidate_id = str(candidate.get("candidate_id") or "")
        if not candidate_id:
            continue
        raw_metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
        metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
        source_candidate = raw_metadata.get("source_candidate") if isinstance(raw_metadata.get("source_candidate"), dict) else {}
        source_metrics = source_candidate.get("metrics") if isinstance(source_candidate.get("metrics"), dict) else {}
        header_values = _parse_sequence_design_fasta_header(raw_metadata.get("fasta_header"))
        binder_sequence = str(candidate.get("binder_sequence") or "")
        parent_sequence = str(source_candidate.get("binder_sequence") or source_candidate.get("sequence") or "")
        sequence_metrics = _sequence_quality_metrics(binder_sequence)
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "design_source_tool": candidate.get("source_tool"),
            "binder_sequence": binder_sequence,
            "parent_sequence": parent_sequence,
            "parent_candidate_id": source_candidate.get("candidate_id"),
            "_parent_metrics": source_metrics,
            "parent_id": (candidate.get("parents") or [""])[0],
            "design_index": raw_metadata.get("ligandmpnn_design_index") or header_values.get("id"),
            "sequence_index": raw_metadata.get("sequence_index"),
            "fixed_residue_count": len(raw_metadata.get("fixed_residues") or []),
            "design_overall_confidence": header_values.get("overall_confidence"),
            "design_ligand_confidence": header_values.get("ligand_confidence"),
            "design_seq_rec": header_values.get("seq_rec"),
            "design_temperature": header_values.get("T"),
            "design_seed": header_values.get("seed"),
        }
        row.update(sequence_metrics)
        for key, value in metrics.items():
            column = f"design_{key}"
            if column not in row:
                row[column] = value
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(subset=["candidate_id"], keep="last")


def _sequence_validation_merge_design_info(table: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    if table.empty or "candidate_id" not in table.columns:
        return table
    design_info = _sequence_design_candidate_info_table(str(run_dir))
    if design_info.empty or "candidate_id" not in design_info.columns:
        return table
    info_cols = [column for column in design_info.columns if column == "candidate_id" or column not in table.columns]
    if len(info_cols) <= 1:
        return table
    return table.merge(design_info[info_cols], on="candidate_id", how="left")


def _pdb_sidechain_summary(pdb_text: str, chains: list[str]) -> dict[str, int]:
    keep = set(chains)
    residues: set[tuple[str, str, str]] = set()
    sidechain_atoms = 0
    backbone_names = {"N", "CA", "C", "O", "OXT"}
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
            continue
        chain = line[21].strip() or "_"
        if keep and chain not in keep:
            continue
        atom_name = line[12:16].strip()
        residues.add((chain, line[22:26].strip(), line[26].strip()))
        if atom_name not in backbone_names:
            sidechain_atoms += 1
    return {"residues": len(residues), "sidechain_atoms": sidechain_atoms}


def _sequence_validation_rule_label(metric: str) -> str:
    labels = {
        "ipsae__x__dg_sasa": "ipSAE min × Rosetta dG/dSASA (lower better)",
        "shape_complementarity": "Rosetta shape complementarity (higher better)",
        "target_aligned_binder_rmsd": "Target-aligned binder RMSD (lower better)",
        "combined_score": "Normalized rank score (higher better)",
    }
    return labels.get(str(metric), str(metric).replace("_", " "))


def _sequence_validation_apply_rule(values: pd.Series, operator: str, threshold: object) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    try:
        cutoff = float(threshold)
    except (TypeError, ValueError):
        return pd.Series(False, index=values.index)
    if operator == ">=":
        return numeric >= cutoff
    if operator == ">":
        return numeric > cutoff
    if operator == "<=":
        return numeric <= cutoff
    if operator == "<":
        return numeric < cutoff
    if operator in {"=", "=="}:
        return numeric == cutoff
    return pd.Series(False, index=values.index)


def _sequence_validation_filter_diagnosis(ranking: pd.DataFrame, filter_config: dict[str, Any] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = ranking.copy()
    rules = [
        rule
        for rule in (filter_config or {}).get("filters", [])
        if isinstance(rule, dict) and bool(rule.get("enabled", True))
    ]
    summary_rows: list[dict[str, Any]] = []
    failed_labels: list[list[str]] = [[] for _ in range(len(table))]
    for rule_index, rule in enumerate(rules, start=1):
        metric = str(rule.get("metric") or "")
        operator = str(rule.get("operator") or "")
        threshold = rule.get("threshold")
        label = _sequence_validation_rule_label(metric)
        pass_column = f"rule_{rule_index}_pass"
        value_column = metric if metric in table.columns else None
        if value_column:
            passes = _sequence_validation_apply_rule(table[value_column], operator, threshold).fillna(False)
        else:
            passes = pd.Series(False, index=table.index)
        table[pass_column] = passes
        for pos, ok in enumerate(passes.tolist()):
            if not ok:
                failed_labels[pos].append(label)
        summary_rows.append(
            {
                "rule": label,
                "metric": metric,
                "operator": operator,
                "threshold": threshold,
                "passing": int(passes.sum()),
                "failing": int((~passes).sum()),
                "available": bool(value_column),
            }
        )
    if rules:
        table["failed_rules"] = ["; ".join(labels) if labels else "" for labels in failed_labels]
    elif "passes_filters" in table.columns:
        table["failed_rules"] = ""
    return table, pd.DataFrame(summary_rows)


def _show_sequence_design_validation_results(run_dir: Path, result: dict) -> None:
    benchmark_dir = run_dir / "artifacts" / "benchmark"
    merged = _read_csv(benchmark_dir / "merged_benchmark_metrics.csv")
    ranking = _read_csv(benchmark_dir / "sequence_design_weighted_ranking.csv")
    filter_config = read_json(benchmark_dir / "sequence_design_filter_config.json")
    metrics = result.get("metrics") or {}
    available_engines = _sequence_validation_available_engines(benchmark_dir, merged)

    st.header("Sequence Design Validation Results")
    retained_count = 0
    if ranking is not None and "retained" in ranking.columns:
        retained_count = int(ranking["retained"].astype(str).str.lower().isin({"true", "1"}).sum())
    cols = st.columns(5)
    cols[0].metric("Validated sequences", int(len(merged)) if merged is not None else metrics.get("record_count", "n/a"))
    cols[1].metric("Ranked", int(len(ranking)) if ranking is not None else "n/a")
    cols[2].metric("Retained", retained_count if ranking is not None else "n/a")
    cols[3].metric("Engines", len(available_engines) or "n/a")
    cols[4].metric("Pose RMSD ok", metrics.get("pose_rmsd_ok", "n/a"))
    if filter_config:
        st.caption(
            f"Scoring engine: {filter_config.get('engine', 'n/a')}; "
            f"keep per parent: {filter_config.get('keep_per_parent', 'n/a')}; "
            f"filter logic: {filter_config.get('filter_logic', 'all')}."
        )

    view = st.segmented_control(
        "Sequence-design result view",
        ["Ranked Sequences", "Filter Diagnosis", "Metrics", "Structure Viewer", "Inputs and Outputs"],
        default="Ranked Sequences",
        key=f"{run_dir.name}_sequence_validation_result_view_v1",
    ) or "Ranked Sequences"

    if view == "Ranked Sequences":
        ranking_source = "weighted ranking"
        if ranking is None or ranking.empty:
            if merged is None or merged.empty:
                st.info("No sequence-design ranking table was written, and no merged metrics are available.")
                return
            st.info(
                "No weighted sequence-design ranking table was written for this run. "
                "Showing the merged validation metrics in their recorded order instead."
            )
            table = merged.copy()
            ranking_source = "merged metrics fallback"
            if "rank" not in table.columns:
                table.insert(0, "rank", range(1, len(table) + 1))
        else:
            table = ranking.copy()
        if table.empty:
            st.info("No sequence-design rows are available.")
            return
        if filter_config:
            table, _summary = _sequence_validation_filter_diagnosis(table, filter_config)
        table = _sequence_validation_add_parent_delta_columns(
            table,
            run_dir=run_dir,
            merged=merged,
            filter_config=filter_config,
        )
        table = _sequence_validation_merge_design_info(table, run_dir)
        if "retained" in table.columns:
            show_retained_only = st.checkbox("Show retained only", value=False, key=f"{run_dir.name}_sequence_validation_retained_only_v1")
        else:
            show_retained_only = False
        if show_retained_only and "retained" in table.columns:
            table = table[table["retained"].astype(str).str.lower().isin({"true", "1"})].copy()
        comparison_cols = []
        for metric in _sequence_validation_filter_metric_keys(filter_config):
            comparison_cols.extend(
                column
                for column in [metric, f"parent_{metric}", f"delta_{metric}"]
                if column in table.columns
            )
        display_cols = [
            column
            for column in [
                "rank",
                "candidate_id",
                "parent_id",
                "design_source_tool",
                "binder_sequence",
                "sequence_length",
                "max_same_aa_run",
                "aaa_runs",
                "alanine_fraction",
                "low_complexity_fraction",
                "design_overall_confidence",
                "design_ligand_confidence",
                "design_seq_rec",
                "retained",
                "passes_filters",
                *comparison_cols,
                "failed_rules",
            ]
            if column in table.columns
        ] or list(table.columns[: min(12, len(table.columns))])
        st.caption(f"Table source: {ranking_source}; rows shown: {len(table)}.")
        st.dataframe(table[display_cols], hide_index=True, width="stretch", height=520)
        st.download_button(
            "Download ranked sequence table CSV",
            data=table.to_csv(index=False),
            file_name=f"{run_dir.name}_sequence_design_ranked_sequences.csv",
            mime="text/csv",
        )
        return

    if view == "Filter Diagnosis":
        if ranking is None or ranking.empty:
            st.info("No sequence-design ranking table was written.")
            return
        diagnosis, rule_summary = _sequence_validation_filter_diagnosis(ranking, filter_config)
        diagnosis = _sequence_validation_add_parent_delta_columns(
            diagnosis,
            run_dir=run_dir,
            merged=merged,
            filter_config=filter_config,
        )
        if filter_config:
            logic = str(filter_config.get("filter_logic") or "all")
            st.caption(
                f"Filtering uses `{logic}` logic on the declared rules. "
                "The table below is pre-filtering: retained and rejected structures are both shown."
            )
        if not rule_summary.empty:
            st.subheader("Declared Filter Rules")
            st.dataframe(rule_summary, hide_index=True, width="stretch")
        if "passes_filters" in diagnosis.columns:
            passing = diagnosis["passes_filters"].astype(str).str.lower().isin({"true", "1"})
            cols = st.columns(3)
            cols[0].metric("Before filtering", len(diagnosis))
            cols[1].metric("Pass filters", int(passing.sum()))
            cols[2].metric("Rejected", int((~passing).sum()))
        mode = st.segmented_control(
            "Rows",
            ["All", "Rejected only", "Passing only"],
            default="Rejected only",
            key=f"{run_dir.name}_sequence_validation_filter_rows_v1",
        ) or "Rejected only"
        shown = diagnosis.copy()
        if "passes_filters" in shown.columns:
            passing = shown["passes_filters"].astype(str).str.lower().isin({"true", "1"})
            if mode == "Rejected only":
                shown = shown[~passing].copy()
            elif mode == "Passing only":
                shown = shown[passing].copy()
        comparison_cols = []
        for metric in _sequence_validation_filter_metric_keys(filter_config):
            comparison_cols.extend(
                column
                for column in [metric, f"parent_{metric}", f"delta_{metric}"]
                if column in shown.columns
            )
        display_cols = [
            column
            for column in [
                "rank",
                "candidate_id",
                "parent_id",
                "passes_filters",
                "retained",
                "failed_rules",
                *comparison_cols,
            ]
            if column in shown.columns
        ] or list(shown.columns[: min(16, len(shown.columns))])
        st.subheader("Prefiltered Structures")
        st.dataframe(shown[display_cols], hide_index=True, width="stretch", height=560)
        st.download_button(
            "Download filter diagnosis CSV",
            data=diagnosis.to_csv(index=False),
            file_name=f"{run_dir.name}_sequence_design_filter_diagnosis.csv",
            mime="text/csv",
        )
        return

    if view == "Metrics":
        if merged is None or merged.empty:
            st.info("No merged validation metrics are available.")
            return
        metrics_table = merged.copy()
        ranking_metric_cols: list[str] = []
        if ranking is not None and not ranking.empty and "candidate_id" in ranking.columns and "candidate_id" in metrics_table.columns:
            ranking_metric_cols = [
                column
                for column in [
                    "rank",
                    "parent_id",
                    "retained",
                    "passes_filters",
                    "ipsae__x__dg_sasa",
                    "shape_complementarity",
                    "target_aligned_binder_rmsd",
                ]
                if column in ranking.columns and column not in metrics_table.columns
            ]
            weighted_cols = [
                column
                for column in ranking.columns
                if str(column).startswith("weighted_") and column not in metrics_table.columns
            ]
            ranking_metric_cols.extend(weighted_cols)
            if ranking_metric_cols:
                metrics_table = metrics_table.merge(
                    ranking[["candidate_id", *ranking_metric_cols]],
                    on="candidate_id",
                    how="left",
                )
        metrics_table = _sequence_validation_add_parent_delta_columns(
            metrics_table,
            run_dir=run_dir,
            merged=metrics_table,
            filter_config=filter_config,
        )
        metrics_table = _sequence_validation_merge_design_info(metrics_table, run_dir)
        engine_labels = [engine[0] for engine in available_engines] or ["All"]
        selected_label = st.selectbox("Engine", engine_labels, key=f"{run_dir.name}_sequence_validation_metric_engine_v1")
        selected_engine = next((engine for engine in available_engines if engine[0] == selected_label), None)
        base_cols = [
            column
            for column in [
                "binder_id",
                "candidate_id",
                "target_id",
                "parent_id",
                "design_source_tool",
                "binder_sequence",
                "sequence_length",
                "max_same_aa_run",
                "aaa_runs",
                "has_4x_same_aa_run",
                "alanine_fraction",
                "glycine_fraction",
                "proline_fraction",
                "cysteine_fraction",
                "charged_fraction",
                "hydrophobic_fraction",
                "net_charge_per_residue",
                "aa_entropy",
                "low_complexity_fraction",
                "design_overall_confidence",
                "design_ligand_confidence",
                "design_seq_rec",
                "design_temperature",
                "design_seed",
                "design_index",
                "sequence_index",
                "fixed_residue_count",
                "rank",
                "retained",
                "passes_filters",
            ]
            if column in metrics_table.columns
        ]
        comparison_cols = []
        for metric in _sequence_validation_filter_metric_keys(filter_config):
            comparison_cols.extend(
                column
                for column in [metric, f"parent_{metric}", f"delta_{metric}"]
                if column in metrics_table.columns
            )
        ranking_cols = [
            column
            for column in ranking_metric_cols
            if column in metrics_table.columns and column not in base_cols and column not in comparison_cols
        ]
        metric_cols = _sequence_validation_metric_columns(list(metrics_table.columns), selected_engine[1]) if selected_engine else []
        if not metric_cols and selected_engine:
            metric_cols = [column for column in metrics_table.columns if str(column).startswith(f"{selected_engine[1]}_")][:40]
        show_cols = list(dict.fromkeys(base_cols + comparison_cols + ranking_cols + metric_cols)) or list(metrics_table.columns[: min(40, len(metrics_table.columns))])
        st.dataframe(metrics_table[show_cols], hide_index=True, width="stretch", height=560)
        st.download_button(
            "Download merged metrics CSV",
            data=metrics_table.to_csv(index=False),
            file_name=f"{run_dir.name}_merged_benchmark_metrics.csv",
            mime="text/csv",
        )
        return

    if view == "Structure Viewer":
        if merged is None or merged.empty:
            st.info("No merged metrics are available for structure selection.")
            return
        selector_table = merged.copy()
        if ranking is not None and not ranking.empty and "candidate_id" in ranking.columns and "candidate_id" in selector_table.columns:
            ranking_extra_cols = [
                column
                for column in ["rank", "parent_id", "retained", "passes_filters"]
                if column in ranking.columns and column not in selector_table.columns
            ]
            if ranking_extra_cols:
                selector_table = selector_table.merge(
                    ranking[["candidate_id", *ranking_extra_cols]].drop_duplicates(subset=["candidate_id"], keep="last"),
                    on="candidate_id",
                    how="left",
                )
        selector_table = _sequence_validation_add_parent_delta_columns(
            selector_table,
            run_dir=run_dir,
            merged=selector_table,
            filter_config=filter_config,
        )
        selector_table = _sequence_validation_merge_design_info(selector_table, run_dir)
        combo_table, combo_features = _cached_pae_rosetta_combo_metric_columns(selector_table, 8)
        if combo_table is not None and not combo_table.empty:
            selector_table = combo_table
        input_combo_table, input_combo_features = _cached_pae_input_rosetta_combo_metric_columns(selector_table, 8)
        if input_combo_table is not None and not input_combo_table.empty:
            selector_table = input_combo_table
        id_column = _sequence_validation_id_column(selector_table)
        if not available_engines:
            st.info("No staged prediction structures were found for this validation run.")
            return
        engine_labels = [engine[0] for engine in available_engines]
        default_engine = str(filter_config.get("engine") or "AF3")
        default_index = engine_labels.index(default_engine) if default_engine in engine_labels else 0
        selected_engine_label = st.selectbox(
            "Prediction engine / ranking source",
            engine_labels,
            index=default_index,
            key=f"{run_dir.name}_sequence_validation_structure_engine_v1",
        )
        selected_engine = next(engine for engine in available_engines if engine[0] == selected_engine_label)
        metric_groups = _sequence_validation_rank_metric_options(selector_table, selected_engine[1])
        if not metric_groups:
            st.info(f"No numeric interface metrics were found for {selected_engine_label}.")
            return
        category_options = list(metric_groups)
        default_category = "PAE × Rosetta combinations" if "PAE × Rosetta combinations" in category_options else category_options[0]
        selected_category = st.segmented_control(
            "Ranking metric category",
            category_options,
            default=default_category,
            key=f"{run_dir.name}_sequence_validation_structure_rank_category_v1",
        ) or default_category
        metric_filter = st.text_input(
            "Filter ranking metrics",
            value="",
            placeholder="e.g. ipsae, pDockQ, rosetta, rmsd",
            key=f"{run_dir.name}_sequence_validation_structure_metric_filter_v1",
        )
        metric_options = metric_groups.get(selected_category, [])
        if metric_filter.strip():
            query = metric_filter.strip().lower()
            metric_options = [column for column in metric_options if query in str(column).lower()]
        metric_options = sorted(metric_options, key=_sequence_validation_rank_metric_priority)
        if not metric_options:
            st.info("No ranking metrics match the current category/filter.")
            return
        preferred_metrics = [
            f"{selected_engine_label} {selected_engine[1]}_ipSAE_min × {selected_engine[1]}_rosetta_interface_dG_dSASA_ratio",
            f"{selected_engine_label} {selected_engine[1]}_ipsae_min × {selected_engine[1]}_rosetta_interface_dG_dSASA_ratio",
            f"{selected_engine_label} {selected_engine[1]}_pDockQ2_min × {selected_engine[1]}_rosetta_interface_dG_dSASA_ratio",
            f"{selected_engine[1]}_ipSAE_min",
            f"{selected_engine[1]}_ipsae_min",
            f"{selected_engine[1]}_pDockQ2_min",
            f"{selected_engine[1]}_pDockQ_min",
            f"{selected_engine[1]}_rosetta_interface_dG_dSASA_ratio",
            f"{selected_engine[1]}_rosetta_interface_sc",
            f"{selected_engine[1]}_target_aligned_binder_rmsd",
        ]
        default_metric = next((metric for metric in preferred_metrics if metric in metric_options), metric_options[0])
        rank_metric = st.selectbox(
            "Rank structures by",
            metric_options,
            index=metric_options.index(default_metric),
            format_func=_sequence_validation_metric_label,
            key=f"{run_dir.name}_sequence_validation_structure_rank_metric_v1",
        )
        inferred_direction = _sequence_validation_metric_direction(rank_metric)
        rank_direction = st.segmented_control(
            "Ranking direction",
            ["higher", "lower"],
            default=inferred_direction,
            key=f"{run_dir.name}_sequence_validation_structure_rank_direction_v1",
        ) or inferred_direction
        working_table = selector_table.copy()
        working_table[rank_metric] = pd.to_numeric(working_table[rank_metric], errors="coerce")
        working_table = working_table[working_table[rank_metric].notna()].copy()
        rank_metric_count = len(working_table)
        rmsd_filter_column = f"{selected_engine[1]}_target_aligned_binder_rmsd"
        if rmsd_filter_column not in working_table.columns and "target_aligned_binder_rmsd" in working_table.columns:
            rmsd_filter_column = "target_aligned_binder_rmsd"
        has_rmsd_filter = rmsd_filter_column in working_table.columns
        with st.expander("Sequence-quality filters before ranking", expanded=True):
            st.caption("These filters use sequence/design/pose properties; interface metrics remain the main ranking driver.")
            filter_cols = st.columns(5)
            with filter_cols[0]:
                use_max_run = st.checkbox("Max same-AA run", value=True, key=f"{run_dir.name}_sequence_validation_filter_max_run_use_v1")
                max_run_cutoff = st.number_input("≤ run length", min_value=1, max_value=20, value=4, step=1, key=f"{run_dir.name}_sequence_validation_filter_max_run_v1")
            with filter_cols[1]:
                use_alanine = st.checkbox("Alanine fraction", value=False, key=f"{run_dir.name}_sequence_validation_filter_ala_use_v1")
                alanine_cutoff = st.number_input("≤ alanine fraction", min_value=0.0, max_value=1.0, value=0.35, step=0.01, key=f"{run_dir.name}_sequence_validation_filter_ala_v1")
            with filter_cols[2]:
                use_low_complexity = st.checkbox("Low complexity", value=False, key=f"{run_dir.name}_sequence_validation_filter_low_complexity_use_v1")
                low_complexity_cutoff = st.number_input("≤ low complexity", min_value=0.0, max_value=1.0, value=0.35, step=0.01, key=f"{run_dir.name}_sequence_validation_filter_low_complexity_v1")
            with filter_cols[3]:
                use_binder_rmsd = st.checkbox(
                    "Binder pose RMSD",
                    value=has_rmsd_filter,
                    disabled=not has_rmsd_filter,
                    key=f"{run_dir.name}_sequence_validation_filter_binder_rmsd_use_v1",
                )
                binder_rmsd_cutoff = st.number_input(
                    "≤ binder RMSD (Å)",
                    min_value=0.0,
                    max_value=100.0,
                    value=3.5,
                    step=0.1,
                    disabled=not has_rmsd_filter,
                    key=f"{run_dir.name}_sequence_validation_filter_binder_rmsd_v1",
                )
            with filter_cols[4]:
                use_seq_rec = st.checkbox("Sequence recovery window", value=False, key=f"{run_dir.name}_sequence_validation_filter_seqrec_use_v1")
                seq_rec_min = st.number_input("seq_rec ≥", min_value=0.0, max_value=1.0, value=0.25, step=0.01, key=f"{run_dir.name}_sequence_validation_filter_seqrec_min_v1")
                seq_rec_max = st.number_input("seq_rec ≤", min_value=0.0, max_value=1.0, value=0.85, step=0.01, key=f"{run_dir.name}_sequence_validation_filter_seqrec_max_v1")
            extra_count = st.number_input(
                "Additional custom metric filters",
                min_value=0,
                max_value=8,
                value=0,
                step=1,
                key=f"{run_dir.name}_sequence_validation_custom_filter_count_v1",
            )
            numeric_columns = _sequence_validation_numeric_columns(working_table)
            for filter_index in range(int(extra_count)):
                row_cols = st.columns([4, 1, 2])
                with row_cols[0]:
                    custom_metric = st.selectbox(
                        f"Custom filter {filter_index + 1}",
                        numeric_columns,
                        key=f"{run_dir.name}_sequence_validation_custom_filter_metric_{filter_index}_v1",
                    )
                with row_cols[1]:
                    custom_operator = st.segmented_control(
                        "Operator",
                        ["<=", ">="],
                        default="<=",
                        key=f"{run_dir.name}_sequence_validation_custom_filter_operator_{filter_index}_v1",
                    ) or "<="
                with row_cols[2]:
                    custom_threshold = st.number_input(
                        "Threshold",
                        value=0.0,
                        step=0.1,
                        key=f"{run_dir.name}_sequence_validation_custom_filter_threshold_{filter_index}_v1",
                    )
                working_table = _sequence_validation_apply_threshold(working_table, custom_metric, custom_operator, custom_threshold)
        sequence_filter_input_count = len(working_table)
        if use_max_run and "max_same_aa_run" in working_table.columns:
            working_table = _sequence_validation_apply_threshold(working_table, "max_same_aa_run", "<=", max_run_cutoff)
        if use_alanine and "alanine_fraction" in working_table.columns:
            working_table = _sequence_validation_apply_threshold(working_table, "alanine_fraction", "<=", alanine_cutoff)
        if use_low_complexity and "low_complexity_fraction" in working_table.columns:
            working_table = _sequence_validation_apply_threshold(working_table, "low_complexity_fraction", "<=", low_complexity_cutoff)
        if use_binder_rmsd and has_rmsd_filter:
            working_table = _sequence_validation_apply_threshold(working_table, rmsd_filter_column, "<=", binder_rmsd_cutoff)
        if use_seq_rec and "design_seq_rec" in working_table.columns:
            working_table = _sequence_validation_apply_threshold(working_table, "design_seq_rec", ">=", seq_rec_min)
            working_table = _sequence_validation_apply_threshold(working_table, "design_seq_rec", "<=", seq_rec_max)
        sequence_filtered_count = len(working_table)
        sort_ascending = rank_direction == "lower"
        working_table = working_table.sort_values([rank_metric, id_column], ascending=[sort_ascending, True]).copy()
        working_table["feature_value"] = pd.to_numeric(working_table[rank_metric], errors="coerce")
        working_table["rank_score"] = -working_table["feature_value"] if rank_direction == "lower" else working_table["feature_value"]
        if "_parent_metrics" in working_table.columns:
            parent_values = []
            parent_statuses = []
            parent_components = []
            for _, parent_row in working_table.iterrows():
                parent_metrics = parent_row.get("_parent_metrics")
                if not isinstance(parent_metrics, dict):
                    parent_values.append(np.nan)
                    parent_statuses.append("missing_parent_metric")
                    parent_components.append("")
                    continue
                parent_value, parent_status, parent_component = _sequence_validation_feature_value_from_metrics(parent_metrics, rank_metric)
                parent_values.append(np.nan if parent_value is None else parent_value)
                parent_statuses.append(parent_status)
                parent_components.append(parent_component)
            working_table["parent_feature_value"] = parent_values
            working_table["parent_score_status"] = parent_statuses
            working_table["parent_score_components"] = parent_components
            working_table["parent_rank_score"] = (
                -pd.to_numeric(working_table["parent_feature_value"], errors="coerce")
                if rank_direction == "lower"
                else pd.to_numeric(working_table["parent_feature_value"], errors="coerce")
            )
            safe_parent_status = working_table["parent_score_status"].isin({"same_feature", "reconstructed_combo"})
            working_table["parent_rank_score"] = working_table["parent_rank_score"].where(safe_parent_status)
            working_table["rank_score_delta_vs_parent"] = (
                pd.to_numeric(working_table["rank_score"], errors="coerce")
                - pd.to_numeric(working_table["parent_rank_score"], errors="coerce")
            ).where(safe_parent_status)
        working_table["prethreshold_rank"] = range(1, len(working_table) + 1)
        threshold_cols = st.columns([1, 2, 1, 2])
        with threshold_cols[0]:
            use_manual_threshold = st.checkbox(
                "Use manual threshold",
                value=True,
                key=f"{run_dir.name}_sequence_validation_structure_manual_threshold_use_v2",
            )
        values_for_threshold = pd.to_numeric(working_table[rank_metric], errors="coerce").dropna() if not working_table.empty else pd.Series(dtype=float)
        default_threshold = float(values_for_threshold.iloc[min(1, len(values_for_threshold) - 1)]) if not values_for_threshold.empty else 0.0
        threshold_operator = ">=" if rank_direction == "higher" else "<="
        with threshold_cols[1]:
            manual_threshold = st.number_input(
                f"Manual threshold ({_combo_feature_term(rank_metric)} {threshold_operator})",
                value=default_threshold,
                step=0.01,
                format="%.6f",
                disabled=not use_manual_threshold,
                key=f"{run_dir.name}_sequence_validation_structure_manual_threshold_value_v2",
            )
        with threshold_cols[2]:
            show_only_passing = st.checkbox(
                "Only passing sequences",
                value=True,
                disabled=not use_manual_threshold,
                key=f"{run_dir.name}_sequence_validation_structure_manual_threshold_filter_v1",
            )
        passing_table = working_table.copy()
        threshold_input_count = len(working_table)
        if use_manual_threshold:
            passing_table = _sequence_validation_apply_threshold(working_table, rank_metric, threshold_operator, manual_threshold)
            if "parent_feature_value" in working_table.columns:
                parent_values_for_threshold = pd.to_numeric(working_table["parent_feature_value"], errors="coerce")
                cutoff = float(manual_threshold)
                if threshold_operator == "<=":
                    parent_passes = parent_values_for_threshold <= cutoff
                elif threshold_operator == ">=":
                    parent_passes = parent_values_for_threshold >= cutoff
                elif threshold_operator == "<":
                    parent_passes = parent_values_for_threshold < cutoff
                elif threshold_operator == ">":
                    parent_passes = parent_values_for_threshold > cutoff
                else:
                    parent_passes = parent_values_for_threshold == cutoff
                if "parent_score_status" in working_table.columns:
                    parent_passes = parent_passes & working_table["parent_score_status"].isin({"same_feature", "reconstructed_combo"})
                working_table["parent_passes_current_threshold"] = parent_passes.fillna(False)
                passing_table = passing_table.merge(
                    working_table[[id_column, "parent_passes_current_threshold"]],
                    on=id_column,
                    how="left",
                )
            with threshold_cols[3]:
                st.metric("Feature threshold", len(passing_table), delta=f"of {threshold_input_count} after sequence filters")
        elif "parent_feature_value" in working_table.columns:
            working_table["parent_passes_current_threshold"] = pd.NA
        selector_table_ranked = passing_table.copy() if use_manual_threshold and show_only_passing else working_table.copy()
        selector_table_ranked = selector_table_ranked.sort_values([rank_metric, id_column], ascending=[sort_ascending, True]).copy()
        selector_table_ranked["feature_value"] = pd.to_numeric(selector_table_ranked[rank_metric], errors="coerce")
        selector_table_ranked["rank_score"] = -selector_table_ranked["feature_value"] if rank_direction == "lower" else selector_table_ranked["feature_value"]
        selector_table_ranked["viewer_rank"] = range(1, len(selector_table_ranked) + 1)
        parent_column = "parent_id" if "parent_id" in working_table.columns else ""
        keep_cols = st.columns(3)
        with keep_cols[0]:
            keep_per_parent = st.number_input(
                "Keep top N per redesign parent",
                min_value=0,
                max_value=1000,
                value=2,
                step=1,
                help="0 keeps all passing sequences. Use this to avoid many variants from one parent backbone dominating the viewer.",
                key=f"{run_dir.name}_sequence_validation_keep_per_parent_v2",
            )
        if selector_table_ranked.empty:
            st.info("No structures remain after the selected sequence-quality/threshold filters.")
            return
        pre_topn_count = len(selector_table_ranked)
        working_table = selector_table_ranked.copy()
        if int(keep_per_parent) > 0 and parent_column:
            working_table = working_table.groupby(parent_column, dropna=False, group_keys=False).head(int(keep_per_parent)).copy()
            working_table = working_table.sort_values([rank_metric, id_column], ascending=[sort_ascending, True]).copy()
            working_table["viewer_rank"] = range(1, len(working_table) + 1)
        if working_table.empty:
            st.info("No structures remain after the selected top-N-per-parent filter.")
            return
        final_selectable_count = len(working_table)
        count_cols = st.columns(4)
        count_cols[0].metric("Numeric rank values", rank_metric_count)
        count_cols[1].metric("After sequence filters", sequence_filtered_count, delta=f"of {sequence_filter_input_count}")
        count_cols[2].metric(
            "After feature threshold",
            len(passing_table) if use_manual_threshold else threshold_input_count,
            delta=f"of {threshold_input_count}",
        )
        count_cols[3].metric("Selectable after top-N", final_selectable_count, delta=f"of {pre_topn_count}")
        st.caption(
            "Count order: numeric metric values -> sequence/custom filters -> optional feature threshold -> top-N per redesign parent. "
            "The scatter, dropdown, and table below use the final selectable set."
        )
        plot_table = working_table[[id_column, rank_metric, "feature_value", "rank_score", "viewer_rank"]].copy()
        plot_table["candidate_id"] = plot_table[id_column].astype(str)
        y_values = pd.to_numeric(plot_table["rank_score"], errors="coerce").dropna()
        y_domain = None
        if not y_values.empty:
            y_min = float(y_values.min())
            y_max = float(y_values.max())
            span = y_max - y_min
            padding = span * 0.08 if span > 0 else max(abs(y_max) * 0.08, 0.05)
            y_domain = [y_min - padding, y_max + padding]
        base_chart = (
            alt.Chart(plot_table)
            .mark_circle(size=70, opacity=0.9)
            .encode(
                x=alt.X("viewer_rank:Q", title="rank"),
                y=alt.Y(
                    "rank_score:Q",
                    title="rank score",
                    scale=alt.Scale(domain=y_domain, zero=False) if y_domain else alt.Undefined,
                ),
                tooltip=[
                    alt.Tooltip("viewer_rank:Q", title="rank", format=".0f"),
                    alt.Tooltip("candidate_id:N", title="candidate"),
                    alt.Tooltip("feature_value:Q", title=_combo_feature_term(rank_metric), format=".4g"),
                    alt.Tooltip("rank_score:Q", title="rank score", format=".4g"),
                ],
            )
            .properties(height=260)
        )
        st.altair_chart(base_chart.interactive(), width="stretch")
        metric_cols = [
            column
            for column in [
                "viewer_rank",
                id_column,
                "parent_id",
                rank_metric,
                "rank_score",
                "parent_rank_score",
                "rank_score_delta_vs_parent",
                "parent_score_status",
                "parent_passes_current_threshold",
                "parent_score_components",
                rmsd_filter_column,
                "binder_sequence",
                "parent_sequence",
                "sequence_length",
                "max_same_aa_run",
                "aaa_runs",
                "alanine_fraction",
                "low_complexity_fraction",
                "design_overall_confidence",
                "design_ligand_confidence",
                "design_seq_rec",
            ]
            if column in working_table.columns
        ]
        st.caption(
            f"Ranking `{rank_metric}` ({rank_direction} is better). "
            f"{final_selectable_count} selectable structures after top-N; "
            f"{sequence_filtered_count} after sequence filters; "
            f"{rank_metric_count} had a numeric ranking value."
        )
        st.dataframe(working_table[metric_cols].head(200), hide_index=True, width="stretch", height=260)
        with st.expander("Create Candidate Set From This Filtered Redesign Selection", expanded=False):
            st.caption(
                "Exports the final selectable sequence-redesign set shown above as a normal candidate set. "
                "The new set records this validation run, the selected engine/feature, thresholds, top-N filter, parent ids, "
                "parent scores, and parent sequences."
            )
            export_cols = st.columns([3, 1])
            default_export_name = (
                f"{run_dir.name}_{selected_engine_label}_{_combo_feature_term(rank_metric)}_"
                f"{final_selectable_count}_sequence_redesigns"
            )
            default_export_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", default_export_name).strip("_")
            with export_cols[0]:
                export_name = st.text_input(
                    "Candidate set name",
                    value=default_export_name,
                    key=f"{run_dir.name}_sequence_validation_export_name_v1",
                )
            with export_cols[1]:
                max_export = st.number_input(
                    "Max candidates",
                    min_value=0,
                    max_value=int(final_selectable_count),
                    value=int(final_selectable_count),
                    step=1,
                    help="0 exports all currently selectable redesigned sequences.",
                    key=f"{run_dir.name}_sequence_validation_export_max_v1",
                )
            export_filter_config = {
                "ranking_feature": rank_metric,
                "ranking_direction": rank_direction,
                "manual_threshold_enabled": bool(use_manual_threshold),
                "manual_threshold_operator": threshold_operator if use_manual_threshold else None,
                "manual_threshold": _jsonable_value(manual_threshold) if use_manual_threshold else None,
                "only_passing_sequences": bool(show_only_passing),
                "keep_per_parent": int(keep_per_parent),
                "sequence_filters": {
                    "max_same_aa_run": {"enabled": bool(use_max_run), "operator": "<=", "threshold": _jsonable_value(max_run_cutoff)},
                    "alanine_fraction": {"enabled": bool(use_alanine), "operator": "<=", "threshold": _jsonable_value(alanine_cutoff)},
                    "low_complexity_fraction": {"enabled": bool(use_low_complexity), "operator": "<=", "threshold": _jsonable_value(low_complexity_cutoff)},
                    "binder_pose_rmsd": {
                        "enabled": bool(use_binder_rmsd and has_rmsd_filter),
                        "feature": rmsd_filter_column if has_rmsd_filter else "",
                        "operator": "<=",
                        "threshold": _jsonable_value(binder_rmsd_cutoff),
                    },
                    "seq_rec": {
                        "enabled": bool(use_seq_rec),
                        "min": _jsonable_value(seq_rec_min),
                        "max": _jsonable_value(seq_rec_max),
                    },
                },
            }
            if st.button(
                "Create candidate set",
                type="primary",
                key=f"{run_dir.name}_sequence_validation_export_button_v1",
                disabled=working_table.empty,
            ):
                try:
                    export_run = _create_candidate_set_from_sequence_redesign_selection(
                        parent_run_dir=run_dir,
                        benchmark_dir=benchmark_dir,
                        merged=merged if merged is not None else pd.DataFrame(),
                        ranked_table=working_table,
                        selected_engine=selected_engine,
                        selected_feature=rank_metric,
                        direction=rank_direction,
                        filter_config=export_filter_config,
                        export_name=export_name.strip() or default_export_name,
                        max_candidates=int(max_export or 0),
                    )
                except Exception as exc:
                    st.error(f"Could not create candidate set: {exc}")
                else:
                    st.success(f"Created candidate set `{export_run.name}`.")
                    st.link_button(
                        "Open candidate set",
                        result_link("candidate-import", export_run.name),
                    )
        ids = [str(value) for value in working_table[id_column].dropna().tolist()]
        selected_id = st.selectbox(
            f"Ranked design ({len(ids)} structures)",
            ids,
            format_func=lambda value: (
                f"#{int(working_table[working_table[id_column].astype(str) == str(value)].iloc[0].get('viewer_rank'))} | "
                f"{value} | {rank_metric}={working_table[working_table[id_column].astype(str) == str(value)].iloc[0].get(rank_metric):.4g}"
                if not working_table[working_table[id_column].astype(str) == str(value)].empty
                else str(value)
            ),
            key=f"{run_dir.name}_sequence_validation_structure_design_v2",
        )
        design_key = _candidate_structure_id(selected_id)
        prediction_path = _preferred_structure_path(benchmark_dir, selected_engine[2], design_key)
        if prediction_path is None or not prediction_path.exists():
            structure_records = _benchmark_structure_records(run_dir, benchmark_dir, allow_input_fallback=True)
            if not structure_records.empty:
                matches = structure_records[
                    (structure_records["engine_key"].astype(str) == str(selected_engine[2]))
                    & (
                        (structure_records["binder_id"].astype(str) == str(design_key))
                        | (structure_records["candidate_id"].astype(str) == str(selected_id))
                        | (structure_records["candidate_id"].astype(str).map(_candidate_structure_id) == str(design_key))
                    )
                ]
                if not matches.empty:
                    linked_path = Path(str(matches.iloc[0].get("path") or ""))
                    if linked_path.exists():
                        prediction_path = linked_path
        if prediction_path is None:
            prediction_path = benchmark_dir / "predicted_metric_pdbs" / selected_engine[1] / f"{design_key}.pdb"
        if prediction_path is None or not prediction_path.exists():
            st.warning(f"No staged {selected_engine_label} prediction structure was found for `{selected_id}`.")
            return
        row = _sequence_validation_row_for_id(merged, selected_id)
        rank_row = _sequence_validation_row_for_id(ranking, selected_id) if ranking is not None and not ranking.empty else None
        parent_reference_path, parent_binder_chains, parent_target_chains, reference_source = _sequence_validation_parent_reference(
            run_dir,
            row,
            selected_id,
        )
        fallback_reference_path = _resolve_run_relative_path(run_dir, row.get("complex_pdb")) if row is not None and "complex_pdb" in row.index else None
        reference_path = parent_reference_path or fallback_reference_path
        prediction_target_chains = _parse_chain_list(row.get("target_chains")) if row is not None and "target_chains" in row.index else ["B"]
        prediction_binder_chains = _parse_chain_list(row.get("binder_chains")) if row is not None and "binder_chains" in row.index else ["A"]
        reference_target_chains = parent_target_chains or prediction_target_chains
        reference_binder_chains = parent_binder_chains or prediction_binder_chains
        show_reference = st.checkbox(
            "Show parent/source reference complex",
            value=True,
            key=f"{run_dir.name}_sequence_validation_structure_reference_v1",
            help="Uses the original selected complex when available. The generated sequence-design staging scaffold is not treated as the reference pose.",
        )
        align_prediction = st.checkbox("Align prediction on target", value=True, key=f"{run_dir.name}_sequence_validation_structure_align_v1")
        prediction_text = _read_structure_text(prediction_path)
        alignment_note = ""
        if show_reference and align_prediction and reference_path is not None and reference_path.exists() and reference_target_chains and prediction_target_chains:
            try:
                reference_structure = _parse_structure_file(reference_path)
                aligned_text, _target_rmsd, alignment_note = _aligned_structure_text(
                    prediction_path,
                    reference_structure,
                    reference_target_chains,
                    prediction_target_chains,
                )
                if aligned_text:
                    prediction_text = aligned_text
            except Exception as exc:
                alignment_note = f"alignment failed: {type(exc).__name__}: {exc}"
        structures = []
        reference_text = ""
        if show_reference and reference_path is not None and reference_path.exists():
            reference_text = _structure_to_pdb_text(_parse_structure_file(reference_path))
            structures.append(
                StructureVisualization(
                    pdb=reference_text,
                    color="chain-id",
                    representation_type="cartoon",
                    chains=[
                        *[
                            ChainVisualization(chain_id=chain, color="uniform", color_params={"value": "0x9ca3af"}, representation_type="cartoon")
                            for chain in reference_target_chains
                        ],
                        *[
                            ChainVisualization(chain_id=chain, color="uniform", color_params={"value": "0x9ca3af"}, representation_type="cartoon")
                            for chain in reference_binder_chains
                        ],
                    ],
                )
            )
        structures.append(
            StructureVisualization(
                pdb=prediction_text,
                color="chain-id",
                representation_type="cartoon",
                chains=[
                    *[
                        ChainVisualization(chain_id=chain, color="uniform", color_params={"value": "0x22c55e"}, representation_type="cartoon")
                        for chain in prediction_target_chains
                    ],
                    *[
                        ChainVisualization(chain_id=chain, color="uniform", color_params={"value": "0xdb2777"}, representation_type="cartoon")
                        for chain in prediction_binder_chains
                    ],
                ],
            )
        )
        selected_rank_matches = working_table[working_table[id_column].astype(str) == str(selected_id)]
        selected_rank_row = selected_rank_matches.iloc[0] if not selected_rank_matches.empty else None

        def _score_text(source: pd.Series | None, column: str) -> str:
            if source is None or column not in source.index:
                return "n/a"
            value = pd.to_numeric(pd.Series([source.get(column)]), errors="coerce").iloc[0]
            if pd.isna(value):
                return "n/a"
            return f"{float(value):.3f}"

        parent_score = _score_text(selected_rank_row, "parent_rank_score")
        delta_score = _score_text(selected_rank_row, "rank_score_delta_vs_parent")
        parent_status = (
            str(selected_rank_row.get("parent_score_status") or "missing_parent_metric")
            if selected_rank_row is not None
            else "missing_parent_metric"
        )
        delta_label = None if delta_score == "n/a" else f"vs parent {delta_score}"
        prefix = selected_engine[1]
        metric_cards: list[tuple[str, str, str | None]] = [
            ("Binder pose RMSD", _sequence_validation_metric_value(row, f"{prefix}_target_aligned_binder_rmsd", ".2f", " Å"), None),
            ("Rank score", _score_text(selected_rank_row, "rank_score"), delta_label),
            ("Parent rank score", parent_score, parent_status.replace("_", " ")),
        ]
        if delta_score != "n/a":
            metric_cards.append(("Score change", delta_score, "positive is better"))
        deduped_metric_cards: list[tuple[str, str, str | None]] = []
        seen_metric_labels: set[str] = set()
        for label, value, delta in metric_cards:
            if label in seen_metric_labels:
                continue
            seen_metric_labels.add(label)
            deduped_metric_cards.append((label, value, delta))
        metric_cols = st.columns(min(4, max(1, len(deduped_metric_cards))))
        for index, (label, value, delta) in enumerate(deduped_metric_cards):
            metric_cols[index % len(metric_cols)].metric(label, value, delta=delta)
        if parent_status not in {"same_feature", "reconstructed_combo"}:
            st.caption(
                "Parent score comparison is hidden because the parent/source candidate does not contain a safely matching "
                f"`{rank_metric}` metric."
            )
        molstar_custom_component(
            structures,
            key=f"{run_dir.name}_sequence_validation_molstar_{selected_id}_{selected_engine[1]}_{show_reference}_{align_prediction}",
            height=620,
            show_controls=True,
            selection_mode=False,
        )
        if alignment_note:
            st.caption(alignment_note)
        if show_reference:
            st.caption("Parent/source reference complex: gray | prediction target: green | prediction binder: magenta")
            st.caption(
                "Reference source: "
                f"{reference_source or 'validation input'} · {_path_label(reference_path, run_dir)}"
            )
            if fallback_reference_path is not None and parent_reference_path is not None:
                st.caption(
                    "Sequence-design staging scaffold, not used as reference: "
                    f"{_path_label(fallback_reference_path, run_dir)}"
                )
        else:
            st.caption("Prediction target: green | prediction binder: magenta")
        prediction_sidechains = _pdb_sidechain_summary(prediction_text, prediction_binder_chains)
        if reference_text:
            reference_sidechains = _pdb_sidechain_summary(reference_text, reference_binder_chains)
            st.caption(
                "Binder side-chain atoms: "
                f"reference {reference_sidechains['sidechain_atoms']} across {reference_sidechains['residues']} residues; "
                f"prediction {prediction_sidechains['sidechain_atoms']} across {prediction_sidechains['residues']} residues."
            )
        else:
            st.caption(
                "Prediction binder side-chain atoms: "
                f"{prediction_sidechains['sidechain_atoms']} across {prediction_sidechains['residues']} residues."
            )
        return

    st.subheader("Sequence Design Validation Artifacts")
    artifact_rows = [
        {"artifact": "merged metrics", "path": benchmark_dir / "merged_benchmark_metrics.csv"},
        {"artifact": "weighted ranking", "path": benchmark_dir / "sequence_design_weighted_ranking.csv"},
        {"artifact": "filter config", "path": benchmark_dir / "sequence_design_filter_config.json"},
        {"artifact": "pose RMSD metrics", "path": benchmark_dir / "target_aligned_binder_rmsd_metrics.csv"},
        {
            "artifact": "staging scaffold PDBs",
            "path": run_dir / "artifacts" / "queued_inputs" / "candidate_repo_dataset" / "input_pdbs",
        },
    ]
    st.dataframe(
        pd.DataFrame(
            [
                {"artifact": row["artifact"], "path": _path_label(row["path"], run_dir), "exists": row["path"].exists()}
                for row in artifact_rows
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    if filter_config:
        with st.expander("Filter configuration", expanded=True):
            st.json(filter_config)


def _monomer_prediction_path(run_dir: Path, candidate: dict[str, Any]) -> Path | None:
    for key in ["complex_pdb", "binder_pdb"]:
        path = _resolve_run_relative_path(run_dir, candidate.get(key))
        if path is not None:
            return path
    raw_metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
    prediction_dir = _resolve_run_relative_path(run_dir, raw_metadata.get("prediction_dir"))
    if prediction_dir is not None:
        for suffix in ("*.pdb", "*.cif", "*.mmcif"):
            matches = sorted(prediction_dir.glob(suffix))
            if matches:
                return matches[0]
    return None


def _monomer_parent_reference(
    run_dir: Path,
    candidate: dict[str, Any],
) -> tuple[Path | None, list[str], list[str], dict[str, Any]]:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    raw_metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
    source_candidate = raw_metadata.get("source_candidate") if isinstance(raw_metadata.get("source_candidate"), dict) else {}
    source_run_dir = _resolve_run_relative_path(run_dir, raw_metadata.get("upstream_source_run_dir"))
    if source_run_dir is None and raw_metadata.get("upstream_source_run_dir"):
        candidate_dir = Path(str(raw_metadata.get("upstream_source_run_dir"))).expanduser()
        if candidate_dir.exists():
            source_run_dir = candidate_dir
    source_run_dir = source_run_dir or run_dir
    reference_values = [
        metrics.get("monomer_refolding_reference"),
        raw_metadata.get("input_complex"),
        source_candidate.get("complex_pdb"),
        candidate.get("complex_pdb"),
    ]
    for value in reference_values:
        path = _resolve_run_relative_path(source_run_dir, value) or _resolve_run_relative_path(run_dir, value)
        if path is not None:
            return (
                path,
                _parse_chain_list(source_candidate.get("binder_chains")) or _parse_chain_list(candidate.get("binder_chains")) or ["A"],
                _parse_chain_list(source_candidate.get("target_chains")) or _parse_chain_list(candidate.get("target_chains")),
                source_candidate,
            )
    return None, [], [], source_candidate


def _monomer_result_rows(run_dir: Path, candidates: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
        raw_metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
        source_candidate = raw_metadata.get("source_candidate") if isinstance(raw_metadata.get("source_candidate"), dict) else {}
        source_metrics = source_candidate.get("metrics") if isinstance(source_candidate.get("metrics"), dict) else {}
        prediction_path = _monomer_prediction_path(run_dir, candidate)
        parent_path, parent_binder_chains, parent_target_chains, source_candidate = _monomer_parent_reference(run_dir, candidate)
        original_rank = metrics.get("result_selection_rank", source_metrics.get("result_selection_rank"))
        original_feature = (
            metrics.get("result_selection_feature")
            or source_metrics.get("result_selection_feature")
            or raw_metadata.get("selected_feature")
        )
        original_feature_value = metrics.get(
            "result_selection_feature_value",
            source_metrics.get("result_selection_feature_value"),
        )
        original_rank_score = metrics.get(
            "result_selection_rank_score",
            source_metrics.get("result_selection_rank_score"),
        )
        original_engine = (
            metrics.get("result_selection_engine")
            or source_metrics.get("result_selection_engine")
            or raw_metadata.get("selected_engine")
        )
        original_direction = (
            metrics.get("result_selection_direction")
            or source_metrics.get("result_selection_direction")
            or raw_metadata.get("direction")
        )
        rows.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "parent_id": (candidate.get("parents") or [source_candidate.get("candidate_id") or ""])[0],
                "source_tool": candidate.get("source_tool") or candidate.get("tool"),
                "binder_sequence": candidate.get("binder_sequence") or candidate.get("sequence") or source_candidate.get("binder_sequence"),
                "sequence_length": len(str(candidate.get("binder_sequence") or candidate.get("sequence") or source_candidate.get("binder_sequence") or "")),
                "monomer_rmsd": metrics.get("monomer_refolding_rmsd"),
                "monomer_rmsd_ca_count": metrics.get("monomer_refolding_rmsd_ca_count"),
                "plddt": metrics.get("plddt"),
                "ptm": metrics.get("ptm"),
                "iptm": metrics.get("iptm"),
                "plddt_mean": metrics.get("plddt_mean"),
                "esmfold2_monomer_ptm": metrics.get("esmfold2_monomer_ptm"),
                "esmfold2_monomer_iptm": metrics.get("esmfold2_monomer_iptm"),
                "esmfold2_monomer_plddt_mean": metrics.get("esmfold2_monomer_plddt_mean"),
                "boltz2_confidence_score": metrics.get("boltz2_confidence_score"),
                "boltz2_ptm": metrics.get("boltz2_ptm"),
                "boltz2_iptm": metrics.get("boltz2_iptm"),
                "original_selection_rank": original_rank,
                "original_selection_rank_score": original_rank_score,
                "original_selection_feature_value": original_feature_value,
                "original_selection_feature": original_feature,
                "original_selection_engine": original_engine,
                "original_selection_direction": original_direction,
                "original_parent_rank_score": metrics.get(
                    "result_selection_parent_rank_score",
                    source_metrics.get("result_selection_parent_rank_score"),
                ),
                "original_rank_score_delta_vs_parent": metrics.get(
                    "result_selection_rank_score_delta_vs_parent",
                    source_metrics.get("result_selection_rank_score_delta_vs_parent"),
                ),
                "original_parent_score_components": metrics.get(
                    "result_selection_parent_score_components",
                    source_metrics.get("result_selection_parent_score_components"),
                ),
                "prediction_path": str(prediction_path) if prediction_path is not None else "",
                "parent_path": str(parent_path) if parent_path is not None else "",
                "binder_chains": ",".join(_parse_chain_list(candidate.get("binder_chains")) or ["A"]),
                "parent_binder_chains": ",".join(parent_binder_chains),
                "parent_target_chains": ",".join(parent_target_chains),
                "_candidate": candidate,
            }
        )
    return pd.DataFrame(rows)


def _aligned_monomer_on_parent_binder(
    prediction_path: Path,
    parent_path: Path,
    parent_binder_chains: list[str],
    prediction_binder_chains: list[str],
) -> tuple[str | None, float | None, str, dict[str, str]]:
    from Bio.PDB import Superimposer

    try:
        parent_structure = _parse_structure_file(parent_path)
        prediction_structure = _parse_structure_file(prediction_path)
        prediction_chain_map = _pdb_safe_chain_id_map(prediction_structure)
        fixed_atoms = _target_ca_atoms(parent_structure, parent_binder_chains)
        moving_atoms = _target_ca_atoms(prediction_structure, prediction_binder_chains)
        if not moving_atoms and prediction_binder_chains != ["A"]:
            moving_atoms = _target_ca_atoms(prediction_structure, ["A"])
        atom_count = min(len(fixed_atoms), len(moving_atoms))
        if atom_count < 3:
            return None, None, f"alignment skipped: only {atom_count} paired binder CA atoms", prediction_chain_map
        superimposer = Superimposer()
        superimposer.set_atoms(fixed_atoms[:atom_count], moving_atoms[:atom_count])
        superimposer.apply(prediction_structure.get_atoms())
        return (
            _structure_to_pdb_text(prediction_structure),
            float(superimposer.rms),
            f"aligned on {atom_count} binder CA atoms",
            prediction_chain_map,
        )
    except Exception as exc:
        return None, None, f"alignment failed: {type(exc).__name__}: {exc}", {}


def _show_monomer_refolding_results(run_dir: Path, result: dict) -> None:
    st.header("Binder Monomer Refolding Results")
    input_json = read_json(run_dir / "input.json")
    params = input_json.get("params") if isinstance(input_json.get("params"), dict) else {}
    candidates = read_candidates(run_dir)
    table = _monomer_result_rows(run_dir, candidates)
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    success = bool(result.get("success"))
    cols = st.columns(5)
    cols[0].metric("Monomers", len(table))
    cols[1].metric("With structures", int(table["prediction_path"].astype(bool).sum()) if not table.empty else 0)
    cols[2].metric("With parent", int(table["parent_path"].astype(bool).sum()) if not table.empty else 0)
    cols[3].metric("Tool", str(input_json.get("tool") or "n/a"))
    cols[4].metric("GPU", str(params.get("gpu_device") or "n/a"))
    if metrics:
        st.caption(
            f"Job candidates: {metrics.get('candidate_count', len(table))}; artifacts: {metrics.get('artifact_count', 'n/a')}; "
            f"source set: {params.get('source_job_code') or params.get('source_run_id') or 'n/a'}."
        )
    if table.empty:
        if not success:
            st.error("Monomer refolding failed before normalized monomer candidates were written.")
            stderr_text = (run_dir / "stderr.log").read_text(errors="ignore") if (run_dir / "stderr.log").exists() else ""
            worker_stderr_text = (
                (run_dir / "worker_stderr.log").read_text(errors="ignore")
                if (run_dir / "worker_stderr.log").exists()
                else ""
            )
            if stderr_text.strip():
                with st.expander("Backend stderr", expanded=True):
                    st.code(stderr_text[-6000:], language="text")
            if worker_stderr_text.strip():
                with st.expander("Worker stderr", expanded=False):
                    st.code(worker_stderr_text[-6000:], language="text")
        else:
            st.info("No monomer-refolding candidates were written.")
        return

    numeric_columns = [
        column
        for column in [
            "original_selection_rank",
            "original_selection_rank_score",
            "original_selection_feature_value",
            "monomer_rmsd",
            "plddt",
            "plddt_mean",
            "ptm",
            "iptm",
            "esmfold2_monomer_plddt_mean",
            "esmfold2_monomer_ptm",
            "esmfold2_monomer_iptm",
            "boltz2_confidence_score",
            "boltz2_ptm",
            "boltz2_iptm",
            "sequence_length",
        ]
        if column in table.columns and pd.to_numeric(table[column], errors="coerce").notna().any()
    ]
    default_metric = (
        "original_selection_rank"
        if "original_selection_rank" in numeric_columns
        else "monomer_rmsd"
        if "monomer_rmsd" in numeric_columns
        else (numeric_columns[0] if numeric_columns else "")
    )
    rank_metric_labels = {
        "original_selection_rank": "original candidate rank",
        "original_selection_rank_score": "original candidate rank score",
        "original_selection_feature_value": "original selected feature value",
        "monomer_rmsd": "monomer RMSD",
        "boltz2_iptm": "boltz2_iptm (monomer confidence JSON; no target interface)",
        "iptm": "iptm (monomer/native confidence; no target interface)",
        "esmfold2_monomer_iptm": "esmfold2_monomer_iptm (monomer; no target interface)",
    }
    filter_cols = st.columns([2, 1, 1, 1])
    with filter_cols[0]:
        rank_metric = st.selectbox(
            "Ranking metric",
            numeric_columns,
            index=numeric_columns.index(default_metric) if default_metric in numeric_columns else 0,
            format_func=lambda column: rank_metric_labels.get(str(column), str(column)),
            key=f"{run_dir.name}_monomer_rank_metric_v2",
        )
    with filter_cols[1]:
        default_direction = "lower" if rank_metric in {"original_selection_rank", "monomer_rmsd", "sequence_length"} else "higher"
        direction = st.segmented_control(
            "Ranking direction",
            ["lower", "higher"],
            default=default_direction,
            key=f"{run_dir.name}_monomer_rank_direction_{rank_metric}_v2",
        ) or default_direction
    with filter_cols[2]:
        use_rmsd_filter = st.checkbox(
            "Filter by RMSD",
            value="monomer_rmsd" in table.columns,
            key=f"{run_dir.name}_monomer_use_rmsd_filter",
        )
    with filter_cols[3]:
        rmsd_threshold = st.number_input(
            "RMSD <= Å",
            min_value=0.0,
            value=3.5,
            step=0.1,
            disabled=not use_rmsd_filter,
            key=f"{run_dir.name}_monomer_rmsd_threshold",
        )

    working = table.copy()
    working["_rank_value"] = pd.to_numeric(working[rank_metric], errors="coerce")
    rankable_count = int(working["_rank_value"].notna().sum())
    working = working.dropna(subset=["_rank_value"])
    rmsd_passing_count = len(working)
    if use_rmsd_filter and "monomer_rmsd" in working.columns:
        rmsd_values = pd.to_numeric(working["monomer_rmsd"], errors="coerce")
        rmsd_passing_count = int((rmsd_values <= float(rmsd_threshold)).sum())
        working = working[rmsd_values <= float(rmsd_threshold)].copy()
    working = working.sort_values("_rank_value", ascending=(direction == "lower")).copy()
    working["rank"] = np.arange(1, len(working) + 1)
    pass_cols = st.columns(4)
    pass_cols[0].metric("Total monomers", len(table))
    pass_cols[1].metric("Rankable", rankable_count, delta=f"by {rank_metric_labels.get(str(rank_metric), str(rank_metric))}")
    pass_cols[2].metric(
        "Pass RMSD",
        rmsd_passing_count if use_rmsd_filter and "monomer_rmsd" in working.columns else "not filtered",
        delta=f"<= {float(rmsd_threshold):.2f} A" if use_rmsd_filter and "monomer_rmsd" in table.columns else None,
    )
    pass_cols[3].metric("Shown", len(working), delta=f"of {len(table)} total")
    if any(column in table.columns for column in ["boltz2_iptm", "iptm", "esmfold2_monomer_iptm"]):
        st.caption(
            "Monomer runs contain only the binder chain, so ipTM-style fields are not target-interface scores here. "
            "They may come from monomer backend confidence JSON or inherited source metrics; use original selection rank/score or RMSD for triage."
        )
    if "original_selection_rank" in table.columns and pd.to_numeric(table["original_selection_rank"], errors="coerce").notna().any():
        feature_text = ""
        if "original_selection_feature" in table.columns:
            features = [str(value) for value in table["original_selection_feature"].dropna().unique().tolist() if str(value)]
            if features:
                feature_text = f" Original ranking feature: `{features[0]}`."
        st.caption(f"Original candidate ordering is available from the source selection/export.{feature_text}")

    view = st.segmented_control(
        "Monomer result view",
        ["Ranked Monomers", "Structure Viewer", "Inputs and Outputs"],
        default="Ranked Monomers",
        key=f"{run_dir.name}_monomer_result_view",
    ) or "Ranked Monomers"

    if view == "Ranked Monomers":
        st.caption(f"{len(working)} monomers shown after the current ranking-value and RMSD filters.")
        if not working.empty:
            chart = (
                alt.Chart(working)
                .mark_circle(size=70, opacity=0.78)
                .encode(
                    x=alt.X("rank:Q", title="rank"),
                    y=alt.Y("_rank_value:Q", title=rank_metric),
                    tooltip=[
                        "rank:Q",
                        "candidate_id:N",
                        "parent_id:N",
                        alt.Tooltip("monomer_rmsd:Q", format=".3f"),
                        alt.Tooltip("plddt:Q", format=".3f"),
                        alt.Tooltip("esmfold2_monomer_plddt_mean:Q", format=".3f"),
                        alt.Tooltip("esmfold2_monomer_ptm:Q", format=".3f"),
                        alt.Tooltip("boltz2_confidence_score:Q", format=".3f"),
                        alt.Tooltip("original_selection_rank:Q", title="original rank"),
                        alt.Tooltip("original_selection_rank_score:Q", title="original rank score", format=".4g"),
                    ],
                )
                .properties(height=320, title=f"{rank_metric} by rank")
                .interactive()
            )
            st.altair_chart(chart, width="stretch", key=f"{run_dir.name}_monomer_rank_chart")
        visible = [
            "rank",
            "original_selection_rank",
            "original_selection_feature",
            "original_selection_feature_value",
            "original_selection_rank_score",
            "original_selection_engine",
            "candidate_id",
            "parent_id",
            "source_tool",
            "monomer_rmsd",
            "monomer_rmsd_ca_count",
            "plddt",
            "ptm",
            "iptm",
            "esmfold2_monomer_plddt_mean",
            "esmfold2_monomer_ptm",
            "esmfold2_monomer_iptm",
            "boltz2_confidence_score",
            "boltz2_ptm",
            "boltz2_iptm",
            "original_parent_rank_score",
            "original_rank_score_delta_vs_parent",
            "original_parent_score_components",
            "sequence_length",
            "binder_sequence",
            "prediction_path",
            "parent_path",
        ]
        st.dataframe(working[[column for column in visible if column in working.columns]], hide_index=True, width="stretch")

    elif view == "Structure Viewer":
        if working.empty:
            st.info("No monomer structures match the current filters.")
            return
        labels = [
            (
                f"#{int(row['rank'])} | original #{int(float(row['original_selection_rank']))} | "
                f"{row['candidate_id']} | RMSD {float(row['monomer_rmsd']):.2f} Å"
            )
            if pd.notna(row.get("monomer_rmsd")) and pd.notna(row.get("original_selection_rank"))
            else (
                f"#{int(row['rank'])} | {row['candidate_id']} | RMSD {float(row['monomer_rmsd']):.2f} Å"
                if pd.notna(row.get("monomer_rmsd"))
                else f"#{int(row['rank'])} | {row['candidate_id']}"
            )
            for _, row in working.iterrows()
        ]
        selected_label = st.selectbox(
            "Ranked monomer",
            labels,
            key=f"{run_dir.name}_monomer_structure_choice",
        )
        selected_row = working.iloc[labels.index(selected_label)]
        candidate = selected_row["_candidate"]
        prediction_path = Path(str(selected_row.get("prediction_path") or ""))
        parent_path = Path(str(selected_row.get("parent_path") or ""))
        def _selected_metric_text(column: str, digits: int = 3) -> str:
            value = selected_row.get(column)
            numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
            if pd.notna(numeric):
                return f"{float(numeric):.{digits}f}"
            if value is None or pd.isna(value):
                return "n/a"
            return str(value)

        selected_metric_cols = st.columns(6)
        selected_metric_cols[0].metric("View rank", int(selected_row["rank"]))
        selected_metric_cols[1].metric("Original rank", _selected_metric_text("original_selection_rank", digits=0))
        selected_rmsd_text = _selected_metric_text("monomer_rmsd")
        selected_metric_cols[2].metric("Monomer RMSD", f"{selected_rmsd_text} A" if selected_rmsd_text != "n/a" else "n/a")
        selected_metric_cols[3].metric("Boltz confidence", _selected_metric_text("boltz2_confidence_score"))
        selected_metric_cols[4].metric("Boltz pTM", _selected_metric_text("boltz2_ptm"))
        selected_metric_cols[5].metric("Original score", _selected_metric_text("original_selection_rank_score"))
        if selected_row.get("original_selection_feature"):
            st.caption(f"Original ranking feature: `{selected_row.get('original_selection_feature')}`.")
        if not prediction_path.exists():
            st.info("No readable monomer prediction structure is available for this candidate.")
            return
        parent_binder_chains = _parse_chain_list(selected_row.get("parent_binder_chains")) or ["A"]
        parent_target_chains = _parse_chain_list(selected_row.get("parent_target_chains"))
        prediction_binder_chains = _parse_chain_list(candidate.get("binder_chains")) or ["A"]
        overlay_parent = st.checkbox(
            "Overlay parent/source design",
            value=parent_path.exists(),
            disabled=not parent_path.exists(),
            key=f"{run_dir.name}_monomer_overlay_parent",
        )
        structures: list[StructureVisualization] = []
        alignment_rmsd = None
        alignment_note = ""
        prediction_chain_map: dict[str, str] = {}
        prediction_text = _read_structure_text(prediction_path)
        if overlay_parent and parent_path.exists():
            parent_structure = _parse_structure_file(parent_path)
            parent_chain_map = _pdb_safe_chain_id_map(parent_structure)
            structures.append(
                StructureVisualization(
                    pdb=_structure_to_pdb_text(parent_structure),
                    chains=[
                        *[
                            ChainVisualization(
                                chain_id=parent_chain_map.get(chain, chain),
                                color="uniform",
                                color_params={"value": "0x9ca3af"},
                                representation_type="cartoon",
                            )
                            for chain in parent_target_chains
                        ],
                        *[
                            ChainVisualization(
                                chain_id=parent_chain_map.get(chain, chain),
                                color="uniform",
                                color_params={"value": "0x2563eb"},
                                representation_type="cartoon",
                            )
                            for chain in parent_binder_chains
                        ],
                    ],
                )
            )
            aligned_text, alignment_rmsd, alignment_note, prediction_chain_map = _aligned_monomer_on_parent_binder(
                prediction_path,
                parent_path,
                parent_binder_chains,
                prediction_binder_chains,
            )
            if aligned_text:
                prediction_text = aligned_text
        structures.append(
            StructureVisualization(
                pdb=prediction_text,
                color="uniform",
                color_params={"value": "0xef4444"},
                representation_type="cartoon",
                chains=(
                    [
                        ChainVisualization(
                            chain_id=prediction_chain_map.get(chain, chain),
                            color="uniform",
                            color_params={"value": "0xef4444"},
                            representation_type="cartoon",
                        )
                        for chain in prediction_binder_chains
                    ]
                    if prediction_chain_map
                    else None
                ),
            )
        )
        molstar_custom_component(
            structures,
            key=f"{run_dir.name}_monomer_molstar_{selected_row['candidate_id']}_{bool(overlay_parent)}",
            height=620,
            show_controls=True,
            selection_mode=False,
        )
        metric_cols = st.columns(4)
        metric_cols[0].metric(
            "Stored monomer RMSD",
            f"{float(selected_row['monomer_rmsd']):.2f} Å" if pd.notna(selected_row.get("monomer_rmsd")) else "n/a",
        )
        metric_cols[1].metric(
            "Viewer alignment RMSD",
            f"{alignment_rmsd:.2f} Å" if alignment_rmsd is not None else "n/a",
        )
        metric_cols[2].metric("Parent", str(selected_row.get("parent_id") or "n/a"))
        metric_cols[3].metric("Rank score", f"{float(selected_row['_rank_value']):.3g}")
        if alignment_note:
            st.caption(alignment_note)
        st.caption("Parent target: gray | parent binder: blue | monomer fold: red")
        st.caption(f"Monomer structure: `{_path_label(prediction_path, run_dir)}`")
        if parent_path.exists():
            st.caption(f"Parent/source design: `{_path_label(parent_path, run_dir)}`")

    elif view == "Inputs and Outputs":
        st.subheader("Monomer Refolding Artifacts")
        st.dataframe(
            pd.DataFrame(
                [
                    {"artifact": "normalized candidates", "path": run_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl"},
                    {"artifact": "raw artifacts", "path": run_dir / "artifacts" / "raw"},
                ]
            ).assign(exists=lambda df: df["path"].map(lambda path: Path(path).exists())),
            hide_index=True,
            width="stretch",
        )


def _show_design_campaign_plot(df: pd.DataFrame, x: str, y: str, title: str, run_key: str) -> None:
    if x not in df.columns or y not in df.columns:
        return
    plot_df = df.dropna(subset=[x, y]).copy()
    if plot_df.empty:
        return
    chart = (
        alt.Chart(plot_df)
        .mark_circle(size=80, opacity=0.78)
        .encode(
            x=alt.X(f"{x}:Q", title=x),
            y=alt.Y(f"{y}:Q", title=y),
            color=alt.Color("engine label:N", title="engine"),
            tooltip=[
                alt.Tooltip("rank:N"),
                alt.Tooltip("engine label:N", title="engine"),
                alt.Tooltip("candidate:N"),
                alt.Tooltip(f"{x}:Q"),
                alt.Tooltip(f"{y}:Q"),
                alt.Tooltip("common pass:N"),
            ],
        )
        .properties(height=360, title=title)
        .interactive()
    )
    st.altair_chart(chart, width="stretch", key=f"{run_key}_{x}_{y}_campaign_plot")


def _show_design_campaign_scout_statistics(
    run_dir: Path,
    selector_df: pd.DataFrame,
    record_by_label: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    inputs: dict[str, Any],
    params: dict[str, Any],
) -> None:
    if selector_df.empty:
        return
    with st.expander("Scout Statistics", expanded=True):
        reference_path = _resolve_run_relative_path(run_dir, inputs.get("target_pdb"))
        fixed_target_chains = [str(chain) for chain in inputs.get("target_chains") or [] if str(chain)]
        configured_hotspots = _design_campaign_hotspot_residues(params.get("hotspots"))
        if not configured_hotspots:
            for candidate, _stage_record in record_by_label.values():
                configured_hotspots.update(_design_campaign_hotspot_residues(candidate.get("hotspots")))
        if reference_path is None or not reference_path.exists() or not fixed_target_chains:
            st.info("Target reference is missing, so geometry statistics cannot be computed.")
            return
        try:
            reference_structure = _parse_structure_file(reference_path)
        except Exception as exc:
            st.info(f"Could not read target reference for scout statistics: {exc}")
            return

        stats_cols = st.columns([1, 1, 2])
        max_stats = int(
            stats_cols[0].number_input(
                "Candidates to summarize",
                min_value=1,
                max_value=2000,
                value=min(300, len(selector_df)),
                step=25,
                key=f"{run_dir.name}_design_campaign_stats_limit_v1",
                help="Caps structure parsing for very large scout runs.",
            )
        )
        contact_cutoff = float(
            stats_cols[1].number_input(
                "Hotspot contact cutoff (A)",
                min_value=2.0,
                max_value=12.0,
                value=4.5,
                step=0.5,
                key=f"{run_dir.name}_design_campaign_stats_contact_cutoff_v1",
            )
        )
        stats_cols[2].caption(
            "Hotspot contacts use configured target hotspot side-chain heavy atoms against any binder heavy atom. "
            "Radius of gyration uses binder heavy atoms."
        )

        stats_rows: list[dict[str, Any]] = []
        hotspot_rows: list[dict[str, Any]] = []
        source_df = selector_df.sort_values(["engine", "_rank_sort", "candidate"], na_position="last").head(max_stats)
        for row in source_df.to_dict("records"):
            label = str(row.get("label") or "")
            candidate, stage_record = record_by_label.get(label, ({}, {}))
            stage_path = stage_record.get("path")
            if not isinstance(stage_path, Path) or not stage_path.exists():
                continue
            geometry_path = _design_campaign_secondary_structure_source_path(run_dir, candidate) or stage_path
            metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
            target_chains = _design_campaign_chain_ids(candidate, "input_target_chains", "target_chains")
            if not target_chains:
                target_chains = [str(chain) for chain in candidate.get("target_chains") or [] if str(chain)]
            moving_target_chains = _resolve_moving_target_chains_for_alignment(
                geometry_path,
                reference_structure,
                fixed_target_chains,
                target_chains,
            )
            binder_chains = _design_campaign_chain_ids(candidate, "input_binder_chains", "binder_chains")
            if not binder_chains:
                binder_chains = [str(chain) for chain in candidate.get("binder_chains") or [] if str(chain)]
            try:
                stat = geometry_path.stat()
                geometry = _cached_design_campaign_structure_geometry_stats(
                    str(geometry_path),
                    int(stat.st_mtime_ns),
                    int(stat.st_size),
                    tuple(binder_chains),
                    tuple(fixed_target_chains),
                    tuple(moving_target_chains),
                    tuple(sorted(configured_hotspots)),
                    float(contact_cutoff),
                )
            except Exception as exc:
                geometry = {"geometry_status": f"failed: {exc}"}
            metric_fraction = _design_campaign_number(metrics, "hotspot_atom_contact_fraction", "hotspot_ca_contact_fraction")
            computed_fraction = geometry.get("computed_hotspot_fraction")
            hotspot_fraction = metric_fraction if metric_fraction is not None else computed_fraction
            metric_min_distance = _design_campaign_number(metrics, "hotspot_atom_min_distance", "hotspot_ca_min_distance")
            contacted_text = (
                _design_campaign_hotspot_metric(metrics, "contacted")
                or metrics.get("hotspots_contacted")
                or geometry.get("computed_contacted_hotspots")
                or ""
            )
            contacted_residues = _design_campaign_hotspot_residues(contacted_text)
            metric_secondary_status = metrics.get("binder_secondary_structure_status")
            metric_helix_residues = metrics.get("binder_helix_residues")
            metric_sheet_residues = metrics.get("binder_sheet_residues")
            metric_coil_residues = metrics.get("binder_coil_residues")
            metric_structured_residues = metrics.get("binder_structured_residues")
            metric_ss_elements = metrics.get("binder_secondary_structure_elements")
            metric_helix_segments = metrics.get("binder_helix_segment_count")
            metric_sheet_segments = metrics.get("binder_sheet_segment_count")
            metric_coil_segments = metrics.get("binder_coil_segment_count")
            metric_total_segments = metrics.get("binder_secondary_structure_segment_count")
            ca_trace_helix_like_elements = metrics.get("binder_ca_trace_helix_like_elements")
            ca_trace_extended_elements = metrics.get("binder_ca_trace_extended_elements")
            ca_trace_elements = metrics.get("binder_ca_trace_elements")
            ca_trace_helix_like_fraction = metrics.get("binder_ca_trace_helix_like_fraction")
            ca_trace_extended_fraction = metrics.get("binder_ca_trace_extended_fraction")
            ca_trace_other_fraction = metrics.get("binder_ca_trace_other_fraction")
            stats_rows.append(
                {
                    "engine": row.get("engine"),
                    "candidate": row.get("candidate"),
                    "rank": row.get("rank"),
                    "stage": row.get("stage"),
                    "hotspot coverage": hotspot_fraction,
                    "hotspot contacts": row.get("hotspot contacts")
                    or (
                        f"{geometry.get('computed_hotspot_contacted_count')}/{geometry.get('computed_hotspot_count')}"
                        if geometry.get("computed_hotspot_count")
                        else ""
                    ),
                    "hotspot min distance": metric_min_distance
                    if metric_min_distance is not None
                    else geometry.get("computed_hotspot_min_distance"),
                    "contacted hotspots": ",".join(_design_campaign_hotspot_tokens(contacted_residues)),
                    "binder Rg": geometry.get("binder_radius_of_gyration"),
                    "binder residues": geometry.get("binder_residues"),
                    "binder heavy atoms": geometry.get("binder_heavy_atoms"),
                    "binder helix residues": metric_helix_residues
                    if metric_helix_residues is not None
                    else geometry.get("binder_helix_residues"),
                    "binder sheet residues": metric_sheet_residues
                    if metric_sheet_residues is not None
                    else geometry.get("binder_sheet_residues"),
                    "binder coil residues": metric_coil_residues
                    if metric_coil_residues is not None
                    else geometry.get("binder_coil_residues"),
                    "binder structured residues": metric_structured_residues
                    if metric_structured_residues is not None
                    else geometry.get("binder_structured_residues"),
                    "binder secondary structure elements": metric_ss_elements
                    if metric_ss_elements is not None
                    else geometry.get("binder_secondary_structure_elements"),
                    "binder helix elements": metric_helix_segments,
                    "binder sheet elements": metric_sheet_segments,
                    "binder coil elements": metric_coil_segments,
                    "binder secondary structure segment count": metric_total_segments,
                    "CA-trace helix-like elements": ca_trace_helix_like_elements,
                    "CA-trace extended elements": ca_trace_extended_elements,
                    "CA-trace structured elements": ca_trace_elements,
                    "CA-trace helix-like fraction": ca_trace_helix_like_fraction,
                    "CA-trace extended fraction": ca_trace_extended_fraction,
                    "CA-trace other fraction": ca_trace_other_fraction,
                    "CA-trace structure method": metrics.get("binder_ca_trace_structure_method") or "",
                    "geometry status": geometry.get("geometry_status"),
                    "secondary structure status": metric_secondary_status or geometry.get("secondary_structure_status"),
                    "secondary structure method": metrics.get("binder_secondary_structure_method") or "",
                    "secondary structure rebuilt backbone": metrics.get("binder_secondary_structure_reconstructed_backbone"),
                    "path": row.get("path"),
                }
            )
            for hotspot in configured_hotspots:
                hotspot_rows.append(
                    {
                        "engine": row.get("engine"),
                        "hotspot": _design_campaign_hotspot_tokens({hotspot})[0],
                        "contacted": 1 if hotspot in contacted_residues else 0,
                    }
                )

        if not stats_rows:
            st.info("No readable candidate structures were available for scout statistics.")
            return
        stats_df = pd.DataFrame(stats_rows)
        numeric_cols = [
            "hotspot coverage",
            "hotspot min distance",
            "binder Rg",
            "binder residues",
            "binder heavy atoms",
            "binder helix residues",
            "binder sheet residues",
            "binder coil residues",
            "binder structured residues",
            "binder secondary structure elements",
            "binder helix elements",
            "binder sheet elements",
            "binder coil elements",
            "binder secondary structure segment count",
            "CA-trace helix-like elements",
            "CA-trace extended elements",
            "CA-trace structured elements",
            "CA-trace helix-like fraction",
            "CA-trace extended fraction",
            "CA-trace other fraction",
        ]
        for column in numeric_cols:
            if column in stats_df:
                stats_df[column] = pd.to_numeric(stats_df[column], errors="coerce")
        if {"binder Rg", "binder residues"}.issubset(stats_df.columns):
            residue_counts = pd.to_numeric(stats_df["binder residues"], errors="coerce")
            normalized_rg = pd.to_numeric(stats_df["binder Rg"], errors="coerce") / residue_counts.where(residue_counts > 0)
            stats_df["binder Rg normalized by length"] = normalized_rg
        if {"binder Rg", "binder structured residues"}.issubset(stats_df.columns):
            structured_counts = pd.to_numeric(stats_df["binder structured residues"], errors="coerce")
            stats_df["binder Rg normalized by structured residues"] = (
                pd.to_numeric(stats_df["binder Rg"], errors="coerce") / structured_counts.where(structured_counts > 0)
            )
        if {"binder Rg", "binder secondary structure elements"}.issubset(stats_df.columns):
            element_counts = pd.to_numeric(stats_df["binder secondary structure elements"], errors="coerce")
            stats_df["binder Rg normalized by secondary structure elements"] = (
                pd.to_numeric(stats_df["binder Rg"], errors="coerce") / element_counts.where(element_counts > 0)
            )
        if {"binder Rg", "CA-trace structured elements"}.issubset(stats_df.columns):
            ca_trace_element_counts = pd.to_numeric(stats_df["CA-trace structured elements"], errors="coerce")
            stats_df["binder Rg normalized by CA-trace elements"] = (
                pd.to_numeric(stats_df["binder Rg"], errors="coerce") / ca_trace_element_counts.where(ca_trace_element_counts > 0)
            )
        ss_count_cols = ["binder helix residues", "binder sheet residues", "binder coil residues"]
        if all(column in stats_df.columns for column in ss_count_cols):
            ss_totals = stats_df[ss_count_cols].sum(axis=1, min_count=1)
            stats_df["binder helix fraction"] = stats_df["binder helix residues"] / ss_totals.where(ss_totals > 0)
            stats_df["binder sheet fraction"] = stats_df["binder sheet residues"] / ss_totals.where(ss_totals > 0)
            stats_df["binder coil fraction"] = stats_df["binder coil residues"] / ss_totals.where(ss_totals > 0)
        summary_df = (
            stats_df.groupby("engine", dropna=False)
            .agg(
                designs=("candidate", "count"),
                readable=("geometry status", lambda values: int(sum(str(value) == "ok" for value in values))),
                mean_hotspot_coverage=("hotspot coverage", "mean"),
                median_hotspot_coverage=("hotspot coverage", "median"),
                best_hotspot_coverage=("hotspot coverage", "max"),
                median_binder_rg=("binder Rg", "median"),
                median_hotspot_min_distance=("hotspot min distance", "median"),
                median_binder_residues=("binder residues", "median"),
                median_structured_residues=("binder structured residues", "median"),
                median_ss_elements=("binder secondary structure elements", "median"),
                median_helix_elements=("binder helix elements", "median"),
                median_sheet_elements=("binder sheet elements", "median"),
                median_ca_trace_elements=("CA-trace structured elements", "median"),
                mean_helix_fraction=("binder helix fraction", "mean"),
                mean_sheet_fraction=("binder sheet fraction", "mean"),
                mean_coil_fraction=("binder coil fraction", "mean"),
            )
            .reset_index()
        )
        if "readable" in summary_df and "designs" in summary_df:
            summary_df["readable %"] = summary_df["readable"] / summary_df["designs"]
        st.dataframe(
            summary_df,
            hide_index=True,
            width="stretch",
            column_config={
                "mean_hotspot_coverage": st.column_config.NumberColumn("mean hotspot coverage", format="%.2f"),
                "median_hotspot_coverage": st.column_config.NumberColumn("median hotspot coverage", format="%.2f"),
                "best_hotspot_coverage": st.column_config.NumberColumn("best hotspot coverage", format="%.2f"),
                "median_binder_rg": st.column_config.NumberColumn("median binder Rg", format="%.2f"),
                "median_hotspot_min_distance": st.column_config.NumberColumn("median hotspot distance", format="%.2f"),
                "median_binder_residues": st.column_config.NumberColumn("median binder residues", format="%.0f"),
                "median_structured_residues": st.column_config.NumberColumn("median structured residues", format="%.0f"),
                "median_ss_elements": st.column_config.NumberColumn("median SS elements", format="%.0f"),
                "median_helix_elements": st.column_config.NumberColumn("median helix elements", format="%.0f"),
                "median_sheet_elements": st.column_config.NumberColumn("median sheet elements", format="%.0f"),
                "median_ca_trace_elements": st.column_config.NumberColumn("median CA-trace elements", format="%.0f"),
                "mean_helix_fraction": st.column_config.NumberColumn("mean helix %", format="%.0%"),
                "mean_sheet_fraction": st.column_config.NumberColumn("mean sheet %", format="%.0%"),
                "mean_coil_fraction": st.column_config.NumberColumn("mean coil %", format="%.0%"),
                "readable %": st.column_config.NumberColumn("readable %", format="%.0%"),
            },
        )
        plot_tabs = st.tabs(["Distributions", "Hotspot Matrix", "Candidate Stats"])
        with plot_tabs[0]:
            plot_df = stats_df.dropna(subset=["hotspot coverage"]).copy()
            if not plot_df.empty:
                st.altair_chart(
                    alt.Chart(plot_df)
                    .mark_bar(opacity=0.82)
                    .encode(
                        x=alt.X(
                            "hotspot coverage:Q",
                            bin=alt.Bin(step=0.1, extent=[0, 1]),
                            title="hotspot coverage fraction",
                            scale=alt.Scale(domain=[0, 1]),
                        ),
                        y=alt.Y("count():Q", title="designs"),
                        color=alt.Color("engine:N", legend=None),
                        row=alt.Row("engine:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
                        tooltip=["engine:N", alt.Tooltip("count():Q", title="designs")],
                    )
                    .resolve_scale(y="independent")
                    .properties(height=90, width=900),
                    width="content",
                    key=f"{run_dir.name}_design_campaign_stats_hotspot_hist_by_engine",
                )
                st.altair_chart(
                    alt.Chart(plot_df)
                    .mark_boxplot(size=42)
                    .encode(
                        x=alt.X("engine:N", title="engine", sort="-y"),
                        y=alt.Y("hotspot coverage:Q", title="hotspot coverage fraction", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("engine:N", legend=None),
                        tooltip=["engine:N"],
                    )
                    .properties(height=320),
                    width="stretch",
                    key=f"{run_dir.name}_design_campaign_stats_hotspot_box",
                )
            ss_fraction_cols = ["binder helix fraction", "binder sheet fraction", "binder coil fraction"]
            if all(column in stats_df.columns for column in ss_fraction_cols):
                ss_plot_source = stats_df.dropna(subset=ss_fraction_cols, how="all").copy()
                if not ss_plot_source.empty:
                    ss_summary = (
                        ss_plot_source.groupby("engine", dropna=False)
                        .agg(
                            helix=("binder helix fraction", "mean"),
                            sheet=("binder sheet fraction", "mean"),
                            coil=("binder coil fraction", "mean"),
                            designs=("candidate", "count"),
                            rebuilt_backbones=(
                                "secondary structure rebuilt backbone",
                                lambda values: int(sum(bool(value) for value in values)),
                            ),
                        )
                        .reset_index()
                    )
                    ss_long = ss_summary.melt(
                        id_vars=["engine", "designs", "rebuilt_backbones"],
                        value_vars=["helix", "sheet", "coil"],
                        var_name="secondary structure",
                        value_name="fraction",
                    ).dropna(subset=["fraction"])
                    if not ss_long.empty:
                        st.altair_chart(
                            alt.Chart(ss_long)
                            .mark_bar(opacity=0.86)
                            .encode(
                                x=alt.X("engine:N", title="engine", sort="-y"),
                                y=alt.Y(
                                    "fraction:Q",
                                    title="mean binder secondary-structure fraction",
                                    stack="normalize",
                                    axis=alt.Axis(format="%"),
                                ),
                                color=alt.Color(
                                    "secondary structure:N",
                                    scale=alt.Scale(
                                        domain=["helix", "sheet", "coil"],
                                        range=["#2563eb", "#f59e0b", "#9ca3af"],
                                    ),
                                ),
                                tooltip=[
                                    "engine:N",
                                    "secondary structure:N",
                                    alt.Tooltip("fraction:Q", title="mean fraction", format=".1%"),
                                    "designs:Q",
                                    alt.Tooltip(
                                        "rebuilt_backbones:Q",
                                        title="CA-rebuilt backbones",
                                        format="d",
                                    ),
                                ],
                            )
                            .properties(height=280, width=900, title="Binder secondary-structure composition"),
                            width="content",
                            key=f"{run_dir.name}_design_campaign_stats_secondary_structure_bar",
                        )
            ss_element_cols = [
                "binder helix elements",
                "binder sheet elements",
                "binder secondary structure elements",
            ]
            if all(column in stats_df.columns for column in ss_element_cols):
                element_source = stats_df.copy()
                if "binder secondary structure elements" in element_source:
                    element_source["binder secondary structure elements"] = element_source[
                        "binder secondary structure elements"
                    ].fillna(
                        pd.to_numeric(element_source.get("binder helix elements"), errors="coerce").fillna(0)
                        + pd.to_numeric(element_source.get("binder sheet elements"), errors="coerce").fillna(0)
                    )
                element_summary = (
                    element_source.groupby("engine", dropna=False)
                    .agg(
                        helix=("binder helix elements", "mean"),
                        sheet=("binder sheet elements", "mean"),
                        total=("binder secondary structure elements", "mean"),
                        designs=("candidate", "count"),
                    )
                    .reset_index()
                )
                element_long = element_summary.melt(
                    id_vars=["engine", "designs"],
                    value_vars=["helix", "sheet", "total"],
                    var_name="element type",
                    value_name="mean count",
                ).dropna(subset=["mean count"])
                if not element_long.empty:
                    st.altair_chart(
                        alt.Chart(element_long)
                        .mark_bar(opacity=0.86)
                        .encode(
                            x=alt.X("engine:N", title="engine", sort="-y"),
                            xOffset=alt.XOffset("element type:N"),
                            y=alt.Y("mean count:Q", title="mean secondary-structure elements per binder"),
                            color=alt.Color(
                                "element type:N",
                                scale=alt.Scale(
                                    domain=["helix", "sheet", "total"],
                                    range=["#2563eb", "#f59e0b", "#111827"],
                                ),
                            ),
                            tooltip=[
                                "engine:N",
                                "element type:N",
                                alt.Tooltip("mean count:Q", format=".2f"),
                                "designs:Q",
                            ],
                        )
                        .properties(height=280, width=900, title="Binder secondary-structure element counts"),
                        width="content",
                        key=f"{run_dir.name}_design_campaign_stats_secondary_structure_elements_bar",
                    )
            ca_trace_element_cols = [
                "CA-trace helix-like elements",
                "CA-trace extended elements",
                "CA-trace structured elements",
            ]
            if all(column in stats_df.columns for column in ca_trace_element_cols):
                ca_trace_summary = (
                    stats_df.groupby("engine", dropna=False)
                    .agg(
                        helix_like=("CA-trace helix-like elements", "mean"),
                        extended=("CA-trace extended elements", "mean"),
                        total=("CA-trace structured elements", "mean"),
                        designs=("candidate", "count"),
                    )
                    .reset_index()
                )
                ca_trace_long = ca_trace_summary.melt(
                    id_vars=["engine", "designs"],
                    value_vars=["helix_like", "extended", "total"],
                    var_name="CA-trace element type",
                    value_name="mean count",
                ).dropna(subset=["mean count"])
                if not ca_trace_long.empty:
                    st.altair_chart(
                        alt.Chart(ca_trace_long)
                        .mark_bar(opacity=0.86)
                        .encode(
                            x=alt.X("engine:N", title="engine", sort="-y"),
                            xOffset=alt.XOffset("CA-trace element type:N"),
                            y=alt.Y("mean count:Q", title="mean CA-trace elements per binder"),
                            color=alt.Color(
                                "CA-trace element type:N",
                                scale=alt.Scale(
                                    domain=["helix_like", "extended", "total"],
                                    range=["#06b6d4", "#84cc16", "#111827"],
                                ),
                            ),
                            tooltip=[
                                "engine:N",
                                "CA-trace element type:N",
                                alt.Tooltip("mean count:Q", format=".2f"),
                                "designs:Q",
                            ],
                        )
                        .properties(height=280, width=900, title="Binder CA-trace geometry element counts"),
                        width="content",
                        key=f"{run_dir.name}_design_campaign_stats_ca_trace_elements_bar",
                    )
            rg_source_df = stats_df.dropna(subset=["hotspot coverage", "binder Rg"]).copy()
            if not rg_source_df.empty:
                st.altair_chart(
                    alt.Chart(rg_source_df)
                    .mark_bar(opacity=0.82)
                    .encode(
                        x=alt.X("binder Rg:Q", bin=alt.Bin(maxbins=20), title="binder radius of gyration"),
                        y=alt.Y("count():Q", title="designs"),
                        color=alt.Color("engine:N", legend=None),
                        row=alt.Row("engine:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
                        tooltip=["engine:N", alt.Tooltip("count():Q", title="designs")],
                    )
                    .resolve_scale(x="independent", y="independent")
                    .properties(height=90, width=900),
                    width="content",
                    key=f"{run_dir.name}_design_campaign_stats_rg_hist_by_engine",
                )
                rg_plot_mode = st.segmented_control(
                    "Rg / hotspot plot",
                    ["Scatter", "Median per engine", "Mean per engine"],
                    default="Scatter",
                    key=f"{run_dir.name}_design_campaign_stats_rg_plot_mode",
                ) or "Scatter"
                rg_axis_mode = st.segmented_control(
                    "Rg axis",
                    [
                        "Raw Rg",
                        "Rg normalized by length",
                        "Rg normalized by structured residues",
                        "Rg normalized by PyDSSP total elements",
                        "Rg normalized by CA-trace total elements",
                    ],
                    default="Raw Rg",
                    key=f"{run_dir.name}_design_campaign_stats_rg_axis_mode",
                ) or "Raw Rg"
                rg_axis_fields = {
                    "Raw Rg": ("binder Rg", "binder radius of gyration"),
                    "Rg normalized by length": (
                        "binder Rg normalized by length",
                        "binder radius of gyration / binder residues",
                    ),
                    "Rg normalized by structured residues": (
                        "binder Rg normalized by structured residues",
                        "binder radius of gyration / helix+sheet residues",
                    ),
                    "Rg normalized by PyDSSP total elements": (
                        "binder Rg normalized by secondary structure elements",
                        "binder radius of gyration / PyDSSP helix+sheet elements",
                    ),
                    "Rg normalized by CA-trace total elements": (
                        "binder Rg normalized by CA-trace elements",
                        "binder radius of gyration / CA-trace geometry elements",
                    ),
                }
                rg_x_field, rg_x_title = rg_axis_fields.get(rg_axis_mode, rg_axis_fields["Raw Rg"])
                axis_state_suffix = re.sub(r"[^a-z0-9]+", "_", f"{rg_plot_mode}_{rg_x_field}".lower()).strip("_")
                scatter_df = rg_source_df.dropna(subset=[rg_x_field]).copy()
                if scatter_df.empty:
                    st.info("No candidates have the selected Rg axis metric.")
                manual_axis_range = st.checkbox(
                    "Manual axis range",
                    value=True,
                    key=f"{run_dir.name}_design_campaign_stats_rg_manual_axis_v2",
                )
                x_scale = alt.Undefined
                y_scale = alt.Scale(domain=[0, 1])
                if manual_axis_range:
                    x_values = pd.to_numeric(scatter_df[rg_x_field], errors="coerce").dropna()
                    y_values = pd.to_numeric(scatter_df["hotspot coverage"], errors="coerce").dropna()
                    x_default_min = float(x_values.min()) if not x_values.empty else 0.0
                    x_default_max = float(x_values.max()) if not x_values.empty else 1.0
                    if x_default_min == x_default_max:
                        x_default_max = x_default_min + 1.0
                    y_default_min = max(0.0, float(y_values.min()) if not y_values.empty else 0.0)
                    y_default_max = min(1.0, float(y_values.max()) if not y_values.empty else 1.0)
                    if y_default_min == y_default_max:
                        y_default_min = 0.0
                        y_default_max = 1.0
                    axis_cols = st.columns(4)
                    x_min = axis_cols[0].number_input(
                        "X min",
                        value=x_default_min,
                        step=0.5,
                        format="%.3f",
                        key=f"{run_dir.name}_design_campaign_stats_rg_x_min_{axis_state_suffix}",
                    )
                    x_max = axis_cols[1].number_input(
                        "X max",
                        value=x_default_max,
                        step=0.5,
                        format="%.3f",
                        key=f"{run_dir.name}_design_campaign_stats_rg_x_max_{axis_state_suffix}",
                    )
                    y_min = axis_cols[2].number_input(
                        "Y min",
                        value=y_default_min,
                        step=0.05,
                        format="%.3f",
                        key=f"{run_dir.name}_design_campaign_stats_rg_y_min_{axis_state_suffix}",
                    )
                    y_max = axis_cols[3].number_input(
                        "Y max",
                        value=y_default_max,
                        step=0.05,
                        format="%.3f",
                        key=f"{run_dir.name}_design_campaign_stats_rg_y_max_{axis_state_suffix}",
                    )
                    if float(x_max) > float(x_min):
                        x_scale = alt.Scale(domain=[float(x_min), float(x_max)])
                    else:
                        st.warning("X max must be greater than X min; using automatic X range.")
                    if float(y_max) > float(y_min):
                        y_scale = alt.Scale(domain=[float(y_min), float(y_max)])
                    else:
                        st.warning("Y max must be greater than Y min; using 0-1 Y range.")
                if rg_plot_mode == "Scatter":
                    rg_chart = (
                        alt.Chart(scatter_df)
                        .mark_circle(size=70, opacity=0.75)
                        .encode(
                            x=alt.X(f"{rg_x_field}:Q", title=rg_x_title, scale=x_scale),
                            y=alt.Y(
                                "hotspot coverage:Q",
                                title="hotspot coverage fraction",
                                scale=y_scale,
                            ),
                            color=alt.Color("engine:N", title="engine"),
                            tooltip=[
                                "engine:N",
                                "candidate:N",
                                alt.Tooltip("hotspot coverage:Q", format=".2f"),
                                alt.Tooltip("binder Rg:Q", format=".2f"),
                                alt.Tooltip("binder Rg normalized by length:Q", title="Rg / length", format=".4f"),
                                alt.Tooltip(
                                    "binder Rg normalized by structured residues:Q",
                                    title="Rg / structured residues",
                                    format=".4f",
                                ),
                                alt.Tooltip(
                                    "binder Rg normalized by secondary structure elements:Q",
                                    title="Rg / PyDSSP elements",
                                    format=".4f",
                                ),
                                alt.Tooltip(
                                    "binder Rg normalized by CA-trace elements:Q",
                                    title="Rg / CA-trace elements",
                                    format=".4f",
                                ),
                                "binder residues:Q",
                                "binder structured residues:Q",
                                alt.Tooltip(
                                    "binder secondary structure elements:Q",
                                    title="PyDSSP total elements",
                                    format=".0f",
                                ),
                                alt.Tooltip(
                                    "CA-trace structured elements:Q",
                                    title="CA-trace total elements",
                                    format=".0f",
                                ),
                            ],
                        )
                    )
                else:
                    agg_name = "median" if rg_plot_mode == "Median per engine" else "mean"
                    grouped = scatter_df.groupby("engine", dropna=False)
                    if agg_name == "median":
                        agg_df = grouped.agg(
                            designs=("candidate", "count"),
                            hotspot_coverage=("hotspot coverage", "median"),
                            hotspot_low=("hotspot coverage", lambda values: values.quantile(0.25)),
                            hotspot_high=("hotspot coverage", lambda values: values.quantile(0.75)),
                            binder_rg=(rg_x_field, "median"),
                            binder_rg_low=(rg_x_field, lambda values: values.quantile(0.25)),
                            binder_rg_high=(rg_x_field, lambda values: values.quantile(0.75)),
                            raw_binder_rg=("binder Rg", "median"),
                            binder_residues=("binder residues", "median"),
                            structured_residues=("binder structured residues", "median"),
                            ss_elements=("binder secondary structure elements", "median"),
                            ca_trace_elements=("CA-trace structured elements", "median"),
                        ).reset_index()
                        spread_label = "IQR"
                    else:
                        agg_df = grouped.agg(
                            designs=("candidate", "count"),
                            hotspot_coverage=("hotspot coverage", "mean"),
                            hotspot_std=("hotspot coverage", "std"),
                            binder_rg=(rg_x_field, "mean"),
                            binder_rg_std=(rg_x_field, "std"),
                            raw_binder_rg=("binder Rg", "mean"),
                            binder_residues=("binder residues", "mean"),
                            structured_residues=("binder structured residues", "mean"),
                            ss_elements=("binder secondary structure elements", "mean"),
                            ca_trace_elements=("CA-trace structured elements", "mean"),
                        ).reset_index()
                        agg_df["hotspot_low"] = agg_df["hotspot_coverage"] - agg_df["hotspot_std"].fillna(0)
                        agg_df["hotspot_high"] = agg_df["hotspot_coverage"] + agg_df["hotspot_std"].fillna(0)
                        agg_df["binder_rg_low"] = agg_df["binder_rg"] - agg_df["binder_rg_std"].fillna(0)
                        agg_df["binder_rg_high"] = agg_df["binder_rg"] + agg_df["binder_rg_std"].fillna(0)
                        spread_label = "SD"
                    agg_df["hotspot_low"] = agg_df["hotspot_low"].clip(lower=0, upper=1)
                    agg_df["hotspot_high"] = agg_df["hotspot_high"].clip(lower=0, upper=1)
                    agg_df["binder_rg_low"] = agg_df["binder_rg_low"].clip(lower=0)
                    base = alt.Chart(agg_df)
                    x_axis = alt.X(
                        "binder_rg:Q",
                        title=f"{agg_name} {rg_x_title}",
                        scale=x_scale,
                    )
                    y_axis = alt.Y(
                        "hotspot_coverage:Q",
                        title=f"{agg_name} hotspot coverage fraction",
                        scale=y_scale,
                    )
                    error_color = alt.Color("engine:N", title="engine")
                    vertical_error = base.mark_rule(opacity=0.65).encode(
                        x=x_axis,
                        y=alt.Y("hotspot_low:Q"),
                        y2="hotspot_high:Q",
                        color=error_color,
                    )
                    horizontal_error = base.mark_rule(opacity=0.65).encode(
                        x=alt.X("binder_rg_low:Q"),
                        x2="binder_rg_high:Q",
                        y=y_axis,
                        color=error_color,
                    )
                    points = base.mark_circle(size=95, opacity=0.9, stroke="#111827", strokeWidth=0.8).encode(
                        x=x_axis,
                        y=y_axis,
                        color=error_color,
                        tooltip=[
                            "engine:N",
                            "designs:Q",
                            alt.Tooltip("hotspot_coverage:Q", title=f"{agg_name} hotspot coverage", format=".2f"),
                            alt.Tooltip("binder_rg:Q", title=f"{agg_name} selected Rg axis", format=".4f"),
                            alt.Tooltip("raw_binder_rg:Q", title=f"{agg_name} raw binder Rg", format=".2f"),
                            alt.Tooltip("hotspot_low:Q", title=f"hotspot low ({spread_label})", format=".2f"),
                            alt.Tooltip("hotspot_high:Q", title=f"hotspot high ({spread_label})", format=".2f"),
                            alt.Tooltip("binder_rg_low:Q", title=f"binder Rg low ({spread_label})", format=".2f"),
                            alt.Tooltip("binder_rg_high:Q", title=f"binder Rg high ({spread_label})", format=".2f"),
                            alt.Tooltip("binder_residues:Q", title=f"{agg_name} binder residues", format=".1f"),
                            alt.Tooltip(
                                "structured_residues:Q",
                                title=f"{agg_name} structured residues",
                                format=".1f",
                            ),
                            alt.Tooltip("ss_elements:Q", title=f"{agg_name} PyDSSP total elements", format=".1f"),
                            alt.Tooltip(
                                "ca_trace_elements:Q",
                                title=f"{agg_name} CA-trace total elements",
                                format=".1f",
                            ),
                        ],
                    )
                    rg_chart = vertical_error + horizontal_error + points
                st.altair_chart(
                    rg_chart.properties(height=340),
                    width="stretch",
                    key=f"{run_dir.name}_design_campaign_stats_rg_{axis_state_suffix}",
                )
        with plot_tabs[1]:
            if hotspot_rows:
                hotspot_df = pd.DataFrame(hotspot_rows)
                matrix_df = (
                    hotspot_df.groupby(["engine", "hotspot"], dropna=False)["contacted"].mean().reset_index()
                )
                st.altair_chart(
                    alt.Chart(matrix_df)
                    .mark_rect()
                    .encode(
                        x=alt.X("hotspot:N", title="hotspot"),
                        y=alt.Y("engine:N", title="engine"),
                        color=alt.Color(
                            "contacted:Q",
                            title="contact frequency",
                            scale=alt.Scale(domain=[0, 1], scheme="greens"),
                        ),
                        tooltip=["engine:N", "hotspot:N", alt.Tooltip("contacted:Q", format=".0%")],
                    )
                    .properties(height=max(220, 28 * max(1, matrix_df["engine"].nunique()))),
                    width="stretch",
                    key=f"{run_dir.name}_design_campaign_stats_hotspot_matrix",
                )
            else:
                st.info("No configured hotspots were available for a hotspot matrix.")
        with plot_tabs[2]:
            st.dataframe(
                stats_df,
                hide_index=True,
                width="stretch",
                column_config={
                    "hotspot coverage": st.column_config.NumberColumn("hotspot coverage", format="%.2f"),
                    "hotspot min distance": st.column_config.NumberColumn("hotspot min distance", format="%.2f"),
                    "binder Rg": st.column_config.NumberColumn("binder Rg", format="%.2f"),
                    "binder Rg normalized by length": st.column_config.NumberColumn("Rg / length", format="%.4f"),
                    "binder Rg normalized by structured residues": st.column_config.NumberColumn(
                        "Rg / structured residues",
                        format="%.4f",
                    ),
                    "binder Rg normalized by secondary structure elements": st.column_config.NumberColumn(
                        "Rg / SS elements",
                        format="%.4f",
                    ),
                },
            )


def _design_campaign_structure_selector_data(
    run_dir: Path,
    display_candidates: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    stage_mode: str,
) -> tuple[pd.DataFrame, dict[str, tuple[dict[str, Any], dict[str, Any]]]]:
    unique_rows = _design_campaign_unique_rows(rows)
    if not unique_rows:
        return pd.DataFrame(), {}
    selector_rows: list[dict[str, Any]] = []
    record_by_label: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for row in unique_rows:
        candidate_id = str(row.get("_candidate_id") or row.get("candidate") or "")
        matching_candidates = _design_campaign_candidates_by_id(display_candidates, candidate_id)
        if not matching_candidates:
            continue
        candidate = matching_candidates[-1]
        stage_records = _design_campaign_stage_structures(run_dir, matching_candidates, summary)
        if not stage_records:
            structure = _design_campaign_candidate_path(run_dir, candidate, "complex_pdb", "binder_pdb")
            stage_records = (
                [{"label": "Candidate structure", "path": structure, "kind": "candidate", "reference": None}]
                if structure is not None
                else []
            )
        if not stage_records:
            continue
        matching_stage_records = [
            record
            for record in stage_records
            if str(record.get("candidate_id") or "") in {"", candidate_id}
        ]
        candidate_stage_records = [
            record
            for record in stage_records
            if str(record.get("candidate_id") or "") == candidate_id
        ]
        stage_pool = candidate_stage_records or matching_stage_records or stage_records
        if stage_mode == "Generator or handoff":
            stage_record = next(
                (
                    record
                    for record in stage_pool
                    if str(record.get("kind") or "") in {"reconstructed", "generator", "screen", "candidate"}
                ),
                stage_pool[0],
            )
        elif stage_mode == "First available":
            stage_record = stage_pool[0]
        else:
            stage_record = stage_pool[-1]
        stage_path = stage_record.get("path")
        if not isinstance(stage_path, Path) or not stage_path.exists():
            continue
        engine_label = str(row.get("engine label") or candidate.get("source_tool") or "unknown")
        rank = row.get("rank") or "-"
        label = f"{rank} | {engine_label} | {candidate_id} | {stage_record.get('label')}"
        metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
        selector_rows.append(
            {
                "label": label,
                "engine": engine_label,
                "candidate": candidate_id,
                "stage": stage_record.get("label"),
                "rank": row.get("rank"),
                "hotspot contacts": _design_campaign_hotspot_contact_label(metrics),
                "target-aligned binder RMSD": metrics.get("target_aligned_binder_rmsd")
                or metrics.get("af2_target_aligned_binder_rmsd")
                or metrics.get("first_refolding_screen_screen_rmsd")
                or metrics.get("second_refolding_screen_screen_rmsd"),
                "path": _path_label(stage_path, run_dir),
            }
        )
        record_by_label[label] = (candidate, stage_record)
    if not selector_rows:
        return pd.DataFrame(), {}
    selector_df = pd.DataFrame(selector_rows)
    selector_df["_rank_sort"] = pd.to_numeric(selector_df["rank"], errors="coerce")
    selector_df = selector_df.sort_values(["engine", "_rank_sort", "candidate"], na_position="last")
    return selector_df, record_by_label


def _show_design_campaign_all_structure_viewer(
    run_dir: Path,
    display_candidates: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    inputs: dict[str, Any],
    params: dict[str, Any],
) -> None:
    st.subheader("All Candidate Structure Viewer")
    stage_mode = st.selectbox(
        "Structure stage",
        ["Best available / final", "Generator or handoff", "First available"],
        index=1,
        key=f"{run_dir.name}_design_campaign_all_stage_mode",
        help="Engine scout usually wants generator/handoff outputs; staged or evaluated runs may prefer the final available structure.",
    )
    selector_df, record_by_label = _design_campaign_structure_selector_data(
        run_dir,
        display_candidates,
        rows,
        summary,
        stage_mode=stage_mode,
    )
    if selector_df.empty:
        st.info("No readable structures are available for the campaign candidates.")
        return
    filter_cols = st.columns([1.2, 1, 1, 1.6])
    engine_options = sorted(selector_df["engine"].dropna().astype(str).unique())
    selected_engines = filter_cols[0].multiselect(
        "Engines",
        engine_options,
        default=engine_options,
        key=f"{run_dir.name}_design_campaign_all_structure_engines",
    )
    per_engine_limit = int(
        filter_cols[1].number_input(
            "Top per engine",
            min_value=1,
            max_value=50,
            value=1,
            step=1,
            key=f"{run_dir.name}_design_campaign_all_structure_per_engine_v2",
            help="Limits the default overlay candidate pool for large scout runs.",
        )
    )
    max_overlay = int(
        filter_cols[2].number_input(
            "Max overlay",
            min_value=1,
            max_value=100,
            value=min(12, max(1, len(engine_options))),
            step=1,
            key=f"{run_dir.name}_design_campaign_all_structure_max_overlay_v2",
        )
    )
    candidate_query = filter_cols[3].text_input(
        "Find candidate",
        value="",
        placeholder="engine, rank, or candidate id",
        key=f"{run_dir.name}_design_campaign_all_structure_search_v2",
    ).strip().lower()
    filtered_df = selector_df
    if selected_engines:
        filtered_df = filtered_df[filtered_df["engine"].astype(str).isin(selected_engines)]
    if candidate_query:
        query_blob = filtered_df[["label", "engine", "candidate", "stage"]].astype(str).agg(" ".join, axis=1).str.lower()
        filtered_df = filtered_df[query_blob.str.contains(re.escape(candidate_query), regex=True, na=False)]
        candidate_pool_df = filtered_df.head(max_overlay)
    else:
        candidate_pool_df = filtered_df.groupby("engine", sort=False, group_keys=False).head(per_engine_limit).head(max_overlay)
    st.caption(
        f"Showing {len(candidate_pool_df)} overlay candidates from {len(filtered_df)} matching structures. "
        "Increase limits or search when you want to inspect more."
    )
    selected_labels = st.multiselect(
        "Candidates for overlay",
        candidate_pool_df["label"].tolist(),
        default=candidate_pool_df["label"].tolist(),
        key=f"{run_dir.name}_design_campaign_all_structure_labels_v2",
    )
    shown_df = candidate_pool_df[candidate_pool_df["label"].isin(selected_labels)] if selected_labels else candidate_pool_df.iloc[0:0]
    st.dataframe(
        shown_df[
            [
                "engine",
                "candidate",
                "stage",
                "hotspot contacts",
                "target-aligned binder RMSD",
                "path",
            ]
        ],
        hide_index=True,
        width="stretch",
    )
    if shown_df.empty:
        st.info("No candidates match the current structure filters.")
        return

    reference_path = _resolve_run_relative_path(run_dir, inputs.get("target_pdb"))
    fixed_target_chains = [str(chain) for chain in inputs.get("target_chains") or [] if str(chain)]
    if reference_path is None or not reference_path.exists() or not fixed_target_chains:
        st.info("The campaign input target is missing, so the all-candidate overlay cannot be target-aligned.")
        return
    try:
        reference_structure = _parse_structure_file(reference_path)
        reference_text = _read_structure_text(reference_path)
    except Exception as exc:
        st.info(f"Could not read target reference for all-candidate alignment: {exc}")
        return

    colors = ["0x2563eb", "0xef4444", "0xd97706", "0x7c3aed", "0x059669", "0xdb2777", "0x0891b2", "0xbe123c"]
    reference_target_text = _structure_chains_to_pdb_text(reference_path, fixed_target_chains) or reference_text
    configured_hotspots = _design_campaign_hotspot_residues(params.get("hotspots"))
    if not configured_hotspots:
        for candidate in display_candidates:
            configured_hotspots.update(_design_campaign_hotspot_residues(candidate.get("hotspots")))
    contacted_hotspots: set[tuple[str, int]] = set()
    for label in shown_df["label"].tolist():
        candidate, _stage_record = record_by_label[label]
        metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
        contacted_hotspots.update(_design_campaign_hotspot_residues(_design_campaign_hotspot_metric(metrics, "contacted")))
        contacted_hotspots.update(_design_campaign_hotspot_residues(metrics.get("hotspots_contacted")))
    reference_chains = [
        ChainVisualization(
            chain_id=chain,
            color="uniform",
            color_params={"value": "0xd1d5db"},
            representation_type="cartoon",
        )
        for chain in fixed_target_chains
    ]
    reference_chains.extend(
        _design_campaign_hotspot_chains(
            configured_hotspots - contacted_hotspots,
            color="0xfacc15",
            label="Configured hotspot",
        )
    )
    reference_chains.extend(
        _design_campaign_hotspot_chains(
            contacted_hotspots,
            color="0x22c55e",
            label="Contacted hotspot",
        )
    )
    structures: list[StructureVisualization] = [
        StructureVisualization(
            pdb=reference_target_text,
            color="uniform",
            color_params={"value": "0xd1d5db"},
            representation_type="cartoon",
            chains=reference_chains,
        )
    ]
    overlay_rows: list[dict[str, Any]] = []
    for index, label in enumerate(shown_df["label"].tolist()):
        candidate, stage_record = record_by_label[label]
        stage_path = stage_record["path"]
        color = colors[index % len(colors)]
        target_chains = _design_campaign_chain_ids(candidate, "input_target_chains", "target_chains")
        if not target_chains:
            target_chains = [str(chain) for chain in candidate.get("target_chains") or [] if str(chain)]
        moving_target_chains = _resolve_moving_target_chains_for_alignment(
            stage_path,
            reference_structure,
            fixed_target_chains,
            target_chains,
        )
        aligned_text, target_rmsd, alignment_note = _aligned_structure_text(
            stage_path,
            reference_structure,
            fixed_target_chains,
            moving_target_chains,
        )
        if not aligned_text:
            overlay_rows.append(
                {
                    "candidate": candidate.get("candidate_id") or "",
                    "stage": stage_record.get("label") or "",
                    "alignment": alignment_note,
                    "target RMSD": None,
                    "shown": False,
                    "path": _path_label(stage_path, run_dir),
                }
            )
            continue
        try:
            chain_map = _pdb_safe_chain_id_map(_parse_structure_file(stage_path))
        except Exception:
            chain_map = {}
        binder_chains = _design_campaign_chain_ids(candidate, "input_binder_chains", "binder_chains")
        if not binder_chains:
            binder_chains = [str(chain) for chain in candidate.get("binder_chains") or [] if str(chain)]
        display_binder_chains = [chain_map.get(chain, chain) for chain in binder_chains]
        display_text = filter_pdb_text(aligned_text, keep_chains=set(display_binder_chains)) if display_binder_chains else aligned_text
        chain_visuals = None
        if display_binder_chains:
            chain_visuals = [
                *[
                    ChainVisualization(
                        chain_id=chain,
                        color="uniform",
                        color_params={"value": color},
                        representation_type="cartoon",
                    )
                    for chain in display_binder_chains
                ],
            ]
        structures.append(
            StructureVisualization(
                pdb=display_text,
                color="uniform",
                color_params={"value": color},
                representation_type="cartoon",
                chains=chain_visuals,
            )
        )
        overlay_rows.append(
            {
                "candidate": candidate.get("candidate_id") or "",
                "stage": stage_record.get("label") or "",
                "alignment": alignment_note,
                "target RMSD": target_rmsd,
                "shown": True,
                "path": _path_label(stage_path, run_dir),
            }
        )
    if overlay_rows:
        st.dataframe(pd.DataFrame(overlay_rows), hide_index=True, width="stretch")
    molstar_custom_component(
        structures,
        key=(
            f"{run_dir.name}_design_campaign_all_molstar_"
            f"{hashlib.sha1('|'.join(shown_df['label'].tolist()).encode()).hexdigest()[:12]}_"
            f"{hashlib.sha1(stage_mode.encode()).hexdigest()[:8]}"
        ),
        height=620,
        show_controls=True,
        selection_mode=False,
    )
    st.caption(
        f"All shown structures are target-aligned to `{_path_label(reference_path, run_dir)}` on chains "
        f"{','.join(fixed_target_chains)}. Target chains are gray; configured hotspots are yellow, contacted hotspots "
        "are green, and binder/generated chains use one color per candidate."
    )


def _design_campaign_workflow_elapsed_seconds(run_dir_value: object) -> tuple[float | None, pd.Timestamp | None, pd.Timestamp | None]:
    run_dir = Path(str(run_dir_value or "")).expanduser()
    if not run_dir.exists():
        return None, None, None
    metadata = read_json(run_dir / "metadata.json")
    created = pd.to_datetime(metadata.get("created_at"), errors="coerce")
    completed = pd.to_datetime(metadata.get("completed_at") or metadata.get("updated_at"), errors="coerce")
    if pd.isna(created):
        created = None
    if pd.isna(completed):
        completed = None
    elapsed = None
    if created is not None and completed is not None:
        elapsed = max(0.0, float((completed - created).total_seconds()))
    return elapsed, created, completed


def _design_campaign_workflow_funnel_df(workflows: list[dict[str, Any]]) -> pd.DataFrame:
    if not workflows:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for workflow in workflows:
        row = dict(workflow)
        engine = str(row.get("engine") or row.get("workflow") or "")
        row["engine"] = engine
        row["workflow"] = DESIGN_CAMPAIGN_ENGINE_LABELS.get(engine, str(row.get("label") or engine))
        elapsed, started, finished = _design_campaign_workflow_elapsed_seconds(row.get("run_dir"))
        row["_elapsed_seconds"] = elapsed
        row["_started_at"] = started
        row["_finished_at"] = finished
        rows.append(row)
    raw_df = pd.DataFrame(rows)
    if raw_df.empty:
        return raw_df
    for column in ["candidate_count", "native_passing_count", "common_passing_count", "selected_count"]:
        if column in raw_df:
            raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce").fillna(0)

    def collapsed_status(values: pd.Series) -> str:
        statuses = {str(value or "").strip().lower() for value in values if str(value or "").strip()}
        if not statuses:
            return ""
        if statuses == {"completed"}:
            return "completed"
        if "running" in statuses or "queued" in statuses:
            return "running/queued"
        if "failed" in statuses and len(statuses) == 1:
            return "failed"
        if "failed" in statuses:
            return "mixed failed"
        return ", ".join(sorted(statuses))

    def merged_text(values: pd.Series) -> str:
        items = [str(value).strip() for value in values if str(value).strip() and str(value).strip().lower() != "nan"]
        return ", ".join(sorted(set(items)))

    group_cols = ["engine", "workflow"]
    agg_spec: dict[str, Any] = {
        "child runs": ("run_id", "count") if "run_id" in raw_df else ("engine", "count"),
        "status": ("status", collapsed_status) if "status" in raw_df else ("engine", lambda _: ""),
        "sum child time": ("_elapsed_seconds", "sum"),
        "_started_at": ("_started_at", "min"),
        "_finished_at": ("_finished_at", "max"),
    }
    for column in ["candidate_count", "native_passing_count", "common_passing_count", "selected_count"]:
        if column in raw_df:
            agg_spec[column] = (column, "sum")
    for column in ["selection_mode", "candidate_pool_coverage", "error"]:
        if column in raw_df:
            agg_spec[column] = (column, merged_text)
    collapsed = raw_df.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()
    collapsed["wall time"] = [
        format_duration(float((finish - start).total_seconds()))
        if start is not None and finish is not None and not pd.isna(start) and not pd.isna(finish)
        else "n/a"
        for start, finish in zip(collapsed["_started_at"], collapsed["_finished_at"], strict=False)
    ]
    collapsed["sum child time"] = collapsed["sum child time"].map(lambda value: format_duration(float(value)) if pd.notna(value) else "n/a")
    collapsed["avg child time"] = [
        format_duration(float(seconds) / float(child_runs))
        if pd.notna(seconds) and child_runs
        else "n/a"
        for seconds, child_runs in zip(raw_df.groupby(group_cols, dropna=False)["_elapsed_seconds"].sum().to_numpy(), collapsed["child runs"], strict=False)
    ]
    return collapsed.drop(columns=["_started_at", "_finished_at"], errors="ignore")


def _design_campaign_result_views(workflow_recipe: str) -> tuple[list[str], dict[str, str], str]:
    workflow_recipe = str(workflow_recipe or "vanilla")
    if workflow_recipe == "engine_scout":
        labels = [
            "Scout Overview",
            "Scout Candidates",
            "Scout Plots",
            "Scout Structure Viewer",
            "Inputs and Outputs",
        ]
        return labels, {
            "Scout Overview": "overview",
            "Scout Candidates": "candidates",
            "Scout Plots": "plots",
            "Scout Structure Viewer": "structure",
            "Inputs and Outputs": "io",
        }, "Scout Overview"
    if workflow_recipe == "staged":
        labels = [
            "Staged Overview",
            "Staged Candidates",
            "Staged Plots",
            "Staged Structure Viewer",
            "Inputs and Outputs",
        ]
        return labels, {
            "Staged Overview": "overview",
            "Staged Candidates": "candidates",
            "Staged Plots": "plots",
            "Staged Structure Viewer": "structure",
            "Inputs and Outputs": "io",
        }, "Staged Overview"
    labels = ["Overview", "Candidates", "Plots", "Structure Viewer", "Inputs and Outputs"]
    return labels, {
        "Overview": "overview",
        "Candidates": "candidates",
        "Plots": "plots",
        "Structure Viewer": "structure",
        "Inputs and Outputs": "io",
    }, "Overview"


def _show_design_campaign_results(run_dir: Path, result: dict) -> None:
    aggregate = bool(result.get("_aggregate_design_campaign"))
    st.header("Design Campaign Results" if not aggregate else "Combined Design Campaign Results")
    input_json = (
        result.get("_aggregate_input")
        if isinstance(result.get("_aggregate_input"), dict)
        else read_json(run_dir / "input.json")
    )
    params = input_json.get("params") if isinstance(input_json.get("params"), dict) else {}
    inputs = input_json.get("inputs") if isinstance(input_json.get("inputs"), dict) else {}
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    outputs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
    candidates = (
        outputs.get("candidates")
        if aggregate and isinstance(outputs.get("candidates"), list)
        else read_candidates(run_dir)
    )
    display_candidates = _display_design_campaign_candidates(candidates, params)
    summary = (
        outputs.get("workflow_summary")
        if aggregate and isinstance(outputs.get("workflow_summary"), dict)
        else read_json(run_dir / "artifacts" / "design_campaign" / "workflow_summary.json")
    )
    workflows = summary.get("workflows") if isinstance(summary.get("workflows"), list) else []
    workflow_funnel_df = _design_campaign_workflow_funnel_df(workflows)
    common_summary = summary.get("common_validation") if isinstance(summary.get("common_validation"), dict) else {}
    refinement_summary = summary.get("sequence_refinement") if isinstance(summary.get("sequence_refinement"), dict) else {}
    evaluation_summary = summary.get("evaluation") if isinstance(summary.get("evaluation"), dict) else {}

    cols = st.columns(6)
    cols[0].metric("Final candidates", len(display_candidates))
    harmonized_count = metrics.get("harmonized_candidate_count", summary.get("harmonized_candidate_count", ""))
    if len(display_candidates) != len(candidates):
        harmonized_count = len(display_candidates)
    cols[1].metric("Harmonized", harmonized_count)
    cols[2].metric("Engines", metrics.get("engine_count", len(params.get("engines") or [])))
    completed_engines = (
        int((workflow_funnel_df["status"].astype(str) == "completed").sum())
        if not workflow_funnel_df.empty and "status" in workflow_funnel_df
        else metrics.get("completed_engine_count", "")
    )
    cols[3].metric("Completed", completed_engines)
    cols[4].metric("Common passes", common_summary.get("common_bindcraft_passing_count", "n/a"))
    cols[5].metric("Evaluation", evaluation_summary.get("status") or "n/a")
    target_text = ", ".join(str(chain) for chain in inputs.get("target_chains") or [])
    workflow_recipe = str(summary.get("workflow_recipe") or params.get("workflow_recipe") or "vanilla")
    st.caption(
        f"{params.get('campaign_name') or run_dir.name} | target chains: {target_text or 'n/a'} | "
        f"workflow: {workflow_recipe} | GPU: {params.get('gpu_device', 'n/a')}"
    )

    rows = _design_campaign_candidate_rows(display_candidates)
    display_rows = _design_campaign_unique_rows(rows)
    candidate_df = pd.DataFrame(display_rows)
    view_options, view_kind_by_label, default_view = _design_campaign_result_views(workflow_recipe)
    selected_view = st.segmented_control(
        "Campaign result view",
        view_options,
        default=default_view,
        key=f"{run_dir.name}_{workflow_recipe}_design_campaign_result_view",
    ) or default_view
    view_kind = view_kind_by_label.get(str(selected_view), "overview")

    if view_kind == "overview":
        if not workflow_funnel_df.empty:
            workflow_df = workflow_funnel_df.copy()
            preferred = [
                "workflow",
                "child runs",
                "status",
                "candidate_count",
                "native_passing_count",
                "common_passing_count",
                "selected_count",
                "selection_mode",
                "candidate_pool_coverage",
                "wall time",
                "sum child time",
                "avg child time",
                "error",
            ]
            st.subheader("Workflow Funnel")
            st.dataframe(workflow_df[[col for col in preferred if col in workflow_df.columns]], hide_index=True, width="stretch")
            count_cols = [col for col in ["candidate_count", "common_passing_count", "selected_count"] if col in workflow_df]
            if count_cols and "workflow" in workflow_df:
                chart_df = workflow_df[["workflow", *count_cols]].melt(
                    id_vars=["workflow"], var_name="count type", value_name="count"
                )
                chart_df["count"] = pd.to_numeric(chart_df["count"], errors="coerce")
                chart_df = chart_df.dropna(subset=["count"])
                if not chart_df.empty:
                    st.altair_chart(
                        alt.Chart(chart_df)
                        .mark_bar()
                        .encode(
                            x=alt.X("workflow:N", sort=None, title="workflow"),
                            y=alt.Y("count:Q", title="candidates"),
                            color=alt.Color("count type:N", title="count"),
                            xOffset="count type:N",
                            tooltip=["workflow:N", "count type:N", "count:Q"],
                        )
                        .properties(height=320),
                        width="stretch",
                        key=f"{run_dir.name}_design_campaign_funnel",
                    )
        summary_rows = []
        if common_summary:
            pool_counts = common_summary.get("candidate_pool_level_counts")
            pool_text = (
                ", ".join(f"{key}: {value}" for key, value in sorted(pool_counts.items()))
                if isinstance(pool_counts, dict)
                else ""
            )
            summary_rows.append(
                {
                    "stage": "Common validation",
                    "mode/status": common_summary.get("profile", "bindcraft_default_target_template"),
                    "input": common_summary.get("native_candidate_count"),
                    "output": common_summary.get("common_bindcraft_passing_count"),
                    "details": pool_text,
                }
            )
        if refinement_summary:
            summary_rows.append(
                {
                    "stage": "Sequence refinement",
                    "mode/status": f"{refinement_summary.get('mode')} / {refinement_summary.get('status')}",
                    "input": refinement_summary.get("input_candidate_count"),
                    "output": refinement_summary.get("refined_candidate_count"),
                    "details": f"{refinement_summary.get('engine', '')}; skipped {refinement_summary.get('skipped_candidate_count', 0)}",
                }
            )
        if evaluation_summary:
            summary_rows.append(
                {
                    "stage": "Evaluation",
                    "mode/status": f"{evaluation_summary.get('mode')} / {evaluation_summary.get('status')}",
                    "input": evaluation_summary.get("candidate_count"),
                    "output": None,
                    "details": evaluation_summary.get("refolder") or "existing structures",
                }
            )
        if summary_rows:
            st.subheader("Shared Stages")
            st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")
        attempt_summaries = (
            refinement_summary.get("attempt_summaries")
            if isinstance(refinement_summary.get("attempt_summaries"), list)
            else []
        )
        if attempt_summaries:
            start_rows: list[dict[str, object]] = []
            for attempt in attempt_summaries:
                if not isinstance(attempt, dict):
                    continue
                first_screen = attempt.get("first_screen") if isinstance(attempt.get("first_screen"), dict) else {}
                second_screen = attempt.get("second_screen") if isinstance(attempt.get("second_screen"), dict) else {}
                start_rows.append(
                    {
                        "start": attempt.get("generator_start_index"),
                        "engine": attempt.get("engine"),
                        "source_candidate": attempt.get("source_candidate_id"),
                        "status": attempt.get("status"),
                        "output_candidates": attempt.get("output_candidate_count"),
                        "first_screen": first_screen.get("status") or attempt.get("result"),
                        "first_screen_job": first_screen.get("child_run_id"),
                        "second_screen": second_screen.get("status"),
                        "second_screen_job": second_screen.get("child_run_id"),
                        "reason": attempt.get("reason") or attempt.get("error"),
                    }
                )
            if start_rows:
                st.subheader("Generator Starts")
                st.dataframe(pd.DataFrame(start_rows), hide_index=True, width="stretch")

    elif view_kind == "candidates":
        st.subheader("Candidate Table")
        if candidate_df.empty:
            st.info("No final candidates were written for this campaign.")
        else:
            engines = sorted(candidate_df["engine label"].dropna().astype(str).unique())
            selected_engines = st.multiselect(
                "Engines",
                engines,
                default=engines,
                key=f"{run_dir.name}_design_campaign_candidate_engines",
            )
            shown = candidate_df[candidate_df["engine label"].isin(selected_engines)] if selected_engines else candidate_df
            if "campaign" in candidate_df and candidate_df["campaign"].astype(str).str.len().any():
                campaigns = sorted(candidate_df["campaign"].dropna().astype(str).unique())
                selected_campaigns = st.multiselect(
                    "Campaigns",
                    campaigns,
                    default=campaigns,
                    key=f"{run_dir.name}_design_campaign_candidate_campaigns",
                )
                if selected_campaigns:
                    shown = shown[shown["campaign"].isin(selected_campaigns)]
            pass_filter = st.segmented_control(
                "Common filter",
                ["All", "Passing", "Failing or unscored"],
                default="All",
                key=f"{run_dir.name}_design_campaign_pass_filter",
            )
            if pass_filter == "Passing":
                shown = shown[shown["common pass"] == True]  # noqa: E712
            elif pass_filter == "Failing or unscored":
                shown = shown[shown["common pass"] != True]  # noqa: E712
            visible = [
                "rank",
                "campaign",
                "engine label",
                "candidate",
                "stage",
                "selection",
                "complex AF2",
                "binder-fold AF2",
                "common pass",
                "ipSAE",
                "iPAE",
                "ipTM",
                "binder pLDDT",
                "Rosetta dG",
                "interface sc",
                "hotspot contacts",
                "hotspot fraction",
                "hotspot min distance",
                "target-aligned binder RMSD",
                "binder fold RMSD",
                "complex pose RMSD",
                "binder length",
                "failures",
            ]
            st.dataframe(shown[visible], hide_index=True, width="stretch")
            st.download_button(
                "Download candidate table CSV",
                shown[visible].to_csv(index=False).encode("utf-8"),
                file_name=f"{run_dir.name}_design_campaign_candidates.csv",
                mime="text/csv",
                key=f"{run_dir.name}_design_campaign_candidates_csv",
            )

    elif view_kind == "plots":
        if candidate_df.empty:
            st.info("No final candidates are available to plot.")
        else:
            if workflow_recipe == "engine_scout":
                scout_selector_df, scout_record_by_label = _design_campaign_structure_selector_data(
                    run_dir,
                    display_candidates,
                    rows,
                    summary,
                    stage_mode="Generator or handoff",
                )
                if scout_selector_df.empty:
                    st.info("No readable generator or handoff structures are available for scout statistics.")
                else:
                    _show_design_campaign_scout_statistics(
                        run_dir,
                        scout_selector_df,
                        scout_record_by_label,
                        inputs,
                        params,
                    )
            plot_cols = st.columns(2)
            with plot_cols[0]:
                _show_design_campaign_plot(candidate_df, "iPAE", "ipSAE", "ipSAE vs iPAE", run_dir.name)
                _show_design_campaign_plot(candidate_df, "ipTM", "binder pLDDT", "ipTM vs binder pLDDT", run_dir.name)
            with plot_cols[1]:
                _show_design_campaign_plot(candidate_df, "Rosetta dG", "ipSAE", "Rosetta dG vs ipSAE", run_dir.name)
                if workflow_recipe != "engine_scout" and "engine label" in candidate_df and "common pass" in candidate_df:
                    pass_df = (
                        candidate_df.assign(_pass=candidate_df["common pass"] == True)  # noqa: E712
                        .groupby("engine label", as_index=False)
                        .agg(candidates=("candidate", "count"), passing=("_pass", "sum"))
                    )
                    if not pass_df.empty:
                        pass_df["pass rate"] = pass_df["passing"] / pass_df["candidates"]
                        st.altair_chart(
                            alt.Chart(pass_df)
                            .mark_bar()
                            .encode(
                                x=alt.X("engine label:N", sort=None, title="engine"),
                                y=alt.Y("pass rate:Q", title="common pass rate", scale=alt.Scale(domain=[0, 1])),
                                tooltip=["engine label:N", "candidates:Q", "passing:Q", alt.Tooltip("pass rate:Q", format=".2f")],
                            )
                            .properties(height=360, title="Common Pass Rate"),
                            width="stretch",
                            key=f"{run_dir.name}_design_campaign_pass_rate",
                        )

    elif view_kind == "structure":
        if not candidates:
            st.info("No final candidates are available.")
        elif workflow_recipe == "engine_scout":
            _show_design_campaign_all_structure_viewer(run_dir, display_candidates, rows, summary, inputs, params)
        else:
            unique_rows = _design_campaign_unique_rows(rows)
            choices = {
                f"{row.get('rank') or '-'} | {row.get('engine label')} | {row.get('candidate')}": row["_candidate_id"]
                for row in unique_rows
            }
            selected_label = st.selectbox("Candidate", list(choices), key=f"{run_dir.name}_design_campaign_structure_candidate")
            selected_id = choices[selected_label]
            matching_candidates = _design_campaign_candidates_by_id(display_candidates, selected_id)
            candidate = matching_candidates[-1]
            stage_records = _design_campaign_stage_structures(run_dir, matching_candidates, summary)
            if stage_records:
                stage_labels = [str(record["label"]) for record in stage_records]
                preferred_stage_labels = [
                    str(record["label"])
                    for record in stage_records
                    if str(record.get("kind") or "") == "reconstructed"
                ] or stage_labels
                stage_key = f"{run_dir.name}_{selected_id}_design_campaign_structure_stages_v2"
                current_stage_selection = st.session_state.get(stage_key)
                if isinstance(current_stage_selection, list):
                    valid_selection = [label for label in current_stage_selection if label in set(stage_labels)]
                    if valid_selection != current_stage_selection:
                        st.session_state[stage_key] = valid_selection or preferred_stage_labels
                selected_stage_labels = st.multiselect(
                    "Structure stages",
                    stage_labels,
                    default=preferred_stage_labels,
                    key=stage_key,
                )
                selected_stage_records = [
                    record for record in stage_records if str(record["label"]) in set(selected_stage_labels)
                ] or stage_records[:1]
                stage_record = selected_stage_records[0]
                structure = stage_record["path"]
                stage_reference = (
                    next((record["path"] for record in stage_records if record.get("kind") == "generator"), None)
                    or stage_record.get("reference")
                    or stage_record["path"]
                )
                st.caption(
                    "Showing "
                    + ", ".join(f"{record['label']} (`{_path_label(record['path'], run_dir)}`)" for record in selected_stage_records)
                )
            else:
                structure = _design_campaign_candidate_path(run_dir, candidate, "complex_pdb", "binder_pdb")
                stage_reference = None
                selected_stage_records = (
                    [{"label": "Candidate structure", "path": structure, "kind": "candidate", "reference": None}]
                    if structure is not None
                    else []
                )
            target = _design_campaign_candidate_path(run_dir, candidate, "target_pdb")
            design_reference = stage_reference or _design_campaign_reference_path(run_dir, candidate)
            prediction_target_chains = [str(chain) for chain in candidate.get("target_chains") or []]
            prediction_binder_chains = [str(chain) for chain in candidate.get("binder_chains") or []]
            reference_target_chains = _design_campaign_chain_ids(
                candidate,
                "input_target_chains",
                "target_chains",
            )
            reference_binder_chains = _design_campaign_chain_ids(
                candidate,
                "input_binder_chains",
                "binder_chains",
            )
            overlay_available = bool(
                design_reference is not None
                and structure is not None
                and reference_target_chains
                and reference_binder_chains
                and prediction_target_chains
                and prediction_binder_chains
            )
            show_design_overlay = overlay_available
            if not overlay_available and stage_records:
                st.caption("Stage structures are shown from the recorded workflow artifacts; target-aligned overlay needs reference chain metadata for this stage.")
            if structure is None:
                st.info("No readable PDB/CIF structure path is available for this candidate.")
            else:
                structures: list[StructureVisualization] = []
                comparison_rows: list[dict[str, Any]] = []
                stage_colors = ["0x2563eb", "0xef4444", "0xd97706", "0x7c3aed", "0x059669", "0xdb2777"]
                reference_structure = _parse_structure_file(design_reference) if show_design_overlay else None
                reference_key = str(design_reference.resolve()) if design_reference and design_reference.exists() else ""
                for stage_index, record in enumerate(selected_stage_records):
                    stage_path = record["path"]
                    stage_label = str(record["label"])
                    stage_color = stage_colors[stage_index % len(stage_colors)]
                    stage_text = _read_structure_text(stage_path)
                    target_rmsd = None
                    binder_rmsd = None
                    reference_contacts = None
                    stage_contacts = None
                    reference_min_distance = None
                    stage_min_distance = None
                    alignment_note = ""
                    if (
                        show_design_overlay
                        and reference_structure is not None
                        and str(stage_path.resolve()) != reference_key
                        and reference_target_chains
                        and reference_binder_chains
                        and prediction_target_chains
                        and prediction_binder_chains
                    ):
                        comparison: dict[str, float | int | str | None]
                        aligned_text, comparison, _prediction_chain_map = _design_campaign_pose_comparison(
                            design_reference,
                            stage_path,
                            reference_target_chains,
                            prediction_target_chains,
                            reference_binder_chains,
                            prediction_binder_chains,
                        )
                        if aligned_text:
                            stage_text = aligned_text
                        target_rmsd = comparison.get("target_rmsd")
                        binder_rmsd = comparison.get("binder_rmsd")
                        reference_contacts = comparison.get("reference_contacts")
                        stage_contacts = comparison.get("prediction_contacts")
                        reference_min_distance = comparison.get("reference_min_distance")
                        stage_min_distance = comparison.get("prediction_min_distance")
                        alignment_note = str(comparison.get("alignment") or "")
                    elif show_design_overlay and str(stage_path.resolve()) == reference_key:
                        target_rmsd = 0.0
                        binder_rmsd = 0.0
                        alignment_note = "reference frame"
                    structures.append(
                        StructureVisualization(
                            pdb=stage_text,
                            color="uniform",
                            color_params={"value": stage_color},
                            representation_type="cartoon",
                            chains=(
                                [
                                    *[
                                        ChainVisualization(
                                            chain_id=chain,
                                            color="uniform",
                                            color_params={"value": "0x9ca3af"},
                                            representation_type="cartoon",
                                        )
                                        for chain in prediction_target_chains
                                    ],
                                    *[
                                        ChainVisualization(
                                            chain_id=chain,
                                            color="uniform",
                                            color_params={"value": stage_color},
                                            representation_type="cartoon",
                                        )
                                        for chain in prediction_binder_chains
                                    ],
                                ]
                                if prediction_target_chains or prediction_binder_chains
                                else None
                            ),
                        )
                    )
                    comparison_rows.append(
                        {
                            "stage": stage_label,
                            "candidate": record.get("candidate_id") or "",
                            "sequence level": record.get("sequence_level") or "",
                            "role": record.get("role") or "",
                            "ranking metric": record.get("rank_metric") or "",
                            "ranking score": record.get("rank_score"),
                            "rank": record.get("rank") or "",
                            "target alignment RMSD": target_rmsd,
                            "binder pose RMSD after target alignment": binder_rmsd,
                            "reference contacts": reference_contacts,
                            "stage contacts": stage_contacts,
                            "reference min Cα distance": reference_min_distance,
                            "stage min Cα distance": stage_min_distance,
                            "alignment": alignment_note or ("reference frame" if str(stage_path.resolve()) == reference_key else ""),
                            "path": _path_label(stage_path, run_dir),
                        }
                    )
                molstar_custom_component(
                    structures,
                    key=(
                        f"{run_dir.name}_design_campaign_molstar_{selected_id}_"
                        f"{'-'.join(selected_stage_labels)}_{bool(show_design_overlay)}"
                    ),
                    height=560,
                    show_controls=True,
                    selection_mode=False,
                )
                if show_design_overlay and design_reference is not None:
                    st.caption(
                        "Stages are superposed on the RF/input target frame. Targets are light gray; binders use one color per selected stage."
                    )
                    st.dataframe(pd.DataFrame(comparison_rows), hide_index=True, width="stretch")
                st.caption(f"Structure: `{_path_label(structure, run_dir)}`")
                if show_design_overlay and design_reference is not None:
                    st.caption(f"Target/reference input frame: `{_path_label(design_reference, run_dir)}`")
                with structure.open("rb") as handle:
                    st.download_button(
                        "Download structure",
                        handle.read(),
                        file_name=structure.name,
                        mime="chemical/x-pdb" if structure.suffix.lower() == ".pdb" else "application/octet-stream",
                        key=f"{run_dir.name}_{selected_id}_download_structure",
                    )
            if target is not None:
                st.caption(f"Target: `{_path_label(target, run_dir)}`")
            candidate_metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
            metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
            st.subheader("Selected Candidate Details")
            detail_cols = st.columns(2)
            with detail_cols[0]:
                st.json(
                    {
                        "candidate_id": candidate.get("candidate_id"),
                        "stage": candidate.get("stage"),
                        "source_tool": candidate.get("source_tool"),
                        "target_chains": candidate.get("target_chains"),
                        "binder_chains": candidate.get("binder_chains"),
                        "binder_sequence": candidate.get("binder_sequence"),
                    },
                    expanded=False,
                )
            with detail_cols[1]:
                st.json(
                    {
                        key: candidate_metrics.get(key)
                        for key in [
                            "design_campaign_rank",
                            "design_campaign_engine_rank",
                            "common_bindcraft_pass",
                            "common_filter_failures",
                            "ipsae",
                            "ipae",
                            "iptm",
                            "rosetta_interface_dG",
                            "rosetta_interface_sc",
                            "af2_validation_parameter_family",
                            "complex_af2_parameter_family",
                            "binder_fold_af2_parameter_family",
                            "complex_prediction_protocol",
                            "binder_fold_prediction_protocol",
                            "average_binder_rmsd",
                            "average_hotspot_rmsd",
                            "target_aligned_binder_rmsd",
                            "af2_target_aligned_binder_rmsd",
                            "first_refolding_screen_screen_rmsd",
                            "second_refolding_screen_screen_rmsd",
                            "target_aligned_binder_rmsd_status",
                            "sequence_refinement_mode",
                            "campaign_evaluation_status",
                        ]
                        if key in candidate_metrics
                    },
                    expanded=False,
                )
            with st.expander("Provenance", expanded=False):
                st.json(metadata)
            st.divider()
            _show_design_campaign_all_structure_viewer(run_dir, display_candidates, rows, summary, inputs, params)

    elif view_kind == "io":
        for name in ["metadata.json", "input.json", "command.json", "result.json"]:
            with st.expander(name, expanded=False):
                st.json(result if name == "result.json" else read_json(run_dir / name))
        for log_name in ["stdout.log", "stderr.log"]:
            log_path = run_dir / log_name
            with st.expander(log_name):
                st.code(log_path.read_text(errors="ignore") if log_path.exists() else "", language="text")
        artifacts_dir = run_dir / "artifacts"
        if artifacts_dir.exists():
            st.subheader("Artifacts")
            for path in sorted(p for p in artifacts_dir.rglob("*") if p.is_file()):
                st.write(str(path.relative_to(run_dir)))
                if path.suffix.lower() == ".csv":
                    with st.expander(f"Preview {path.name}"):
                        try:
                            st.dataframe(pd.read_csv(path).head(200), hide_index=True, width="stretch")
                        except Exception as exc:
                            st.warning(f"Could not preview CSV: {exc}")


def _combined_design_campaign_result() -> tuple[Path | None, dict]:
    jobs = [
        job
        for job in collect_jobs("design-campaign")
        if (Path(str(job.get("run_dir") or "")) / "artifacts" / "design_campaign" / "common_validation").exists()
    ]
    jobs.sort(key=lambda row: row.get("updated_at") or row.get("created_at") or "")
    if not jobs:
        return None, {}

    all_candidates: list[dict[str, Any]] = []
    workflows: list[dict[str, Any]] = []
    total_metrics = {
        "candidate_count": 0,
        "harmonized_candidate_count": 0,
        "native_candidate_count": 0,
        "common_bindcraft_passing_count": 0,
        "completed_engine_count": 0,
        "failed_engine_count": 0,
    }
    campaign_rows: list[dict[str, Any]] = []
    target_chains: list[str] = []

    for job in jobs:
        campaign_run_dir = Path(str(job.get("run_dir") or ""))
        result = read_json(campaign_run_dir / "result.json")
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        input_json = read_json(campaign_run_dir / "input.json")
        inputs = input_json.get("inputs") if isinstance(input_json.get("inputs"), dict) else {}
        params = input_json.get("params") if isinstance(input_json.get("params"), dict) else {}
        campaign_name = str(
            job.get("campaign_name")
            or params.get("campaign_name")
            or campaign_run_dir.name
        )
        campaign_label = f"{job.get('job_code') or campaign_run_dir.name[:5]} | {campaign_name}"
        for chain in inputs.get("target_chains") or []:
            chain_text = str(chain)
            if chain_text not in target_chains:
                target_chains.append(chain_text)

        candidates = read_candidates(campaign_run_dir)
        for candidate in candidates:
            merged = dict(candidate)
            metadata = (
                dict(merged.get("raw_metadata"))
                if isinstance(merged.get("raw_metadata"), dict)
                else {}
            )
            metrics_for_candidate = (
                dict(merged.get("metrics"))
                if isinstance(merged.get("metrics"), dict)
                else {}
            )
            metadata["design_campaign_source_run_id"] = campaign_run_dir.name
            metadata["design_campaign_source_job_code"] = str(job.get("job_code") or "")
            metadata["design_campaign_source_campaign"] = campaign_label
            metrics_for_candidate["source_campaign"] = campaign_label
            merged["raw_metadata"] = metadata
            merged["metrics"] = metrics_for_candidate
            all_candidates.append(merged)

        summary = read_json(campaign_run_dir / "artifacts" / "design_campaign" / "workflow_summary.json")
        for workflow in summary.get("workflows") if isinstance(summary.get("workflows"), list) else []:
            row = dict(workflow)
            row["campaign"] = campaign_label
            workflows.append(row)

        for key in total_metrics:
            try:
                total_metrics[key] += int(metrics.get(key) or 0)
            except (TypeError, ValueError):
                pass
        campaign_rows.append(
            {
                "campaign": campaign_label,
                "run_id": campaign_run_dir.name,
                "status": job.get("status"),
                "candidates": len(candidates),
                "completed_engines": metrics.get("completed_engine_count", ""),
                "failed_engines": metrics.get("failed_engine_count", ""),
            }
        )

    first_run_dir = Path(str(jobs[0].get("run_dir") or ""))
    result = {
        "_aggregate_design_campaign": True,
        "_aggregate_input": {
            "inputs": {"target_chains": target_chains},
            "params": {
                "campaign_name": f"Combined completed design campaigns ({len(jobs)})",
                "gpu_device": "mixed",
            },
        },
        "metrics": {
            **total_metrics,
            "engine_count": len(
                {
                    str(candidate.get("raw_metadata", {}).get("design_campaign_engine") or candidate.get("source_tool") or "")
                    for candidate in all_candidates
                }
                - {""}
            ),
        },
        "outputs": {
            "candidates": all_candidates,
            "workflow_summary": {
                "workflows": workflows,
                "common_validation": {
                    "common_bindcraft_passing_count": total_metrics["common_bindcraft_passing_count"],
                    "native_candidate_count": total_metrics["native_candidate_count"],
                    "profile": "combined completed campaigns",
                },
                "evaluation": {"status": "mixed"},
                "campaigns": campaign_rows,
            },
        },
    }
    return first_run_dir, result


task_group = st.query_params.get("task_group", "")
run_id = st.query_params.get("run_id", "")
combine = st.query_params.get("combine", "")

st.title("Result Details")
refresh_results_button("result_details_refresh")
if not task_group or not run_id:
    st.info("Select a job from a jobs page.")
    st.stop()

run_dir = get_run_dir(task_group, run_id)
if not run_dir.exists():
    st.error(f"Run not found: {run_dir}")
    st.stop()

if task_group == "design-campaign" and combine in {"completed", "validated"}:
    combined_run_dir, combined_result = _combined_design_campaign_result()
    if combined_run_dir is None:
        st.info("No completed design campaigns are available to combine.")
    else:
        st.caption("Combined view of all design-campaign rows with common-validation outputs. No files were duplicated.")
        _show_design_campaign_results(combined_run_dir, combined_result)
    st.stop()

st.caption(str(run_dir))
show_contract_files(run_dir)
result_json = read_json(run_dir / "result.json")

zip_state_key = f"{task_group}_{run_id}_result_zip_bytes"
if st.button(
    "Prepare result zip",
    key=f"{task_group}_{run_id}_prepare_result_zip",
    help="Build the downloadable archive only when needed. Large benchmark runs can take a moment.",
):
    with st.spinner("Preparing result zip..."):
        st.session_state[zip_state_key] = build_run_zip(run_dir)

if zip_state_key in st.session_state:
    st.download_button(
        "Download result zip",
        data=st.session_state[zip_state_key],
        file_name=f"{task_group}_{run_id}.zip",
        mime="application/zip",
        key=f"{task_group}_{run_id}_download_result_zip",
        help="Creates a fresh zip from this run's contract files, logs, and artifacts. The zip is not stored in the workdir.",
    )
else:
    st.caption("Result zip is generated on demand so large benchmark pages stay responsive.")

input_json = read_json(run_dir / "input.json")
benchmark_result_groups = {"benchmark", "target-refolding"}

if task_group == "benchmark" and input_json.get("job_type") == "refolding_capacity_benchmark":
    _show_capacity_benchmark_results(run_dir, result_json)
elif task_group == "benchmark" and input_json.get("job_type") == "benchmark_matrix_workspace":
    _show_benchmark_matrix_workspace_results(run_dir, result_json)
elif task_group in benchmark_result_groups:
    _show_benchmark_results(run_dir, result_json)
elif task_group == "design-campaign":
    _show_design_campaign_results(run_dir, result_json)
    st.stop()
elif task_group == "refolding-validation" and str(input_json.get("job_type") or "") == "monomer_refolding":
    _show_monomer_refolding_results(run_dir, result_json)
    st.stop()

candidates = read_candidates(run_dir)
if candidates and task_group not in benchmark_result_groups:
    st.subheader("Candidate Set")
    stage_counts = candidate_stage_counts(candidates)
    cols = st.columns(4)
    cols[0].metric("Candidates", len(candidates))
    cols[1].metric("Stages", len(stage_counts))
    if stage_counts:
        cols[2].metric("Primary stage", max(stage_counts, key=stage_counts.get))
    source_tools = sorted({str(candidate.get("source_tool") or "unknown") for candidate in candidates})
    cols[3].metric("Source tools", len(source_tools))
    st.caption(f"Source tools: {', '.join(source_tools)}")
    preview_rows: list[dict[str, Any]] = []
    for candidate in candidates[:200]:
        metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
        preview_rows.append(
            {
                "candidate_id": candidate.get("candidate_id"),
                "stage": candidate.get("stage"),
                "source_tool": candidate.get("source_tool"),
                "binder_chains": ",".join(candidate.get("binder_chains") or []),
                "target_chains": ",".join(candidate.get("target_chains") or []),
                "binder_sequence_len": len(str(candidate.get("binder_sequence") or "")),
                "complex_pdb": candidate.get("complex_pdb"),
                "target_pdb": candidate.get("target_pdb"),
                "metric_count": len(metrics),
            }
        )
    st.dataframe(pd.DataFrame(preview_rows), hide_index=True, width="stretch")
    if task_group == "candidate-import":
        st.info(
            "Imported candidate sets are kept separate from original design jobs. "
            "Use them as sources in Refolding / Validation or Analysis to reevaluate external designs."
        )
        metric_table_path = run_dir / "artifacts" / "normalized_candidates" / "imported_candidate_metrics.csv"
        if metric_table_path.exists():
            try:
                imported_metrics_df = pd.read_csv(metric_table_path)
            except Exception as exc:
                st.warning(f"Could not read imported candidate metrics: {exc}")
                imported_metrics_df = pd.DataFrame()
        else:
            imported_metric_rows = [
                {"candidate_id": candidate.get("candidate_id"), **(candidate.get("metrics") or {})}
                for candidate in candidates
            ]
            imported_metrics_df = pd.DataFrame(imported_metric_rows)
        result_selection_manifest = read_json(
            run_dir / "artifacts" / "normalized_candidates" / "result_selection_manifest.json"
        )
        result_selection_summary_path = run_dir / "artifacts" / "normalized_candidates" / "result_selection_summary.csv"
        is_result_selection = (
            str(input_json.get("job_type") or "") == "result_selection_export"
            or bool(result_selection_manifest)
            or (
                not imported_metrics_df.empty
                and "result_selection_rank" in imported_metrics_df.columns
            )
        )
        if is_result_selection:
            st.subheader("Result Selection Ranking")
            selected_feature = (
                result_selection_manifest.get("selected_feature")
                or input_json.get("inputs", {}).get("selected_feature")
                or (
                    imported_metrics_df["result_selection_feature"].dropna().iloc[0]
                    if "result_selection_feature" in imported_metrics_df.columns
                    and not imported_metrics_df["result_selection_feature"].dropna().empty
                    else ""
                )
            )
            selected_engine = (
                result_selection_manifest.get("selected_engine")
                or input_json.get("inputs", {}).get("selected_engine")
                or (
                    imported_metrics_df["result_selection_engine"].dropna().iloc[0]
                    if "result_selection_engine" in imported_metrics_df.columns
                    and not imported_metrics_df["result_selection_engine"].dropna().empty
                    else ""
                )
            )
            direction = (
                result_selection_manifest.get("direction")
                or input_json.get("params", {}).get("direction")
                or (
                    imported_metrics_df["result_selection_direction"].dropna().iloc[0]
                    if "result_selection_direction" in imported_metrics_df.columns
                    and not imported_metrics_df["result_selection_direction"].dropna().empty
                    else ""
                )
            )
            metric_cols = st.columns(4)
            metric_cols[0].metric("Exported candidates", len(candidates))
            metric_cols[1].metric("Engine", selected_engine or "n/a")
            metric_cols[2].metric("Direction", direction or "n/a")
            metric_cols[3].metric("Source ranked rows", result_json.get("metrics", {}).get("source_ranked_count", len(candidates)))
            if selected_feature:
                st.caption(f"Ranking/filter metric: `{selected_feature}`.")
            filter_config = (
                result_selection_manifest.get("filter_config")
                or input_json.get("params", {}).get("filter_config")
                or {}
            )
            if filter_config:
                with st.expander("Filtering parameters", expanded=True):
                    st.json(filter_config, expanded=False)
            else:
                st.caption(
                    "This candidate set stores the filtered candidate order and selected ranking metric, "
                    "but this export was created before explicit filter-rule recording was added."
                )
            selection_df = pd.DataFrame()
            if result_selection_summary_path.exists():
                try:
                    selection_df = pd.read_csv(result_selection_summary_path)
                except Exception as exc:
                    st.warning(f"Could not read result-selection summary: {exc}")
            if selection_df.empty and not imported_metrics_df.empty:
                selection_df = imported_metrics_df.copy()
            if not selection_df.empty:
                visible_cols = [
                    col
                    for col in [
                        "candidate_id",
                        "parent_id",
                        "rank",
                        "result_selection_rank",
                        "feature",
                        "result_selection_feature",
                        "feature_value",
                        "result_selection_feature_value",
                        "result_selection_rank_score",
                        "parent_rank_score",
                        "rank_score_delta_vs_parent",
                        "parent_score_status",
                        "parent_passes_current_threshold",
                        "parent_score_components",
                        "binder_sequence",
                        "parent_sequence",
                        "result_selection_engine",
                        "result_selection_direction",
                        "boltzbio_rank",
                        "source_rank",
                        "complex_pdb",
                        "target_pdb",
                    ]
                    if col in selection_df.columns
                ]
                rank_sort = (
                    "rank"
                    if "rank" in selection_df.columns
                    else "result_selection_rank"
                    if "result_selection_rank" in selection_df.columns
                    else ""
                )
                if rank_sort:
                    selection_df[rank_sort] = pd.to_numeric(selection_df[rank_sort], errors="coerce")
                    selection_df = selection_df.sort_values([rank_sort, "candidate_id"], na_position="last")
                st.dataframe(
                    selection_df[visible_cols] if visible_cols else selection_df,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "rank": st.column_config.NumberColumn("filtered rank", format="%d"),
                        "result_selection_rank": st.column_config.NumberColumn("filtered rank", format="%d"),
                        "feature_value": st.column_config.NumberColumn("selected metric", format="%.4f"),
                        "result_selection_feature_value": st.column_config.NumberColumn("selected metric", format="%.4f"),
                        "result_selection_rank_score": st.column_config.NumberColumn("rank score", format="%.4f"),
                        "boltzbio_rank": st.column_config.NumberColumn("BoltzBio rank", format="%d"),
                        "source_rank": st.column_config.NumberColumn("source rank", format="%d"),
                    },
                )
        if not imported_metrics_df.empty:
            rank_cols = [
                column
                for column in [
                    "candidate_id",
                    "bindcraft_original_rank",
                    "bindcraft_final_rank",
                    "bindcraft_ranked_pdb_rank",
                    "bindcraft_rank_consistent",
                    "bindcraft_original_design",
                    "bindcraft_Rank",
                    "bindcraft_Design",
                    "bindcraft_Sequence",
                    "bindcraft_Protocol",
                    "bindcraft_Average_pLDDT",
                    "bindcraft_Average_i_pTM",
                    "bindcraft_Average_i_pAE",
                    "bindcraft_Average_dG",
                    "bindcraft_Average_dSASA",
                    "bindcraft_Average_dG/dSASA",
                    "bindcraft_Average_ShapeComplementarity",
                    "bindcraft_import_mode",
                    "sequence_source",
                    "bindcraft_metrics_csv",
                    "bindcraft_metric_sources",
                ]
                if column in imported_metrics_df.columns
            ]
            bindcraft_metric_cols = [col for col in rank_cols if col != "candidate_id"]
            if bindcraft_metric_cols:
                st.subheader("Imported BindCraft Ranking")
                st.caption(
                    "BindCraft-native ranking and score columns imported from the run CSV files, "
                    "with final_design_stats.csv used as the canonical source when present."
                )
                rank_df = imported_metrics_df[rank_cols].copy()
                rank_sort_col = (
                    "bindcraft_original_rank"
                    if "bindcraft_original_rank" in rank_df.columns
                    else "bindcraft_final_rank"
                )
                if rank_sort_col in rank_df.columns:
                    rank_df[rank_sort_col] = pd.to_numeric(rank_df[rank_sort_col], errors="coerce")
                    rank_df = rank_df.sort_values([rank_sort_col, "candidate_id"], na_position="last")
                st.dataframe(rank_df, hide_index=True, width="stretch")

analysis_csv = run_dir / "artifacts" / "analysis" / "ranked_candidates.csv"
if analysis_csv.exists():
    st.subheader("Analysis Results")
    try:
        df = pd.read_csv(analysis_csv)
        if df.empty:
            st.info("The ranked candidates table is empty.")
        else:
            metric_cols = st.columns(4)
            metric_cols[0].metric("Candidates", len(df))
            if "passes_filters" in df:
                metric_cols[1].metric("Passing", int(df["passes_filters"].fillna(False).sum()))
            if "analysis_score" in df:
                metric_cols[2].metric("Best score", f"{pd.to_numeric(df['analysis_score'], errors='coerce').max():.2f}")
            if "ipsae_error" in df:
                metric_cols[3].metric("IPSAE errors", int(df["ipsae_error"].fillna("").astype(bool).sum()))
            visible_cols = [
                "analysis_rank",
                "candidate_id",
                "passes_filters",
                "analysis_score",
                "binder_plddt",
                "confidence",
                "iptm",
                "ipsae",
                "ipsae_min",
                "ipsae_max",
                "ipsae_avg",
                "lis",
                "ipsae_min_in_calculation",
                "ipae",
                "pdockq_min",
                "pdockq_max",
                "pdockq2_min",
                "pdockq2_max",
                "ipde",
                "monomer_rmsd",
                "binder_rmsd",
                "hotspot_contact_fraction",
                "hotspots_contacted",
                "min_binder_to_hotspot_distance",
                "binder_interface_contacts",
                "hotspot_interface_contact_fraction",
                "filter_failures",
                "ipsae_error",
                "complex_pdb",
            ]
            st.dataframe(df[[col for col in visible_cols if col in df.columns]], hide_index=True, width="stretch")
            if "analysis_rank" in df and "analysis_score" in df:
                st.caption("Score by rank")
                st.line_chart(df.set_index("analysis_rank")[["analysis_score"]])
    except Exception as exc:
        st.warning(f"Could not preview analysis CSV: {exc}")

for name in ["metadata.json", "input.json", "command.json", "result.json"]:
    with st.expander(name, expanded=False):
        st.json(result_json if name == "result.json" else read_json(run_dir / name))

for log_name in ["stdout.log", "stderr.log"]:
    log_path = run_dir / log_name
    with st.expander(log_name):
        st.code(log_path.read_text(errors="ignore") if log_path.exists() else "", language="text")

artifacts_dir = run_dir / "artifacts"
if artifacts_dir.exists() and task_group not in benchmark_result_groups:
    if result_json.get("tool") == "scannet" or (artifacts_dir / "scannet").exists():
        _show_scannet_results(run_dir)
    if result_json.get("tool") == "pesto" or (artifacts_dir / "pesto").exists():
        _show_pesto_results(run_dir)
    if result_json.get("tool") == "surf2spot" or (artifacts_dir / "surf2spot").exists():
        _show_surf2spot_results(run_dir)
    if result_json.get("tool") == "masif_seed" or (artifacts_dir / "masif_seed").exists():
        _show_masif_seed_visualization(run_dir)
    st.subheader("Artifacts")
    for path in sorted(p for p in artifacts_dir.rglob("*") if p.is_file()):
        st.write(str(path.relative_to(run_dir)))
        if path.suffix.lower() == ".csv":
            with st.expander(f"Preview {path.name}"):
                try:
                    st.dataframe(pd.read_csv(path).head(200), hide_index=True, width="stretch")
                except Exception as exc:
                    st.warning(f"Could not preview CSV: {exc}")
