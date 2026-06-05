from __future__ import annotations

import ast
import json
import shlex
from io import StringIO
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from mn_protein_design.app.components.molstar_viewer import StructureVisualization, molstar_custom_component
from mn_protein_design.app.pages.common import show_contract_files
from mn_protein_design.core.artifacts import build_run_zip
from mn_protein_design.core.benchmark_presets import load_feature_presets, save_feature_preset
from mn_protein_design.core.candidates import candidate_stage_counts, read_candidates
from mn_protein_design.core.jobs import get_run_dir, read_json
from mn_protein_design.core.runtime_estimator import ENGINE_LABELS, format_duration


FEATURE_ENGINE_PREFIXES = [
    "alphafast_af3_",
    "af3_",
    "colabfold_",
    "colab_",
    "af2_",
    "esmfold2_",
    "boltz2_",
    "rf3_",
    "protenix_",
    "boltzgen_fold_",
    "input_",
]


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
        ("protenix", "Protenix"),
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


def _show_benchmark_collection_notice(benchmark_dir: Path) -> None:
    collection = read_json(benchmark_dir / "collection.json")
    if not collection:
        return
    sources = _read_csv(benchmark_dir / "benchmark_collection_sources.csv")
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
            job_code = job_code_by_run.get(run_id, run_id[:10])
            engines = str(row.get("engines") or "").strip()
            records = row.get("records")
            coverage = row.get("coverage_records")
            backbone = str(row.get("is_collection_backbone") or "").lower() == "true"
            bits = [job_code]
            if engines and engines.lower() != "nan":
                bits.append(engines)
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
    name = collection.get("name") or "Benchmark collection"
    st.info(
        f"This result is a benchmark collection: **{name}**. "
        "It combines selected engine outputs from existing benchmark jobs. "
        "Deleting this collection removes only this collection run; the source benchmark jobs and their prediction artifacts are kept."
    )
    if source_lines:
        with st.expander("Collection source jobs", expanded=True):
            st.write("\n".join(f"- {line}" for line in source_lines))


CLASS_COLOR_SCALE = alt.Scale(domain=["binder", "nonbinder", "candidate"], range=["#D55E00", "#0072B2", "#6B7280"])
LINE_COLOR_SCALE = alt.Scale(domain=["precision", "recall"], range=["#009E73", "#CC79A7"])
ENGINE_COLOR_DOMAIN = [
    "AF3",
    "AF2-IG",
    "Boltz-2",
    "BoltzGen Fold",
    "ColabFold",
    "ESMFold2",
    "Protenix",
    "RF3",
    "Input",
    "Published metrics",
    "Other",
]
ENGINE_COLOR_RANGE = [
    "#56B4E9",
    "#0072B2",
    "#E41A1C",
    "#F4A3A3",
    "#009E73",
    "#7AD99F",
    "#E69F00",
    "#F0C05A",
    "#6B7280",
    "#8B5CF6",
    "#9CA3AF",
]
ENGINE_COLOR_SCALE = alt.Scale(domain=ENGINE_COLOR_DOMAIN, range=ENGINE_COLOR_RANGE)
ENGINE_COLOR_BY_NAME = dict(zip(ENGINE_COLOR_DOMAIN, ENGINE_COLOR_RANGE))


def _engine_color_scale(engines: list[object] | pd.Series | pd.Index) -> alt.Scale:
    present = {str(engine) for engine in engines if str(engine)}
    domain = [engine for engine in ENGINE_COLOR_DOMAIN if engine in present]
    domain.extend(sorted(engine for engine in present if engine not in set(domain)))
    fallback = "#7a869a"
    color_range = [ENGINE_COLOR_BY_NAME.get(engine, fallback) for engine in domain]
    return alt.Scale(domain=domain, range=color_range)


def _feature_engine(feature: object, fallback: str = "Other") -> str:
    text = str(feature or "")
    known_prefixes = [
        ("alphafast_af3", "AF3"),
        ("af3_", "AF3"),
        ("colabfold", "ColabFold"),
        ("colab_", "ColabFold"),
        ("af2_", "AF2-IG"),
        ("esmfold2", "ESMFold2"),
        ("boltz2", "Boltz-2"),
        ("rf3_", "RF3"),
        ("protenix_", "Protenix"),
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


def _is_informative_feature(feature: object) -> bool:
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
    raw = _raw_feature_name(feature)
    recalculated_tokens = [
        "ipsae",
        "lis",
        "pdockq",
        "pdockq2",
        "contact_ipae",
        "contact_interface_pae",
        "interaction_pae",
        "min_interaction_pae",
        "pae_interaction",
        "ipae_binder_to_target",
        "ipae_target_to_binder",
        "pair_chains_iptm",
    ]
    if any(token in raw for token in recalculated_tokens) or any(token in text for token in recalculated_tokens):
        return True
    return raw in {"ipae", "interface_pae", "ipae_mean", "ipae_min", "ipae_max"}


def _is_engine_confidence_pae_feature(feature: object) -> bool:
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


def _ranking_preview(path: Path, *, rows: int = 100, feature_prefix: str | None = None) -> bool:
    df = _read_csv(path)
    if df is None or df.empty:
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
    direction = str(ranking_row.get("direction") or "higher")
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
        st.subheader("Precision / Recall By Rank")
        pr_df = ranked_df[["rank", "precision", "recall"]].melt(
            id_vars="rank",
            var_name="metric",
            value_name="value",
        )
        pr_chart = (
            alt.Chart(pr_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("rank:Q", title="rank"),
                y=alt.Y("value:Q", title="value", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("metric:N", scale=LINE_COLOR_SCALE),
                tooltip=["rank:Q", "metric:N", alt.Tooltip("value:Q", format=".3f")],
            )
            .interactive()
        )
        st.altair_chart(pr_chart, width="stretch", key=f"{chart_key}_precision_recall")
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
    rows: list[pd.Series] = []
    for _engine, group in ranking.groupby("engine", dropna=False):
        rows.append(group.sort_values(["best_average_precision", "best_auroc"], ascending=[False, False]).iloc[0])
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["best_average_precision", "best_auroc"], ascending=[False, False]).reset_index(drop=True)


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


def _ranked_design_selector(
    benchmark_dir: Path,
    records: pd.DataFrame,
    metrics: pd.DataFrame | None,
    *,
    run_key: str,
) -> str | None:
    if metrics is None or metrics.empty or "binder_id" not in metrics.columns:
        return None
    st.subheader("Design Ranking")
    st.caption(
        "Pick an engine-specific best-AP metric, then click a point or choose a ranked design. "
        "The structure matrix below updates to that design."
    )
    current_best = _best_rankable_features_by_engine(benchmark_dir, metrics)
    preset_options = load_feature_presets()
    source_options: list[tuple[str, pd.DataFrame]] = []
    if not current_best.empty:
        source_options.append(("Current run best AP features", current_best))
    for preset in preset_options:
        preset_best = _preset_rankable_features(preset, metrics)
        if not preset_best.empty:
            name = str(preset.get("name") or Path(str(preset.get("_path") or "preset")).stem)
            source_options.append((f"Preset: {name}", preset_best))
    if not source_options:
        return None
    if len(source_options) > 1:
        source_label = st.selectbox(
            "Ranking preset",
            [label for label, _features in source_options],
            index=0 if not current_best.empty else 0,
            key=f"{run_key}_benchmark_structure_rank_preset_v1",
        )
        best_features = dict(source_options)[source_label]
    else:
        source_label, best_features = source_options[0]
        st.caption(f"Ranking source: {source_label}")
    choices = [f"{row.engine}: {row.feature}" for row in best_features.itertuples(index=False)]
    selected_choice = st.selectbox(
        "Rank designs by best AP metric",
        choices,
        index=0,
        key=f"{run_key}_benchmark_structure_rank_feature_v1",
    )
    selected_row = best_features.iloc[choices.index(selected_choice)]
    feature = str(selected_row["feature"])
    direction = str(selected_row.get("direction") or "higher")
    label_col = _label_column(metrics)
    if feature not in metrics.columns:
        return None
    allowed_designs = set(records["binder_id"].dropna().astype(str))
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
    hover_columns = [
        "binder_id",
        "target_id",
        "class",
        "rank",
        "rank_score",
        feature,
    ]
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
            color=alt.Color("class:N", scale=CLASS_COLOR_SCALE),
            size=alt.condition("datum.viewer_selection", alt.value(180), alt.value(75)),
            stroke=alt.condition("datum.viewer_selection", alt.value("#111827"), alt.value(None)),
            strokeWidth=alt.condition("datum.viewer_selection", alt.value(2.5), alt.value(0)),
            tooltip=tooltip_fields,
        )
        .add_params(selector)
        .properties(height=260)
        .interactive()
    )
    event = st.altair_chart(
        chart,
        width="stretch",
        key=f"{run_key}_benchmark_structure_rank_chart_v1",
        on_select="rerun",
        selection_mode="design_pick",
    )
    clicked_design = _selected_design_from_altair_event(event)
    if clicked_design in ranked_options and st.session_state.get(ranked_design_key) != str(clicked_design):
        st.session_state[ranked_design_key] = str(clicked_design)
        st.session_state[viewer_design_key] = str(clicked_design)
        st.rerun()
    default_design = str(clicked_design) if clicked_design in ranked_options else str(highlighted_design)
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


def _show_grouped_feature_rankings(benchmark_dir: Path) -> None:
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
            _ranking_preview(merged)

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
                ("Protenix", benchmark_dir / "protenix_feature_benchmark.csv", None),
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
                ("Protenix", benchmark_dir / "protenix_common_interface_feature_benchmark.csv", None),
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
                ("Protenix", benchmark_dir / "predicted_rosetta_feature_benchmark.csv", "protenix_rosetta_"),
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
                ("Protenix", benchmark_dir / "pymol_files" / "pymol_metrics_protenix_feature_benchmark.csv", None),
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
                shown = _ranking_preview(path, feature_prefix=feature_prefix)
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


def _show_benchmark_plots(benchmark_dir: Path) -> None:
    run_key = benchmark_dir.parents[1].name if len(benchmark_dir.parents) > 1 else benchmark_dir.name
    ranking_files = _benchmark_ranking_files(benchmark_dir)
    if not ranking_files:
        st.info("No feature ranking CSV is available for plots.")
        return
    selected_ranking_path = ranking_files[0]
    if len(ranking_files) > 1:
        selected_ranking_path = st.selectbox(
            "Ranking table",
            ranking_files,
            index=0,
            format_func=_ranking_label,
            key=f"{run_key}_benchmark_ranking_table",
        )
    ranking_key = selected_ranking_path.stem
    ranking = _read_csv(selected_ranking_path)
    metrics_path = _metrics_file_for_ranking(benchmark_dir, selected_ranking_path)
    metrics = _read_csv(metrics_path) if metrics_path else None
    if ranking is None or ranking.empty:
        st.info("No merged feature ranking is available for plots.")
        return
    ranking = _with_legacy_af2_ranking_rows(benchmark_dir, selected_ranking_path, ranking)
    merged_metrics = _read_csv(benchmark_dir / "merged_benchmark_metrics.csv")
    target_filter_label = "all targets"
    if merged_metrics is not None and not merged_metrics.empty and "target_id" in merged_metrics.columns:
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
    fallback_engine = _ranking_engine_fallback(selected_ranking_path)
    ranking["engine"] = ranking["feature"].map(lambda feature: _feature_engine(feature, fallback=fallback_engine))
    for column in ["best_average_precision", "best_auroc", "average_precision", "auroc"]:
        if column in ranking.columns:
            ranking[column] = pd.to_numeric(ranking[column], errors="coerce")
    ranking["alias_count"] = 1
    ranking["aliases"] = ranking["feature"].astype(str)
    ranking["category"] = ranking["feature"].map(_feature_category)

    show_aliases = st.checkbox("Show repeated alias metrics", value=False, key=f"{run_key}_{ranking_key}_show_aliases_v2")
    show_setup_features = st.checkbox("Show setup/status features", value=False, key=f"{run_key}_{ranking_key}_show_setup_v2")
    plot_ranking = ranking if show_aliases else _deduplicate_feature_aliases(ranking)
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
        "Protenix",
        "RF3",
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
        key=f"{run_key}_{ranking_key}_engine_filter_v1",
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
        "Overath key metrics",
        "Engine Confidence / PAE",
        "Recalculated PAE / Interface",
        "Interface Energy / Rosetta",
        "Interface Geometry / PyMOL",
        "Other",
    ]
    selected_category = st.segmented_control(
        "Feature category",
        category_options,
        selection_mode="single",
        default="Overath key metrics",
        key=f"{run_key}_{ranking_key}_feature_category_v3",
    )
    selected_category = selected_category or "Overath key metrics"
    if selected_category == "Overath key metrics":
        st.caption(
            "Paper-prioritized view: interface confidence scores such as ipSAE/LIS/interface PAE/ipTM, "
            "plus orthogonal descriptors highlighted by Overath et al., including interface dG/dSASA, "
            "shape complementarity, SAP, and binder/input RMSD filters."
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
    if selected_category == "Overath key metrics":
        plot_ranking = plot_ranking[plot_ranking["feature"].map(_is_overath_key_feature)].copy()
    elif selected_category != "All":
        plot_ranking = plot_ranking[plot_ranking["category"] == selected_category].copy()
    if plot_ranking.empty:
        st.info("No features are available for this category after the current filters.")
        return
    show_best_per_engine = st.checkbox(
        "Show only best AP feature per engine in the bar plot",
        value=False,
        key=f"{run_key}_{ranking_key}_best_per_engine_bar_v1",
        help=(
            "The left plot is reduced to each engine's highest-AP feature. "
            "The AP/AUROC scatter still shows all features in the selected category and highlights those engine winners."
        ),
    )

    top_n = 20
    if not show_best_per_engine:
        top_n = st.slider("Top features shown", min_value=5, max_value=50, value=20, step=5, key=f"{run_key}_{ranking_key}_top_n_v2")
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
    chart_state_key = (
        f"{run_key}_{ranking_key}_{target_filter_label}_{selected_category}_"
        f"{int(bool(show_aliases))}_{int(bool(show_setup_features))}_{int(bool(show_best_per_engine))}_{int(top_n)}"
    ).replace(" ", "_").replace("/", "_")
    if show_best_per_engine and not engine_summary.empty:
        top = engine_summary.copy()
        bar_title = "Best Feature Per Engine By AP"
        bar_height = max(260, len(top) * 36)
    else:
        top = (
            plot_ranking.dropna(subset=["best_average_precision"])
            .sort_values("best_average_precision", ascending=False)
            .head(int(top_n))
            .copy()
        )
        bar_title = "Top Features By AP"
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
            top_chart = (
                alt.Chart(top)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "best_average_precision:Q",
                        title="average precision",
                        scale=alt.Scale(domain=[0, 1]),
                    ),
                    y=alt.Y(
                        "feature:N",
                        title="feature",
                        sort=alt.SortField(field="best_average_precision", order="descending"),
                        axis=alt.Axis(labelLimit=260),
                    ),
                    color=alt.Color("engine:N", scale=top_engine_color_scale),
                    tooltip=[
                        "feature:N",
                        "engine:N",
                        "category:N",
                        "direction:N",
                        "aliases:N",
                        alt.Tooltip("best_average_precision:Q", title="AP", format=".3f"),
                        alt.Tooltip("best_auroc:Q", title="AUROC", format=".3f"),
                        alt.Tooltip("positive_mean:Q", title="binder mean", format=".3f"),
                        alt.Tooltip("negative_mean:Q", title="nonbinder mean", format=".3f"),
                    ],
                )
                .properties(height=bar_height)
                .interactive()
            )
            st.altair_chart(top_chart, width="stretch", key=f"{chart_state_key}_top_ap")
        else:
            st.info("Average precision columns are missing.")
    with chart_cols[1]:
        st.subheader("AP vs AUROC")
        if {"best_average_precision", "best_auroc"}.issubset(plot_ranking.columns):
            scatter_data = plot_ranking.dropna(subset=["best_average_precision", "best_auroc"]).copy()
            base_scatter = (
                alt.Chart(scatter_data)
                .mark_circle(size=55, opacity=0.45 if show_best_per_engine else 0.85)
                .encode(
                    x=alt.X("best_auroc:Q", title="AUROC", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("best_average_precision:Q", title="average precision", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("engine:N", scale=plot_engine_color_scale),
                    tooltip=[
                        "feature:N",
                        "engine:N",
                        "category:N",
                        "direction:N",
                        "aliases:N",
                        alt.Tooltip("best_average_precision:Q", title="AP", format=".3f"),
                        alt.Tooltip("best_auroc:Q", title="AUROC", format=".3f"),
                    ],
                )
                .properties(height=max(260, int(top_n) * 26))
            )
            if show_best_per_engine and not engine_summary.empty:
                highlight_data = engine_summary.dropna(subset=["best_average_precision", "best_auroc"]).copy()
                highlight_scatter = (
                    alt.Chart(highlight_data)
                    .mark_point(size=180, filled=True, stroke="black", strokeWidth=2, opacity=1.0)
                    .encode(
                        x=alt.X("best_auroc:Q", title="AUROC", scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y("best_average_precision:Q", title="average precision", scale=alt.Scale(domain=[0, 1])),
                        color=alt.Color("engine:N", scale=plot_engine_color_scale),
                        tooltip=[
                            "feature:N",
                            "engine:N",
                            "category:N",
                            "direction:N",
                            "aliases:N",
                            alt.Tooltip("best_average_precision:Q", title="AP", format=".3f"),
                            alt.Tooltip("best_auroc:Q", title="AUROC", format=".3f"),
                        ],
                    )
                )
                scatter_chart = (base_scatter + highlight_scatter).interactive()
            else:
                scatter_chart = base_scatter.interactive()
            st.altair_chart(scatter_chart, width="stretch", key=f"{chart_state_key}_ap_auroc")
        else:
            st.info("AP/AUROC columns are missing.")

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
                    "feature:N",
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
    selector_ranking = engine_summary if show_best_per_engine and not engine_summary.empty else plot_ranking
    feature_options = [feature for feature in selector_ranking["feature"].tolist() if feature in metrics.columns]
    if not label_col or not feature_options:
        st.info("Merged metrics do not contain compatible labels and ranked feature columns.")
        return
    selected_feature = st.selectbox("Feature", feature_options, index=0, key=f"{chart_state_key}_feature_v2")
    feature_chart_key = f"{chart_state_key}_{str(selected_feature).replace(' ', '_').replace('/', '_')}"
    selected_row = selector_ranking[selector_ranking["feature"] == selected_feature].head(1)
    direction = selected_row["direction"].iloc[0] if "direction" in selected_row else "higher"
    plot_df = metrics[[label_col, selected_feature]].copy()
    plot_df[selected_feature] = pd.to_numeric(plot_df[selected_feature], errors="coerce")
    plot_df = plot_df.dropna(subset=[selected_feature]).reset_index(drop=True)
    plot_df["record"] = plot_df.index + 1
    plot_df["class"] = plot_df[label_col].map(lambda value: "binder" if str(value).lower() in {"1", "true", "yes", "y", "binder"} else "nonbinder")
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
        st.altair_chart(ranked_chart, width="stretch", key=f"{feature_chart_key}_ranked")
    with ranked_cols[1]:
        st.subheader("Precision / Recall By Rank")
        pr_df = ranked_df[["rank", "precision", "recall"]].melt(
            id_vars="rank",
            var_name="metric",
            value_name="value",
        )
        pr_chart = (
            alt.Chart(pr_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("rank:Q", title="rank"),
                y=alt.Y("value:Q", title="value", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("metric:N", scale=LINE_COLOR_SCALE),
                tooltip=["rank:Q", "metric:N", alt.Tooltip("value:Q", format=".3f")],
            )
            .interactive()
        )
        st.altair_chart(pr_chart, width="stretch", key=f"{feature_chart_key}_precision_recall")
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
    with st.expander("Input-order scatter", expanded=False):
        input_chart = (
            alt.Chart(plot_df)
            .mark_circle(size=75, opacity=0.9)
            .encode(
                x=alt.X("record:Q", title="record"),
                y=alt.Y(f"{selected_feature}:Q", title=selected_feature),
                color=alt.Color("class:N", scale=CLASS_COLOR_SCALE),
                tooltip=["record:Q", "class:N", alt.Tooltip(f"{selected_feature}:Q", format=".4f")],
            )
            .interactive()
        )
    st.altair_chart(input_chart, width="stretch", key=f"{feature_chart_key}_input_order")
    st.caption(f"Ranking direction for this feature: {direction}. `rank_score` is higher-is-better after applying that direction.")


STRUCTURE_ENGINE_TABLES = [
    ("AF3", "alphafast_af3", "alphafast_af3_metrics.csv"),
    ("AF2-IG", "af2_initial_guess", "af2_initial_guess_metrics.csv"),
    ("Boltz-2", "boltz2", "boltz2_initial_guess_metrics.csv"),
    ("BoltzGen Fold", "boltzgen_fold", "boltzgen_fold_metrics.csv"),
    ("ColabFold", "colabfold", "colabfold_metrics.csv"),
    ("ESMFold2", "esmfold2", "esmfold2_metrics.csv"),
    ("Protenix", "protenix", "protenix_metrics.csv"),
    ("RF3", "rf3", "rf3_metrics.csv"),
]


def _molstar_color(value: str) -> str:
    return "0x" + value.lstrip("#")


def _resolve_engine_structure_path(run_dir: Path, engine_key: str, value: object) -> Path | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text)
    candidates = [path] if path.is_absolute() else [run_dir / path]
    if not path.is_absolute():
        candidates.append(run_dir / "artifacts" / "engines" / engine_key / path)
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


def _benchmark_structure_records(run_dir: Path, benchmark_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for engine_label, engine_key, csv_name in STRUCTURE_ENGINE_TABLES:
        table = _read_csv(benchmark_dir / csv_name)
        if table is None or table.empty:
            continue
        for _, row in table.iterrows():
            design_id = str(row.get("binder_id") or _candidate_structure_id(row.get("candidate_id")) or "").strip()
            if not design_id:
                continue
            path = None
            if "complex_pdb" in table.columns:
                path = _resolve_engine_structure_path(run_dir, engine_key, row.get("complex_pdb"))
            if path is None and engine_key == "alphafast_af3" and "summary_confidences" in table.columns:
                path = _alphafast_model_from_summary(run_dir, row.get("summary_confidences"))
            if path is None and engine_key == "colabfold":
                path = _colabfold_model_for_design(run_dir, design_id)
            if path is None:
                continue
            rows.append(
                {
                    "binder_id": design_id,
                    "candidate_id": row.get("candidate_id") or design_id,
                    "engine": engine_label,
                    "engine_key": engine_key,
                    "path": str(path),
                    "target_id": row.get("target_id"),
                    "label": row.get("label") if "label" in table.columns else row.get("binder"),
                }
            )
    return pd.DataFrame(rows)


def _reference_structure_for_design(run_dir: Path, benchmark_dir: Path, design_id: str) -> Path | None:
    for csv_name in ["af2_initial_guess_metrics.csv", "boltz2_initial_guess_metrics.csv", "merged_benchmark_metrics.csv"]:
        table = _read_csv(benchmark_dir / csv_name)
        if table is None or table.empty or "binder_id" not in table.columns:
            continue
        subset = table[table["binder_id"].astype(str) == str(design_id)]
        if subset.empty:
            continue
        for column in ["monomer_refolding_reference", "input_pdb", "reference_pdb", "complex_pdb"]:
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


def _target_chains_for_design(merged_metrics: pd.DataFrame | None, design_id: str) -> list[str]:
    if merged_metrics is None or merged_metrics.empty or "binder_id" not in merged_metrics.columns:
        return []
    if "target_chains" not in merged_metrics.columns:
        return []
    subset = merged_metrics[merged_metrics["binder_id"].astype(str) == str(design_id)]
    if subset.empty:
        return []
    for value in subset["target_chains"].dropna().tolist():
        chains = _parse_chain_list(value)
        if chains:
            return chains
    return []


def _parse_structure_file(path: Path):
    from Bio.PDB import MMCIFParser, PDBParser

    if path.suffix.lower() in {".cif", ".mmcif"}:
        parser = MMCIFParser(QUIET=True)
        try:
            return parser.get_structure(path.stem, str(path))
        except KeyError as exc:
            if str(exc).strip("'\"") != "_atom_site.occupancy":
                raise
            patched_text = _mmcif_text_with_default_occupancy(path.read_text(errors="ignore"))
            return parser.get_structure(path.stem, StringIO(patched_text))
    return PDBParser(QUIET=True).get_structure(path.stem, str(path))


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


def _structure_to_pdb_text(structure: object) -> str:
    from Bio.PDB import PDBIO

    output = StringIO()
    writer = PDBIO()
    writer.set_structure(structure)
    writer.save(output)
    return output.getvalue()


def _aligned_structure_text(path: Path, fixed_structure: object, target_chains: list[str]) -> tuple[str | None, float | None, str]:
    from Bio.PDB import Superimposer

    try:
        moving_structure = _parse_structure_file(path)
        fixed_atoms = _target_ca_atoms(fixed_structure, target_chains)
        moving_atoms = _target_ca_atoms(moving_structure, target_chains)
    except Exception as exc:
        return None, None, f"alignment failed: {exc}"
    atom_count = min(len(fixed_atoms), len(moving_atoms))
    if atom_count < 3:
        return None, None, f"alignment skipped: only {atom_count} shared target CA atoms"
    superimposer = Superimposer()
    superimposer.set_atoms(fixed_atoms[:atom_count], moving_atoms[:atom_count])
    superimposer.apply(moving_structure.get_atoms())
    return _structure_to_pdb_text(moving_structure), float(superimposer.rms), f"aligned on {atom_count} target CA atoms"


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
        const referenceModel = viewer.addModel(referencePdb, 'pdb');
        referenceModel.setStyle({{}}, {{cartoon: {{color: '#8a8f98', opacity: 0.45}}}});
      }}
      const engineModel = viewer.addModel(panels[idx].pdb, 'pdb');
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


def _show_benchmark_structure_viewer(run_dir: Path, benchmark_dir: Path, merged_metrics: pd.DataFrame | None) -> None:
    st.subheader("Predicted Structure Overlay")
    st.caption(
        "Select one design and overlay the available engine predictions in the same coordinate frame. "
        "If an input/reference structure is available it is shown in gray."
    )
    records = _benchmark_structure_records(run_dir, benchmark_dir)
    if records.empty:
        st.info("No predicted PDB/mmCIF structures were found for this benchmark run.")
        return
    metadata = pd.DataFrame()
    if merged_metrics is not None and not merged_metrics.empty and "binder_id" in merged_metrics.columns:
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
    class_filter = st.segmented_control(
        "Design class",
        ["All", "Binders", "Nonbinders"],
        default="All",
        key=f"{run_dir.name}_benchmark_structure_class_filter_v1",
        help="Filter the design selector by the benchmark binder/nonbinder label.",
    )
    if label_source_column:
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
    elif class_filter != "All":
        st.info("This benchmark run does not expose binder/nonbinder labels for structure filtering.")
    design_options = sorted(records["binder_id"].dropna().astype(str).unique())
    if not design_options:
        st.info("No designs match the selected class filter.")
        return
    ranked_design = _ranked_design_selector(
        benchmark_dir,
        records,
        merged_metrics,
        run_key=run_dir.name,
    )
    default_design = ranked_design if ranked_design in design_options else design_options[0]
    design_key = f"{run_dir.name}_benchmark_structure_design_v1"
    if ranked_design in design_options and st.session_state.get(design_key) != ranked_design:
        st.session_state[design_key] = ranked_design
    selected_design = st.selectbox(
        "Design",
        design_options,
        index=design_options.index(default_design),
        key=design_key,
    )
    design_records = records[records["binder_id"].astype(str) == str(selected_design)].copy()
    engine_order = [label for label, _key, _csv in STRUCTURE_ENGINE_TABLES]
    available_engines = [engine for engine in engine_order if engine in set(design_records["engine"].astype(str))]
    selected_engines = st.multiselect(
        "Prediction engines",
        available_engines,
        default=available_engines,
        key=f"{run_dir.name}_benchmark_structure_engines_v1",
    )
    show_reference = st.checkbox(
        "Show input/reference structure",
        value=True,
        key=f"{run_dir.name}_benchmark_structure_reference_v1",
    )
    design_records = design_records[design_records["engine"].astype(str).isin(selected_engines)].copy()
    target_chains = _target_chains_for_design(merged_metrics, selected_design)
    align_on_target = st.checkbox(
        "Align predictions on target chains",
        value=True,
        disabled=not target_chains,
        key=f"{run_dir.name}_benchmark_structure_align_target_v1",
        help="Superpose each selected prediction on the declared target chain(s) before display. Binder coordinates move with the target alignment.",
    )
    if align_on_target and target_chains:
        st.caption(f"Target-chain alignment uses: {', '.join(target_chains)}")
    elif not target_chains:
        st.caption("Target-chain IDs were not found for this design; showing native coordinates.")
    viewer_layout = st.segmented_control(
        "Structure layout",
        ["Linked 3Dmol grid", "Mol* overlay"],
        default="Linked 3Dmol grid",
        key=f"{run_dir.name}_benchmark_structure_layout_v1",
        help="Linked 3Dmol grid shows one aligned prediction per panel with synchronized camera controls. Mol* overlay stacks selected predictions in one scene.",
    )
    show_reference_in_grid = False
    if viewer_layout == "Linked 3Dmol grid":
        show_reference_in_grid = st.checkbox(
            "Show input/reference in each grid panel",
            value=True,
            key=f"{run_dir.name}_benchmark_structure_grid_reference_v1",
            help="Add the input/reference structure in gray to each per-engine panel when available.",
        )
    structures: list[StructureVisualization] = []
    viewer_entries: list[dict[str, object]] = []
    reference_path = _reference_structure_for_design(run_dir, benchmark_dir, selected_design) if show_reference else None
    structure_paths = [Path(str(row["path"])) for _, row in design_records.iterrows()]
    fixed_path = reference_path or (structure_paths[0] if structure_paths else None)
    fixed_structure = None
    alignment_rows: list[dict[str, object]] = []
    if align_on_target and target_chains and fixed_path is not None:
        try:
            fixed_structure = _parse_structure_file(fixed_path)
        except Exception as exc:
            st.warning(f"Could not parse the target-alignment reference structure: {exc}")
    if reference_path is not None:
        reference_structure = StructureVisualization(
            pdb=reference_path.read_text(errors="ignore"),
            color="uniform",
            color_params={"value": "0x8a8f98"},
            representation_type="cartoon",
        )
        structures.append(reference_structure)
        viewer_entries.append(
            {
                "engine": "Input/reference",
                "structure": reference_structure,
                "pdb": reference_structure.pdb,
                "color": "#8a8f98",
                "is_reference": True,
            }
        )
        alignment_rows.append(
            {
                "engine": "Input/reference",
                "path": str(reference_path),
                "target_alignment_rmsd": None,
                "alignment": "reference frame" if fixed_path == reference_path and fixed_structure is not None else "native coordinates",
            }
        )
    for _, row in design_records.iterrows():
        path = Path(str(row["path"]))
        color = ENGINE_COLOR_BY_NAME.get(str(row["engine"]), "#7a869a")
        pdb_text = path.read_text(errors="ignore")
        rmsd = None
        alignment_note = "native coordinates"
        if fixed_structure is not None and align_on_target and target_chains:
            if fixed_path is not None and path == fixed_path:
                alignment_note = "reference frame"
            else:
                aligned_text, rmsd, alignment_note = _aligned_structure_text(path, fixed_structure, target_chains)
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
                "engine": row["engine"],
                "candidate_id": row["candidate_id"],
                "structure": engine_structure,
                "pdb": pdb_text,
                "color": color,
                "is_reference": False,
            }
        )
        alignment_rows.append(
            {
                "engine": row["engine"],
                "candidate_id": row["candidate_id"],
                "path": row["path"],
                "target_alignment_rmsd": rmsd,
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
    summary_bits = [selected_design]
    if target_id:
        summary_bits.append(f"target {target_id}")
    if label_value:
        summary_bits.append(label_value)
    st.caption(" | ".join(summary_bits))
    if viewer_layout == "Mol* overlay":
        molstar_custom_component(
            structures=structures,
            key=(
                f"{run_dir.name}_benchmark_structure_viewer_"
                f"{selected_design}_{'_'.join(selected_engines)}_{bool(reference_path)}_"
                f"{bool(align_on_target)}_{'-'.join(target_chains)}_overlay"
            ),
            height=720,
            show_controls=True,
            html_filename=f"{run_dir.name}_{selected_design}_overlay",
        )
    else:
        reference_entry = next((entry for entry in viewer_entries if entry.get("is_reference")), None)
        engine_entries = [entry for entry in viewer_entries if not entry.get("is_reference")]
        if not engine_entries and reference_entry is not None:
            engine_entries = [reference_entry]
        if len(engine_entries) > 9:
            st.caption("Showing the first 9 selected engines in the linked grid.")
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


def _show_benchmark_results(run_dir: Path, result: dict) -> None:
    benchmark_dir = run_dir / "artifacts" / "benchmark"
    if not benchmark_dir.exists():
        return

    input_json = read_json(run_dir / "input.json")
    is_refolding_evaluation = str(input_json.get("job_type") or "") == "refolding_evaluation"
    st.header("Refolding Evaluation Results" if is_refolding_evaluation else "Benchmark Results")
    metrics = result.get("metrics") or {}
    outputs = result.get("outputs") or {}
    colab_summary = outputs.get("colabfold_summary_metrics") or {}
    engine_artifacts = outputs.get("engine_artifacts") or {}
    feature_summary = _first_benchmark_feature_summary(benchmark_dir)
    merged_metrics = _read_csv(benchmark_dir / "merged_benchmark_metrics.csv")
    target_summary = _target_summary(merged_metrics)
    _show_benchmark_collection_notice(benchmark_dir)

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
                feature_summary.get("record_count")
                or metrics.get("record_count")
                or colab_summary.get("record_count"),
            )
        with cols[1]:
            _display_metric("Targets", target_summary.get("target_count") or metrics.get("target_count"))
        with cols[2]:
            _display_metric(
                "Binders",
                feature_summary.get("positive_count")
                or metrics.get("positive_count")
                or colab_summary.get("positive_count"),
            )
        with cols[3]:
            _display_metric(
                "Nonbinders",
                feature_summary.get("negative_count")
                or metrics.get("negative_count")
                or colab_summary.get("negative_count"),
            )
        with cols[4]:
            _display_metric(
                "Top feature",
                feature_summary.get("top_feature")
                or metrics.get("merged_benchmark_top_feature")
                or metrics.get("top_feature")
                or colab_summary.get("top_feature"),
            )
        with cols[5]:
            _display_metric(
                "Top AP",
                feature_summary.get("top_feature_average_precision")
                or metrics.get("merged_benchmark_top_feature_average_precision")
                or metrics.get("top_feature_average_precision")
                or colab_summary.get("top_feature_average_precision"),
            )
        with cols[6]:
            _display_metric("ColabFold jobs", metrics.get("colabfold_selected_count"))
    target_text = _target_summary_text(target_summary)
    if target_text:
        st.caption(target_text)

    collection_sources = _read_csv(benchmark_dir / "benchmark_collection_sources.csv")
    if collection_sources is not None and not collection_sources.empty:
        if "is_collection_backbone" in collection_sources.columns:
            backbone = collection_sources[collection_sources["is_collection_backbone"].astype(str).str.lower() == "true"]
        else:
            backbone = pd.DataFrame()
        if not backbone.empty:
            row = backbone.iloc[0]
            st.caption(
                "Collection backbone: "
                f"{row.get('run_id')} ({row.get('records')} records). "
                "Partial engine reruns are merged onto this record set."
            )
        partial_sources = []
        for _, row in collection_sources.iterrows():
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

    view = st.segmented_control(
        "Benchmark result view",
        ["Plots", "Feature Ranking", "Per Record Evaluation", "Structure Viewer", "Inputs and Outputs"],
        default="Plots",
        key=f"{run_dir.name}_benchmark_result_view",
    )
    view = view or "Plots"

    if view == "Plots":
        _show_benchmark_plots(benchmark_dir)

    elif view == "Feature Ranking":
        _show_grouped_feature_rankings(benchmark_dir)

    elif view == "Per Record Evaluation":
        engine_dirs = (
            {
                str(engine): run_dir / str(relative_path)
                for engine, relative_path in engine_artifacts.items()
                if relative_path
            }
            if isinstance(engine_artifacts, dict)
            else {}
        )

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
            "Protenix native metrics",
            "Protenix confidence values parsed from the prediction output, including ranking score, ipTM, pTM, pLDDT, gPDE, and recycle count.",
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
                (benchmark_dir / "protenix_common_interface_metrics.csv", "Protenix adapter"),
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
        _show_benchmark_structure_viewer(run_dir, benchmark_dir, merged_metrics)

    elif view == "Inputs and Outputs":
        chain_msa_rel = outputs.get("chain_msa_map") or metrics.get("chain_msa_map")
        chain_msa_path = run_dir / str(chain_msa_rel) if chain_msa_rel else benchmark_dir / "chain_msa_map.json"
        if chain_msa_path.exists():
            payload = read_json(chain_msa_path)
            summary = payload.get("summary") or {}
            st.subheader("Chain and MSA Usage")
            msa_cols = st.columns(4)
            with msa_cols[0]:
                _display_metric("Records", summary.get("record_count"))
            with msa_cols[1]:
                _display_metric("Target chains", summary.get("target_chain_count"))
            with msa_cols[2]:
                _display_metric("Target MSAs", summary.get("target_msa_available_count"))
            with msa_cols[3]:
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


task_group = st.query_params.get("task_group", "")
run_id = st.query_params.get("run_id", "")

st.title("Result Details")
if not task_group or not run_id:
    st.info("Select a job from a jobs page.")
    st.stop()

run_dir = get_run_dir(task_group, run_id)
if not run_dir.exists():
    st.error(f"Run not found: {run_dir}")
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

if task_group == "benchmark":
    _show_benchmark_results(run_dir, result_json)

candidates = read_candidates(run_dir)
if candidates and task_group != "benchmark":
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
if artifacts_dir.exists() and task_group != "benchmark":
    st.subheader("Artifacts")
    for path in sorted(p for p in artifacts_dir.rglob("*") if p.is_file()):
        st.write(str(path.relative_to(run_dir)))
        if path.suffix.lower() == ".csv":
            with st.expander(f"Preview {path.name}"):
                try:
                    st.dataframe(pd.read_csv(path).head(200), hide_index=True, width="stretch")
                except Exception as exc:
                    st.warning(f"Could not preview CSV: {exc}")
