from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from mn_protein_design.app.pages.common import show_contract_files
from mn_protein_design.core.artifacts import build_run_zip
from mn_protein_design.core.jobs import get_run_dir, read_json
from mn_protein_design.core.runtime_estimator import ENGINE_LABELS, format_duration


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
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


def _show_csv_preview(path: Path, title: str, columns: list[str] | None = None, rows: int = 200) -> None:
    df = _read_csv(path)
    if df is None:
        return
    st.subheader(title)
    if df.empty:
        st.info(f"{title} is empty.")
        return
    shown = df
    if columns:
        keep = [col for col in columns if col in df.columns]
        if keep:
            shown = df[keep]
    st.dataframe(shown.head(rows), hide_index=True, width="stretch")


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


CLASS_COLOR_SCALE = alt.Scale(domain=["binder", "nonbinder"], range=["#D55E00", "#0072B2"])
LINE_COLOR_SCALE = alt.Scale(domain=["precision", "recall"], range=["#009E73", "#CC79A7"])


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
        ("input_", "Input"),
    ]
    for prefix, label in known_prefixes:
        if text.startswith(prefix):
            return label
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
    prefixes = [
        "alphafast_af3_",
        "af3_",
        "colabfold_",
        "colab_",
        "af2_",
        "esmfold2_",
        "boltz2_",
        "input_",
    ]
    for prefix in prefixes:
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
    for prefix in ["alphafast_af3_", "af3_", "colabfold_", "colab_", "af2_", "esmfold2_", "boltz2_", "input_"]:
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
    for prefix in ["alphafast_af3_", "af3_", "colabfold_", "colab_", "af2_", "esmfold2_", "boltz2_", "input_"]:
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


def _feature_category(feature: object) -> str:
    text = str(feature or "").lower()
    raw = _raw_feature_name(feature)
    if "_x_" in text or "interaction(" in text:
        return "Interaction Terms"
    if any(token in raw for token in ["ipsae", "ipae", "pae", "lis", "pdockq", "iptm", "ptm", "plddt", "actifptm", "ranking_confidence"]):
        return "Confidence / PAE"
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
    if any(token in raw for token in ["rmsd", "dockq", "tm_score", "lddt"]):
        return "Structure Agreement"
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
    category_counts = plot_ranking["category"].value_counts().to_dict() if "category" in plot_ranking.columns else {}
    if category_counts:
        counts_text = ", ".join(f"{name}: {count}" for name, count in sorted(category_counts.items()))
        st.caption(f"Categories in selected ranking table: {counts_text}")
    category_options = [
        "All",
        "Confidence / PAE",
        "Interface Energy / Rosetta",
        "Interface Geometry / PyMOL",
        "Structure Agreement",
        "Interaction Terms",
        "Other",
    ]
    selected_category = st.segmented_control(
        "Feature category",
        category_options,
        selection_mode="single",
        default="All",
        key=f"{run_key}_{ranking_key}_feature_category_v2",
    )
    selected_category = selected_category or "All"
    if selected_category != "All":
        plot_ranking = plot_ranking[plot_ranking["category"] == selected_category].copy()
    if plot_ranking.empty:
        st.info("No features are available for this category after the current filters.")
        return

    top_n = st.slider("Top features shown", min_value=5, max_value=50, value=20, step=5, key=f"{run_key}_{ranking_key}_top_n_v2")
    chart_state_key = (
        f"{run_key}_{ranking_key}_{selected_category}_"
        f"{int(bool(show_aliases))}_{int(bool(show_setup_features))}_{int(top_n)}"
    ).replace(" ", "_").replace("/", "_")
    top = (
        plot_ranking.dropna(subset=["best_average_precision"])
        .sort_values("best_average_precision", ascending=False)
        .head(int(top_n))
        .copy()
    )

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.subheader("Top Features By AP")
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
                    color=alt.Color("engine:N"),
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
                .properties(height=max(260, int(top_n) * 26))
                .interactive()
            )
            st.altair_chart(top_chart, width="stretch", key=f"{chart_state_key}_top_ap")
        else:
            st.info("Average precision columns are missing.")
    with chart_cols[1]:
        st.subheader("AP vs AUROC")
        if {"best_average_precision", "best_auroc"}.issubset(ranking.columns):
            scatter_chart = (
                alt.Chart(plot_ranking.dropna(subset=["best_average_precision", "best_auroc"]))
                .mark_circle(size=70, opacity=0.85)
                .encode(
                    x=alt.X("best_auroc:Q", title="AUROC", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("best_average_precision:Q", title="average precision", scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color("engine:N"),
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
                .interactive()
            )
            st.altair_chart(scatter_chart, width="stretch", key=f"{chart_state_key}_ap_auroc")
        else:
            st.info("AP/AUROC columns are missing.")

    if "best_average_precision" in ranking.columns:
        st.subheader("Best Feature Per Engine")
        engine_summary = (
            plot_ranking.dropna(subset=["best_average_precision"])
            .sort_values(["engine", "best_average_precision", "best_auroc"], ascending=[True, False, False])
            .groupby("engine", as_index=False)
            .head(1)
            .sort_values("best_average_precision", ascending=False)
        )
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
                color=alt.Color("engine:N"),
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
    feature_options = [feature for feature in plot_ranking["feature"].tolist() if feature in metrics.columns]
    if not label_col or not feature_options:
        st.info("Merged metrics do not contain compatible labels and ranked feature columns.")
        return
    selected_feature = st.selectbox("Feature", feature_options, index=0, key=f"{chart_state_key}_feature_v2")
    feature_chart_key = f"{chart_state_key}_{str(selected_feature).replace(' ', '_').replace('/', '_')}"
    selected_row = plot_ranking[plot_ranking["feature"] == selected_feature].head(1)
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


def _show_benchmark_results(run_dir: Path, result: dict) -> None:
    benchmark_dir = run_dir / "artifacts" / "benchmark"
    if not benchmark_dir.exists():
        return

    st.header("Benchmark Results")
    metrics = result.get("metrics") or {}
    outputs = result.get("outputs") or {}
    colab_summary = outputs.get("colabfold_summary_metrics") or {}
    feature_summary = _first_benchmark_feature_summary(benchmark_dir)

    cols = st.columns(6)
    with cols[0]:
        _display_metric(
            "Records",
            feature_summary.get("record_count")
            or metrics.get("record_count")
            or colab_summary.get("record_count"),
        )
    with cols[1]:
        _display_metric(
            "Binders",
            feature_summary.get("positive_count")
            or metrics.get("positive_count")
            or colab_summary.get("positive_count"),
        )
    with cols[2]:
        _display_metric(
            "Nonbinders",
            feature_summary.get("negative_count")
            or metrics.get("negative_count")
            or colab_summary.get("negative_count"),
        )
    with cols[3]:
        _display_metric(
            "Top feature",
            feature_summary.get("top_feature")
            or metrics.get("merged_benchmark_top_feature")
            or metrics.get("top_feature")
            or colab_summary.get("top_feature"),
        )
    with cols[4]:
        _display_metric(
            "Top AP",
            feature_summary.get("top_feature_average_precision")
            or metrics.get("merged_benchmark_top_feature_average_precision")
            or metrics.get("top_feature_average_precision")
            or colab_summary.get("top_feature_average_precision"),
        )
    with cols[5]:
        _display_metric("ColabFold jobs", metrics.get("colabfold_selected_count"))

    child_count = sum(
        len(outputs.get(key) or [])
        for key in ("esmfold2_child_runs", "af2_initial_guess_child_runs", "boltz2_initial_guess_child_runs")
    )
    if child_count:
        st.caption(
            f"Merged {child_count} engine sub-run(s) into this benchmark result. "
            "Use the parent tables, plots, and staged PDB outputs below for comparison."
        )

    _show_runtime_timings(metrics)

    tabs = st.tabs(["Plots", "Feature Ranking", "Per Record Metrics", "Inputs and Outputs"])
    with tabs[0]:
        _show_benchmark_plots(benchmark_dir)

    with tabs[1]:
        feature_files = _benchmark_ranking_files(benchmark_dir)
        found = False
        for path in feature_files:
            df = _read_csv(path)
            if df is None:
                continue
            expand_this = not found
            found = True
            with st.expander(_ranking_label(path), expanded=expand_this):
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
                st.dataframe((df[keep] if keep else df).head(100), hide_index=True, width="stretch")
        if not found:
            st.info("No feature-ranking CSV was produced for this run.")

    with tabs[2]:
        _show_csv_preview(
            benchmark_dir / "alphafast_af3_metrics.csv",
            "AlphaFast AF3 Metrics",
            [
                "candidate_id",
                "label",
                "alphafast_af3_ranking_score",
                "alphafast_af3_iptm",
                "alphafast_af3_ptm",
                "alphafast_af3_chain_pair_1_2_pae_min",
                "alphafast_af3_chain_pair_2_1_pae_min",
            ],
        )
        _show_csv_preview(
            benchmark_dir / "colabfold_metrics.csv",
            "ColabFold Metrics",
            [
                "binder_id",
                "candidate_id",
                "label",
                "colab_actifptm_avg",
                "colab_ptm_avg",
                "colab_iptm_avg",
            ],
        )
        _show_csv_preview(
            benchmark_dir / "benchmark_table.csv",
            "ESMFold2 Benchmark Table",
            [
                "candidate_id",
                "label",
                "benchmark_mode",
                "esmfold2_benchmark_score",
                "interface_contacts",
                "binder_plddt",
                "complex_ptm",
                "complex_iptm",
            ],
        )
        _show_csv_preview(
            benchmark_dir / "boltz2_initial_guess_metrics.csv",
            "Boltz-2 Metrics",
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
        _show_csv_preview(
            benchmark_dir / "common_interface_metrics.csv",
            "Common Interface Metrics",
            [
                "binder_id",
                "af3_ipSAE_min",
                "af3_LIS",
                "af3_pDockQ2_min",
                "colab_ipSAE_min",
                "colab_LIS",
                "colab_pDockQ2_min",
                "af2_ipSAE_min",
                "af2_LIS",
                "af2_pDockQ2_min",
                "boltz2_ipSAE_min",
                "boltz2_LIS",
                "boltz2_pDockQ2_min",
            ],
        )
        _show_csv_preview(
            benchmark_dir / "af2_common_interface_metrics.csv",
            "AF2 Common Interface Metrics",
            [
                "binder_id",
                "af2_ipSAE_min",
                "af2_ipSAE_max",
                "af2_ipSAE_avg",
                "af2_LIS",
                "af2_pDockQ_min",
                "af2_pDockQ2_min",
                "af2_ipae",
            ],
        )
        _show_csv_preview(
            benchmark_dir / "esmfold2_common_interface_metrics.csv",
            "ESMFold2 Common Interface Metrics",
            [
                "binder_id",
                "esmfold2_ipSAE_min",
                "esmfold2_ipSAE_max",
                "esmfold2_ipSAE_avg",
                "esmfold2_LIS",
                "esmfold2_pDockQ_min",
                "esmfold2_pDockQ2_min",
                "esmfold2_ipae",
            ],
        )
        _show_csv_preview(
            benchmark_dir / "merged_benchmark_metrics.csv",
            "Merged Benchmark Metrics",
            rows=200,
        )

    with tabs[3]:
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
