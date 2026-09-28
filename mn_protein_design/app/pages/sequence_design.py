from __future__ import annotations

import gzip
from pathlib import Path

import pandas as pd
import streamlit as st

from mn_protein_design.app.components.molstar_viewer import molstar_custom_component
from mn_protein_design.app.components.molstar_viewer.dataclasses import ChainVisualization, StructureVisualization
from mn_protein_design.core.candidates import STAGE_COMPLEX_REFOLDING, STAGE_GENERATION_BACKBONE, read_candidates
from mn_protein_design.core.jobs import collect_jobs, read_json, write_json
from mn_protein_design.core.local_worker import spawn_worker_for_run
from mn_protein_design.app.pages.common import gpu_run_panel, refresh_results_button, result_link, selected_dataframe_rows
from mn_protein_design.workflows.modules import candidate_sources, load_source_candidates
from mn_protein_design.workflows import benchmark as benchmark_workflow
from mn_protein_design.workflows import refolding as refolding_workflow
from mn_protein_design.workflows.sequence_design import enqueue_sequence_design_pipeline
from mn_protein_design.workflows.sequence_design import derive_interface_masks


st.title("Sequence Design / Optimization")
st.caption("Consumes normalized backbone or predicted-complex candidates and produces sequence-designed binder candidates.")

ACCEPTED_INPUT_STAGES = [STAGE_GENERATION_BACKBONE, STAGE_COMPLEX_REFOLDING]

METRIC_DEFINITIONS = {
    "ipsae": ("ipSAE min", ["ipSAE_min", "ipsae_min"], True, "pae"),
    "ipae": ("iPAE", ["ipae"], False, "pae"),
    "pdockq": ("pDockQ min", ["pDockQ_min", "pdockq_min"], True, "pae"),
    "pdockq2": ("pDockQ2 min", ["pDockQ2_min", "pdockq2_min"], True, "pae"),
    "dg_sasa": ("Rosetta dG/dSASA", ["rosetta_interface_dG_dSASA_ratio"], False, "rosetta"),
    "shape_complementarity": ("Rosetta shape complementarity", ["rosetta_interface_sc"], True, "rosetta"),
    "dg": ("Rosetta interface dG", ["rosetta_interface_dG"], False, "rosetta"),
    "packstat": ("Rosetta packstat", ["rosetta_interface_packstat"], True, "rosetta"),
    "dsasa": ("Rosetta interface dSASA", ["rosetta_interface_dSASA"], True, "rosetta"),
    "interface_hbonds": ("Rosetta interface H-bonds", ["rosetta_interface_interface_hbonds"], True, "rosetta"),
    "unsat_hbonds": ("Rosetta unsatisfied H-bonds", ["rosetta_interface_unsat_hbonds"], False, "rosetta"),
    "target_aligned_binder_rmsd": ("Target-aligned binder RMSD", ["target_aligned_binder_rmsd"], False, "pose"),
}

RANK_KEYS = [
    "rank",
    "result_selection_rank",
    "boltzbio_rank",
    "source_rank",
    "bindcraft_original_rank",
    "bindcraft_final_rank",
    "bindcraft_ranked_pdb_rank",
    "design_campaign_rank",
    "design_campaign_engine_rank",
    "analysis_rank",
]


def _first_text(*values: object) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _format_threshold_value(value: object) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value or "").strip()
    return f"{number:.4g}"


def _sequence_source_rule_text(manifest: dict) -> str:
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


def _sequence_source_summary(source_run_dir: object) -> dict:
    source_path = Path(str(source_run_dir or ""))
    if not str(source_path):
        return {}
    metadata = read_json(source_path / "metadata.json")
    manifest = read_json(source_path / "artifacts" / "normalized_candidates" / "result_selection_manifest.json")
    label = _first_text(
        manifest.get("export_name"),
        metadata.get("title"),
        metadata.get("job_code"),
        source_path.name,
    )
    task_group = _first_text(metadata.get("task_group"))
    run_id = _first_text(metadata.get("run_id"), source_path.name)
    return {
        "source": label,
        "source_result": result_link(task_group, run_id) if task_group and run_id else "",
        "source_job_code": _first_text(metadata.get("job_code")),
        "source_tool": _first_text(manifest.get("selected_engine"), metadata.get("tool")),
        "source_feature": _first_text(manifest.get("selected_feature")),
        "source_selection_rule": _sequence_source_rule_text(manifest),
        "source_selection_kind": _first_text(manifest.get("selection_kind")),
    }


def _resolve_candidate_structure(source_run_dir: Path, candidate: dict) -> tuple[Path | None, str | None]:
    for key in ["complex_pdb", "binder_pdb", "target_pdb", "structure_pdb", "structure_cif"]:
        value = candidate.get(key)
        if not value:
            continue
        path = Path(str(value))
        path = path if path.is_absolute() else source_run_dir / path
        if path.exists():
            return path, key
    return None, None


def _resolve_candidate_path(source_run_dir: Path, candidate: dict, *keys: str) -> Path | None:
    for key in keys:
        value = candidate.get(key)
        if not value:
            continue
        path = Path(str(value))
        path = path if path.is_absolute() else source_run_dir / path
        if path.exists():
            return path
    return None


def _source_candidate_reference_path(source_run_dir: Path, candidate: dict) -> Path | None:
    raw = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
    source_candidate = raw.get("source_candidate") if isinstance(raw.get("source_candidate"), dict) else {}
    if not source_candidate:
        source_candidate = raw.get("pre_refinement_candidate") if isinstance(raw.get("pre_refinement_candidate"), dict) else {}
    source_raw = source_candidate.get("raw_metadata") if isinstance(source_candidate.get("raw_metadata"), dict) else {}
    source_run_dir_text = raw.get("source_run_dir") or source_raw.get("source_run_dir")
    reference_run_dir = Path(str(source_run_dir_text)) if source_run_dir_text else source_run_dir
    return _resolve_candidate_path(reference_run_dir, source_candidate, "complex_pdb", "target_pdb", "structure_pdb", "structure_cif")


def _candidate_rank_value(candidate: dict, fallback: int) -> int | float | str:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    for key in RANK_KEYS:
        value = candidate.get(key, metrics.get(key))
        if value is not None and str(value).strip():
            try:
                numeric = float(str(value))
                return int(numeric) if numeric.is_integer() else numeric
            except ValueError:
                return str(value)
    return fallback


def _candidate_rank_sort_value(candidate: dict, fallback: int) -> tuple[int, float | str]:
    rank = _candidate_rank_value(candidate, fallback)
    try:
        return (0, float(str(rank)))
    except ValueError:
        return (1, str(rank))


def _read_structure_text(path: Path) -> str:
    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", errors="ignore") as handle:
            return handle.read()
    return path.read_text(errors="ignore")


def _engine_prefixes(engine: str) -> list[str]:
    return {
        "AF3": ["af3"], "ColabFold": ["colab"], "AF2 initial guess": ["af2"], "ESMFold2": ["esmfold2"],
        "Boltz-2": ["boltz2", "boltz"], "RF3": ["rf3"], "OpenFold-3": ["openfold3"],
        "Protenix": ["protenix"], "BoltzGen Fold": ["boltzgen_fold"],
    }.get(engine, [])


def _advanced_metric_options(engine: str) -> dict[str, dict]:
    options: dict[str, dict] = {}
    for key, (label, _suffixes, higher, group) in METRIC_DEFINITIONS.items():
        options[key] = {"label": label, "higher": higher, "group": group, "components": [key]}
    for pae_key, pae in METRIC_DEFINITIONS.items():
        if pae[3] != "pae":
            continue
        for rosetta_key, rosetta in METRIC_DEFINITIONS.items():
            if rosetta[3] != "rosetta":
                continue
            key = f"{pae_key}__x__{rosetta_key}"
            options[key] = {
                "label": f"{pae[0]} × {rosetta[0]}",
                "higher": bool(pae[2] and rosetta[2]),
                "group": "combo",
                "components": [pae_key, rosetta_key],
            }
    return options


def _metric_option_label(options: dict[str, dict], key: str) -> str:
    spec = options.get(key, {})
    label = str(spec.get("label") or key)
    direction = "higher better" if bool(spec.get("higher")) else "lower better"
    return f"{label} ({direction})"


def _write_weighted_ranking(
    metrics_path: Path,
    candidates_path: Path,
    output_path: Path,
    engine: str,
    weights: dict[str, float],
    filters: list[dict],
    filter_logic: str,
    keep_per_parent: int,
) -> dict[str, int]:
    prefixes = _engine_prefixes(engine)
    table = pd.read_csv(metrics_path)
    metric_options = _advanced_metric_options(engine)

    def metric_values(key: str) -> pd.Series | None:
        spec = metric_options.get(key)
        if not spec:
            return None
        components = list(spec["components"])
        if len(components) == 2:
            left, right = (metric_values(component) for component in components)
            return None if left is None or right is None else left * right
        definition = METRIC_DEFINITIONS.get(components[0])
        if not definition:
            return None
        column = next(
            (f"{prefix}_{suffix}" for prefix in prefixes for suffix in definition[1] if f"{prefix}_{suffix}" in table.columns),
            None,
        )
        return pd.to_numeric(table[column], errors="coerce") if column else None

    required = [key for key, weight in weights.items() if weight > 0]
    active_keys = set(required) | {str(rule.get("metric") or "") for rule in filters if rule.get("enabled")}
    values = {key: metric_values(key) for key in active_keys}
    missing = [key for key in required if values.get(key) is None]
    if missing:
        raise ValueError(f"The {engine} validation output is missing ranking metrics: {', '.join(missing)}.")
    parents = {
        str(row.get("candidate_id") or ""): str((row.get("parents") or [row.get("candidate_id")])[0] or "")
        for row in read_candidates(candidates_path)
    }
    ranking = pd.DataFrame({"candidate_id": table["candidate_id"].astype(str)})
    ranking["parent_id"] = ranking["candidate_id"].map(parents).fillna(ranking["candidate_id"])
    for key, value in values.items():
        if value is not None:
            ranking[key] = value
    ranking = ranking.dropna(subset=required).copy()
    if ranking.empty:
        raise ValueError("No completed validation rows contained every selected ranking metric.")
    rule_masks: list[pd.Series] = []
    for rule in filters:
        if not rule.get("enabled"):
            continue
        key = str(rule.get("metric") or "")
        if key not in ranking:
            rule_masks.append(pd.Series(False, index=ranking.index))
            continue
        threshold = float(rule.get("threshold") or 0.0)
        rule_masks.append(ranking[key] >= threshold if rule.get("operator") == ">=" else ranking[key] <= threshold)
    if not rule_masks:
        ranking["passes_filters"] = True
    elif filter_logic == "any":
        ranking["passes_filters"] = pd.concat(rule_masks, axis=1).any(axis=1)
    else:
        ranking["passes_filters"] = pd.concat(rule_masks, axis=1).all(axis=1)
    ranking["combined_score"] = 0.0
    for key in required:
        low, high = ranking[key].min(), ranking[key].max()
        normalized = pd.Series(1.0, index=ranking.index) if low == high else (ranking[key] - low) / (high - low)
        if not bool(metric_options[key]["higher"]):
            normalized = 1.0 - normalized
        ranking[f"weighted_{key}"] = normalized * weights[key]
        ranking["combined_score"] += ranking[f"weighted_{key}"]
    ranking = ranking.sort_values(["passes_filters", "combined_score", "candidate_id"], ascending=[False, False, True]).reset_index(drop=True)
    ranking.insert(0, "rank", ranking.index + 1)
    ranking["parent_rank"] = ranking.groupby("parent_id").cumcount() + 1
    ranking["retained"] = ranking["passes_filters"] & (ranking["parent_rank"] <= max(1, int(keep_per_parent)))
    ranking.to_csv(output_path, index=False)
    return {"ranked_count": len(ranking), "retained_count": int(ranking["retained"].sum())}


def _sequence_design_resolved_path_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).resolve())
    except (OSError, RuntimeError):
        return text


def _sequence_design_result_rows(source_run_dir: Path | str | None = None) -> list[dict]:
    source_run_dir_text = _sequence_design_resolved_path_text(source_run_dir)
    design_jobs = [
        row
        for row in collect_jobs("design")
        if str(row.get("job_type") or "") == "sequence_design"
        and str(row.get("tool") or "") in {"ligandmpnn", "foundry_mpnn"}
    ]
    validation_by_source: dict[str, list[dict]] = {}
    for row in collect_jobs("benchmark"):
        run_dir = Path(str(row.get("run_dir") or ""))
        input_payload = read_json(run_dir / "input.json")
        if str(input_payload.get("job_type") or row.get("job_type") or "") != "sequence_design_validation":
            continue
        inputs = input_payload.get("inputs") if isinstance(input_payload.get("inputs"), dict) else {}
        source_run_dir = str(inputs.get("source_run_dir") or "")
        if source_run_dir:
            validation_by_source.setdefault(source_run_dir, []).append(row)

    rows: list[dict] = []
    for design in design_jobs:
        design_run_dir = str(design.get("run_dir") or "")
        input_payload = read_json(Path(design_run_dir) / "input.json") if design_run_dir else {}
        inputs = input_payload.get("inputs") if isinstance(input_payload.get("inputs"), dict) else {}
        design_source_run_dir = str(inputs.get("source_run_dir") or "")
        source_summary = _sequence_source_summary(design_source_run_dir)
        design_source_match = bool(source_run_dir_text) and _sequence_design_resolved_path_text(design_source_run_dir) == source_run_dir_text
        if source_run_dir_text and not design_source_match:
            continue
        validations = validation_by_source.get(design_run_dir, [])
        latest_validation = validations[0] if validations else {}
        result_payload = read_json(Path(design_run_dir) / "result.json") if design_run_dir else {}
        metrics = result_payload.get("metrics") if isinstance(result_payload.get("metrics"), dict) else {}
        rows.append(
            {
                "sequence_result": result_link("design", str(design.get("run_id") or "")),
                "validation_result": (
                    result_link("benchmark", str(latest_validation.get("run_id") or ""))
                    if latest_validation
                    else ""
                ),
                "job_code": design.get("job_code"),
                "source": source_summary.get("source"),
                "source_result": source_summary.get("source_result"),
                "source_job_code": source_summary.get("source_job_code"),
                "source_tool": source_summary.get("source_tool"),
                "source_feature": source_summary.get("source_feature"),
                "source_selection_rule": source_summary.get("source_selection_rule"),
                "source_selection_kind": source_summary.get("source_selection_kind"),
                "sequence_status": design.get("status"),
                "validation_status": latest_validation.get("status") if latest_validation else "",
                "tool": design.get("tool"),
                "candidates": metrics.get("candidate_count", ""),
                "validation_runs": len(validations),
                "source_run_id": Path(design_source_run_dir).name if design_source_run_dir else "",
                "created_at": design.get("created_at"),
                "updated_at": latest_validation.get("updated_at") or design.get("updated_at"),
                "sequence_run_id": design.get("run_id"),
                "validation_run_id": latest_validation.get("run_id") if latest_validation else "",
            }
        )
    rows.sort(key=lambda row: str(row.get("updated_at") or row.get("created_at") or ""), reverse=True)
    return rows


sources = [
    source
    for source in candidate_sources()
    if any(stage in source["stage_counts"] for stage in ACCEPTED_INPUT_STAGES)
]
sources.sort(key=lambda source: str(source.get("updated_at") or source.get("created_at") or ""), reverse=True)

if not sources:
    st.info("No backbone or predicted-complex candidate sets are available yet. Run a design tool or export a filtered result selection first.")
    st.stop()

candidates_tab, sequence_tab, interface_tab, engines_tab, metrics_tab, ranking_tab, run_tab, results_tab = st.tabs(
    ["Candidates", "Sequence Engine", "Interface Mask", "Refolding Engines", "Metrics & Filters", "Ranking", "Run", "Results"]
)
with candidates_tab:
    labels = [
        " | ".join(
            part
            for part in [
                str(source.get("job_code") or source.get("run_id") or ""),
                str(source.get("tool") or ""),
                str(source.get("campaign_name") or ""),
                f"{source.get('candidate_count', 0)} candidates",
                str(source.get("stage_counts") or {}),
                str(source.get("run_id") or ""),
            ]
            if part
        )
        for source in sources
    ]
    selected = st.selectbox("Backbone candidate set", range(len(sources)), format_func=lambda index: labels[index])
    source = sources[selected]
    candidates = load_source_candidates(source)
    source_input_stages = [stage for stage in ACCEPTED_INPUT_STAGES if stage in source["stage_counts"]]
    source_candidates = [row for row in candidates if row.get("stage") in set(source_input_stages)]
    candidate_ids = [str(row.get("candidate_id") or "") for row in source_candidates]
    selection_key = f"sequence_design_candidate_selection_{source['run_id']}"
    table_widget_key = f"sequence_design_candidate_table_{source['run_id']}"
    selection_mode = st.segmented_control(
        "Candidate selection",
        ["all", "rows"],
        default="all",
        format_func={"all": "All candidates", "rows": "Choose rows"}.get,
        key=f"sequence_design_candidate_selection_mode_{source['run_id']}",
    ) or "all"
    candidate_table = pd.DataFrame([
        {
            "rank": _candidate_rank_value(row, fallback=index),
            "candidate_id": candidate_id,
            "stage": row.get("stage"),
            "source_tool": row.get("source_tool"),
            "binder_chains": ",".join(row.get("binder_chains") or []),
            "target_chains": ",".join(row.get("target_chains") or []),
            "complex_pdb": row.get("complex_pdb"),
            "contig": row.get("contig"),
        }
        for index, (row, candidate_id) in enumerate(zip(source_candidates, candidate_ids), start=1)
    ])
    if selection_mode == "rows":
        selection_event = st.dataframe(
            candidate_table,
            width="stretch",
            height=420,
            hide_index=True,
            on_select="rerun",
            selection_mode="multi-row",
            key=table_widget_key,
        )
        selected_indices = [
            index
            for index in selected_dataframe_rows(selection_event, table_widget_key)
            if 0 <= index < len(candidate_ids)
        ]
        selected_candidate_ids = [candidate_ids[index] for index in selected_indices]
        if not selected_candidate_ids:
            st.warning("Choose one or more table rows, or switch back to All candidates.")
    else:
        st.dataframe(candidate_table, width="stretch", height=420, hide_index=True)
        selected_candidate_ids = candidate_ids
    st.session_state[selection_key] = selected_candidate_ids
    selected_candidates = [row for row in source_candidates if str(row.get("candidate_id") or "") in set(selected_candidate_ids)]
    st.info(f"{len(selected_candidate_ids):,} of {len(source_candidates):,} candidates selected for sequence design.")
    st.subheader("Structure Preview")
    ranked_preview_candidates = sorted(
        enumerate(source_candidates, start=1),
        key=lambda item: _candidate_rank_sort_value(item[1], item[0]),
    )
    preview_options = []
    for fallback_rank, candidate in ranked_preview_candidates:
        structure_path, structure_kind = _resolve_candidate_structure(source["run_dir"], candidate)
        if structure_path is None:
            continue
        rank = _candidate_rank_value(candidate, fallback=fallback_rank)
        candidate_id = str(candidate.get("candidate_id") or f"candidate_{fallback_rank}")
        preview_options.append(
            {
                "rank": rank,
                "candidate_id": candidate_id,
                "candidate": candidate,
                "path": structure_path,
                "kind": structure_kind,
                "label": f"#{rank} | {candidate_id} | {structure_path.name}",
            }
        )
    if not preview_options:
        st.info("No structure files are available for the candidates in this set.")
    else:
        preview_index = st.selectbox(
            "Ranked structure",
            range(len(preview_options)),
            format_func=lambda index: preview_options[index]["label"],
            key=f"sequence_design_structure_preview_{source['run_id']}",
        )
        preview = preview_options[int(preview_index)]
        st.caption(f"{preview['kind']} · {preview['path']}")
        reference_path = _source_candidate_reference_path(source["run_dir"], preview["candidate"])
        show_reference_overlay = st.checkbox(
            "Show parent/reference structure",
            value=reference_path is not None,
            disabled=reference_path is None,
            key=f"sequence_design_reference_overlay_{source['run_id']}_{preview['candidate_id']}",
        )
        structures = []
        if show_reference_overlay and reference_path is not None:
            structures.append(
                StructureVisualization(
                    pdb=_read_structure_text(reference_path),
                    chains=[
                        ChainVisualization(
                            chain_id=str(chain),
                            color="uniform",
                            color_params={"value": "0x9ca3af"},
                            representation_type="cartoon",
                        )
                        for chain in preview["candidate"].get("target_chains") or []
                    ]
                    or None,
                    color="uniform",
                    color_params={"value": "0x9ca3af"},
                    representation_type="cartoon",
                )
            )
            st.caption(f"Parent/reference overlay: `{reference_path}`")
        structures.append(
            StructureVisualization(
                pdb=_read_structure_text(preview["path"]),
                color="chain-id",
                representation_type="cartoon",
            )
        )
        molstar_custom_component(
            structures,
            key=(
                f"sequence_design_molstar_{source['run_id']}_{preview['candidate_id']}_"
                f"{preview['path'].name}_{bool(show_reference_overlay)}"
            ),
            height=560,
            show_controls=True,
            selection_mode=False,
            download_filename=f"{preview['candidate_id']}_structure",
        )

with sequence_tab:
    source_tool = str(source.get("tool") or "")
    default_backend = "foundry_native" if source_tool == "rfdiffusion3_foundry" and STAGE_GENERATION_BACKBONE in source_input_stages else "shared_ligandmpnn"
    backend = st.radio(
        "Sequence backend",
        ["shared_ligandmpnn", "foundry_native"],
        format_func={"shared_ligandmpnn": "Shared LigandMPNN container for PDB candidates", "foundry_native": "Foundry-native MPNN for CIF candidates"}.get,
        index=1 if default_backend == "foundry_native" else 0,
        horizontal=True,
    )
    if STAGE_COMPLEX_REFOLDING in source_input_stages:
        st.caption("This source includes predicted complexes. Shared LigandMPNN will redesign the binder chain from those predicted structures; interface-fixing can be added from contact-derived fixed residues.")
    if backend == "shared_ligandmpnn":
        model_type = st.radio("MPNN model", ["soluble_mpnn", "protein_mpnn", "ligand_mpnn"], format_func={"protein_mpnn": "ProteinMPNN weights through LigandMPNN", "ligand_mpnn": "LigandMPNN", "soluble_mpnn": "SolubleMPNN"}.get, horizontal=True)
        settings_cols = st.columns(3)
        with settings_cols[0]:
            num_seq_per_target = st.number_input("Sequences per backbone", min_value=1, max_value=1000, value=20, step=1)
        with settings_cols[1]:
            sampling_temp = st.number_input("Sampling temperature", min_value=0.0, max_value=5.0, value=0.0001, step=0.0001, format="%g")
        with settings_cols[2]:
            omit_aas = st.text_input("Omit amino acids", value="C,X")
        design_chains = st.text_input("Chains to design", value="", placeholder="Leave empty to infer non-target binder chain")
        seed = st.number_input("Seed", min_value=0, max_value=999999999, value=0, step=1)
    else:
        model_type = "ligand_mpnn"
        settings_cols = st.columns(2)
        with settings_cols[0]:
            foundry_batches = st.number_input("Number of batches", min_value=1, max_value=1000, value=1, step=1)
        with settings_cols[1]:
            foundry_batch_size = st.number_input("Batch size", min_value=1, max_value=1000, value=10, step=1)
        foundry_checkpoint = st.text_input("Foundry MPNN checkpoint", value="/weights/ligandmpnn_v_32_010_25.pt")
fixed_residues_by_candidate: dict[str, list[str]] = {}
with interface_tab:
    interface_mode = st.segmented_control(
        "Binder redesign mode",
        ["redesign_all", "fix_recalculated_interface"],
        default="fix_recalculated_interface" if STAGE_COMPLEX_REFOLDING in source_input_stages else "redesign_all",
        format_func={"redesign_all": "Redesign all", "fix_recalculated_interface": "Fix recalculated interface"}.get,
    )
    if interface_mode == "fix_recalculated_interface":
        interface_cutoff = st.number_input("Atom-contact cutoff (Å)", 2.0, 10.0, 4.0, step=0.5)
        if backend != "shared_ligandmpnn":
            st.warning("Interface locking requires the shared PDB-based LigandMPNN backend.")
        else:
            fixed_residues_by_candidate, preview = derive_interface_masks(
                source["run_dir"], selected_candidates, design_chains=design_chains, cutoff=float(interface_cutoff)
            )
            st.dataframe(pd.DataFrame(preview), hide_index=True, width="stretch")
    else:
        st.info("All binder residues will be available for redesign.")

with engines_tab:
    run_validation = st.checkbox("Refold and validate generated sequences", value=True)
    engine_state = {
        "sequence_validation_run_af3": True,
        "sequence_validation_run_colab": False,
        "sequence_validation_run_af2": False,
        "sequence_validation_run_esmfold2": False,
        "sequence_validation_run_boltz2": False,
        "sequence_validation_run_rf3": False,
        "sequence_validation_run_openfold3": False,
        "sequence_validation_run_protenix": False,
        "sequence_validation_run_protenix_v1": False,
        "sequence_validation_run_protenix_v2": False,
        "sequence_validation_run_boltzgen": False,
    }
    for state_key, default in engine_state.items():
        st.session_state.setdefault(state_key, default)
    template_msa_engine_keys = {
        "sequence_validation_run_af3",
        "sequence_validation_run_colab",
        "sequence_validation_run_esmfold2",
        "sequence_validation_run_boltz2",
        "sequence_validation_run_rf3",
        "sequence_validation_run_protenix_v1",
        "sequence_validation_run_protenix_v2",
    }
    template_only_engine_keys = set(template_msa_engine_keys)
    template_only_engine_keys.add("sequence_validation_run_af2")
    msa_engine_keys = {
        "sequence_validation_run_af3",
        "sequence_validation_run_colab",
        "sequence_validation_run_esmfold2",
        "sequence_validation_run_boltz2",
        "sequence_validation_run_rf3",
        "sequence_validation_run_openfold3",
        "sequence_validation_run_protenix",
        "sequence_validation_run_protenix_v1",
        "sequence_validation_run_protenix_v2",
    }
    bulk_cols = st.columns([1, 1, 1.5, 1.5, 1.35, 2.65])
    if bulk_cols[0].button("Select all engines", key="sequence_validation_select_all_engines"):
        for state_key in engine_state:
            st.session_state[state_key] = True
        st.rerun()
    if bulk_cols[1].button("Deselect all engines", key="sequence_validation_deselect_all_engines"):
        for state_key in engine_state:
            st.session_state[state_key] = False
        st.rerun()
    if bulk_cols[2].button("Template + MSA engines", key="sequence_validation_select_template_msa_engines"):
        for state_key in engine_state:
            st.session_state[state_key] = state_key in template_msa_engine_keys
        st.session_state["sequence_validation_af3_templates"] = True
        st.session_state["sequence_validation_af3_msa"] = True
        st.session_state["sequence_validation_colab_templates"] = True
        st.session_state["sequence_validation_colab_msa"] = True
        st.session_state["sequence_validation_af2_initial_guess"] = True
        st.session_state["sequence_validation_esm_modes"] = ["initial_guess"]
        st.session_state["sequence_validation_esm_msa"] = True
        st.session_state["sequence_validation_af2_binder_template"] = True
        st.session_state["sequence_validation_af2_interface_template"] = True
        st.session_state["sequence_validation_boltz_template"] = True
        st.session_state["sequence_validation_boltz_msa"] = True
        st.session_state["sequence_validation_rf3_template"] = True
        st.session_state["sequence_validation_rf3_msa"] = True
        st.session_state["sequence_validation_protenix_v1_template"] = True
        st.session_state["sequence_validation_protenix_v1_msa"] = True
        st.session_state["sequence_validation_protenix_v2_template"] = True
        st.session_state["sequence_validation_protenix_v2_msa"] = True
        st.rerun()
    if bulk_cols[3].button("Template-only engines", key="sequence_validation_select_template_only_engines"):
        for state_key in engine_state:
            st.session_state[state_key] = state_key in template_only_engine_keys
        st.session_state["sequence_validation_af3_templates"] = True
        st.session_state["sequence_validation_af3_msa"] = False
        st.session_state["sequence_validation_colab_templates"] = True
        st.session_state["sequence_validation_colab_msa"] = False
        st.session_state["sequence_validation_af2_initial_guess"] = True
        st.session_state["sequence_validation_esm_modes"] = ["initial_guess"]
        st.session_state["sequence_validation_esm_msa"] = False
        st.session_state["sequence_validation_af2_binder_template"] = True
        st.session_state["sequence_validation_af2_interface_template"] = True
        st.session_state["sequence_validation_boltz_template"] = True
        st.session_state["sequence_validation_boltz_msa"] = False
        st.session_state["sequence_validation_rf3_template"] = True
        st.session_state["sequence_validation_rf3_msa"] = False
        st.session_state["sequence_validation_protenix_v1_template"] = True
        st.session_state["sequence_validation_protenix_v1_msa"] = False
        st.session_state["sequence_validation_protenix_v2_template"] = True
        st.session_state["sequence_validation_protenix_v2_msa"] = False
        st.rerun()
    if bulk_cols[4].button("MSA-only engines", key="sequence_validation_select_msa_only_engines"):
        for state_key in engine_state:
            st.session_state[state_key] = state_key in msa_engine_keys
        st.session_state["sequence_validation_af3_templates"] = False
        st.session_state["sequence_validation_af3_msa"] = True
        st.session_state["sequence_validation_colab_templates"] = False
        st.session_state["sequence_validation_colab_msa"] = True
        st.session_state["sequence_validation_boltz_template"] = False
        st.session_state["sequence_validation_boltz_msa"] = True
        st.session_state["sequence_validation_rf3_template"] = False
        st.session_state["sequence_validation_rf3_msa"] = True
        st.session_state["sequence_validation_protenix_v1_template"] = False
        st.session_state["sequence_validation_protenix_v1_msa"] = True
        st.session_state["sequence_validation_protenix_v2_template"] = False
        st.session_state["sequence_validation_protenix_v2_msa"] = True
        st.session_state["sequence_validation_esm_modes"] = ["sequence"]
        st.session_state["sequence_validation_esm_msa"] = True
        st.session_state["sequence_validation_openfold3_msa"] = True
        st.session_state["sequence_validation_protenix_msa"] = True
        st.rerun()
    engine_cols = st.columns(11)
    run_af3 = engine_cols[0].checkbox("AlphaFast AF3", key="sequence_validation_run_af3", disabled=not run_validation)
    run_colab = engine_cols[1].checkbox("ColabFold", key="sequence_validation_run_colab", disabled=not run_validation)
    run_af2 = engine_cols[2].checkbox("AF2 initial guess", key="sequence_validation_run_af2", disabled=not run_validation)
    if run_af2:
        st.session_state["sequence_validation_af2_initial_guess"] = True
    run_esmfold2 = engine_cols[3].checkbox("ESMFold2", key="sequence_validation_run_esmfold2", disabled=not run_validation)
    run_boltz2 = engine_cols[4].checkbox("Boltz-2", key="sequence_validation_run_boltz2", disabled=not run_validation)
    run_rf3 = engine_cols[5].checkbox("RF3", key="sequence_validation_run_rf3", disabled=not run_validation)
    run_openfold3 = engine_cols[6].checkbox("OpenFold-3", key="sequence_validation_run_openfold3", disabled=not run_validation)
    run_protenix = engine_cols[7].checkbox("Protenix", key="sequence_validation_run_protenix", disabled=not run_validation)
    run_protenix_v1 = engine_cols[8].checkbox("Protenix v1", key="sequence_validation_run_protenix_v1", disabled=not run_validation)
    run_protenix_v2 = engine_cols[9].checkbox("Protenix v2", key="sequence_validation_run_protenix_v2", disabled=not run_validation)
    run_boltzgen = engine_cols[10].checkbox("BoltzGen fold", key="sequence_validation_run_boltzgen", disabled=not run_validation)
    selected_engines = [
        label for label, enabled in [
            ("AF3", run_af3), ("ColabFold", run_colab), ("AF2 initial guess", run_af2),
            ("ESMFold2", run_esmfold2), ("Boltz-2", run_boltz2), ("RF3", run_rf3),
            ("OpenFold-3", run_openfold3), ("Protenix", run_protenix),
            ("Protenix v1", run_protenix_v1), ("Protenix v2", run_protenix_v2),
            ("BoltzGen Fold", run_boltzgen),
        ] if enabled
    ]
    st.caption(f"Prediction engines will process {len(selected_candidate_ids):,} selected candidates.")
    with st.expander("AlphaFast AF3 Settings", expanded=run_af3):
        cols = st.columns(3)
        af3_recycles = cols[0].number_input("AF3 recycles", 1, 48, 10, disabled=not run_af3, key="sequence_validation_af3_recycles")
        af3_templates = cols[1].checkbox(
            "Use templates",
            value=True,
            disabled=not run_af3,
            key="sequence_validation_af3_templates",
            help="Embeds the staged input target chains as AF3 templates. Binder chains and binder-interface geometry are not templated.",
        )
        af3_msa = cols[2].checkbox(
            "Use target MSAs",
            value=True,
            disabled=not run_af3,
            key="sequence_validation_af3_msa",
            help="When off, AlphaFast AF3 runs with query-only/no target MSA input.",
        )
    with st.expander("ColabFold Settings", expanded=run_colab):
        cols = st.columns(4)
        colab_recycles = cols[0].number_input("ColabFold recycles", 1, 48, 3, disabled=not run_colab, key="sequence_validation_colab_recycles")
        colab_models = cols[1].number_input("ColabFold models", 1, 5, 3, disabled=not run_colab, key="sequence_validation_colab_models")
        colab_templates = cols[2].checkbox("Use templates", value=True, disabled=not run_colab, key="sequence_validation_colab_templates", help="Uses the staged input target as the template.")
        colab_msa = cols[3].checkbox("Use target MSAs", value=True, disabled=not run_colab, key="sequence_validation_colab_msa", help="When off, ColabFold does not request or inject real target MSAs.")
        colab_template_hits = 4
    with st.expander("AF2 Target-Only Initial Guess Settings", expanded=run_af2):
        cols = st.columns(4)
        af2_recycles = cols[0].number_input("AF2 recycles", 1, 24, 3, disabled=not run_af2, key="sequence_validation_af2_recycles")
        af2_multimer = cols[1].checkbox("AF2 multimer", value=True, disabled=not run_af2, key="sequence_validation_af2_multimer")
        af2_initial_guess = cols[2].checkbox(
            "Whole-complex initial guess",
            value=True,
            disabled=True,
            key="sequence_validation_af2_initial_guess",
            help="AF2 runs in this workflow always use initial-guess conditioning. Deselect AF2 to omit it from no-template/no-MSA runs.",
        )
        af2_binder_template = cols[3].checkbox("Binder/interface template (legacy)", value=False, disabled=not run_af2 or not af2_initial_guess, key="sequence_validation_af2_binder_template")
        af2_interface_template = st.checkbox("Preserve template interface geometry (legacy)", value=False, disabled=not run_af2 or not af2_initial_guess or not af2_binder_template, key="sequence_validation_af2_interface_template")
    with st.expander("ESMFold2 Settings", expanded=run_esmfold2):
        esm_modes = st.multiselect("Modes", ["sequence", "initial_guess"], default=["initial_guess"], disabled=not run_esmfold2, format_func={"sequence": "Sequence only", "initial_guess": "Selected-target distogram"}.get, key="sequence_validation_esm_modes")
        cols = st.columns(4)
        esm_msa = cols[0].checkbox(
            "Use ESMFold2 target MSAs",
            value=False,
            disabled=not run_esmfold2,
            key="sequence_validation_esm_msa",
            help=(
                "Pass prepared per-target-chain A3M files into ESMFold2 ProteinInput.msa. "
                "This is per-chain target MSA conditioning, not a ColabFold-style paired multimer A3M; "
                "missing, query-only, or mismatched MSAs are skipped and noted in the ESMFold2 metrics."
            ),
        )
        esm_steps = cols[1].number_input("ESMFold2 sampling steps", 1, 256, 68, disabled=not run_esmfold2, key="sequence_validation_esm_steps")
        esm_loops = cols[2].number_input("ESMFold2 recycling loops", 1, 64, 10, disabled=not run_esmfold2, key="sequence_validation_esm_loops")
        esm_seed = cols[3].number_input("ESMFold2 seed", 0, 999999, 0, disabled=not run_esmfold2, key="sequence_validation_esm_seed")
    with st.expander("Boltz-2 Settings", expanded=run_boltz2):
        cols = st.columns(5)
        boltz_template = cols[0].checkbox("Use templates", value=True, disabled=not run_boltz2, key="sequence_validation_boltz_template", help="Uses the staged input target as the template.")
        boltz_msa = cols[1].checkbox("Use target MSAs", value=True, disabled=not run_boltz2, key="sequence_validation_boltz_msa")
        boltz_recycles = cols[2].number_input("Recycling steps", 1, 48, 10, disabled=not run_boltz2, key="sequence_validation_boltz_recycles")
        boltz_steps = cols[3].number_input("Sampling steps", 1, 1000, 200, disabled=not run_boltz2, key="sequence_validation_boltz_steps")
        boltz_samples = cols[4].number_input("Diffusion samples", 1, 20, 3, disabled=not run_boltz2, key="sequence_validation_boltz_samples")
        boltz_full_pae = st.checkbox("Write full PAE", value=True, disabled=not run_boltz2, key="sequence_validation_boltz_full_pae")
    with st.expander("RF3 Settings", expanded=run_rf3):
        cols = st.columns(5)
        rf3_template = cols[0].checkbox("Use templates", value=True, disabled=not run_rf3, key="sequence_validation_rf3_template", help="Uses the staged input target chains as RF3 template coordinates.")
        rf3_msa = cols[1].checkbox("Use target MSAs", value=True, disabled=not run_rf3, key="sequence_validation_rf3_msa")
        rf3_recycles = cols[2].number_input("RF3 recycles", 1, 48, 10, disabled=not run_rf3, key="sequence_validation_rf3_recycles")
        rf3_steps = cols[3].number_input("RF3 diffusion steps", 1, 1000, 50, disabled=not run_rf3, key="sequence_validation_rf3_steps")
        rf3_samples = cols[4].number_input("RF3 samples", 1, 20, 5, disabled=not run_rf3, key="sequence_validation_rf3_samples")
        rf3_seed = st.number_input("RF3 seed", 0, 999999, 0, disabled=not run_rf3, key="sequence_validation_rf3_seed")
    with st.expander("OpenFold-3 Settings", expanded=run_openfold3):
        cols = st.columns(5)
        openfold3_msa = cols[0].checkbox("Use target MSAs", value=True, disabled=not run_openfold3, key="sequence_validation_openfold3_msa")
        openfold3_samples = cols[1].number_input("Diffusion samples", 1, 20, 5, disabled=not run_openfold3, key="sequence_validation_openfold3_samples")
        openfold3_seeds = cols[2].number_input("Model seeds", 1, 20, 1, disabled=not run_openfold3, key="sequence_validation_openfold3_seeds")
        openfold3_recycles = cols[3].number_input("Recycles", 1, 48, 3, disabled=not run_openfold3, key="sequence_validation_openfold3_recycles")
        openfold3_msa_server = cols[4].checkbox("Use MSA server", value=False, disabled=not run_openfold3, key="sequence_validation_openfold3_msa_server")
    with st.expander("Protenix Settings", expanded=run_protenix):
        cols = st.columns(4)
        protenix_msa = cols[0].checkbox("Use target MSAs", value=True, disabled=not run_protenix, key="sequence_validation_protenix_msa")
        protenix_cycle = cols[1].number_input("Pairformer cycles", 1, 48, 3, disabled=not run_protenix, key="sequence_validation_protenix_cycle")
        protenix_steps = cols[2].number_input("Diffusion steps", 1, 1000, 50, disabled=not run_protenix, key="sequence_validation_protenix_steps")
        protenix_samples = cols[3].number_input("Samples", 1, 20, 5, disabled=not run_protenix, key="sequence_validation_protenix_samples")
    with st.expander("Protenix v1 Settings", expanded=run_protenix_v1):
        cols = st.columns(6)
        protenix_v1_model = cols[0].selectbox(
            "Model",
            [
                refolding_workflow.PROTENIX_V1_MODEL,
                refolding_workflow.PROTENIX_V1_20250630_MODEL,
            ],
            disabled=not run_protenix_v1,
            key="sequence_validation_protenix_v1_model",
        )
        protenix_v1_msa = cols[1].checkbox("Use target MSAs", value=True, disabled=not run_protenix_v1, key="sequence_validation_protenix_v1_msa")
        protenix_v1_template = cols[2].checkbox("Use templates", value=False, disabled=not run_protenix_v1, key="sequence_validation_protenix_v1_template")
        protenix_v1_cycle = cols[3].number_input("Pairformer cycles", 1, 48, 10, disabled=not run_protenix_v1, key="sequence_validation_protenix_v1_cycle")
        protenix_v1_steps = cols[4].number_input("Diffusion steps", 1, 1000, 200, disabled=not run_protenix_v1, key="sequence_validation_protenix_v1_steps")
        protenix_v1_samples = cols[5].number_input("Samples", 1, 20, 5, disabled=not run_protenix_v1, key="sequence_validation_protenix_v1_samples")
    with st.expander("Protenix v2 Settings", expanded=run_protenix_v2):
        cols = st.columns(6)
        protenix_v2_model = cols[0].text_input("Model", value=refolding_workflow.PROTENIX_V2_MODEL, disabled=not run_protenix_v2, key="sequence_validation_protenix_v2_model")
        protenix_v2_msa = cols[1].checkbox("Use target MSAs", value=True, disabled=not run_protenix_v2, key="sequence_validation_protenix_v2_msa")
        protenix_v2_template = cols[2].checkbox("Use templates", value=False, disabled=not run_protenix_v2, key="sequence_validation_protenix_v2_template")
        protenix_v2_cycle = cols[3].number_input("Pairformer cycles", 1, 48, 10, disabled=not run_protenix_v2, key="sequence_validation_protenix_v2_cycle")
        protenix_v2_steps = cols[4].number_input("Diffusion steps", 1, 1000, 200, disabled=not run_protenix_v2, key="sequence_validation_protenix_v2_steps")
        protenix_v2_samples = cols[5].number_input("Samples", 1, 20, 5, disabled=not run_protenix_v2, key="sequence_validation_protenix_v2_samples")
    with st.expander("BoltzGen Fold Settings", expanded=run_boltzgen):
        cols = st.columns(3)
        boltzgen_recycles = cols[0].number_input("Recycling steps", 1, 48, 3, disabled=not run_boltzgen, key="sequence_validation_boltzgen_recycles")
        boltzgen_steps = cols[1].number_input("Sampling steps", 1, 1000, 200, disabled=not run_boltzgen, key="sequence_validation_boltzgen_steps")
        boltzgen_samples = cols[2].number_input("Diffusion samples", 1, 20, 5, disabled=not run_boltzgen, key="sequence_validation_boltzgen_samples")

with metrics_tab:
    metric_cols = st.columns(3)
    run_ipsae = metric_cols[0].checkbox("IP-SAE", value=True, disabled=not run_validation)
    run_rosetta = metric_cols[1].checkbox("Rosetta interface metrics", value=True, disabled=not run_validation)
    run_pymol = metric_cols[2].checkbox("PyMOL interface metrics", value=True, disabled=not run_validation)
    pyrosetta_nprocs = st.number_input("PyRosetta CPU processes", 1, 64, 32, disabled=not run_rosetta)
    filter_engine = st.selectbox("Engine for filtering and scoring", selected_engines or ["AF3"], disabled=not selected_engines)
    advanced_options = _advanced_metric_options(filter_engine)
    option_keys = list(advanced_options)
    default_filter_rules = [
        ("ipsae__x__dg_sasa", "<=", -1.664),
        ("shape_complementarity", ">=", 0.620),
        ("target_aligned_binder_rmsd", "<=", 3.5),
    ]
    with st.expander("Advanced filtering", expanded=True):
        filter_logic = st.segmented_control("Combine rules", ["all", "any"], default="all", format_func={"all": "All rules", "any": "Any rule"}.get, key="sequence_filter_logic_v2") or "all"
        rule_count = int(st.number_input("Declared filter rules", 0, 12, 3, step=1, key="sequence_filter_rule_count_v2"))
        configured_filters: list[dict] = []
        for index in range(rule_count):
            default_metric, default_operator, default_threshold = default_filter_rules[index] if index < len(default_filter_rules) else (option_keys[0], ">=", 0.0)
            cols = st.columns([0.8, 3.4, 0.9, 1.3])
            enabled = cols[0].checkbox("Use", value=index < len(default_filter_rules), key=f"sequence_filter_enabled_{index}_v2")
            metric = cols[1].selectbox(
                "Metric",
                option_keys,
                index=option_keys.index(default_metric),
                format_func=lambda key: _metric_option_label(advanced_options, key),
                key=f"sequence_filter_metric_{index}_v2",
            )
            operator = cols[2].segmented_control("Operator", [">=", "<="], default=default_operator, key=f"sequence_filter_operator_{index}_v2") or default_operator
            threshold = cols[3].number_input("Threshold", value=float(default_threshold), step=0.01, format="%.6f", key=f"sequence_filter_threshold_{index}_v2")
            configured_filters.append({"enabled": enabled, "metric": metric, "operator": operator, "threshold": float(threshold)})

with ranking_tab:
    ranking_engine = filter_engine
    ranking_metric = st.selectbox(
        "Metric used for ranking",
        option_keys,
        index=option_keys.index("ipsae__x__dg_sasa"),
        format_func=lambda key: _metric_option_label(advanced_options, key),
        key="sequence_ranking_metric_v1",
    )
    keep_per_parent = st.number_input("Keep per parent", 1, 100, 2, step=1)
    score_weights = {ranking_metric: 1.0}
    direction = "higher" if advanced_options[ranking_metric]["higher"] else "lower"
    st.caption(
        f"Rank direction for `{advanced_options[ranking_metric]['label']}`: {direction} is better. "
        "The default ipSAE min × Rosetta dG/dSASA is lower-better; Rosetta shape complementarity is higher-better."
    )

with run_tab:
    sequence_gpu_device = gpu_run_panel(key="sequence_design", default="0")
    launch = st.button(
        "Run sequence design and validation",
        type="primary",
        disabled=not selected_candidate_ids or (run_validation and not selected_engines),
    )

with results_tab:
    st.subheader("Sequence Design Results")
    refresh_results_button("sequence_design_refresh_results")
    scope = st.segmented_control(
        "Result scope",
        ["current_source", "all"],
        default="current_source",
        format_func={"current_source": "Current candidate set", "all": "All sequence-design jobs"}.get,
        key=f"sequence_design_results_scope_{source['run_id']}",
    ) or "current_source"
    rows = _sequence_design_result_rows(source["run_dir"] if scope == "current_source" else None)
    if not rows:
        if scope == "current_source":
            st.info("No sequence-design jobs were created from the currently selected candidate set yet.")
        else:
            st.info("No sequence-design jobs have been created yet.")
    else:
        if scope == "current_source":
            st.caption("Showing only sequence-design jobs created from the currently selected candidate set.")
        result_df = pd.DataFrame(rows)
        st.dataframe(
            result_df[
                [
                    "sequence_result",
                    "validation_result",
                    "job_code",
                    "source",
                    "source_result",
                    "source_job_code",
                    "source_tool",
                    "source_feature",
                    "source_selection_rule",
                    "sequence_status",
                    "validation_status",
                    "tool",
                    "candidates",
                    "validation_runs",
                    "source_run_id",
                    "updated_at",
                    "sequence_run_id",
                    "validation_run_id",
                ]
            ],
            hide_index=True,
            width="stretch",
            column_config={
                "sequence_result": st.column_config.LinkColumn("sequence result", display_text="Open"),
                "validation_result": st.column_config.LinkColumn("validation result", display_text="Open"),
                "source_result": st.column_config.LinkColumn("source result", display_text="Open"),
            },
        )

if launch:
    try:
        if backend == "shared_ligandmpnn":
            design_kwargs = {
                "model_type": model_type,
                "design_chains": design_chains,
                "num_seq_per_target": int(num_seq_per_target),
                "sampling_temp": float(sampling_temp),
                "omit_aas": omit_aas,
                "seed": int(seed) if int(seed) else None,
                "accepted_stages": source_input_stages,
                "fixed_residues_by_candidate": fixed_residues_by_candidate,
                "selected_candidate_ids": selected_candidate_ids,
                "gpu_device": sequence_gpu_device,
            }
        else:
            design_kwargs = {
                "number_of_batches": int(foundry_batches),
                "batch_size": int(foundry_batch_size),
                "checkpoint_path": foundry_checkpoint,
                "selected_candidate_ids": selected_candidate_ids,
                "gpu_device": sequence_gpu_device,
            }
        validation_kwargs = None
        if run_validation:
            validation_kwargs = {
                "run_alphafast_af3": run_af3,
                "run_colabfold": run_colab,
                "run_af2_initial_guess": run_af2,
                "run_esmfold2": run_esmfold2,
                "run_boltz2_initial_guess": run_boltz2,
                "run_rf3": run_rf3,
                "run_openfold3": run_openfold3,
                "run_protenix": run_protenix,
                "run_protenix_v1": run_protenix_v1,
                "run_protenix_v2": run_protenix_v2,
                "run_boltzgen_fold": run_boltzgen,
                "run_common_interface_metrics": run_ipsae,
                "run_pyrosetta_input_metrics": run_rosetta,
                "run_predicted_rosetta_metrics": run_rosetta,
                "run_pymol_metrics": run_pymol,
                "pyrosetta_nprocs": int(pyrosetta_nprocs),
                "esmfold2_modes": esm_modes,
                "esmfold2_use_target_msa": bool(esm_msa),
                "num_loops": int(esm_loops),
                "num_sampling_steps": int(esm_steps),
                "seed": int(esm_seed),
                "af2_num_recycles": int(af2_recycles),
                "af2_multimer": bool(af2_multimer),
                "af2_use_initial_guess": bool(run_af2),
                "af2_use_binder_template": bool(af2_binder_template),
                "af2_use_interface_template": bool(af2_interface_template),
                "colabfold_num_recycles": int(colab_recycles),
                "colabfold_num_models": int(colab_models),
                "colabfold_use_target_templates": bool(colab_templates),
                "colabfold_use_target_msa": bool(colab_msa),
                "colabfold_max_template_hits": int(colab_template_hits),
                "boltz2_use_target_template": bool(boltz_template),
                "boltz2_use_target_msa": bool(boltz_msa),
                "boltz2_recycling_steps": int(boltz_recycles),
                "boltz2_sampling_steps": int(boltz_steps),
                "boltz2_diffusion_samples": int(boltz_samples),
                "boltz2_write_full_pae": bool(boltz_full_pae),
                "rf3_use_target_msa": bool(rf3_msa),
                "rf3_use_target_template": bool(rf3_template),
                "rf3_recycles": int(rf3_recycles),
                "rf3_num_steps": int(rf3_steps),
                "rf3_diffusion_batch_size": int(rf3_samples),
                "rf3_seed": int(rf3_seed),
                "openfold3_use_target_msa": bool(openfold3_msa),
                "openfold3_num_recycles": int(openfold3_recycles),
                "openfold3_num_diffusion_samples": int(openfold3_samples),
                "openfold3_num_model_seeds": int(openfold3_seeds),
                "openfold3_use_msa_server": bool(openfold3_msa_server),
                "protenix_use_msa": bool(protenix_msa),
                "protenix_cycle": int(protenix_cycle),
                "protenix_diffusion_steps": int(protenix_steps),
                "protenix_samples": int(protenix_samples),
                "protenix_v1_model_name": str(protenix_v1_model),
                "protenix_v1_use_msa": bool(protenix_v1_msa),
                "protenix_v1_use_template": bool(protenix_v1_template),
                "protenix_v1_cycle": int(protenix_v1_cycle),
                "protenix_v1_diffusion_steps": int(protenix_v1_steps),
                "protenix_v1_samples": int(protenix_v1_samples),
                "protenix_v2_model_name": str(protenix_v2_model),
                "protenix_v2_use_msa": bool(protenix_v2_msa),
                "protenix_v2_use_template": bool(protenix_v2_template),
                "protenix_v2_cycle": int(protenix_v2_cycle),
                "protenix_v2_diffusion_steps": int(protenix_v2_steps),
                "protenix_v2_samples": int(protenix_v2_samples),
                "boltzgen_recycling_steps": int(boltzgen_recycles),
                "boltzgen_sampling_steps": int(boltzgen_steps),
                "boltzgen_diffusion_samples": int(boltzgen_samples),
                "alphafast_num_recycles": int(af3_recycles),
                "alphafast_use_target_templates": bool(af3_templates),
                "alphafast_query_only_msa": not bool(af3_msa),
                "alphafast_gpu_device": sequence_gpu_device,
                "gpu_device": sequence_gpu_device,
            }
        run_dir = enqueue_sequence_design_pipeline(
            backend=backend,
            source_run_dir=source["run_dir"],
            candidates_jsonl=source["candidates_jsonl"],
            design_kwargs=design_kwargs,
            validation_kwargs=validation_kwargs,
        )
        spawn_worker_for_run(run_dir)
        st.success("Sequence design was queued. Validation will be queued automatically after design completes.")
        st.link_button("Open sequence-design result", result_link("design", run_dir.name))
    except Exception as exc:
        st.error(str(exc))


if False:  # Legacy synchronous path retained temporarily for reference during the worker migration.
    try:
        with st.spinner("Running sequence design..."):
            if backend == "shared_ligandmpnn":
                run_dir = run_ligandmpnn_sequence_design(
                    source_run_dir=source["run_dir"],
                    candidates_jsonl=source["candidates_jsonl"],
                    model_type=model_type,
                    design_chains=design_chains,
                    num_seq_per_target=int(num_seq_per_target),
                    sampling_temp=float(sampling_temp),
                    omit_aas=omit_aas,
                    seed=int(seed) if int(seed) else None,
                    accepted_stages=source_input_stages,
                    fixed_residues_by_candidate=fixed_residues_by_candidate,
                    selected_candidate_ids=selected_candidate_ids,
                    gpu_device=sequence_gpu_device,
                )
            else:
                run_dir = run_foundry_mpnn_sequence_design(
                    source_run_dir=source["run_dir"],
                    candidates_jsonl=source["candidates_jsonl"],
                    number_of_batches=int(foundry_batches),
                    batch_size=int(foundry_batch_size),
                    checkpoint_path=foundry_checkpoint,
                    selected_candidate_ids=selected_candidate_ids,
                    gpu_device=sequence_gpu_device,
                )
        st.success("Sequence design job finished.")
        st.link_button("Open sequence-design result", result_link("design", run_dir.name))
        if run_validation:
            engine_flags = {
                "run_alphafast_af3": run_af3,
                "run_colabfold": run_colab,
                "run_af2_initial_guess": run_af2,
                "run_esmfold2": run_esmfold2,
                "run_boltz2_initial_guess": run_boltz2,
                "run_rf3": run_rf3,
                "run_openfold3": run_openfold3,
                "run_protenix": run_protenix,
                "run_protenix_v1": run_protenix_v1,
                "run_protenix_v2": run_protenix_v2,
                "run_boltzgen_fold": run_boltzgen,
            }
            with st.spinner("Refolding and calculating interface metrics..."):
                validation_run = benchmark_workflow.run_de_novo_binder_scoring_dataset(
                    source_run_dir=run_dir,
                    candidates_jsonl=run_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl",
                    selected_candidate_ids=[str(row.get("candidate_id") or "") for row in read_candidates(run_dir)],
                    mode="seq_only_csv",
                    generate_inputs=True,
                    models=[],
                    run_common_interface_metrics=run_ipsae,
                    run_pyrosetta_input_metrics=run_rosetta,
                    run_predicted_rosetta_metrics=run_rosetta,
                    run_pymol_metrics=run_pymol,
                    pyrosetta_nprocs=int(pyrosetta_nprocs),
                    esmfold2_modes=esm_modes,
                    esmfold2_use_target_msa=bool(esm_msa),
                    num_loops=int(esm_loops),
                    num_sampling_steps=int(esm_steps),
                    seed=int(esm_seed),
                    af2_num_recycles=int(af2_recycles),
                    af2_multimer=bool(af2_multimer),
                    af2_use_initial_guess=bool(run_af2),
                    af2_use_binder_template=bool(af2_binder_template),
                    af2_use_interface_template=bool(af2_interface_template),
                    colabfold_num_recycles=int(colab_recycles),
                    colabfold_num_models=int(colab_models),
                    colabfold_use_target_templates=bool(colab_templates),
                    colabfold_use_target_msa=bool(colab_msa),
                    colabfold_max_template_hits=int(colab_template_hits),
                    boltz2_use_target_template=bool(boltz_template),
                    boltz2_use_target_msa=bool(boltz_msa),
                    boltz2_recycling_steps=int(boltz_recycles),
                    boltz2_sampling_steps=int(boltz_steps),
                    boltz2_diffusion_samples=int(boltz_samples),
                    boltz2_write_full_pae=bool(boltz_full_pae),
                    rf3_use_target_msa=bool(rf3_msa),
                    rf3_use_target_template=bool(rf3_template),
                    rf3_recycles=int(rf3_recycles),
                    rf3_num_steps=int(rf3_steps),
                    rf3_diffusion_batch_size=int(rf3_samples),
                    rf3_seed=int(rf3_seed),
                    openfold3_use_target_msa=bool(openfold3_msa),
                    openfold3_num_recycles=int(openfold3_recycles),
                    openfold3_num_diffusion_samples=int(openfold3_samples),
                    openfold3_num_model_seeds=int(openfold3_seeds),
                    openfold3_use_msa_server=bool(openfold3_msa_server),
                    protenix_use_msa=bool(protenix_msa),
                    protenix_cycle=int(protenix_cycle),
                    protenix_diffusion_steps=int(protenix_steps),
                    protenix_samples=int(protenix_samples),
                    protenix_v1_model_name=str(protenix_v1_model),
                    protenix_v1_use_msa=bool(protenix_v1_msa),
                    protenix_v1_use_template=bool(protenix_v1_template),
                    protenix_v1_cycle=int(protenix_v1_cycle),
                    protenix_v1_diffusion_steps=int(protenix_v1_steps),
                    protenix_v1_samples=int(protenix_v1_samples),
                    protenix_v2_model_name=str(protenix_v2_model),
                    protenix_v2_use_msa=bool(protenix_v2_msa),
                    protenix_v2_use_template=bool(protenix_v2_template),
                    protenix_v2_cycle=int(protenix_v2_cycle),
                    protenix_v2_diffusion_steps=int(protenix_v2_steps),
                    protenix_v2_samples=int(protenix_v2_samples),
                    boltzgen_recycling_steps=int(boltzgen_recycles),
                    boltzgen_sampling_steps=int(boltzgen_steps),
                    boltzgen_diffusion_samples=int(boltzgen_samples),
                    alphafast_num_recycles=int(af3_recycles),
                    alphafast_use_target_templates=bool(af3_templates),
                    alphafast_query_only_msa=not bool(af3_msa),
                    alphafast_gpu_device=sequence_gpu_device,
                    gpu_device=sequence_gpu_device,
                    max_records=0,
                    job_type="sequence_design_validation",
                    tool_name="sequence_design_validation",
                    **engine_flags,
                )
            validation_result = read_json(validation_run / "result.json")
            if validation_result.get("success") is True:
                merged_path = str((validation_result.get("outputs") or {}).get("merged_benchmark_metrics") or "")
                if merged_path:
                    ranking_path = validation_run / "artifacts" / "benchmark" / "sequence_design_weighted_ranking.csv"
                    ranking_summary = _write_weighted_ranking(
                        validation_run / merged_path,
                        run_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl",
                        ranking_path,
                        ranking_engine,
                        score_weights,
                        configured_filters,
                        filter_logic,
                        int(keep_per_parent),
                    )
                    write_json(
                        ranking_path.with_name("sequence_design_filter_config.json"),
                        {
                            "engine": ranking_engine,
                            "filter_logic": filter_logic,
                            "filters": configured_filters,
                            "scoring_weights": score_weights,
                            "keep_per_parent": int(keep_per_parent),
                        },
                    )
                    st.success(f"Validation finished. Retained {ranking_summary['retained_count']} of {ranking_summary['ranked_count']} ranked sequences.")
                else:
                    st.warning("Validation finished without a merged metrics table, so no combined ranking was written.")
            else:
                st.warning("Validation finished with errors. Open the result for engine-level details.")
            st.link_button("Open validation result", result_link("benchmark", validation_run.name))
    except Exception as exc:
        st.error(str(exc))
