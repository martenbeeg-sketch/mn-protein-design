from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
from io import StringIO
from typing import Any
from urllib.parse import quote

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from mn_protein_design.app.components.molstar_viewer import (
    ChainVisualization,
    StructureVisualization,
    molstar_custom_component,
)
from mn_protein_design.app.pages.common import gpu_run_panel, refresh_results_button, result_link, show_pipeline_links
from mn_protein_design.core.candidates import read_candidates
from mn_protein_design.core.detection_annotations import detection_score_sources, sidechain_sasa_by_residue
from mn_protein_design.core.jobs import collect_jobs, read_json, write_json
from mn_protein_design.core.runtime_estimator import format_duration
from mn_protein_design.core.structures import filter_pdb_text, pdb_summary
from mn_protein_design.workflows import refolding as refolding_workflow
from mn_protein_design.workflows.benchmark import (
    ALPHAFAST_DB_DIR,
    ALPHAFAST_WEIGHTS_DIR,
    COLABFOLD_CACHE_DIR,
    MSA_REPOSITORY_DIR,
)
from mn_protein_design.workflows.capacity_benchmark import (
    ENGINE_LABELS,
    capacity_child_rows,
    capacity_limit_rows,
    capacity_parent_rows,
    create_practical_capacity_benchmark_from_parent,
    create_design_capacity_benchmark,
    create_refolding_capacity_benchmark,
    extend_refolding_capacity_benchmark,
)
from mn_protein_design.workflows.design import prepared_design_targets, target_label
from mn_protein_design.workflows.detection import ppi_target_jobs
from mn_protein_design.workflows.esm_binder import ESMFOLD2_MODEL_DIR
from mn_protein_design.workflows.refolding import BOLTZ_MODELS_DIR, _sequences_by_chain, _structure_chains
from mn_protein_design.workflows.target_prep import target_chain_break_summary
from mn_protein_design.workflows.target_msa import boltz_msa_paths, validate_a3m_file


DESIGN_GENERATOR_LABELS = {
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

TARGET_CATEGORY_ORDER = ["imported", "trimmed", "cleaned", "mutated", "cropped", "benchmark", "split_fragments", "unknown"]

DETECTION_TOOL_STYLES = {
    "pesto": {"label": "PeSTo", "color": "0xd946ef"},
    "masif_seed": {"label": "MaSIF", "color": "0xf97316"},
    "scannet": {"label": "ScanNet", "color": "0x22c55e"},
    "surf2spot": {"label": "Surf2Spot", "color": "0x2563eb"},
}


def _query_param_value(name: str) -> str:
    value = st.query_params.get(name, "")
    if isinstance(value, list):
        return str(value[0] if value else "")
    return str(value or "")


def _design_capacity_batch_url(batch_id: object) -> str:
    return f"/design-capacity-results?design_capacity_batch={quote(str(batch_id), safe='')}"


def _parse_lengths(text: str) -> list[int]:
    values: list[int] = []
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError:
            continue
        if value > 0 and value not in values:
            values.append(value)
    return values


def _hotspot_residues(text: str) -> set[tuple[str, int]]:
    residues: set[tuple[str, int]] = set()
    for token in str(text or "").replace(";", ",").replace(" ", ",").split(","):
        token = token.strip().replace(":", "")
        if len(token) < 2:
            continue
        try:
            residues.add((token[0].upper(), int(token[1:])))
        except ValueError:
            continue
    return residues


def _hotspot_text(residues: set[tuple[str, int]]) -> str:
    return ",".join(f"{chain}{residue}" for chain, residue in sorted(residues))


def _pdb_residue_index_map(pdb_text: str) -> dict[str, dict[int, int]]:
    mapping: dict[str, dict[int, int]] = {}
    seen: dict[str, list[int]] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
            continue
        chain = line[21].strip() or "_"
        try:
            residue = int(line[22:26])
        except ValueError:
            continue
        seen.setdefault(chain, []).append(residue)
    for chain, residues in seen.items():
        mapping[chain] = {index: residue for index, residue in enumerate(residues, start=1)}
    return mapping


def _pdb_residue_number_set(pdb_text: str) -> dict[str, set[int]]:
    residues: dict[str, set[int]] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
            continue
        try:
            residue = int(line[22:26])
        except ValueError:
            continue
        residues.setdefault(line[21].strip() or "_", set()).add(residue)
    return residues


def _viewer_residues(value: object, pdb_text: str) -> set[tuple[str, int]]:
    if not value:
        return set()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return set()
    if not isinstance(value, dict):
        return set()
    index_map = _pdb_residue_index_map(pdb_text)
    number_set = _pdb_residue_number_set(pdb_text)
    residues: set[tuple[str, int]] = set()
    for selection in value.get("sequenceSelections") or []:
        if not isinstance(selection, dict):
            continue
        chain = str(selection.get("chainId") or "").strip()
        for raw in selection.get("residues") or []:
            try:
                residue = int(raw)
            except (TypeError, ValueError):
                continue
            pdb_residue = residue if residue in number_set.get(chain, set()) else index_map.get(chain, {}).get(residue, residue)
            residues.add((chain, pdb_residue))
    return residues


def _hotspot_chains(
    residues: set[tuple[str, int]],
    *,
    color: str = "0x2563eb",
    label: str = "Selected hotspots",
) -> list[ChainVisualization] | None:
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
    ] or None


def _detection_tool_chains(scores: pd.DataFrame) -> list[ChainVisualization] | None:
    if scores.empty or "include" not in scores or "tool" not in scores:
        return None
    included = scores[scores["include"]].copy()
    if included.empty:
        return None
    chains: list[ChainVisualization] = []
    for tool, tool_rows in included.groupby("tool", sort=True):
        style = DETECTION_TOOL_STYLES.get(str(tool), {"label": str(tool), "color": "0x64748b"})
        for chain, chain_rows in tool_rows.groupby("chain", sort=True):
            residues = sorted({int(residue) for residue in chain_rows["residue"].dropna()})
            if not residues:
                continue
            chains.append(
                ChainVisualization(
                    chain_id=str(chain),
                    residues=residues,
                    color="uniform",
                    color_params={"value": style["color"]},
                    representation_type="cartoon+ball-and-stick",
                    label=f"{style['label']} predicted residues",
                )
            )
    return chains or None


def _detection_tool_legend(tools: set[str]) -> None:
    if not tools:
        return
    chips = []
    for tool in sorted(tools):
        style = DETECTION_TOOL_STYLES.get(tool, {"label": tool, "color": "0x64748b"})
        color = "#" + style["color"].replace("0x", "")
        chips.append(
            f"<span style='display:inline-flex;align-items:center;margin-right:1rem;'>"
            f"<span style='width:0.8rem;height:0.8rem;border-radius:999px;background:{color};"
            f"display:inline-block;margin-right:0.35rem;'></span>{style['label']}</span>"
        )
    st.markdown(" ".join(chips), unsafe_allow_html=True)


def _fallback_target_hotspots(target_pdb: Path | None, target_chains: list[str], count: int = 3) -> str:
    if target_pdb is None or not target_pdb.exists():
        return ""
    allowed = set(str(chain) for chain in target_chains if str(chain))
    residues: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for line in target_pdb.read_text(errors="ignore").splitlines():
        if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
            continue
        chain = line[21].strip() or "_"
        if allowed and chain not in allowed:
            continue
        try:
            residue = int(line[22:26])
        except ValueError:
            continue
        key = (chain, residue)
        if key not in seen:
            seen.add(key)
            residues.append(key)
    if not residues:
        return ""
    if len(residues) <= count:
        selected = residues
    else:
        selected = [residues[round(index * (len(residues) - 1) / max(1, count - 1))] for index in range(count)]
    return ",".join(f"{chain}{residue}" for chain, residue in selected)


def _capacity_genie_hotspots(target_pdb: Path | None, target_chains: list[str], count: int = 3) -> str:
    if target_pdb is None or not target_pdb.exists():
        return ""
    allowed = set(str(chain) for chain in target_chains if str(chain))
    try:
        sasa = sidechain_sasa_by_residue(target_pdb)
    except Exception:
        sasa = {}
    ranked = sorted(
        ((chain, residue, score) for (chain, residue), score in sasa.items() if not allowed or chain in allowed),
        key=lambda item: (-float(item[2]), item[0], item[1]),
    )
    selected: list[tuple[str, int]] = []
    for chain, residue, score in ranked:
        if score <= 0:
            continue
        if any(chain == used_chain and abs(residue - used_residue) < 6 for used_chain, used_residue in selected):
            continue
        selected.append((chain, residue))
        if len(selected) >= count:
            break
    if not selected:
        return _fallback_target_hotspots(target_pdb, target_chains, count=count)
    return ",".join(f"{chain}{residue}" for chain, residue in selected)


def _capacity_target_label(row: dict) -> str:
    try:
        return target_label(row)
    except Exception:
        source = row.get("source_label") or row.get("source_category") or "target"
        chains = ",".join(str(chain) for chain in row.get("chains") or [] if str(chain))
        suffix = f", chains {chains}" if chains else ""
        return f"{row.get('target_name') or row.get('job_code') or 'target'} ({source}{suffix})"


def _target_category(row: dict) -> str:
    return str(row.get("source_category") or row.get("prepared_kind") or row.get("task_group") or "unknown").strip() or "unknown"


def _target_category_options(categories: list[str] | set[str]) -> list[str]:
    present = {str(category) for category in categories if str(category)}
    ordered = [category for category in TARGET_CATEGORY_ORDER if category in present]
    ordered.extend(sorted(category for category in present if category not in TARGET_CATEGORY_ORDER))
    return ordered


def _target_detection_summary(target_pdb: Path, chains: list[str]) -> str:
    try:
        sources = detection_score_sources(target_pdb, chains)
    except Exception:
        sources = {}
    if not sources:
        return ""
    labels: list[str] = []
    for source in sources:
        label = str(source).split(" | ")[0]
        if label and label not in labels:
            labels.append(label)
    return ", ".join(labels)


def _prepared_target_structure_rows(targets: list[dict]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, row in enumerate(targets):
        target_pdb = Path(str(row.get("target_pdb") or "")).expanduser()
        if not target_pdb.exists():
            continue
        try:
            summary = pdb_summary(target_pdb.read_text(errors="ignore"))
            sequence_by_chain = _sequences_by_chain(target_pdb)
        except Exception:
            summary = {"chains": []}
            sequence_by_chain = {}
        declared = {str(chain) for chain in (row.get("chains") or []) if str(chain)}
        chain_ids: list[str] = []
        total_aa_length = 0
        total_fragments = 0
        total_breaks = 0
        for chain in summary.get("chains") or []:
            chain_id = str(chain.get("chain_id") or "")
            if not chain_id or (declared and chain_id not in declared):
                continue
            chain_ids.append(chain_id)
            sequence = str(sequence_by_chain.get(chain_id) or "").replace("X", "")
            length = len(sequence) or int(chain.get("residue_count") or 0)
            fragment_count = 1 if length else 0
            break_count = 0
            try:
                break_summary = target_chain_break_summary(target_pdb, chain_id)
                fragment_count = int(break_summary.get("fragment_count") or fragment_count)
                break_count = int(break_summary.get("break_count") or 0)
            except Exception:
                pass
            total_aa_length += length
            total_fragments += fragment_count
            total_breaks += break_count
        if not chain_ids:
            continue
        source_category = _target_category(row)
        source_label = row.get("source_label") or source_category or ""
        rows.append(
            {
                "select": False,
                "target": row.get("target_name") or target_pdb.stem,
                "chains": ",".join(chain_ids),
                "chain_count": len(chain_ids),
                "aa_length": total_aa_length,
                "fragments": total_fragments,
                "breaks": total_breaks,
                "source": source_label,
                "category": source_category,
                "job": row.get("job_code") or "",
                "records": row.get("records") or 1,
                "ppi_hotspot": _target_detection_summary(target_pdb, chain_ids),
                "path": str(target_pdb),
                "_target_index": index,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["category", "aa_length", "target"],
        ascending=[True, True, True],
    ).copy()


def _msa_readiness(sequence: str) -> tuple[str, str]:
    if not sequence:
        return "no sequence", ""
    host_path, _container_path = boltz_msa_paths(sequence)
    valid, _reason = validate_a3m_file(host_path)
    return ("yes" if valid else "no"), str(host_path)


def _prepared_target_chain_rows(targets: list[dict]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, row in enumerate(targets):
        target_pdb = Path(str(row.get("target_pdb") or ""))
        if not target_pdb.exists():
            continue
        try:
            from mn_protein_design.core.structures import pdb_summary
            from mn_protein_design.workflows.refolding import _sequences_by_chain

            summary = pdb_summary(target_pdb.read_text(errors="ignore"))
            sequence_by_chain = _sequences_by_chain(target_pdb)
        except Exception:
            summary = {"chains": []}
            sequence_by_chain = {}
        declared = set(str(chain) for chain in (row.get("chains") or []) if str(chain))
        for chain in summary.get("chains") or []:
            chain_id = str(chain.get("chain_id") or "")
            if not chain_id or (declared and chain_id not in declared):
                continue
            sequence = str(sequence_by_chain.get(chain_id) or "").replace("X", "")
            length = len(sequence) or int(chain.get("residue_count") or 0)
            msa_ready, msa_path = _msa_readiness(sequence)
            source_category = row.get("source_category") or row.get("prepared_kind") or ""
            source_label = row.get("source_label") or source_category or ""
            if str(source_category).lower() == "benchmark":
                source_label = f"Benchmark target: {source_label or 'Overath 2025'}"
            rows.append(
                {
                    "target": row.get("target_name") or target_pdb.stem,
                    "chain": chain_id,
                    "aa_length": length,
                    "copy_count": 1,
                    "total_sequence_length": length,
                    "source": source_label,
                    "source_category": source_category,
                    "target_msa": msa_ready,
                    "af3_msa": msa_ready,
                    "colabfold_msa": msa_ready,
                    "boltz_msa": msa_ready,
                    "rf3_msa": msa_ready,
                    "openfold3_msa": msa_ready,
                    "protenix_msa": msa_ready,
                    "protenix_v1_msa": msa_ready,
                    "protenix_v2_msa": msa_ready,
                    "esmfold2_msa": msa_ready,
                    "af2_target_template": "yes",
                    "msa_path": msa_path,
                    "job_code": row.get("job_code") or "",
                    "target_pdb": str(target_pdb),
                    "target_index": index,
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["source_category", "aa_length", "target", "chain"],
        ascending=[True, True, True, True],
    ).copy()


def _target_controls(targets: list[dict], *, key_prefix: str) -> tuple[dict | None, Path | None, list[str]]:
    if not targets:
        st.warning("Prepare, crop, or import a target first.")
        return None, None, []
    target_rows = _prepared_target_structure_rows(targets)
    if target_rows.empty:
        st.info("No readable prepared, imported, cropped, mutated, benchmark, or split-fragment target structures are available yet.")
        return None, None, []
    filter_cols = st.columns([1, 4])
    category_options = _target_category_options(str(value) for value in target_rows["category"].dropna().unique() if str(value))
    selected_categories = filter_cols[0].multiselect(
        "Category",
        category_options,
        default=category_options,
        key=f"{key_prefix}_target_categories",
    )
    search_text = filter_cols[1].text_input(
        "Search targets",
        value="",
        placeholder="Target name, chain, source, or job code",
        key=f"{key_prefix}_target_search",
    )
    filtered_rows = target_rows.copy()
    if selected_categories:
        filtered_rows = filtered_rows[filtered_rows["category"].astype(str).isin(selected_categories)]
    if search_text.strip():
        needle = search_text.strip().lower()
        filtered_rows = filtered_rows[
            filtered_rows.apply(lambda row: needle in " ".join(str(value).lower() for value in row.values), axis=1)
        ]
    if filtered_rows.empty:
        st.info("No targets match the current filters.")
        return None, None, []
    selected_path_state = str(st.session_state.get(f"{key_prefix}_selected_target_pdb") or "")
    selected_chains_state = set(st.session_state.get(f"{key_prefix}_selected_target_chains") or [])
    table_rows = filtered_rows.copy()
    if selected_path_state:
        table_rows["select"] = table_rows.apply(
            lambda row: str(row["path"]) == selected_path_state
            and (
                not selected_chains_state
                or set(str(row.get("chains") or "").split(",")) == selected_chains_state
            ),
            axis=1,
        )
    if not bool(table_rows["select"].any()):
        first_path = str(table_rows.iloc[0]["path"])
        table_rows["select"] = table_rows["path"].astype(str) == first_path
    edited_targets = st.data_editor(
        table_rows.drop(columns=["_target_index"]),
        hide_index=True,
        width="stretch",
        height=290,
        disabled=[
            "target",
            "chains",
            "chain_count",
            "aa_length",
            "fragments",
            "breaks",
            "source",
            "category",
            "job",
            "records",
            "ppi_hotspot",
            "path",
        ],
        column_config={
            "select": st.column_config.CheckboxColumn("Select"),
            "chains": st.column_config.TextColumn("Chains", width="small"),
            "chain_count": st.column_config.NumberColumn("Chain count", format="%d"),
            "aa_length": st.column_config.NumberColumn("AA", format="%d"),
            "fragments": st.column_config.NumberColumn("Fragments", format="%d"),
            "breaks": st.column_config.NumberColumn("Breaks", format="%d"),
            "ppi_hotspot": st.column_config.TextColumn("PPI/hotspot runs", width="medium"),
            "path": st.column_config.TextColumn("PDB path", width="large"),
        },
        key=f"{key_prefix}_target_structure_table",
    )
    selected_visible = edited_targets[edited_targets["select"] == True] if "select" in edited_targets.columns else pd.DataFrame()
    if selected_visible.empty:
        st.info("Select one target structure.")
        return None, None, []
    selected_row = selected_visible.iloc[0]
    if len(selected_visible) > 1:
        st.warning("Multiple target structures were selected; using the first selected structure.")
    matches = table_rows[
        (table_rows["target"].astype(str) == str(selected_row.get("target")))
        & (table_rows["path"].astype(str) == str(selected_row.get("path")))
    ]
    if matches.empty:
        return None, None, []
    source = targets[int(matches.iloc[0]["_target_index"])]
    target = dict(source)
    target_pdb = Path(str(selected_row.get("path") or ""))
    selected_chains = [chain.strip() for chain in str(selected_row.get("chains") or "").split(",") if chain.strip()]
    if not selected_chains:
        try:
            selected_chains = _structure_chains(target_pdb)
        except Exception:
            selected_chains = list(target.get("chains") or [])
    st.session_state[f"{key_prefix}_selected_target_pdb"] = str(target_pdb)
    st.session_state[f"{key_prefix}_selected_target_chains"] = selected_chains
    st.caption(f"Target PDB: `{target_pdb}`")
    return target, target_pdb, selected_chains


def _capacity_hotspot_controls(target_pdb: Path | None, selected_chains: list[str], *, key_prefix: str) -> str:
    if target_pdb is None or not target_pdb.exists() or not selected_chains:
        return ""
    st.markdown("**Target hotspot selection**")
    st.caption(
        "Select target residues for hotspot-aware generators and downstream hotspot/contact metrics. "
        "Use detection-guided suggestions when available, or click residues in the Mol* viewer."
    )
    hotspot_key = f"{key_prefix}_hotspots_{target_pdb}_{','.join(selected_chains)}"
    hotspot_text_key = f"{hotspot_key}_text"
    st.session_state.setdefault(hotspot_key, set())
    if isinstance(st.session_state.get(hotspot_key), str):
        st.session_state[hotspot_key] = _hotspot_residues(str(st.session_state.get(hotspot_key) or ""))
    st.session_state.setdefault(hotspot_text_key, _hotspot_text(set(st.session_state[hotspot_key])))
    typed_hotspots = {
        residue for residue in _hotspot_residues(str(st.session_state.get(hotspot_text_key) or "")) if residue[0] in set(selected_chains)
    }
    if typed_hotspots != set(st.session_state[hotspot_key]):
        st.session_state[hotspot_key] = typed_hotspots
    active_hotspots = {
        residue for residue in set(st.session_state[hotspot_key]) if residue[0] in set(selected_chains)
    }
    target_text = target_pdb.read_text(errors="ignore")
    target_view_text = filter_pdb_text(target_text, keep_chains=set(selected_chains))
    sources = detection_score_sources(target_pdb, selected_chains)
    detected_residues: set[tuple[str, int]] = set()
    show_existing_hotspots = False
    detection_view_token = "none"
    detection_preview_structures: list[StructureVisualization] | None = None
    detection_preview_key = ""
    preview_tool_color: str | None = None
    preview_sequence_indices: list[int] = []
    viewer_square_height = "min(44vw, 620px)"

    with st.expander("Detection-guided hotspots", expanded=bool(sources)):
        if not sources:
            st.info(
                "No PPI/hotspot detection results were found for this selected target. "
                "You can still choose hotspots manually in the Mol* selector below."
            )
        else:
            source_names = st.multiselect(
                "Detection results",
                list(sources),
                default=list(sources),
                key=f"{hotspot_key}_sources",
            )
            threshold_cols = st.columns(2)
            prediction_threshold = threshold_cols[0].number_input(
                "Prediction threshold",
                0.0,
                1.0,
                0.50,
                0.01,
                key=f"{hotspot_key}_prediction_threshold",
            )
            masif_threshold = threshold_cols[1].number_input(
                "MaSIF score threshold",
                0.0,
                1.0,
                0.50,
                0.01,
                key=f"{hotspot_key}_masif_threshold",
            )
            sasa_threshold = st.number_input(
                "Minimum side-chain SASA (A2)",
                0.0,
                500.0,
                5.0,
                1.0,
                key=f"{hotspot_key}_sasa_threshold",
            )
            show_existing_hotspots = st.checkbox("Also show existing hotspots", value=False, key=f"{hotspot_key}_existing")
            scores = (
                pd.concat([sources[name].assign(source=name) for name in source_names], ignore_index=True)
                if source_names
                else pd.DataFrame(columns=["chain", "residue", "amino_acid", "score", "source"])
            )
            scores["tool"] = scores["source"].astype(str).str.split(" | ").str[0]
            scores["score_threshold"] = scores["tool"].map(lambda tool: masif_threshold if tool == "masif_seed" else prediction_threshold)
            try:
                sasa = sidechain_sasa_by_residue(target_pdb)
            except Exception:
                sasa = {}
            scores["sidechain_sasa_a2"] = [sasa.get((str(row.chain), int(row.residue)), 0.0) for row in scores.itertuples()]
            scores["passes_score_threshold"] = scores["score"] >= scores["score_threshold"]
            scores["solvent_facing"] = scores["sidechain_sasa_a2"] >= sasa_threshold
            selected_scores = scores[scores["passes_score_threshold"] & scores["solvent_facing"]].copy()
            agreement = selected_scores.groupby(["chain", "residue"], as_index=False).agg(
                detected_by_tools=("tool", lambda values: ", ".join(sorted(set(values)))),
                tool_count=("tool", "nunique"),
            )
            selected_scores = selected_scores.merge(agreement, on=["chain", "residue"], how="left")
            if not scores.empty:
                summary_rows = []
                for source, source_scores in scores.groupby("source", sort=True):
                    all_residues = source_scores.drop_duplicates(["chain", "residue"])
                    score_pass = source_scores[source_scores["passes_score_threshold"]].drop_duplicates(["chain", "residue"])
                    suggested = source_scores[source_scores["passes_score_threshold"] & source_scores["solvent_facing"]].drop_duplicates(["chain", "residue"])
                    summary_rows.append(
                        {
                            "source": source,
                            "residues scored": len(all_residues),
                            "above score threshold": len(score_pass),
                            "surface-accessible suggestions": len(suggested),
                            "best score": round(float(source_scores["score"].max()), 3) if len(source_scores) else None,
                        }
                    )
                st.markdown("**Detection summary**")
                st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")
            if not selected_scores.empty:
                residue_summary = selected_scores.groupby(["chain", "residue"], as_index=False).agg(
                    amino_acid=("amino_acid", lambda values: next((str(value) for value in values if str(value).strip()), "")),
                    best_score=("score", "max"),
                    sidechain_sasa_a2=("sidechain_sasa_a2", "max"),
                    tools=("tool", lambda values: ", ".join(sorted(set(str(value) for value in values)))),
                    tool_count=("tool", "nunique"),
                )
                residue_summary = residue_summary.sort_values(
                    ["tool_count", "best_score", "sidechain_sasa_a2"],
                    ascending=[False, False, False],
                )
                residue_summary["best_score"] = residue_summary["best_score"].round(3)
                residue_summary["sidechain_sasa_a2"] = residue_summary["sidechain_sasa_a2"].round(1)
                metric_cols = st.columns(4)
                metric_cols[0].metric("Suggested residues", len(residue_summary))
                metric_cols[1].metric("Consensus residues", int((residue_summary["tool_count"] > 1).sum()))
                metric_cols[2].metric("Tools", len(set(selected_scores["tool"].dropna().astype(str))))
                metric_cols[3].metric("Selected chains", len(set(selected_scores["chain"].dropna().astype(str))))
                st.markdown("**Top hotspot suggestions**")
                st.dataframe(
                    residue_summary.head(20).rename(
                        columns={
                            "amino_acid": "AA",
                            "best_score": "best score",
                            "sidechain_sasa_a2": "side-chain SASA (A2)",
                            "tool_count": "tool count",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
            selected_scores["include"] = True
            edited_scores = st.data_editor(selected_scores, hide_index=True, width="stretch", key=f"{hotspot_key}_rows")
            include_mask = edited_scores["include"].fillna(False) if "include" in edited_scores else pd.Series(False, index=edited_scores.index)
            checked_scores = edited_scores[include_mask].copy()
            detected_residues = {(str(row.chain), int(row.residue)) for row in checked_scores.itertuples()}
            tool_options = sorted(set(checked_scores["tool"].dropna().astype(str))) if "tool" in checked_scores else []
            preview_tools = st.multiselect(
                "Preview tools",
                tool_options,
                default=tool_options,
                key=f"{hotspot_key}_preview_tools",
                help="Choose which checked PPI/hotspot calculations are shown in the preview and sequence strip.",
            )
            preview_scores = checked_scores[checked_scores["tool"].astype(str).isin(preview_tools)].copy() if preview_tools else checked_scores.iloc[0:0].copy()
            preview_residues = {(str(row.chain), int(row.residue)) for row in preview_scores.itertuples()}
            detection_view_token = (
                f"{','.join(source_names)}:{prediction_threshold:.2f}:{masif_threshold:.2f}:"
                f"{sasa_threshold:.1f}:{','.join(preview_tools)}"
            )
            if st.button("Add selected predicted residues as hotspots", key=f"{hotspot_key}_add"):
                st.session_state[hotspot_key] = active_hotspots | detected_residues
                st.session_state[hotspot_text_key] = _hotspot_text(set(st.session_state[hotspot_key]))
                st.rerun()
            _detection_tool_legend(set(preview_scores["tool"].dropna().astype(str)) if "tool" in preview_scores else set())
            st.caption(
                f"{len(detected_residues)} checked residues will be added as hotspots; "
                f"{len(preview_residues)} residues from the selected preview tools are highlighted."
            )
            detection_chains = _detection_tool_chains(preview_scores) or []
            preview_tool_color = None
            if len(preview_tools) == 1:
                preview_tool_color = "#" + DETECTION_TOOL_STYLES.get(
                    preview_tools[0],
                    {"color": "0x2563eb"},
                )["color"].replace("0x", "")
            elif preview_tools:
                preview_tool_color = "#2563eb"
            preview_sequence_indices = sorted({int(residue) for _chain, residue in preview_residues})
            if show_existing_hotspots:
                detection_chains.extend(_hotspot_chains(active_hotspots) or [])
            detection_preview_structures = [
                StructureVisualization(
                    pdb=target_view_text,
                    color="chain-id",
                    color_params={"palette": "pastel-1"},
                    representation_type="cartoon",
                    highlighted_selections=None,
                    chains=detection_chains or None,
                )
            ]
            detection_preview_key = f"{hotspot_key}_detection_viewer_{detection_view_token}"

    st.caption("Campaign hotspot selector")
    main_viewer_structures = [
        StructureVisualization(
            pdb=target_view_text,
            color="chain-id",
            color_params={"palette": "pastel-1"},
            representation_type="cartoon",
            highlighted_selections=[f"{chain}{residue}" for chain, residue in sorted(active_hotspots)],
            chains=_hotspot_chains(active_hotspots),
        )
    ]
    if detection_preview_structures:
        preview_col, select_col = st.columns(2)
        with preview_col:
            st.caption("PPI/hotspot preview")
            molstar_custom_component(
                detection_preview_structures,
                key=detection_preview_key,
                height=viewer_square_height,
                show_controls=True,
                selection_mode=False,
                sequence_highlight_color=preview_tool_color,
                sequence_highlight_indices=preview_sequence_indices,
                force_reload=True,
            )
        with select_col:
            st.caption("Campaign hotspot selection")
            viewer_value = molstar_custom_component(
                main_viewer_structures,
                key=f"{hotspot_key}_target_viewer_{detection_view_token}",
                height=viewer_square_height,
                show_controls=True,
                selection_mode=True,
            )
    else:
        viewer_value = molstar_custom_component(
            main_viewer_structures,
            key=f"{hotspot_key}_target_viewer_{detection_view_token}",
            height=560,
            show_controls=True,
            selection_mode=True,
        )
    clicked_hotspots = {
        residue for residue in _viewer_residues(viewer_value, target_view_text) if residue[0] in set(selected_chains)
    }
    if clicked_hotspots and not clicked_hotspots.issubset(active_hotspots):
        active_hotspots = active_hotspots | clicked_hotspots
        st.session_state[hotspot_key] = active_hotspots
        st.session_state[hotspot_text_key] = _hotspot_text(active_hotspots)
    hotspot_buttons = st.columns([1, 1, 1, 3])
    if hotspot_buttons[0].button("Add clicked", disabled=not clicked_hotspots, key=f"{hotspot_key}_add_clicked"):
        st.session_state[hotspot_key] = active_hotspots | clicked_hotspots
        st.session_state[hotspot_text_key] = _hotspot_text(set(st.session_state[hotspot_key]))
        st.rerun()
    if hotspot_buttons[1].button("Remove clicked", disabled=not clicked_hotspots, key=f"{hotspot_key}_remove_clicked"):
        st.session_state[hotspot_key] = active_hotspots - clicked_hotspots
        st.session_state[hotspot_text_key] = _hotspot_text(set(st.session_state[hotspot_key]))
        st.rerun()
    if hotspot_buttons[2].button("Clear", disabled=not active_hotspots, key=f"{hotspot_key}_clear"):
        st.session_state[hotspot_key] = set()
        st.session_state[hotspot_text_key] = ""
        st.rerun()
    hotspot_buttons[3].caption(f"Clicked: `{_hotspot_text(clicked_hotspots) or 'none'}`")
    hotspot_input = st.text_input(
        "Hotspots",
        placeholder="A40,A99,A107",
        key=hotspot_text_key,
    )
    parsed_hotspots = {
        residue for residue in _hotspot_residues(hotspot_input) if residue[0] in set(selected_chains)
    }
    if parsed_hotspots != active_hotspots:
        st.session_state[hotspot_key] = parsed_hotspots
        active_hotspots = parsed_hotspots
    normalized = _hotspot_text(active_hotspots)
    if normalized:
        st.caption(f"Selected hotspots: `{normalized}`")
    else:
        st.caption("No hotspots selected. These capacity jobs will run without hotspots; Genie3 will be skipped.")
    return normalized


def _selected_target_length(target_pdb: Path | None, selected_chains: list[str]) -> int | None:
    if target_pdb is None or not target_pdb.exists():
        return None
    try:
        from mn_protein_design.core.structures import pdb_summary

        summary = pdb_summary(target_pdb.read_text(errors="ignore"))
        selected = set(selected_chains)
        return sum(
            int(row.get("residue_count") or len(row.get("residues") or []) or 0)
            for row in summary.get("chains") or []
            if not selected or row.get("chain_id") in selected
        )
    except Exception:
        return None


def _source_chain_length(target_pdb: Path | None, selected_chains: list[str]) -> int | None:
    if target_pdb is None or not target_pdb.exists() or not selected_chains:
        return None
    try:
        from mn_protein_design.workflows.refolding import _sequences_by_chain

        sequence = str(_sequences_by_chain(target_pdb).get(selected_chains[0]) or "").replace("X", "")
        return len(sequence) or None
    except Exception:
        return None


def _show_matrix(matrix_rows: list[dict[str, object]], *, sequence_copy: bool = False) -> None:
    matrix_df = pd.DataFrame(matrix_rows)
    if sequence_copy and not matrix_df.empty:
        matrix_df = matrix_df.rename(columns={"total_length": "total_sequence_length"})
        matrix_df = matrix_df[
            [
                col
                for col in ["test", "source_chain", "sequence_length", "copy_count", "total_sequence_length"]
                if col in matrix_df.columns
            ]
        ]
    st.dataframe(matrix_df, hide_index=True, width="stretch")
    if matrix_rows:
        totals = [row["total_length"] for row in matrix_rows if row.get("total_length") is not None]
        st.caption(f"{len(matrix_rows)} systems" + (f" | total residues {min(totals)}-{max(totals)}" if totals else ""))


def _settings_dataframe(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "setting": str(row.get("setting") or ""),
                "value": "" if row.get("value") is None else str(row.get("value")),
            }
            for row in rows
        ]
    )


def _tag_design_capacity_child(
    run_dir: Path,
    *,
    batch_id: str,
    benchmark_name: str,
    generator: str,
    binder_length: int,
    target_name: str,
) -> None:
    metadata_path = run_dir / "metadata.json"
    metadata = read_json(metadata_path)
    metadata.update(
        {
            "capacity_kind": "design_generator",
            "capacity_batch_id": batch_id,
            "capacity_benchmark_name": benchmark_name,
            "capacity_generator": generator,
            "capacity_generator_label": DESIGN_GENERATOR_LABELS.get(generator, generator),
            "capacity_binder_length": int(binder_length),
            "capacity_target_name": target_name,
        }
    )
    write_json(metadata_path, metadata)
    input_path = run_dir / "input.json"
    payload = read_json(input_path)
    params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
    params.update(
        {
            "capacity_kind": "design_generator",
            "capacity_batch_id": batch_id,
            "capacity_benchmark_name": benchmark_name,
            "capacity_generator": generator,
            "capacity_binder_length": int(binder_length),
            "capacity_target_name": target_name,
        }
    )
    payload["params"] = params
    write_json(input_path, payload)


def _design_capacity_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for job in collect_jobs("design-campaign", include_hidden=True):
        run_dir = Path(str(job.get("run_dir") or ""))
        payload = read_json(run_dir / "input.json")
        params = payload.get("params") if isinstance(payload.get("params"), dict) else {}
        campaign_name = str(params.get("campaign_name") or job.get("campaign_name") or "")
        parsed_benchmark = ""
        parsed_generator = ""
        parsed_length = ""
        if " | " in campaign_name:
            parts = [part.strip() for part in campaign_name.split(" | ")]
            if len(parts) >= 3 and parts[-1].startswith("L") and parts[-1][1:].isdigit():
                parsed_benchmark = " | ".join(parts[:-2])
                parsed_generator = parts[-2]
                parsed_length = parts[-1][1:]
        is_tagged = str(params.get("capacity_kind") or "") == "design_generator"
        is_legacy_capacity_name = bool(parsed_benchmark and parsed_generator and parsed_length)
        if not is_tagged and not is_legacy_capacity_name:
            continue
        metadata = read_json(run_dir / "metadata.json")
        result = read_json(run_dir / "result.json")
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        inputs = payload.get("inputs") if isinstance(payload.get("inputs"), dict) else {}
        generator = str(params.get("capacity_generator") or "")
        generator_label = DESIGN_GENERATOR_LABELS.get(generator, generator) if generator else parsed_generator
        benchmark = str(params.get("capacity_benchmark_name") or parsed_benchmark or job.get("campaign_name") or "")
        batch_id = str(params.get("capacity_batch_id") or f"legacy:{benchmark}")
        target_pdb = inputs.get("target_pdb") or ""
        target_chains = [str(chain) for chain in (inputs.get("target_chains") or []) if str(chain)]
        target_length = params.get("capacity_target_length")
        if target_length in {None, ""}:
            target_length = _selected_target_length(Path(str(target_pdb)).expanduser(), target_chains) if str(target_pdb).strip() else None
        binder_length = params.get("capacity_binder_length") or parsed_length
        total_length = params.get("capacity_total_length")
        if total_length in {None, ""}:
            try:
                total_length = int(target_length) + int(binder_length) if target_length not in {None, ""} and binder_length not in {None, ""} else None
            except (TypeError, ValueError):
                total_length = None
        rows.append(
            {
                "result": result_link(str(job.get("task_group") or "design-campaign"), str(job.get("run_id") or ""), "Open"),
                "batch_id": batch_id,
                "benchmark": benchmark or batch_id,
                "status": job.get("status") or "",
                "target": params.get("capacity_target_name") or "",
                "generator": generator_label,
                "generator_key": generator,
                "binder_length": binder_length,
                "target_length": target_length,
                "total_length": total_length,
                "candidate_count": metrics.get("candidate_count", ""),
                "target_pdb": target_pdb,
                "target_chains": ",".join(target_chains),
                "phase": job.get("current_phase") or "",
                "job_code": job.get("job_code") or "",
                "run_id": job.get("run_id") or "",
                "run_dir": str(run_dir),
                "created_at": job.get("created_at") or "",
                "worker_started_at": metadata.get("worker_started_at") or "",
                "queue_started_at": metadata.get("queue_started_at") or "",
                "completed_at": metadata.get("completed_at") or "",
                "updated_at": job.get("updated_at") or "",
            }
        )
    return rows


def _read_structure_text(path: Path) -> str:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", errors="ignore").read()
    return path.read_text(errors="ignore")


def _cif_atom_rows(text: str) -> list[dict[str, str]]:
    atom_headers: list[str] = []
    in_atom_loop = False
    rows: list[dict[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "loop_":
            atom_headers = []
            in_atom_loop = False
            continue
        if line.startswith("_atom_site."):
            atom_headers.append(line.split(".", 1)[1])
            in_atom_loop = True
            continue
        if not in_atom_loop:
            continue
        if line.startswith("_"):
            in_atom_loop = False
            continue
        if not line.startswith(("ATOM ", "HETATM ")):
            continue
        parts = line.split()
        if len(parts) < len(atom_headers):
            continue
        rows.append(dict(zip(atom_headers, parts)))
    return rows


def _pdb_chain_ids_for_cif_chains(chains: list[str]) -> dict[str, str]:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    used: set[str] = set()
    mapping: dict[str, str] = {}
    for chain in chains:
        preferred = chain[:1] if chain[:1] and chain[:1] not in used else ""
        if not preferred:
            preferred = next((candidate for candidate in alphabet if candidate not in used), "_")
        mapping[chain] = preferred
        used.add(preferred)
    return mapping


def _cif_to_pdb_text(text: str) -> str:
    rows = _cif_atom_rows(text)
    chain_map = _pdb_chain_ids_for_cif_chains(
        list(
            dict.fromkeys(
                (row.get("auth_asym_id") or row.get("label_asym_id") or "_").strip("\"'")
                for row in rows
            )
        )
    )
    lines: list[str] = []
    for serial, row in enumerate(rows, start=1):
        record = (row.get("group_PDB") or "ATOM").strip("\"'")[:6]
        atom_name = (row.get("auth_atom_id") or row.get("label_atom_id") or "").strip("\"'")
        residue_name = (row.get("auth_comp_id") or row.get("label_comp_id") or "UNK").strip("\"'")[:3].upper()
        original_chain = (row.get("auth_asym_id") or row.get("label_asym_id") or "_").strip("\"'")
        chain_id = chain_map.get(original_chain, original_chain[:1] or "_")
        insertion = (row.get("pdbx_PDB_ins_code") or "").strip("\"'")
        insertion = " " if insertion in {"", ".", "?"} else insertion[:1]
        try:
            residue_id = int(float((row.get("auth_seq_id") or row.get("label_seq_id") or "0").strip("\"'")))
            x = float(row.get("Cartn_x") or row.get("pdbx_model_Cartn_x"))
            y = float(row.get("Cartn_y") or row.get("pdbx_model_Cartn_y"))
            z = float(row.get("Cartn_z") or row.get("pdbx_model_Cartn_z"))
        except (TypeError, ValueError):
            continue
        try:
            occupancy = float((row.get("occupancy") or "1.0").strip("\"'"))
        except ValueError:
            occupancy = 1.0
        try:
            b_factor = float((row.get("B_iso_or_equiv") or "0.0").strip("\"'"))
        except ValueError:
            b_factor = 0.0
        element = (row.get("type_symbol") or atom_name[:1] or "").strip("\"'")[:2].upper()
        lines.append(
            f"{record:<6}{serial % 100000:5d} {atom_name:<4.4s} {residue_name:>3s} {chain_id:1s}"
            f"{residue_id:4d}{insertion:1s}   {x:8.3f}{y:8.3f}{z:8.3f}"
            f"{occupancy:6.2f}{b_factor:6.2f}          {element:>2s}"
        )
    return "\n".join(lines) + ("\nEND\n" if lines else "")


def _parse_structure_file(path: Path):
    from Bio.PDB import MMCIFParser, PDBParser

    if path.suffix.lower() in {".cif", ".mmcif"} or path.name.endswith((".cif.gz", ".mmcif.gz")):
        text = _read_structure_text(path)
        try:
            structure = MMCIFParser(QUIET=True).get_structure(path.stem, StringIO(text))
            model = next(structure.get_models(), None)
            if model is not None and any(len(str(getattr(chain, "id", ""))) > 1 for chain in model):
                return PDBParser(QUIET=True).get_structure(path.stem, StringIO(_cif_to_pdb_text(text)))
            return structure
        except Exception:
            return PDBParser(QUIET=True).get_structure(path.stem, StringIO(_cif_to_pdb_text(text)))
    return PDBParser(QUIET=True).get_structure(path.stem, str(path))


def _structure_to_pdb_text(structure: object) -> str:
    import warnings
    from Bio.PDB import PDBIO
    from Bio.PDB.PDBExceptions import PDBIOWarning

    output = StringIO()
    writer = PDBIO()
    writer.set_structure(structure)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBIOWarning)
        writer.save(output)
    return output.getvalue()


def _pdb_display_chain_id(chain_id: str) -> str:
    return str(chain_id or "")[:1]


def _filter_pdb_chains_text(pdb_text: str, keep_chains: list[str]) -> str:
    keep = {str(chain)[:1] for chain in keep_chains if str(chain)}
    if not keep:
        return pdb_text
    lines: list[str] = []
    for line in pdb_text.splitlines():
        if line.startswith(("ATOM  ", "HETATM", "TER   ")):
            if len(line) > 21 and line[21].strip() not in keep:
                continue
        lines.append(line)
    return "\n".join(lines) + ("\n" if lines else "")


def _pdb_is_ca_only_trace(pdb_text: str) -> bool:
    atom_names = {
        line[12:16].strip()
        for line in pdb_text.splitlines()
        if line.startswith(("ATOM  ", "HETATM")) and len(line) >= 16
    }
    return bool(atom_names) and atom_names <= {"CA"}


def _hotspot_metric(metrics: dict[str, Any], key: str) -> object:
    atom_key = f"hotspot_atom_{key}"
    ca_key = f"hotspot_ca_{key}"
    value = metrics.get(atom_key)
    if value not in {None, ""}:
        return value
    return metrics.get(ca_key)


def _hotspot_contact_label(metrics: dict[str, Any]) -> str:
    contacted = _hotspot_metric(metrics, "contacted_count")
    total = _hotspot_metric(metrics, "count")
    if total in {None, ""}:
        return ""
    return f"{contacted or 0}/{total}"


def _hotspot_contacted_tokens(metrics: dict[str, Any]) -> str:
    return str(_hotspot_metric(metrics, "contacted") or "")


_AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def _chain_ca_sequence_atoms(chain: object) -> tuple[str, list[object]]:
    sequence = ""
    atoms: list[object] = []
    for residue in chain:
        residue_id = getattr(residue, "id", ("", None, ""))
        if residue_id[0] != " " or "CA" not in residue:
            continue
        sequence += _AA3_TO_1.get(str(getattr(residue, "resname", "")).upper(), "X")
        atoms.append(residue["CA"])
    return sequence, atoms


def _model_chain_id(model: object, chain_id: str) -> str | None:
    if chain_id in model:
        return chain_id
    if chain_id and chain_id[:1] in model:
        return chain_id[:1]
    for model_chain in model:
        model_chain_id = str(getattr(model_chain, "id", ""))
        if model_chain_id and (model_chain_id.startswith(chain_id) or chain_id.startswith(model_chain_id)):
            return model_chain_id
    return None


def _sequence_target_ca_pairs(
    fixed_structure: object,
    moving_structure: object,
    fixed_chains: list[str],
    moving_chains: list[str],
) -> tuple[list[object], list[object]]:
    from Bio.Align import PairwiseAligner

    fixed_model = next(fixed_structure.get_models(), None)
    moving_model = next(moving_structure.get_models(), None)
    if fixed_model is None or moving_model is None:
        return [], []
    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -0.5
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5
    fixed_atoms: list[object] = []
    moving_atoms: list[object] = []
    for fixed_chain, moving_chain in zip(fixed_chains, moving_chains):
        fixed_chain_id = _model_chain_id(fixed_model, fixed_chain)
        moving_chain_id = _model_chain_id(moving_model, moving_chain)
        if fixed_chain_id is None or moving_chain_id is None:
            continue
        fixed_sequence, fixed_chain_atoms = _chain_ca_sequence_atoms(fixed_model[fixed_chain_id])
        moving_sequence, moving_chain_atoms = _chain_ca_sequence_atoms(moving_model[moving_chain_id])
        if not fixed_sequence or not moving_sequence:
            continue
        try:
            alignment = aligner.align(fixed_sequence, moving_sequence)[0]
        except Exception:
            continue
        for fixed_block, moving_block in zip(alignment.aligned[0], alignment.aligned[1]):
            fixed_start, fixed_end = int(fixed_block[0]), int(fixed_block[1])
            moving_start, moving_end = int(moving_block[0]), int(moving_block[1])
            block_count = min(fixed_end - fixed_start, moving_end - moving_start)
            for offset in range(block_count):
                fixed_atoms.append(fixed_chain_atoms[fixed_start + offset])
                moving_atoms.append(moving_chain_atoms[moving_start + offset])
    return fixed_atoms, moving_atoms


def _positional_target_ca_pairs(
    fixed_structure: object,
    moving_structure: object,
    fixed_chains: list[str],
    moving_chains: list[str],
) -> tuple[list[object], list[object]]:
    fixed_model = next(fixed_structure.get_models(), None)
    moving_model = next(moving_structure.get_models(), None)
    if fixed_model is None or moving_model is None:
        return [], []
    fixed_atoms: list[object] = []
    moving_atoms: list[object] = []
    for fixed_chain, moving_chain in zip(fixed_chains, moving_chains):
        fixed_chain_id = _model_chain_id(fixed_model, fixed_chain)
        moving_chain_id = _model_chain_id(moving_model, moving_chain)
        if fixed_chain_id is None or moving_chain_id is None:
            continue
        _fixed_sequence, fixed_chain_atoms = _chain_ca_sequence_atoms(fixed_model[fixed_chain_id])
        _moving_sequence, moving_chain_atoms = _chain_ca_sequence_atoms(moving_model[moving_chain_id])
        for index in range(min(len(fixed_chain_atoms), len(moving_chain_atoms))):
            fixed_atoms.append(fixed_chain_atoms[index])
            moving_atoms.append(moving_chain_atoms[index])
    return fixed_atoms, moving_atoms


def _target_ca_pairs(fixed_structure: object, moving_structure: object, fixed_chains: list[str], moving_chains: list[str]) -> tuple[list[object], list[object]]:
    fixed_model = next(fixed_structure.get_models(), None)
    moving_model = next(moving_structure.get_models(), None)
    if fixed_model is None or moving_model is None:
        return [], []
    fixed_atoms: list[object] = []
    moving_atoms: list[object] = []
    for fixed_chain, moving_chain in zip(fixed_chains, moving_chains):
        fixed_chain_id = _model_chain_id(fixed_model, fixed_chain)
        moving_chain_id = _model_chain_id(moving_model, moving_chain)
        if fixed_chain_id is None or moving_chain_id is None:
            continue
        moving_by_residue: dict[tuple[int, str], object] = {}
        for residue in moving_model[moving_chain_id]:
            residue_id = getattr(residue, "id", ("", None, ""))
            if residue_id[0] == " " and "CA" in residue:
                moving_by_residue[(int(residue_id[1]), str(residue_id[2]).strip())] = residue["CA"]
        for residue in fixed_model[fixed_chain_id]:
            residue_id = getattr(residue, "id", ("", None, ""))
            if residue_id[0] != " " or "CA" not in residue:
                continue
            moving_atom = moving_by_residue.get((int(residue_id[1]), str(residue_id[2]).strip()))
            if moving_atom is not None:
                fixed_atoms.append(residue["CA"])
                moving_atoms.append(moving_atom)
    sequence_fixed_atoms, sequence_moving_atoms = _sequence_target_ca_pairs(
        fixed_structure,
        moving_structure,
        fixed_chains,
        moving_chains,
    )
    if len(sequence_fixed_atoms) >= len(fixed_atoms):
        return sequence_fixed_atoms, sequence_moving_atoms
    if len(fixed_atoms) >= 3:
        return fixed_atoms, moving_atoms
    return _positional_target_ca_pairs(fixed_structure, moving_structure, fixed_chains, moving_chains)


def _aligned_capacity_structure_text(path: Path, reference_structure: object, reference_target_chains: list[str], moving_target_chains: list[str]) -> tuple[str | None, float | None, int]:
    from Bio.PDB import Superimposer

    try:
        moving_structure = _parse_structure_file(path)
        fixed_atoms, moving_atoms = _target_ca_pairs(reference_structure, moving_structure, reference_target_chains, moving_target_chains)
        atom_count = min(len(fixed_atoms), len(moving_atoms))
        if atom_count < 3:
            return None, None, atom_count
        superimposer = Superimposer()
        superimposer.set_atoms(fixed_atoms[:atom_count], moving_atoms[:atom_count])
        superimposer.apply(moving_structure.get_atoms())
        return _structure_to_pdb_text(moving_structure), float(superimposer.rms), atom_count
    except Exception:
        return None, None, 0


def _resolve_capacity_candidate_path(run_dir: Path, candidate: dict[str, Any], key: str = "complex_pdb") -> Path | None:
    raw_metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
    roots = [run_dir]
    source_run_dir = raw_metadata.get("source_run_dir")
    if source_run_dir:
        source_path = Path(str(source_run_dir)).expanduser()
        if source_path.exists():
            roots.insert(0, source_path)
    value = candidate.get(key)
    if not value:
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute() and path.exists():
        return path
    for root in roots:
        candidate_path = root / path
        if candidate_path.exists():
            return candidate_path
    return None


def _capacity_design_candidates(child_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, child in child_df.iterrows():
        run_dir = Path(str(child.get("run_dir") or ""))
        if not run_dir.exists():
            continue
        for candidate in read_candidates(run_dir):
            metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
            metadata = candidate.get("raw_metadata") if isinstance(candidate.get("raw_metadata"), dict) else {}
            rows.append(
                {
                    "batch_run_dir": run_dir,
                    "generator": child.get("generator"),
                    "generator_key": child.get("generator_key"),
                    "status": child.get("status"),
                    "binder_length": child.get("binder_length"),
                    "job_code": child.get("job_code"),
                    "run_id": child.get("run_id"),
                    "candidate_id": candidate.get("candidate_id"),
                    "candidate": candidate,
                    "metrics": metrics,
                    "metadata": metadata,
                    "structure_path": _resolve_capacity_candidate_path(run_dir, candidate, "complex_pdb"),
                    "target_path": _resolve_capacity_candidate_path(run_dir, candidate, "target_pdb"),
                    "original_target_path": Path(str(child.get("target_pdb") or "")).expanduser()
                    if str(child.get("target_pdb") or "").strip()
                    else None,
                    "original_target_chains": [
                        chain.strip()
                        for chain in str(child.get("target_chains") or "").split(",")
                        if chain.strip()
                    ],
                }
            )
    return rows


def _timestamp_seconds(start: object, end: object) -> float | None:
    try:
        start_ts = pd.to_datetime(start, utc=True, errors="coerce")
        end_ts = pd.to_datetime(end, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(start_ts) or pd.isna(end_ts):
        return None
    return float((end_ts - start_ts).total_seconds())


def _design_cell_runtime_seconds(row: pd.Series) -> float | None:
    status = str(row.get("status") or "")
    start = row.get("worker_started_at") or row.get("queue_started_at") or row.get("created_at")
    if not start:
        return None
    if status in {"queued", "skipped"} and not row.get("worker_started_at"):
        return None
    end = row.get("completed_at") or row.get("updated_at")
    if status in {"running", "preparing"}:
        end = pd.Timestamp.now(tz="UTC").isoformat()
    return _timestamp_seconds(start, end)


def _design_capacity_matrix_value(row: pd.Series) -> str:
    status = str(row.get("status") or "").strip().lower() or "unknown"
    seconds = _design_cell_runtime_seconds(row)
    duration = "" if seconds is None else f" · {format_duration(seconds)}"
    candidate_count = row.get("candidate_count")
    try:
        candidate_count_int = int(float(str(candidate_count)))
    except (TypeError, ValueError):
        candidate_count_int = None
    if status == "completed":
        if candidate_count_int == 0:
            return f"completed, 0 designs{duration}"
        return f"passed{duration}"
    if status == "failed":
        return f"failed{duration}"
    if status == "skipped":
        return "skipped"
    if status in {"running", "preparing"}:
        return f"{status}{duration}"
    if status == "queued":
        return "queued"
    return f"{status}{duration}"


def _py3dmol_design_capacity_matrix_html(
    panels: list[dict[str, object]],
    *,
    div_id: str,
    height: int,
) -> str:
    panel_count = max(1, len(panels))
    cols = 2 if panel_count <= 4 else 3
    rows = max(1, (panel_count + cols - 1) // cols)
    js_url = "https://cdn.jsdelivr.net/npm/3dmol@2.5.5/build/3Dmol-min.js"
    panels_json = json.dumps(panels)
    return f"""
<div id="{div_id}" style="width:100%; height:{height}px; position:relative; border:1px solid #d8dee9; border-radius:6px; overflow:hidden;"></div>
<div style="display:flex; flex-wrap:wrap; gap:14px; margin:8px 0 0 0; font:12px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
  <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#d1d5db;margin-right:5px;"></span>target reference</div>
  <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#facc15;margin-right:5px;"></span>configured hotspot</div>
  <div><span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:#22c55e;margin-right:5px;"></span>contacted hotspot</div>
</div>
<script>
(function() {{
  const panels = {panels_json};
  const rows = {rows};
  const cols = {cols};
  const jsUrl = {json.dumps(js_url)};
  function loadScriptAsync(uri) {{
    return new Promise((resolve, reject) => {{
      if (window.$3Dmol) {{ resolve(); return; }}
      const tag = document.createElement('script');
      tag.src = uri;
      tag.async = true;
      tag.onload = resolve;
      tag.onerror = reject;
      document.head.appendChild(tag);
    }});
  }}
  function styleHotspots(model, residues, color) {{
    for (const residue of residues || []) {{
      const selection = {{chain: residue.chain, resi: residue.resi}};
      model.setStyle(selection, {{
        cartoon: {{color: '#d1d5db', opacity: 0.55}},
        stick: {{color: color, radius: 0.22}}
      }});
    }}
  }}
  function caTraceAtoms(pdbText) {{
    const atoms = [];
    const lines = (pdbText || '').split(/\\r?\\n/);
    for (const line of lines) {{
      if (!line.startsWith('ATOM') && !line.startsWith('HETATM')) continue;
      const atomName = line.slice(12, 16).trim();
      if (atomName !== 'CA') continue;
      const chain = line.slice(21, 22).trim();
      const resi = parseInt(line.slice(22, 26).trim(), 10);
      const x = parseFloat(line.slice(30, 38).trim());
      const y = parseFloat(line.slice(38, 46).trim());
      const z = parseFloat(line.slice(46, 54).trim());
      if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) {{
        atoms.push({{chain, resi, x, y, z}});
      }}
    }}
    atoms.sort((a, b) => a.chain.localeCompare(b.chain) || a.resi - b.resi);
    return atoms;
  }}
  function addCaTraceSticks(viewer, pdbText, color) {{
    const atoms = caTraceAtoms(pdbText);
    for (const atom of atoms) {{
      viewer.addSphere({{
        center: {{x: atom.x, y: atom.y, z: atom.z}},
        radius: 0.34,
        color: color
      }});
    }}
    for (let idx = 1; idx < atoms.length; idx += 1) {{
      const previous = atoms[idx - 1];
      const current = atoms[idx];
      if (previous.chain !== current.chain) continue;
      viewer.addCylinder({{
        start: {{x: previous.x, y: previous.y, z: previous.z}},
        end: {{x: current.x, y: current.y, z: current.z}},
        radius: 0.13,
        color: color,
        fromCap: 1,
        toCap: 1
      }});
    }}
  }}
  loadScriptAsync(jsUrl).then(function() {{
    const container = document.getElementById({json.dumps(div_id)});
    if (!container) return;
    container.innerHTML = '';
    const grid = $3Dmol.createViewerGrid(container, {{rows: rows, cols: cols, control_all: true}}, {{backgroundColor: 'white'}});
    for (let idx = 0; idx < panels.length; idx += 1) {{
      const row = Math.floor(idx / cols);
      const col = idx % cols;
      const viewer = grid[row][col];
      const panel = panels[idx];
      const targetModel = viewer.addModel(panel.reference_pdb || '', 'pdb');
      targetModel.setStyle({{}}, {{cartoon: {{color: '#d1d5db', opacity: 0.55}}}});
      styleHotspots(targetModel, panel.configured_hotspots, '#facc15');
      styleHotspots(targetModel, panel.contacted_hotspots, '#22c55e');
      const binderModel = viewer.addModel(panel.binder_pdb || '', 'pdb');
      if (panel.binder_ca_only) {{
        binderModel.setStyle({{}}, {{sphere: {{color: panel.color || '#2563eb', scale: 0.18}}}});
        addCaTraceSticks(viewer, panel.binder_pdb || '', panel.color || '#2563eb');
      }} else {{
        binderModel.setStyle({{}}, {{cartoon: {{color: panel.color || '#2563eb', opacity: 0.94}}}});
      }}
      viewer.zoomTo();
      viewer.render();
    }}
  }}).catch(function() {{
    const container = document.getElementById({json.dumps(div_id)});
    if (container) {{
      container.innerHTML = '<p style="padding:12px; background:#fff3cd; color:#664d03;">3Dmol.js could not be loaded. Check browser network access to jsDelivr.</p>';
    }}
  }});
}})();
</script>
<div style="display:grid; grid-template-columns:repeat({cols}, 1fr); gap:6px; margin-top:6px; font:12px system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color:#374151;">
  {''.join('<div style="text-align:center; font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">' + str(panel.get("title") or "") + '</div>' for panel in panels)}
</div>
"""


def _design_capacity_plot_status(row: pd.Series) -> str:
    status = str(row.get("status") or "").strip().lower() or "unknown"
    if status == "completed":
        candidate_count = row.get("candidate_count")
        try:
            if int(float(str(candidate_count))) == 0:
                return "completed, 0 designs"
        except (TypeError, ValueError):
            pass
        return "passed"
    return status


def _numeric_metric(value: object) -> float | None:
    try:
        if value in {None, ""}:
            return None
        parsed = float(str(value))
    except (TypeError, ValueError):
        return None
    return parsed if pd.notna(parsed) else None


def _int_metric(value: object) -> int:
    parsed = _numeric_metric(value)
    return int(parsed) if parsed is not None else 0


def _render_design_capacity_status_matrix(child_df: pd.DataFrame) -> None:
    if child_df.empty:
        return
    matrix_df = child_df.copy()
    matrix_df["binder_length_numeric"] = pd.to_numeric(matrix_df["binder_length"], errors="coerce")
    matrix_df = matrix_df.dropna(subset=["binder_length_numeric"])
    if matrix_df.empty:
        return
    lengths = sorted({int(value) for value in matrix_df["binder_length_numeric"].dropna()})
    length_labels: dict[int, str] = {}
    for length in lengths:
        length_rows = matrix_df[matrix_df["binder_length_numeric"].astype(int) == int(length)]
        total_values = [
            _int_metric(value)
            for value in length_rows.get("total_length", pd.Series(dtype=object)).tolist()
            if _int_metric(value)
        ]
        total_label = min(total_values) if total_values and len(set(total_values)) == 1 else None
        length_labels[int(length)] = f"L{length} ({total_label} aa)" if total_label else f"L{length}"
    rows: list[dict[str, object]] = []
    plot_rows: list[dict[str, object]] = []
    for generator, group in matrix_df.groupby("generator", sort=True):
        group = group.sort_values("binder_length_numeric")
        row: dict[str, object] = {"generator": generator}
        for length in lengths:
            column_label = length_labels[int(length)]
            matches = group[group["binder_length_numeric"].astype(int) == int(length)]
            if matches.empty:
                row[column_label] = ""
                continue
            match = matches.iloc[0]
            total_length = _int_metric(match.get("total_length"))
            seconds = _design_cell_runtime_seconds(match)
            duration = "" if seconds is None else format_duration(seconds)
            row[column_label] = _design_capacity_matrix_value(match)
            plot_rows.append(
                {
                    "generator": generator,
                    "binder_length": int(length),
                    "target_length": _int_metric(match.get("target_length")) or None,
                    "total_length": total_length or None,
                    "status": _design_capacity_plot_status(match),
                    "duration": duration,
                    "runtime_seconds": seconds,
                    "candidate_count": match.get("candidate_count"),
                    "job_code": match.get("job_code"),
                    "phase": match.get("phase"),
                }
            )
        rows.append(row)
    st.markdown("**Generator-Length Capacity Matrix**")
    st.caption("Each cell shows whether that binder length passed, failed, is queued/running, or was skipped, plus design runtime when available.")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if plot_rows:
        plot_df = pd.DataFrame(plot_rows)
        status_order = ["passed", "completed, 0 designs", "failed", "skipped", "running", "preparing", "queued", "unknown"]
        status_colors = ["#16a34a", "#84cc16", "#dc2626", "#9ca3af", "#f59e0b", "#fbbf24", "#d1d5db", "#64748b"]
        base = alt.Chart(plot_df).encode(
            x=alt.X("binder_length:O", title="binder length", sort=lengths),
            y=alt.Y("generator:N", title="generator", sort=sorted(plot_df["generator"].dropna().unique().tolist())),
        )
        heatmap = base.mark_rect().encode(
            color=alt.Color(
                "status:N",
                title="status",
                scale=alt.Scale(domain=status_order, range=status_colors),
            ),
            tooltip=[
                alt.Tooltip("generator:N", title="generator"),
                alt.Tooltip("binder_length:O", title="binder length"),
                alt.Tooltip("target_length:Q", title="target length"),
                alt.Tooltip("total_length:Q", title="total length"),
                alt.Tooltip("status:N", title="status"),
                alt.Tooltip("duration:N", title="runtime"),
                alt.Tooltip("candidate_count:N", title="candidates"),
                alt.Tooltip("job_code:N", title="job"),
                alt.Tooltip("phase:N", title="phase"),
            ],
        )
        labels = base.mark_text(fontSize=11, color="#111827").encode(
            text=alt.Text("duration:N"),
        )
        st.altair_chart((heatmap + labels).properties(height=max(280, 34 * plot_df["generator"].nunique())), width="stretch")


def _render_design_capacity_batch_summary(child_df: pd.DataFrame, selected_batch: str) -> None:
    _render_design_capacity_status_matrix(child_df)
    candidate_rows = _capacity_design_candidates(child_df)
    if not candidate_rows:
        return
    st.markdown("**Candidate / Hotspot Details**")
    summary_rows = []
    for row in candidate_rows:
        metrics = row["metrics"]
        summary_rows.append(
            {
                "generator": row.get("generator"),
                "status": row.get("status"),
                "candidate": row.get("candidate_id"),
                "binder_length": row.get("binder_length"),
                "hotspot contacts": _hotspot_contact_label(metrics),
                "hotspot fraction": _hotspot_metric(metrics, "contact_fraction"),
                "hotspot min distance": _hotspot_metric(metrics, "min_distance"),
                "contacted hotspots": _hotspot_contacted_tokens(metrics),
                "missing hotspots": _hotspot_metric(metrics, "missing") or "",
                "binder atoms": metrics.get("binder_atom_count_for_hotspot_contacts")
                or metrics.get("binder_ca_count_for_hotspot_contacts"),
                "contact mode": metrics.get("hotspot_contact_mode") or "CA fallback",
                "structure": str(row.get("structure_path") or ""),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, hide_index=True, width="stretch")

    hotspots = sorted(
        {
            str(token)
            for row in candidate_rows
            for token in ((row.get("candidate") or {}).get("hotspots") or [])
            if str(token)
        }
    )
    if hotspots:
        st.markdown("**Hotspot Contact Matrix**")
        matrix_rows = []
        for row in candidate_rows:
            metrics = row["metrics"]
            contacted = {token for token in _hotspot_contacted_tokens(metrics).split(",") if token}
            matrix_row: dict[str, object] = {
                "generator": row.get("generator"),
                "candidate": row.get("candidate_id"),
            }
            for hotspot in hotspots:
                matrix_row[hotspot] = 1 if hotspot in contacted else 0
            matrix_rows.append(matrix_row)
        matrix_df = pd.DataFrame(matrix_rows)
        st.dataframe(matrix_df, hide_index=True, width="stretch")

    structure_rows = [row for row in candidate_rows if row.get("structure_path") is not None]
    if not structure_rows:
        return
    st.markdown("**Target-Aligned Design Overlay**")
    selector_rows: list[dict[str, object]] = []
    for index, row in enumerate(structure_rows):
        metrics = row.get("metrics") or {}
        selector_rows.append(
            {
                "row_index": index,
                "label": f"{row.get('generator')} | L{row.get('binder_length')} | {row.get('candidate_id')}",
                "generator": row.get("generator"),
                "binder_length": _int_metric(row.get("binder_length")),
                "candidate": row.get("candidate_id"),
                "hotspot_contacts": _hotspot_contact_label(metrics),
                "contacted_count": _int_metric(_hotspot_metric(metrics, "contacted_count")),
                "hotspot_fraction": _numeric_metric(_hotspot_metric(metrics, "contact_fraction")),
                "hotspot_min_distance": _numeric_metric(_hotspot_metric(metrics, "min_distance")),
                "target_rmsd_hint": _numeric_metric(metrics.get("target_alignment_rmsd")),
                "status": row.get("status"),
            }
        )
    selector_df = pd.DataFrame(selector_rows)
    selector_lengths = sorted({int(value) for value in selector_df["binder_length"].dropna() if int(value) > 0})
    default_length = selector_lengths[-1] if selector_lengths else None
    filter_cols = st.columns([1.2, 2.2, 1.6])
    selected_lengths = filter_cols[0].multiselect(
        "Binder lengths",
        selector_lengths,
        default=[default_length] if default_length is not None else [],
        key=f"{selected_batch}_design_capacity_overlay_lengths",
    )
    generator_options = sorted(selector_df["generator"].dropna().astype(str).unique().tolist())
    selected_generators = filter_cols[1].multiselect(
        "Generators",
        generator_options,
        default=generator_options,
        key=f"{selected_batch}_design_capacity_overlay_generators",
    )
    rank_mode = filter_cols[2].selectbox(
        "Sort by",
        ["hotspot contacts", "hotspot distance", "binder length", "generator"],
        index=0,
        key=f"{selected_batch}_design_capacity_overlay_sort",
    )
    filtered_selector = selector_df.copy()
    if selected_lengths:
        filtered_selector = filtered_selector[filtered_selector["binder_length"].isin(selected_lengths)]
    if selected_generators:
        filtered_selector = filtered_selector[filtered_selector["generator"].astype(str).isin(selected_generators)]
    if rank_mode == "hotspot contacts":
        filtered_selector = filtered_selector.sort_values(
            ["contacted_count", "hotspot_fraction", "hotspot_min_distance", "binder_length", "generator"],
            ascending=[False, False, True, True, True],
            na_position="last",
        )
    elif rank_mode == "hotspot distance":
        filtered_selector = filtered_selector.sort_values(
            ["hotspot_min_distance", "contacted_count", "binder_length", "generator"],
            ascending=[True, False, True, True],
            na_position="last",
        )
    elif rank_mode == "binder length":
        filtered_selector = filtered_selector.sort_values(["binder_length", "generator"], ascending=[True, True])
    else:
        filtered_selector = filtered_selector.sort_values(["generator", "binder_length"], ascending=[True, True])
    selected_labels = filtered_selector["label"].tolist()
    selected_lookup = dict(zip(selector_df["label"], selector_df["row_index"]))
    selected_rows = [
        structure_rows[int(selected_lookup[label])]
        for label in selected_labels
        if label in selected_lookup
    ]
    selector_display = filtered_selector[
        [
            "label",
            "status",
            "hotspot_contacts",
            "hotspot_fraction",
            "hotspot_min_distance",
            "target_rmsd_hint",
        ]
    ].rename(
        columns={
            "label": "design",
            "hotspot_contacts": "hotspot contacts",
            "hotspot_fraction": "hotspot fraction",
            "hotspot_min_distance": "hotspot min distance",
            "target_rmsd_hint": "target RMSD hint",
        }
    )
    st.dataframe(
        selector_display,
        hide_index=True,
        width="stretch",
    )
    st.caption(f"Mol* overlay includes all {len(selected_rows)} designs shown in the selector table.")
    if not selected_rows:
        st.info("No designs match the current Mol* overlay filters.")
        return
    overlay_hotspots = {
        residue
        for row in selected_rows
        for residue in _hotspot_residues(
            ",".join(str(token) for token in ((row.get("candidate") or {}).get("hotspots") or []) if str(token))
        )
    }
    contacted_overlay_hotspots = {
        residue
        for row in selected_rows
        for residue in _hotspot_residues(_hotspot_contacted_tokens(row.get("metrics") or {}))
    }
    reference_row = next((row for row in selected_rows if row.get("original_target_path") is not None), selected_rows[0])
    reference_path = reference_row.get("original_target_path") or reference_row.get("target_path") or reference_row.get("structure_path")
    if reference_path is None:
        return
    reference_target_chains = [
        str(chain)
        for chain in (
            reference_row.get("original_target_chains")
            or (reference_row["candidate"].get("target_chains") or [])
        )
        if str(chain)
    ]
    if not reference_target_chains:
        st.info("Target chain metadata is missing, so target-aligned overlay cannot be built.")
        return
    try:
        reference_structure = _parse_structure_file(Path(reference_path))
        reference_text = _read_structure_text(Path(reference_path))
    except Exception as exc:
        st.info(f"Could not read target reference for overlay: {exc}")
        return
    reference_chains = [
        ChainVisualization(
            chain_id=chain,
            color="uniform",
            color_params={"value": "0xd1d5db"},
            representation_type="cartoon",
        )
        for chain in reference_target_chains
    ]
    reference_chains.extend(
        _hotspot_chains(
            overlay_hotspots,
            color="0xfacc15",
            label="Configured hotspots",
        )
        or []
    )
    reference_chains.extend(
        _hotspot_chains(
            contacted_overlay_hotspots,
            color="0x22c55e",
            label="Contacted hotspots",
        )
        or []
    )
    structures: list[StructureVisualization] = [
        StructureVisualization(
            pdb=reference_text,
            color="uniform",
            color_params={"value": "0xd1d5db"},
            representation_type="cartoon",
            chains=reference_chains,
        )
    ]
    reference_target_text = _filter_pdb_chains_text(
        reference_text,
        [_pdb_display_chain_id(chain) for chain in reference_target_chains],
    )
    if not reference_target_text.strip():
        reference_target_text = reference_text
    configured_hotspot_payload = [
        {"chain": chain, "resi": residue}
        for chain, residue in sorted(overlay_hotspots)
    ]
    contacted_hotspot_payload = [
        {"chain": chain, "resi": residue}
        for chain, residue in sorted(contacted_overlay_hotspots)
    ]
    colors = ["0x2563eb", "0xef4444", "0xd97706", "0x7c3aed", "0x059669", "0xdb2777", "0x0891b2", "0xbe123c"]
    overlay_rows = []
    grid_panels: list[dict[str, object]] = []
    for index, row in enumerate(selected_rows):
        candidate = row["candidate"]
        structure_path = row.get("structure_path")
        if structure_path is None:
            continue
        target_chains = [str(chain) for chain in (candidate.get("target_chains") or []) if str(chain)]
        binder_chains = [str(chain) for chain in (candidate.get("binder_chains") or []) if str(chain)]
        aligned_text, rmsd, atom_count = _aligned_capacity_structure_text(
            Path(structure_path),
            reference_structure,
            reference_target_chains,
            target_chains or reference_target_chains,
        )
        metrics = row["metrics"]
        if aligned_text is None:
            overlay_rows.append(
                {
                    "generator": row.get("generator"),
                    "candidate": row.get("candidate_id"),
                    "target RMSD": None,
                    "aligned target CA": atom_count,
                    "hotspot contacts": _hotspot_contact_label(metrics),
                    "contacted hotspots": _hotspot_contacted_tokens(metrics),
                    "alignment": "not shown; target alignment failed",
                    "structure": str(structure_path),
                }
            )
            continue
        display_binder_chains = [_pdb_display_chain_id(chain) for chain in binder_chains]
        structure_text = _filter_pdb_chains_text(aligned_text, display_binder_chains)
        color = colors[index % len(colors)]
        web_color = f"#{color[2:]}" if color.startswith("0x") else color
        structures.append(
            StructureVisualization(
                pdb=structure_text,
                color="uniform",
                color_params={"value": color},
                representation_type="cartoon",
            )
        )
        overlay_rows.append(
            {
                "generator": row.get("generator"),
                "candidate": row.get("candidate_id"),
                "target RMSD": rmsd,
                "aligned target CA": atom_count,
                "hotspot contacts": _hotspot_contact_label(metrics),
                "contacted hotspots": _hotspot_contacted_tokens(metrics),
                "alignment": "shown",
                "structure": str(structure_path),
            }
        )
        if structure_text.strip():
            grid_panels.append(
                {
                    "title": f"{row.get('generator')} | L{row.get('binder_length')} | {row.get('candidate_id')}",
                    "reference_pdb": reference_target_text,
                    "binder_pdb": structure_text,
                    "color": web_color,
                    "binder_ca_only": _pdb_is_ca_only_trace(structure_text),
                    "configured_hotspots": configured_hotspot_payload,
                    "contacted_hotspots": contacted_hotspot_payload,
                }
            )
    overlay_tab, matrix_tab, table_tab = st.tabs(["Mol* Overlay", "3Dmol Matrix", "Alignment Table"])
    with overlay_tab:
        molstar_custom_component(
            structures,
            key=(
                f"{selected_batch}_design_capacity_overlay_molstar_"
                f"{hashlib.sha1('|'.join(selected_labels).encode()).hexdigest()[:12]}"
            ),
            height=620,
            show_controls=True,
            selection_mode=False,
        )
        st.caption(
            "Target reference is the original capacity input target in light gray. "
            "Configured hotspots are yellow, contacted hotspots are green. Each selected design is sequence-aligned on "
            "target CA atoms; only binder chains are overlaid, with one color per design."
        )
    with matrix_tab:
        if grid_panels:
            grid_height = max(360, 310 * ((len(grid_panels) + 2) // 3))
            components.html(
                _py3dmol_design_capacity_matrix_html(
                    grid_panels,
                    div_id=(
                        f"{selected_batch}_design_capacity_py3dmol_grid_"
                        f"{hashlib.sha1('|'.join(selected_labels).encode()).hexdigest()[:12]}"
                    ),
                    height=grid_height,
                ),
                height=grid_height + 92,
                scrolling=False,
            )
            st.caption(
                "Each panel shows the same target reference and one aligned binder, with linked camera controls. "
                "Configured hotspots are yellow and contacted hotspots are green."
            )
        else:
            st.info("No aligned binder structures are available for the 3Dmol matrix.")
    with table_tab:
        if overlay_rows:
            st.dataframe(pd.DataFrame(overlay_rows), hide_index=True, width="stretch")


def _design_capacity_batch_table(df: pd.DataFrame) -> pd.DataFrame:
    batches = []
    for batch_id, group in df.groupby("batch_id", dropna=False):
        completed = int((group["status"].astype(str) == "completed").sum())
        failed = int((group["status"].astype(str) == "failed").sum())
        skipped = int((group["status"].astype(str) == "skipped").sum())
        generators = sorted(set(group["generator"].astype(str)))
        batches.append(
            {
                "open": _design_capacity_batch_url(batch_id),
                "batch_id": batch_id,
                "benchmark": group["benchmark"].iloc[0],
                "target": group["target"].iloc[0],
                "cells": len(group),
                "completed": completed,
                "failed": failed,
                "skipped": skipped,
                "running/queued": int(group["status"].astype(str).isin({"queued", "running", "preparing"}).sum()),
                "generator_count": len(generators),
                "generators": ", ".join(generators),
                "binder_lengths": ", ".join(str(value) for value in sorted(set(pd.to_numeric(group["binder_length"], errors="coerce").dropna().astype(int)))),
                "created_at": group["created_at"].min(),
            }
        )
    return pd.DataFrame(batches).sort_values("created_at", ascending=False)


def _design_capacity_child_table(df: pd.DataFrame, selected_batch: str) -> pd.DataFrame:
    child_df = df[df["batch_id"].astype(str) == selected_batch].copy()
    child_df["binder_length_numeric"] = pd.to_numeric(child_df["binder_length"], errors="coerce")
    child_df = child_df.sort_values(["binder_length_numeric", "generator"])
    for display_col in ["candidate_count", "binder_length"]:
        if display_col in child_df.columns:
            child_df[display_col] = child_df[display_col].map(lambda value: "" if pd.isna(value) else str(value))
    return child_df


def _render_design_capacity_batch_detail(
    df: pd.DataFrame,
    batch_df: pd.DataFrame,
    selected_batch: str,
    *,
    back_url: str = "?",
) -> None:
    batch_match = batch_df[batch_df["batch_id"].astype(str) == selected_batch]
    if batch_match.empty:
        st.warning(f"Design capacity batch not found: {selected_batch}")
        st.link_button("Back to batch list", back_url)
        return
    batch_row = batch_match.iloc[0]
    st.subheader("Design Capacity Batch")
    st.link_button("Back to batch list", back_url)
    st.caption(str(selected_batch))
    summary_cols = st.columns(5)
    summary_cols[0].metric("Cells", str(batch_row.get("cells") or 0))
    summary_cols[1].metric("Completed", str(batch_row.get("completed") or 0))
    summary_cols[2].metric("Failed", str(batch_row.get("failed") or 0))
    summary_cols[3].metric("Skipped", str(batch_row.get("skipped") or 0))
    summary_cols[4].metric("Running/queued", str(batch_row.get("running/queued") or 0))
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "benchmark": batch_row.get("benchmark"),
                    "target": batch_row.get("target"),
                    "binder_lengths": batch_row.get("binder_lengths"),
                    "generators": batch_row.get("generators"),
                    "created_at": batch_row.get("created_at"),
                }
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.markdown("**Generator Cells**")
    child_df = _design_capacity_child_table(df, selected_batch)
    st.dataframe(
        child_df[
            [
                col
                for col in [
                    "result",
                    "status",
                    "target",
                    "generator",
                    "binder_length",
                    "target_length",
                    "total_length",
                    "candidate_count",
                    "phase",
                    "job_code",
                    "run_id",
                    "updated_at",
                ]
                if col in child_df.columns
            ]
        ],
        hide_index=True,
        width="stretch",
        column_config={"result": st.column_config.LinkColumn("result", display_text="Open")},
    )
    _render_design_capacity_batch_summary(child_df, selected_batch)


def _render_design_capacity_results() -> None:
    st.subheader("Results")
    rows = _design_capacity_rows()
    if not rows:
        st.info("No design capacity runs yet.")
        return
    df = pd.DataFrame(rows)
    batch_df = _design_capacity_batch_table(df)
    refresh_results_button("design_capacity_refresh_results")
    st.caption("Open a batch on its dedicated results page to inspect generator cells, hotspot matrix, and target-aligned overlay.")
    display_cols = [
        "open",
        "batch_id",
        "benchmark",
        "target",
        "cells",
        "completed",
        "failed",
        "skipped",
        "running/queued",
        "generator_count",
        "binder_lengths",
        "created_at",
    ]
    st.dataframe(
        batch_df[[col for col in display_cols if col in batch_df.columns]],
        hide_index=True,
        width="stretch",
        key="design_capacity_batch_table",
        column_config={"open": st.column_config.LinkColumn("open", display_text="Open batch")},
    )


def _render_target_chain_inventory(target_chain_table: pd.DataFrame, *, expanded: bool) -> None:
    if target_chain_table.empty:
        return
    with st.expander("Available target chains", expanded=expanded):
        st.dataframe(
            target_chain_table[
                [
                    col
                    for col in [
                        "target",
                        "chain",
                        "aa_length",
                        "source",
                        "target_msa",
                        "af3_msa",
                        "colabfold_msa",
                        "boltz_msa",
                        "rf3_msa",
                        "openfold3_msa",
                        "protenix_msa",
                        "protenix_v1_msa",
                        "protenix_v2_msa",
                        "esmfold2_msa",
                        "af2_target_template",
                        "job_code",
                        "msa_path",
                        "target_pdb",
                    ]
                    if col in target_chain_table.columns
                ]
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Imported PDB rows show MSA-ready only when their exact chain sequence already exists in the "
            "sequence-hashed A3M cache. Prepared/cropped rows often already have MSAs."
        )


def _target_panel_selection(target_chain_table: pd.DataFrame) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    st.caption(
        "Select target chains to test as independent systems. Set copy_count to 1 for monomer capacity, "
        "or higher to make a repeated-chain multimer for that target."
    )
    if target_chain_table.empty:
        st.info("No target-chain rows are available yet.")
        return [], []
    panel_df = target_chain_table.copy()
    panel_df["row_key"] = panel_df["target_pdb"].astype(str) + "::" + panel_df["chain"].astype(str)
    selected_state = st.session_state.get("capacity_target_panel_selected", {})
    if isinstance(selected_state, dict):
        panel_df["copy_count"] = panel_df["row_key"].map(lambda key: int(selected_state.get(f"{key}::copy_count", 1) or 1))
    display_panel = panel_df[
        [
            col
            for col in [
                "target",
                "chain",
                "aa_length",
                "copy_count",
                "total_sequence_length",
                "source",
                "target_msa",
                "af3_msa",
                "colabfold_msa",
                "boltz_msa",
                "rf3_msa",
                "openfold3_msa",
                "protenix_msa",
                "protenix_v1_msa",
                "protenix_v2_msa",
                "esmfold2_msa",
                "af2_target_template",
                "job_code",
                "target_pdb",
                "row_key",
            ]
            if col in panel_df.columns
        ]
    ].copy()
    table_event = st.dataframe(
        display_panel,
        hide_index=True,
        width="stretch",
        key="capacity_target_panel_table",
        on_select="rerun",
        selection_mode="multi-row",
    )
    selected_indices = list((getattr(table_event, "selection", {}) or {}).get("rows") or [])
    selected_panel = display_panel.iloc[selected_indices].copy() if selected_indices else display_panel.iloc[0:0].copy()
    if not selected_panel.empty:
        selected_panel = st.data_editor(
            selected_panel[
                [
                    col
                    for col in ["target", "chain", "aa_length", "copy_count", "total_sequence_length", "source", "target_msa", "target_pdb", "row_key"]
                    if col in selected_panel.columns
                ]
            ],
            hide_index=True,
            width="stretch",
            key="capacity_target_panel_copy_count_editor",
            column_config={"copy_count": st.column_config.NumberColumn("copy_count", min_value=1, max_value=26, step=1)},
            disabled=[col for col in selected_panel.columns if col != "copy_count"],
        )
        selected_panel["copy_count"] = pd.to_numeric(selected_panel["copy_count"], errors="coerce").fillna(1).clip(lower=1, upper=26).astype(int)
        selected_panel["total_sequence_length"] = pd.to_numeric(selected_panel["aa_length"], errors="coerce").fillna(0).astype(int) * selected_panel["copy_count"]
        st.session_state["capacity_target_panel_selected"] = {
            f"{row['row_key']}::copy_count": int(row.get("copy_count") or 1)
            for row in selected_panel.to_dict(orient="records")
            if row.get("row_key")
        }
    target_panel_entries = [
        {
            "target_name": row.get("target"),
            "target_pdb": row.get("target_pdb"),
            "chain": row.get("chain"),
            "copy_count": int(row.get("copy_count") or 1),
        }
        for row in selected_panel.to_dict(orient="records")
    ]
    matrix_rows = [
        {
            "test": "target panel",
            "target": row.get("target"),
            "chain": row.get("chain"),
            "sequence_length": int(row.get("aa_length") or 0),
            "copy_count": int(row.get("copy_count") or 1),
            "total_length": int(row.get("total_sequence_length") or 0),
        }
        for row in selected_panel.to_dict(orient="records")
    ]
    return target_panel_entries, matrix_rows


def _render_refolding_results() -> None:
    st.subheader("Results")
    refresh_results_button("refolding_capacity_refresh_results")
    parents = capacity_parent_rows()
    if not parents:
        st.info("No capacity benchmark runs yet.")
        return

    st.markdown("**Benchmark Runs (Umbrella Jobs)**")
    st.caption("Each row owns one target, run depth, test matrix, scheduler, and its child engine jobs.")
    parent_df = pd.DataFrame(parents)
    display_cols = [
        "result",
        "benchmark",
        "status",
        "umbrella_type",
        "preset",
        "matrix_mode",
        "target",
        "target_chains",
        "sequence_length",
        "systems",
        "cells",
        "completed_cells",
        "failed_cells",
        "skipped_cells",
        "running_cells",
        "engines",
        "source_capacity_run_id",
        "created_at",
        "job_code",
        "run_id",
    ]
    event = st.dataframe(
        parent_df[[col for col in display_cols if col in parent_df.columns]],
        hide_index=True,
        width="stretch",
        column_config={"result": st.column_config.LinkColumn("result", display_text="Open")},
        key="capacity_parent_table",
        on_select="rerun",
        selection_mode="single-row",
    )
    st.caption("Select a row to inspect the matrix, plots, capacity profile, and additional run actions.")
    selected_indices = list((getattr(event, "selection", {}) or {}).get("rows") or [])
    selected_parent = parent_df.iloc[selected_indices[0]] if selected_indices else parent_df.iloc[0]
    selected_run_dir = Path(str(selected_parent["run_dir"]))
    children = capacity_child_rows(selected_run_dir)
    selected_payload = read_json(selected_run_dir / "input.json")
    selected_params = selected_payload.get("params") if isinstance(selected_payload.get("params"), dict) else {}
    selected_engine_keys = [str(engine) for engine in (selected_params.get("engines") or []) if str(engine)]
    selected_copy_counts = sorted(
        {
            int(float(str(value)))
            for value in (selected_params.get("copy_counts") or [])
            if str(value).strip()
        }
    )

    st.markdown(f"**Selected Umbrella: {selected_parent.get('benchmark', selected_run_dir.name)}**")
    selected_summary_cols = st.columns(5)
    selected_summary_cols[0].metric("Run depth", str(selected_params.get("preset") or ""))
    selected_summary_cols[1].metric("Systems", str(selected_parent.get("systems") or 0))
    selected_summary_cols[2].metric("Engine cells", str(selected_parent.get("cells") or 0))
    selected_summary_cols[3].metric("Completed", str(selected_parent.get("completed_cells") or 0))
    selected_summary_cols[4].metric("Failed", str(selected_parent.get("failed_cells") or 0))

    limit_rows = capacity_limit_rows(parent_run_dir=selected_run_dir)
    if limit_rows:
        with st.expander("Selected umbrella capacity profile", expanded=True):
            st.dataframe(
                pd.DataFrame(limit_rows),
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
            "Creates a new practical umbrella for the same target. Each engine starts at its own largest completed "
            "system from the selected capacity run."
        )
        practical_seeds: list[dict[str, object]] = []
        if children:
            child_seed_df = pd.DataFrame(children)
            child_seed_df["total_length_numeric"] = pd.to_numeric(child_seed_df["total_length"], errors="coerce")
            for _engine_key, group in child_seed_df.groupby("engine_key", dropna=True):
                valid = group.dropna(subset=["total_length_numeric"]).copy()
                completed = valid[valid["status"].astype(str) == "completed"]
                if completed.empty:
                    continue
                best = completed.sort_values("total_length_numeric").iloc[-1]
                practical_seeds.append(
                    {
                        "engine": best.get("engine"),
                        "copy_count": best.get("copy_count"),
                        "sequence_length": best.get("sequence_length"),
                        "total_sequence_length": int(best["total_length_numeric"]),
                        "source_run": best.get("run_id"),
                    }
                )
        if practical_seeds:
            st.dataframe(pd.DataFrame(practical_seeds), hide_index=True, width="stretch")
            practical_name = st.text_input(
                "Practical umbrella name",
                value=f"{selected_params.get('benchmark_name') or selected_run_dir.name} - practical",
                key=f"capacity_practical_name_{selected_run_dir.name}",
            )
            practical_gpu = st.text_input(
                "GPU device",
                value=str(selected_params.get("capacity_device") or "0"),
                key=f"capacity_practical_gpu_{selected_run_dir.name}",
            )
            practical_launch = st.checkbox("Launch practical cells now", value=True, key=f"capacity_practical_launch_{selected_run_dir.name}")
            if st.button("Create practical benchmark umbrella", type="primary", key=f"capacity_create_practical_{selected_run_dir.name}"):
                try:
                    practical_run = create_practical_capacity_benchmark_from_parent(
                        selected_run_dir,
                        benchmark_name=practical_name,
                        gpu_device=practical_gpu,
                        launch=practical_launch,
                    )
                    st.success(f"Practical umbrella created: {practical_run.name}")
                    show_pipeline_links(practical_run, [practical_run])
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        else:
            st.info("This umbrella has no completed cells available as practical starting points.")

    with st.expander("Extend or recalculate selected benchmark", expanded=False):
        st.caption("Adds cells to the selected umbrella and keeps the same parent result.")
        missing_engines = [engine for engine in ENGINE_LABELS if engine not in selected_engine_keys]
        add_engines = st.multiselect(
            "Add refolding engines",
            missing_engines,
            default=["boltzgen_fold"] if "boltzgen_fold" in missing_engines else [],
            format_func=lambda key: ENGINE_LABELS.get(key, key),
            key=f"capacity_add_engines_{selected_run_dir.name}",
        )
        next_copy = (max(selected_copy_counts) + 1) if selected_copy_counts else 2
        add_copy_text = st.text_input(
            "Add multimer copy counts",
            value="",
            placeholder=str(next_copy),
            key=f"capacity_add_copies_{selected_run_dir.name}",
            disabled=str(selected_params.get("matrix_mode") or "") != "sequence_copy_multimer",
        )
        terminal_children = [
            row
            for row in children
            if str(row.get("status") or "") in {"completed", "failed", "stopped", "skipped", "cancelled"}
        ]
        recalc_labels = {
            str(row.get("run_id") or ""): f"{row.get('engine')} | {row.get('total_length')} residues | {row.get('status')} | {row.get('job_code')}"
            for row in terminal_children
            if str(row.get("run_id") or "")
        }
        recalculate_ids = st.multiselect(
            "Recalculate selected cells",
            list(recalc_labels),
            format_func=lambda run_id: recalc_labels.get(run_id, run_id),
            key=f"capacity_recalculate_{selected_run_dir.name}",
            help="The old cell is retained for provenance but replaced in the active matrix.",
        )
        launch_extension = st.checkbox("Launch added cells now", value=True, key=f"capacity_launch_extension_{selected_run_dir.name}")
        parsed_copy_counts = _parse_lengths(add_copy_text)
        if st.button(
            "Apply additions / recalculations",
            type="primary",
            disabled=not (add_engines or parsed_copy_counts or recalculate_ids),
            key=f"capacity_extend_{selected_run_dir.name}",
        ):
            try:
                extension = extend_refolding_capacity_benchmark(
                    selected_run_dir,
                    engines_to_add=list(add_engines),
                    copy_counts_to_add=parsed_copy_counts,
                    recalculate_run_ids=list(recalculate_ids),
                    launch=launch_extension,
                )
                st.success(
                    f"Added {extension['added_systems']} systems, {extension['added_engines']} engines, "
                    f"{extension['added_cells']} new cells, and {extension['recalculated_cells']} recalculated cells."
                )
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

    st.markdown("**Selected Capacity Matrix**")
    if not children:
        st.info("This capacity run has no child jobs yet.")
        return
    child_df = pd.DataFrame(children)
    child_modes = set(child_df.get("matrix_mode", pd.Series(dtype=str)).dropna().astype(str))
    folding_only = child_modes <= {"sequence_copy_multimer", "target_panel"}
    if folding_only:
        child_cols = [
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
        child_display = child_df[[col for col in child_cols if col in child_df.columns]].copy()
        child_display = child_display.rename(columns={"total_length": "total_sequence_length"})
    else:
        child_cols = [
            "result",
            "engine",
            "status",
            "matrix_mode",
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
        child_display = child_df[[col for col in child_cols if col in child_df.columns]].copy()
    st.dataframe(
        child_display,
        hide_index=True,
        width="stretch",
        column_config={"result": st.column_config.LinkColumn("result", display_text="Open")},
    )

    plot_df = child_df.copy()
    plot_df["total_length"] = pd.to_numeric(plot_df["total_length"], errors="coerce")
    target_panel_only = child_modes <= {"target_panel"}
    if folding_only:
        plot_df["total_sequence_length"] = plot_df["total_length"]
    plot_df["target_label"] = (
        plot_df["target_name"].fillna("").astype(str)
        if "target_name" in plot_df.columns
        else pd.Series([""] * len(plot_df), index=plot_df.index, dtype=str)
    )
    if "candidate_id" in plot_df.columns:
        empty_target_label = plot_df["target_label"].str.strip() == ""
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
        x_encoding = alt.X("total_length:O", title="total residues")
    status_order = ["completed", "failed", "queued", "running", "preparing", "skipped", "cancelled", "paused"]
    status_colors = ["#0072CE", "#7CC1F2", "#FF2D2D", "#F4A3A8", "#F59E0B", "#9CA3AF", "#6B7280", "#A78BFA"]
    tooltip = (
        ["engine", "target_label", "sequence_length", "copy_count", "total_sequence_length", "peak_gpu_memory_mib", "gpu_total_memory_mib", "status", "failure_kind", "worker_error"]
        if folding_only
        else ["engine", "copy_count", "sequence_length", "target_length", "binder_length", "total_length", "peak_gpu_memory_mib", "gpu_total_memory_mib", "status", "failure_kind", "worker_error"]
    )
    chart = (
        alt.Chart(plot_df)
        .mark_rect()
        .encode(
            x=x_encoding,
            y=alt.Y("engine:N", title="engine"),
            color=alt.Color("status:N", title="status", scale=alt.Scale(domain=status_order, range=status_colors)),
            tooltip=tooltip,
        )
        .properties(width=720, height=720)
    )
    st.altair_chart(chart, width="content")


def _render_refolding_capacity() -> None:
    setup_tab, engines_tab, review_tab, results_tab = st.tabs(["Setup", "Engines", "Review & Run", "Results"])
    targets = ppi_target_jobs()
    target_chain_table = _prepared_target_chain_rows(targets)
    preset = "practical"

    with setup_tab:
        st.subheader("Refolding Engine Capacity")
        st.caption(
            "Use this to find how large a system each installed folding/refolding engine can handle on this GPU."
        )
        benchmark_name = st.text_input("Benchmark name", value="Refolding capacity check", key="refolding_capacity_name")
        matrix_mode = "target_panel"
        launch_now = st.checkbox("Launch child jobs immediately", value=True, key="refolding_capacity_launch")
        st.info("Target panel capacity uses selected prepared target chains and the normal refolding engine parameters.")

        target = None
        target_pdb = None
        selected_chains: list[str] = []
        target_lengths: list[int] = []
        binder_lengths: list[int] = []
        copy_counts: list[int] = []
        _render_target_chain_inventory(target_chain_table, expanded=True)
        target_panel_entries, matrix_rows = _target_panel_selection(target_chain_table)
        _show_matrix(matrix_rows)

    with engines_tab:
        st.subheader("Prediction / Refolding Engines")
        st.caption("These generate structures or confidence outputs. Capacity metrics are collected separately per engine-size cell.")
        engine_state_keys = {
            "alphafast_af3": "capacity_run_af3",
            "colabfold": "capacity_run_colab",
            "af2_initial_guess": "capacity_run_af2",
            "esmfold2": "capacity_run_esmfold2",
            "boltz2_initial_guess": "capacity_run_boltz2",
            "rf3": "capacity_run_rf3",
            "openfold3": "capacity_run_openfold3",
            "protenix": "capacity_run_protenix",
            "protenix_v1": "capacity_run_protenix_v1",
            "protenix_v2": "capacity_run_protenix_v2",
            "boltzgen_fold": "capacity_run_boltzgen",
        }
        if "capacity_engine_defaults_v2" not in st.session_state:
            for state_key in engine_state_keys.values():
                st.session_state[state_key] = True
            st.session_state["capacity_engine_defaults_v2"] = True
        for state_key in engine_state_keys.values():
            st.session_state.setdefault(state_key, True)

        template_msa_engine_keys = {
            "alphafast_af3",
            "colabfold",
            "boltz2_initial_guess",
            "rf3",
            "protenix_v1",
            "protenix_v2",
        }
        template_only_engine_keys = set(template_msa_engine_keys)
        template_only_engine_keys.add("af2_initial_guess")
        msa_engine_keys = {
            "alphafast_af3",
            "colabfold",
            "esmfold2",
            "boltz2_initial_guess",
            "rf3",
            "openfold3",
            "protenix",
            "protenix_v1",
            "protenix_v2",
        }
        st.session_state.setdefault("capacity_refolding_conditioning_mode", "default")
        bulk_cols = st.columns([1, 1, 1.5, 1.5, 1.35, 2.65])
        if bulk_cols[0].button("Select all engines", key="capacity_select_all"):
            for state_key in engine_state_keys.values():
                st.session_state[state_key] = True
            st.session_state["capacity_refolding_conditioning_mode"] = "default"
            st.rerun()
        if bulk_cols[1].button("Deselect all engines", key="capacity_deselect_all"):
            for state_key in engine_state_keys.values():
                st.session_state[state_key] = False
            st.session_state["capacity_refolding_conditioning_mode"] = "default"
            st.rerun()
        if bulk_cols[2].button("Template + MSA engines", key="capacity_select_template_msa_engines"):
            for engine_key, state_key in engine_state_keys.items():
                st.session_state[state_key] = engine_key in template_msa_engine_keys
            st.session_state["capacity_refolding_conditioning_mode"] = "template_msa"
            st.rerun()
        if bulk_cols[3].button("Template-only engines", key="capacity_select_template_only_engines"):
            for engine_key, state_key in engine_state_keys.items():
                st.session_state[state_key] = engine_key in template_only_engine_keys
            st.session_state["capacity_refolding_conditioning_mode"] = "template_only"
            st.rerun()
        if bulk_cols[4].button("MSA-only engines", key="capacity_select_msa_only_engines"):
            for engine_key, state_key in engine_state_keys.items():
                st.session_state[state_key] = engine_key in msa_engine_keys
            st.session_state["capacity_refolding_conditioning_mode"] = "msa_only"
            st.rerun()

        engine_cols = st.columns(len(engine_state_keys))
        for column, (engine_key, state_key) in zip(engine_cols, engine_state_keys.items()):
            with column:
                st.checkbox(ENGINE_LABELS.get(engine_key, engine_key), value=True, key=state_key)
        selected_engines = [
            engine_key
            for engine_key, state_key in engine_state_keys.items()
            if bool(st.session_state.get(state_key))
        ]

        st.segmented_control(
            "Prediction input contract",
            ["engine_specific"],
            selection_mode="single",
            default="engine_specific",
            disabled=True,
            key="capacity_input_mode_display",
            format_func={"engine_specific": "Capacity system + staged role contract"}.get,
            help="Child refolding jobs stage candidates through the normal role contract: targets A..., binders Z...",
        )
        st.caption(f"Selected engines will process {len(matrix_rows):,} capacity systems.")
        with st.expander("Normal refolding engine settings", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"engine": ENGINE_LABELS[key], "normal refolding settings": practical, "MSA use": msa_use}
                        for key, _capacity, practical, _full, msa_use in [
                            ("alphafast_af3", "1 recycle", "10 recycles", "10 recycles", "practical/full use cached target MSAs"),
                            ("colabfold", "1 model, 1 recycle", "3 models, 3 recycles", "3 models, 3 recycles", "practical/full use cached target MSAs"),
                            ("af2_initial_guess", "1 recycle", "3 recycles", "3 recycles", "target template; not shared-MSA driven"),
                            ("esmfold2", "1 loop, 16 steps", "10 loops, 68 steps", "10 loops, 68 steps", "practical/full use cached target MSAs"),
                            ("boltz2_initial_guess", "1 recycle, 20 steps, 1 sample", "10 recycles, 200 steps, 3 samples", "10 recycles, 200 steps, 3 samples", "practical/full use cached target MSAs"),
                            ("rf3", "2 recycles, 10 steps, 1 sample", "10 recycles, 50 steps, 5 samples", "10 recycles, 50 steps, 5 samples", "practical/full use cached target MSAs"),
                            ("openfold3", "1 recycle, 1 sample", "3 recycles, 5 samples", "3 recycles, 5 samples", "practical/full use cached target MSAs"),
                            ("protenix", "1 cycle, 10 steps, 1 sample", "3 cycles, 50 steps, 5 samples", "3 cycles, 50 steps, 5 samples", "practical/full use cached target MSAs"),
                            ("protenix_v1", "1 cycle, 10 steps, 1 sample", "3 cycles, 50 steps, 5 samples", "3 cycles, 50 steps, 5 samples", "practical/full use cached target MSAs"),
                            ("protenix_v2", "1 cycle, 10 steps, 1 sample", "3 cycles, 50 steps, 5 samples", "3 cycles, 50 steps, 5 samples", "practical/full use cached target MSAs"),
                            ("boltzgen_fold", "1 recycle, 20 steps, 1 sample", "3 recycles, 200 steps, 5 samples", "3 recycles, 200 steps, 5 samples", "coordinate/template conditioned; target MSA not central"),
                        ]
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

    with review_tab:
        st.subheader("Review & Run")
        gpu_device = gpu_run_panel(key="capacity_benchmark", default="0")
        cells = len(matrix_rows) * len(selected_engines)
        st.dataframe(
            _settings_dataframe(
                [
                    {"setting": "Benchmark", "value": benchmark_name},
                    {"setting": "Engine parameters", "value": "Normal refolding settings"},
                    {"setting": "GPU", "value": gpu_device},
                    {"setting": "Systems", "value": len(matrix_rows)},
                    {"setting": "Engines", "value": len(selected_engines)},
                    {"setting": "Child jobs / cells", "value": cells},
                    {"setting": "Launch immediately", "value": "yes" if launch_now else "no"},
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        run_disabled = not matrix_rows or not selected_engines
        if matrix_mode == "sequence_copy_multimer" and (target_pdb is None or not target_pdb.exists() or not selected_chains):
            run_disabled = True
            st.warning("Select a readable target and source chain before running.")
        if matrix_mode == "target_panel" and not target_panel_entries:
            run_disabled = True
            st.warning("Select at least one target-chain row in Setup before running.")
        if st.button("Create refolding capacity benchmark", type="primary", disabled=run_disabled):
            try:
                run_dir = create_refolding_capacity_benchmark(
                    benchmark_name=str(benchmark_name or "Refolding capacity benchmark"),
                    target_lengths=target_lengths,
                    binder_lengths=binder_lengths,
                    copy_counts=copy_counts,
                    engines=list(selected_engines),
                    gpu_device=str(gpu_device),
                    preset=str(preset or "capacity_only"),
                    conditioning_mode=str(st.session_state.get("capacity_refolding_conditioning_mode") or "default"),
                    launch=bool(launch_now),
                    target_pdb=target_pdb if matrix_mode == "sequence_copy_multimer" else None,
                    target_chains=list(selected_chains),
                    target_name=str(target.get("target_name") if target else ""),
                    matrix_mode=str(matrix_mode),
                    target_panel_entries=target_panel_entries,
                )
                st.success("Capacity benchmark created.")
                show_pipeline_links(run_dir, [run_dir])
            except Exception as exc:
                st.error(str(exc))

    with results_tab:
        _render_refolding_results()


def _render_design_capacity() -> None:
    setup_tab, generator_tab, review_tab, results_tab = st.tabs(["Setup", "Design Engines", "Review & Run", "Results"])
    targets = prepared_design_targets()

    with setup_tab:
        st.subheader("Minibinder Design Capacity")
        st.caption(
            "Use this to test how far each backbone generator gets as binder length increases for one fixed target."
        )
        benchmark_name = st.text_input("Benchmark name", value="Minibinder design capacity check", key="design_capacity_name")
        target, target_pdb, selected_chains = _target_controls(targets, key_prefix="design_capacity")
        ladder_cols = st.columns(3)
        binder_start = ladder_cols[0].number_input("Start binder length", min_value=1, max_value=2000, value=50, step=1, key="design_capacity_binder_start")
        binder_step = ladder_cols[1].number_input("Increase by", min_value=1, max_value=500, value=50, step=1, key="design_capacity_binder_step")
        binder_max = ladder_cols[2].number_input("Stop at length", min_value=1, max_value=5000, value=300, step=50, key="design_capacity_binder_max")
        if binder_max < binder_start:
            st.warning("Stop length is smaller than the start length.")
            binder_lengths = []
        else:
            binder_lengths = list(range(int(binder_start), int(binder_max) + 1, int(binder_step)))
        st.caption(
            "The benchmark creates one generator-only cell per selected engine and binder length. "
            "Each generator ladder runs from short to long and skips larger lengths after the first failed cell."
        )
        selected_hotspots = _capacity_hotspot_controls(target_pdb, selected_chains, key_prefix="design_capacity")
        fixed_target_len = _selected_target_length(target_pdb, selected_chains)
        matrix_rows = [
            {
                "test": "design generator",
                "target": target.get("target_name") if target else "prepared target",
                "target_length": fixed_target_len,
                "binder_length": binder_len,
                "total_length": (fixed_target_len + binder_len) if fixed_target_len else None,
            }
            for binder_len in binder_lengths
        ]
        _show_matrix(matrix_rows)

    with generator_tab:
        st.subheader("Design Engines")
        st.caption("Backbone/generator output is the endpoint for this capacity test. No sequence selection, refolding, ranking, or metrics run after it.")
        generator_keys = list(DESIGN_GENERATOR_LABELS)
        if "capacity_selected_generators" not in st.session_state:
            st.session_state["capacity_selected_generators"] = [
                "rfdiffusion_classic",
                "bindcraft",
                "rfdiffusion3_foundry",
                "boltzgen",
                "pxdesign",
                "genie3",
                "esmfold2_binder_design",
                "protpardelle_1c",
                "proteina_complexa",
            ]
        selected_generators = st.multiselect(
            "Backbone engines",
            generator_keys,
            default=st.session_state["capacity_selected_generators"],
            format_func=lambda key: DESIGN_GENERATOR_LABELS.get(key, key),
            key="capacity_selected_generators",
        )
        runnable_generators = [key for key in selected_generators if key != "genie3" or bool(selected_hotspots)]
        if "genie3" in selected_generators and not selected_hotspots:
            st.warning("Genie3 requires target hotspots in this app. Select hotspots in Setup or Genie3 will be skipped.")
        st.dataframe(
            pd.DataFrame(
                [
                    {"engine": "RFdiffusion classic", "staged handoff": "RFdiffusion backbone PDB", "native downstream skipped": "MPNN, monomer refold, complex refold, ranking"},
                    {"engine": "BindCraft", "staged handoff": "BindCraft AFDesign trajectory/backbone before MPNN", "native downstream skipped": "BindCraft ProteinMPNN redesign and accepted-MPNN validation"},
                    {"engine": "RFdiffusion3 / Foundry", "staged handoff": "RFdiffusion3 generated CIF", "native downstream skipped": "Foundry MPNN, RF3 folding"},
                    {"engine": "BoltzGen", "staged handoff": "BoltzGen intermediate design CIF", "native downstream skipped": "final budget/ranking stage"},
                    {"engine": "PXDesign", "staged handoff": "PXDesign generated structure", "native downstream skipped": "pipeline evaluation mode"},
                    {"engine": "Genie3", "staged handoff": "Genie3 generated structure", "native downstream skipped": "native folding evaluation"},
                    {"engine": "Protpardelle-1c", "staged handoff": "Protpardelle scaffold sample", "native downstream skipped": "native MPNN sequence generation"},
                    {"engine": "ESMFold2 binder design", "staged handoff": "native ESMFold2 designed complex", "native downstream skipped": "none; this engine designs sequence and structure together"},
                    {"engine": "Proteina-Complexa", "staged handoff": "native Proteina-Complexa design candidate", "native downstream skipped": "none; adapter emits its designed candidate set"},
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        show_generator_settings = st.checkbox("Show advanced design-engine settings", value=False, key="capacity_show_generator_settings")
        if show_generator_settings:
            with st.expander("RFdiffusion classic", expanded=True):
                cols = st.columns(2)
                cols[0].number_input("Diffusion timesteps", min_value=1, max_value=1000, value=50, step=1, key="capacity_rfdiffusion_timesteps")
                cols[1].selectbox("Model weights", ["Complex_base"], index=0, key="capacity_rfdiffusion_weights")
                st.text_input("Contig override", value="Automatically derived from target chains and binder length", disabled=True, key="capacity_rfdiffusion_contig")
            with st.expander("Shared capacity behavior", expanded=False):
                st.caption(
                    "Each engine receives the same target and current binder length. The benchmark stops after "
                    "generator artifacts are emitted; larger lengths for that generator are skipped after the first failure."
                )

    with review_tab:
        st.subheader("Review & Run")
        gpu_device = gpu_run_panel(key="design_capacity_benchmark", default="0")
        runnable_generators = [key for key in selected_generators if key != "genie3" or bool(selected_hotspots)]
        skipped_generators = [key for key in selected_generators if key not in runnable_generators]
        cells = len(matrix_rows) * len(runnable_generators)
        st.dataframe(
            _settings_dataframe(
                [
                    {"setting": "Benchmark", "value": benchmark_name},
                    {"setting": "GPU", "value": gpu_device},
                    {"setting": "Target", "value": target.get("target_name") if target else ""},
                    {"setting": "Target chains", "value": ",".join(selected_chains)},
                    {"setting": "Binder ladder", "value": ",".join(str(value) for value in binder_lengths)},
                    {"setting": "Target hotspots", "value": selected_hotspots or "none"},
                    {"setting": "Generators", "value": len(runnable_generators)},
                    {"setting": "Skipped generators", "value": ", ".join(DESIGN_GENERATOR_LABELS.get(key, key) for key in skipped_generators) or "none"},
                    {"setting": "Generator-length cells", "value": cells},
                    {"setting": "Ladder stop rule", "value": "per generator: skip larger lengths after first failed cell"},
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        if skipped_generators:
            st.warning("Genie3 will not be queued because no target hotspots are selected.")
        run_disabled = not target or not target_pdb or not target_pdb.exists() or not selected_chains or not binder_lengths or not runnable_generators
        if st.button("Create design capacity benchmark", type="primary", disabled=run_disabled):
            try:
                target_name = str(target.get("target_name") if target else target_pdb.stem)
                parent_run = create_design_capacity_benchmark(
                    benchmark_name=str(benchmark_name or "Design capacity benchmark"),
                    target_pdb=target_pdb,
                    target_chains=list(selected_chains),
                    target_name=target_name,
                    hotspots=selected_hotspots,
                    binder_lengths=[int(value) for value in binder_lengths],
                    generators=list(runnable_generators),
                    gpu_device=str(gpu_device),
                )
                st.success(
                    f"Created a sequential design-capacity ladder with {cells} generator-length cells. "
                    "Each generator stops after its first failed binder length."
                )
                show_pipeline_links(parent_run, [parent_run])
            except Exception as exc:
                st.error(str(exc))

    with results_tab:
        _render_design_capacity_results()


def main() -> None:
    st.title("Capacity Benchmark")
    st.caption("Measure installed-GPU capacity with separate workflows for refolding engines and minibinder design generators.")

    refolding_tab, design_tab = st.tabs(["Refolding Capacity", "Design Capacity"])

    with refolding_tab:
        _render_refolding_capacity()

    with design_tab:
        _render_design_capacity()


if os.getenv("MN_PROTEIN_DESIGN_IMPORT_CAPACITY_HELPERS") != "1":
    main()
