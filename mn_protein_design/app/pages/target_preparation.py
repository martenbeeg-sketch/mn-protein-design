from __future__ import annotations

import json
import hashlib
import math
import re
from pathlib import Path
from uuid import uuid4

import altair as alt
import pandas as pd
import streamlit as st

from mn_protein_design.app.components.molstar_viewer import ChainVisualization, StructureVisualization, molstar_custom_component
from mn_protein_design.app.pages.common import (
    gpu_run_panel,
    refresh_results_button,
    result_link,
    selected_dataframe_rows,
    show_delete_jobs_dialog,
    show_pipeline_links,
)
from mn_protein_design.core.jobs import ACTIVE_STATUSES, collect_jobs, derive_job_warning, read_json
from mn_protein_design.core.local_worker import spawn_worker_for_run
from mn_protein_design.core.residue_selection import ResidueSelection
from mn_protein_design.core.runtime_estimator import estimate_engines, format_duration
from mn_protein_design.core.structures import (
    detect_nonstandard_residues,
    download_alphafold_db_pdb,
    download_pdb,
    filter_pdb_text,
    pdb_summary,
)
from mn_protein_design.runtime import app_home
from mn_protein_design.workflows.benchmark import (
    ALPHAFAST_DB_DIR,
    ALPHAFAST_WEIGHTS_DIR,
    COLABFOLD_CACHE_DIR,
    MSA_REPOSITORY_DIR,
)
from mn_protein_design.workflows.detection import ppi_target_jobs
from mn_protein_design.workflows.refolding import (
    OPENFOLD3_CHECKPOINT,
    PROTENIX_V1_MODEL,
    PROTENIX_V2_MODEL,
    RF3_CHECKPOINT,
    _sequences_by_chain,
)
from mn_protein_design.workflows import target_msa as target_msa_workflow
from mn_protein_design.workflows.target_prep import (
    analyze_target_chain,
    enqueue_target_preparation,
    enqueue_target_refolding_evaluation,
    split_target_chain_fragments,
    target_chain_break_summary,
)
from mn_protein_design.workflows.target_masking import (
    HYDROPHOBIC_RESIDUES,
    create_masked_target,
    crop_exposure_table,
    mutation_labels,
    parse_mutation_text,
)


ENGINE_METRIC_FILES = {
    "AF3": ("alphafast_af3", "alphafast_af3_metrics.csv"),
    "ColabFold": ("colab", "colabfold_metrics.csv"),
    "AF2": ("af2", "af2_initial_guess_metrics.csv"),
    "Boltz-2": ("boltz2", "boltz2_initial_guess_metrics.csv"),
    "ESMFold2": ("esmfold2", "esmfold2_metrics.csv"),
    "RF3": ("rf3", "rf3_metrics.csv"),
    "OpenFold-3": ("openfold3", "openfold3_metrics.csv"),
    "Protenix v0.5": ("protenix", "protenix_metrics.csv"),
    "Protenix v1": ("protenix_v1", "protenix_v1_metrics.csv"),
    "Protenix v2": ("protenix_v2", "protenix_v2_metrics.csv"),
    "BoltzGen Fold": ("boltzgen_fold", "boltzgen_fold_metrics.csv"),
}
TARGET_CATEGORY_ORDER = ["imported", "trimmed", "cleaned", "split_fragments", "mutated", "cropped", "benchmark", "unknown"]


st.title("Target Preparation")
st.caption("Create, inspect, and refold target artifacts for downstream detection, design, benchmark, and validation tasks.")

state = st.session_state.setdefault(
    "target_prep",
    {
        "source_name": "",
        "source_path": "",
        "pdb_text": "",
        "selected_chains": [],
        "trim_ranges": {},
        "remove_waters": True,
        "remove_hetero": False,
        "map_known_modified_residues": False,
        "replace_nonstandard_residues": False,
        "prepare_msa": True,
    },
)


def _store_source(name: str, pdb_text: str) -> Path:
    upload_dir = app_home() / "workdir" / "targets" / "imports"
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in name).strip("._") or "target"
    if not safe_name.lower().endswith(".pdb"):
        safe_name = f"{safe_name}.pdb"
    path = upload_dir / f"{Path(safe_name).stem}-{uuid4().hex[:8]}.pdb"
    path.write_text(pdb_text)
    state.update({"source_name": Path(safe_name).stem, "source_path": str(path), "pdb_text": pdb_text})
    summary = pdb_summary(pdb_text)
    state["selected_chains"] = [chain["chain_id"] for chain in summary["chains"]]
    state["trim_ranges"] = {chain["chain_id"]: [chain["start"], chain["end"]] for chain in summary["chains"]}
    return path


def _residue_selections() -> list[ResidueSelection]:
    selections: list[ResidueSelection] = []
    for chain in state["selected_chains"]:
        start, end = state["trim_ranges"].get(chain, [None, None])
        if start is not None and end is not None:
            selections.append(ResidueSelection(chain, int(start), int(end)))
    return selections


def _trimmed_preview_text() -> str:
    if not state["pdb_text"]:
        return ""
    return filter_pdb_text(
        state["pdb_text"],
        keep_chains=set(state["selected_chains"]) or None,
        remove_waters=state["remove_waters"],
        remove_hetero=state["remove_hetero"],
        residue_selections=_residue_selections(),
    )


def _selected_text_for_nonstandard_detection() -> str:
    if not state["pdb_text"]:
        return ""
    return filter_pdb_text(
        state["pdb_text"],
        keep_chains=set(state["selected_chains"]) or None,
        remove_waters=True,
        remove_hetero=False,
        residue_selections=_residue_selections(),
    )


def _target_chain_rows(targets: list[dict]) -> pd.DataFrame:
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
        declared = set()
        for item in row.get("chains") or []:
            chain_id = str(item.get("chain_id") if isinstance(item, dict) else item).strip()
            if chain_id:
                declared.add(chain_id)
        for chain in summary.get("chains") or []:
            chain_id = str(chain.get("chain_id") or "")
            if not chain_id or (declared and chain_id not in declared):
                continue
            sequence = str(sequence_by_chain.get(chain_id) or "").replace("X", "")
            break_count = 0
            fragment_count = 1 if sequence else 0
            try:
                break_summary = target_chain_break_summary(target_pdb, chain_id)
                break_count = int(break_summary.get("break_count") or 0)
                fragment_count = int(break_summary.get("fragment_count") or fragment_count)
            except Exception:
                pass
            rows.append(
                {
                    "select": False,
                    "target": row.get("target_name") or target_pdb.stem,
                    "chain": chain_id,
                    "aa_length": len(sequence) or int(chain.get("residue_count") or 0),
                    "fragments": fragment_count,
                    "breaks": break_count,
                    "source": row.get("source_label") or row.get("source_category") or "",
                    "category": row.get("source_category") or row.get("prepared_kind") or "",
                    "job": row.get("job_code") or "",
                    "records": row.get("records") or 1,
                    "path": str(target_pdb),
                    "_target_index": index,
                }
            )
    return pd.DataFrame(rows)


def _target_structure_rows(targets: list[dict]) -> pd.DataFrame:
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
        declared = set()
        for item in row.get("chains") or []:
            chain_id = str(item.get("chain_id") if isinstance(item, dict) else item).strip()
            if chain_id:
                declared.add(chain_id)
        chains: list[str] = []
        aa_length = 0
        break_count = 0
        fragment_count = 0
        for chain in summary.get("chains") or []:
            chain_id = str(chain.get("chain_id") or "")
            if not chain_id or (declared and chain_id not in declared):
                continue
            chains.append(chain_id)
            sequence = str(sequence_by_chain.get(chain_id) or "").replace("X", "")
            aa_length += len(sequence) or int(chain.get("residue_count") or 0)
            try:
                break_summary = target_chain_break_summary(target_pdb, chain_id)
                break_count += int(break_summary.get("break_count") or 0)
                fragment_count += int(break_summary.get("fragment_count") or (1 if sequence else 0))
            except Exception:
                fragment_count += 1 if sequence else 0
        rows.append(
            {
                "target": row.get("target_name") or target_pdb.stem,
                "chains": ",".join(chains),
                "chain_count": len(chains),
                "aa_length": aa_length or row.get("residue_count") or 0,
                "fragments": fragment_count,
                "breaks": break_count,
                "source": row.get("source_label") or row.get("source_category") or "",
                "category": row.get("source_category") or row.get("prepared_kind") or "",
                "job": row.get("job_code") or "",
                "delete_job": _target_delete_label(row),
                "records": row.get("records") or 1,
                "path": str(target_pdb),
                "_target_index": index,
            }
        )
    return pd.DataFrame(rows)


def _selected_target_entries(edited: pd.DataFrame, filtered_rows: pd.DataFrame, targets: list[dict]) -> list[dict]:
    selected_visible = edited[edited["select"] == True] if "select" in edited.columns else pd.DataFrame()
    selected_entries = []
    for _, selected_row in selected_visible.iterrows():
        matches = filtered_rows[
            (filtered_rows["target"].astype(str) == str(selected_row.get("target")))
            & (filtered_rows["chain"].astype(str) == str(selected_row.get("chain")))
            & (filtered_rows["path"].astype(str) == str(selected_row.get("path")))
        ]
        if matches.empty:
            continue
        source = targets[int(matches.iloc[0]["_target_index"])]
        target_entity_chains = []
        target_entities = source.get("target_entities") if isinstance(source.get("target_entities"), list) else []
        for entity in target_entities:
            if isinstance(entity, dict) and str(entity.get("role") or "") == "target":
                target_entity_chains = [str(chain) for chain in entity.get("chains") or [] if str(chain)]
                if target_entity_chains:
                    break
        if not target_entity_chains and str(source.get("prepared_kind") or source.get("source_category") or "") == "split_fragments":
            target_entity_chains = [str(chain) for chain in source.get("chains") or [] if str(chain)]
        selected_entries.append(
            {
                "target_name": selected_row.get("target"),
                "target_pdb": selected_row.get("path"),
                "chain": selected_row.get("chain"),
                "target_entity_chains": target_entity_chains,
                "target_entities": target_entities,
                "source_category": selected_row.get("category"),
                "source_label": selected_row.get("source"),
                "source_job_code": selected_row.get("job"),
                "records": selected_row.get("records"),
                "source_run_id": source.get("run_id"),
                "source_task_group": source.get("task_group"),
            }
        )
    return selected_entries


def _selected_target_structure_entries(edited: pd.DataFrame, filtered_rows: pd.DataFrame, targets: list[dict]) -> list[dict]:
    selected_visible = edited[edited["select"] == True] if "select" in edited.columns else pd.DataFrame()
    selected_entries = []
    for _, selected_row in selected_visible.iterrows():
        matches = filtered_rows[
            (filtered_rows["target"].astype(str) == str(selected_row.get("target")))
            & (filtered_rows["path"].astype(str) == str(selected_row.get("path")))
        ]
        if matches.empty:
            continue
        source = targets[int(matches.iloc[0]["_target_index"])]
        chains = [item.strip() for item in str(selected_row.get("chains") or "").split(",") if item.strip()]
        target_entities = source.get("target_entities") if isinstance(source.get("target_entities"), list) else []
        target_entity_chains = []
        for entity in target_entities:
            if isinstance(entity, dict) and str(entity.get("role") or "") == "target":
                target_entity_chains = [str(chain) for chain in entity.get("chains") or [] if str(chain)]
                if target_entity_chains:
                    break
        if not target_entity_chains:
            target_entity_chains = chains
        selected_entries.append(
            {
                "target_name": selected_row.get("target"),
                "target_pdb": selected_row.get("path"),
                "chain": target_entity_chains[0] if target_entity_chains else (chains[0] if chains else ""),
                "chains": chains,
                "chain_count": selected_row.get("chain_count"),
                "aa_length": selected_row.get("aa_length"),
                "target_entity_chains": target_entity_chains,
                "target_entities": target_entities,
                "source_category": selected_row.get("category"),
                "source_label": selected_row.get("source"),
                "source_job_code": selected_row.get("job"),
                "records": selected_row.get("records"),
                "source_run_id": source.get("run_id"),
                "source_task_group": source.get("task_group"),
            }
        )
    return selected_entries


def _target_delete_ref(row: dict[str, object]) -> tuple[str, str] | None:
    task_group = str(row.get("task_group") or "").strip()
    run_id = str(row.get("run_id") or "").strip()
    run_dir = Path(str(row.get("run_dir") or "")).expanduser()
    if not task_group or not run_id or not run_dir.exists() or not (run_dir / "metadata.json").exists():
        return None
    return task_group, run_id


def _target_delete_label(row: dict[str, object]) -> str:
    ref = _target_delete_ref(row)
    if ref is None:
        return "reference / not deletable"
    return str(row.get("job_code") or row.get("run_id") or "")


def _render_target_library_manager(*, key_prefix: str = "target_library") -> None:
    targets = [row for row in ppi_target_jobs() if Path(str(row.get("target_pdb") or "")).expanduser().exists()]
    if not targets:
        st.info("No targets are currently registered.")
        return

    filter_cols = st.columns([1.2, 3.0])
    category_options = _target_category_options({_target_category(row) for row in targets})
    selected_categories = filter_cols[0].multiselect(
        "Target categories",
        category_options,
        default=category_options,
        key=f"{key_prefix}_categories",
    )
    search_text = filter_cols[1].text_input(
        "Search targets",
        value="",
        placeholder="Target name, source, job code, run id, or path",
        key=f"{key_prefix}_search",
    )
    filtered_targets = [
        row
        for row in targets
        if (not selected_categories or _target_category(row) in selected_categories)
        and (not search_text.strip() or search_text.strip().lower() in _target_search_blob(row))
    ]
    if not filtered_targets:
        st.info("No targets match the current filters.")
        return

    target_rows = _target_structure_rows(filtered_targets)
    if target_rows.empty:
        st.info("No target rows are available for the current target filters.")
        return

    display_df = target_rows.drop(columns=["_target_index"], errors="ignore").copy()
    table_key = f"{key_prefix}_table"
    event = st.dataframe(
        display_df,
        hide_index=True,
        width="stretch",
        height=330,
        key=table_key,
        on_select="rerun",
        selection_mode="multi-row",
        column_config={
            "aa_length": st.column_config.NumberColumn("AA", format="%d"),
            "chain_count": st.column_config.NumberColumn("Chains", format="%d"),
            "fragments": st.column_config.NumberColumn("Fragments", format="%d"),
            "breaks": st.column_config.NumberColumn("Breaks", format="%d"),
            "path": st.column_config.TextColumn("PDB path", width="large"),
        },
    )

    delete_result = st.session_state.pop(f"{key_prefix}_delete_result", None)
    if delete_result:
        level, message = delete_result
        if level == "success":
            st.success(str(message))
        else:
            st.error(str(message))

    selected_indices = [idx for idx in selected_dataframe_rows(event, table_key) if 0 <= idx < len(target_rows)]
    selected_refs: list[tuple[str, str]] = []
    skipped = 0
    seen_refs: set[tuple[str, str]] = set()
    for index in selected_indices:
        target = filtered_targets[int(target_rows.iloc[index]["_target_index"])]
        ref = _target_delete_ref(target)
        if ref is None:
            skipped += 1
            continue
        if ref not in seen_refs:
            seen_refs.add(ref)
            selected_refs.append(ref)

    selected_refs_key = f"{key_prefix}_selected_delete_refs"
    if selected_refs:
        st.session_state[selected_refs_key] = selected_refs
    cached_selected_refs = st.session_state.get(selected_refs_key) or []

    if skipped:
        st.caption(f"{skipped} selected reference row{' was' if skipped == 1 else 's were'} skipped because it is not a deletable job.")
    if selected_refs:
        st.caption(
            f"Selected {len(selected_refs):,} deletable target job"
            f"{'' if len(selected_refs) == 1 else 's'}."
        )
    delete_clicked = st.button(
        "Delete selected target jobs",
        type="primary",
        disabled=not cached_selected_refs,
        key=f"{key_prefix}_request_delete_jobs",
    )
    if delete_clicked and cached_selected_refs:
        show_delete_jobs_dialog(
            table_key=key_prefix,
            pending_refs=cached_selected_refs,
            selected_refs_key=selected_refs_key,
            label="target job",
        )
    else:
        st.caption("Select target rows created by a job to enable deletion. Benchmark/reference rows stay read-only.")


def _target_break_count(entry: dict[str, object]) -> int:
    target_pdb = Path(str(entry.get("target_pdb") or ""))
    chains = [str(chain) for chain in entry.get("target_entity_chains") or [] if str(chain)]
    if not chains and str(entry.get("chain") or ""):
        chains = [str(entry.get("chain") or "")]
    total = 0
    for chain in chains:
        try:
            total += int(target_chain_break_summary(target_pdb, chain).get("break_count") or 0)
        except Exception:
            continue
    return total


def _target_selection_signature(entries: list[dict[str, object]]) -> str:
    parts = []
    for entry in entries:
        parts.append(
            "|".join(
                [
                    str(entry.get("target_pdb") or ""),
                    str(entry.get("chain") or ""),
                    str(entry.get("target_name") or ""),
                ]
            )
        )
    return ";;".join(sorted(parts))


def _target_entry_sequence_length(entry: dict[str, object]) -> int:
    chains = [str(chain) for chain in entry.get("target_entity_chains") or [] if str(chain)]
    if not chains:
        chains = [str(chain) for chain in entry.get("chains") or [] if str(chain)]
    if not chains and str(entry.get("chain") or ""):
        chains = [str(entry.get("chain") or "")]
    if len(chains) > 1:
        total = 0
        for chain in chains:
            try:
                summary = target_chain_break_summary(Path(str(entry.get("target_pdb") or "")), chain)
                total += int(summary.get("sequence_length") or 0)
            except Exception:
                try:
                    total += len(str(_sequences_by_chain(Path(str(entry.get("target_pdb") or ""))).get(chain) or "").replace("X", ""))
                except Exception:
                    pass
        if total > 0:
            return total
    try:
        length = int(float(str(entry.get("aa_length") or "0")))
        if length > 0 and (not chains or len(chains) > 1):
            return length
    except Exception:
        pass
    try:
        summary = target_chain_break_summary(
            Path(str(entry.get("target_pdb") or "")),
            str(entry.get("chain") or ""),
        )
        length = int(summary.get("sequence_length") or 0)
        if length > 0:
            return length
    except Exception:
        pass
    try:
        return len(str(_sequences_by_chain(Path(str(entry.get("target_pdb") or ""))).get(str(entry.get("chain") or "")) or "").replace("X", ""))
    except Exception:
        return 0


def _target_entry_chain_count(entry: dict[str, object]) -> int:
    chains = [str(chain) for chain in entry.get("target_entity_chains") or [] if str(chain)]
    if not chains:
        chains = [str(chain) for chain in entry.get("chains") or [] if str(chain)]
    if chains:
        return len(dict.fromkeys(chains))
    try:
        count = int(float(str(entry.get("chain_count") or "0")))
        if count > 0:
            return count
    except Exception:
        pass
    return 1 if str(entry.get("chain") or "") else 0


def _target_entry_missing_cached_msa_count(entry: dict[str, object], msa_repository_dir: Path) -> int:
    target_pdb = Path(str(entry.get("target_pdb") or "")).expanduser()
    if not target_pdb.exists():
        return 0
    chains = [str(chain) for chain in entry.get("target_entity_chains") or [] if str(chain)]
    if not chains:
        chains = [str(chain) for chain in entry.get("chains") or [] if str(chain)]
    if not chains and str(entry.get("chain") or ""):
        chains = [str(entry.get("chain") or "")]
    if not chains:
        return 0
    try:
        sequences = _sequences_by_chain(target_pdb)
    except Exception:
        return len(chains)
    missing = 0
    for chain in dict.fromkeys(chains):
        sequence = str(sequences.get(chain) or "").replace("X", "")
        if not sequence:
            missing += 1
            continue
        cached_path, _source = target_msa_workflow.find_cached_msa_for_sequence(
            sequence,
            msa_repository_dir=msa_repository_dir,
        )
        if cached_path is None:
            missing += 1
    return missing


def _selected_missing_cached_msa_count(entries: list[dict[str, object]], msa_repository_dir: Path) -> int:
    return sum(_target_entry_missing_cached_msa_count(entry, msa_repository_dir) for entry in entries)


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


def _target_refolding_source_counts(source_run_dir: Path, source_result: dict[str, object]) -> tuple[object, object]:
    manifest = read_json(source_run_dir / "artifacts" / "target_refolding_manifest.json")
    targets = manifest.get("targets") if isinstance(manifest, dict) else None
    if isinstance(targets, list):
        structure_count = len(targets)
        chain_count = sum(
            len([chain for chain in row.get("staged_target_chains") or row.get("target_entity_chains") or [] if str(chain)])
            for row in targets
            if isinstance(row, dict)
        )
        return structure_count, chain_count
    metrics = source_result.get("metrics") if isinstance(source_result.get("metrics"), dict) else {}
    return metrics.get("target_structure_count") or metrics.get("target_chain_count"), metrics.get("target_chain_count")


def _target_refolding_source_summary(source_run_dir: Path, source_result: dict[str, object]) -> dict[str, object]:
    manifest = read_json(source_run_dir / "artifacts" / "target_refolding_manifest.json")
    targets = manifest.get("targets") if isinstance(manifest, dict) else None
    if isinstance(targets, list) and targets:
        names: list[str] = []
        chains: list[str] = []
        pdbs: list[str] = []
        for row in targets:
            if not isinstance(row, dict):
                continue
            name = str(row.get("target_name") or Path(str(row.get("target_pdb") or "")).stem or "").strip()
            if name and name not in names:
                names.append(name)
            for chain in row.get("staged_target_chains") or row.get("target_entity_chains") or row.get("biological_target_chains") or []:
                text = str(chain or "").strip()
                if text and text not in chains:
                    chains.append(text)
            pdb = str(row.get("target_pdb") or "").strip()
            if pdb and pdb not in pdbs:
                pdbs.append(pdb)
        return {
            "target": "; ".join(names) if names else None,
            "target_chain_ids": ",".join(chains) if chains else None,
            "target_pdb": "; ".join(pdbs) if pdbs else None,
            "target_structures": len([row for row in targets if isinstance(row, dict)]),
            "target_chains": len(chains) if chains else None,
        }
    metrics = source_result.get("metrics") if isinstance(source_result.get("metrics"), dict) else {}
    return {
        "target": metrics.get("target_name") or metrics.get("target"),
        "target_chain_ids": metrics.get("target_chains") or metrics.get("target_chain"),
        "target_pdb": metrics.get("target_pdb"),
        "target_structures": metrics.get("target_structure_count") or metrics.get("target_chain_count"),
        "target_chains": metrics.get("target_chain_count"),
    }


def _target_refolding_bool(config: dict[str, object], key: str, default: bool = False) -> bool:
    value = config.get(key)
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _target_refolding_evidence_label(*, template: bool = False, msa: bool = False) -> str:
    if template and msa:
        return "template+MSA"
    if template:
        return "template"
    if msa:
        return "MSA"
    return "none"


def _target_refolding_evidence_summary(params: dict[str, object], metrics: dict[str, object]) -> tuple[str, str]:
    config = {**params, **metrics}
    engine_modes: list[tuple[str, str]] = []

    def enabled(key: str) -> bool:
        return _target_refolding_bool(config, key, False)

    def add(engine: str, *, template: bool = False, msa: bool = False) -> None:
        engine_modes.append((engine, _target_refolding_evidence_label(template=template, msa=msa)))

    if enabled("run_alphafast_af3") or "alphafast_af3_return_code" in config or "alphafast_af3_metrics_table" in config:
        add(
            "AF3",
            template=_target_refolding_bool(config, "alphafast_use_target_templates", True),
            msa=not _target_refolding_bool(config, "alphafast_query_only_msa", False),
        )
    if enabled("run_colabfold") or "colabfold_metrics_table" in config:
        add(
            "ColabFold",
            template=_target_refolding_bool(config, "colabfold_use_target_templates", True),
            msa=_target_refolding_bool(config, "colabfold_use_target_msa", True),
        )
    if enabled("run_af2_initial_guess") or "af2_initial_guess_metrics_table" in config:
        add(
            "AF2 template",
            template=any(
                _target_refolding_bool(config, key, False)
                for key in ("af2_use_initial_guess", "af2_use_binder_template", "af2_use_interface_template")
            ),
            msa=False,
        )
    if enabled("run_boltz2_initial_guess") or "boltz2_initial_guess_metrics_table" in config:
        add(
            "Boltz-2",
            template=_target_refolding_bool(config, "boltz2_use_target_template", True),
            msa=_target_refolding_bool(config, "boltz2_use_target_msa", True),
        )
    if enabled("run_rf3") or "rf3_metrics_table" in config:
        add(
            "RF3",
            template=_target_refolding_bool(config, "rf3_use_target_template", True),
            msa=_target_refolding_bool(config, "rf3_use_target_msa", True),
        )
    if enabled("run_openfold3") or "openfold3_metrics_table" in config:
        add("OpenFold-3", template=False, msa=_target_refolding_bool(config, "openfold3_use_target_msa", True))
    if enabled("run_esmfold2") or "esmfold2_metrics_table" in config:
        esm_modes = config.get("esmfold2_modes") or []
        if isinstance(esm_modes, str):
            esm_modes = [part.strip() for part in esm_modes.split(",") if part.strip()]
        add(
            "ESMFold2",
            template="initial_guess" in set(esm_modes),
            msa=_target_refolding_bool(config, "esmfold2_use_target_msa", False),
        )
    if enabled("run_protenix") or "protenix_metrics_table" in config:
        add("Protenix v0.5", template=False, msa=_target_refolding_bool(config, "protenix_use_msa", True))
    if enabled("run_protenix_v1") or "protenix_v1_metrics_table" in config:
        add(
            "Protenix v1",
            template=_target_refolding_bool(config, "protenix_v1_use_template", False),
            msa=_target_refolding_bool(config, "protenix_v1_use_msa", True),
        )
    if enabled("run_protenix_v2") or "protenix_v2_metrics_table" in config:
        add(
            "Protenix v2",
            template=_target_refolding_bool(config, "protenix_v2_use_template", False),
            msa=_target_refolding_bool(config, "protenix_v2_use_msa", True),
        )
    if enabled("run_boltzgen_fold") or "boltzgen_fold_metrics_table" in config:
        add("BoltzGen Fold", template=True, msa=False)

    if not engine_modes:
        return "", ""
    modes = {mode for _engine, mode in engine_modes}
    summary = next(iter(modes)) if len(modes) == 1 else "mixed"
    detail = "; ".join(f"{engine}: {mode}" for engine, mode in engine_modes)
    return summary, detail


def _target_refolding_runs() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task_group in ("target-refolding", "benchmark"):
        for job in collect_jobs(task_group):
            run_dir = Path(str(job.get("run_dir") or ""))
            input_json = read_json(run_dir / "input.json")
            if str(input_json.get("job_type") or "") != "refolding_evaluation":
                continue
            inputs = input_json.get("inputs") if isinstance(input_json.get("inputs"), dict) else {}
            params = input_json.get("params") if isinstance(input_json.get("params"), dict) else {}
            source_run_dir = Path(str(inputs.get("source_run_dir") or ""))
            source_input = read_json(source_run_dir / "input.json") if source_run_dir.exists() else {}
            is_target_refolding = bool(
                params.get("target_refolding")
                or inputs.get("target_refolding")
                or str(input_json.get("tool") or "") == "target_refolding_evaluation_engines"
                or str(source_input.get("job_type") or "") == "target_refolding_candidate_set"
            )
            if not is_target_refolding:
                continue
            result = read_json(run_dir / "result.json")
            metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
            source_result = read_json(source_run_dir / "result.json")
            source_summary = _target_refolding_source_summary(source_run_dir, source_result)
            evidence_mode, evidence_detail = _target_refolding_evidence_summary(params, metrics)
            rows.append(
                {
                    "result": result_link(str(job.get("task_group") or task_group), str(job.get("run_id"))),
                    "job_code": job.get("job_code"),
                    "task_group": str(job.get("task_group") or task_group),
                    "status": job.get("status"),
                    "warning": _target_refolding_warning(run_dir),
                    "description": job.get("description"),
                    "target": source_summary.get("target"),
                    "target_chain_ids": source_summary.get("target_chain_ids"),
                    "evidence_mode": evidence_mode,
                    "evidence_detail": evidence_detail,
                    "target_structures": source_summary.get("target_structures"),
                    "target_chains": source_summary.get("target_chains"),
                    "records": metrics.get("record_count"),
                    "current_phase": "Predicting target folds"
                    if job.get("current_phase") == "Predicting complexes"
                    else job.get("current_phase"),
                    "current_engine": job.get("current_engine"),
                    "created_at": job.get("created_at"),
                    "run_id": job.get("run_id"),
                    "run_dir": str(run_dir),
                    "source_run_dir": str(source_run_dir),
                }
            )
    return rows


def _metrics_for_run(run_dir: Path) -> pd.DataFrame:
    merged = run_dir / "artifacts" / "benchmark" / "merged_benchmark_metrics.csv"
    if not merged.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(merged)
    except Exception:
        return pd.DataFrame()


def _engine_score_columns(metrics: pd.DataFrame) -> list[str]:
    if metrics.empty:
        return []
    ignored = {"binder_id", "target_id", "target_source", "label", "source"}
    columns = []
    for column in metrics.columns:
        if column in ignored:
            continue
        if not any(str(column).startswith(prefix) for prefix, _csv in ENGINE_METRIC_FILES.values()):
            continue
        numeric = pd.to_numeric(metrics[column], errors="coerce")
        if numeric.notna().any():
            columns.append(column)
    return columns


def _prediction_path(run_dir: Path, engine_label: str, design_id: str) -> Path | None:
    metric_key, _csv = ENGINE_METRIC_FILES.get(engine_label, ("", ""))
    for suffix in [".pdb", ".cif", ".mmcif"]:
        path = run_dir / "artifacts" / "benchmark" / "predicted_metric_pdbs" / metric_key / f"{design_id}{suffix}"
        if path.exists():
            return path
    return None


def _target_refolding_warning(run_dir: Path) -> str:
    metadata = read_json(run_dir / "metadata.json")
    input_json = read_json(run_dir / "input.json")
    result = read_json(run_dir / "result.json")
    return derive_job_warning(metadata, input_json, result)


def _target_option_label(row: dict[str, object]) -> str:
    name = str(row.get("target_name") or Path(str(row.get("target_pdb") or "")).stem)
    source = str(row.get("source_label") or row.get("source_category") or row.get("task_group") or "")
    run_id = str(row.get("run_id") or row.get("job_code") or "")
    return " | ".join(part for part in [name, source, run_id] if part)


def _target_path(row: dict[str, object]) -> Path:
    return Path(str(row.get("target_pdb") or "")).expanduser()


def _a3m_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    lines: list[str] = []
    for raw_line in path.read_text(errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(lines)))
            header = line[1:].strip() or f"sequence_{len(records) + 1}"
            lines = []
        else:
            lines.append(line)
    if header is not None:
        records.append((header, "".join(lines)))
    return records


def _a3m_match_columns(sequence: str, query_length: int) -> list[str]:
    columns: list[str] = []
    for char in str(sequence or ""):
        if char == "-" or (char.isalpha() and char.isupper()):
            columns.append(char.upper())
            if len(columns) >= query_length:
                break
    if len(columns) < query_length:
        columns.extend(["-"] * (query_length - len(columns)))
    return columns[:query_length]


def _msa_depth_summary(msa_path: Path, query_sequence: str) -> dict[str, object]:
    query = re.sub(r"[^A-Za-z]+", "", str(query_sequence or "")).upper()
    records = _a3m_records(msa_path)
    query_length = len(query)
    if not records or not query_length:
        return {
            "sequence_count": len(records),
            "effective_depth": 0,
            "mean_depth": 0.0,
            "min_depth": 0,
            "max_depth": 0,
            "mean_identity": 0.0,
            "position_rows": [],
            "heatmap_rows": [],
        }
    depth = [0] * query_length
    identity_sum = [0.0] * query_length
    aligned_records = [_a3m_match_columns(sequence, query_length) for _header, sequence in records]
    for aligned in aligned_records:
        for index, char in enumerate(aligned):
            if char == "-" or not char.isalpha():
                continue
            depth[index] += 1
            if char == query[index]:
                identity_sum[index] += 1.0
    position_rows = [
        {
            "position": index + 1,
            "depth": value,
            "depth_fraction": value / float(len(records)) if records else 0.0,
            "identity": identity_sum[index] / float(value) if value else 0.0,
        }
        for index, value in enumerate(depth)
    ]
    if len(aligned_records) <= 260:
        sampled_indexes = list(range(len(aligned_records)))
    else:
        sampled_indexes = sorted({round(value) for value in [i * (len(aligned_records) - 1) / 259 for i in range(260)]})
    sampled_records: list[tuple[float, int, int, int, list[str]]] = []
    for record_index in sampled_indexes:
        aligned = aligned_records[record_index]
        covered = [
            (position, char)
            for position, char in enumerate(aligned, start=1)
            if char != "-" and char.isalpha()
        ]
        row_identity = (
            sum(1 for position, char in covered if char == query[position - 1]) / float(len(covered))
            if covered
            else None
        )
        first_covered = covered[0][0] if covered else query_length + 1
        sampled_records.append((float(row_identity or 0.0), first_covered, -len(covered), record_index, aligned))
    sampled_records.sort(key=lambda item: (item[2], item[1], -item[0], item[3]))
    heatmap_rows: list[dict[str, object]] = []
    for display_index, (row_identity, _first_covered, _negative_coverage, _record_index, aligned) in enumerate(sampled_records, start=1):
        for position, char in enumerate(aligned, start=1):
            if char == "-" or not char.isalpha():
                value = None
            else:
                value = row_identity
            heatmap_rows.append({"position": position, "sequence": display_index, "identity": value})
    nonzero_depth = [value for value in depth if value > 0]
    identities = [row["identity"] for row in position_rows if row["depth"]]
    return {
        "sequence_count": len(records),
        "effective_depth": max(0, len(records) - 1),
        "mean_depth": sum(depth) / float(query_length),
        "min_depth": min(depth) if depth else 0,
        "max_depth": max(depth) if depth else 0,
        "covered_positions": len(nonzero_depth),
        "mean_identity": sum(identities) / float(len(identities)) if identities else 0.0,
        "position_rows": position_rows,
        "heatmap_rows": heatmap_rows,
        "sampled_sequence_count": len(sampled_indexes),
    }


def _msa_depth_png_path(msa_path: Path, query_sequence: str) -> Path:
    digest = hashlib.sha256(f"v6|{msa_path}|{msa_path.stat().st_mtime_ns}|{query_sequence}".encode("utf-8")).hexdigest()
    return app_home() / "workdir" / "artifacts" / "msa_depth_plots" / f"{digest[:24]}.png"


def _identity_color(value: object) -> tuple[int, int, int]:
    if value is None or pd.isna(value):
        return (255, 255, 255)
    value = max(0.0, min(1.0, float(value)))
    stops = [
        (0.0, (239, 68, 68)),
        (0.18, (249, 115, 22)),
        (0.35, (250, 204, 21)),
        (0.52, (134, 239, 172)),
        (0.70, (45, 212, 191)),
        (0.86, (56, 189, 248)),
        (1.0, (99, 102, 241)),
    ]
    for (left_value, left_color), (right_value, right_color) in zip(stops, stops[1:], strict=False):
        if value <= right_value:
            span = right_value - left_value
            t = 0.0 if span <= 0 else (value - left_value) / span
            return tuple(int(left_color[i] + (right_color[i] - left_color[i]) * t) for i in range(3))
    return stops[-1][1]


def _write_msa_depth_png(msa_path: Path, query_sequence: str, summary: dict[str, object]) -> Path | None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None
    position_rows = summary.get("position_rows") if isinstance(summary, dict) else None
    heatmap_rows = summary.get("heatmap_rows") if isinstance(summary, dict) else None
    if not position_rows or not heatmap_rows:
        return None
    output_path = _msa_depth_png_path(msa_path, query_sequence)
    if output_path.exists():
        return output_path
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    query_length = len(position_rows)
    sequence_count = int(summary.get("sampled_sequence_count") or 1)
    margin_left = 156
    margin_right = 150
    margin_top = 54
    heatmap_height = min(420, max(180, sequence_count * 2))
    plot_width = min(1700, max(880, query_length * 3))
    width = margin_left + plot_width + margin_right
    height = margin_top + heatmap_height + 78
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 11)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = small_font = title_font = ImageFont.load_default()

    def draw_rotated_label(text: str, center: tuple[int, int], *, label_font: object = font) -> None:
        bbox = draw.textbbox((0, 0), text, font=label_font)
        label_width = max(1, bbox[2] - bbox[0] + 8)
        label_height = max(1, bbox[3] - bbox[1] + 8)
        label = Image.new("RGBA", (label_width, label_height), (255, 255, 255, 0))
        label_draw = ImageDraw.Draw(label)
        label_draw.text((4 - bbox[0], 4 - bbox[1]), text, fill=(55, 65, 81, 255), font=label_font)
        rotated = label.rotate(90, expand=True)
        image.paste(
            rotated,
            (int(center[0] - rotated.width / 2), int(center[1] - rotated.height / 2)),
            rotated,
        )

    x0 = margin_left
    y0 = margin_top
    x_scale = plot_width / float(max(1, query_length))
    y_scale = heatmap_height / float(max(1, sequence_count))
    draw.text((x0, 18), "MSA sequence coverage", fill=(17, 24, 39), font=title_font)
    draw.text((x0, 38), f"{msa_path.name} | sampled {sequence_count} of {summary.get('sequence_count')} sequences", fill=(75, 85, 99), font=small_font)

    for row in heatmap_rows:
        try:
            position = int(row["position"]) - 1
            sequence = int(row["sequence"]) - 1
        except Exception:
            continue
        left = int(x0 + position * x_scale)
        right = int(x0 + (position + 1) * x_scale + 0.999)
        top = int(y0 + sequence * y_scale)
        bottom = int(y0 + (sequence + 1) * y_scale + 0.999)
        draw.rectangle([left, top, max(left, right), max(top, bottom)], fill=_identity_color(row.get("identity")))
    draw.rectangle([x0, y0, x0 + plot_width, y0 + heatmap_height], outline=(156, 163, 175), width=1)
    draw_rotated_label("Sequences", (30, y0 + heatmap_height // 2))

    max_depth = max(1, int(max(float(row.get("depth") or 0) for row in position_rows)))
    points: list[tuple[int, int]] = []
    for row in position_rows:
        position = int(row.get("position") or 1) - 1
        depth = float(row.get("depth") or 0)
        x = int(x0 + (position + 0.5) * x_scale)
        y = int(y0 + heatmap_height - (depth / max_depth) * heatmap_height)
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=(17, 24, 39), width=3)
    for fraction in [0.0, 0.5, 1.0]:
        y = int(y0 + heatmap_height - fraction * heatmap_height)
        draw.text((x0 - 66, y - 7), str(int(max_depth * fraction)), fill=(107, 114, 128), font=small_font)
    draw.text((x0 + plot_width // 2 - 30, height - 34), "Positions", fill=(55, 65, 81), font=font)

    tick_count = 8
    for tick in range(tick_count + 1):
        pos = 1 + round((query_length - 1) * tick / tick_count) if query_length > 1 else 1
        x = int(x0 + (pos - 1) * x_scale)
        draw.line([x, y0 + heatmap_height, x, y0 + heatmap_height + 5], fill=(107, 114, 128), width=1)
        draw.text((x - 10, y0 + heatmap_height + 8), str(pos), fill=(107, 114, 128), font=small_font)

    legend_x = x0 + plot_width + 20
    legend_y = y0 + 18
    legend_height = min(220, max(120, heatmap_height - 34))
    legend_width = 22
    draw.text((legend_x, legend_y - 20), "Identity", fill=(55, 65, 81), font=small_font)
    for offset in range(legend_height):
        value = 1.0 - offset / float(max(1, legend_height - 1))
        draw.line(
            [legend_x, legend_y + offset, legend_x + legend_width, legend_y + offset],
            fill=_identity_color(value),
            width=1,
        )
    draw.rectangle([legend_x, legend_y, legend_x + legend_width, legend_y + legend_height], outline=(209, 213, 219), width=1)
    for value in [1.0, 0.66, 0.33, 0.0]:
        y = int(legend_y + (1.0 - value) * legend_height)
        draw.line([legend_x + legend_width, y, legend_x + legend_width + 5, y], fill=(107, 114, 128), width=1)
        draw.text((legend_x + legend_width + 10, y - 7), f"{value:.2g}", fill=(75, 85, 99), font=small_font)

    try:
        image.save(output_path)
    except OSError:
        return None
    return output_path


def _target_msa_entries(
    selected_target_pdb: Path,
    selected_chain: str,
    analysis: dict[str, object],
    *,
    msa_repository_dir: Path = MSA_REPOSITORY_DIR,
) -> list[dict[str, object]]:
    sequence_by_chain = _sequences_by_chain(selected_target_pdb)
    entries: list[dict[str, object]] = []
    full_sequence = str(sequence_by_chain.get(selected_chain) or "").replace("X", "")
    if full_sequence:
        entries.append(
            {
                "scope": "full chain",
                "chain": selected_chain,
                "fragment": "",
                "sequence": full_sequence,
            }
        )
    detected_fragments: list[dict[str, object]] = []
    try:
        detected_summary = target_chain_break_summary(selected_target_pdb, selected_chain)
        detected_fragments = [
            fragment
            for fragment in detected_summary.get("fragments") or []
            if isinstance(fragment, dict)
        ]
    except Exception:
        detected_fragments = []
    analysis_fragments = [
        fragment
        for fragment in analysis.get("fragments") or []
        if isinstance(fragment, dict)
    ]
    fragment_rows = detected_fragments if detected_fragments else analysis_fragments
    for fragment in fragment_rows:
        if not isinstance(fragment, dict):
            continue
        sequence = str(fragment.get("sequence") or "").replace("X", "")
        entries.append(
            {
                "scope": "fragment",
                "chain": str(fragment.get("source_chain") or selected_chain),
                "fragment": int(fragment.get("fragment_index") or len(entries)),
                "sequence": sequence,
                "source_start": fragment.get("source_start"),
                "source_end": fragment.get("source_end"),
            }
        )
    seen: set[tuple[str, str, str, str, str, str]] = set()
    rows: list[dict[str, object]] = []
    for entry in entries:
        sequence = str(entry.get("sequence") or "")
        key = (
            str(entry.get("scope") or ""),
            str(entry.get("chain") or ""),
            str(entry.get("fragment") or ""),
            str(entry.get("source_start") or ""),
            str(entry.get("source_end") or ""),
            sequence,
        )
        if key in seen:
            continue
        seen.add(key)
        if not sequence:
            rows.append(
                {
                    **entry,
                    "length": 0,
                    "status": "empty",
                    "cache_source": "",
                    "msa_path": "",
                    "sequence_count": None,
                    "effective_depth": None,
                    "mean_depth": None,
                    "min_depth": None,
                    "max_depth": None,
                    "mean_identity": None,
                    "plot_png": "",
                    "_summary": {},
                }
            )
            continue
        expected_path, _container_path = target_msa_workflow.boltz_msa_paths(
            sequence,
            msa_repository_dir=Path(msa_repository_dir),
        )
        msa_path, source = target_msa_workflow.find_cached_msa_for_sequence(
            sequence,
            msa_repository_dir=Path(msa_repository_dir),
        )
        valid = bool(msa_path and Path(msa_path).exists())
        summary = _msa_depth_summary(Path(msa_path), sequence) if valid else {}
        png_path = _write_msa_depth_png(Path(msa_path), sequence, summary) if valid else None
        rows.append(
            {
                **entry,
                "length": len(sequence),
                "status": "cached" if valid else "missing",
                "cache_source": source,
                "msa_path": str(msa_path or expected_path),
                "sequence_count": summary.get("sequence_count") if summary else None,
                "effective_depth": summary.get("effective_depth") if summary else None,
                "mean_depth": summary.get("mean_depth") if summary else None,
                "min_depth": summary.get("min_depth") if summary else None,
                "max_depth": summary.get("max_depth") if summary else None,
                "mean_identity": summary.get("mean_identity") if summary else None,
                "plot_png": str(png_path) if png_path else "",
                "_summary": summary,
            }
        )
    return rows


def _render_msa_depth_plot(summary: dict[str, object], *, key: str) -> None:
    position_rows = summary.get("position_rows") if isinstance(summary, dict) else None
    heatmap_rows = summary.get("heatmap_rows") if isinstance(summary, dict) else None
    if not position_rows:
        return
    depth_df = pd.DataFrame(position_rows)
    heatmap_df = pd.DataFrame(heatmap_rows or [])
    depth = (
        alt.Chart(depth_df)
        .mark_line(color="#111827", strokeWidth=2)
        .encode(
            x=alt.X("position:Q", title="Position"),
            y=alt.Y("depth:Q", title="Sequences covering position"),
            tooltip=[
                alt.Tooltip("position:Q", title="Position", format=".0f"),
                alt.Tooltip("depth:Q", title="Depth", format=".0f"),
                alt.Tooltip("depth_fraction:Q", title="Depth fraction", format=".2f"),
                alt.Tooltip("identity:Q", title="Mean identity", format=".2f"),
            ],
        )
        .properties(height=140)
    )
    if heatmap_df.empty:
        st.altair_chart(depth, width="stretch", key=f"{key}_depth")
        return
    heatmap = (
        alt.Chart(heatmap_df)
        .mark_rect()
        .encode(
            x=alt.X("position:Q", title="Position"),
            y=alt.Y("sequence:Q", title="Sampled MSA sequences", sort="descending"),
            color=alt.Color(
                "identity:Q",
                title="Identity to query",
                scale=alt.Scale(domain=[0, 1], range=["#ef4444", "#fbbf24", "#6ee7b7", "#2563eb"]),
                legend=alt.Legend(format=".1f"),
            ),
            tooltip=[
                alt.Tooltip("position:Q", title="Position", format=".0f"),
                alt.Tooltip("sequence:Q", title="Sampled sequence", format=".0f"),
                alt.Tooltip("identity:Q", title="Identity", format=".1f"),
            ],
        )
        .properties(height=min(280, max(120, int(math.sqrt(len(heatmap_df)) * 2))))
    )
    st.altair_chart(alt.vconcat(heatmap, depth).resolve_scale(x="shared"), width="stretch", key=key)


def _target_category(row: dict[str, object]) -> str:
    return str(row.get("source_category") or row.get("prepared_kind") or row.get("task_group") or "unknown").strip() or "unknown"


def _target_category_options(categories: list[str] | set[str]) -> list[str]:
    present = {str(category) for category in categories if str(category)}
    present.add("mutated")
    ordered = [category for category in TARGET_CATEGORY_ORDER if category in present]
    ordered.extend(sorted(category for category in present if category not in TARGET_CATEGORY_ORDER))
    return ordered


def _target_search_blob(row: dict[str, object]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in [
            "target_name",
            "source_label",
            "source_category",
            "prepared_kind",
            "task_group",
            "run_id",
            "job_code",
            "target_pdb",
        ]
    ).lower()


def _target_analysis_report_path(target_pdb: Path, chain: str) -> Path:
    report_dir = app_home() / "workdir" / "targets" / "analysis"
    safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in target_pdb.stem).strip("_") or "target"
    safe_chain = "".join(ch if ch.isalnum() else "_" for ch in str(chain or "chain")).strip("_") or "chain"
    return report_dir / f"{safe_stem}_{safe_chain}_analysis.json"


def _write_target_analysis_report(target_pdb: Path, chain: str, analysis: dict[str, object]) -> Path:
    report_path = _target_analysis_report_path(target_pdb, chain)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "target_pdb": str(target_pdb),
                "chain": str(chain),
                "analysis": analysis,
            },
            indent=2,
        )
    )
    return report_path


def _target_break_residues(analysis: dict[str, object]) -> dict[str, list[int]]:
    residues_by_chain: dict[str, set[int]] = {}
    for row in analysis.get("breaks") or []:
        if not isinstance(row, dict):
            continue
        chain = str(row.get("source_chain") or "").strip()
        if not chain:
            continue
        for key in ("before_residue", "after_residue"):
            try:
                residue = int(row.get(key) or 0)
            except (TypeError, ValueError):
                continue
            if residue > 0:
                residues_by_chain.setdefault(chain, set()).add(residue)
    return {chain: sorted(residues) for chain, residues in sorted(residues_by_chain.items())}


def _target_break_selection_labels(residues_by_chain: dict[str, list[int]]) -> list[str]:
    labels: list[str] = []
    for chain, residues in residues_by_chain.items():
        safe_chain = "".join(ch for ch in str(chain).upper() if ch.isalpha())
        if not safe_chain:
            continue
        labels.extend(f"{safe_chain}{int(residue)}" for residue in residues)
    return labels


def _crop_source_reference(row: dict[str, object]) -> tuple[Path | None, dict[str, object]]:
    run_dir = Path(str(row.get("run_dir") or ""))
    details: dict[str, object] = {}
    if not run_dir.exists():
        return None, details
    result = read_json(run_dir / "result.json")
    input_json = read_json(run_dir / "input.json")
    crop = (result.get("outputs") or {}).get("crop") if isinstance(result.get("outputs"), dict) else {}
    inputs = result.get("inputs") if isinstance(result.get("inputs"), dict) else {}
    if not isinstance(inputs, dict):
        inputs = input_json.get("inputs") if isinstance(input_json.get("inputs"), dict) else {}
    source_text = str((crop or {}).get("source_pdb") or inputs.get("target_pdb") or "").strip()
    source_path = Path(source_text).expanduser() if source_text else None
    details = {
        "source_pdb": str(source_path or ""),
        "manual_residue_count": (crop or {}).get("manual_residue_count") or (result.get("metrics") or {}).get("manual_residue_count"),
        "viewer_residue_count": (crop or {}).get("viewer_residue_count") or (result.get("metrics") or {}).get("viewer_residue_count"),
        "selected_residue_count": (crop or {}).get("selected_residue_count") or (result.get("metrics") or {}).get("selected_residue_count"),
        "crop_run_id": row.get("run_id"),
        "crop_job_code": row.get("job_code"),
    }
    return source_path if source_path and source_path.exists() else None, details


def _residue_number_sets(pdb_text: str) -> tuple[dict[str, dict[int, int]], dict[str, set[int]]]:
    residues_by_chain: dict[str, list[int]] = {}
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
            continue
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        chain = line[21].strip() or "_"
        if resseq not in residues_by_chain.setdefault(chain, []):
            residues_by_chain[chain].append(resseq)
    index_map = {
        chain: {index + 1: residue for index, residue in enumerate(sorted(values))}
        for chain, values in residues_by_chain.items()
    }
    residue_sets = {chain: set(values) for chain, values in residues_by_chain.items()}
    return index_map, residue_sets


def _viewer_selected_residues(value: object, pdb_text: str) -> set[tuple[str, int]]:
    if not value:
        return set()
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return set()
    if not isinstance(value, dict):
        return set()
    residue_index_map, residue_number_set = _residue_number_sets(pdb_text)
    residues: set[tuple[str, int]] = set()
    for selection in value.get("sequenceSelections") or []:
        if not isinstance(selection, dict):
            continue
        chain = str(selection.get("chainId") or "").strip()
        if not chain:
            continue
        real_numbers = residue_number_set.get(chain, set())
        chain_map = residue_index_map.get(chain, {})
        for residue in selection.get("residues") or []:
            try:
                raw_residue = int(residue)
            except (TypeError, ValueError):
                continue
            residues.add((chain, raw_residue if raw_residue in real_numbers else chain_map.get(raw_residue, raw_residue)))
    return residues


def _compress_highlight_labels(rows: list[dict[str, object]] | set[tuple[str, int]]) -> list[str]:
    by_chain: dict[str, list[int]] = {}
    if isinstance(rows, set):
        items = [{"chain": chain, "residue_number": residue} for chain, residue in rows]
    else:
        items = rows
    for row in items:
        chain = str(row.get("chain") or "").strip()
        if not chain:
            continue
        try:
            residue = int(row.get("residue_number") or 0)
        except (TypeError, ValueError):
            continue
        by_chain.setdefault(chain, []).append(residue)
    labels: list[str] = []
    for chain, residues in sorted(by_chain.items()):
        values = sorted(set(residues))
        if not values:
            continue
        start = previous = values[0]
        for residue in values[1:]:
            if residue == previous + 1:
                previous = residue
                continue
            labels.append(f"{chain}{start}" if start == previous else f"{chain}{start}-{previous}")
            start = previous = residue
        labels.append(f"{chain}{start}" if start == previous else f"{chain}{start}-{previous}")
    return labels


def _mutation_rows_from_table(edited: pd.DataFrame, *, default_mutation: str) -> list[dict[str, object]]:
    if edited.empty or "select" not in edited.columns:
        return []
    rows: list[dict[str, object]] = []
    selected = edited[edited["select"] == True]
    for _, row in selected.iterrows():
        mutation = str(row.get("mutation") or default_mutation or "LYS").strip().upper()
        if len(mutation) == 1 and mutation == "K":
            mutation = "LYS"
        if not mutation:
            continue
        rows.append(
            {
                "chain": str(row.get("chain") or "").strip(),
                "residue_number": int(row.get("residue_number")),
                "insertion_code": str(row.get("insertion_code") or ""),
                "resname": str(row.get("resname") or ""),
                "category": str(row.get("category") or ""),
                "mutation": mutation,
                "reason": "crop_exposed_hydrophobic_mask" if bool(row.get("crop_exposed")) else "manual_table_selection",
            }
        )
    return rows


def _render_target_refolding_result_browser(runs: list[dict[str, object]], *, key_prefix: str) -> None:
    if not runs:
        st.info("No target-refolding runs are available yet.")
        return
    labels = [
        f"{row.get('job_code')} | {row.get('status')} | {row.get('created_at') or row.get('run_id')}"
        for row in runs
    ]
    selected_index = st.selectbox(
        "Run",
        range(len(runs)),
        format_func=lambda index: labels[int(index)],
        key=f"{key_prefix}_run",
    )
    selected_run = runs[int(selected_index)]
    run_dir = Path(str(selected_run["run_dir"]))
    metrics = _metrics_for_run(run_dir)
    warning = _target_refolding_warning(run_dir)
    if warning:
        st.warning(warning)
    st.link_button(
        "Open Full Result",
        result_link(str(selected_run.get("task_group") or "target-refolding"), str(selected_run.get("run_id"))),
    )
    if metrics.empty:
        st.info("Merged metrics are not available yet. The worker may still be running.")
        return
    score_columns = _engine_score_columns(metrics)
    cols = st.columns(4)
    cols[0].metric("Rows", len(metrics))
    cols[1].metric("Score columns", len(score_columns))
    cols[2].metric("Status", str(selected_run.get("status") or ""))
    cols[3].metric("Engine", str(selected_run.get("current_engine") or ""))
    if score_columns:
        score_column = st.selectbox("Engine score", score_columns, key=f"{key_prefix}_score")
        plot_df = metrics[["binder_id", score_column]].copy()
        plot_df[score_column] = pd.to_numeric(plot_df[score_column], errors="coerce")
        plot_df = plot_df.dropna(subset=[score_column]).sort_values(score_column)
        st.altair_chart(
            alt.Chart(plot_df)
            .mark_bar()
            .encode(
                x=alt.X("binder_id:N", sort=None, title="Target chain"),
                y=alt.Y(f"{score_column}:Q", title=score_column),
                tooltip=["binder_id:N", alt.Tooltip(f"{score_column}:Q", format=".3f")],
            )
            .properties(height=260),
            width="stretch",
        )
    st.dataframe(metrics, hide_index=True, width="stretch")
    if "binder_id" not in metrics.columns:
        return
    view_cols = st.columns(2)
    design_id = view_cols[0].selectbox(
        "Target chain",
        [str(value) for value in metrics["binder_id"].dropna().unique()],
        key=f"{key_prefix}_design",
    )
    engine_label = view_cols[1].selectbox(
        "Structure engine",
        list(ENGINE_METRIC_FILES),
        key=f"{key_prefix}_engine",
    )
    prediction = _prediction_path(run_dir, engine_label, str(design_id))
    if prediction is None:
        st.info("No predicted structure is available for that engine/target chain yet.")
        return
    molstar_custom_component(
        structures=[
            StructureVisualization(
                pdb=prediction.read_text(errors="ignore"),
                color="chain-id",
                representation_type="cartoon+ball-and-stick",
            )
        ],
        key=f"{key_prefix}_prediction_{run_dir.name}_{engine_label}_{design_id}",
        height=620,
        show_controls=True,
        download_filename=f"{run_dir.name}_{engine_label}_{design_id}",
    )


import_tab, analysis_tab, masking_tab, refolding_tab = st.tabs(
    ["Import / Prepare", "Target Analysis", "Mutation / Masking", "Target Refolding"],
    key="target_preparation_tabs",
    on_change="rerun",
)

if import_tab.open:
    with import_tab:
        left, right = st.columns([0.9, 1.25], gap="large")
        with left:
            st.subheader("Import")
            source_mode = st.radio(
                "Source",
                ["Download from PDB", "AlphaFold DB", "Upload PDB"],
                horizontal=True,
                label_visibility="collapsed",
            )
            if source_mode == "Download from PDB":
                pdb_id = st.text_input("PDB ID", placeholder="1BRS").strip()
                if st.button("Download structure", disabled=not pdb_id):
                    try:
                        with st.spinner(f"Downloading {pdb_id.upper()} from RCSB PDB..."):
                            _store_source(pdb_id.upper(), download_pdb(pdb_id))
                        st.success(f"Imported {pdb_id.upper()}")
                    except Exception as exc:
                        st.error(str(exc))
            elif source_mode == "AlphaFold DB":
                afdb_id = st.text_input(
                    "UniProt accession or AlphaFold DB PDB URL",
                    placeholder="P0DTC2 or https://alphafold.ebi.ac.uk/files/AF-P0DTC2-F1-model_v4.pdb",
                ).strip()
                if st.button("Download AlphaFold DB structure", disabled=not afdb_id):
                    try:
                        label = Path(afdb_id.rstrip("/")).stem if afdb_id.startswith(("http://", "https://")) else afdb_id.upper()
                        with st.spinner(f"Downloading {label} from AlphaFold DB..."):
                            _store_source(label, download_alphafold_db_pdb(afdb_id))
                        st.success(f"Imported {label}")
                    except Exception as exc:
                        st.error(str(exc))
            else:
                uploaded = st.file_uploader("PDB file", type=["pdb", "ent"])
                if uploaded is not None and st.button("Import uploaded structure"):
                    pdb_text = uploaded.getvalue().decode("utf-8", errors="replace")
                    _store_source(uploaded.name, pdb_text)
                    st.success(f"Imported {uploaded.name}")

            if state["pdb_text"]:
                summary = pdb_summary(state["pdb_text"])
                st.subheader("Prepare")
                chain_options = [chain["chain_id"] for chain in summary["chains"]]
                state["selected_chains"] = st.multiselect(
                    "Chains to keep",
                    options=chain_options,
                    default=[chain for chain in state["selected_chains"] if chain in chain_options] or chain_options,
                )
                state["remove_waters"] = st.checkbox("Remove waters", value=state["remove_waters"])
                state["remove_hetero"] = st.checkbox("Remove hetero atoms", value=state["remove_hetero"])
                state["prepare_msa"] = st.checkbox(
                    "Prepare shared target MSAs",
                    value=state.get("prepare_msa", True),
                    help="Caches one A3M per selected target chain in the shared sequence-hashed MSA repository.",
                )
                nonstandard = detect_nonstandard_residues(_selected_text_for_nonstandard_detection())
                if nonstandard:
                    st.warning(f"Detected {len(nonstandard)} nonstandard residue(s) in the selected target.")
                    with st.expander("Nonstandard residues", expanded=True):
                        st.dataframe(nonstandard, hide_index=True, width="stretch")
                    known_mappable = [residue for residue in nonstandard if residue.get("known_mapping")]
                    state["map_known_modified_residues"] = bool(known_mappable) and st.checkbox(
                        "Repair known non-canonical residues",
                        value=state["map_known_modified_residues"],
                    )
                    state["replace_nonstandard_residues"] = st.checkbox(
                        "Replace nonstandard residues with PDBFixer",
                        value=state["replace_nonstandard_residues"],
                    )
                else:
                    state["map_known_modified_residues"] = False
                    state["replace_nonstandard_residues"] = False
                    st.caption("No nonstandard residues detected in the selected target.")

                st.subheader("Trim")
                for chain in summary["chains"]:
                    chain_id = chain["chain_id"]
                    if chain_id not in state["selected_chains"]:
                        continue
                    residues = chain["residues"]
                    if not residues:
                        continue
                    current = state["trim_ranges"].get(chain_id, [residues[0], residues[-1]])
                    start_default = min(max(int(current[0]), residues[0]), residues[-1])
                    end_default = min(max(int(current[1]), residues[0]), residues[-1])
                    start_col, end_col = st.columns(2)
                    start = start_col.number_input(
                        f"{chain_id} start",
                        min_value=residues[0],
                        max_value=residues[-1],
                        value=start_default,
                        step=1,
                        key=f"target_prep_{chain_id}_start",
                    )
                    end = end_col.number_input(
                        f"{chain_id} end",
                        min_value=residues[0],
                        max_value=residues[-1],
                        value=max(end_default, start),
                        step=1,
                        key=f"target_prep_{chain_id}_end",
                    )
                    if end < start:
                        st.error(f"Chain {chain_id}: end residue must be >= start residue.")
                    state["trim_ranges"][chain_id] = [int(start), int(end)]

                target_name = st.text_input("Target name", value=state["source_name"] or "target")
                can_submit = bool(state["source_path"] and state["selected_chains"] and _trimmed_preview_text().strip())
                if st.button("Create target preparation job", type="primary", disabled=not can_submit):
                    ranges = [f"{selection.chain_id}:{selection.start}-{selection.end}" for selection in _residue_selections()]
                    run_dir = enqueue_target_preparation(
                        Path(state["source_path"]),
                        target_name=target_name.strip() or state["source_name"] or "target",
                        keep_chains=state["selected_chains"],
                        residue_range_text=", ".join(ranges),
                        remove_waters=state["remove_waters"],
                        remove_hetero=state["remove_hetero"],
                        map_known_modified_residues=state["map_known_modified_residues"],
                        replace_nonstandard_residues=state["replace_nonstandard_residues"],
                        prepare_msa=state.get("prepare_msa", True),
                    )
                    spawn_worker_for_run(run_dir)
                    st.success(f"Target preparation queued: {run_dir.name}")
                    st.markdown(f"[Open result](/results?task_group=target-prep&run_id={run_dir.name})")

        with right:
            st.subheader("Preview")
            if not state["pdb_text"]:
                st.info("Import a target structure to show the Mol* viewer.")
            else:
                preview_mode = st.radio("Preview", ["Trimmed preview", "Original"], horizontal=True, label_visibility="collapsed")
                pdb_text = _trimmed_preview_text() if preview_mode == "Trimmed preview" else state["pdb_text"]
                summary = pdb_summary(pdb_text)
                cols = st.columns(4)
                cols[0].metric("Chains", len(summary["chains"]))
                cols[1].metric("Residues", sum(chain["residue_count"] for chain in summary["chains"]))
                cols[2].metric("Atoms", summary["atom_count"])
                cols[3].metric("Waters", summary["water_count"])
                molstar_custom_component(
                    structures=[
                        StructureVisualization(
                            pdb=pdb_text,
                            color="chain-id",
                            representation_type="cartoon+ball-and-stick",
                        )
                    ],
                    key=f"target_prep_viewer_{preview_mode}_{state['source_name']}",
                    height=680,
                    show_controls=True,
                    download_filename=f"{state['source_name'] or 'target'}_preview",
                )

        st.divider()
        st.subheader("Target Library")
        st.caption(
            "Select target rows to remove completed target-preparation jobs from the registry. "
            "Rows generated from bundled benchmark/reference files are shown for context but are not deletable here."
        )
        _render_target_library_manager(key_prefix="target_import_library")

def _render_target_masking_tab() -> None:
    targets = ppi_target_jobs()
    if not targets:
        st.info("No prepared, imported, cropped, or benchmark targets are available yet.")
    else:
        st.subheader("Target Mutation And Crop-Edge Masking")
        st.caption(
            "Create a mutated target copy for design/refolding. Lysine masking is recommended for artificial crop-exposed buried residues."
        )
        target_options = [
            row
            for row in targets
            if Path(str(row.get("target_pdb") or "")).expanduser().exists()
        ]
        if not target_options:
            st.info("No target PDB paths from the target registry are currently readable.")
            return
        filter_cols = st.columns([1.3, 3.0])
        category_options = _target_category_options({_target_category(row) for row in target_options})
        selected_categories = filter_cols[0].multiselect(
            "Target categories",
            category_options,
            default=category_options,
            key="target_masking_categories",
        )
        search_text = filter_cols[1].text_input(
            "Search targets",
            value="",
            placeholder="Target name, source, job code, run id, or path",
            key="target_masking_search",
        )
        filtered_target_options = [
            row
            for row in target_options
            if (not selected_categories or _target_category(row) in selected_categories)
            and (not search_text.strip() or search_text.strip().lower() in _target_search_blob(row))
        ]
        if not filtered_target_options:
            st.info("No targets match the current filters.")
            return
        target_rows = _target_chain_rows(filtered_target_options)
        if target_rows.empty:
            st.info("No chain rows are available for the current target filters.")
            return
        edited_targets = st.data_editor(
            target_rows.drop(columns=["_target_index"]),
            hide_index=True,
            width="stretch",
            height=290,
            disabled=["target", "chain", "aa_length", "fragments", "breaks", "source", "category", "job", "records", "path"],
            column_config={
                "select": st.column_config.CheckboxColumn("Select"),
                "aa_length": st.column_config.NumberColumn("AA", format="%d"),
                "fragments": st.column_config.NumberColumn("Fragments", format="%d"),
                "breaks": st.column_config.NumberColumn("Breaks", format="%d"),
                "path": st.column_config.TextColumn("PDB path", width="large"),
            },
            key="target_masking_target_table",
        )
        selected_target_rows = edited_targets[edited_targets["select"] == True] if "select" in edited_targets.columns else pd.DataFrame()
        if selected_target_rows.empty:
            st.info("Select one target/chain row to inspect mutations.")
            return
        selected_row = selected_target_rows.iloc[0]
        if len(selected_target_rows) > 1:
            st.warning("Multiple rows selected; using the first selected target/chain.")
        row_matches = target_rows[
            (target_rows["target"].astype(str) == str(selected_row.get("target")))
            & (target_rows["chain"].astype(str) == str(selected_row.get("chain")))
            & (target_rows["path"].astype(str) == str(selected_row.get("path")))
        ]
        if row_matches.empty:
            st.warning("The selected target row could not be resolved.")
            return
        selected_target = filtered_target_options[int(row_matches.iloc[0]["_target_index"])]
        selected_chain = str(selected_row.get("chain") or "").strip()
        selected_target_pdb = _target_path(selected_target)
        selected_target_text = selected_target_pdb.read_text(errors="ignore")

        parent_reference, parent_details = _crop_source_reference(selected_target)
        details_cols = st.columns(4)
        details_cols[0].metric("Category", _target_category(selected_target))
        details_cols[1].metric("Run", str(selected_target.get("run_id") or selected_target.get("job_code") or "n/a"))
        details_cols[2].metric("Source", str(selected_target.get("source_label") or ""))
        details_cols[3].metric("Chain", selected_chain or "all")
        st.caption(f"Selected PDB: `{selected_target_pdb}`")
        if parent_reference is not None:
            st.success(f"Crop parent/source target detected: `{parent_reference}`")
            parent_counts = [
                f"selected residues: {parent_details.get('selected_residue_count')}"
                if parent_details.get("selected_residue_count") is not None
                else "",
                f"manual: {parent_details.get('manual_residue_count')}"
                if parent_details.get("manual_residue_count") is not None
                else "",
                f"Mol*: {parent_details.get('viewer_residue_count')}"
                if parent_details.get("viewer_residue_count") is not None
                else "",
            ]
            st.caption("; ".join(part for part in parent_counts if part))
        elif _target_category(selected_target) == "cropped":
            st.warning("This looks like a cropped target, but its parent/source PDB was not found on disk.")

        reference_choices: list[tuple[str, Path | None]] = []
        if parent_reference is not None:
            reference_choices.append((f"Use crop parent/source target | {parent_reference.name}", parent_reference))
        reference_choices.append(("Use selected target only", None))
        seen_reference_paths = {str(parent_reference.resolve())} if parent_reference is not None else set()
        for row in target_options:
            path = _target_path(row)
            try:
                resolved = str(path.resolve())
            except Exception:
                resolved = str(path)
            if resolved in seen_reference_paths:
                continue
            seen_reference_paths.add(resolved)
            reference_choices.append((_target_option_label(row), path))
        reference_choice = st.selectbox(
            "Full/reference target for crop-difference scoring",
            range(len(reference_choices)),
            format_func=lambda index: reference_choices[int(index)][0],
            key="target_masking_reference",
            help="Choose the full un-cropped target when mutating a cropped/trimmed target. The app suggests residues that lost buried neighbors during the crop.",
        )
        reference_pdb = reference_choices[int(reference_choice)][1]

        controls = st.columns([1, 1, 1, 1])
        default_mutation = controls[0].selectbox("Mask mutation", ["LYS", "ARG", "GLU", "ALA", "GLY"], index=0)
        show_mode = controls[1].selectbox(
            "Table preset",
            ["Suggested crop masks", "Hydrophobic residues", "Surface-exposed residues", "All residues"],
            index=0,
        )
        rebuild_atoms = controls[2].checkbox(
            "Rebuild sidechains",
            value=True,
            help="Uses PDBFixer/OpenMM when available after mutating residue names and pruning incompatible sidechain atoms.",
        )
        neighbor_cutoff = controls[3].number_input("Neighbor cutoff Å", min_value=3.0, max_value=8.0, value=5.0, step=0.5)

        exposure_rows = crop_exposure_table(
            selected_target_pdb,
            reference_pdb=reference_pdb,
            neighbor_cutoff_angstrom=float(neighbor_cutoff),
        )
        for row in exposure_rows:
            if row.get("select"):
                row["mutation"] = default_mutation
        exposure_df = pd.DataFrame(exposure_rows)
        if exposure_df.empty:
            st.warning("No protein residues were found in the selected target.")
            return
        if selected_chain and selected_chain in set(exposure_df["chain"].astype(str)):
            exposure_df = exposure_df[exposure_df["chain"].astype(str) == selected_chain].copy()

        if show_mode == "Suggested crop masks":
            table_df = exposure_df[exposure_df["suggested_mask"] == True].copy()
        elif show_mode == "Hydrophobic residues":
            table_df = exposure_df[exposure_df["resname"].isin(sorted(HYDROPHOBIC_RESIDUES))].copy()
            table_df["select"] = False
            table_df["mutation"] = default_mutation
        elif show_mode == "Surface-exposed residues":
            table_df = exposure_df[exposure_df["cropped_neighbors"] <= 9].copy()
            table_df["select"] = False
            table_df["mutation"] = default_mutation
        else:
            table_df = exposure_df.copy()
            table_df["select"] = False
            table_df["mutation"] = table_df["mutation"].replace("", default_mutation)

        metric_cols = st.columns(4)
        metric_cols[0].metric("Residues", len(exposure_df))
        metric_cols[1].metric("Crop-exposed", int(exposure_df["crop_exposed"].sum()))
        metric_cols[2].metric("Suggested masks", int(exposure_df["suggested_mask"].sum()))
        metric_cols[3].metric("Hydrophobics", int(exposure_df["resname"].isin(sorted(HYDROPHOBIC_RESIDUES)).sum()))
        if reference_pdb is None:
            st.info("Choose a full/reference target to calculate crop-induced exposure. Manual mutation and hydrophobic/surface presets still work.")
        elif not bool(exposure_df["suggested_mask"].any()):
            st.info("No hydrophobic crop-exposed residues were suggested with the current thresholds.")

        st.markdown("**Suggested / Manual Table**")
        edited_mutations = st.data_editor(
            table_df[
                [
                    "select",
                    "chain",
                    "residue_number",
                    "resname",
                    "category",
                    "full_neighbors",
                    "cropped_neighbors",
                    "removed_neighbors",
                    "crop_exposed",
                    "mutation",
                ]
            ],
            hide_index=True,
            width="stretch",
            height=310,
            disabled=[
                "chain",
                "residue_number",
                "resname",
                "category",
                "full_neighbors",
                "cropped_neighbors",
                "removed_neighbors",
                "crop_exposed",
            ],
            column_config={
                "select": st.column_config.CheckboxColumn("Mutate"),
                "mutation": st.column_config.SelectboxColumn("To", options=["LYS", "ARG", "GLU", "ALA", "GLY", "SER"]),
                "full_neighbors": st.column_config.NumberColumn("Full nbrs", format="%d"),
                "cropped_neighbors": st.column_config.NumberColumn("Crop nbrs", format="%d"),
                "removed_neighbors": st.column_config.NumberColumn("Lost nbrs", format="%d"),
            },
            key="target_masking_mutation_table",
        )

        manual_text = st.text_area(
            "Manual mutations",
            value=st.session_state.get("target_masking_manual_text", ""),
            placeholder="B177K, B180:LYS, C22A",
            key="target_masking_manual_text",
            help="Items without a target amino acid use the selected mask mutation, for example B177.",
        )

        table_mutations = _mutation_rows_from_table(edited_mutations, default_mutation=default_mutation)
        manual_mutations = []
        try:
            manual_mutations = parse_mutation_text(manual_text, default_target=default_mutation)
        except ValueError as exc:
            st.warning(str(exc))

        selected_table_labels = _compress_highlight_labels(table_mutations)
        viewer_value = molstar_custom_component(
            structures=[
                StructureVisualization(
                    pdb=selected_target_text,
                    color="chain-id",
                    representation_type="cartoon+ball-and-stick",
                    highlighted_selections=selected_table_labels,
                )
            ],
            key=f"target_masking_viewer_{selected_target_pdb}_{reference_pdb}_{show_mode}",
            height=620,
            show_controls=True,
            selection_mode=True,
            download_filename=f"{Path(selected_target_pdb).stem}_masking_input",
        )
        clicked_residues = _viewer_selected_residues(viewer_value, selected_target_text)
        viewer_mutations = [
            {
                "chain": chain,
                "residue_number": residue,
                "insertion_code": "",
                "resname": "",
                "category": "",
                "mutation": default_mutation,
                "reason": "manual_molstar_selection",
            }
            for chain, residue in sorted(clicked_residues)
        ]
        if clicked_residues:
            st.caption("Mol* clicked residues: " + ", ".join(_compress_highlight_labels(clicked_residues)))

        mutation_by_key: dict[tuple[str, int, str], dict[str, object]] = {}
        for row in [*table_mutations, *manual_mutations, *viewer_mutations]:
            key = (str(row.get("chain") or ""), int(row.get("residue_number") or 0), str(row.get("insertion_code") or ""))
            if key[0] and key[1]:
                mutation_by_key[key] = {
                    "chain": key[0],
                    "residue_number": key[1],
                    "insertion_code": key[2],
                    "resname": row.get("resname", ""),
                    "category": row.get("category", ""),
                    "mutation": str(row.get("mutation") or default_mutation).upper(),
                    "reason": row.get("reason", "manual"),
                }
        final_mutations = list(mutation_by_key.values())
        st.info(f"{len(final_mutations)} residue mutation(s) selected: {', '.join(mutation_labels(final_mutations[:12]))}{' ...' if len(final_mutations) > 12 else ''}")

        target_name_default = f"{str(selected_target.get('target_name') or selected_target_pdb.stem)} lysine-mutated"
        target_name = st.text_input("New target name", value=target_name_default, key="target_masking_name")
        if st.button("Create mutated target", type="primary", disabled=not final_mutations, key="target_masking_create"):
            try:
                with st.spinner("Writing mutated target artifacts..."):
                    run_dir = create_masked_target(
                        selected_target_pdb,
                        target_name=target_name.strip() or target_name_default,
                        mutations=final_mutations,
                        reference_pdb=reference_pdb,
                        mutation_source="crop_exposure_and_manual",
                        rebuild_missing_atoms=bool(rebuild_atoms),
                    )
                st.success(f"Created mutated target job {run_dir.name}")
                st.markdown(f"[Open result](/results?task_group=target-prep&run_id={run_dir.name})")
            except Exception as exc:
                st.error(str(exc))


def _render_target_analysis_tab() -> None:
    targets = ppi_target_jobs()
    if not targets:
        st.info("No prepared, imported, cropped, or benchmark targets are available yet.")
        return
    target_options = [row for row in targets if _target_path(row).exists()]
    if not target_options:
        st.info("No target PDB paths from the target registry are currently readable.")
        return
    st.subheader("Target Structure Analysis")
    filter_cols = st.columns([1.3, 3.0])
    category_options = _target_category_options({_target_category(row) for row in target_options})
    selected_categories = filter_cols[0].multiselect(
        "Target categories",
        category_options,
        default=category_options,
        key="target_analysis_categories",
    )
    search_text = filter_cols[1].text_input(
        "Search targets",
        value="",
        placeholder="Target name, source, job code, run id, or path",
        key="target_analysis_search",
    )
    filtered_target_options = [
        row
        for row in target_options
        if (not selected_categories or _target_category(row) in selected_categories)
        and (not search_text.strip() or search_text.strip().lower() in _target_search_blob(row))
    ]
    target_rows = _target_structure_rows(filtered_target_options)
    if target_rows.empty:
        st.info("No target rows are available for the current target filters.")
        return
    table_rows = target_rows.drop(columns=["_target_index", "delete_job"], errors="ignore").copy()
    table_rows.insert(0, "select", False)
    show_only_warnings = st.checkbox("Only targets with warnings", value=False, key="target_analysis_warning_only")
    if show_only_warnings:
        table_rows = table_rows[pd.to_numeric(table_rows["breaks"], errors="coerce").fillna(0) > 0]
    edited_targets = st.data_editor(
        table_rows,
        hide_index=True,
        width="stretch",
        height=310,
        disabled=["target", "chains", "chain_count", "aa_length", "fragments", "breaks", "source", "category", "job", "records", "path"],
        column_config={
            "select": st.column_config.CheckboxColumn("Select"),
            "chain_count": st.column_config.NumberColumn("Chains", format="%d"),
            "aa_length": st.column_config.NumberColumn("AA", format="%d"),
            "fragments": st.column_config.NumberColumn("Fragments", format="%d"),
            "breaks": st.column_config.NumberColumn("Breaks", format="%d"),
            "path": st.column_config.TextColumn("PDB path", width="large"),
        },
        key="target_analysis_target_table",
    )
    selected_target_rows = edited_targets[edited_targets["select"] == True] if "select" in edited_targets.columns else pd.DataFrame()
    if selected_target_rows.empty:
        warning_count = int((pd.to_numeric(target_rows["breaks"], errors="coerce").fillna(0) > 0).sum())
        cols = st.columns(4)
        cols[0].metric("Targets", len(target_rows))
        cols[1].metric("Chains", int(pd.to_numeric(target_rows["chain_count"], errors="coerce").fillna(0).sum()))
        cols[2].metric("Warning targets", warning_count)
        cols[3].metric("Readable PDBs", len({str(path) for path in target_rows["path"]}))
        if st.button("Write reports for visible targets", key="target_analysis_write_visible"):
            written = 0
            errors: list[str] = []
            visible_paths = {str(row.get("path") or "") for _, row in table_rows.iterrows()}
            for _, row in target_rows[target_rows["path"].astype(str).isin(visible_paths)].iterrows():
                if show_only_warnings and int(row.get("breaks") or 0) <= 0:
                    continue
                try:
                    path = Path(str(row.get("path") or "")).expanduser()
                    for chain in [item.strip() for item in str(row.get("chains") or "").split(",") if item.strip()]:
                        _write_target_analysis_report(path, chain, analyze_target_chain(path, chain))
                        written += 1
                except Exception as exc:
                    errors.append(f"{row.get('target')}: {exc}")
            if written:
                st.success(f"Wrote {written} target analysis report(s).")
            if errors:
                st.warning("; ".join(errors[:5]))
        st.info("Select one target structure to inspect its fragment map.")
        return
    selected_row = selected_target_rows.iloc[0]
    row_matches = target_rows[
        (target_rows["target"].astype(str) == str(selected_row.get("target")))
        & (target_rows["path"].astype(str) == str(selected_row.get("path")))
    ]
    if row_matches.empty:
        st.warning("The selected target row could not be resolved.")
        return
    selected_target = filtered_target_options[int(row_matches.iloc[0]["_target_index"])]
    selected_target_pdb = _target_path(selected_target)
    chain_options = [item.strip() for item in str(row_matches.iloc[0].get("chains") or "").split(",") if item.strip()]
    if not chain_options:
        chain_options = [chain["chain_id"] for chain in pdb_summary(selected_target_pdb.read_text(errors="ignore")).get("chains") or []]
    if len(chain_options) > 1:
        selected_chain = st.selectbox(
            "Chain to analyze",
            chain_options,
            key=f"target_analysis_chain_{selected_target_pdb}",
        )
    else:
        selected_chain = chain_options[0] if chain_options else ""
        st.caption(f"Chain to analyze: `{selected_chain or 'n/a'}`")
    if not selected_chain:
        st.warning("The selected target has no analyzable chain.")
        return
    analysis = analyze_target_chain(selected_target_pdb, selected_chain)
    cols = st.columns(5)
    cols[0].metric("Residues", int(analysis.get("sequence_length") or 0))
    cols[1].metric("Fragments", int(analysis.get("fragment_count") or 0))
    cols[2].metric("Breaks", int(analysis.get("break_count") or 0))
    cols[3].metric("Coordinate breaks", int(analysis.get("coordinate_break_count") or 0))
    cols[4].metric("Residue gaps", int(analysis.get("residue_number_break_count") or 0))
    st.caption(f"Selected PDB: `{selected_target_pdb}`")
    msa_entries = _target_msa_entries(selected_target_pdb, selected_chain, analysis)
    with st.expander("MSA depth", expanded=any(str(row.get("status")) == "cached" for row in msa_entries)):
        if not msa_entries:
            st.info("No protein sequence was available for MSA lookup.")
        else:
            msa_table = pd.DataFrame(
                [
                    {
                        "scope": row.get("scope"),
                        "chain": row.get("chain"),
                        "fragment": row.get("fragment"),
                        "source_start": row.get("source_start"),
                        "source_end": row.get("source_end"),
                        "length": row.get("length"),
                        "status": row.get("status"),
                        "sequences": row.get("sequence_count"),
                        "effective_depth": row.get("effective_depth"),
                        "mean_depth": row.get("mean_depth"),
                        "min_depth": row.get("min_depth"),
                        "max_depth": row.get("max_depth"),
                        "mean_identity": row.get("mean_identity"),
                        "source": row.get("cache_source"),
                        "msa_path": row.get("msa_path"),
                        "plot_png": row.get("plot_png"),
                    }
                    for row in msa_entries
                ]
            )
            st.dataframe(
                msa_table,
                hide_index=True,
                width="stretch",
                column_config={
                    "length": st.column_config.NumberColumn("AA", format="%d"),
                    "source_start": st.column_config.NumberColumn("Start", format="%d"),
                    "source_end": st.column_config.NumberColumn("End", format="%d"),
                    "sequences": st.column_config.NumberColumn("Sequences", format="%d"),
                    "effective_depth": st.column_config.NumberColumn("Depth excl. query", format="%d"),
                    "mean_depth": st.column_config.NumberColumn("Mean depth", format="%.1f"),
                    "min_depth": st.column_config.NumberColumn("Min depth", format="%d"),
                    "max_depth": st.column_config.NumberColumn("Max depth", format="%d"),
                    "mean_identity": st.column_config.NumberColumn("Mean identity", format="%.2f"),
                    "msa_path": st.column_config.TextColumn("MSA path", width="large"),
                    "plot_png": st.column_config.TextColumn("PNG", width="large"),
                },
            )
            plot_options = [
                index
                for index, row in enumerate(msa_entries)
                if str(row.get("plot_png") or "").strip() and Path(str(row.get("plot_png"))).exists()
            ]
            if plot_options:
                selected_plot_index = st.selectbox(
                    "MSA plot",
                    plot_options,
                    format_func=lambda index: (
                        f"{msa_entries[int(index)].get('scope')} "
                        f"{msa_entries[int(index)].get('chain')}"
                        f"{':' + str(msa_entries[int(index)].get('fragment')) if msa_entries[int(index)].get('fragment') not in ('', None) else ''}"
                        f" ({msa_entries[int(index)].get('effective_depth')} seqs)"
                    ),
                    key=f"target_analysis_msa_plot_{selected_target_pdb}_{selected_chain}",
                )
                selected_msa = msa_entries[int(selected_plot_index)]
                st.caption(f"MSA source: `{selected_msa.get('msa_path')}`")
                png_path = Path(str(selected_msa.get("plot_png") or ""))
                st.image(str(png_path), caption=f"MSA depth image: {png_path}", use_container_width=True)
            else:
                st.info("No cached A3M matched the selected chain or fragments. Refolding runs that require target MSAs can create/populate this cache.")
    report_path = _target_analysis_report_path(selected_target_pdb, selected_chain)
    report_cols = st.columns([1, 1.4, 2.6])
    if report_cols[0].button("Write analysis report", key="target_analysis_write_selected"):
        written = _write_target_analysis_report(selected_target_pdb, selected_chain, analysis)
        st.success(f"Wrote `{written}`")
    split_default_name = f"{str(selected_target.get('target_name') or selected_target_pdb.stem)} chain {selected_chain} as fragment chains"
    if report_cols[1].button(
        "Create one multi-chain target",
        key="target_analysis_split_fragments",
        disabled=int(analysis.get("fragment_count") or 0) < 2,
        help="Writes one new target PDB. Detected fragments become separate chains inside that single structure.",
    ):
        try:
            run_dir = split_target_chain_fragments(
                selected_target_pdb,
                selected_chain,
                target_name=split_default_name,
                source_label=f"Split fragments from {selected_target.get('target_name') or selected_target_pdb.stem} chain {selected_chain}",
            )
            st.success(f"Created one split-fragment target structure in job {run_dir.name}")
            st.markdown(f"[Open result](/results?task_group=target-prep&run_id={run_dir.name})")
        except Exception as exc:
            st.error(str(exc))
    if report_path.exists():
        report_cols[2].caption(f"Existing report: `{report_path}`")
    for warning in analysis.get("warnings") or []:
        st.warning(str(warning))
    fragments = pd.DataFrame(analysis.get("fragments") or [])
    if not fragments.empty:
        fragment_view = fragments.drop(columns=["break_before", "sequence"], errors="ignore")
        st.dataframe(fragment_view, hide_index=True, width="stretch")
    breaks = pd.DataFrame(analysis.get("breaks") or [])
    if not breaks.empty:
        st.subheader("Breaks")
        st.dataframe(
            breaks,
            hide_index=True,
            width="stretch",
            column_config={
                "ca_distance": st.column_config.NumberColumn("CA distance Å", format="%.2f"),
                "peptide_distance": st.column_config.NumberColumn("C-N distance Å", format="%.2f"),
            },
        )
    break_residues = _target_break_residues(analysis)
    if break_residues:
        st.caption(
            "Highlighted break-boundary residues: "
            + ", ".join(f"{chain}{residue}" for chain, residues in break_residues.items() for residue in residues)
        )
    preview_text = filter_pdb_text(
        selected_target_pdb.read_text(errors="ignore"),
        keep_chains=None,
        remove_waters=True,
        remove_hetero=True,
    )
    highlight_chains = [
        ChainVisualization(
            chain_id=chain,
            color="uniform",
            color_params={"value": "0xff2d7a"},
            representation_type="ball-and-stick",
            residues=residues,
            label="break boundary",
        )
        for chain, residues in break_residues.items()
    ]
    molstar_custom_component(
        structures=[
            StructureVisualization(
                pdb=preview_text,
                color="chain-id",
                representation_type="cartoon+ball-and-stick",
                highlighted_selections=_target_break_selection_labels(break_residues),
                chains=highlight_chains or None,
            )
        ],
        key=f"target_analysis_viewer_{selected_target_pdb}_{selected_chain}",
        height=620,
        show_controls=True,
        force_reload=True,
        sequence_highlight_color="#ff2d7a" if break_residues else None,
        sequence_highlight_indices=[residue for residues in break_residues.values() for residue in residues],
        download_filename=f"{selected_row.get('target')}_analysis",
    )
    with st.expander("Repair route", expanded=bool(analysis.get("warnings"))):
        st.markdown(
            "- For fixed-sequence target reconstruction, use Target Refolding with Boltz-2, OpenFold-3, Protenix, RF3, ESMFold2, or AF3.\n"
            "- For disconnected coordinates, prefer separate fragment chains for diagnostic runs, then compare against a full-sequence refolded target.\n"
            "- PyRosetta relax can clean local geometry, but it should not be trusted to invent a large missing backbone segment from sequence alone.\n"
            "- ProteinMPNN/LigandMPNN design sequences for a backbone; they are not fixed-sequence backbone repair tools."
        )


if analysis_tab.open:
    with analysis_tab:
        _render_target_analysis_tab()


if masking_tab.open:
    with masking_tab:
        _render_target_masking_tab()

if refolding_tab.open:
    with refolding_tab:
        select_tab, engines_tab, run_tab, refold_results_tab = st.tabs(["Select Target", "Engines", "Run Refolding", "Results"])
        targets = ppi_target_jobs()
        target_rows = _target_structure_rows(targets) if targets else pd.DataFrame()
        selected_entries: list[dict] = []

        with select_tab:
            if target_rows.empty:
                st.info("No prepared, imported, cropped, or benchmark targets are available yet.")
            else:
                filter_cols = st.columns([1, 4])
                category_options = _target_category_options(str(value) for value in target_rows["category"].dropna().unique() if str(value))
                selected_categories = filter_cols[0].multiselect("Category", category_options, default=category_options)
                search_text = filter_cols[1].text_input("Search targets", value="", placeholder="Target name, chain, source, or job code")
                filtered_rows = target_rows.copy()
                if selected_categories:
                    filtered_rows = filtered_rows[filtered_rows["category"].astype(str).isin(selected_categories)]
                if search_text.strip():
                    needle = search_text.strip().lower()
                    filtered_rows = filtered_rows[
                        filtered_rows.apply(lambda row: needle in " ".join(str(value).lower() for value in row.values), axis=1)
                    ]
                table_rows = filtered_rows.drop(columns=["_target_index", "delete_job"], errors="ignore").copy()
                table_rows.insert(0, "select", False)
                edited = st.data_editor(
                    table_rows,
                    hide_index=True,
                    width="stretch",
                    height=290,
                    disabled=["target", "chains", "chain_count", "aa_length", "fragments", "breaks", "source", "category", "job", "records", "path"],
                    column_config={
                        "select": st.column_config.CheckboxColumn("Select"),
                        "chain_count": st.column_config.NumberColumn("Chains", format="%d"),
                        "aa_length": st.column_config.NumberColumn("AA", format="%d"),
                        "fragments": st.column_config.NumberColumn("Fragments", format="%d"),
                        "breaks": st.column_config.NumberColumn("Breaks", format="%d"),
                        "path": st.column_config.TextColumn("PDB path", width="large"),
                    },
                    key="target_refolding_chain_table",
                )
                selected_entries = _selected_target_structure_entries(edited, filtered_rows, targets)
                if len(selected_entries) > 1:
                    st.warning("Target refolding runs one target structure at a time. Keeping the first selected row.")
                    selected_entries = selected_entries[:1]
                if selected_entries:
                    selected = selected_entries[0]
                    st.session_state["target_refolding_selected_entries"] = selected_entries
                    selected_with_breaks = [entry for entry in selected_entries if _target_break_count(entry) > 0]
                    st.session_state["target_refold_selected_break_count"] = len(selected_with_breaks)
                    if selected_with_breaks:
                        st.warning(
                            f"{len(selected_with_breaks)} selected target structure(s) contain residue-number breaks. "
                            "Default engines are restricted to AF2-IG and Boltz-2 for structure-assisted refolding."
                        )
                    preview_chains = [str(chain) for chain in selected.get("target_entity_chains") or [] if str(chain)] or [str(selected["chain"])]
                    st.markdown(f"**Preview:** {selected['target_name']} chain(s) {','.join(preview_chains)}")
                    target_text = filter_pdb_text(
                        Path(str(selected["target_pdb"])).read_text(errors="ignore"),
                        keep_chains=set(preview_chains),
                        remove_waters=True,
                        remove_hetero=True,
                    )
                    molstar_custom_component(
                        structures=[
                            StructureVisualization(
                                pdb=target_text,
                                color="chain-id",
                                representation_type="cartoon+ball-and-stick",
                            )
                        ],
                        key=f"target_refolding_selected_viewer_{selected['target_pdb']}_{'-'.join(preview_chains)}",
                        height=560,
                        show_controls=True,
                        download_filename=f"{selected['target_name']}_{'_'.join(preview_chains)}",
                    )
                else:
                    st.session_state["target_refolding_selected_entries"] = []
                    st.session_state["target_refold_selected_break_count"] = 0
                    st.info("Select one target structure to configure a refolding run.")

        with engines_tab:
            st.subheader("Engines")
            target_refold_engine_keys = [
                "target_refold_run_af3",
                "target_refold_run_colab",
                "target_refold_run_af2",
                "target_refold_run_esmfold2",
                "target_refold_run_boltz2",
                "target_refold_run_rf3",
                "target_refold_run_openfold3",
                "target_refold_run_protenix",
                "target_refold_run_protenix_v1",
                "target_refold_run_protenix_v2",
                "target_refold_run_boltzgen",
            ]
            if "target_refold_engine_defaults_v1" not in st.session_state:
                for engine_key in target_refold_engine_keys:
                    st.session_state[engine_key] = True
                st.session_state["target_refold_run_boltzgen"] = False
                st.session_state["target_refold_engine_defaults_v1"] = True
            if "target_refold_template_defaults_v2" not in st.session_state:
                st.session_state["target_refold_af2_multimer"] = True
                st.session_state["target_refold_af2_initial_guess"] = True
                st.session_state["target_refold_af2_fragment_template"] = True
                st.session_state["target_refold_af2_layout_template"] = True
                st.session_state["target_refold_boltz_template"] = True
                st.session_state["target_refold_template_defaults_v2"] = True
            for engine_key in target_refold_engine_keys:
                st.session_state.setdefault(engine_key, engine_key != "target_refold_run_boltzgen")
            selected_engine_entries = list(st.session_state.get("target_refolding_selected_entries") or [])
            selected_break_count = sum(1 for entry in selected_engine_entries if _target_break_count(entry) > 0)
            selected_split_fragment_count = sum(
                1
                for entry in selected_engine_entries
                if str(entry.get("source_category") or "") == "split_fragments"
                and len([chain for chain in entry.get("target_entity_chains") or [] if str(chain)]) > 1
            )
            selection_signature = _target_selection_signature(selected_engine_entries)
            if (
                (selected_break_count or selected_split_fragment_count)
                and selection_signature
                and st.session_state.get("target_refold_break_mode_signature") != selection_signature
            ):
                st.session_state["target_refold_chain_break_mode"] = "Fragment chains for broken targets (recommended)"
                st.session_state["target_refold_break_mode_signature"] = selection_signature

            target_template_msa_engine_keys = {
                "target_refold_run_af3",
                "target_refold_run_colab",
                "target_refold_run_esmfold2",
                "target_refold_run_boltz2",
                "target_refold_run_rf3",
                "target_refold_run_protenix_v1",
                "target_refold_run_protenix_v2",
            }
            target_template_only_engine_keys = set(target_template_msa_engine_keys)
            target_template_only_engine_keys.add("target_refold_run_af2")
            target_msa_engine_keys = {
                "target_refold_run_af3",
                "target_refold_run_colab",
                "target_refold_run_esmfold2",
                "target_refold_run_boltz2",
                "target_refold_run_rf3",
                "target_refold_run_openfold3",
                "target_refold_run_protenix",
                "target_refold_run_protenix_v1",
                "target_refold_run_protenix_v2",
            }
            bulk_cols = st.columns([1, 1, 1.5, 1.5, 1.35, 2.65])
            if bulk_cols[0].button("Select all engines", key="target_refold_select_all_engines"):
                for engine_key in target_refold_engine_keys:
                    st.session_state[engine_key] = True
                st.rerun()
            if bulk_cols[1].button("Deselect all engines", key="target_refold_deselect_all_engines"):
                for engine_key in target_refold_engine_keys:
                    st.session_state[engine_key] = False
                st.rerun()
            if bulk_cols[2].button("Template + MSA engines", key="target_refold_select_template_msa_engines"):
                for engine_key in target_refold_engine_keys:
                    st.session_state[engine_key] = engine_key in target_template_msa_engine_keys
                st.session_state["target_refold_af3_templates"] = True
                st.session_state["target_refold_af3_msa"] = True
                st.session_state["target_refold_colab_templates"] = True
                st.session_state["target_refold_colab_msa"] = True
                st.session_state["target_refold_af2_initial_guess"] = True
                st.session_state["target_refold_esm_modes"] = ["initial_guess"]
                st.session_state["target_refold_esm_msa"] = True
                st.session_state["target_refold_boltz_template"] = True
                st.session_state["target_refold_boltz_msa"] = True
                st.session_state["target_refold_rf3_template"] = True
                st.session_state["target_refold_rf3_msa"] = True
                st.session_state["target_refold_protenix_v1_template"] = True
                st.session_state["target_refold_protenix_v1_msa"] = True
                st.session_state["target_refold_protenix_v2_template"] = True
                st.session_state["target_refold_protenix_v2_msa"] = True
                st.session_state["target_refold_openfold3_msa"] = False
                st.session_state["target_refold_protenix_msa"] = False
                st.session_state["target_refold_require_real_msa"] = True
                st.rerun()
            if bulk_cols[3].button("Template-only engines", key="target_refold_select_template_only_engines"):
                for engine_key in target_refold_engine_keys:
                    st.session_state[engine_key] = engine_key in target_template_only_engine_keys
                st.session_state["target_refold_af3_templates"] = True
                st.session_state["target_refold_af3_msa"] = False
                st.session_state["target_refold_colab_templates"] = True
                st.session_state["target_refold_colab_msa"] = False
                st.session_state["target_refold_af2_initial_guess"] = True
                st.session_state["target_refold_esm_modes"] = ["initial_guess"]
                st.session_state["target_refold_esm_msa"] = False
                st.session_state["target_refold_boltz_template"] = True
                st.session_state["target_refold_boltz_msa"] = False
                st.session_state["target_refold_rf3_template"] = True
                st.session_state["target_refold_rf3_msa"] = False
                st.session_state["target_refold_protenix_v1_template"] = True
                st.session_state["target_refold_protenix_v1_msa"] = False
                st.session_state["target_refold_protenix_v2_template"] = True
                st.session_state["target_refold_protenix_v2_msa"] = False
                st.session_state["target_refold_openfold3_msa"] = False
                st.session_state["target_refold_protenix_msa"] = False
                st.session_state["target_refold_require_real_msa"] = False
                st.rerun()
            if bulk_cols[4].button("MSA-only engines", key="target_refold_select_msa_only_engines"):
                for engine_key in target_refold_engine_keys:
                    st.session_state[engine_key] = engine_key in target_msa_engine_keys
                st.session_state["target_refold_af3_templates"] = False
                st.session_state["target_refold_af3_msa"] = True
                st.session_state["target_refold_colab_templates"] = False
                st.session_state["target_refold_colab_msa"] = True
                st.session_state["target_refold_boltz_template"] = False
                st.session_state["target_refold_boltz_msa"] = True
                st.session_state["target_refold_rf3_template"] = False
                st.session_state["target_refold_rf3_msa"] = True
                st.session_state["target_refold_protenix_v1_template"] = False
                st.session_state["target_refold_protenix_v1_msa"] = True
                st.session_state["target_refold_protenix_v2_template"] = False
                st.session_state["target_refold_protenix_v2_msa"] = True
                st.session_state["target_refold_esm_modes"] = ["sequence"]
                st.session_state["target_refold_esm_msa"] = True
                st.session_state["target_refold_openfold3_msa"] = True
                st.session_state["target_refold_protenix_msa"] = True
                st.session_state["target_refold_require_real_msa"] = True
                st.rerun()
            if bulk_cols[5].button("No MSA + no template", key="target_refold_disable_msa_template"):
                st.session_state["target_refold_run_af2"] = False
                st.session_state["target_refold_af3_templates"] = False
                st.session_state["target_refold_af3_msa"] = False
                st.session_state["target_refold_colab_templates"] = False
                st.session_state["target_refold_colab_msa"] = False
                st.session_state["target_refold_esm_modes"] = ["sequence"]
                st.session_state["target_refold_esm_msa"] = False
                st.session_state["target_refold_af2_initial_guess"] = True
                st.session_state["target_refold_af2_fragment_template"] = False
                st.session_state["target_refold_af2_layout_template"] = False
                st.session_state["target_refold_boltz_template"] = False
                st.session_state["target_refold_boltz_msa"] = False
                st.session_state["target_refold_rf3_template"] = False
                st.session_state["target_refold_rf3_msa"] = False
                st.session_state["target_refold_openfold3_msa"] = False
                st.session_state["target_refold_protenix_msa"] = False
                st.session_state["target_refold_protenix_v1_template"] = False
                st.session_state["target_refold_protenix_v1_msa"] = False
                st.session_state["target_refold_protenix_v2_template"] = False
                st.session_state["target_refold_protenix_v2_msa"] = False
                st.session_state["target_refold_require_real_msa"] = False
                st.rerun()
            if selected_break_count:
                st.warning(
                    f"{selected_break_count} selected target structure(s) contain residue-number breaks. "
                    "Fragment-chain mode is selected by default; all compatible refolding engines remain available."
                )
            if selected_split_fragment_count:
                st.warning(
                    f"{selected_split_fragment_count} selected target structure(s) were created by splitting one broken chain into fragments. "
                    "They are forced to run as fragment chains so the staged target stays multi-chain. "
                    "All compatible refolding engines remain available."
                )
                st.session_state["target_refold_chain_break_mode"] = "Fragment chains for broken targets (recommended)"
            if st.session_state.get("target_refold_run_boltzgen", False):
                st.session_state["target_refold_run_boltzgen"] = False
                st.warning(
                    "BoltzGen Fold is disabled for target-only refolding because its fold entrypoint requires "
                    "at least one designed residue; launching it here would fail without producing a structure."
                )
            legacy_chain_break_mode = st.session_state.get("target_refold_chain_break_mode")
            if legacy_chain_break_mode == "Preserve original chain":
                st.session_state["target_refold_chain_break_mode"] = "Legacy single-chain gaps (diagnostic)"
            elif legacy_chain_break_mode == "Split into fragment chains":
                st.session_state["target_refold_chain_break_mode"] = "Fragment chains for broken targets (recommended)"
            elif legacy_chain_break_mode == "Residue-gap single chain":
                st.session_state["target_refold_chain_break_mode"] = "Legacy single-chain gaps (diagnostic)"
            elif legacy_chain_break_mode == "Separate fragment chains":
                st.session_state["target_refold_chain_break_mode"] = "Fragment chains for broken targets (recommended)"
            st.radio(
                "Chain-break mode",
                ["Fragment chains for broken targets (recommended)", "Legacy single-chain gaps (diagnostic)"],
                horizontal=True,
                key="target_refold_chain_break_mode",
                disabled=not selected_engine_entries or bool(selected_split_fragment_count),
                help=(
                    "Fragment chains rewrites each discontinuous crop fragment as its own chain and is the default for fragmented targets. "
                    "Legacy single-chain gaps keeps the original chain with missing residue numbers for explicit diagnostic comparisons only."
                ),
            )

            engine_cols = st.columns(11)
            target_run_af3 = engine_cols[0].checkbox("AlphaFast AF3", value=True, key="target_refold_run_af3")
            target_run_colab = engine_cols[1].checkbox("ColabFold", value=True, key="target_refold_run_colab")
            target_run_af2 = engine_cols[2].checkbox("AF2 template", value=True, key="target_refold_run_af2")
            if target_run_af2:
                st.session_state["target_refold_af2_initial_guess"] = True
            target_run_esmfold2 = engine_cols[3].checkbox("ESMFold2", value=True, key="target_refold_run_esmfold2")
            target_run_boltz2 = engine_cols[4].checkbox("Boltz-2", value=True, key="target_refold_run_boltz2")
            target_run_rf3 = engine_cols[5].checkbox("RF3", value=True, key="target_refold_run_rf3")
            target_run_openfold3 = engine_cols[6].checkbox("OpenFold-3", value=True, key="target_refold_run_openfold3")
            target_run_protenix = engine_cols[7].checkbox("Protenix v0.5", value=True, key="target_refold_run_protenix")
            target_run_protenix_v1 = engine_cols[8].checkbox("Protenix v1", value=True, key="target_refold_run_protenix_v1")
            target_run_protenix_v2 = engine_cols[9].checkbox("Protenix v2", value=True, key="target_refold_run_protenix_v2")
            target_run_boltzgen = engine_cols[10].checkbox(
                "BoltzGen Fold",
                value=False,
                key="target_refold_run_boltzgen",
                disabled=True,
                help="Not available for target-only refolding: BoltzGen Fold requires at least one designed residue in the input mask.",
            )

            msa_enabled = bool(
                (target_run_af3 and bool(st.session_state.get("target_refold_af3_msa", True)))
                or (target_run_colab and bool(st.session_state.get("target_refold_colab_msa", True)))
                or (target_run_boltz2 and bool(st.session_state.get("target_refold_boltz_msa", True)))
                or (target_run_esmfold2 and bool(st.session_state.get("target_refold_esm_msa", True)))
                or (target_run_rf3 and bool(st.session_state.get("target_refold_rf3_msa", True)))
                or (target_run_openfold3 and bool(st.session_state.get("target_refold_openfold3_msa", True)))
                or (target_run_protenix and bool(st.session_state.get("target_refold_protenix_msa", True)))
                or (target_run_protenix_v1 and bool(st.session_state.get("target_refold_protenix_v1_msa", True)))
                or (target_run_protenix_v2 and bool(st.session_state.get("target_refold_protenix_v2_msa", True)))
            )
            with st.expander("MSA Reference Data", expanded=True):
                msa_cols = st.columns(3)
                msa_cols[0].text_input(
                    "MSA repository",
                    value=str(MSA_REPOSITORY_DIR),
                    key="target_refold_msa_repository",
                    disabled=not msa_enabled,
                )
                msa_cols[1].text_input(
                    "Alignment/MMseqs DB dir",
                    value=str(ALPHAFAST_DB_DIR),
                    key="target_refold_alphafast_db",
                    disabled=not msa_enabled,
                )
                msa_cols[2].checkbox(
                    "Require real MSAs",
                    value=True,
                    key="target_refold_require_real_msa",
                    disabled=not msa_enabled,
                )

            with st.expander("AlphaFast AF3 Settings", expanded=target_run_af3):
                af3_cols = st.columns(4)
                af3_cols[0].text_input(
                    "AF3 weights dir",
                    value=str(ALPHAFAST_WEIGHTS_DIR),
                    disabled=not target_run_af3,
                    key="target_refold_af3_weights",
                )
                af3_cols[1].number_input(
                    "AF3 recycles",
                    min_value=1,
                    max_value=48,
                    value=10,
                    step=1,
                    disabled=not target_run_af3,
                    key="target_refold_af3_recycles",
                )
                af3_cols[2].checkbox(
                    "Use templates",
                    value=True,
                    disabled=not target_run_af3,
                    key="target_refold_af3_templates",
                    help="Embeds the staged input target chains as AF3 templates.",
                )
                af3_cols[3].checkbox(
                    "Use target MSAs",
                    value=True,
                    disabled=not target_run_af3,
                    key="target_refold_af3_msa",
                    help="When off, AlphaFast AF3 runs with query-only/no target MSA input.",
                )

            with st.expander("ColabFold Settings", expanded=target_run_colab):
                colab_cols = st.columns(5)
                colab_cols[0].text_input(
                    "ColabFold / AF2 model cache",
                    value=str(COLABFOLD_CACHE_DIR),
                    disabled=not target_run_colab,
                    key="target_refold_colab_cache",
                )
                colab_cols[1].number_input("ColabFold recycles", 1, 48, 3, disabled=not target_run_colab, key="target_refold_colab_recycles")
                colab_cols[2].number_input("ColabFold models", 1, 5, 3, disabled=not target_run_colab, key="target_refold_colab_models")
                colab_templates = colab_cols[3].checkbox(
                    "Use templates",
                    value=True,
                    disabled=not target_run_colab,
                    key="target_refold_colab_templates",
                    help="Uses the staged input target as the template.",
                )
                colab_msa = colab_cols[4].checkbox(
                    "Use target MSAs",
                    value=True,
                    disabled=not target_run_colab,
                    key="target_refold_colab_msa",
                    help="When off, ColabFold does not request or inject real target MSAs.",
                )

            with st.expander("AF2 Template Settings", expanded=target_run_af2):
                af2_cols = st.columns(5)
                af2_cols[0].number_input("AF2 recycles", 1, 24, 3, disabled=not target_run_af2, key="target_refold_af2_recycles")
                af2_cols[1].checkbox("AF2 multimer", value=True, disabled=not target_run_af2, key="target_refold_af2_multimer")
                af2_cols[2].checkbox(
                    "Whole input initial guess",
                    value=True,
                    disabled=True,
                    key="target_refold_af2_initial_guess",
                    help="AF2 target refolding always uses initial-guess conditioning. Deselect AF2 to omit it from no-template/no-MSA runs.",
                )
                af2_cols[3].checkbox(
                    "Template fragments",
                    value=True,
                    disabled=not target_run_af2,
                    key="target_refold_af2_fragment_template",
                )
                af2_cols[4].checkbox(
                    "Template fragment layout",
                    value=True,
                    disabled=not target_run_af2,
                    key="target_refold_af2_layout_template",
                )

            with st.expander("ESMFold2 Settings", expanded=target_run_esmfold2):
                target_esm_modes = st.multiselect(
                    "Modes",
                    ["sequence", "initial_guess"],
                    default=["initial_guess"],
                    disabled=not target_run_esmfold2,
                    format_func={"sequence": "Sequence only", "initial_guess": "Selected-target distogram"}.get,
                    key="target_refold_esm_modes",
                    help="Selected-target distogram is ESMFold2's template-like mode: it conditions on the selected target structure, without binder/interface geometry.",
                )
                esm_cols = st.columns(4)
                esm_cols[0].checkbox(
                    "Use ESMFold2 target MSAs",
                    value=True,
                    disabled=not target_run_esmfold2,
                    key="target_refold_esm_msa",
                    help=(
                        "Pass prepared per-target-chain A3M files into ESMFold2 ProteinInput.msa. "
                        "This is per-chain target MSA conditioning, not a ColabFold-style paired multimer A3M; "
                        "missing, query-only, or mismatched MSAs are skipped and noted in the ESMFold2 metrics."
                    ),
                )
                esm_cols[1].number_input("ESMFold2 sampling steps", 1, 256, 68, disabled=not target_run_esmfold2, key="target_refold_esm_steps")
                esm_cols[2].number_input("ESMFold2 recycling loops", 1, 64, 10, disabled=not target_run_esmfold2, key="target_refold_esm_loops")
                esm_cols[3].number_input("ESMFold2 seed", 0, 999999, 0, disabled=not target_run_esmfold2, key="target_refold_esm_seed")

            with st.expander("Boltz-2 Settings", expanded=target_run_boltz2):
                boltz_cols = st.columns(6)
                boltz_cols[0].checkbox("Use templates", value=True, disabled=not target_run_boltz2, key="target_refold_boltz_template", help="Uses the staged input target as the template.")
                boltz_cols[1].checkbox("Use target MSAs", value=True, disabled=not target_run_boltz2, key="target_refold_boltz_msa")
                boltz_cols[2].number_input("Recycling steps", 1, 48, 10, disabled=not target_run_boltz2, key="target_refold_boltz_recycles")
                boltz_cols[3].number_input("Sampling steps", 1, 1000, 100, disabled=not target_run_boltz2, key="target_refold_boltz_steps")
                boltz_cols[4].number_input("Diffusion samples", 1, 20, 3, disabled=not target_run_boltz2, key="target_refold_boltz_samples")
                boltz_cols[5].checkbox("Write full PAE", value=True, disabled=not target_run_boltz2, key="target_refold_boltz_full_pae")

            with st.expander("RF3 Settings", expanded=target_run_rf3):
                rf3_cols = st.columns(6)
                rf3_cols[0].text_input("RF3 checkpoint", value=str(RF3_CHECKPOINT), disabled=not target_run_rf3, key="target_refold_rf3_checkpoint")
                rf3_cols[1].checkbox("Use templates", value=True, disabled=not target_run_rf3, key="target_refold_rf3_template", help="Uses the staged input target chains as RF3 template coordinates.")
                rf3_cols[2].checkbox("Use target MSAs", value=True, disabled=not target_run_rf3, key="target_refold_rf3_msa")
                rf3_cols[3].number_input("RF3 recycles", 1, 48, 10, disabled=not target_run_rf3, key="target_refold_rf3_recycles")
                rf3_cols[4].number_input("RF3 diffusion steps", 1, 1000, 50, disabled=not target_run_rf3, key="target_refold_rf3_steps")
                rf3_cols[5].number_input("RF3 samples", 1, 20, 5, disabled=not target_run_rf3, key="target_refold_rf3_samples")
                st.number_input("RF3 seed", 0, 999999, 0, disabled=not target_run_rf3, key="target_refold_rf3_seed")

            with st.expander("OpenFold-3 Settings", expanded=target_run_openfold3):
                of3_cols = st.columns(5)
                of3_cols[0].text_input("OpenFold-3 checkpoint", value=str(OPENFOLD3_CHECKPOINT), disabled=not target_run_openfold3, key="target_refold_openfold3_checkpoint")
                of3_cols[1].checkbox("Use target MSAs", value=True, disabled=not target_run_openfold3, key="target_refold_openfold3_msa")
                of3_cols[2].number_input("Diffusion samples", 1, 20, 5, disabled=not target_run_openfold3, key="target_refold_openfold3_samples")
                of3_cols[3].number_input("Model seeds", 1, 20, 1, disabled=not target_run_openfold3, key="target_refold_openfold3_seeds")
                of3_cols[4].number_input("Recycles", 1, 48, 3, disabled=not target_run_openfold3, key="target_refold_openfold3_recycles")
                st.checkbox("Use MSA server", value=False, disabled=not target_run_openfold3, key="target_refold_openfold3_msa_server")

            with st.expander("Protenix v0.5 Settings", expanded=target_run_protenix):
                protenix_cols = st.columns(4)
                protenix_cols[0].checkbox("Use target MSAs", value=True, disabled=not target_run_protenix, key="target_refold_protenix_msa")
                protenix_cols[1].number_input("Pairformer cycles", 1, 48, 3, disabled=not target_run_protenix, key="target_refold_protenix_cycle")
                protenix_cols[2].number_input("Diffusion steps", 1, 1000, 50, disabled=not target_run_protenix, key="target_refold_protenix_steps")
                protenix_cols[3].number_input("Samples", 1, 20, 5, disabled=not target_run_protenix, key="target_refold_protenix_samples")

            with st.expander("Protenix v1 Settings", expanded=target_run_protenix_v1):
                protenix_v1_cols = st.columns(6)
                protenix_v1_cols[0].text_input("Model", value=PROTENIX_V1_MODEL, disabled=not target_run_protenix_v1, key="target_refold_protenix_v1_model")
                protenix_v1_cols[1].checkbox("Use target MSAs", value=True, disabled=not target_run_protenix_v1, key="target_refold_protenix_v1_msa")
                protenix_v1_cols[2].checkbox("Use templates", value=True, disabled=not target_run_protenix_v1, key="target_refold_protenix_v1_template")
                protenix_v1_cols[3].number_input("Pairformer cycles", 1, 48, 10, disabled=not target_run_protenix_v1, key="target_refold_protenix_v1_cycle")
                protenix_v1_cols[4].number_input("Diffusion steps", 1, 1000, 200, disabled=not target_run_protenix_v1, key="target_refold_protenix_v1_steps")
                protenix_v1_cols[5].number_input("Samples", 1, 20, 5, disabled=not target_run_protenix_v1, key="target_refold_protenix_v1_samples")

            with st.expander("Protenix v2 Settings", expanded=target_run_protenix_v2):
                protenix_v2_cols = st.columns(6)
                protenix_v2_cols[0].text_input("Model", value=PROTENIX_V2_MODEL, disabled=not target_run_protenix_v2, key="target_refold_protenix_v2_model")
                protenix_v2_cols[1].checkbox("Use target MSAs", value=True, disabled=not target_run_protenix_v2, key="target_refold_protenix_v2_msa")
                protenix_v2_cols[2].checkbox("Use templates", value=True, disabled=not target_run_protenix_v2, key="target_refold_protenix_v2_template")
                protenix_v2_cols[3].number_input("Pairformer cycles", 1, 48, 10, disabled=not target_run_protenix_v2, key="target_refold_protenix_v2_cycle")
                protenix_v2_cols[4].number_input("Diffusion steps", 1, 1000, 200, disabled=not target_run_protenix_v2, key="target_refold_protenix_v2_steps")
                protenix_v2_cols[5].number_input("Samples", 1, 20, 5, disabled=not target_run_protenix_v2, key="target_refold_protenix_v2_samples")

            with st.expander("BoltzGen Fold Settings", expanded=target_run_boltzgen):
                boltzgen_cols = st.columns(4)
                boltzgen_cols[0].checkbox(
                    "Use templates",
                    value=True,
                    disabled=True,
                    key="target_refold_boltzgen_template",
                    help="BoltzGen Fold runs in template mode; the staged input target is always used as the template.",
                )
                boltzgen_cols[1].number_input("Recycling steps", 1, 48, 3, disabled=not target_run_boltzgen, key="target_refold_boltzgen_recycles")
                boltzgen_cols[2].number_input("Sampling steps", 1, 1000, 100, disabled=not target_run_boltzgen, key="target_refold_boltzgen_steps")
                boltzgen_cols[3].number_input("Diffusion samples", 1, 20, 3, disabled=not target_run_boltzgen, key="target_refold_boltzgen_samples")

            with st.expander("Metrics", expanded=True):
                st.checkbox("Predicted Rosetta metrics", value=False, key="target_refold_rosetta")

        with run_tab:
            selected_entries = list(st.session_state.get("target_refolding_selected_entries") or [])
            selected_break_count = sum(1 for entry in selected_entries if _target_break_count(entry) > 0)
            selected_split_fragment_count = sum(
                1
                for entry in selected_entries
                if str(entry.get("source_category") or "") == "split_fragments"
                and len([chain for chain in entry.get("target_entity_chains") or [] if str(chain)]) > 1
            )
            selected_engines = [
                label
                for label, enabled in [
                    ("AF3", st.session_state.get("target_refold_run_af3", True)),
                    ("ColabFold", st.session_state.get("target_refold_run_colab", True)),
                    ("AF2 template", st.session_state.get("target_refold_run_af2", True)),
                    ("ESMFold2", st.session_state.get("target_refold_run_esmfold2", True)),
                    ("Boltz-2", st.session_state.get("target_refold_run_boltz2", True)),
                    ("RF3", st.session_state.get("target_refold_run_rf3", True)),
                    ("OpenFold-3", st.session_state.get("target_refold_run_openfold3", True)),
                    ("Protenix v0.5", st.session_state.get("target_refold_run_protenix", True)),
                    ("Protenix v1", st.session_state.get("target_refold_run_protenix_v1", True)),
                    ("Protenix v2", st.session_state.get("target_refold_run_protenix_v2", True)),
                    ("BoltzGen Fold", st.session_state.get("target_refold_run_boltzgen", False)),
                ]
                if enabled
            ]
            eval_name = st.text_input("Evaluation name", value="Target refolding evaluation", key="target_refold_eval_name")
            chain_break_mode = str(
                st.session_state.get("target_refold_chain_break_mode")
                or "Fragment chains for broken targets (recommended)"
            )
            split_chain_breaks = bool(
                selected_split_fragment_count
                or chain_break_mode
                in {
                    "Fragment chains for broken targets (recommended)",
                    "Separate fragment chains",
                    "Split into fragment chains",
                }
            )
            target_gpu = gpu_run_panel(key="target_refold_queue", default="0")
            st.info(
                f"{len(selected_entries):,} target structure(s) selected. "
                f"Selected engines: {', '.join(selected_engines) or 'none'}. "
                f"Chain breaks: {'separate fragment chains' if split_chain_breaks else 'residue-gap single chain'}."
            )
            if (
                split_chain_breaks
                and st.session_state.get("target_refold_run_af2", True)
                and (
                    not st.session_state.get("target_refold_af2_multimer", True)
                    or not st.session_state.get("target_refold_af2_initial_guess", True)
                )
            ):
                st.warning(
                    "AF2-IG will not preserve the split crop layout unless AF2 multimer "
                    "and whole-input initial guess are enabled."
                )
            non_assisted_engines = [
                label
                for label, key in [
                    ("AF3", "target_refold_run_af3"),
                    ("ColabFold", "target_refold_run_colab"),
                    ("ESMFold2", "target_refold_run_esmfold2"),
                    ("RF3", "target_refold_run_rf3"),
                    ("OpenFold-3", "target_refold_run_openfold3"),
                    ("Protenix v0.5", "target_refold_run_protenix"),
                    ("Protenix v1", "target_refold_run_protenix_v1"),
                    ("Protenix v2", "target_refold_run_protenix_v2"),
                ]
                if st.session_state.get(key, True)
            ]
            if split_chain_breaks and selected_break_count and non_assisted_engines:
                st.warning(
                    "This selection has residue-number breaks. "
                    f"Non-template-assisted engines selected: {', '.join(non_assisted_engines)}."
                )
            unsafe_single_chain_template_engines = [
                label
                for label, key in [
                    ("AF2 template", "target_refold_run_af2"),
                    ("Boltz-2", "target_refold_run_boltz2"),
                ]
                if st.session_state.get(key, True)
            ]
            block_single_chain_template_run = bool(
                selected_break_count and not split_chain_breaks and unsafe_single_chain_template_engines
            )
            if selected_break_count and not split_chain_breaks:
                message = (
                    "Legacy single-chain gaps keeps discontinuous coordinates on one chain. "
                    "For fragmented targets this is diagnostic only; template engines can collapse to the first fragment or fail."
                )
                if block_single_chain_template_run:
                    st.error(
                        message
                        + " Switch to fragment-chain mode before running "
                        + ", ".join(unsafe_single_chain_template_engines)
                        + "."
                    )
                else:
                    st.warning(message)
            if selected_split_fragment_count:
                st.info("Split-fragment targets will be staged as one multi-chain target structure, regardless of the legacy residue-gap mode.")
            estimate_engine_keys: list[str] = []
            if st.session_state.get("target_refold_run_af3", True):
                estimate_engine_keys.append("alphafast_af3")
            if st.session_state.get("target_refold_run_colab", True):
                estimate_engine_keys.append("colabfold")
            if st.session_state.get("target_refold_run_af2", True):
                estimate_engine_keys.append("af2_initial_guess")
            if st.session_state.get("target_refold_run_boltz2", True):
                estimate_engine_keys.append("boltz2_initial_guess")
            if st.session_state.get("target_refold_run_esmfold2", True):
                estimate_engine_keys.append("esmfold2")
            if st.session_state.get("target_refold_run_rf3", True):
                estimate_engine_keys.append("rf3")
            if st.session_state.get("target_refold_run_openfold3", True):
                estimate_engine_keys.append("openfold3")
            if st.session_state.get("target_refold_run_protenix", True):
                estimate_engine_keys.append("protenix")
            if st.session_state.get("target_refold_run_protenix_v1", True):
                estimate_engine_keys.append("protenix_v1")
            if st.session_state.get("target_refold_run_protenix_v2", True):
                estimate_engine_keys.append("protenix_v2")
            if st.session_state.get("target_refold_run_boltzgen", False):
                estimate_engine_keys.append("boltzgen_fold")
            msa_needed = bool(
                (
                    st.session_state.get("target_refold_run_af3", True)
                    and st.session_state.get("target_refold_af3_msa", True)
                )
                or (
                    st.session_state.get("target_refold_run_colab", True)
                    and st.session_state.get("target_refold_colab_msa", True)
                )
                or (st.session_state.get("target_refold_run_boltz2", True) and st.session_state.get("target_refold_boltz_msa", True))
                or (st.session_state.get("target_refold_run_esmfold2", True) and st.session_state.get("target_refold_esm_msa", True))
                or (st.session_state.get("target_refold_run_rf3", True) and st.session_state.get("target_refold_rf3_msa", True))
                or (st.session_state.get("target_refold_run_openfold3", True) and st.session_state.get("target_refold_openfold3_msa", True))
                or (st.session_state.get("target_refold_run_protenix", True) and st.session_state.get("target_refold_protenix_msa", True))
                or (st.session_state.get("target_refold_run_protenix_v1", True) and st.session_state.get("target_refold_protenix_v1_msa", True))
                or (st.session_state.get("target_refold_run_protenix_v2", True) and st.session_state.get("target_refold_protenix_v2_msa", True))
            )
            msa_repository_dir = Path(str(st.session_state.get("target_refold_msa_repository") or MSA_REPOSITORY_DIR))
            missing_msa_chain_count = _selected_missing_cached_msa_count(selected_entries, msa_repository_dir) if msa_needed else 0
            if msa_needed and missing_msa_chain_count:
                estimate_engine_keys.insert(0, "alphafast_msa")
            if st.session_state.get("target_refold_rosetta", False):
                estimate_engine_keys.append("postprocessing")
            estimate_count = len(selected_entries)
            estimate_total_residues = sum(_target_entry_sequence_length(entry) for entry in selected_entries) or None
            estimate_chain_count = sum(_target_entry_chain_count(entry) for entry in selected_entries)
            estimate_engine_params: dict[str, dict[str, int]] = {}
            if "esmfold2" in estimate_engine_keys:
                estimate_engine_params["esmfold2"] = {
                    "num_loops": int(st.session_state.get("target_refold_esm_loops", 10)),
                    "num_sampling_steps": int(st.session_state.get("target_refold_esm_steps", 68)),
                }
            if estimate_engine_keys and estimate_count:
                estimate = estimate_engines(
                    engines=estimate_engine_keys,
                    candidate_count=estimate_count,
                    total_residues=int(estimate_total_residues) if estimate_total_residues else None,
                    engine_params=estimate_engine_params,
                )
                if "alphafast_msa" in estimate_engine_keys and estimate_count:
                    msa_estimate = estimate_engines(
                        engines=["alphafast_msa"],
                        candidate_count=int(missing_msa_chain_count),
                        observations=[],
                    )
                    msa_seconds = float((msa_estimate.get("rows") or [{}])[0].get("estimated_seconds") or 0.0)
                    for row in estimate.get("rows") or []:
                        if row.get("engine") == "alphafast_msa":
                            previous_seconds = float(row.get("estimated_seconds") or 0.0)
                            row["estimated_seconds"] = msa_seconds
                            row["estimated_time"] = msa_estimate.get("total_time")
                            row["seconds_per_candidate"] = msa_seconds / float(missing_msa_chain_count) if missing_msa_chain_count else None
                            row["seconds_per_residue"] = None
                            row["basis"] = f"missing cached MSA chains ({missing_msa_chain_count})"
                            estimate["total_seconds"] = max(0.0, float(estimate.get("total_seconds") or 0.0) - previous_seconds + msa_seconds)
                            break
                    estimate["total_time"] = format_duration(float(estimate.get("total_seconds") or 0.0))
                with st.expander("Runtime estimate", expanded=True):
                    runtime_cols = st.columns(5)
                    runtime_cols[0].metric("Estimated total", estimate["total_time"])
                    runtime_cols[1].metric("Target structures", f"{estimate_count:,}")
                    runtime_cols[2].metric("Target chains", f"{estimate_chain_count:,}" if estimate_chain_count else "n/a")
                    runtime_cols[3].metric("Residues", f"{int(estimate_total_residues):,}" if estimate_total_residues else "n/a")
                    runtime_cols[4].metric("Engines/steps", len(estimate_engine_keys))
                    estimate_df = _estimate_rows_dataframe(estimate)
                    if not estimate_df.empty:
                        st.dataframe(
                            estimate_df,
                            width="stretch",
                            hide_index=True,
                            column_config={
                            "seconds_per_candidate": st.column_config.NumberColumn("sec / target", format="%.1f"),
                            "seconds_per_residue": st.column_config.NumberColumn("sec / residue", format="%.3f"),
                        },
                    )
                    st.caption(
                        "Estimates use previous completed jobs when available and fallback rates otherwise. "
                        "The MSA row is included when selected engines need shared target MSAs."
                    )
            elif estimate_engine_keys:
                st.caption("Runtime estimate needs at least one selected target chain.")
            run_disabled = not selected_entries or not selected_engines or block_single_chain_template_run
            if st.button("Run target refolding", type="primary", disabled=run_disabled, key="target_hub_run_refolding"):
                try:
                    source_run_dir, run_dir = enqueue_target_refolding_evaluation(
                        selected_entries,
                        evaluation_name=str(eval_name or "Target refolding evaluation"),
                        models=["af3", "boltz", "colabfold"],
                        split_chain_breaks=bool(split_chain_breaks),
                        chain_break_mode="split_fragments" if split_chain_breaks else "preserve_original_chain",
                        run_common_interface_metrics=False,
                        run_pyrosetta_input_metrics=False,
                        run_predicted_rosetta_metrics=bool(st.session_state.get("target_refold_rosetta", False)),
                        run_pymol_metrics=False,
                        pyrosetta_nprocs=16,
                        run_alphafast_af3=bool(st.session_state.get("target_refold_run_af3", True)),
                        run_colabfold=bool(st.session_state.get("target_refold_run_colab", True)),
                        run_af2_initial_guess=bool(st.session_state.get("target_refold_run_af2", True)),
                        run_boltz2_initial_guess=bool(st.session_state.get("target_refold_run_boltz2", True)),
                        run_esmfold2=bool(st.session_state.get("target_refold_run_esmfold2", True)),
                        run_rf3=bool(st.session_state.get("target_refold_run_rf3", True)),
                        run_openfold3=bool(st.session_state.get("target_refold_run_openfold3", True)),
                        run_protenix=bool(st.session_state.get("target_refold_run_protenix", True)),
                        run_protenix_v1=bool(st.session_state.get("target_refold_run_protenix_v1", True)),
                        run_protenix_v2=bool(st.session_state.get("target_refold_run_protenix_v2", True)),
                        run_boltzgen_fold=bool(st.session_state.get("target_refold_run_boltzgen", False)),
                        colabfold_msa_source=(
                            "msa_repository_then_alphafast_mmseqs_gpu"
                            if bool(st.session_state.get("target_refold_colab_msa", True))
                            else "repo_run_csv"
                        ),
                        msa_repository_dir=Path(str(st.session_state.get("target_refold_msa_repository") or MSA_REPOSITORY_DIR)),
                        require_real_target_msa=bool(st.session_state.get("target_refold_require_real_msa", True)),
                        alphafast_db_dir=Path(str(st.session_state.get("target_refold_alphafast_db") or ALPHAFAST_DB_DIR)),
                        alphafast_weights_dir=Path(str(st.session_state.get("target_refold_af3_weights") or ALPHAFAST_WEIGHTS_DIR)),
                        colabfold_cache_dir=Path(str(st.session_state.get("target_refold_colab_cache") or COLABFOLD_CACHE_DIR)),
                        alphafast_num_recycles=int(st.session_state.get("target_refold_af3_recycles", 10)),
                        alphafast_use_target_templates=bool(st.session_state.get("target_refold_af3_templates", True)),
                        alphafast_query_only_msa=not bool(st.session_state.get("target_refold_af3_msa", True)),
                        alphafast_gpu_device=str(target_gpu),
                        af2_num_recycles=int(st.session_state.get("target_refold_af2_recycles", 3)),
                        af2_multimer=bool(st.session_state.get("target_refold_af2_multimer", True)),
                        af2_use_initial_guess=bool(st.session_state.get("target_refold_run_af2", True)),
                        af2_use_binder_template=bool(st.session_state.get("target_refold_af2_fragment_template", True)),
                        af2_use_interface_template=bool(st.session_state.get("target_refold_af2_layout_template", True)),
                        colabfold_num_recycles=int(st.session_state.get("target_refold_colab_recycles", 3)),
                        colabfold_num_models=int(st.session_state.get("target_refold_colab_models", 3)),
                        colabfold_gpu_device=str(target_gpu),
                        gpu_device=str(target_gpu),
                        colabfold_use_target_templates=bool(st.session_state.get("target_refold_colab_templates", True)),
                        colabfold_use_target_msa=bool(st.session_state.get("target_refold_colab_msa", True)),
                        colabfold_max_template_hits=4,
                        boltz2_use_target_template=bool(st.session_state.get("target_refold_boltz_template", True)),
                        boltz2_use_target_msa=bool(st.session_state.get("target_refold_boltz_msa", True)),
                        boltz2_recycling_steps=int(st.session_state.get("target_refold_boltz_recycles", 10)),
                        boltz2_sampling_steps=int(st.session_state.get("target_refold_boltz_steps", 100)),
                        boltz2_diffusion_samples=int(st.session_state.get("target_refold_boltz_samples", 3)),
                        boltz2_write_full_pae=bool(st.session_state.get("target_refold_boltz_full_pae", True)),
                        esmfold2_modes=list(st.session_state.get("target_refold_esm_modes") or target_esm_modes or ["initial_guess"]),
                        esmfold2_use_target_msa=bool(st.session_state.get("target_refold_esm_msa", True)),
                        num_sampling_steps=int(st.session_state.get("target_refold_esm_steps", 68)),
                        num_loops=int(st.session_state.get("target_refold_esm_loops", 10)),
                        seed=int(st.session_state.get("target_refold_esm_seed", 0)),
                        rf3_checkpoint_path=Path(str(st.session_state.get("target_refold_rf3_checkpoint") or RF3_CHECKPOINT)),
                        rf3_use_target_msa=bool(st.session_state.get("target_refold_rf3_msa", True)),
                        rf3_use_target_template=bool(st.session_state.get("target_refold_rf3_template", True)),
                        rf3_recycles=int(st.session_state.get("target_refold_rf3_recycles", 10)),
                        rf3_num_steps=int(st.session_state.get("target_refold_rf3_steps", 50)),
                        rf3_diffusion_batch_size=int(st.session_state.get("target_refold_rf3_samples", 5)),
                        rf3_seed=int(st.session_state.get("target_refold_rf3_seed", 0)),
                        openfold3_checkpoint_path=Path(str(st.session_state.get("target_refold_openfold3_checkpoint") or OPENFOLD3_CHECKPOINT)),
                        openfold3_use_target_msa=bool(st.session_state.get("target_refold_openfold3_msa", True)),
                        openfold3_num_diffusion_samples=int(st.session_state.get("target_refold_openfold3_samples", 5)),
                        openfold3_num_model_seeds=int(st.session_state.get("target_refold_openfold3_seeds", 1)),
                        openfold3_num_recycles=int(st.session_state.get("target_refold_openfold3_recycles", 3)),
                        openfold3_use_msa_server=bool(st.session_state.get("target_refold_openfold3_msa_server", False)),
                        protenix_use_msa=bool(st.session_state.get("target_refold_protenix_msa", True)),
                        protenix_cycle=int(st.session_state.get("target_refold_protenix_cycle", 3)),
                        protenix_diffusion_steps=int(st.session_state.get("target_refold_protenix_steps", 50)),
                        protenix_samples=int(st.session_state.get("target_refold_protenix_samples", 5)),
                        protenix_v1_model_name=str(st.session_state.get("target_refold_protenix_v1_model") or PROTENIX_V1_MODEL),
                        protenix_v1_use_msa=bool(st.session_state.get("target_refold_protenix_v1_msa", True)),
                        protenix_v1_use_template=bool(st.session_state.get("target_refold_protenix_v1_template", True)),
                        protenix_v1_use_default_params=True,
                        protenix_v1_cycle=int(st.session_state.get("target_refold_protenix_v1_cycle", 10)),
                        protenix_v1_diffusion_steps=int(st.session_state.get("target_refold_protenix_v1_steps", 200)),
                        protenix_v1_samples=int(st.session_state.get("target_refold_protenix_v1_samples", 5)),
                        protenix_v2_model_name=str(st.session_state.get("target_refold_protenix_v2_model") or PROTENIX_V2_MODEL),
                        protenix_v2_use_msa=bool(st.session_state.get("target_refold_protenix_v2_msa", True)),
                        protenix_v2_use_template=bool(st.session_state.get("target_refold_protenix_v2_template", True)),
                        protenix_v2_use_default_params=True,
                        protenix_v2_cycle=int(st.session_state.get("target_refold_protenix_v2_cycle", 10)),
                        protenix_v2_diffusion_steps=int(st.session_state.get("target_refold_protenix_v2_steps", 200)),
                        protenix_v2_samples=int(st.session_state.get("target_refold_protenix_v2_samples", 5)),
                        boltzgen_recycling_steps=int(st.session_state.get("target_refold_boltzgen_recycles", 3)),
                        boltzgen_sampling_steps=int(st.session_state.get("target_refold_boltzgen_steps", 100)),
                        boltzgen_diffusion_samples=int(st.session_state.get("target_refold_boltzgen_samples", 3)),
                    )
                    spawn_worker_for_run(run_dir)
                    st.success("Target refolding queued. The worker will keep running independently of Streamlit.")
                    show_pipeline_links(run_dir, [source_run_dir, run_dir])
                except Exception as exc:
                    st.error(str(exc))

        with refold_results_tab:
            refresh_results_button("target_preparation_refolding_refresh_results")
            runs = _target_refolding_runs()
            if not runs:
                st.info("No target-refolding runs are available yet.")
            else:
                runs_df = pd.DataFrame(runs)
                hidden_columns = {"result", "run_dir", "source_run_dir"}
                preferred_columns = [
                    "job_code",
                    "status",
                    "target",
                    "target_chain_ids",
                    "evidence_mode",
                    "evidence_detail",
                    "warning",
                    "target_structures",
                    "target_chains",
                    "records",
                    "current_phase",
                    "current_engine",
                    "created_at",
                    "run_id",
                    "task_group",
                ]
                visible_columns = [
                    column for column in preferred_columns if column in runs_df.columns and column not in hidden_columns
                ]
                visible_columns.extend(
                    column for column in runs_df.columns if column not in hidden_columns and column not in visible_columns
                )
                display_df = runs_df.copy()
                if {"job_code", "result"}.issubset(display_df.columns):
                    display_df["job_code"] = display_df.apply(
                        lambda row: f"{row['result']}&job_code={row['job_code']}",
                        axis=1,
                    )
                table_key = "target_refolding_results"
                event = st.dataframe(
                    display_df[visible_columns],
                    hide_index=True,
                    width="stretch",
                    key=f"{table_key}_jobs_table",
                    on_select="rerun",
                    selection_mode="multi-row",
                    column_config={
                        "job_code": st.column_config.LinkColumn("job_code", display_text=r"job_code=([^&]+)"),
                    },
                )
                delete_result = st.session_state.pop(f"{table_key}_delete_result", None)
                if delete_result:
                    level, message = delete_result
                    if level == "success":
                        st.success(str(message))
                    else:
                        st.error(str(message))

                selected_indices = [
                    index
                    for index in selected_dataframe_rows(event, f"{table_key}_jobs_table")
                    if 0 <= index < len(runs_df)
                ]
                selected_rows = runs_df.iloc[selected_indices].copy() if selected_indices else runs_df.iloc[0:0].copy()
                active_selected = selected_rows[selected_rows["status"].isin(ACTIVE_STATUSES)] if "status" in selected_rows.columns else selected_rows.iloc[0:0]
                if not active_selected.empty:
                    st.warning("Running, queued, or preparing target-refolding jobs cannot be deleted.")
                if "status" in selected_rows.columns:
                    selected_rows = selected_rows[~selected_rows["status"].isin(ACTIVE_STATUSES)]
                selected_refs = [
                    (str(row.get("task_group") or "target-refolding"), str(row.get("run_id") or ""))
                    for row in selected_rows.to_dict(orient="records")
                    if str(row.get("run_id") or "")
                ]
                selected_refs_key = f"{table_key}_selected_delete_refs"
                if selected_refs:
                    st.session_state[selected_refs_key] = selected_refs
                cached_selected_refs = st.session_state.get(selected_refs_key) or []
                delete_clicked = st.button(
                    "Delete selected target-refolding jobs",
                    type="primary",
                    disabled=not cached_selected_refs,
                    key=f"{table_key}_request_delete_jobs",
                )
                if delete_clicked and cached_selected_refs:
                    show_delete_jobs_dialog(
                        table_key=table_key,
                        pending_refs=cached_selected_refs,
                        selected_refs_key=selected_refs_key,
                        label="target-refolding job",
                    )
                else:
                    st.caption("Select finished target-refolding rows in the table to enable deletion.")
                _render_target_refolding_result_browser(runs, key_prefix="target_refold_nested_results")
