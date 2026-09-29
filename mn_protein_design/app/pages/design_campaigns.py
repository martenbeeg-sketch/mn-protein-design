from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from mn_protein_design.app.components.molstar_viewer import (
    ChainVisualization,
    StructureVisualization,
    molstar_custom_component,
)
from mn_protein_design.app.pages.common import gpu_run_panel, refresh_results_button, result_link
from mn_protein_design.core.jobs import collect_jobs, read_json
from mn_protein_design.core.detection_annotations import detection_score_sources, sidechain_sasa_by_residue
from mn_protein_design.workflows.detection import detection_jobs_for_target
from mn_protein_design.core.structures import filter_pdb_text, pdb_summary
from mn_protein_design.runtime import runs_root
from mn_protein_design.workflows.design import (
    default_target_contig,
    prepared_design_targets,
    rfdiffusion_scaffold_library_status,
    target_label,
)
from mn_protein_design.workflows.refolding import (
    PROTENIX_V1_20250630_MODEL,
    PROTENIX_V1_MODEL,
    PROTENIX_V2_MODEL,
    _sequences_by_chain,
)
from mn_protein_design.workflows.target_prep import target_chain_break_summary
from mn_protein_design.workflows.design_campaigns import (
    CANDIDATE_POOL_COVERAGE,
    DESIGN_CAMPAIGN_GROUP,
    ENGINE_LABELS,
    ENGINE_ORDER,
    VANILLA_ONLY_ENGINES,
    create_design_campaign,
    create_design_campaign_collection,
)
from mn_protein_design.workflows import bindcraft2 as bindcraft2_workflow

ResidueId = tuple[str, int]


DETECTION_TOOL_STYLES = {
    "pesto": {"label": "PeSTo", "color": "0xd946ef"},
    "masif_seed": {"label": "MaSIF", "color": "0xf97316"},
    "scannet": {"label": "ScanNet", "color": "0x22c55e"},
    "surf2spot": {"label": "Surf2Spot", "color": "0x2563eb"},
}


REFOLDING_ENGINE_OPTIONS = [
    "AlphaFast AF3",
    "ColabFold",
    "AF2-IG",
    "ESMFold2",
    "Boltz-2",
    "RF3",
    "OpenFold-3",
    "Protenix",
    "Protenix v1",
    "Protenix v2",
    "BoltzGen Fold",
]

STAGED_NATIVE_SEQUENCE_ENGINES = {"esmfold2_binder_design", "proteina_complexa"}
GENERATOR_ONLY_RECIPES = {"staged", "engine_scout"}

MPNN_MODEL_LABELS = {
    "protein_mpnn": "ProteinMPNN model (LigandMPNN implementation)",
    "soluble_mpnn": "Soluble ProteinMPNN model (LigandMPNN implementation)",
    "ligand_mpnn": "LigandMPNN model (LigandMPNN implementation)",
}

TARGET_CATEGORY_ORDER = ["imported", "trimmed", "cleaned", "mutated", "cropped", "benchmark", "unknown"]

BINDER_LENGTH_SUPPORT = [
    {"engine": "RFdiffusion classic", "range handling": "Uses fixed or min-max range in the generated contig"},
    {"engine": "BindCraft", "range handling": "Uses fixed or min-max range in BindCraft input settings"},
    {"engine": "RFdiffusion3 / Foundry", "range handling": "Uses fixed or min-max range in the generated contig"},
    {"engine": "BoltzGen", "range handling": "Uses fixed length or min..max sequence spec"},
    {"engine": "Genie3", "range handling": "Uses fixed or min-max length in the generated problem spec"},
    {"engine": "Protpardelle-1c", "range handling": "Uses fixed or min-max binder range"},
    {"engine": "Proteina-Complexa", "range handling": "Uses fixed or min-max binder_length list"},
    {"engine": "PXDesign", "range handling": "Design campaigns sample one fixed length per attempt from the range"},
    {"engine": "ESMFold2 binder design", "range handling": "Design campaigns sample one fixed length per attempt from the range"},
]

BINDCRAFT_VANILLA_PRESETS = {
    "Default 4-stage + MPNN": "default_4stage_multimer_mpnn.json",
    "Default flexible + MPNN": "default_4stage_multimer_mpnn_flexible.json",
    "Default hard target + MPNN": "default_4stage_multimer_mpnn_hardtarget.json",
    "Default flexible hard target + MPNN": "default_4stage_multimer_mpnn_flexible_hardtarget.json",
    "Beta-sheet 4-stage + MPNN": "betasheet_4stage_multimer_mpnn.json",
    "Beta-sheet hard target + MPNN": "betasheet_4stage_multimer_mpnn_hardtarget.json",
    "Beta-sheet flexible hard target + MPNN": "betasheet_4stage_multimer_mpnn_flexible_hardtarget.json",
    "Peptide 3-stage + MPNN": "peptide_3stage_multimer_mpnn.json",
    "Peptide flexible + MPNN": "peptide_3stage_multimer_mpnn_flexible.json",
}

BINDCRAFT_GENERATOR_PRESETS = {
    "Default 4-stage generator": "default_4stage_multimer.json",
    "Default flexible generator": "default_4stage_multimer_flexible.json",
    "Default hard target generator": "default_4stage_multimer_hardtarget.json",
    "Default flexible hard target generator": "default_4stage_multimer_flexible_hardtarget.json",
    "Beta-sheet 4-stage generator": "betasheet_4stage_multimer.json",
    "Beta-sheet hard target generator": "betasheet_4stage_multimer_hardtarget.json",
    "Beta-sheet flexible hard target generator": "betasheet_4stage_multimer_flexible_hardtarget.json",
    "Peptide 3-stage generator": "peptide_3stage_multimer.json",
    "Peptide flexible generator": "peptide_3stage_multimer_flexible.json",
}

BINDCRAFT_FILTER_PRESETS = {
    "Default filters": "default_filters.json",
    "Relaxed filters": "relaxed_filters.json",
    "Peptide filters": "peptide_filters.json",
    "Peptide relaxed filters": "peptide_relaxed_filters.json",
    "No native filters": "no_filters.json",
}

RANK_OPTION_LABELS = {
    "ranking_score": "native ranking/confidence (higher)",
    "iptm": "ipTM (higher)",
    "ipae": "iPAE (lower)",
    "binder_plddt": "binder pLDDT (higher)",
    "ptm": "pTM (higher)",
    "target_aligned_binder_rmsd": "target-aligned binder RMSD (lower)",
}

RANK_OPTIONS_BY_REFOLDER = {
    "AlphaFast AF3": ["ranking_score", "iptm", "ptm", "target_aligned_binder_rmsd"],
    "ColabFold": ["iptm", "ipae", "binder_plddt", "ptm", "target_aligned_binder_rmsd"],
    "AF2-IG": ["iptm", "ipae", "binder_plddt", "ptm", "target_aligned_binder_rmsd"],
    "ESMFold2": ["ipae", "binder_plddt", "iptm", "target_aligned_binder_rmsd"],
    "Boltz-2": ["ranking_score", "iptm", "ptm", "binder_plddt", "ipae", "target_aligned_binder_rmsd"],
    "RF3": ["ranking_score", "binder_plddt", "target_aligned_binder_rmsd"],
    "OpenFold-3": ["ranking_score", "binder_plddt", "target_aligned_binder_rmsd"],
    "Protenix": ["ranking_score", "binder_plddt", "target_aligned_binder_rmsd"],
    "Protenix v1": ["ranking_score", "binder_plddt", "target_aligned_binder_rmsd"],
    "Protenix v2": ["ranking_score", "binder_plddt", "target_aligned_binder_rmsd"],
    "BoltzGen Fold": ["ranking_score", "binder_plddt", "target_aligned_binder_rmsd"],
}


def _rank_options_for_refolder(refolder: str) -> dict[str, str]:
    keys = RANK_OPTIONS_BY_REFOLDER.get(str(refolder), ["target_aligned_binder_rmsd"])
    return {key: RANK_OPTION_LABELS[key] for key in keys}


def _select_rank_metric(column, label: str, *, refolder: str, key: str, disabled: bool) -> str:
    options = _rank_options_for_refolder(refolder)
    if st.session_state.get(key) not in options:
        st.session_state[key] = next(iter(options))
    return column.selectbox(
        label,
        list(options),
        format_func=options.get,
        disabled=disabled,
        key=key,
    )


def _rfdiffusion_vanilla_pipeline_table(
    *,
    rf_sequence_method: str,
    rf_max_binder_rmsd: float,
    rf_rank_metric: str,
    rf_keep_per_attempt: int,
) -> None:
    st.caption(
        "Vanilla RFdiffusion runs with Docker: backbone generation -> selected MPNN model -> "
        "Boltz-2 monomer refold -> AF2-IG target-template complex refold -> RMSD-filtered ranking."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {"step": "Backbone generation", "tool": "RFdiffusion classic"},
                {"step": "Sequence design", "tool": MPNN_MODEL_LABELS.get(str(rf_sequence_method), str(rf_sequence_method))},
                {"step": "Monomer refolding", "tool": "Boltz-2 monomer"},
                {"step": "Complex refolding", "tool": "AF2-IG target-template, multimer, 3 recycles"},
                {
                    "step": "Analysis",
                    "tool": (
                        f"Filter binder RMSD <= {float(rf_max_binder_rmsd):g} A; "
                        f"rank by {rf_rank_metric}; keep {int(rf_keep_per_attempt)} per backbone attempt"
                    ),
                },
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def _screen_refolder_settings(prefix: str, refolder: str, *, disabled: bool, caption: str | None = None) -> dict[str, object]:
    settings: dict[str, object] = {}
    if caption is None:
        caption = (
            "These settings are used only for this internal sequence-selection screen. "
            "ipSAE/Rosetta/PyMOL stay in the Evaluation tab."
        )
    if caption:
        st.caption(caption)
    if refolder == "AlphaFast AF3":
        cols = st.columns(5)
        settings["num_recycles"] = cols[0].number_input("AF3 recycles", 1, 48, 10, key=f"{prefix}_af3_recycles", disabled=disabled)
        settings["num_sampling_steps"] = cols[1].number_input("Sampling steps", 1, 1000, 68, key=f"{prefix}_af3_steps", disabled=disabled)
        settings["num_samples"] = cols[2].number_input("Samples", 1, 20, 1, key=f"{prefix}_af3_samples", disabled=disabled)
        settings["alphafast_use_target_templates"] = cols[3].checkbox(
            "Use templates",
            value=True,
            key=f"{prefix}_af3_templates",
            disabled=disabled,
            help="Embeds the staged input target chains as AF3 templates. Binder chains and binder-interface geometry are not templated.",
        )
        settings["alphafast_use_target_msa"] = cols[4].checkbox(
            "Use target MSAs",
            value=True,
            key=f"{prefix}_af3_msa",
            disabled=disabled,
            help="When off, AlphaFast AF3 runs with query-only/no target MSA input.",
        )
        settings["use_target_msa"] = bool(settings["alphafast_use_target_msa"])
    elif refolder == "ColabFold":
        cols = st.columns(3)
        settings["num_recycles"] = cols[0].number_input("ColabFold recycles", 1, 48, 3, key=f"{prefix}_colab_recycles", disabled=disabled)
        settings["num_samples"] = cols[1].number_input("Models", 1, 20, 3, key=f"{prefix}_colab_models", disabled=disabled)
        settings["colabfold_use_target_templates"] = cols[2].checkbox("Use templates", value=True, key=f"{prefix}_colab_templates", disabled=disabled, help="Uses the staged input target as the template.")
        settings["colabfold_max_template_hits"] = 4
        settings["use_target_msa"] = True
    elif refolder == "AF2-IG":
        cols = st.columns(5)
        settings["num_recycles"] = cols[0].number_input("AF2 recycles", 1, 48, 3, key=f"{prefix}_af2_recycles", disabled=disabled)
        settings["af2_multimer"] = cols[1].checkbox("AF2 multimer", value=True, key=f"{prefix}_af2_multimer", disabled=disabled)
        settings["af2_use_initial_guess"] = cols[2].checkbox("Initial guess", value=True, key=f"{prefix}_af2_initial_guess", disabled=disabled)
        settings["af2_use_binder_template"] = cols[3].checkbox("Binder template", value=False, key=f"{prefix}_af2_binder_template", disabled=disabled)
        settings["af2_use_interface_template"] = cols[4].checkbox("Interface template", value=False, key=f"{prefix}_af2_interface_template", disabled=disabled)
        settings["use_target_msa"] = False
    elif refolder == "ESMFold2":
        cols = st.columns(4)
        settings["num_sampling_steps"] = cols[0].number_input("Sampling steps", 1, 256, 68, key=f"{prefix}_esm_steps", disabled=disabled)
        settings["num_recycles"] = cols[1].number_input("Recycling loops", 1, 64, 10, key=f"{prefix}_esm_loops", disabled=disabled)
        settings["seed"] = cols[2].number_input("Seed", 0, 999999, 0, key=f"{prefix}_esm_seed", disabled=disabled)
        settings["use_target_msa"] = cols[3].checkbox("Use target MSAs", value=True, key=f"{prefix}_esm_msa", disabled=disabled)
    elif refolder == "Boltz-2":
        cols = st.columns(6)
        settings["boltz2_use_target_template"] = cols[0].checkbox("Use templates", value=True, key=f"{prefix}_boltz_template", disabled=disabled, help="Uses the staged input target as the template.")
        settings["use_target_msa"] = cols[1].checkbox("Use target MSAs", value=True, key=f"{prefix}_boltz_msa", disabled=disabled)
        settings["num_recycles"] = cols[2].number_input("Recycling steps", 1, 48, 10, key=f"{prefix}_boltz_recycles", disabled=disabled)
        settings["num_sampling_steps"] = cols[3].number_input("Sampling steps", 1, 1000, 200, key=f"{prefix}_boltz_steps", disabled=disabled)
        settings["num_samples"] = cols[4].number_input("Diffusion samples", 1, 20, 3, key=f"{prefix}_boltz_samples", disabled=disabled)
        settings["boltz2_write_full_pae"] = cols[5].checkbox("Write full PAE", value=True, key=f"{prefix}_boltz_full_pae", disabled=disabled)
    elif refolder == "RF3":
        cols = st.columns(6)
        settings["rf3_use_target_template"] = cols[0].checkbox("Use templates", value=True, key=f"{prefix}_rf3_template", disabled=disabled, help="Uses the staged input target chains as RF3 template coordinates.")
        settings["use_target_msa"] = cols[1].checkbox("Use target MSAs", value=True, key=f"{prefix}_rf3_msa", disabled=disabled)
        settings["num_recycles"] = cols[2].number_input("RF3 recycles", 1, 48, 10, key=f"{prefix}_rf3_recycles", disabled=disabled)
        settings["num_sampling_steps"] = cols[3].number_input("Diffusion steps", 1, 1000, 50, key=f"{prefix}_rf3_steps", disabled=disabled)
        settings["num_samples"] = cols[4].number_input("Samples", 1, 20, 5, key=f"{prefix}_rf3_samples", disabled=disabled)
        settings["seed"] = cols[5].number_input("Seed", 0, 999999, 0, key=f"{prefix}_rf3_seed", disabled=disabled)
    elif refolder == "OpenFold-3":
        cols = st.columns(5)
        settings["use_target_msa"] = cols[0].checkbox("Use target MSAs", value=True, key=f"{prefix}_of3_msa", disabled=disabled)
        settings["num_samples"] = cols[1].number_input("Diffusion samples", 1, 20, 5, key=f"{prefix}_of3_samples", disabled=disabled)
        settings["openfold3_num_model_seeds"] = cols[2].number_input("Model seeds", 1, 20, 1, key=f"{prefix}_of3_model_seeds", disabled=disabled)
        settings["num_recycles"] = cols[3].number_input("Recycles", 1, 48, 3, key=f"{prefix}_of3_recycles", disabled=disabled)
        settings["openfold3_use_msa_server"] = cols[4].checkbox("Use MSA server", value=False, key=f"{prefix}_of3_msa_server", disabled=disabled)
    elif refolder == "Protenix":
        cols = st.columns(4)
        settings["use_target_msa"] = cols[0].checkbox("Use target MSAs", value=True, key=f"{prefix}_protenix_msa", disabled=disabled)
        settings["num_recycles"] = cols[1].number_input("Pairformer cycles", 1, 48, 3, key=f"{prefix}_protenix_cycles", disabled=disabled)
        settings["num_sampling_steps"] = cols[2].number_input("Diffusion steps", 1, 1000, 50, key=f"{prefix}_protenix_steps", disabled=disabled)
        settings["num_samples"] = cols[3].number_input("Samples", 1, 20, 5, key=f"{prefix}_protenix_samples", disabled=disabled)
    elif refolder == "Protenix v1":
        cols = st.columns(6)
        settings["protenix_v1_model_name"] = cols[0].selectbox(
            "Model",
            [PROTENIX_V1_MODEL, PROTENIX_V1_20250630_MODEL],
            key=f"{prefix}_protenix_v1_model",
            disabled=disabled,
        )
        settings["use_target_msa"] = cols[1].checkbox("Use target MSAs", value=True, key=f"{prefix}_protenix_v1_msa", disabled=disabled)
        settings["protenix_v1_use_template"] = cols[2].checkbox("Use templates", value=True, key=f"{prefix}_protenix_v1_template", disabled=disabled)
        settings["num_recycles"] = cols[3].number_input("Pairformer cycles", 1, 48, 10, key=f"{prefix}_protenix_v1_cycles", disabled=disabled)
        settings["num_sampling_steps"] = cols[4].number_input("Diffusion steps", 1, 1000, 200, key=f"{prefix}_protenix_v1_steps", disabled=disabled)
        settings["num_samples"] = cols[5].number_input("Samples", 1, 20, 5, key=f"{prefix}_protenix_v1_samples", disabled=disabled)
    elif refolder == "Protenix v2":
        cols = st.columns(6)
        settings["protenix_v2_model_name"] = cols[0].text_input("Model", value=PROTENIX_V2_MODEL, key=f"{prefix}_protenix_v2_model", disabled=disabled)
        settings["use_target_msa"] = cols[1].checkbox("Use target MSAs", value=True, key=f"{prefix}_protenix_v2_msa", disabled=disabled)
        settings["protenix_v2_use_template"] = cols[2].checkbox("Use templates", value=True, key=f"{prefix}_protenix_v2_template", disabled=disabled)
        settings["num_recycles"] = cols[3].number_input("Pairformer cycles", 1, 48, 10, key=f"{prefix}_protenix_v2_cycles", disabled=disabled)
        settings["num_sampling_steps"] = cols[4].number_input("Diffusion steps", 1, 1000, 200, key=f"{prefix}_protenix_v2_steps", disabled=disabled)
        settings["num_samples"] = cols[5].number_input("Samples", 1, 20, 5, key=f"{prefix}_protenix_v2_samples", disabled=disabled)
    elif refolder == "BoltzGen Fold":
        cols = st.columns(3)
        settings["num_recycles"] = cols[0].number_input("Recycling steps", 1, 48, 3, key=f"{prefix}_boltzgen_recycles", disabled=disabled)
        settings["num_sampling_steps"] = cols[1].number_input("Sampling steps", 1, 1000, 200, key=f"{prefix}_boltzgen_steps", disabled=disabled)
        settings["num_samples"] = cols[2].number_input("Diffusion samples", 1, 20, 5, key=f"{prefix}_boltzgen_samples", disabled=disabled)
        settings["use_target_msa"] = True
    return settings


def _target_chain_ids(target: dict) -> list[str]:
    chains = target.get("chains") or []
    if chains and isinstance(chains[0], dict):
        return [str(row.get("chain_id") or "") for row in chains if row.get("chain_id")]
    return [str(chain) for chain in chains]


def _design_target_category(row: dict[str, object]) -> str:
    return str(row.get("source_category") or row.get("prepared_kind") or row.get("task_group") or "unknown").strip() or "unknown"


def _design_target_category_options(categories: list[str] | set[str]) -> list[str]:
    present = {str(category) for category in categories if str(category)}
    present.add("mutated")
    ordered = [category for category in TARGET_CATEGORY_ORDER if category in present]
    ordered.extend(sorted(category for category in present if category not in TARGET_CATEGORY_ORDER))
    return ordered


def _target_detection_summary(target_pdb: Path) -> str:
    jobs = [job for job in detection_jobs_for_target(target_pdb) if job.get("status") == "completed"]
    if not jobs:
        return ""
    labels: list[str] = []
    for job in jobs:
        tool = str(job.get("tool") or job.get("job_type") or "detection").strip() or "detection"
        code = str(job.get("job_code") or "").strip()
        label = f"{tool} {code}" if code else tool
        if label not in labels:
            labels.append(label)
    return ", ".join(labels)


def _design_target_chain_rows(targets: list[dict]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    detection_cache: dict[str, str] = {}
    for index, row in enumerate(targets):
        target_pdb = Path(str(row.get("target_pdb") or "")).expanduser()
        if not target_pdb.exists():
            continue
        detection_key = str(target_pdb.resolve())
        if detection_key not in detection_cache:
            detection_cache[detection_key] = _target_detection_summary(target_pdb)
        try:
            summary = pdb_summary(target_pdb.read_text(errors="ignore"))
            sequence_by_chain = _sequences_by_chain(target_pdb)
        except Exception:
            summary = {"chains": []}
            sequence_by_chain = {}
        declared = set(_target_chain_ids(row))
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
            fragment_count = 1 if sequence else 0
            break_count = 0
            try:
                break_summary = target_chain_break_summary(target_pdb, chain_id)
                fragment_count = int(break_summary.get("fragment_count") or fragment_count)
                break_count = int(break_summary.get("break_count") or 0)
            except Exception:
                pass
            total_aa_length += len(sequence) or int(chain.get("residue_count") or 0)
            total_fragments += fragment_count
            total_breaks += break_count
        if not chain_ids:
            continue
        rows.append(
            {
                "target": row.get("target_name") or target_pdb.stem,
                "chains": ",".join(chain_ids),
                "chain_count": len(chain_ids),
                "aa_length": total_aa_length,
                "fragments": total_fragments,
                "breaks": total_breaks,
                "source": row.get("source_label") or row.get("source_category") or "",
                "category": _design_target_category(row),
                "job": row.get("job_code") or "",
                "records": row.get("records") or 1,
                "ppi_hotspot": detection_cache.get(detection_key, ""),
                "path": str(target_pdb),
                "_target_index": index,
            }
        )
    return pd.DataFrame(rows)


def _design_selected_target_entry(selected_row: pd.Series, targets: list[dict]) -> dict[str, object]:
    source = targets[int(selected_row["_target_index"])]
    chains = [chain.strip() for chain in str(selected_row.get("chains") or "").split(",") if chain.strip()]
    return {
        "target_name": selected_row.get("target"),
        "target_pdb": selected_row.get("path"),
        "chains": chains,
        "source_category": selected_row.get("category"),
        "source_label": selected_row.get("source"),
        "source_job_code": selected_row.get("job"),
        "records": selected_row.get("records"),
        "source_run_id": source.get("run_id"),
        "source_task_group": source.get("task_group"),
        "breaks": selected_row.get("breaks"),
        "_target": source,
    }


def _pdb_residue_index_map(pdb_text: str) -> dict[str, dict[int, int]]:
    mapping: dict[str, dict[int, int]] = {}
    residues_by_chain: dict[str, list[int]] = {}
    last_seen: dict[str, tuple[int, str] | None] = {}
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        chain = line[21].strip() or "_"
        try:
            residue = int(line[22:26])
        except ValueError:
            continue
        residue_key = (residue, line[26].strip())
        residues_by_chain.setdefault(chain, [])
        last_seen.setdefault(chain, None)
        if last_seen[chain] != residue_key:
            residues_by_chain[chain].append(residue)
            last_seen[chain] = residue_key
    for chain, residues in residues_by_chain.items():
        mapping[chain] = {index: residue for index, residue in enumerate(residues, start=1)}
    return mapping


def _pdb_residue_number_set(pdb_text: str) -> dict[str, set[int]]:
    residues: dict[str, set[int]] = {}
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        try:
            residue = int(line[22:26])
        except ValueError:
            continue
        residues.setdefault(line[21].strip() or "_", set()).add(residue)
    return residues


def _viewer_residues(value: object, pdb_text: str) -> set[ResidueId]:
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
    residues: set[ResidueId] = set()
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


def _hotspot_residues(text: str) -> set[ResidueId]:
    residues: set[ResidueId] = set()
    for token in text.replace(";", ",").replace(" ", ",").split(","):
        token = token.strip().replace(":", "")
        if len(token) < 2:
            continue
        try:
            residues.add((token[0].upper(), int(token[1:])))
        except ValueError:
            continue
    return residues


def _hotspot_text(residues: set[ResidueId]) -> str:
    return ",".join(f"{chain}{residue}" for chain, residue in sorted(residues))


def _hotspot_chains(residues: set[ResidueId]) -> list[ChainVisualization] | None:
    by_chain: dict[str, list[int]] = {}
    for chain, residue in sorted(residues):
        by_chain.setdefault(chain, []).append(residue)
    return [
        ChainVisualization(
            chain_id=chain,
            residues=values,
            color="uniform",
            color_params={"value": "0x2563eb"},
            representation_type="cartoon+ball-and-stick",
            label="Selected hotspots",
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


def _campaign_rows() -> list[dict]:
    rows: list[dict] = []
    for job in collect_jobs(DESIGN_CAMPAIGN_GROUP):
        run_dir = Path(job["run_dir"])
        input_payload = read_json(run_dir / "input.json")
        inputs = input_payload.get("inputs") if isinstance(input_payload.get("inputs"), dict) else {}
        params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
        result = read_json(run_dir / "result.json")
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        target_key = _campaign_target_key(input_payload)
        rows.append(
            {
                "result": result_link(DESIGN_CAMPAIGN_GROUP, job["run_id"]),
                "job": job["job_code"],
                "campaign": job.get("campaign_name") or params.get("campaign_name") or metrics.get("collection_name") or "",
                "status": job["status"],
                "job_type": input_payload.get("job_type") or "",
                "phase": job.get("current_phase") or "",
                "engine": job.get("current_engine") or "",
                "candidates": metrics.get("candidate_count", ""),
                "completed engines": metrics.get("completed_engine_count", ""),
                "failed engines": metrics.get("failed_engine_count", ""),
                "updated": job.get("updated_at") or "",
                "run_dir": str(run_dir),
                "run_id": job["run_id"],
                "target_key": target_key,
                "target_pdb": str(inputs.get("target_pdb") or ""),
                "target_chains": ", ".join(str(chain) for chain in inputs.get("target_chains") or []),
            }
        )
    return rows


def _campaign_target_key(input_payload: dict) -> str:
    inputs = input_payload.get("inputs") if isinstance(input_payload.get("inputs"), dict) else {}
    target_pdb = str(inputs.get("target_pdb") or "").strip()
    target_chains = sorted(str(chain) for chain in inputs.get("target_chains") or [])
    if not target_pdb:
        return ""
    return json.dumps({"target_pdb": target_pdb, "target_chains": target_chains}, sort_keys=True)


def _target_key_label(target_key: str) -> str:
    try:
        payload = json.loads(target_key)
    except Exception:
        return target_key or "unknown target"
    chains = ", ".join(payload.get("target_chains") or []) or "n/a"
    path = Path(str(payload.get("target_pdb") or ""))
    return f"{path.name or 'target'} | chains: {chains}"


def _campaign_history_for_target(target_pdb: Path, target_chains: list[str]) -> list[dict[str, object]]:
    selected_path = str(target_pdb)
    selected_resolved = str(target_pdb.expanduser().resolve()) if target_pdb.exists() else selected_path
    selected_chain_set = set(str(chain) for chain in target_chains or [])
    rows: list[dict[str, object]] = []
    for job in collect_jobs(DESIGN_CAMPAIGN_GROUP):
        run_dir = Path(str(job.get("run_dir") or ""))
        input_payload = read_json(run_dir / "input.json")
        inputs = input_payload.get("inputs") if isinstance(input_payload.get("inputs"), dict) else {}
        params = input_payload.get("params") if isinstance(input_payload.get("params"), dict) else {}
        job_target = str(inputs.get("target_pdb") or "").strip()
        if not job_target:
            continue
        try:
            job_resolved = str(Path(job_target).expanduser().resolve())
        except Exception:
            job_resolved = job_target
        if job_target != selected_path and job_resolved != selected_resolved:
            continue
        job_chains = [str(chain) for chain in inputs.get("target_chains") or []]
        if selected_chain_set and job_chains and set(job_chains) != selected_chain_set:
            continue
        engines = [
            ENGINE_LABELS.get(str(engine), str(engine))
            for engine in params.get("engines") or []
            if str(engine)
        ]
        hotspots = str(params.get("hotspots") or "").strip()
        result = read_json(run_dir / "result.json")
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        job_code = str(job.get("job_code") or "")
        result_url = result_link(DESIGN_CAMPAIGN_GROUP, str(job.get("run_id") or ""))
        if job_code:
            result_url = f"{result_url}&job_code={job_code}"
        rows.append(
            {
                "job": result_url,
                "status": job.get("status") or "",
                "workflow": str(params.get("workflow_recipe") or "vanilla"),
                "campaign": job.get("campaign_name") or params.get("campaign_name") or "",
                "engines": ", ".join(engines),
                "hotspots": hotspots or "none",
                "candidates": metrics.get("candidate_count", ""),
                "updated": job.get("updated_at") or "",
            }
        )
    return rows


def _structure_chain_ids(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    chains: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) <= 21:
            continue
        chain = line[21].strip() or "_"
        if chain not in seen:
            seen.add(chain)
            chains.append(chain)
    return chains


def _target_has_residue_gaps(path: Path, chains: list[str]) -> bool:
    selected = set(chains or [])
    residues_by_chain: dict[str, list[int]] = {}
    last_seen: dict[str, tuple[int, str] | None] = {}
    for line in path.read_text(errors="ignore").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
            continue
        chain = line[21].strip() or "_"
        if selected and chain not in selected:
            continue
        try:
            residue = int(line[22:26])
        except ValueError:
            continue
        residue_key = (residue, line[26].strip())
        if last_seen.get(chain) == residue_key:
            continue
        residues_by_chain.setdefault(chain, []).append(residue)
        last_seen[chain] = residue_key
    return any(
        any(next_residue > residue + 1 for residue, next_residue in zip(residues, residues[1:]))
        for residues in residues_by_chain.values()
    )


def _restricted_refolding_options_for_target(target: dict, target_pdb: Path, chains: list[str]) -> list[str]:
    category = str(target.get("source_category") or target.get("prepared_kind") or "").lower()
    if category == "cropped" or _target_has_residue_gaps(target_pdb, chains):
        return ["AF2-IG", "Boltz-2"]
    return list(REFOLDING_ENGINE_OPTIONS)


def _uploaded_template_path(uploaded_file: object | None) -> Path | None:
    if uploaded_file is None:
        return None
    upload_dir = runs_root().parent / "uploads" / "design_campaign_templates"
    upload_dir.mkdir(parents=True, exist_ok=True)
    name = Path(str(getattr(uploaded_file, "name", "template_complex.pdb"))).name
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name) or "template_complex.pdb"
    path = upload_dir / safe_name
    path.write_bytes(uploaded_file.getvalue())
    return path


st.title("Design Campaigns")

targets = prepared_design_targets()
if not targets:
    st.info("Add, prepare, crop, or select an installed benchmark target before starting a design campaign.")
    st.stop()

active_workflow_recipe = str(st.session_state.get("design_campaign_workflow_recipe") or "vanilla")
if active_workflow_recipe in GENERATOR_ONLY_RECIPES:
    target_tab, workflows_tab, evaluation_tab, run_tab, results_tab = st.tabs(
        ["Target", "Workflows", "Evaluation Refolding/Metric", "Review & Run", "Results"]
    )
    harmonize_tab = None
else:
    target_tab, workflows_tab, evaluation_tab, run_tab, results_tab = st.tabs(
        ["Target", "Workflows", "Evaluation Refolding/Metric", "Review & Run", "Results"]
    )
    harmonize_tab = workflows_tab

target_rows = _design_target_chain_rows(targets)
with target_tab:
    if target_rows.empty:
        st.info("No readable prepared, imported, cropped, mutated, or benchmark target structures are available yet.")
        st.stop()
    filter_cols = st.columns([1, 4])
    category_options = _design_target_category_options(str(value) for value in target_rows["category"].dropna().unique() if str(value))
    selected_categories = filter_cols[0].multiselect(
        "Category",
        category_options,
        default=category_options,
        key="design_campaign_target_categories",
    )
    search_text = filter_cols[1].text_input(
        "Search targets",
        value="",
        placeholder="Target name, chain, source, or job code",
        key="design_campaign_target_search",
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
        st.stop()

    selected_path_state = str(st.session_state.get("design_campaign_selected_target_pdb") or "")
    selected_chains_state = set(st.session_state.get("design_campaign_selected_target_chains") or [])
    table_rows = filtered_rows.reset_index(drop=True)
    target_table_event = st.dataframe(
        table_rows.drop(columns=["_target_index"]),
        hide_index=True,
        width="stretch",
        height=290,
        column_config={
            "chains": st.column_config.TextColumn("Chains", width="small"),
            "chain_count": st.column_config.NumberColumn("Chain count", format="%d"),
            "aa_length": st.column_config.NumberColumn("AA", format="%d"),
            "fragments": st.column_config.NumberColumn("Fragments", format="%d"),
            "breaks": st.column_config.NumberColumn("Breaks", format="%d"),
            "ppi_hotspot": st.column_config.TextColumn("PPI/hotspot runs", width="medium"),
            "path": st.column_config.TextColumn("PDB path", width="large"),
        },
        selection_mode="single-row",
        on_select="rerun",
        key="design_campaign_target_structure_table",
    )
    selected_positions = list(getattr(target_table_event.selection, "rows", []) or [])
    if selected_positions:
        selected_position = int(selected_positions[0])
    else:
        selected_position = 0
        if selected_path_state:
            visible_matches = table_rows[
                (table_rows["path"].astype(str) == selected_path_state)
                & (
                    table_rows["chains"].astype(str).apply(
                        lambda value: not selected_chains_state
                        or set(part.strip() for part in value.split(",") if part.strip()) == selected_chains_state
                    )
                )
            ]
            if not visible_matches.empty:
                selected_position = int(visible_matches.index[0])
    selected_entry = _design_selected_target_entry(table_rows.iloc[selected_position], targets)
    selected_path = str(selected_entry["target_pdb"])
    selected_entries = [selected_entry]
    target = dict(selected_entry.get("_target") or {})
    target_pdb = Path(selected_path)
    available_chains = _target_chain_ids(target)
    selected_chains = []
    for entry in selected_entries:
        for chain in entry.get("chains") or []:
            if str(chain) and str(chain) not in selected_chains:
                selected_chains.append(str(chain))
    if not selected_chains:
        selected_chains = available_chains
    st.session_state["design_campaign_selected_target_pdb"] = str(target_pdb)
    st.session_state["design_campaign_selected_target_chains"] = selected_chains
    selected_breaks = sum(int(entry.get("breaks") or 0) for entry in selected_entries)
    if selected_breaks:
        st.warning(
            f"The selected target contains {selected_breaks} residue-number break(s). "
            "Fragment-compatible refolding choices will be used later where needed."
        )
    history_rows = _campaign_history_for_target(target_pdb, selected_chains)
    with st.expander("Previous campaigns for selected target", expanded=bool(history_rows)):
        if history_rows:
            st.dataframe(
                pd.DataFrame(history_rows),
                hide_index=True,
                width="stretch",
                column_config={
                    "job": st.column_config.LinkColumn("job", display_text=r".*[?&]job_code=([^&]+).*"),
                    "workflow": st.column_config.TextColumn("workflow"),
                    "campaign": st.column_config.TextColumn("campaign", width="medium"),
                    "engines": st.column_config.TextColumn("engines", width="large"),
                    "hotspots": st.column_config.TextColumn("hotspots", width="large"),
                    "updated": st.column_config.TextColumn("updated"),
                },
            )
        else:
            st.caption("No previous design campaigns found for this exact target/chains selection.")

    designs_per_attempt_forward = max(
        1,
        int(
            st.session_state.get(
                "dc_staged_level1_keep_per_attempt",
                st.session_state.get("design_campaign_designs_per_attempt_forward", 2),
            )
            or 2
        ),
    )
    sequence_count_min = max(1, int(designs_per_attempt_forward))
    target_cols = st.columns([2, 1, 1])
    with target_cols[0]:
        campaign_name = st.text_input("Campaign name", placeholder="PDL1 multi-engine 60aa")
    with target_cols[1]:
        design_attempts = st.number_input(
            "Design attempts",
            min_value=1,
            max_value=1000,
            value=2,
            step=1,
            key="design_campaign_design_attempts",
        )
    with target_cols[2]:
        random_seed = st.number_input("Random seed", min_value=0, max_value=2147483647, value=0, step=1)
    sequences_per_backbone_key = "design_campaign_sequences_per_backbone"
    sequences_per_backbone = int(st.session_state.get(sequences_per_backbone_key, 20) or 20)

    length_mode = st.radio(
        "Binder length",
        ["range", "fixed"],
        horizontal=True,
        index=0,
        format_func={"fixed": "Fixed length", "range": "Length range"}.get,
        help="Range-aware engines receive min-max; fixed-only engines currently use the minimum value.",
        key="design_campaign_binder_length_mode",
    )
    length_cols = st.columns(2 if length_mode == "range" else 1)
    if length_mode == "range":
        if int(st.session_state.get("design_campaign_binder_length_min", 60) or 60) == 55:
            st.session_state["design_campaign_binder_length_min"] = 60
        if int(st.session_state.get("design_campaign_binder_length_max", 120) or 120) == 80:
            st.session_state["design_campaign_binder_length_max"] = 120
        with length_cols[0]:
            binder_length_min = st.number_input(
                "Minimum binder length",
                min_value=1,
                max_value=1000,
                value=60,
                step=1,
                key="design_campaign_binder_length_min",
            )
        with length_cols[1]:
            binder_length_max = st.number_input(
                "Maximum binder length",
                min_value=1,
                max_value=1000,
                value=max(120, int(binder_length_min)),
                step=1,
                key="design_campaign_binder_length_max",
            )
        if int(binder_length_max) < int(binder_length_min):
            st.error("Maximum binder length must be greater than or equal to the minimum.")
        binder_length = f"{int(binder_length_min)}-{int(binder_length_max)}"
    else:
        with length_cols[0]:
            binder_length_fixed = st.number_input(
                "Fixed binder length",
                min_value=1,
                max_value=1000,
                value=60,
                step=1,
                key="design_campaign_binder_length_fixed",
            )
        binder_length = str(int(binder_length_fixed))
    with st.expander("Binder length support by generator", expanded=False):
        st.dataframe(pd.DataFrame(BINDER_LENGTH_SUPPORT), hide_index=True, use_container_width=True)
    st.markdown("**Target hotspot selection**")
    st.caption(
        "Select target residues for hotspot-aware generators and downstream hotspot/contact metrics. "
        "Use detection-guided suggestions when available, or click residues in the Mol* viewer."
    )
    target_text = target_pdb.read_text(errors="ignore")
    target_view_text = (
        filter_pdb_text(target_text, keep_chains=set(selected_chains))
        if selected_chains
        else target_text
    )
    hotspot_key = f"design_campaign_hotspots_{target_pdb}"
    hotspot_text_key = f"{hotspot_key}_text"
    hotspot_suppress_auto_add_key = f"{hotspot_key}_suppress_auto_add"
    st.session_state.setdefault(hotspot_key, set())
    st.session_state.setdefault(hotspot_text_key, _hotspot_text(set(st.session_state[hotspot_key])))
    typed_hotspots = {
        residue for residue in _hotspot_residues(str(st.session_state.get(hotspot_text_key) or "")) if residue[0] in set(selected_chains)
    }
    if typed_hotspots != set(st.session_state[hotspot_key]):
        st.session_state[hotspot_key] = typed_hotspots
    active_hotspots = {
        residue for residue in set(st.session_state[hotspot_key]) if residue[0] in set(selected_chains)
    }
    sources = detection_score_sources(target_pdb, selected_chains)
    detected_residues: set[ResidueId] = set()
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
            source_names = st.multiselect("Detection results", list(sources), default=list(sources), key=f"dc_detection_sources_{target_pdb}")
            threshold_cols = st.columns(2)
            prediction_threshold = threshold_cols[0].number_input("Prediction threshold", 0.0, 1.0, 0.50, 0.01, key=f"dc_detection_threshold_{target_pdb}")
            masif_threshold = threshold_cols[1].number_input("MaSIF score threshold", 0.0, 1.0, 0.50, 0.01, key=f"dc_masif_threshold_{target_pdb}")
            sasa_threshold = st.number_input("Minimum side-chain SASA (A2)", 0.0, 500.0, 5.0, 1.0, key=f"dc_detection_sasa_{target_pdb}")
            show_existing_hotspots = st.checkbox("Also show existing hotspots", value=False, key=f"dc_detection_existing_{target_pdb}")
            scores = pd.concat([sources[name].assign(source=name) for name in source_names], ignore_index=True) if source_names else pd.DataFrame(columns=["chain", "residue", "amino_acid", "score", "source"])
            scores["tool"] = scores["source"].str.split(" | ").str[0]
            scores["score_threshold"] = scores["tool"].map(lambda tool: masif_threshold if tool == "masif_seed" else prediction_threshold)
            sasa = sidechain_sasa_by_residue(target_pdb)
            scores["sidechain_sasa_a2"] = [sasa.get((row.chain, int(row.residue)), 0.0) for row in scores.itertuples()]
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
            edited_scores = st.data_editor(selected_scores, hide_index=True, width="stretch", key=f"dc_detection_rows_{target_pdb}")
            include_mask = edited_scores["include"].fillna(False) if "include" in edited_scores else pd.Series(False, index=edited_scores.index)
            checked_scores = edited_scores[include_mask].copy()
            detected_residues = {(str(row.chain), int(row.residue)) for row in checked_scores.itertuples()}
            tool_options = sorted(set(checked_scores["tool"].dropna().astype(str))) if "tool" in checked_scores else []
            preview_tools = st.multiselect(
                "Preview tools",
                tool_options,
                default=tool_options,
                key=f"dc_detection_preview_tools_{target_pdb}",
                help="Choose which checked PPI/hotspot calculations are shown in the preview and sequence strip.",
            )
            preview_scores = checked_scores[checked_scores["tool"].astype(str).isin(preview_tools)].copy() if preview_tools else checked_scores.iloc[0:0].copy()
            preview_residues = {(str(row.chain), int(row.residue)) for row in preview_scores.itertuples()}
            detection_view_token = (
                f"{','.join(source_names)}:{prediction_threshold:.2f}:{masif_threshold:.2f}:"
                f"{sasa_threshold:.1f}:{','.join(preview_tools)}"
            )
            add_cols = st.columns([1, 1, 4])
            if add_cols[0].button("Add checked rows", key=f"dc_detection_add_{target_pdb}"):
                st.session_state[hotspot_key] = active_hotspots | detected_residues
                st.session_state[hotspot_text_key] = _hotspot_text(set(st.session_state[hotspot_key]))
                st.rerun()
            preview_differs = bool(preview_residues) and preview_residues != detected_residues
            if add_cols[1].button(
                "Add preview only",
                disabled=not preview_differs,
                key=f"dc_detection_add_preview_{target_pdb}",
                help="Add only residues from the selected Preview tools instead of every checked table row.",
            ):
                st.session_state[hotspot_key] = active_hotspots | preview_residues
                st.session_state[hotspot_text_key] = _hotspot_text(set(st.session_state[hotspot_key]))
                st.rerun()
            _detection_tool_legend(set(preview_scores["tool"].dropna().astype(str)) if "tool" in preview_scores else set())
            st.caption(
                f"{len(detected_residues)} checked table residues are selected; "
                f"{len(preview_residues)} residues from the selected Preview tools are highlighted in the left viewer."
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
            detection_preview_key = f"dc_detection_viewer_{target.get('run_id', target_pdb.name)}_{detection_view_token}"
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
                key=f"design_campaign_target_viewer_{target.get('run_id', target_pdb.name)}_{','.join(selected_chains)}_{detection_view_token}",
                height=viewer_square_height,
                show_controls=True,
                selection_mode=True,
            )
    else:
        viewer_value = molstar_custom_component(
            main_viewer_structures,
            key=f"design_campaign_target_viewer_{target.get('run_id', target_pdb.name)}_{','.join(selected_chains)}_{detection_view_token}",
            height=560,
            show_controls=True,
            selection_mode=True,
        )
    clicked_hotspots = _viewer_residues(viewer_value, target_view_text)
    suppressed_auto_add = _hotspot_residues(str(st.session_state.get(hotspot_suppress_auto_add_key) or ""))
    if clicked_hotspots != suppressed_auto_add:
        st.session_state[hotspot_suppress_auto_add_key] = ""
        suppressed_auto_add = set()
    if clicked_hotspots and clicked_hotspots != suppressed_auto_add and not clicked_hotspots.issubset(active_hotspots):
        active_hotspots = active_hotspots | clicked_hotspots
        st.session_state[hotspot_key] = active_hotspots
        st.session_state[hotspot_text_key] = _hotspot_text(active_hotspots)
        st.rerun()
    hotspot_buttons = st.columns([1, 1, 1, 3])
    if hotspot_buttons[0].button("Add clicked", disabled=not clicked_hotspots, key=f"dc_hotspot_add_clicked_{target_pdb}"):
        st.session_state[hotspot_suppress_auto_add_key] = ""
        st.session_state[hotspot_key] = active_hotspots | clicked_hotspots
        st.session_state[hotspot_text_key] = _hotspot_text(set(st.session_state[hotspot_key]))
        st.rerun()
    if hotspot_buttons[1].button("Remove clicked", disabled=not clicked_hotspots, key=f"dc_hotspot_remove_clicked_{target_pdb}"):
        st.session_state[hotspot_key] = active_hotspots - clicked_hotspots
        st.session_state[hotspot_text_key] = _hotspot_text(set(st.session_state[hotspot_key]))
        st.session_state[hotspot_suppress_auto_add_key] = _hotspot_text(clicked_hotspots)
        st.rerun()
    if hotspot_buttons[2].button("Clear", disabled=not active_hotspots, key=f"dc_hotspot_clear_{target_pdb}"):
        st.session_state[hotspot_key] = set()
        st.session_state[hotspot_text_key] = ""
        st.session_state[hotspot_suppress_auto_add_key] = _hotspot_text(clicked_hotspots)
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
        st.session_state[hotspot_text_key] = _hotspot_text(parsed_hotspots)
        active_hotspots = parsed_hotspots
    hotspots = _hotspot_text(active_hotspots)
    template_enabled = st.checkbox(
        "Start from an existing target-binder complex",
        value=False,
        key="design_campaign_template_enabled",
    )
    template_complex_pdb: Path | None = None
    template_binder_chains: list[str] = []
    template_target_chains: list[str] = list(selected_chains)
    template_lock_mode = "interface"
    template_locked_residues = ""
    template_unlocked_residues = ""
    if template_enabled:
        template_upload = st.file_uploader(
            "Template complex PDB",
            type=["pdb"],
            key="design_campaign_template_upload",
        )
        uploaded_path = _uploaded_template_path(template_upload)
        template_path_text = st.text_input(
            "Template complex path",
            value=str(uploaded_path or ""),
            placeholder="/path/to/target_binder_complex.pdb",
            key="design_campaign_template_path",
        )
        if template_path_text.strip():
            template_complex_pdb = Path(template_path_text).expanduser()
        template_chains = _structure_chain_ids(template_complex_pdb) if template_complex_pdb else []
        template_cols = st.columns(3)
        with template_cols[0]:
            template_target_chains = st.multiselect(
                "Template target chains",
                template_chains,
                default=[chain for chain in selected_chains if chain in template_chains],
                key="design_campaign_template_target_chains",
            )
        with template_cols[1]:
            default_binders = [chain for chain in template_chains if chain not in set(template_target_chains)]
            template_binder_chains = st.multiselect(
                "Template binder chains",
                template_chains,
                default=default_binders[-1:] if default_binders else [],
                key="design_campaign_template_binder_chains",
            )
        with template_cols[2]:
            template_lock_mode = st.selectbox(
                "Template redesign",
                [
                    "interface",
                    "all",
                    "selected_unlocked",
                    "selected_locked",
                ],
                format_func={
                    "interface": "Redesign non-interface",
                    "all": "Redesign full binder",
                    "selected_unlocked": "Redesign selected residues",
                    "selected_locked": "Lock selected residues",
                }.get,
                key="design_campaign_template_lock_mode",
            )
        if template_lock_mode == "selected_unlocked":
            template_unlocked_residues = st.text_input(
                "Residues to redesign",
                placeholder="B12,B18,B22 or 12,18,22 for the first binder chain",
                key="design_campaign_template_unlocked_residues",
            )
        elif template_lock_mode == "selected_locked":
            template_locked_residues = st.text_input(
                "Residues to keep fixed",
                placeholder="B12,B18,B22 or 12,18,22 for the first binder chain",
                key="design_campaign_template_locked_residues",
            )
        if template_complex_pdb and template_complex_pdb.exists():
            template_view_text = filter_pdb_text(
                template_complex_pdb.read_text(errors="ignore"),
                keep_chains=set([*template_target_chains, *template_binder_chains]) or None,
            )
            molstar_custom_component(
                [
                    StructureVisualization(
                        pdb=template_view_text,
                        color="chain-id",
                        color_params={"palette": "pastel-2"},
                        representation_type="cartoon",
                    )
                ],
                key=f"design_campaign_template_viewer_{template_complex_pdb}",
                height=420,
                show_controls=True,
                selection_mode=False,
            )
        elif template_path_text.strip():
            st.warning("Template complex path is not readable.")
    st.caption(f"{target_pdb.name} | {','.join(selected_chains) or 'no chains selected'}")

with workflows_tab:
    target_refolding_options = _restricted_refolding_options_for_target(target, target_pdb, selected_chains)
    target_refolding_restricted = set(target_refolding_options) != set(REFOLDING_ENGINE_OPTIONS)
    if target_refolding_restricted:
        st.warning(
            "The selected target is cropped or contains residue-number gaps. "
            "Refolding choices are restricted to AF2-IG and Boltz-2 so the target/template handling remains compatible with fragmented targets."
        )
    workflow_recipe = st.segmented_control(
        "Workflow recipe",
        ["vanilla", "staged", "engine_scout"],
        default="vanilla",
        format_func={
            "vanilla": "Vanilla multi-engine",
            "staged": "Backbone -> sequence selection",
            "engine_scout": "Engine scout / pilot",
        }.get,
        key="design_campaign_workflow_recipe",
    )
    generator_only_recipe = workflow_recipe in GENERATOR_ONLY_RECIPES
    staged_engine_settings_slot = None
    scout_pilot_outputs_per_engine = int(design_attempts)
    if generator_only_recipe:
        if workflow_recipe == "staged":
            staged_design_tab, staged_sequence_tab = st.tabs(["Design Engines", "Sequence Selection"])
        else:
            staged_design_tab = st.container()
            staged_sequence_tab = None
        staged_engine_order = [
            engine
            for engine in ENGINE_ORDER
            if engine != "template_redesign" and engine not in VANILLA_ONLY_ENGINES
        ]
        with staged_design_tab:
            staged_default_engines = list(staged_engine_order)
            selected_engines = st.multiselect(
                "Design engines" if workflow_recipe == "engine_scout" else "Backbone engines",
                staged_engine_order,
                default=staged_default_engines,
                format_func=lambda value: ENGINE_LABELS[value],
                key="dc_staged_design_engines",
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "engine": ENGINE_LABELS["rfdiffusion_classic"],
                            "staged handoff": "RFdiffusion backbone PDB",
                            "native downstream skipped": "MPNN, monomer refold, complex refold, ranking",
                        },
                        {
                            "engine": ENGINE_LABELS["bindcraft"],
                            "staged handoff": "BindCraft AFDesign trajectory/backbone before MPNN",
                            "native downstream skipped": "BindCraft ProteinMPNN redesign and accepted-MPNN validation",
                        },
                        {
                            "engine": ENGINE_LABELS["rfdiffusion3_foundry"],
                            "staged handoff": "RFdiffusion3 generated CIF",
                            "native downstream skipped": "Foundry MPNN, RF3 folding",
                        },
                        {
                            "engine": ENGINE_LABELS["boltzgen"],
                            "staged handoff": "BoltzGen intermediate design CIF",
                            "native downstream skipped": "final budget/ranking stage",
                        },
                        {
                            "engine": ENGINE_LABELS["pxdesign"],
                            "staged handoff": "PXDesign generated structure",
                            "native downstream skipped": "pipeline evaluation mode",
                        },
                        {
                            "engine": ENGINE_LABELS["genie3"],
                            "staged handoff": "Genie3 generated structure",
                            "native downstream skipped": "native folding evaluation",
                        },
                        {
                            "engine": ENGINE_LABELS["protpardelle_1c"],
                            "staged handoff": "Protpardelle scaffold sample",
                            "native downstream skipped": "native MPNN sequence generation",
                        },
                        {
                            "engine": ENGINE_LABELS["esmfold2_binder_design"],
                            "staged handoff": "native ESMFold2 designed complex",
                            "native downstream skipped": "none; this engine designs sequence and structure together",
                        },
                        {
                            "engine": ENGINE_LABELS["proteina_complexa"],
                            "staged handoff": "native Proteina-Complexa design candidate",
                            "native downstream skipped": "none; adapter emits its designed candidate set",
                        },
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            if workflow_recipe == "engine_scout":
                scout_cols = st.columns(3)
                scout_cols[0].metric("Pilot outputs per engine", str(int(design_attempts)))
                scout_cols[1].metric("Selected engines", str(len(selected_engines)))
                scout_cols[2].metric("Maximum pilot pool", str(int(design_attempts) * len(selected_engines)))
                st.caption(
                    "Engine scout runs generator-only outputs for all selected engines. "
                    "The per-engine pilot size follows Design attempts from the Target tab. "
                    "No sequence redesign or evaluation refolding is run in this workflow."
                )
            show_staged_engine_settings = st.checkbox(
                "Show advanced design-engine settings",
                value=False,
                key="dc_staged_show_engine_settings",
            )
            staged_engine_settings_slot = st.container()
        staged_sequence_by_engine: dict[str, dict] = {}
        if workflow_recipe == "engine_scout":
            staged_two_step = False
            staged_level1_sequence_engine = "ProteinMPNN"
            staged_level1_sequences = int(sequence_count_min)
            staged_level1_temperature = 0.1
            staged_level1_omit_aas = "C,X"
            staged_level1_redesign_engines = []
            staged_screen_refolder = "AF2-IG"
            staged_rank_metric = "iptm"
            staged_rmsd_cutoff = 3.5
            staged_level1_keep_per_attempt = int(designs_per_attempt_forward)
            staged_screen_settings = {"num_recycles": 3}
            staged_run_level2 = False
            staged_second_engine = "Soluble ProteinMPNN"
            staged_second_sequences = int(sequence_count_min)
            staged_second_temperature = 0.1
            staged_second_omit_aas = "C,X"
            staged_level2_redesign_engines = []
            staged_second_screen_refolder = "AF2-IG"
            staged_second_rank_metric = "iptm"
            staged_second_rmsd_cutoff = 3.5
            staged_second_keep_total = 1
            staged_include_first_stage_outputs = True
            staged_structure_filter_enabled = True
            staged_min_structure_elements = 3
            staged_second_screen_settings = {"num_recycles": 3}
            staged_sequence_by_engine = {
                engine: {"enabled": False, "engine": "ProteinMPNN"}
                for engine in selected_engines
            }
            staged_second_sequence_by_engine = {
                engine: {"enabled": False, "engine": "Soluble ProteinMPNN"}
                for engine in selected_engines
            }
            staged_sequence_enabled = False
        else:
            with staged_sequence_tab:
                staged_two_step = st.checkbox(
                    "Two-step redesign with internal refolding screen",
                    value=True,
                    key="dc_staged_two_step_redesign",
                )
                scout_pilot_outputs_per_engine = 1_000_000
                st.caption(
                    "The same sequence-design settings are applied to all selected generators. "
                    "ipSAE is not used here because it is only calculated in the final Evaluation step."
                )
                st.markdown("**Structure prefilter before sequence redesign**")
                structure_filter_cols = st.columns([1, 1, 2])
                staged_structure_filter_enabled = structure_filter_cols[0].checkbox(
                    "Filter by secondary-structure elements",
                    value=True,
                    disabled=not staged_two_step,
                    key="dc_staged_structure_filter_enabled",
                    help=(
                        "Drop generated structures before Level 1 and Level 2 sequence redesign when they have too few "
                        "secondary-structure elements. Genie3 uses CA-trace geometry elements; other generators use PyDSSP elements."
                    ),
                )
                staged_min_structure_elements = structure_filter_cols[1].number_input(
                    "Minimum elements",
                    min_value=0,
                    max_value=100,
                    value=3,
                    step=1,
                    disabled=not (staged_two_step and staged_structure_filter_enabled),
                    key="dc_staged_min_structure_elements",
                    help="Default keeps structures with at least 3 total helix/sheet elements before entering sequence redesign.",
                )
                structure_filter_cols[2].caption(
                    "Applied before Level 1 and again before Level 2. "
                    "Genie3 is scored with CA-trace elements; all other generators use PyDSSP total elements."
                )
                st.markdown("**Level 1: sequence design for every generator attempt**")
                level1_cols = st.columns(4)
                if int(st.session_state.get("dc_staged_level1_sequences", max(20, sequence_count_min)) or 1) < sequence_count_min:
                    st.session_state["dc_staged_level1_sequences"] = sequence_count_min
                staged_level1_sequence_engine = level1_cols[0].selectbox(
                    "Sequence engine",
                    ["ProteinMPNN", "LigandMPNN", "Soluble ProteinMPNN"],
                    index=0,
                    disabled=not staged_two_step,
                    key="dc_staged_level1_sequence_engine",
                )
                staged_level1_sequences = level1_cols[1].number_input(
                    "Sequences per generator output",
                    sequence_count_min,
                    1000,
                    max(20, sequence_count_min),
                    disabled=not staged_two_step,
                    key="dc_staged_level1_sequences",
                    help="How many sequence variants are generated for each eligible generator output before the first refolding screen.",
                )
                staged_level1_temperature = level1_cols[2].number_input(
                    "Temperature",
                    0.0001,
                    2.0,
                    0.1,
                    step=0.05,
                    format="%.4f",
                    disabled=not staged_two_step,
                    key="dc_staged_level1_temperature",
                )
                staged_level1_omit_aas = level1_cols[3].text_input(
                    "Exclude amino acids",
                    value="C,X",
                    disabled=not staged_two_step,
                    key="dc_staged_level1_omit_aas",
                    help="One-letter amino-acid codes passed to LigandMPNN --omit_AA. Default C,X excludes cysteine and unknown residues.",
                ).strip().upper()
                level1_required_engines = [
                    engine for engine in selected_engines if engine not in STAGED_NATIVE_SEQUENCE_ENGINES
                ]
                level1_redesign_key = "dc_staged_level1_redesign_engines"
                level1_existing = (
                    st.session_state.get(level1_redesign_key)
                    if isinstance(st.session_state.get(level1_redesign_key), list)
                    else list(level1_required_engines)
                )
                st.session_state[level1_redesign_key] = list(
                    dict.fromkeys(
                        [
                            *[engine for engine in level1_existing if engine in selected_engines],
                            *level1_required_engines,
                        ]
                    )
                )
                staged_level1_redesign_engines = st.multiselect(
                    "Level-1 sequence redesign applies to generator outputs",
                    selected_engines,
                    default=st.session_state[level1_redesign_key],
                    format_func=lambda value: ENGINE_LABELS[value],
                    disabled=not staged_two_step,
                    key=level1_redesign_key,
                    help=(
                        "ESMFold2 binder design and Proteina-Complexa already emit sequence-bearing candidates, "
                        "so they pass through level 1 by default."
                    ),
                )
                staged_level1_redesign_engines = list(
                    dict.fromkeys([*staged_level1_redesign_engines, *level1_required_engines])
                )
                if level1_required_engines:
                    st.caption(
                        "Backbone-only generators always receive Level-1 sequence design. "
                        "Native sequence generators can pass through Level 1 and still enter Level 2."
                    )
                st.markdown("**Level 1 screen: keep output(s) per generator attempt**")
                staged_two_step_cols = st.columns(4)
                if st.session_state.get("dc_staged_screen_refolder") not in target_refolding_options:
                    st.session_state["dc_staged_screen_refolder"] = "AF2-IG"
                staged_screen_refolder = staged_two_step_cols[0].selectbox(
                    "Refolding engine",
                    target_refolding_options,
                    index=target_refolding_options.index("AF2-IG"),
                    disabled=not staged_two_step,
                    key="dc_staged_screen_refolder",
                )
                staged_rank_metric = _select_rank_metric(
                    staged_two_step_cols[1],
                    "Order by",
                    refolder=str(staged_screen_refolder),
                    disabled=not staged_two_step,
                    key="dc_staged_rank_metric",
                )
                staged_rmsd_cutoff = staged_two_step_cols[2].number_input(
                    "Maximum target-aligned binder RMSD (A)",
                    0.0,
                    50.0,
                    3.5,
                    0.1,
                    disabled=not staged_two_step,
                    key="dc_staged_rmsd_cutoff",
                )
                staged_level1_keep_per_attempt = staged_two_step_cols[3].number_input(
                    "Level-1 output keep per attempt",
                    min_value=1,
                    max_value=1000,
                    value=int(designs_per_attempt_forward),
                    step=1,
                    disabled=not staged_two_step,
                    key="dc_staged_level1_keep_per_attempt",
                    help=(
                        "After the level-1 refolding screen, keep this many ranked outputs per original generator attempt. "
                        "ESMFold2 binder design and Proteina-Complexa use the same number as native generator outputs per attempt."
                    ),
                )
                sequence_count_min = max(1, int(staged_level1_keep_per_attempt))
                staged_screen_settings = _screen_refolder_settings(
                    "dc_staged_level1_screen",
                    str(staged_screen_refolder),
                    disabled=not staged_two_step,
                )
                staged_run_level2 = st.checkbox(
                    "Run level 2 redesign",
                    value=True,
                    disabled=not staged_two_step,
                    key="dc_staged_run_level2",
                )
                st.markdown("**Level 2: redesign level-1 kept sequence(s)**")
                staged_second_cols = st.columns(4)
                if int(st.session_state.get("dc_staged_second_sequences", max(20, sequence_count_min)) or 1) < sequence_count_min:
                    st.session_state["dc_staged_second_sequences"] = sequence_count_min
                staged_second_engine = staged_second_cols[0].selectbox(
                    "Sequence engine",
                    ["ProteinMPNN", "LigandMPNN", "Soluble ProteinMPNN"],
                    index=2,
                    disabled=not (staged_two_step and staged_run_level2),
                    key="dc_staged_second_engine",
                )
                staged_second_sequences = staged_second_cols[1].number_input(
                    "Sequences per Level-1 kept design",
                    sequence_count_min,
                    1000,
                    max(20, sequence_count_min),
                    disabled=not (staged_two_step and staged_run_level2),
                    key="dc_staged_second_sequences",
                    help="How many sequence variants are generated from each Level-1 kept design before the second refolding screen.",
                )
                staged_second_temperature = staged_second_cols[2].number_input(
                    "Temperature",
                    0.0001,
                    2.0,
                    0.1,
                    step=0.05,
                    format="%.4f",
                    disabled=not (staged_two_step and staged_run_level2),
                    key="dc_staged_second_temperature",
                )
                staged_second_omit_aas = staged_second_cols[3].text_input(
                    "Exclude amino acids",
                    value="C,X",
                    disabled=not (staged_two_step and staged_run_level2),
                    key="dc_staged_second_omit_aas",
                    help="One-letter amino-acid codes passed to LigandMPNN --omit_AA. Default C,X excludes cysteine and unknown residues.",
                ).strip().upper()
                level2_redesign_key = "dc_staged_level2_redesign_engines"
                if isinstance(st.session_state.get(level2_redesign_key), list):
                    st.session_state[level2_redesign_key] = [
                        engine for engine in st.session_state[level2_redesign_key] if engine in selected_engines
                    ]
                staged_level2_redesign_engines = st.multiselect(
                    "Level-2 sequence redesign applies to generator outputs",
                    selected_engines,
                    default=list(selected_engines),
                    format_func=lambda value: ENGINE_LABELS[value],
                    disabled=not (staged_two_step and staged_run_level2),
                    key=level2_redesign_key,
                    help="Outputs from unselected generators pass through unchanged to the level-2 refolding screen.",
                )
                st.markdown("**Level 2 screen: final staged outputs**")
                staged_second_screen_cols = st.columns(5)
                if st.session_state.get("dc_staged_second_screen_refolder") not in target_refolding_options:
                    st.session_state["dc_staged_second_screen_refolder"] = "AF2-IG"
                staged_second_screen_refolder = staged_second_screen_cols[0].selectbox(
                    "Refolding engine",
                    target_refolding_options,
                    index=target_refolding_options.index("AF2-IG"),
                    disabled=not (staged_two_step and staged_run_level2),
                    key="dc_staged_second_screen_refolder",
                )
                staged_second_rank_metric = _select_rank_metric(
                    staged_second_screen_cols[1],
                    "Order by",
                    refolder=str(staged_second_screen_refolder),
                    disabled=not (staged_two_step and staged_run_level2),
                    key="dc_staged_second_rank_metric",
                )
                staged_second_rmsd_cutoff = staged_second_screen_cols[2].number_input(
                    "Maximum target-aligned binder RMSD (A)",
                    0.0,
                    50.0,
                    3.5,
                    0.1,
                    disabled=not (staged_two_step and staged_run_level2),
                    key="dc_staged_second_rmsd_cutoff",
                )
                staged_second_keep_total = staged_second_screen_cols[3].number_input(
                    "Keep level-2 outputs",
                    1,
                    1000,
                    2,
                    disabled=not (staged_two_step and staged_run_level2),
                    key="dc_staged_second_keep_total",
                    help="After the level-2 refolding/RMSD screen, keep this many ranked final level-2 sequences.",
                )
                staged_include_first_stage_outputs = staged_second_screen_cols[4].checkbox(
                    "Include level-1 kept designs in final output",
                    value=True,
                    disabled=not staged_two_step,
                    key="dc_staged_include_first_stage_outputs",
                    help=(
                        "When level 2 is enabled, final staged outputs are the kept level-1 designs plus "
                        "the level-2 designs that pass the second screen. If level 2 is disabled, final outputs are the level-1 kept designs."
                    ),
                )
                st.caption(
                    "Final staged outputs include the level-1 kept designs"
                    + (" together with level-2 passing designs." if staged_run_level2 else ". Level 2 is disabled.")
                )
                staged_second_screen_settings = _screen_refolder_settings(
                    "dc_staged_level2_screen",
                    str(staged_second_screen_refolder),
                    disabled=not (staged_two_step and staged_run_level2),
                )
                staged_sequence_by_engine = {
                    engine: {
                        "enabled": engine in level1_required_engines or engine in staged_level1_redesign_engines,
                        "engine": staged_level1_sequence_engine,
                        "omit_aas": staged_level1_omit_aas or "C,X",
                    }
                    for engine in selected_engines
                }
                staged_second_sequence_by_engine = {
                    engine: {
                        "enabled": engine in staged_level2_redesign_engines,
                        "engine": staged_second_engine,
                        "omit_aas": staged_second_omit_aas or "C,X",
                    }
                    for engine in selected_engines
                }
                staged_sequence_enabled = bool(staged_two_step and selected_engines)
        if not selected_engines:
            st.warning("Select at least one backbone engine for the staged workflow.")
    else:
        show_staged_engine_settings = True
        staged_sequence_by_engine = {}
        staged_second_sequence_by_engine = {}
        vanilla_engine_order = [engine for engine in ENGINE_ORDER if engine != "template_redesign"]
        selected_engines = st.multiselect(
            "Vanilla workflows",
            vanilla_engine_order,
            default=["rfdiffusion_classic", "rfdiffusion3_foundry", "boltzgen", "esmfold2_binder_design"],
            format_func=lambda value: ENGINE_LABELS[value],
        )
    engine_settings_host = staged_engine_settings_slot if generator_only_recipe and staged_engine_settings_slot else st
    continue_after_failure = True

    engine_configs: dict[str, dict] = {}
    bindcraft2_campaign_ready = True
    show_engine_settings = not generator_only_recipe or bool(show_staged_engine_settings)
    if "rfdiffusion_classic" in selected_engines and show_engine_settings:
        with engine_settings_host.expander("RFdiffusion classic", expanded=True):
            if generator_only_recipe:
                cols = st.columns(2)
                rf_timesteps = cols[0].number_input("Diffusion timesteps", 1, 200, 50, key="dc_rf_timesteps")
                rf_model = cols[1].selectbox("Model weights", ["Complex_base", "Complex_beta"], key="dc_rf_model")
                rf_sequence_method = "protein_mpnn"
                rf_mpnn_sequences = int(sequences_per_backbone)
                rf_backend = "docker"
                rf_rank_metric = "iptm"
                rf_max_binder_rmsd = 3.5
                rf_keep_per_attempt = 1
            else:
                cols = st.columns(4)
                rf_timesteps = cols[0].number_input("Diffusion timesteps", 1, 200, 50, key="dc_rf_timesteps")
                rf_model = cols[1].selectbox("Model weights", ["Complex_base", "Complex_beta"], key="dc_rf_model")
                if st.session_state.get("dc_rf_sequence_method") not in MPNN_MODEL_LABELS:
                    st.session_state["dc_rf_sequence_method"] = "protein_mpnn"
                rf_sequence_method = cols[2].selectbox(
                    "Sequence design model",
                    ["protein_mpnn", "soluble_mpnn", "ligand_mpnn"],
                    format_func=MPNN_MODEL_LABELS.get,
                    key="dc_rf_sequence_method",
                )
                if int(st.session_state.get("dc_rf_mpnn_sequences", max(int(sequences_per_backbone), sequence_count_min)) or 1) < sequence_count_min:
                    st.session_state["dc_rf_mpnn_sequences"] = sequence_count_min
                rf_mpnn_sequences = cols[3].number_input(
                    "Sequences per backbone",
                    sequence_count_min,
                    1000,
                    max(int(sequences_per_backbone), sequence_count_min),
                    key="dc_rf_mpnn_sequences",
                    help="Must be at least the Target-tab designs-per-attempt-forward value.",
                )
                rf_analysis_cols = st.columns(3)
                rf_rank_metric = rf_analysis_cols[0].selectbox(
                    "Rank passing designs by",
                    ["iptm", "analysis_score", "binder_plddt", "binder_rmsd", "ipae"],
                    index=0,
                    format_func={
                        "iptm": "ipTM (higher)",
                        "analysis_score": "combined analysis score (higher)",
                        "binder_plddt": "binder pLDDT (higher)",
                        "binder_rmsd": "binder/target-aligned RMSD (lower)",
                        "ipae": "interface PAE (lower)",
                    }.get,
                    key="dc_rf_analysis_rank_metric",
                )
                rf_max_binder_rmsd = rf_analysis_cols[1].number_input(
                    "Maximum binder RMSD (A)",
                    0.0,
                    50.0,
                    3.5,
                    0.1,
                    key="dc_rf_max_binder_rmsd",
                    help="Applied by the RFdiffusion vanilla analysis step after AF2-IG complex refolding.",
                )
                rf_keep_per_attempt = rf_analysis_cols[2].number_input(
                    "Keep per backbone attempt",
                    1,
                    1000,
                    1,
                    key="dc_rf_keep_per_attempt",
                    help="After filters and ranking, keep this many sequences from each RFdiffusion backbone attempt.",
                )
                rf_backend = "docker"
            scaffold_status = rfdiffusion_scaffold_library_status()
            rf_scaffoldguided = st.checkbox(
                "Use scaffold-guided binder folds",
                value=bool(generator_only_recipe and scaffold_status.get("ready")),
                disabled=not bool(scaffold_status.get("ready")),
                help=(
                    "Uses the RFdiffusion PPI scaffold subset from "
                    f"{scaffold_status.get('path')}. Install it from the single RFdiffusion design page if needed."
                ),
                key=f"dc_rf_scaffoldguided_{workflow_recipe}",
            )
            if rf_scaffoldguided:
                st.caption(
                    "RFdiffusion will choose one random scaffold from /models/ppi_scaffolds/ at launch "
                    "and record the selected scaffold in the run artifacts."
                )
            elif not scaffold_status.get("ready"):
                st.caption(f"RFdiffusion scaffold library is not installed: `{scaffold_status.get('path')}`")
            rf_default_contig = default_target_contig(target_pdb, selected_chains, binder_length)
            st.text_input("Resolved contig sent to RFdiffusion", value=rf_default_contig, disabled=True)
            rf_contig = st.text_input(
                "Contig override",
                value="",
                placeholder="Automatically derived from target chains and binder length",
                key="dc_rf_contig",
            )
            if not generator_only_recipe:
                _rfdiffusion_vanilla_pipeline_table(
                    rf_sequence_method=str(rf_sequence_method),
                    rf_max_binder_rmsd=float(rf_max_binder_rmsd),
                    rf_rank_metric=str(rf_rank_metric),
                    rf_keep_per_attempt=int(rf_keep_per_attempt),
                )
            if generator_only_recipe:
                st.caption(
                    "Staged RFdiffusion only generates the binder backbone. Sequence design and internal refolding screens happen in the Sequence Selection tab."
                )
            engine_configs["rfdiffusion_classic"] = {
                "timesteps": int(rf_timesteps),
                "model_weights": rf_model,
                "sequence_design_method": rf_sequence_method,
                "execution_backend": rf_backend,
                "mpnn_num_sequences": int(rf_mpnn_sequences),
                "monomer_refolding_tool": "boltz2_monomer",
                "complex_refolding_tool": "af2_initial_guess",
                "complex_template_mode": "target_template",
                "complex_multimer": True,
                "complex_num_recycles": 3,
                "analysis_rank_metric": rf_rank_metric,
                "analysis_keep_per_attempt": int(rf_keep_per_attempt),
                "analysis_thresholds": {
                    "max_binder_rmsd": float(rf_max_binder_rmsd),
                    "min_iptm": 0.0,
                },
                "scaffoldguided": bool(rf_scaffoldguided),
                "contig": rf_contig.strip() or rf_default_contig,
                "analysis_keep_top_n": int(design_attempts),
            }

    if "bindcraft" in selected_engines and show_engine_settings:
        with engine_settings_host.expander("BindCraft", expanded=True):
            bindcraft_preset_options = BINDCRAFT_GENERATOR_PRESETS if generator_only_recipe else BINDCRAFT_VANILLA_PRESETS
            bindcraft_default_preset = "Default 4-stage generator" if generator_only_recipe else "Default 4-stage + MPNN"
            bindcraft_preset_key = f"dc_bindcraft_advanced_preset_{workflow_recipe}"
            bindcraft_preset_label = st.selectbox(
                "Design preset",
                list(bindcraft_preset_options),
                index=list(bindcraft_preset_options).index(bindcraft_default_preset),
                key=bindcraft_preset_key,
                help="Advanced BindCraft settings file. Staged mode uses generator-only presets because MPNN is handled later by the staged sequence-selection workflow.",
            )
            if generator_only_recipe:
                bindcraft_filter_settings = "no_filters.json"
            else:
                bindcraft_filter_label = st.selectbox(
                    "Native filter preset",
                    list(BINDCRAFT_FILTER_PRESETS),
                    index=list(BINDCRAFT_FILTER_PRESETS).index("No native filters"),
                    key="dc_bindcraft_filter_preset",
                    help="Default is no native filtering so campaign outputs can be harmonized first and filtered later.",
                )
                bindcraft_filter_settings = BINDCRAFT_FILTER_PRESETS[str(bindcraft_filter_label)]
                st.caption("Native BindCraft filtering is optional here; downstream scoring/filtering can be run later from Evaluation.")
            if generator_only_recipe:
                bindcraft_final = int(design_attempts)
                bindcraft_minutes = 0
                st.caption(
                    f"Staged BindCraft uses generator-only presets, disables native ProteinMPNN, and hands off up to {bindcraft_final} AFDesign backbones."
                )
            else:
                cols = st.columns(2)
                bindcraft_final = cols[0].number_input(
                    "Desired accepted designs",
                    1,
                    int(design_attempts),
                    min(2, int(design_attempts)),
                    key="dc_bindcraft_final",
                )
                bindcraft_minutes = cols[1].number_input("Time limit (minutes)", 0, 100000, 0, key="dc_bindcraft_minutes")
            engine_configs["bindcraft"] = {
                "number_of_final_designs": int(bindcraft_final),
                "time_limit_seconds": int(bindcraft_minutes) * 60 if bindcraft_minutes else None,
                "advanced_settings_file": bindcraft_preset_options[str(bindcraft_preset_label)],
                "filter_settings": bindcraft_filter_settings,
            }

    if "bindcraft2" in selected_engines and show_engine_settings:
        with engine_settings_host.expander("BindCraft 2", expanded=True):
            st.caption(
                "Runs the complete native campaign: trajectory generation, ProteinMPNN redesign, "
                "AlphaFold validation, native filters, and ranking. This is currently a vanilla-only workflow."
            )
            image_ready = bindcraft2_workflow.bindcraft2_image_available()
            if not image_ready:
                st.warning(
                    f"Container missing: {bindcraft2_workflow.BINDCRAFT2_IMAGE}. "
                    "Build it with `bash containers/bindcraft2/build.sh`."
                )
            reference_path = bindcraft2_workflow.bindcraft2_reference_directory()
            missing_parameters = bindcraft2_workflow.missing_bindcraft2_parameters()
            if missing_parameters:
                st.warning(
                    f"Seven AlphaFold checkpoints are required under {reference_path}; missing or incomplete: "
                    + ", ".join(missing_parameters)
                )
            elif image_ready:
                st.success("BindCraft 2 image and AlphaFold checkpoints are ready.")
            bindcraft2_campaign_ready = image_ready and not missing_parameters

            bc2_cols = st.columns(3)
            with bc2_cols[0]:
                bc2_binder_format = st.selectbox(
                    "BC2 binder format",
                    bindcraft2_workflow.BC2_BINDER_FORMATS,
                    key="dc_bindcraft2_modality",
                    help="These are BindCraft 2's native binder-format presets; BC1 presets are separate.",
                )
            with bc2_cols[1]:
                bc2_max_trajectories = st.number_input(
                    "Maximum trajectories",
                    min_value=1,
                    max_value=1000,
                    value=int(design_attempts),
                    key="dc_bindcraft2_max_trajectories",
                    help="BindCraft 2 stops after this many trajectories or after reaching the accepted-design target.",
                )
            with bc2_cols[2]:
                bc2_final_designs = st.number_input(
                    "Desired accepted designs",
                    min_value=1,
                    max_value=1000,
                    value=min(5, int(design_attempts)),
                    key="dc_bindcraft2_final_designs",
                )
            bc2_objective_options = [None, *bindcraft2_workflow.BC2_CONFORMATIONAL_OBJECTIVES] if bc2_binder_format == "binder" else [None]
            bc2_objective = st.selectbox(
                "BC2 conformational objective",
                bc2_objective_options,
                format_func=lambda value: "None" if value is None else value,
                key="dc_bindcraft2_objective",
                help="induced_fit and fold_switch are optional objectives for the de novo binder format.",
            )
            bc2_modality = [bc2_binder_format] + ([bc2_objective] if bc2_objective else [])
            bc2_binder_length = st.text_input(
                "BC2 binder-length override (optional)",
                value="",
                key=f"dc_bindcraft2_binder_length_{bc2_binder_format}",
                help="Leave blank to use the selected BC2 modality preset's length choices. Enter one length or a min-max range to override.",
            ).strip()
            preset_lengths = bindcraft2_workflow.BC2_PRESET_LENGTHS.get(bc2_binder_format)
            st.caption(
                "BC2 preset lengths: "
                + (f"{preset_lengths[0]}–{preset_lengths[1]} residues" if preset_lengths else "defined by the selected scaffold")
                + ". The shared length setting for other engines does not override this BC2 preset."
            )
            bc2_properties = st.multiselect(
                "BC2 design properties",
                list(bindcraft2_workflow.BC2_PROPERTY_PRESETS),
                format_func=lambda value: value.replace("_", " "),
                key="dc_bindcraft2_properties",
                help="Optional BindCraft 2 property presets. forced_targeting requires hotspots; some combinations are biologically incompatible.",
            )
            st.caption("BC2 loads its own modality and property preset defaults. The campaign hotspot selection above is applied to this workflow.")
            resource_cols = st.columns(2)
            with resource_cols[0]:
                bc2_workers = st.selectbox(
                    "Workers per GPU",
                    bindcraft2_workflow.WORKERS_PER_GPU_OPTIONS,
                    format_func=lambda value: "BC2 automatic" if value == "auto" else str(value),
                    key="dc_bindcraft2_workers_per_gpu",
                    help="Automatic uses BindCraft 2's memory-aware worker packing.",
                )
            with resource_cols[1]:
                bc2_cpu_options, bc2_default_cpu = bindcraft2_workflow.bindcraft2_cpu_options()
                bc2_cpu_cores = st.selectbox(
                    "Reserved CPU cores",
                    bc2_cpu_options,
                    index=bc2_cpu_options.index(bc2_default_cpu),
                    key="dc_bindcraft2_cpu_cores",
                    help="Reserved by the campaign scheduler and applied to the BindCraft 2 Docker container.",
                )
            engine_configs["bindcraft2"] = {
                "modality": bc2_modality,
                "binder_length": bc2_binder_length or None,
                "design_properties": bc2_properties,
                "max_trajectories": int(bc2_max_trajectories),
                "number_of_final_designs": int(bc2_final_designs),
                "campaign_seed": int(random_seed),
                "workers_per_gpu": bc2_workers,
                "cpu_cores": int(bc2_cpu_cores),
            }

    if "rfdiffusion3_foundry" in selected_engines and show_engine_settings:
        with engine_settings_host.expander("RFdiffusion3 / Foundry", expanded=True):
            foundry_timesteps = st.number_input("Diffusion steps", 1, 1000, 50, key="dc_foundry_timesteps")
            if generator_only_recipe:
                foundry_msa = False
                foundry_model = "protein_mpnn"
                st.caption("Staged RFdiffusion3/Foundry emits generated structures only; MPNN/MSA/refolding are handled later.")
            else:
                foundry_msa = st.checkbox("Prepare target MSA", value=True, key="dc_foundry_msa")
                foundry_model = st.selectbox(
                    "MPNN model",
                    ["protein_mpnn", "ligand_mpnn", "soluble_mpnn"],
                    format_func={
                        "protein_mpnn": "ProteinMPNN",
                        "ligand_mpnn": "LigandMPNN",
                        "soluble_mpnn": "Soluble ProteinMPNN",
                    }.get,
                )
            checkpoint_by_model = {
                "protein_mpnn": "/weights/proteinmpnn_v_48_020.pt",
                "ligand_mpnn": "/weights/ligandmpnn_v_32_010_25.pt",
                "soluble_mpnn": "/ligandmpnn_weights/solublempnn_v_48_020.pt",
            }
            engine_configs["rfdiffusion3_foundry"] = {
                "timesteps": int(foundry_timesteps),
                "mpnn_model_type": foundry_model,
                "mpnn_checkpoint_path": checkpoint_by_model[foundry_model],
                "prepare_target_msa": bool(foundry_msa),
            }

    if "boltzgen" in selected_engines and show_engine_settings:
        with engine_settings_host.expander("BoltzGen", expanded=True):
            if generator_only_recipe:
                boltz_budget = int(design_attempts)
                boltz_steps = st.number_input("Sampling steps", 1, 1000, 20, key="dc_boltz_steps")
                st.caption("Staged BoltzGen hands off generated designs before the native final ranking/budget stage.")
            else:
                cols = st.columns(2)
                boltz_budget = cols[0].number_input(
                    "Final budget", 1, 1000, min(2, int(design_attempts)), key="dc_boltz_budget"
                )
                boltz_steps = cols[1].number_input("Sampling steps", 1, 1000, 20, key="dc_boltz_steps")
            engine_configs["boltzgen"] = {
                "budget": min(int(boltz_budget), int(design_attempts)),
                "sampling_steps": int(boltz_steps),
            }

    if "pxdesign" in selected_engines and show_engine_settings:
        with engine_settings_host.expander("PXDesign", expanded=True):
            if generator_only_recipe:
                cols = st.columns(2)
                px_steps = cols[0].number_input("Diffusion steps", 1, 2000, 400, key="dc_px_steps")
                px_dtype = cols[1].selectbox("dtype", ["bf16", "fp32"], key="dc_px_dtype")
                px_preset = "preview"
                px_max_runs = 1
                px_msa = False
                px_fast_ln = True
                px_deepspeed = False
                st.caption("Staged PXDesign uses generation-only mode; pipeline evaluation/MSA settings are skipped.")
            else:
                cols = st.columns(4)
                px_preset = cols[0].selectbox("Pipeline preset", ["preview", "extended"], key="dc_px_preset")
                px_steps = cols[1].number_input("Diffusion steps", 1, 2000, 400, key="dc_px_steps")
                px_max_runs = cols[2].number_input("Maximum internal runs", 1, 1000, 1, key="dc_px_max_runs")
                px_dtype = cols[3].selectbox("dtype", ["bf16", "fp32"], key="dc_px_dtype")
                px_options = st.columns(3)
                px_msa = px_options[0].checkbox("Prepare target MSA", value=True, key="dc_px_msa")
                px_fast_ln = px_options[1].checkbox("Fast layer norm", value=True, key="dc_px_fast_ln")
                px_deepspeed = px_options[2].checkbox("DeepSpeed EvoAttention", value=False, key="dc_px_deepspeed")
            engine_configs["pxdesign"] = {
                "preset": px_preset,
                "n_steps": int(px_steps),
                "n_max_runs": int(px_max_runs),
                "dtype": px_dtype,
                "prepare_target_msa": bool(px_msa),
                "use_fast_ln": bool(px_fast_ln),
                "use_deepspeed_evo_attention": bool(px_deepspeed),
            }

    if "genie3" in selected_engines and show_engine_settings:
        with engine_settings_host.expander("Genie3", expanded=True):
            cols = st.columns(2 if generator_only_recipe else 4)
            genie_condition = cols[0].selectbox(
                "Conditioning",
                ["extended", "hotspot", "common", "iter_common", "iter_common_prob"],
                key="dc_genie_condition",
            )
            if generator_only_recipe:
                genie_direction = cols[1].number_input("Direction scale", 0.0, 2.0, 0.0, 0.1, key="dc_genie_direction")
                genie_folding = "colabfold"
                genie_models = 5
                genie_recycles = 20
                genie_beam = False
                genie_beam_width = 4
                genie_compile = False
                st.caption("Staged Genie3 emits generated structures; native folding evaluation is skipped.")
            else:
                genie_folding = cols[1].selectbox("Native refolder", ["colabfold", "boltz2"], key="dc_genie_folding")
                genie_models = cols[2].number_input("Folding models", 1, 10, 5, key="dc_genie_models")
                genie_recycles = cols[3].number_input("Folding recycles", 1, 50, 20, key="dc_genie_recycles")
                genie_advanced = st.columns(4)
                genie_direction = genie_advanced[0].number_input(
                    "Direction scale", 0.0, 2.0, 0.0, 0.1, key="dc_genie_direction"
                )
                genie_beam = genie_advanced[1].checkbox("Beam search", value=False, key="dc_genie_beam")
                genie_beam_width = genie_advanced[2].number_input(
                    "Beam width", 1, 16, 4, disabled=not genie_beam, key="dc_genie_beam_width"
                )
                genie_compile = genie_advanced[3].checkbox("Compile generation", value=False, key="dc_genie_compile")
            engine_configs["genie3"] = {
                "cond_strategy": genie_condition,
                "folding_model_name": genie_folding,
                "folding_mode": "msa" if genie_folding == "boltz2" else "template",
                "folding_num_models": int(genie_models),
                "folding_num_recycles": int(genie_recycles),
                "direction_scale": float(genie_direction),
                "enable_beam_search": bool(genie_beam),
                "beam_width": int(genie_beam_width),
                "compile_generation": bool(genie_compile),
            }

    if "esmfold2_binder_design" in selected_engines and show_engine_settings:
        with engine_settings_host.expander("ESMFold2 binder design", expanded=True):
            cols = st.columns(4 if generator_only_recipe else 3)
            esm_chain = cols[0].selectbox("Target chain", selected_chains or available_chains, key="dc_esm_chain")
            esm_steps = cols[1].number_input("Optimization steps", 1, 10000, 150, key="dc_esm_steps")
            esm_lr = cols[2].number_input(
                "Learning rate", min_value=0.001, max_value=10.0, value=0.1, step=0.01, format="%.3f", key="dc_esm_lr"
            )
            if generator_only_recipe:
                esm_default_outputs_per_attempt = 1 if workflow_recipe == "engine_scout" else 2
                esm_outputs_per_attempt = cols[3].number_input(
                    "Native designs per attempt",
                    1,
                    1000,
                    esm_default_outputs_per_attempt,
                    key=f"dc_esm_outputs_per_attempt_{workflow_recipe}",
                    help="ESMFold2 already emits sequence+structure candidates, so this sets how many native candidates each generator attempt produces.",
                )
                esm_compile = False
                esm_checkpoint = False
                st.caption("ESMFold2 already emits a sequence+structure candidate; staged sequence selection can pass it through or redesign it.")
            else:
                esm_outputs_per_attempt = 1
                esm_options = st.columns(2)
                esm_compile = esm_options[0].checkbox("Compile model", value=False, key="dc_esm_compile")
                esm_checkpoint = esm_options[1].checkbox("Checkpoint language model", value=False, key="dc_esm_checkpoint")
            engine_configs["esmfold2_binder_design"] = {
                "target_chain": esm_chain,
                "optimization_steps": int(esm_steps),
                "learning_rate": float(esm_lr),
                "generator_outputs_per_attempt": int(esm_outputs_per_attempt),
                "compile_model": bool(esm_compile),
                "checkpoint_lm": bool(esm_checkpoint),
            }

    if "protpardelle_1c" in selected_engines and show_engine_settings:
        with engine_settings_host.expander("Protpardelle-1c", expanded=True):
            cols = st.columns(3 if generator_only_recipe else 4)
            prot_model = cols[0].selectbox(
                "Model preset",
                ["cc83", "cc95"],
                format_func={"cc83": "cc83 epoch 2616", "cc95": "cc95 epoch 3490"}.get,
                key="dc_prot_model",
            )
            prot_batch = cols[1].number_input("Sampling batch size", 1, 128, 1, key="dc_prot_batch")
            prot_step_scale = cols[2].number_input("Step scale", 0.1, 5.0, 1.2, 0.1, key="dc_prot_step")
            if generator_only_recipe:
                prot_schurn = 200
                prot_crop = 0.0
                st.caption("Staged Protpardelle hands scaffold samples to the shared sequence-selection step.")
            else:
                prot_schurn = cols[3].number_input("Schurn", 0, 1000, 200, key="dc_prot_schurn")
                prot_crop = st.number_input(
                    "Crop conditioning start", 0.0, 1.0, 0.0, 0.05, key="dc_prot_crop"
                )
            engine_configs["protpardelle_1c"] = {
                "model_name": prot_model,
                "model_epoch": "3490" if prot_model == "cc95" else "2616",
                "sampling_config": "sampling_sidechain_conditional",
                "batch_size": int(prot_batch),
                "step_scale": float(prot_step_scale),
                "schurn": int(prot_schurn),
                "crop_cond_start": float(prot_crop),
            }

    if "proteina_complexa" in selected_engines and show_engine_settings:
        with engine_settings_host.expander("Proteina-Complexa", expanded=True):
            if generator_only_recipe:
                cols = st.columns(3)
                complexa_steps = cols[0].number_input("Generation steps", 1, 1000, 400, key="dc_complexa_steps")
                complexa_replicas = cols[1].number_input("Replicas", 1, 100, 1, key="dc_complexa_replicas")
                complexa_default_outputs_per_attempt = 1 if workflow_recipe == "engine_scout" else 2
                complexa_outputs_per_attempt = cols[2].number_input(
                    "Native designs per attempt",
                    1,
                    1000,
                    complexa_default_outputs_per_attempt,
                    key=f"dc_complexa_outputs_per_attempt_{workflow_recipe}",
                    help="Proteina-Complexa emits designed sequence+structure candidates directly.",
                )
                complexa_batch = 1
                st.caption("Proteina-Complexa emits designed candidates directly; staged sequence selection can pass through or redesign them.")
            else:
                cols = st.columns(3)
                complexa_steps = cols[0].number_input("Generation steps", 1, 1000, 400, key="dc_complexa_steps")
                complexa_replicas = cols[1].number_input("Best-of-N replicas", 1, 100, 2, key="dc_complexa_replicas")
                complexa_batch = cols[2].number_input("GPU batch size", 1, 16, 1, key="dc_complexa_batch")
                complexa_outputs_per_attempt = 1
            engine_configs["proteina_complexa"] = {
                "n_steps": int(complexa_steps),
                "replicas": int(complexa_replicas),
                "batch_size": int(complexa_batch),
                "generator_outputs_per_attempt": int(complexa_outputs_per_attempt),
            }

    checkpoint_by_model = {
        "protein_mpnn": "/weights/proteinmpnn_v_48_020.pt",
        "ligand_mpnn": "/weights/ligandmpnn_v_32_010_25.pt",
        "soluble_mpnn": "/ligandmpnn_weights/solublempnn_v_48_020.pt",
    }
    default_engine_configs = {
        "rfdiffusion_classic": {
            "timesteps": 50,
            "model_weights": "Complex_base",
            "sequence_design_method": "protein_mpnn",
            "execution_backend": "docker",
            "mpnn_num_sequences": int(sequences_per_backbone),
            "monomer_refolding_tool": "boltz2_monomer",
            "complex_refolding_tool": "af2_initial_guess",
            "complex_template_mode": "target_template",
            "complex_multimer": True,
            "complex_num_recycles": 3,
            "analysis_rank_metric": "iptm",
            "analysis_keep_per_attempt": 1,
            "analysis_thresholds": {"max_binder_rmsd": 3.5, "min_iptm": 0.0},
            "scaffoldguided": bool(
                generator_only_recipe
                and rfdiffusion_scaffold_library_status().get("ready")
            ),
            "contig": "",
            "analysis_keep_top_n": int(design_attempts),
        },
        "bindcraft": {
            "number_of_final_designs": int(design_attempts) if generator_only_recipe else min(2, int(design_attempts)),
            "time_limit_seconds": None,
            "advanced_settings_file": "default_4stage_multimer.json" if generator_only_recipe else "default_4stage_multimer_mpnn.json",
            "filter_settings": "no_filters.json",
        },
        "rfdiffusion3_foundry": {
            "timesteps": 50,
            "mpnn_model_type": "protein_mpnn",
            "mpnn_checkpoint_path": checkpoint_by_model["protein_mpnn"],
            "prepare_target_msa": True,
        },
        "boltzgen": {
            "budget": int(design_attempts) if generator_only_recipe else min(2, int(design_attempts)),
            "sampling_steps": 20,
        },
        "pxdesign": {
            "preset": "preview",
            "n_steps": 400,
            "n_max_runs": 1,
            "dtype": "bf16",
            "prepare_target_msa": True,
            "use_fast_ln": True,
            "use_deepspeed_evo_attention": False,
        },
        "genie3": {
            "cond_strategy": "extended",
            "folding_model_name": "colabfold",
            "folding_mode": "template",
            "folding_num_models": 5,
            "folding_num_recycles": 20,
            "direction_scale": 0.0,
            "enable_beam_search": False,
            "beam_width": 4,
            "compile_generation": False,
        },
        "esmfold2_binder_design": {
            "target_chain": (selected_chains or available_chains or [""])[0],
            "optimization_steps": 150,
            "learning_rate": 0.1,
            "generator_outputs_per_attempt": 2 if workflow_recipe == "staged" else 1,
            "compile_model": False,
            "checkpoint_lm": False,
        },
        "protpardelle_1c": {
            "model_name": "cc83",
            "model_epoch": "2616",
            "sampling_config": "sampling_sidechain_conditional",
            "batch_size": 1,
            "step_scale": 1.2,
            "schurn": 200,
            "crop_cond_start": 0.0,
        },
        "proteina_complexa": {
            "n_steps": 400,
            "replicas": 1 if generator_only_recipe else 2,
            "batch_size": 1,
            "generator_outputs_per_attempt": 2 if workflow_recipe == "staged" else 1,
        },
    }
    for engine in selected_engines:
        if engine in default_engine_configs:
            engine_configs.setdefault(engine, default_engine_configs[engine])

if harmonize_tab is not None:
    with harmonize_tab:
        st.markdown("**Harmonize vanilla workflow outputs**")
        st.caption(
            "Harmonization is always run for vanilla campaigns. It only normalizes and collects emitted workflow "
            "candidates into one downstream-ready result set; refolding, metrics, and filtering are handled later in Evaluation."
        )
        common_validation_enabled = False
        common_recycles = 3
        common_min_ipsae = 0.0
        common_pyrosetta_nprocs = 4
        survivors_per_engine = 1_000_000
        passing_only = False
        keep_best_failed = True
        if selected_engines:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "workflow": ENGINE_LABELS[engine],
                            "candidate pool": CANDIDATE_POOL_COVERAGE.get(engine, "native emitted candidates"),
                        }
                        for engine in selected_engines
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
else:
    common_validation_enabled = False
    common_recycles = 3
    common_min_ipsae = 0.0
    common_pyrosetta_nprocs = 4
    survivors_per_engine = 1_000_000
    passing_only = False
    keep_best_failed = True

refinement_mode = "none"
refinement_engine = "ProteinMPNN"
refinement_sequences = 2
refinement_temperature = 0.1
refinement_omit_aas = "C,X"

with evaluation_tab:
    refolder_options = list(target_refolding_options if target_refolding_restricted else REFOLDING_ENGINE_OPTIONS)
    default_evaluation_refolder = "AF2-IG" if target_refolding_restricted else "AlphaFast AF3"
    if default_evaluation_refolder not in refolder_options:
        default_evaluation_refolder = refolder_options[0] if refolder_options else "AF2-IG"
    if workflow_recipe == "engine_scout":
        st.info(
            "Evaluation refolding is disabled for Engine scout / pilot. "
            "This workflow should first compare generator outputs; run sequence design, refolding, or metric jobs later on the selected pilot candidates."
        )
        evaluation_refold = False
        evaluation_ipsae = False
        evaluation_rosetta = False
        evaluation_pymol = False
        evaluation_refolder = default_evaluation_refolder
        evaluation_refolders = []
        evaluation_refolder_settings = {}
        evaluation_recycles = 10
        evaluation_steps = 68
        evaluation_samples = 3
        evaluation_target_msa = False
        evaluation_rosetta_nprocs = 4
        evaluation_mode = "none"
    else:
        st.caption(
            "Optional final evaluation after the selected campaign recipe. Select metrics to score existing outputs, "
            "and enable final refolding when you want to score a newly refolded structure/PAE from one evaluation engine."
        )
        evaluation_cols = st.columns(4)
        evaluation_refold = evaluation_cols[0].checkbox(
            "Refold final outputs",
            value=True,
            key="design_campaign_evaluation_refold",
            help="Run one final refolding engine before metric calculation. Leave off to score existing workflow outputs/PAE only.",
        )
        evaluation_ipsae = evaluation_cols[1].checkbox("ipSAE", value=True, key="design_campaign_evaluation_ipsae")
        evaluation_rosetta = evaluation_cols[2].checkbox("Rosetta", value=True, key="design_campaign_evaluation_rosetta")
        evaluation_pymol = evaluation_cols[3].checkbox("PyMOL", value=True, key="design_campaign_evaluation_pymol")

        evaluation_refolder_key = "design_campaign_evaluation_refolder"
        if st.session_state.get(evaluation_refolder_key) not in refolder_options:
            st.session_state[evaluation_refolder_key] = default_evaluation_refolder

        if evaluation_refold:
            if target_refolding_restricted:
                st.warning(
                    "The selected target is cropped or contains residue-number gaps. Final refolding is restricted to "
                    "AF2-IG and Boltz-2 for this target representation."
                )
            evaluation_refolder = st.selectbox(
                "Refolding engine",
                refolder_options,
                index=refolder_options.index(st.session_state[evaluation_refolder_key]),
                key=evaluation_refolder_key,
            )
            evaluation_refolders = [evaluation_refolder] if evaluation_refolder in refolder_options else [default_evaluation_refolder]
            evaluation_refolder_settings = _screen_refolder_settings(
                "design_campaign_evaluation_refolder",
                str(evaluation_refolder),
                disabled=False,
                caption="These settings are used for the final refolding pass before ipSAE/Rosetta/PyMOL metrics.",
            )
            evaluation_recycles = int(evaluation_refolder_settings.get("num_recycles") or 10)
            evaluation_steps = int(evaluation_refolder_settings.get("num_sampling_steps") or 68)
            evaluation_samples = int(evaluation_refolder_settings.get("num_samples") or 3)
            evaluation_target_msa = bool(evaluation_refolder_settings.get("use_target_msa", evaluation_refolder != "AF2-IG"))
        else:
            evaluation_refolder = default_evaluation_refolder
            evaluation_refolders = []
            evaluation_refolder_settings = {}
            evaluation_recycles = 10
            evaluation_steps = 68
            evaluation_samples = 3
            evaluation_target_msa = False

        if evaluation_rosetta:
            evaluation_rosetta_nprocs = st.number_input(
                "PyRosetta processes",
                1,
                64,
                4,
                key="design_campaign_evaluation_rosetta_nprocs",
            )
        else:
            evaluation_rosetta_nprocs = 4

        if not any([evaluation_refold, evaluation_ipsae, evaluation_rosetta, evaluation_pymol]):
            st.info("No final evaluation will run; campaign outputs will still be available for downstream jobs.")
        elif evaluation_ipsae and not evaluation_refold:
            st.caption("ipSAE will use compatible PAE artifacts already present in the existing workflow outputs.")

        evaluation_mode = "refold_and_metrics" if evaluation_refold else (
            "metrics_only" if any([evaluation_ipsae, evaluation_rosetta, evaluation_pymol]) else "none"
        )

if generator_only_recipe:
    continue_after_failure = True
    common_validation_enabled = False
    staged_two_step = bool(workflow_recipe == "staged" and staged_two_step and staged_sequence_enabled)
    if staged_sequence_enabled:
        refinement_mode = "redesign_all"
        refinement_engine = str(staged_level1_sequence_engine or "ProteinMPNN")
        refinement_sequences = max(1, int(staged_level1_sequences))
        refinement_temperature = float(staged_level1_temperature)
        refinement_omit_aas = str(staged_level1_omit_aas or "C,X").strip().upper() or "C,X"
    else:
        refinement_mode = "none"
    for config in engine_configs.values():
        config["campaign_workflow_recipe"] = "staged_backbone_sequence_refold"
else:
    staged_two_step = False
    staged_run_level2 = False
    staged_level1_keep_per_attempt = 2
    staged_second_keep_total = 2
    staged_second_rmsd_cutoff = 3.5
    staged_second_sequences = 2
    staged_level1_omit_aas = "C,X"
    staged_second_omit_aas = "C,X"
    staged_screen_settings = {"num_recycles": 3}
    staged_second_screen_settings = {"num_recycles": 3}
    staged_screen_refolder = "AF2-IG"
    staged_second_screen_refolder = "AF2-IG"
    staged_rank_metric = "iptm"
    staged_second_rank_metric = "iptm"
    staged_second_engine = "Soluble ProteinMPNN"
    staged_second_temperature = 0.1
    staged_rmsd_cutoff = 3.5
    staged_include_first_stage_outputs = True
    staged_structure_filter_enabled = True
    staged_min_structure_elements = 3
evaluation_refolders = list(evaluation_refolders) if evaluation_mode == "refold_and_metrics" else []

with run_tab:
    execution_rows = []
    engine_scout_mode = workflow_recipe == "engine_scout"
    staged_mode = workflow_recipe == "staged" and bool(staged_sequence_enabled)

    def _staged_sequence_workload(engine: str, *, native_label: str | None = None) -> str:
        if not staged_mode:
            return ""
        level1_row = staged_sequence_by_engine.get(engine, {}) if isinstance(staged_sequence_by_engine, dict) else {}
        level2_row = (
            staged_second_sequence_by_engine.get(engine, {})
            if isinstance(staged_second_sequence_by_engine, dict)
            else {}
        )
        parts: list[str] = []
        if native_label and not bool(level1_row.get("enabled")):
            parts.append(native_label)
        elif bool(level1_row.get("enabled")):
            parts.append(
                f"Level 1 {staged_level1_sequence_engine} x{int(staged_level1_sequences)} "
                f"omit {staged_level1_omit_aas or 'none'}"
            )
        else:
            parts.append("passes Level 1 unchanged")
        if bool(staged_run_level2):
            if bool(level2_row.get("enabled", True)):
                parts.append(
                    f"Level 2 {staged_second_engine} x{int(staged_second_sequences)} "
                    f"omit {staged_second_omit_aas or 'none'}"
                )
            else:
                parts.append("passes Level 2 unchanged")
        if bool(staged_structure_filter_enabled):
            parts.append(f"prefilter SS elements >= {int(staged_min_structure_elements)}")
        parts.append(f"keep {int(staged_level1_keep_per_attempt)}/attempt after Level 1")
        return "; ".join(parts)

    if "rfdiffusion_classic" in selected_engines:
        rf_config = engine_configs.get("rfdiffusion_classic", {})
        rf_sequence_count = int(rf_config.get("mpnn_num_sequences", sequences_per_backbone))
        rf_keep_per_attempt = int(rf_config.get("analysis_keep_per_attempt") or 1)
        rf_rank_metric = str(rf_config.get("analysis_rank_metric") or "iptm")
        rf_max_rmsd = (rf_config.get("analysis_thresholds") or {}).get("max_binder_rmsd", 3.5)
        execution_rows.append(
            {
                "workflow": "RFdiffusion classic",
                "generation workload": f"{design_attempts} diffusion backbones",
                "sequence workload": (
                    "none; generator-only backbone handoff"
                    if engine_scout_mode
                    else _staged_sequence_workload("rfdiffusion_classic")
                    if staged_mode
                    else f"{rf_sequence_count} MPNN sequences per backbone"
                ),
                "native final limit": (
                    "pilot backbone candidates"
                    if engine_scout_mode
                    else (
                        "Boltz-2 monomer -> AF2-IG complex; "
                        f"RMSD <= {float(rf_max_rmsd):g} A; rank by {rf_rank_metric}; "
                        f"keep {rf_keep_per_attempt}/backbone"
                    )
                ),
            }
        )
    if "bindcraft" in selected_engines:
        execution_rows.append(
            {
                "workflow": "BindCraft",
                "generation workload": f"up to {design_attempts} trajectories",
                "sequence workload": (
                    "none; pre-MPNN trajectory/backbone handoff"
                    if engine_scout_mode
                    else _staged_sequence_workload("bindcraft")
                    if staged_mode
                    else f"{sequences_per_backbone} MPNN sequences per trajectory"
                ),
                "native final limit": (
                    "pilot trajectory candidates"
                    if engine_scout_mode
                    else (
                        f"{design_attempts} completed designs with native filters disabled"
                        if common_validation_enabled
                        else str(engine_configs["bindcraft"]["number_of_final_designs"])
                    )
                ),
            }
        )
    if "bindcraft2" in selected_engines:
        bc2_config = engine_configs.get("bindcraft2", {})
        execution_rows.append(
            {
                "workflow": "BindCraft 2",
                "generation workload": f"up to {int(bc2_config.get('max_trajectories', design_attempts))} trajectories",
                "sequence workload": "native ProteinMPNN redesign and AlphaFold validation",
                "native final limit": f"up to {int(bc2_config.get('number_of_final_designs', 1))} accepted designs",
            }
        )
    if "rfdiffusion3_foundry" in selected_engines:
        execution_rows.append(
            {
                "workflow": "RFdiffusion3 / Foundry",
                "generation workload": f"{design_attempts} diffusion designs",
                "sequence workload": (
                    "none; generated structure handoff"
                    if engine_scout_mode
                    else _staged_sequence_workload("rfdiffusion3_foundry")
                    if staged_mode
                    else f"{sequences_per_backbone} MPNN sequences per backbone"
                ),
                "native final limit": "pilot generated candidates" if engine_scout_mode else "native pipeline",
            }
        )
    if "boltzgen" in selected_engines:
        execution_rows.append(
            {
                "workflow": "BoltzGen",
                "generation workload": f"{design_attempts} generated designs",
                "sequence workload": (
                    "native sequence-design pipeline"
                    if engine_scout_mode
                    else _staged_sequence_workload("boltzgen")
                    if staged_mode
                    else "native sequence-design pipeline"
                ),
                "native final limit": (
                    str(design_attempts)
                    if common_validation_enabled
                    else str(engine_configs["boltzgen"]["budget"])
                ),
            }
        )
    if "pxdesign" in selected_engines:
        execution_rows.append(
            {
                "workflow": "PXDesign",
                "generation workload": f"{design_attempts} diffusion designs",
                "sequence workload": (
                    "native PXDesign pipeline"
                    if engine_scout_mode
                    else _staged_sequence_workload("pxdesign")
                    if staged_mode
                    else "native PXDesign pipeline"
                ),
                "native final limit": engine_configs["pxdesign"]["preset"],
            }
        )
    if "genie3" in selected_engines:
        execution_rows.append(
            {
                "workflow": "Genie3",
                "generation workload": f"{design_attempts} generated structures",
                "sequence workload": (
                    "none; generated structure handoff"
                    if engine_scout_mode
                    else _staged_sequence_workload("genie3")
                    if staged_mode
                    else f"{sequences_per_backbone} inverse-folded sequences per structure"
                ),
                "native final limit": "pilot generated candidates" if engine_scout_mode else "native folding evaluation",
            }
        )
    if "esmfold2_binder_design" in selected_engines:
        execution_rows.append(
            {
                "workflow": "ESMFold2 binder design",
                "generation workload": f"{design_attempts} optimization starts",
                "sequence workload": (
                    "one optimized sequence per start"
                    if engine_scout_mode
                    else _staged_sequence_workload(
                        "esmfold2_binder_design",
                        native_label=f"{engine_configs['esmfold2_binder_design'].get('generator_outputs_per_attempt', 1)} native sequence+structure design(s) per attempt",
                    )
                    if staged_mode
                    else "one optimized sequence per start"
                ),
                "native final limit": "all completed starts",
            }
        )
    if "protpardelle_1c" in selected_engines:
        execution_rows.append(
            {
                "workflow": "Protpardelle-1c",
                "generation workload": f"{design_attempts} scaffold samples",
                "sequence workload": (
                    "none; scaffold sample handoff"
                    if engine_scout_mode
                    else _staged_sequence_workload("protpardelle_1c")
                    if staged_mode
                    else f"{sequences_per_backbone} MPNN sequences per scaffold"
                ),
                "native final limit": "pilot scaffold candidates" if engine_scout_mode else "native ESMFold consistency filters",
            }
        )
    if "proteina_complexa" in selected_engines:
        execution_rows.append(
            {
                "workflow": "Proteina-Complexa",
                "generation workload": f"{design_attempts} requested designs",
                "sequence workload": (
                    "native Complexa pipeline"
                    if engine_scout_mode
                    else _staged_sequence_workload(
                        "proteina_complexa",
                        native_label=f"{engine_configs['proteina_complexa'].get('generator_outputs_per_attempt', 1)} native Complexa design(s) per attempt",
                    )
                    if staged_mode
                    else "native Complexa pipeline"
                ),
                "native final limit": f"best of {engine_configs['proteina_complexa']['replicas']} replicas",
            }
        )
    if workflow_recipe == "engine_scout":
        sequence_refinement_review = "Disabled for scout; normalized pilot candidates can be used later for sequence design"
    elif workflow_recipe == "staged" and staged_sequence_enabled:
        sequence_refinement_review = (
            f"Level 1 {refinement_engine} x{int(refinement_sequences)} omit {refinement_omit_aas or 'none'}; "
            f"keep {int(staged_level1_keep_per_attempt)}/attempt; "
            f"screen {staged_screen_refolder} ordered by {staged_rank_metric}; "
            + (
                f"Level 2 {staged_second_engine} x{int(staged_second_sequences)} omit {staged_second_omit_aas or 'none'}; "
                f"screen {staged_second_screen_refolder} ordered by {staged_second_rank_metric}; "
                f"{'include' if staged_include_first_stage_outputs else 'exclude'} level-1 kept designs"
                if staged_run_level2
                else "Level 2 disabled"
            )
        )
    elif refinement_mode != "none":
        sequence_refinement_review = (
            f"{refinement_mode}; {refinement_engine}; {int(refinement_sequences)} sequences; "
            f"T={float(refinement_temperature):g}; omit {refinement_omit_aas or 'none'}"
        )
    else:
        sequence_refinement_review = "Disabled"
    review_rows = [
        {"setting": "Target", "value": target_pdb.name},
        {"setting": "Chains", "value": ", ".join(selected_chains)},
        {
            "setting": "Workflow recipe",
            "value": {
                "staged": "Backbone -> sequence selection",
                "engine_scout": "Engine scout / pilot",
                "vanilla": "Vanilla multi-engine",
            }.get(str(workflow_recipe), "Vanilla multi-engine"),
        },
        {
            "setting": "Template redesign",
            "value": (
                f"{template_complex_pdb.name if template_complex_pdb else 'unreadable template'}; "
                f"target {','.join(template_target_chains) or '-'}; "
                f"binder {','.join(template_binder_chains) or '-'}; {template_lock_mode}"
                if template_enabled
                else "Disabled"
            ),
        },
        {"setting": "Binder length", "value": binder_length},
        {"setting": "Design attempts", "value": str(design_attempts)},
        {
            "setting": "Level-1 output keep per attempt",
            "value": (
                "Disabled for Engine scout / pilot"
                if workflow_recipe == "engine_scout"
                else str(staged_level1_keep_per_attempt)
                if workflow_recipe == "staged"
                else "n/a"
            ),
        },
        {
            "setting": "Sequences per backbone",
            "value": (
                str(sequences_per_backbone)
                if workflow_recipe == "vanilla"
                else "Configured per staged sequence level"
            ),
        },
        {"setting": "Random seed", "value": str(random_seed)},
        {"setting": "Workflows", "value": ", ".join(ENGINE_LABELS[engine] for engine in selected_engines)},
        {
            "setting": "Harmonized outputs",
            "value": (
                "All generator outputs enter sequence/refold screening"
                if workflow_recipe == "staged"
                else f"up to {int(scout_pilot_outputs_per_engine)} pilot outputs per engine"
                if workflow_recipe == "engine_scout"
                else f"up to {survivors_per_engine} per workflow"
            ),
        },
        {
            "setting": "Shared scoring",
            "value": (
                f"AF2 target-template, {common_recycles} recycles; ipSAE >= {common_min_ipsae:g}; "
                f"PyRosetta x{common_pyrosetta_nprocs}"
                if common_validation_enabled
                else (
                    "Skipped for staged workflow; RMSD gate is applied after internal refolding screens"
                    if generator_only_recipe
                    else "Disabled; harmonize outputs only"
                )
            ),
        },
        {
            "setting": "Sequence refinement",
            "value": sequence_refinement_review,
        },
        {
            "setting": "Evaluation",
            "value": (
                f"{evaluation_mode}; "
                + (
                    f"{', '.join(evaluation_refolders or [evaluation_refolder])}, {int(evaluation_recycles)} recycles, "
                    f"{int(evaluation_steps)} steps, {int(evaluation_samples)} samples; "
                    if evaluation_mode == "refold_and_metrics"
                    else ""
                )
                + ", ".join(
                    name
                    for name, enabled in [
                        ("ipSAE/interface", evaluation_ipsae),
                        ("Rosetta", evaluation_rosetta),
                        ("PyMOL", evaluation_pymol),
                    ]
                    if enabled
                )
                if evaluation_mode != "none"
                else "Disabled"
            ),
        },
    ]
    st.dataframe(pd.DataFrame(review_rows), hide_index=True, width="stretch")
    if execution_rows:
        st.dataframe(pd.DataFrame(execution_rows), hide_index=True, width="stretch")
    gpu_device = gpu_run_panel(key="design_campaign", default="0")
    template_ready = (
        bool(template_enabled)
        and template_complex_pdb is not None
        and template_complex_pdb.exists()
        and bool(template_target_chains)
        and bool(template_binder_chains)
    )
    launch_disabled = (
        not selected_chains
        or (not selected_engines and not template_ready)
        or not binder_length.strip()
        or (template_enabled and not template_ready)
        or ("bindcraft2" in selected_engines and not bindcraft2_campaign_ready)
    )
    if st.button("Run design campaign", type="primary", disabled=launch_disabled, width="stretch"):
        run_dir = create_design_campaign(
            target_pdb=target_pdb,
            target_chains=selected_chains,
            binder_length=binder_length,
            hotspots=hotspots,
            campaign_name=campaign_name,
            design_attempts=int(design_attempts),
            sequences_per_backbone=int(sequences_per_backbone),
            random_seed=int(random_seed),
            engines=selected_engines,
            engine_configs=engine_configs,
            workflow_recipe=workflow_recipe,
            survivors_per_engine=(
                int(scout_pilot_outputs_per_engine)
                if workflow_recipe == "engine_scout"
                else int(survivors_per_engine)
            ),
            passing_only=bool(passing_only),
            keep_best_failed=bool(keep_best_failed),
            continue_after_failure=bool(continue_after_failure),
            common_validation={
                "enabled": bool(common_validation_enabled),
                "profile": "bindcraft_default_target_template",
                "prediction_input_mode": "target_template_plus_binder_sequence",
                "num_recycles": int(common_recycles),
                "min_ipsae": float(common_min_ipsae),
                "pyrosetta_nprocs": int(common_pyrosetta_nprocs),
            },
            sequence_refinement={
                "mode": refinement_mode,
                "engine": refinement_engine,
                "sequences_per_structure": int(refinement_sequences),
                "sampling_temp": float(refinement_temperature),
                "omit_aas": refinement_omit_aas,
                "locked_residues": template_locked_residues,
                "unlocked_residues": template_unlocked_residues,
                "sequence_by_engine": staged_sequence_by_engine,
                "staged_two_step": {
                    "enabled": bool(staged_two_step),
                    "screen_refolder": staged_screen_refolder,
                    "second_screen_refolder": staged_second_screen_refolder,
                    "run_second_level": bool(staged_run_level2),
                    "first_keep_per_attempt": int(staged_level1_keep_per_attempt),
                    "second_keep_total": int(staged_second_keep_total),
                    "second_engine": staged_second_engine,
                    "second_sequences_per_structure": int(staged_second_sequences),
                    "second_sampling_temp": float(staged_second_temperature),
                    "second_omit_aas": str(staged_second_omit_aas or "C,X").strip().upper() or "C,X",
                    "second_sequence_by_engine": staged_second_sequence_by_engine,
                    "screen_settings": dict(staged_screen_settings),
                    "second_screen_settings": dict(staged_second_screen_settings),
                    "screen_num_recycles": int(staged_screen_settings.get("num_recycles") or 3),
                    "max_target_aligned_binder_rmsd": float(staged_rmsd_cutoff),
                    "second_max_target_aligned_binder_rmsd": float(staged_second_rmsd_cutoff),
                    "structure_element_filter_enabled": bool(staged_structure_filter_enabled),
                    "min_structure_elements": int(staged_min_structure_elements),
                    "include_first_stage_outputs": bool(staged_include_first_stage_outputs),
                    "rank_metric": staged_rank_metric,
                    "second_rank_metric": staged_second_rank_metric,
                },
            },
            template_redesign={
                "enabled": bool(template_enabled),
                "complex_pdb": str(template_complex_pdb) if template_complex_pdb else "",
                "target_chains": template_target_chains,
                "binder_chains": template_binder_chains,
                "mode": template_lock_mode,
                "locked_residues": template_locked_residues,
                "unlocked_residues": template_unlocked_residues,
            },
            evaluation={
                "mode": evaluation_mode,
                "ipsae": bool(evaluation_ipsae),
                "rosetta": bool(evaluation_rosetta),
                "pymol": bool(evaluation_pymol),
                "refolder": evaluation_refolder,
                "refolders": evaluation_refolders,
                "num_recycles": int(evaluation_recycles),
                "num_sampling_steps": int(evaluation_steps),
                "num_samples": int(evaluation_samples),
                "use_target_msa": bool(evaluation_target_msa),
                **dict(evaluation_refolder_settings),
                "pyrosetta_nprocs": int(evaluation_rosetta_nprocs),
            },
            gpu_device=gpu_device,
        )
        st.success(f"Campaign queued: {run_dir.name}")
        st.link_button("Open campaign result", result_link(DESIGN_CAMPAIGN_GROUP, run_dir.name))

with results_tab:
    refresh_results_button("refresh_design_campaigns")
    campaigns = _campaign_rows()
    if not campaigns:
        st.info("No design campaigns yet.")
    else:
        source_campaigns = [
            row
            for row in campaigns
            if str(row.get("job_type") or "") == "multi_engine_design_campaign"
            and (Path(str(row.get("run_dir") or "")) / "artifacts" / "design_campaign" / "common_validation").exists()
            and str(row.get("target_key") or "")
        ]
        if source_campaigns:
            with st.expander("Create validated campaign collection", expanded=True):
                target_keys = sorted({str(row["target_key"]) for row in source_campaigns})
                selected_target_key = st.selectbox(
                    "Target",
                    target_keys,
                    format_func=_target_key_label,
                    key="design_campaign_collection_target",
                )
                compatible = [row for row in source_campaigns if row["target_key"] == selected_target_key]
                label_by_run = {
                    row["run_id"]: f"{row['job']} | {row['campaign']} | {row['candidates']} candidates"
                    for row in compatible
                }
                selected_run_ids = st.multiselect(
                    "Source campaigns",
                    [row["run_id"] for row in compatible],
                    default=[row["run_id"] for row in compatible],
                    format_func=label_by_run.get,
                    key="design_campaign_collection_sources",
                )
                collection_name = st.text_input(
                    "Collection name",
                    value=f"Design campaign collection - {_target_key_label(selected_target_key)}",
                    key="design_campaign_collection_name",
                )
                if st.button(
                    "Create design campaign collection",
                    type="primary",
                    disabled=not bool(selected_run_ids),
                    key="create_design_campaign_collection",
                ):
                    try:
                        collection_run_dir = create_design_campaign_collection(
                            name=collection_name,
                            source_run_ids=selected_run_ids,
                            target_key=selected_target_key,
                        )
                    except Exception as exc:
                        st.error(str(exc))
                    else:
                        st.success("Design campaign collection created.")
                        st.link_button("Open collection result", result_link(DESIGN_CAMPAIGN_GROUP, collection_run_dir.name))
        collection_rows = [
            row
            for row in campaigns
            if str(row.get("job_type") or "") == "design_campaign_collection"
        ]
        if collection_rows:
            st.markdown("**Existing Collections**")
            collection_table = pd.DataFrame(collection_rows).drop(columns=["run_dir", "target_key"], errors="ignore")
            st.dataframe(
                collection_table[
                    [
                        col
                        for col in [
                            "result",
                            "job",
                            "campaign",
                            "status",
                            "candidates",
                            "completed engines",
                            "failed engines",
                            "updated",
                            "run_id",
                        ]
                        if col in collection_table.columns
                    ]
                ],
                hide_index=True,
                width="stretch",
                column_config={"result": st.column_config.LinkColumn("result", display_text="Open")},
            )
        table = pd.DataFrame(campaigns).drop(
            columns=["run_dir", "run_id", "target_key", "target_pdb", "target_chains", "job_type"],
            errors="ignore",
        )
        st.dataframe(
            table,
            hide_index=True,
            width="stretch",
            column_config={"result": st.column_config.LinkColumn("result", display_text="Open")},
        )
        st.caption("Open a campaign row to inspect workflow summaries, candidate tables, plots, structures, and artifacts.")
