from __future__ import annotations

import importlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from mn_protein_design.app.components.molstar_viewer import (
    ChainVisualization,
    StructureVisualization,
    molstar_custom_component,
)
from mn_protein_design.app.pages.common import cpu_run_panel, gpu_run_panel, result_link
from mn_protein_design.core.candidates import candidate_stage_counts, read_candidates
from mn_protein_design.core.detection_annotations import detection_score_sources, sidechain_sasa_by_residue
from mn_protein_design.core.jobs import read_json
from mn_protein_design.core.workflow_queue import queued_workflow_alias
from mn_protein_design.core.structures import filter_pdb_text
from mn_protein_design.workflows import design as design_workflow
from mn_protein_design.workflows import esm_binder as esm_binder_workflow
from mn_protein_design.workflows import bindcraft2 as bindcraft2_workflow

design_workflow = importlib.reload(design_workflow)
esm_binder_workflow = importlib.reload(esm_binder_workflow)
bindcraft2_workflow = importlib.reload(bindcraft2_workflow)
build_rfdiffusion_run_parameters = design_workflow.build_rfdiffusion_run_parameters
default_target_contig = design_workflow.default_target_contig
design_jobs_for_target = design_workflow.design_jobs_for_target
prepare_rfdiffusion_scaffold_library = design_workflow.prepare_rfdiffusion_scaffold_library
prepared_design_targets = design_workflow.prepared_design_targets
rfdiffusion_scaffold_library_status = design_workflow.rfdiffusion_scaffold_library_status


def _queued_design_workflow(function, tool: str, job_type: str = "design_campaign"):
    return queued_workflow_alias(function, task_group="design", tool=tool, job_type=job_type)


run_boltzgen = _queued_design_workflow(design_workflow.run_boltzgen, "boltzgen")
run_bindcraft = _queued_design_workflow(design_workflow.run_bindcraft, "bindcraft")
run_genie3 = _queued_design_workflow(design_workflow.run_genie3, "genie3")
run_proteina_complexa = _queued_design_workflow(design_workflow.run_proteina_complexa, "proteina_complexa")
run_protpardelle_1c = _queued_design_workflow(design_workflow.run_protpardelle_1c, "protpardelle_1c")
run_pxdesign = _queued_design_workflow(design_workflow.run_pxdesign, "pxdesign")
run_rfdiffusion3_foundry = _queued_design_workflow(design_workflow.run_rfdiffusion3_foundry, "rfdiffusion3_foundry")
run_rfdiffusion_classic = _queued_design_workflow(design_workflow.run_rfdiffusion_classic, "rfdiffusion_classic")
run_esmfold2_native_binder_design = _queued_design_workflow(
    esm_binder_workflow.run_esmfold2_native_binder_design,
    "esmfold2_binder_design",
)
run_esmfold2_binder_screening = _queued_design_workflow(
    esm_binder_workflow.run_esmfold2_binder_screening,
    "esmfold2_binder",
    "esmfold2_binder_screening",
)
target_label = design_workflow.target_label
normalize_rfdiffusion3_contig = design_workflow._normalize_rfdiffusion3_contig

ResidueId = tuple[str, int]


DETECTION_TOOL_STYLES = {
    "pesto": {"label": "PeSTo", "color": "0xd946ef"},
    "masif_seed": {"label": "MaSIF", "color": "0xf97316"},
    "scannet": {"label": "ScanNet", "color": "0x22c55e"},
    "surf2spot": {"label": "Surf2Spot", "color": "0x2563eb"},
}


RFDIFFUSION_GUIDANCE_PRESETS = {
    "none": {
        "label": "None",
        "parameters": "",
        "description": "No RFdiffusion guiding potential is added.",
    },
    "compact": {
        "label": "Compact binder",
        "parameters": 'potentials.guiding_potentials=["type:binder_ROG,weight:1"] potentials.guide_scale=1 potentials.guide_decay=quadratic',
        "description": "Light radius-of-gyration pressure to discourage extended binders.",
    },
    "compact_strong": {
        "label": "Compact binder strong",
        "parameters": 'potentials.guiding_potentials=["type:binder_ROG,weight:2"] potentials.guide_scale=2 potentials.guide_decay=quadratic',
        "description": "Stronger compactness pressure; useful against long isolated helices but more likely to over-constrain.",
    },
    "compact_interface": {
        "label": "Compact + interface contacts",
        "parameters": 'potentials.guiding_potentials=["type:binder_ROG,weight:1","type:interface_ncontacts,weight:1"] potentials.guide_scale=1 potentials.guide_decay=quadratic',
        "description": "Encourages both compact binders and target-interface contacts.",
    },
    "interface_contacts": {
        "label": "Interface contacts",
        "parameters": 'potentials.guiding_potentials=["type:interface_ncontacts,weight:1"] potentials.guide_scale=1 potentials.guide_decay=quadratic',
        "description": "Biases the binder toward making more target contacts without explicit compactness pressure.",
    },
}


def _pdb_residue_index_map(pdb_text: str) -> dict[str, dict[int, int]]:
    mapping: dict[str, dict[int, int]] = {}
    residues_by_chain: dict[str, list[int]] = {}
    last_seen_by_chain: dict[str, tuple[int, str] | None] = {}
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        chain = line[21].strip() or "_"
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        residue_key = (resseq, line[26].strip())
        residues_by_chain.setdefault(chain, [])
        last_seen_by_chain.setdefault(chain, None)
        if last_seen_by_chain[chain] != residue_key:
            residues_by_chain[chain].append(resseq)
            last_seen_by_chain[chain] = residue_key
    for chain, pdb_residue_numbers in residues_by_chain.items():
        mapping[chain] = {
            sequence_index: pdb_residue_number
            for sequence_index, pdb_residue_number in enumerate(pdb_residue_numbers, start=1)
        }
    return mapping


def _pdb_residue_number_set(pdb_text: str) -> dict[str, set[int]]:
    residues_by_chain: dict[str, set[int]] = {}
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        chain = line[21].strip() or "_"
        try:
            resseq = int(line[22:26])
        except ValueError:
            continue
        residues_by_chain.setdefault(chain, set()).add(resseq)
    return residues_by_chain


def _viewer_residues(
    value: object,
    residue_index_map: dict[str, dict[int, int]],
    residue_number_set: dict[str, set[int]],
) -> set[ResidueId]:
    residues: set[ResidueId] = set()
    if not value:
        return residues
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return residues
    if not isinstance(value, dict):
        return residues
    for selection in value.get("sequenceSelections") or []:
        if not isinstance(selection, dict):
            continue
        chain = str(selection.get("chainId") or "").strip()
        if not chain:
            continue
        chain_map = residue_index_map.get(chain, {})
        real_residue_numbers = residue_number_set.get(chain, set())
        for residue in selection.get("residues") or []:
            try:
                raw_residue = int(residue)
            except (TypeError, ValueError):
                continue
            pdb_residue = raw_residue if raw_residue in real_residue_numbers else chain_map.get(raw_residue, raw_residue)
            residues.add((chain, pdb_residue))
    return residues


def _hotspot_text(residues: set[ResidueId]) -> str:
    return ",".join(f"{chain}{residue}" for chain, residue in sorted(residues))


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


def _highlight_segments(residues: set[ResidueId]) -> list[str]:
    return [f"{chain}{residue}" for chain, residue in sorted(residues)]


def _chain_visualizations(residues: set[ResidueId]) -> list[ChainVisualization] | None:
    if not residues:
        return None
    by_chain: dict[str, list[int]] = {}
    for chain, residue in sorted(residues):
        by_chain.setdefault(chain, []).append(residue)
    return [
        ChainVisualization(
            chain_id=chain,
            residues=sorted(set(values)),
            color="uniform",
            color_params={"value": "0x2563eb"},
            representation_type="cartoon+ball-and-stick",
            label="Selected hotspots",
        )
        for chain, values in sorted(by_chain.items())
    ]


def _detection_tool_visualizations(scores: pd.DataFrame) -> list[ChainVisualization] | None:
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


def _target_chain_ids(target: dict) -> list[str]:
    chains = target.get("chains") or []
    if chains and isinstance(chains[0], dict):
        return [str(row["chain_id"]) for row in chains]
    return [str(chain) for chain in chains]


def _job_params(row: dict | None) -> dict:
    if not row:
        return {}
    payload = read_json(Path(row["run_dir"]) / "input.json")
    return payload.get("params") or {}


def _job_option(row: dict) -> str:
    return f"{row['job_code']} | {row['tool']} | {row['status']} | {row.get('candidate_count', '')} candidates"


def _hotspot_atom_map(hotspots: str) -> dict[str, str]:
    return {token: "CA" for token in hotspots.split(",") if token}


def _rfdiffusion3_contig(target_pdb: Path, target_chains: list[str], binder_length: str) -> str:
    return normalize_rfdiffusion3_contig(default_target_contig(target_pdb, target_chains, binder_length))


def _foundry_mpnn_model_options() -> dict[str, dict[str, str]]:
    return {
        "ligand_mpnn": {
            "label": "LigandMPNN",
            "checkpoint": "/weights/ligandmpnn_v_32_010_25.pt",
            "description": "LigandMPNN weights from the Foundry mount.",
        },
        "protein_mpnn": {
            "label": "ProteinMPNN",
            "checkpoint": "/weights/proteinmpnn_v_48_020.pt",
            "description": "ProteinMPNN weights from the Foundry mount.",
        },
        "soluble_mpnn": {
            "label": "Soluble ProteinMPNN",
            "checkpoint": "/ligandmpnn_weights/solublempnn_v_48_020.pt",
            "description": "Soluble ProteinMPNN weights from the LigandMPNN model-params mount.",
        },
    }


def _foundry_mpnn_model_from_checkpoint(checkpoint_path: str) -> str:
    checkpoint = checkpoint_path.lower()
    if "soluble" in checkpoint:
        return "soluble_mpnn"
    if "proteinmpnn" in checkpoint:
        return "protein_mpnn"
    return "protein_mpnn"


def _bindcraft_advanced_setting_options() -> dict[tuple[str, str], list[str]]:
    settings_dir = Path("/home/user/programs/ovo-git/ovo/resources/bindcraft/settings_advanced")
    options: dict[tuple[str, str], list[str]] = {}
    for path in sorted(settings_dir.glob("*.json")):
        stem = path.stem
        match = stem.split("_", 1)
        if len(match) != 2:
            continue
        family, rest = match
        for stage in ("3stage_multimer", "4stage_multimer"):
            if rest == stage:
                variant = "None"
            elif rest.startswith(f"{stage}_"):
                variant = rest.removeprefix(f"{stage}_")
            else:
                continue
            options.setdefault((family, stage), []).append(variant)
    order = ["None", "flexible", "flexible_hardtarget", "hardtarget", "mpnn", "mpnn_flexible", "mpnn_flexible_hardtarget", "mpnn_hardtarget"]
    for key, values in options.items():
        options[key] = sorted(set(values), key=lambda value: order.index(value) if value in order else len(order))
    return options


def _bindcraft_advanced_settings_file(family: str, stage: str, variant: str) -> str:
    suffix = "" if variant == "None" else f"_{variant}"
    return f"{family}_{stage}{suffix}.json"


def _bindcraft_preset_from_file(settings_file: str) -> tuple[str, str, str]:
    stem = Path(settings_file).stem
    family, rest = stem.split("_", 1) if "_" in stem else ("default", "4stage_multimer_mpnn")
    for stage in ("3stage_multimer", "4stage_multimer"):
        if rest == stage:
            return family, stage, "None"
        if rest.startswith(f"{stage}_"):
            return family, stage, rest.removeprefix(f"{stage}_")
    return "default", "4stage_multimer", "mpnn"


def _bindcraft_resource_json(folder: str, filename: str) -> dict:
    path = Path("/home/user/programs/ovo-git/ovo/resources/bindcraft") / folder / filename
    if not path.exists():
        return {"error": f"Missing BindCraft resource: {path}"}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return {"error": f"Could not parse {path}: {exc}"}


def _auto_contig(target_pdb: Path, target_chains: list[str], binder_length: str) -> str:
    if not target_chains:
        return ""
    return default_target_contig(target_pdb, target_chains, binder_length)


def _tool_payload_expander(tool_name: str, payload: dict) -> None:
    with st.expander(f"{tool_name} payload sent to algorithm", expanded=False):
        st.json(payload)


def _source_from_run_dir(run_dir: Path) -> dict:
    metadata = read_json(run_dir / "metadata.json")
    candidates = read_candidates(run_dir)
    return {
        "task_group": metadata.get("task_group"),
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "job_code": metadata.get("job_code"),
        "tool": metadata.get("tool"),
        "candidates_jsonl": str(run_dir / "artifacts" / "normalized_candidates" / "candidates.jsonl"),
        "candidate_count": len(candidates),
        "stage_counts": candidate_stage_counts(candidates),
    }


def _lineage_links(child_runs: list[Path]) -> None:
    analysis_csv: Path | None = None
    for child_run in child_runs:
        metadata = read_json(child_run / "metadata.json")
        task_group = str(metadata.get("task_group") or "")
        label = str(metadata.get("job_type") or metadata.get("tool") or child_run.name).replace("_", " ")
        if task_group:
            st.link_button(f"Open {label}", result_link(task_group, child_run.name))
        candidate_csv = child_run / "artifacts" / "analysis" / "ranked_candidates.csv"
        if candidate_csv.exists():
            analysis_csv = candidate_csv
    if analysis_csv:
        st.subheader("Analysis Results")
        df = pd.read_csv(analysis_csv)
        if not df.empty:
            metric_cols = st.columns(4)
            metric_cols[0].metric("Candidates", len(df))
            if "passes_filters" in df:
                metric_cols[1].metric("Passing", int(df["passes_filters"].fillna(False).sum()))
            if "analysis_score" in df:
                metric_cols[2].metric("Best score", f"{pd.to_numeric(df['analysis_score'], errors='coerce').max():.2f}")
            if "ipsae_error" in df:
                metric_cols[3].metric("IPSAE errors", int(df["ipsae_error"].fillna("").astype(bool).sum()))
            visible = [
                "analysis_rank",
                "candidate_id",
                "passes_filters",
                "analysis_score",
                "binder_plddt",
                "iptm",
                "ipae",
                "ipsae_min",
                "ipsae_max",
                "lis",
                "monomer_rmsd",
                "binder_rmsd",
                "hotspot_contact_fraction",
                "min_binder_to_hotspot_distance",
                "filter_failures",
            ]
            st.dataframe(df[[col for col in visible if col in df.columns]], hide_index=True, width="stretch")
            if "analysis_rank" in df and "analysis_score" in df:
                st.line_chart(df.set_index("analysis_rank")[["analysis_score"]])


st.title("Design")

targets = prepared_design_targets()
if not targets:
    st.info("Add, prepare, crop, or select an installed benchmark target before starting a design campaign.")
    st.stop()

labels = [target_label(row) for row in targets]
selected_label = st.selectbox("Target", labels)
target = targets[labels.index(selected_label)]
target_pdb = Path(target["target_pdb"])

existing_jobs = design_jobs_for_target(target_pdb)
if existing_jobs:
    st.caption("Existing design results for this target")
    rows = []
    for row in existing_jobs:
        rows.append(
            {
                "job_code": row["job_code"],
                "tool": row["tool"],
                "status": row["status"],
                "candidates": row.get("candidate_count", ""),
                "result": result_link(row["task_group"], row["run_id"], row["job_code"]),
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={"result": st.column_config.LinkColumn("result", display_text="open")},
    )

settings_source = None
if existing_jobs:
    source_options = ["Start fresh"] + [_job_option(row) for row in existing_jobs]
    selected_source = st.selectbox("Reuse settings from previous run", source_options)
    if selected_source != "Start fresh":
        settings_source = existing_jobs[source_options.index(selected_source) - 1]

loaded_params = _job_params(settings_source)
source_key = settings_source["run_id"] if settings_source else "fresh"

summary = ", ".join(
    f"{chain['chain_id']}:{chain['start']}-{chain['end']}" for chain in target.get("chains", []) if isinstance(chain, dict)
)
if not summary:
    summary = ",".join(target.get("chains") or [])
st.caption(f"Starting structure: {target_pdb.name} | chains {summary or 'unknown'}")
target_chains = _target_chain_ids(target)
st.caption(f"Using all chains in the selected prepared structure: `{','.join(target_chains) or 'unknown'}`")

target_text = target_pdb.read_text(errors="ignore")
target_view_text = filter_pdb_text(target_text, keep_chains=set(target_chains) or None) if target_chains else target_text
hotspot_state_key = f"design_hotspots_{target_pdb}"
if hotspot_state_key not in st.session_state:
    st.session_state[hotspot_state_key] = _hotspot_residues(str(loaded_params.get("hotspots") or ""))
elif settings_source and st.session_state.get(f"{hotspot_state_key}_source") != settings_source["run_id"]:
    st.session_state[hotspot_state_key] = _hotspot_residues(str(loaded_params.get("hotspots") or ""))
    st.session_state[f"{hotspot_state_key}_source"] = settings_source["run_id"]

st.subheader("Target And Hotspots")
st.caption("Click residues in the structure or sequence viewer, then add them as design hotspots.")
active_hotspots: set[ResidueId] = set(st.session_state[hotspot_state_key])
sources = detection_score_sources(target_pdb, target_chains)
detected_residues: set[ResidueId] = set()
show_existing_hotspots = False
detection_view_token = "none"
if sources:
    with st.expander("Detection-guided hotspots", expanded=True):
        source_names = st.multiselect("Detection results", list(sources), default=list(sources), key=f"design_detection_sources_{target_pdb}")
        threshold_cols = st.columns(2)
        prediction_threshold = threshold_cols[0].number_input("Prediction threshold", 0.0, 1.0, 0.50, 0.01, key=f"design_detection_threshold_{target_pdb}")
        masif_threshold = threshold_cols[1].number_input("MaSIF score threshold", 0.0, 1.0, 0.50, 0.01, key=f"design_masif_threshold_{target_pdb}")
        sasa_threshold = st.number_input("Minimum side-chain SASA (A2)", 0.0, 500.0, 5.0, 1.0, key=f"design_detection_sasa_{target_pdb}")
        show_existing_hotspots = st.checkbox("Also show existing hotspots", value=False, key=f"design_detection_existing_{target_pdb}")
        scores = pd.concat([sources[name].assign(source=name) for name in source_names], ignore_index=True) if source_names else pd.DataFrame(columns=["chain", "residue", "amino_acid", "score", "source"])
        scores["tool"] = scores["source"].str.split(" | ").str[0]
        scores["score_threshold"] = scores["tool"].map(lambda tool: masif_threshold if tool == "masif_seed" else prediction_threshold)
        sasa = sidechain_sasa_by_residue(target_pdb)
        scores["sidechain_sasa_a2"] = [sasa.get((row.chain, int(row.residue)), 0.0) for row in scores.itertuples()]
        scores["solvent_facing"] = scores["sidechain_sasa_a2"] >= sasa_threshold
        selected_scores = scores[(scores["score"] >= scores["score_threshold"]) & scores["solvent_facing"]].copy()
        agreement = selected_scores.groupby(["chain", "residue"], as_index=False).agg(
            detected_by_tools=("tool", lambda values: ", ".join(sorted(set(values)))),
            tool_count=("tool", "nunique"),
        )
        selected_scores = selected_scores.merge(agreement, on=["chain", "residue"], how="left")
        selected_scores["include"] = True
        edited_scores = st.data_editor(selected_scores, hide_index=True, width="stretch", key=f"design_detection_rows_{target_pdb}")
        include_mask = edited_scores["include"].fillna(False) if "include" in edited_scores else pd.Series(False, index=edited_scores.index)
        checked_scores = edited_scores[include_mask].copy()
        detected_residues = {(str(row.chain), int(row.residue)) for row in checked_scores.itertuples()}
        detection_view_token = f"{','.join(source_names)}:{prediction_threshold:.2f}:{masif_threshold:.2f}:{sasa_threshold:.1f}"
        if st.button("Add selected predicted residues as hotspots", key=f"design_detection_add_{target_pdb}"):
            st.session_state[hotspot_state_key] = active_hotspots | detected_residues
            st.rerun()
        _detection_tool_legend(set(checked_scores["tool"].dropna().astype(str)) if "tool" in checked_scores else set())
        st.caption(f"{len(detected_residues)} checked residues will be highlighted and added as hotspots.")
        detection_chains = _detection_tool_visualizations(checked_scores) or []
        if show_existing_hotspots:
            detection_chains.extend(_chain_visualizations(active_hotspots) or [])
        molstar_custom_component(
            [
                StructureVisualization(
                    pdb=target_view_text,
                    color="uniform",
                    color_params={"value": "0xe8b3ad"},
                    representation_type="cartoon",
                    chains=detection_chains or None,
                )
            ],
            key=f"design_detection_viewer_{target['run_id']}_{detection_view_token}",
            height=420,
            show_controls=True,
            selection_mode=False,
        )
viewer_value = molstar_custom_component(
    [
        StructureVisualization(
            pdb=target_view_text,
            color="uniform",
            color_params={"value": "0xe8b3ad"},
            representation_type="cartoon",
            highlighted_selections=_highlight_segments(active_hotspots),
            chains=_chain_visualizations(active_hotspots),
        )
    ],
    key=f"design_target_viewer_{target['run_id']}_{detection_view_token}",
    height=620,
    show_controls=True,
    selection_mode=True,
)
viewer_selected = _viewer_residues(
    viewer_value,
    _pdb_residue_index_map(target_view_text),
    _pdb_residue_number_set(target_view_text),
)

hotspot_cols = st.columns([1, 1, 1, 3])
with hotspot_cols[0]:
    if st.button("Add clicked hotspots", disabled=not viewer_selected):
        st.session_state[hotspot_state_key] = active_hotspots | viewer_selected
        st.rerun()
with hotspot_cols[1]:
    if st.button("Remove clicked hotspots", disabled=not viewer_selected):
        st.session_state[hotspot_state_key] = active_hotspots - viewer_selected
        st.rerun()
with hotspot_cols[2]:
    if st.button("Clear hotspots", disabled=not active_hotspots):
        st.session_state[hotspot_state_key] = set()
        st.rerun()
with hotspot_cols[3]:
    st.caption(f"Clicked now: `{_hotspot_text(viewer_selected) or 'none'}`")

hotspots_text = st.text_input(
    "Hotspots",
    value=_hotspot_text(active_hotspots),
    placeholder="A40,A99,A107",
    help="Editable fallback. Residues added from Mol* are written here.",
    key=f"hotspots_text_{source_key}",
)
if _hotspot_residues(hotspots_text) != active_hotspots:
    st.session_state[hotspot_state_key] = _hotspot_residues(hotspots_text)
    active_hotspots = set(st.session_state[hotspot_state_key])
hotspots = _hotspot_text(active_hotspots)

st.subheader("Campaign Setup")
campaign_name = st.text_input(
    "Campaign name",
    value=str(loaded_params.get("campaign_name") or ""),
    placeholder="for example PDL1 RFdiffusion 75aa soluble MPNN",
    key=f"campaign_name_{source_key}",
)
campaign_cols = st.columns(3)
with campaign_cols[0]:
    binder_length = st.text_input(
        "Binder length",
        value=str(loaded_params.get("binder_length") or "55-55"),
        help="Use a fixed length like 55 or a range like 55-80. Most generators accept this directly or through their contig/settings file.",
        key=f"binder_length_{source_key}",
    )
with campaign_cols[1]:
    num_candidates = st.number_input(
        "Design attempts",
        min_value=1,
        max_value=1000,
        value=int(loaded_params.get("num_designs") or loaded_params.get("number_of_final_designs") or loaded_params.get("max_trajectories") or 1),
        step=1,
        help="How many generator attempts to launch. For BindCraft this is the max trajectories setting.",
        key=f"num_candidates_{source_key}",
    )
with campaign_cols[2]:
    st.caption("GPU")
with st.expander("Compute", expanded=True):
    compute_cols = st.columns(2)
    with compute_cols[0]:
        design_gpu_device = gpu_run_panel(key=f"design_{source_key}", default=str(loaded_params.get("gpu_device") or "0"))
    with compute_cols[1]:
        design_cpu_cores = cpu_run_panel(key=f"design_{source_key}", default=4)

st.subheader("Generator")
rfdiffusion_tab, bindcraft_tab, bindcraft2_tab, foundry_tab, boltzgen_tab, pxdesign_tab, genie3_tab, esm_tab, protpardelle_tab, complexa_tab = st.tabs(
    [
        "RFdiffusion classic",
        "BindCraft",
        "BindCraft 2",
        "RFdiffusion3 / Foundry",
        "BoltzGen",
        "PXDesign",
        "Genie3",
        "ESM experimental",
        "Protpardelle-1c",
        "Proteina-Complexa",
    ]
)

with rfdiffusion_tab:
    st.caption(
        "App-owned RFdiffusion pipeline. Nextflow is integrated here as this repo's execution backend, "
        "not as an OVO runtime dependency."
    )
    execution_backend = st.radio(
        "Execution backend",
        ["docker", "nextflow"],
        format_func={"nextflow": "Nextflow", "docker": "Direct Docker"}.get,
        index=1 if str(loaded_params.get("execution_backend") or "docker") == "nextflow" else 0,
        horizontal=True,
        help="Nextflow runs the local mn_protein_design/pipelines/rfdiffusion_backbone pipeline.",
        key=f"rfdiffusion_execution_backend_{source_key}",
    )
    st.subheader("Run Mode")
    run_mode = st.segmented_control(
        "RFdiffusion workflow",
        ["generation_only", "full_vanilla_pipeline"],
        selection_mode="single",
        default="full_vanilla_pipeline",
        format_func={
            "generation_only": "Generation only",
            "full_vanilla_pipeline": "Full vanilla pipeline",
        }.get,
        key=f"rfdiffusion_run_mode_{source_key}",
    )
    full_pipeline = str(run_mode or "full_vanilla_pipeline") == "full_vanilla_pipeline"
    if full_pipeline:
        st.caption("Runs RFdiffusion, sequence design, monomer refolding, AF2 initial-guess complex refolding, then analysis.")
    full_mpnn_model = "protein_mpnn"
    monomer_tool = "boltz2_monomer"
    complex_tool = "af2_initial_guess"
    keep_top_n = 100
    min_binder_plddt = 70.0
    max_ipae = 10.0
    min_ipsae = 0.0
    max_binder_rmsd = 5.0
    min_final_hotspot_contact_fraction: float | None = None
    max_final_hotspot_distance: float | None = None
    complex_template_mode = "target_template"
    complex_multimer = True
    complex_num_recycles = 3

    st.markdown("#### 1. RFdiffusion Generation")
    contig_key = f"rfdiffusion_contig_{source_key}"
    auto_contig_key = f"{contig_key}_auto"
    computed_contig = _auto_contig(target_pdb, target_chains, binder_length)
    if loaded_params.get("contig") and contig_key not in st.session_state:
        st.session_state[contig_key] = str(loaded_params.get("contig"))
        st.session_state[auto_contig_key] = computed_contig
    elif contig_key not in st.session_state:
        st.session_state[contig_key] = computed_contig
        st.session_state[auto_contig_key] = computed_contig
    elif st.session_state.get(contig_key) == st.session_state.get(auto_contig_key):
        st.session_state[contig_key] = computed_contig
        st.session_state[auto_contig_key] = computed_contig
    contig = st.text_input(
        "RFdiffusion contig",
        help="Auto-built from the selected prepared target chains plus binder length. Manual edits are preserved.",
        key=contig_key,
    )
    rfdiffusion_num_designs = int(num_candidates)
    settings_cols = st.columns(2)
    with settings_cols[0]:
        model_options = ["Complex_base", "Complex_beta"]
        loaded_model = str(loaded_params.get("model_weights") or "Complex_base")
        model_weights = st.selectbox(
            "Model weights",
            model_options,
            index=model_options.index(loaded_model) if loaded_model in model_options else 0,
            key=f"model_weights_{source_key}",
        )
    with settings_cols[1]:
        timesteps = st.number_input(
            "Diffusion timesteps",
            min_value=1,
            max_value=200,
            value=int(loaded_params.get("timesteps") or 50),
            step=1,
            key=f"timesteps_{source_key}",
        )
    with st.expander("RFdiffusion advanced parameters", expanded=False):
        preset_key = f"rfdiffusion_guidance_preset_{source_key}"
        extra_key = f"extra_run_parameters_{source_key}"
        previous_preset_key = f"{preset_key}_previous"
        preset_changed_key = f"{preset_key}_changed"
        loaded_preset = str(loaded_params.get("rfdiffusion_guidance_preset") or "none")
        if loaded_preset not in RFDIFFUSION_GUIDANCE_PRESETS:
            loaded_preset = "none"
        if preset_key not in st.session_state:
            st.session_state[preset_key] = loaded_preset
            st.session_state[previous_preset_key] = loaded_preset
        selected_guidance_preset = st.selectbox(
            "Backbone guidance preset",
            list(RFDIFFUSION_GUIDANCE_PRESETS),
            index=list(RFDIFFUSION_GUIDANCE_PRESETS).index(st.session_state[preset_key]),
            format_func=lambda key: RFDIFFUSION_GUIDANCE_PRESETS[key]["label"],
            help="Fills the editable RFdiffusion parameter field below. The final text is what is actually sent to RFdiffusion.",
            key=preset_key,
        )
        if st.session_state.get(previous_preset_key) != selected_guidance_preset:
            st.session_state[extra_key] = RFDIFFUSION_GUIDANCE_PRESETS[selected_guidance_preset]["parameters"]
            st.session_state[previous_preset_key] = selected_guidance_preset
            st.session_state[preset_changed_key] = True
        elif extra_key not in st.session_state:
            st.session_state[extra_key] = str(
                loaded_params.get("extra_run_parameters")
                or RFDIFFUSION_GUIDANCE_PRESETS[selected_guidance_preset]["parameters"]
            )
        st.caption(RFDIFFUSION_GUIDANCE_PRESETS[selected_guidance_preset]["description"])
        extra_run_parameters = st.text_input(
            "Extra RFdiffusion parameters",
            placeholder="for example potentials.guiding_potentials=[]",
            help="Editable. This is appended to the generated RFdiffusion settings and stored with the job.",
            key=extra_key,
        )
        advanced_cols = st.columns(3)
        with advanced_cols[0]:
            partial_diffusion = st.checkbox(
                "Use partial diffusion",
                value=bool(loaded_params.get("partial_diffusion", False)),
                key=f"rfdiffusion_partial_diffusion_{source_key}",
            )
        with advanced_cols[1]:
            deterministic = st.checkbox(
                "Deterministic inference",
                value=bool(loaded_params.get("deterministic", False)),
                key=f"rfdiffusion_deterministic_{source_key}",
            )
        with advanced_cols[2]:
            save_trajectory = st.checkbox(
                "Save trajectory",
                value=bool(loaded_params.get("save_trajectory", False)),
                key=f"rfdiffusion_save_trajectory_{source_key}",
            )
        contigmap_length = st.text_input(
            "Total contig length limit",
            value=str(loaded_params.get("contigmap_length") or ""),
            placeholder="123 or 123-456",
            help="Forwarded to RFdiffusion as contigmap.length. Leave empty to let RFdiffusion sample from the contig.",
            key=f"rfdiffusion_contigmap_length_{source_key}",
        )
        inpaint_seq = st.text_input(
            "Sequence inpainting regions",
            value=str(loaded_params.get("inpaint_seq") or ""),
            placeholder="A10-20/A22/B30-40",
            help="Forwarded to RFdiffusion as contigmap.inpaint_seq. Useful when input-structure residues should be sequence-redesigned later.",
            key=f"rfdiffusion_inpaint_seq_{source_key}",
        )
        noise_cols = st.columns(2)
        with noise_cols[0]:
            noise_scale_ca = st.text_input(
                "CA noise scale",
                value=str(loaded_params.get("noise_scale_ca") or ""),
                placeholder="0",
                key=f"rfdiffusion_noise_scale_ca_{source_key}",
            )
        with noise_cols[1]:
            noise_scale_frame = st.text_input(
                "Frame noise scale",
                value=str(loaded_params.get("noise_scale_frame") or ""),
                placeholder="0",
                key=f"rfdiffusion_noise_scale_frame_{source_key}",
            )
        scaffold_status = rfdiffusion_scaffold_library_status()
        scaffold_ready = bool(scaffold_status.get("ready"))
        st.caption(
            "Scaffold-guided RFdiffusion can bias the binder backbone toward folds from a scaffold library "
            f"({scaffold_status.get('ss_count', 0)} SS files, {scaffold_status.get('adj_count', 0)} adjacency files)."
        )
        install_cols = st.columns([1, 3])
        with install_cols[0]:
            if st.button(
                "Install scaffold subset",
                disabled=scaffold_ready or not bool(scaffold_status.get("bundled_tar_exists")),
                key=f"rfdiffusion_install_scaffolds_{source_key}",
                help=f"Unpacks the bundled RFdiffusion ppi_scaffolds_subset.tar.gz into {scaffold_status.get('path')}.",
            ):
                try:
                    scaffold_status = prepare_rfdiffusion_scaffold_library()
                    scaffold_ready = bool(scaffold_status.get("ready"))
                    st.success(f"Installed RFdiffusion scaffold subset in {scaffold_status.get('path')}.")
                except Exception as exc:
                    st.error(f"Could not install RFdiffusion scaffold subset: {exc}")
        with install_cols[1]:
            if scaffold_ready:
                st.caption(f"Scaffold library ready: `{scaffold_status.get('path')}`")
            elif scaffold_status.get("bundled_tar_exists"):
                st.caption(f"Scaffold library not installed yet. Bundled archive: `{scaffold_status.get('bundled_tar')}`")
            else:
                st.caption(f"Bundled scaffold archive missing: `{scaffold_status.get('bundled_tar')}`")
        scaffoldguided = st.checkbox(
            "Use RFdiffusion scaffold-guided binder folds",
            value=bool(loaded_params.get("scaffoldguided", False)),
            disabled=not scaffold_ready,
            help=(
                "Adds scaffold-guided RFdiffusion settings. At launch, the app chooses one random scaffold "
                "from the installed library unless rfdiffusion_run_parameters.txt already contains scaffoldguided.scaffold_list."
            ),
            key=f"rfdiffusion_scaffoldguided_{source_key}",
        )
        if scaffoldguided:
            st.caption(
                "A single scaffold will be chosen randomly at launch and recorded in "
                "`rfdiffusion_scaffold_selection.json`."
            )
        scaffold_options = st.columns(3)
        with scaffold_options[0]:
            scaffold_target_pdb = st.checkbox(
                "Use target PDB in scaffold mode",
                value=bool(loaded_params.get("scaffold_target_pdb", True)),
                disabled=not scaffoldguided,
                key=f"rfdiffusion_scaffold_target_pdb_{source_key}",
            )
        with scaffold_options[1]:
            scaffold_target_ss = st.text_input(
                "Target SS .pt",
                value=str(loaded_params.get("scaffold_target_ss") or ""),
                placeholder="optional, e.g. /models/target_folds/target_ss.pt",
                disabled=not scaffoldguided,
                key=f"rfdiffusion_scaffold_target_ss_{source_key}",
            )
        with scaffold_options[2]:
            scaffold_target_adj = st.text_input(
                "Target adjacency .pt",
                value=str(loaded_params.get("scaffold_target_adj") or ""),
                placeholder="optional, e.g. /models/target_folds/target_adj.pt",
                disabled=not scaffoldguided,
                key=f"rfdiffusion_scaffold_target_adj_{source_key}",
            )
        backbone_filters = st.text_input(
            "Backbone hard filters",
            value=str(loaded_params.get("backbone_filters") or ""),
            placeholder="pydssp_helix_percent<50,N_contact_hotspots>=5",
            help="Stored in the app pipeline contract for the sequence/refolding steps that follow RFdiffusion.",
            key=f"rfdiffusion_backbone_filters_{source_key}",
        )
        apply_backbone_hotspot_prefilter = st.checkbox(
            "Use hotspot/site pre-filter before MPNN",
            value=bool(loaded_params.get("apply_backbone_hotspot_prefilter", False)),
            disabled=not bool(hotspots),
            help="All RFdiffusion backbones are still saved, but failed backbones are skipped by the downstream MPNN step.",
            key=f"rfdiffusion_apply_backbone_hotspot_prefilter_{source_key}",
        )
        prefilter_cols = st.columns(2)
        with prefilter_cols[0]:
            backbone_min_hotspot_contact_fraction = st.number_input(
                "Min hotspot contact fraction",
                min_value=0.0,
                max_value=1.0,
                value=float(loaded_params.get("backbone_min_hotspot_contact_fraction") or 0.25),
                step=0.05,
                disabled=not apply_backbone_hotspot_prefilter,
                key=f"rfdiffusion_backbone_min_hotspot_contact_fraction_{source_key}",
            )
        with prefilter_cols[1]:
            backbone_max_hotspot_distance = st.number_input(
                "Max nearest hotspot distance",
                min_value=0.0,
                max_value=50.0,
                value=float(loaded_params.get("backbone_max_hotspot_distance") or 10.0),
                step=0.5,
                disabled=not apply_backbone_hotspot_prefilter,
                key=f"rfdiffusion_backbone_max_hotspot_distance_{source_key}",
            )
    if not bool(hotspots):
        apply_backbone_hotspot_prefilter = False
        backbone_min_hotspot_contact_fraction = 0.25
        backbone_max_hotspot_distance = 10.0

    st.markdown("#### 2. Sequence Design")
    with st.expander("MPNN settings", expanded=True):
        sequence_design_options = {
            "ligandmpnn": "ProteinMPNN model (LigandMPNN implementation)",
            "fastrelax": "ProteinMPNN-FastRelax",
        }
        if full_pipeline:
            sequence_design_method = "ligandmpnn"
            full_mpnn_model = st.selectbox(
                "MPNN model",
                ["protein_mpnn", "soluble_mpnn", "ligand_mpnn"],
                format_func={
                    "protein_mpnn": "ProteinMPNN model (LigandMPNN implementation)",
                    "soluble_mpnn": "Soluble ProteinMPNN model (LigandMPNN implementation)",
                    "ligand_mpnn": "LigandMPNN model (LigandMPNN implementation)",
                }.get,
                key=f"rfdiffusion_full_mpnn_model_{source_key}",
            )
        else:
            loaded_sequence_method = str(loaded_params.get("sequence_design_method") or "ligandmpnn")
            sequence_design_method = st.radio(
                "Sequence design method",
                list(sequence_design_options),
                format_func=sequence_design_options.get,
                index=list(sequence_design_options).index(loaded_sequence_method)
                if loaded_sequence_method in sequence_design_options
                else 0,
                horizontal=True,
                key=f"rfdiffusion_sequence_design_method_{source_key}",
            )
        if sequence_design_method == "fastrelax":
            mpnn_fastrelax_cycles = st.number_input(
                "FastRelax cycles",
                min_value=1,
                max_value=5,
                value=int(loaded_params.get("mpnn_fastrelax_cycles") or 3),
                step=1,
                key=f"rfdiffusion_mpnn_fastrelax_cycles_{source_key}",
            )
        else:
            mpnn_fastrelax_cycles = 0
        mpnn_cols = st.columns(3)
        with mpnn_cols[0]:
            mpnn_num_sequences = st.number_input(
                "Sequences per backbone",
                min_value=1,
                max_value=1000,
                value=1 if sequence_design_method == "fastrelax" else int(loaded_params.get("mpnn_num_sequences") or 1),
                step=1,
                disabled=sequence_design_method == "fastrelax",
                key=f"rfdiffusion_mpnn_num_sequences_{source_key}",
            )
        with mpnn_cols[1]:
            mpnn_sampling_temp = st.number_input(
                "MPNN temperature",
                min_value=0.0,
                max_value=5.0,
                value=float(loaded_params.get("mpnn_sampling_temp") or 0.0001),
                step=0.0001,
                format="%g",
                key=f"rfdiffusion_mpnn_sampling_temp_{source_key}",
            )
        with mpnn_cols[2]:
            mpnn_omit_aa = st.text_input(
                "Omit amino acids",
                value=str(loaded_params.get("mpnn_omit_aa") or "CX"),
                key=f"rfdiffusion_mpnn_omit_aa_{source_key}",
            )
        mpnn_bias_aa = st.text_input(
            "Amino acid bias",
            value=str(loaded_params.get("mpnn_bias_aa") or ""),
            placeholder="F:1.5,K:-1.5",
            key=f"rfdiffusion_mpnn_bias_aa_{source_key}",
        )
        mpnn_run_parameters = st.text_input(
            "Extra MPNN parameters",
            value=str(loaded_params.get("mpnn_run_parameters") or ""),
            placeholder="--seed 42",
            key=f"rfdiffusion_mpnn_run_parameters_{source_key}",
        )
        refolding_test = ""

    st.markdown("#### 3. Refolding")
    if full_pipeline:
        refold_cols = st.columns(2)
        with refold_cols[0]:
            monomer_tool = st.selectbox(
                "Monomer refolding",
                ["boltz2_monomer", "esmfold", "af2_monomer"],
                format_func={
                    "boltz2_monomer": "Boltz2 monomer",
                    "esmfold": "ESMFold",
                    "af2_monomer": "AF2 monomer contract",
                }.get,
                key=f"rfdiffusion_full_monomer_tool_{source_key}",
            )
        with refold_cols[1]:
            complex_tool = st.selectbox(
                "Complex refolding",
                ["af2_initial_guess", "boltz2_initial_guess"],
                format_func={
                    "af2_initial_guess": "AF2 initial guess",
                    "boltz2_initial_guess": "Boltz2 initial guess",
                }.get,
                key=f"rfdiffusion_full_complex_tool_{source_key}",
            )
        if complex_tool == "af2_initial_guess":
            af2_refolding_options = {
                "af2_model_1_ptm_tt_3rec": {
                    "label": "AF2 monomer model, target template",
                    "template_mode": "target_template",
                    "multimer": False,
                    "num_recycles": 3,
                },
                "af2_model_1_ptm_tbt_3rec": {
                    "label": "AF2 monomer model, target + binder templates",
                    "template_mode": "target_binder_template",
                    "multimer": False,
                    "num_recycles": 3,
                },
                "af2_model_1_ptm_ct_3rec": {
                    "label": "AF2 monomer model, complex template",
                    "template_mode": "complex_template",
                    "multimer": False,
                    "num_recycles": 3,
                },
                "af2_model_1_multimer_tt_3rec": {
                    "label": "AF2 multimer, target template",
                    "template_mode": "target_template",
                    "multimer": True,
                    "num_recycles": 3,
                },
                "af2_model_1_multimer_tbt_3rec": {
                    "label": "AF2 multimer, target + binder templates",
                    "template_mode": "target_binder_template",
                    "multimer": True,
                    "num_recycles": 3,
                },
                "af2_model_1_multimer_ct_3rec": {
                    "label": "AF2 multimer, complex template",
                    "template_mode": "complex_template",
                    "multimer": True,
                    "num_recycles": 3,
                },
            }
            af2_refolding_test = st.selectbox(
                "AF2 complex refolding model",
                list(af2_refolding_options),
                format_func=lambda key: af2_refolding_options[key]["label"],
                index=list(af2_refolding_options).index("af2_model_1_multimer_tt_3rec"),
                key=f"rfdiffusion_full_af2_refolding_test_{source_key}",
            )
            complex_template_mode = str(af2_refolding_options[af2_refolding_test]["template_mode"])
            complex_multimer = bool(af2_refolding_options[af2_refolding_test]["multimer"])
            complex_num_recycles = int(af2_refolding_options[af2_refolding_test]["num_recycles"])
        else:
            boltz_template_mode = st.segmented_control(
                "Boltz2 complex template",
                ["target_template", "no_template"],
                selection_mode="single",
                default="target_template",
                format_func={
                    "target_template": "Target template",
                    "no_template": "No template",
                }.get,
                key=f"rfdiffusion_full_boltz_template_mode_{source_key}",
            )
            complex_template_mode = str(boltz_template_mode or "target_template")
            complex_multimer = True
            complex_num_recycles = 3
    else:
        st.caption("Generation-only mode stops after RFdiffusion. Refolding can be launched later from Campaigns.")

    st.markdown("#### 4. Filtering And Analysis")
    if full_pipeline:
        analysis_cols = st.columns(5)
        with analysis_cols[0]:
            keep_top_n = st.number_input(
                "Keep top results",
                min_value=1,
                max_value=10000,
                value=100,
                step=10,
                key=f"rfdiffusion_full_keep_top_n_{source_key}",
            )
        with analysis_cols[1]:
            min_binder_plddt = st.number_input(
                "Min binder pLDDT",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                step=1.0,
                key=f"rfdiffusion_full_min_binder_plddt_{source_key}",
            )
        with analysis_cols[2]:
            max_ipae = st.number_input(
                "Max iPAE",
                min_value=0.0,
                max_value=100.0,
                value=10.0,
                step=1.0,
                key=f"rfdiffusion_full_max_ipae_{source_key}",
            )
        with analysis_cols[3]:
            min_ipsae = st.number_input(
                "Min ipSAE",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                key=f"rfdiffusion_full_min_ipsae_{source_key}",
            )
        with analysis_cols[4]:
            max_binder_rmsd = st.number_input(
                "Max binder RMSD",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=0.5,
                key=f"rfdiffusion_full_max_binder_rmsd_{source_key}",
            )
        hotspot_filter_enabled = st.checkbox(
            "Filter final complexes by hotspot/site recovery",
            value=False,
            disabled=not bool(hotspots),
            key=f"rfdiffusion_full_hotspot_filter_enabled_{source_key}",
        )
        final_hotspot_cols = st.columns(2)
        with final_hotspot_cols[0]:
            min_final_hotspot_contact_fraction = st.number_input(
                "Final min hotspot contact fraction",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                disabled=not hotspot_filter_enabled,
                key=f"rfdiffusion_full_min_hotspot_contact_fraction_{source_key}",
            )
        with final_hotspot_cols[1]:
            max_final_hotspot_distance = st.number_input(
                "Final max nearest hotspot distance",
                min_value=0.0,
                max_value=50.0,
                value=8.0,
                step=0.5,
                disabled=not hotspot_filter_enabled,
                key=f"rfdiffusion_full_max_hotspot_distance_{source_key}",
            )
        if not hotspot_filter_enabled:
            min_final_hotspot_contact_fraction = None
            max_final_hotspot_distance = None
    else:
        st.caption("Analysis is available after downstream validation candidates exist.")

    try:
        generated_run_parameters = build_rfdiffusion_run_parameters(
            timesteps=int(timesteps),
            partial_diffusion=bool(partial_diffusion),
            contigmap_length=contigmap_length,
            inpaint_seq=inpaint_seq,
            model_weights=model_weights,
            deterministic=bool(deterministic),
            noise_scale_ca=noise_scale_ca,
            noise_scale_frame=noise_scale_frame,
            scaffoldguided=bool(scaffoldguided),
            scaffold_target_pdb=bool(scaffold_target_pdb),
            scaffold_target_ss=scaffold_target_ss,
            scaffold_target_adj=scaffold_target_adj,
            extra_run_parameters=extra_run_parameters,
        )
    except ValueError as exc:
        generated_run_parameters = str(loaded_params.get("rfdiffusion_run_parameters") or f"diffuser.T={int(timesteps)}")
        st.warning(str(exc))
    st.markdown("#### 5. Files, Payload, And Run")
    with st.expander("Editable files and final settings sent to RFdiffusion", expanded=True):
        st.caption("These editable copies are written into the run folder under artifacts before Docker starts.")
        editable_run_parameters_key = f"rfdiffusion_editable_run_parameters_{source_key}"
        scaffold_signature_key = f"rfdiffusion_scaffold_signature_{source_key}"
        scaffold_signature = (
            bool(scaffoldguided),
            bool(scaffold_target_pdb),
            scaffold_target_ss.strip(),
            scaffold_target_adj.strip(),
        )
        if st.session_state.pop(f"rfdiffusion_guidance_preset_{source_key}_changed", False):
            st.session_state[editable_run_parameters_key] = generated_run_parameters
        if st.session_state.get(scaffold_signature_key) != scaffold_signature:
            st.session_state[editable_run_parameters_key] = generated_run_parameters
            st.session_state[scaffold_signature_key] = scaffold_signature
        edited_run_parameters = st.text_area(
            "rfdiffusion_run_parameters.txt",
            value=str(loaded_params.get("rfdiffusion_run_parameters") or generated_run_parameters),
            height=110,
            key=editable_run_parameters_key,
        )
        edited_target_text = st.text_area(
            "target.pdb",
            value=target_view_text,
            height=220,
            key=f"rfdiffusion_editable_target_pdb_{source_key}",
        )
        pipeline_payload = {
            "pipeline": "mn-protein-design-rfdiffusion",
            "campaign_name": campaign_name.strip() if full_pipeline else "",
            "execution_backend": execution_backend,
            "design_type": "binder",
            "rfdiffusion_input_pdb": "/work/artifacts/input/target.pdb",
            "rfdiffusion_num_designs": int(rfdiffusion_num_designs),
            "rfdiffusion_contig": contig,
            "hotspot": hotspots,
            "rfdiffusion_guidance_preset": selected_guidance_preset,
            "rfdiffusion_run_parameters": edited_run_parameters,
            "scaffoldguided": bool(scaffoldguided),
            "scaffold_dir": design_workflow.RFDIFFUSION_SCAFFOLD_LIBRARY_CONTAINER_DIR,
            "scaffold_target_pdb": bool(scaffold_target_pdb),
            "scaffold_target_ss": scaffold_target_ss.strip(),
            "scaffold_target_adj": scaffold_target_adj.strip(),
            "backbone_filters": backbone_filters or "none",
            "backbone_hotspot_prefilter": {
                "enabled": bool(apply_backbone_hotspot_prefilter),
                "min_hotspot_contact_fraction": float(backbone_min_hotspot_contact_fraction),
                "max_min_binder_to_hotspot_distance": float(backbone_max_hotspot_distance),
                "contact_cutoff": 8.0,
            },
            "mpnn_num_sequences": int(mpnn_num_sequences),
            "mpnn_fastrelax_cycles": int(mpnn_fastrelax_cycles),
            "mpnn_run_parameters": (
                (
                    f'-omit_AAs "{mpnn_omit_aa}" -temperature {mpnn_sampling_temp}'
                    if sequence_design_method == "fastrelax"
                    else f'--omit_AA "{mpnn_omit_aa}" --temperature {mpnn_sampling_temp}'
                )
                + (
                    f' -bias_AA "{mpnn_bias_aa}"'
                    if sequence_design_method == "fastrelax" and mpnn_bias_aa
                    else f' --bias_AA "{mpnn_bias_aa}"'
                    if mpnn_bias_aa
                    else ""
                )
                + (f" {mpnn_run_parameters}" if mpnn_run_parameters else "")
            ).strip(),
            "refolding_tests": refolding_test or None,
            "planned_steps": [
                "rfdiffusion_backbone",
                "sequence_design" if full_pipeline else "manual_sequence_design",
                "monomer_refolding" if full_pipeline else "manual_monomer_refolding",
                "complex_refolding" if full_pipeline else "manual_complex_refolding",
                "analysis" if full_pipeline else "manual_analysis",
            ],
        }
        st.json(pipeline_payload)
    _tool_payload_expander(
        "RFdiffusion classic",
        {
            "docker_image": "mn-rfdiffusion:latest",
            "pipeline": "mn_protein_design/pipelines/rfdiffusion_backbone" if execution_backend == "nextflow" else "direct docker run",
            "input_pdb": "artifacts/input/target.pdb",
            "staged_files": {
                "target": "artifacts/input/target.pdb",
                "run_parameters": "artifacts/raw/rfdiffusion/pipeline_inputs/rfdiffusion_run_parameters.txt",
                "pipeline_payload": "artifacts/raw/rfdiffusion/pipeline_inputs/mn_rfdiffusion_pipeline_params.json",
            },
            "command_args": {
                "inference.output_prefix": "artifacts/raw/rfdiffusion/output/design",
                "inference.model_directory_path": "/models",
                "inference.schedule_directory_path": "/opt/RFdiffusion/schedules",
                "inference.input_pdb": "/work/artifacts/input/target.pdb",
                "inference.num_designs": int(rfdiffusion_num_designs),
                "contigmap.contigs": f"[{contig}]",
                "ppi.hotspot_res": f"[{hotspots}]" if hotspots else None,
                "inference.write_trajectory": bool(save_trajectory),
                "rfdiffusion_guidance_preset": selected_guidance_preset,
                "rfdiffusion_run_parameters": edited_run_parameters,
                "scaffoldguided": bool(scaffoldguided),
                "scaffold_dir": design_workflow.RFDIFFUSION_SCAFFOLD_LIBRARY_CONTAINER_DIR,
                "backbone_hotspot_prefilter": bool(apply_backbone_hotspot_prefilter),
            },
            "full_pipeline": {
                "enabled": full_pipeline,
                "sequence_design": {
                    "tool": "ligandmpnn",
                    "model_type": full_mpnn_model,
                    "num_seq_per_target": int(mpnn_num_sequences),
                    "sampling_temp": float(mpnn_sampling_temp),
                    "omit_aas": mpnn_omit_aa,
                    "require_backbone_hotspot_filter_pass": bool(apply_backbone_hotspot_prefilter),
                },
                "monomer_refolding": {"tool": monomer_tool, "min_plddt": float(min_binder_plddt)},
                "complex_refolding": {
                    "tool": complex_tool,
                    "template_mode": complex_template_mode,
                    "multimer": complex_multimer,
                    "num_recycles": complex_num_recycles,
                },
                "analysis": {
                    "keep_top_n": int(keep_top_n),
                    "thresholds": {
                        "min_binder_plddt": float(min_binder_plddt),
                        "min_confidence": 0.0,
                        "min_iptm": 0.0,
                        "min_ipsae": float(min_ipsae),
                        "max_ipae": float(max_ipae),
                        "max_ipde": 20.0,
                        "max_binder_rmsd": float(max_binder_rmsd),
                        **(
                            {
                                "min_hotspot_contact_fraction": float(min_final_hotspot_contact_fraction),
                                "max_hotspot_distance": float(max_final_hotspot_distance),
                            }
                            if min_final_hotspot_contact_fraction is not None
                            and max_final_hotspot_distance is not None
                            else {}
                        ),
                    },
                },
            },
        },
    )
    run_label = "Run RFdiffusion full pipeline" if full_pipeline else "Run RFdiffusion generation"
    if st.button(run_label, type="primary", disabled=not target_chains):
        try:
            with st.spinner("Queueing RFdiffusion generation..."):
                run_dir = run_rfdiffusion_classic(
                    target_pdb=target_pdb,
                    target_chains=target_chains,
                    contig=contig,
                    binder_length=binder_length,
                    hotspots=hotspots,
                    campaign_name=campaign_name,
                    num_designs=int(rfdiffusion_num_designs),
                    timesteps=int(timesteps),
                    model_weights=model_weights,
                    rfdiffusion_guidance_preset=selected_guidance_preset,
                    extra_run_parameters=extra_run_parameters,
                    partial_diffusion=bool(partial_diffusion),
                    contigmap_length=contigmap_length,
                    inpaint_seq=inpaint_seq,
                    backbone_filters=backbone_filters,
                    apply_backbone_hotspot_prefilter=bool(apply_backbone_hotspot_prefilter),
                    backbone_min_hotspot_contact_fraction=float(backbone_min_hotspot_contact_fraction),
                    backbone_max_hotspot_distance=float(backbone_max_hotspot_distance),
                    deterministic=bool(deterministic),
                    noise_scale_ca=noise_scale_ca,
                    noise_scale_frame=noise_scale_frame,
                    scaffoldguided=bool(scaffoldguided),
                    scaffold_dir=design_workflow.RFDIFFUSION_SCAFFOLD_LIBRARY_CONTAINER_DIR,
                    scaffold_target_pdb=bool(scaffold_target_pdb),
                    scaffold_target_ss=scaffold_target_ss,
                    scaffold_target_adj=scaffold_target_adj,
                    save_trajectory=bool(save_trajectory),
                    editable_run_parameters=edited_run_parameters,
                    edited_target_pdb_text=edited_target_text,
                    mpnn_num_sequences=int(mpnn_num_sequences),
                    mpnn_sampling_temp=float(mpnn_sampling_temp),
                    mpnn_omit_aa=mpnn_omit_aa,
                    mpnn_bias_aa=mpnn_bias_aa,
                    mpnn_run_parameters=mpnn_run_parameters,
                    sequence_design_method=full_mpnn_model if full_pipeline else sequence_design_method,
                    mpnn_fastrelax_cycles=int(mpnn_fastrelax_cycles),
                    refolding_test=refolding_test,
                    execution_backend=execution_backend,
                    run_vanilla_pipeline=bool(full_pipeline),
                    monomer_refolding_tool=monomer_tool,
                    complex_refolding_tool=complex_tool,
                    complex_template_mode=complex_template_mode,
                    complex_multimer=bool(complex_multimer),
                    complex_num_recycles=int(complex_num_recycles),
                    analysis_keep_top_n=int(keep_top_n),
                    analysis_thresholds={
                        "min_binder_plddt": float(min_binder_plddt),
                        "min_confidence": 0.0,
                        "min_iptm": 0.0,
                        "min_ipsae": float(min_ipsae),
                        "max_ipae": float(max_ipae),
                        "max_ipde": 20.0,
                        "max_binder_rmsd": float(max_binder_rmsd),
                        **(
                            {
                                "min_hotspot_contact_fraction": float(min_final_hotspot_contact_fraction),
                                "max_hotspot_distance": float(max_final_hotspot_distance),
                            }
                            if min_final_hotspot_contact_fraction is not None
                            and max_final_hotspot_distance is not None
                            else {}
                        ),
                    },
                    gpu_device=design_gpu_device,
                    queue_cpu_cores=design_cpu_cores,
                )
            st.success("RFdiffusion job queued. It will continue if you close Streamlit.")
            st.link_button("Open result", result_link("design", run_dir.name))
        except Exception as exc:
            st.error(str(exc))

with bindcraft_tab:
    bindcraft_campaign_name = campaign_name
    st.caption("Uses the campaign name, binder length, design attempts, target chains, and hotspots from Campaign Setup.")
    st.markdown("#### Output Goal")
    settings_cols = st.columns(2)
    with settings_cols[0]:
        bindcraft_final_designs = st.number_input(
            "Desired final designs",
            min_value=1,
            max_value=100,
            value=int(loaded_params.get("number_of_final_designs") or 1),
            step=1,
            help="BindCraft stops when this many accepted designs are found or when the time limit is reached.",
            key=f"bindcraft_final_designs_{source_key}",
        )
    with settings_cols[1]:
        bindcraft_max_trajectories = int(num_candidates)
        st.metric("Design attempts", bindcraft_max_trajectories)

    st.markdown("#### Runtime")
    runtime_cols = st.columns(2)
    with runtime_cols[0]:
        loaded_time_limit_seconds = loaded_params.get("time_limit_seconds")
        bindcraft_use_time_limit = st.checkbox(
            "Set time limit",
            value=loaded_time_limit_seconds is not None,
            key=f"bindcraft_use_time_limit_{source_key}",
        )
    with runtime_cols[1]:
        bindcraft_time_limit_hours = st.number_input(
            "Time limit hours",
            min_value=0.1,
            max_value=14 * 24.0,
            value=float(loaded_time_limit_seconds or 0) / 3600.0 if loaded_time_limit_seconds else 24.0,
            step=0.5,
            disabled=not bindcraft_use_time_limit,
            key=f"bindcraft_time_limit_hours_{source_key}",
        )
        bindcraft_time_limit_seconds = int(float(bindcraft_time_limit_hours) * 3600) if bindcraft_use_time_limit else None

    st.markdown("#### BindCraft Presets")
    preset_options = _bindcraft_advanced_setting_options()
    loaded_advanced_file = str(loaded_params.get("advanced_settings_file") or "default_4stage_multimer_mpnn.json")
    loaded_family, loaded_stage, loaded_variant = _bindcraft_preset_from_file(loaded_advanced_file)
    family_labels = {"default": "default", "betasheet": "beta", "peptide": "peptide"}
    family_values = ["default", "betasheet", "peptide"]
    preset_cols = st.columns(3)
    with preset_cols[0]:
        bindcraft_family = st.selectbox(
            "Advanced preset family",
            family_values,
            index=family_values.index(loaded_family) if loaded_family in family_values else 0,
            format_func=lambda value: family_labels.get(value, value),
            key=f"bindcraft_family_{source_key}",
        )
    available_stages = [stage for stage in ["4stage_multimer", "3stage_multimer"] if (bindcraft_family, stage) in preset_options]
    if not available_stages:
        available_stages = ["4stage_multimer"]
    with preset_cols[1]:
        bindcraft_stage = st.selectbox(
            "Advanced preset stage",
            available_stages,
            index=available_stages.index(loaded_stage) if loaded_stage in available_stages else 0,
            format_func=lambda value: value.replace("stage_", " stage "),
            key=f"bindcraft_stage_{source_key}",
        )
    available_variants = preset_options.get((bindcraft_family, bindcraft_stage), ["mpnn"])
    default_variant = loaded_variant if loaded_variant in available_variants else ("mpnn" if "mpnn" in available_variants else available_variants[0])
    with preset_cols[2]:
        bindcraft_variant = st.selectbox(
            "Advanced preset mode",
            available_variants,
            index=available_variants.index(default_variant),
            key=f"bindcraft_variant_{source_key}",
        )
    bindcraft_advanced_settings_file = _bindcraft_advanced_settings_file(bindcraft_family, bindcraft_stage, bindcraft_variant)
    bindcraft_advanced_settings = _bindcraft_resource_json("settings_advanced", bindcraft_advanced_settings_file)

    st.markdown("#### Sequence Redesign")
    preset_num_seqs = int(bindcraft_advanced_settings.get("num_seqs") or 1)
    preset_max_mpnn_sequences = int(bindcraft_advanced_settings.get("max_mpnn_sequences") or preset_num_seqs)
    loaded_num_seqs_override = loaded_params.get("num_seqs_override", loaded_params.get("num_sequences"))
    loaded_max_mpnn_override = loaded_params.get("max_mpnn_sequences_override", loaded_params.get("num_sequences"))
    override_loaded = loaded_num_seqs_override is not None or loaded_max_mpnn_override is not None
    bindcraft_override_sequence_settings = st.checkbox(
        "Override preset sequence settings",
        value=override_loaded,
        help="When off, BindCraft uses num_seqs and max_mpnn_sequences directly from the selected advanced JSON.",
        key=f"bindcraft_override_sequence_settings_{source_key}",
    )
    sequence_cols = st.columns(2)
    with sequence_cols[0]:
        bindcraft_num_seqs = st.number_input(
            "num_seqs",
            min_value=1,
            max_value=100,
            value=int(loaded_num_seqs_override or preset_num_seqs),
            step=1,
            disabled=not bindcraft_override_sequence_settings,
            help="Advanced JSON field: number of MPNN sequences generated per trajectory.",
            key=f"bindcraft_num_seqs_{source_key}",
        )
    with sequence_cols[1]:
        bindcraft_max_mpnn_sequences = st.number_input(
            "max_mpnn_sequences",
            min_value=1,
            max_value=100,
            value=int(loaded_max_mpnn_override or preset_max_mpnn_sequences),
            step=1,
            disabled=not bindcraft_override_sequence_settings,
            help="Advanced JSON field: maximum MPNN designs tested/refolded per trajectory.",
            key=f"bindcraft_max_mpnn_sequences_{source_key}",
        )
    if not bindcraft_override_sequence_settings:
        bindcraft_num_seqs = preset_num_seqs
        bindcraft_max_mpnn_sequences = preset_max_mpnn_sequences

    st.markdown("#### Filters")
    filter_options = [
        "default_filters.json",
        "relaxed_filters.json",
        "peptide_filters.json",
        "peptide_relaxed_filters.json",
    ]
    loaded_filter = str(loaded_params.get("filter_settings") or "default_filters.json")
    if loaded_filter == "no_filters.json":
        loaded_filter = "default_filters.json"
    bindcraft_filter_settings = st.selectbox(
        "Filter settings",
        filter_options,
        index=filter_options.index(loaded_filter) if loaded_filter in filter_options else 0,
        key=f"bindcraft_filter_settings_{source_key}",
    )
    inspect_cols = st.columns(2)
    with inspect_cols[0]:
        with st.expander(f"Inspect advanced preset JSON: {bindcraft_advanced_settings_file}", expanded=False):
            st.json(bindcraft_advanced_settings)
    with inspect_cols[1]:
        with st.expander(f"Inspect filter JSON: {bindcraft_filter_settings}", expanded=False):
            st.json(_bindcraft_resource_json("settings_filters", bindcraft_filter_settings))
    _tool_payload_expander(
        "BindCraft",
        {
            "docker_image": "mn-bindcraft:latest",
            "input_json": {
                "design_path": "output",
                "starting_pdb": "target.pdb",
                "binder_name": bindcraft_campaign_name.strip() or "design",
                "chains": ",".join(target_chains),
                "target_hotspot_residues": hotspots,
                "lengths": binder_length,
                "number_of_final_designs": int(bindcraft_final_designs),
            },
            "advanced_settings_file": bindcraft_advanced_settings_file,
            "advanced_settings_effective": {
                "num_seqs": int(bindcraft_num_seqs),
                "max_mpnn_sequences": int(bindcraft_max_mpnn_sequences),
                "max_trajectories": int(bindcraft_max_trajectories),
                "af_params_dir": "alphafold_models_path",
            },
            "advanced_settings_sequence_overrides_enabled": bool(bindcraft_override_sequence_settings),
            "filter_settings": bindcraft_filter_settings,
            "runtime": {
                "time_limit_hours": float(bindcraft_time_limit_hours) if bindcraft_use_time_limit else None,
                "time_limit_seconds": bindcraft_time_limit_seconds,
            },
        },
    )
    if st.button("Run BindCraft campaign", type="primary", disabled=not target_chains):
        try:
            with st.spinner("Queueing BindCraft..."):
                run_dir = run_bindcraft(
                    target_pdb=target_pdb,
                    target_chains=target_chains,
                    binder_length=binder_length,
                    hotspots=hotspots,
                    campaign_name=bindcraft_campaign_name,
                    number_of_final_designs=int(bindcraft_final_designs),
                    num_seqs_override=int(bindcraft_num_seqs) if bindcraft_override_sequence_settings else None,
                    max_mpnn_sequences_override=int(bindcraft_max_mpnn_sequences)
                    if bindcraft_override_sequence_settings
                    else None,
                    time_limit_seconds=bindcraft_time_limit_seconds,
                    enable_mpnn=True,
                    max_trajectories=int(bindcraft_max_trajectories),
                    filter_settings=bindcraft_filter_settings,
                    advanced_settings_file=bindcraft_advanced_settings_file,
                    gpu_device=design_gpu_device,
                    queue_cpu_cores=design_cpu_cores,
                )
            st.success("BindCraft job queued. It will continue if you close Streamlit.")
            st.link_button("Open result", result_link("design", run_dir.name))
        except Exception as exc:
            st.error(str(exc))

with bindcraft2_tab:
    st.caption(
        "Runs the full BindCraft 2 campaign: gradient design, ProteinMPNN redesign, AlphaFold validation, "
        "native filters, and ranking. This is submitted as a background GPU job."
    )
    st.info("BindCraft 2 requires an NVIDIA GPU and cannot run in CPU mode.")

    image_ready = bindcraft2_workflow.bindcraft2_image_available()
    if image_ready:
        st.success(f"Container available: {bindcraft2_workflow.BINDCRAFT2_IMAGE}")
    else:
        st.warning(f"Container is not built: {bindcraft2_workflow.BINDCRAFT2_IMAGE}")
        st.code("bash containers/bindcraft2/build.sh", language="bash")

    reference_dir = bindcraft2_workflow.bindcraft2_reference_directory()
    missing_parameters = bindcraft2_workflow.missing_bindcraft2_parameters()
    if missing_parameters:
        st.error(
            f"BindCraft 2 needs seven AlphaFold checkpoints under {reference_dir}. "
            f"Missing or incomplete: {', '.join(missing_parameters)}"
        )
    else:
        st.caption(f"AlphaFold parameters: {reference_dir} (mounted read-only into the container)")

    bc2_cols = st.columns(3)
    with bc2_cols[0]:
        bc2_binder_format = st.selectbox(
            "BC2 binder format",
            bindcraft2_workflow.BC2_BINDER_FORMATS,
            index=0,
            key=f"bindcraft2_modality_{source_key}",
            help="Choose one of BindCraft 2's native binder-format presets. These are different from BC1 settings presets.",
        )
    with bc2_cols[1]:
        bc2_final_designs = st.number_input(
            "Desired accepted designs",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
            key=f"bindcraft2_final_designs_{source_key}",
        )
    with bc2_cols[2]:
        bc2_attempts = st.number_input(
            "Maximum trajectories",
            min_value=1,
            max_value=1000,
            value=int(num_candidates),
            step=1,
            help="BC2 may stop earlier if it accepts the requested number of designs.",
            key=f"bindcraft2_attempts_{source_key}",
        )

    bc2_objective_options = [None, *bindcraft2_workflow.BC2_CONFORMATIONAL_OBJECTIVES] if bc2_binder_format == "binder" else [None]
    bc2_objective = st.selectbox(
        "BC2 conformational objective",
        bc2_objective_options,
        format_func=lambda value: "None" if value is None else value,
        key=f"bindcraft2_objective_{source_key}",
        help="induced_fit and fold_switch are optional objectives available with the de novo binder format.",
    )
    bc2_modality = [bc2_binder_format] + ([bc2_objective] if bc2_objective else [])
    bc2_binder_length = st.text_input(
        "BC2 binder-length override (optional)",
        value="",
        key=f"bindcraft2_binder_length_{source_key}_{bc2_binder_format}",
        help="Leave blank to use the selected BC2 modality preset's length choices. Enter one length or a min-max range to override.",
    ).strip()
    preset_lengths = bindcraft2_workflow.BC2_PRESET_LENGTHS.get(bc2_binder_format)
    st.caption(
        "BC2 preset lengths: "
        + (f"{preset_lengths[0]}–{preset_lengths[1]} residues" if preset_lengths else "defined by the selected scaffold")
        + ". The shared length setting for other design engines does not override this BC2 preset."
    )
    bc2_properties = st.multiselect(
        "BC2 design properties",
        list(bindcraft2_workflow.BC2_PROPERTY_PRESETS),
        format_func=lambda value: value.replace("_", " "),
        key=f"bindcraft2_properties_{source_key}",
        help="Optional native BC2 properties. forced_targeting requires target hotspots; some combinations are incompatible.",
    )
    st.caption("BC2 loads its own native modality and property presets. The selected target hotspots above are applied here.")

    resource_cols = st.columns(3)
    bc2_cpu_options, bc2_default_cpu_cores = bindcraft2_workflow.bindcraft2_cpu_options()
    with resource_cols[0]:
        bc2_workers = st.selectbox(
            "Workers per GPU",
            bindcraft2_workflow.WORKERS_PER_GPU_OPTIONS,
            index=0,
            format_func=lambda value: "BC2 automatic" if value == "auto" else str(value),
            help="The queue reserves the selected GPU for this job. Automatic uses BC2's memory-aware worker packing.",
            key=f"bindcraft2_workers_per_gpu_{source_key}",
        )
    with resource_cols[1]:
        bc2_cpu_cores = st.selectbox(
            "Reserved CPU cores",
            bc2_cpu_options,
            index=bc2_cpu_options.index(bc2_default_cpu_cores),
            help="The local job scheduler reserves these CPU slots and applies the same limit to Docker.",
            key=f"bindcraft2_cpu_cores_{source_key}",
        )
    with resource_cols[2]:
        bc2_seed = st.number_input(
            "Campaign seed",
            min_value=0,
            max_value=2_147_483_647,
            value=0,
            step=1,
            key=f"bindcraft2_seed_{source_key}",
        )

    with st.expander("Review BindCraft 2 campaign settings", expanded=False):
        try:
            st.json(
                bindcraft2_workflow.build_bindcraft2_settings(
                    target_chains=target_chains,
                    binder_length=bc2_binder_length or None,
                    hotspots=hotspots,
                    campaign_name=campaign_name,
                    modality=bc2_modality,
                    design_properties=bc2_properties,
                    number_of_final_designs=int(bc2_final_designs),
                    max_trajectories=int(bc2_attempts),
                    campaign_seed=int(bc2_seed),
                    workers_per_gpu=bc2_workers,
                )
            )
        except ValueError as exc:
            st.error(str(exc))

    bc2_ready = bool(target_chains) and image_ready and not missing_parameters
    if st.button(
        "Queue BindCraft 2 campaign",
        type="primary",
        disabled=not bc2_ready,
        key=f"run_bindcraft2_{source_key}",
    ):
        try:
            run_dir = bindcraft2_workflow.enqueue_bindcraft2_design(
                target_pdb=target_pdb,
                target_chains=target_chains,
                binder_length=bc2_binder_length or None,
                hotspots=hotspots,
                campaign_name=campaign_name,
                modality=bc2_modality,
                design_properties=bc2_properties,
                number_of_final_designs=int(bc2_final_designs),
                max_trajectories=int(bc2_attempts),
                campaign_seed=int(bc2_seed),
                workers_per_gpu=bc2_workers,
                cpu_cores=int(bc2_cpu_cores),
                gpu_device=design_gpu_device,
            )
            st.success("BindCraft 2 campaign queued. It will continue if you close Streamlit.")
            st.link_button("Open job", result_link("design", run_dir.name))
        except Exception as exc:
            st.error(str(exc))

with foundry_tab:
    st.caption("RFdiffusion3 / Foundry runs the native Foundry chain: RFD3 backbone generation, Foundry MPNN redesign, then RF3 refolding with cached target MSAs.")
    st.subheader("Run Mode")
    foundry_run_mode = st.segmented_control(
        "RFdiffusion3 / Foundry workflow",
        ["vanilla_pipeline", "generation_only"],
        selection_mode="single",
        default="vanilla_pipeline",
        format_func={
            "vanilla_pipeline": "Vanilla Foundry pipeline",
            "generation_only": "Generation only",
        }.get,
        key=f"foundry_run_mode_{source_key}",
    )
    foundry_vanilla_pipeline = str(foundry_run_mode or "vanilla_pipeline") == "vanilla_pipeline"

    foundry_num_designs = int(num_candidates)
    st.markdown("#### Foundry Native Settings")
    foundry_cols = st.columns(3)
    with foundry_cols[0]:
        st.metric("Design attempts", foundry_num_designs)
    with foundry_cols[1]:
        foundry_timesteps = st.number_input(
            "RFD3 diffusion timesteps",
            min_value=1,
            max_value=200,
            value=int(loaded_params.get("timesteps") or 50),
            step=1,
            key=f"foundry_timesteps_{source_key}",
        )
    with foundry_cols[2]:
        foundry_is_non_loopy = st.checkbox(
            "Non-loopy binder",
            value=bool(loaded_params.get("is_non_loopy", True)),
            key=f"foundry_is_non_loopy_{source_key}",
        )

    foundry_contig_key = f"foundry_contig_{source_key}"
    foundry_auto_contig_key = f"{foundry_contig_key}_auto"
    foundry_manual_contig_key = f"{foundry_contig_key}_manual"
    computed_foundry_contig = _rfdiffusion3_contig(target_pdb, target_chains, binder_length)
    foundry_manual_contig = st.checkbox(
        "Edit RFD3 contig manually",
        value=bool(st.session_state.get(foundry_manual_contig_key, False)),
        key=foundry_manual_contig_key,
        help="Leave off to rebuild the contig from the selected target and binder length on every rerun.",
    )
    if not foundry_manual_contig:
        st.session_state[foundry_contig_key] = computed_foundry_contig
        st.session_state[foundry_auto_contig_key] = computed_foundry_contig
    elif loaded_params.get("contig") and foundry_contig_key not in st.session_state:
        st.session_state[foundry_contig_key] = normalize_rfdiffusion3_contig(str(loaded_params.get("contig")))
        st.session_state[foundry_auto_contig_key] = computed_foundry_contig
    elif foundry_contig_key not in st.session_state:
        st.session_state[foundry_contig_key] = computed_foundry_contig
        st.session_state[foundry_auto_contig_key] = computed_foundry_contig
    else:
        st.session_state[foundry_contig_key] = normalize_rfdiffusion3_contig(str(st.session_state.get(foundry_contig_key)))
        st.session_state[foundry_auto_contig_key] = computed_foundry_contig
    foundry_contig = st.text_input(
        "RFD3 contig",
        help="Auto-built from the prepared target plus binder length unless manual editing is enabled.",
        disabled=not foundry_manual_contig,
        key=foundry_contig_key,
    )

    foundry_infer_ori_strategy = "hotspots" if hotspots else "default"
    st.caption("RFD3 uses selected hotspots for binder placement." if hotspots else "RFD3 uses default binder placement because no hotspots are selected.")
    with st.expander("Advanced RFD3 placement override", expanded=False):
        foundry_infer_ori_strategy = st.selectbox(
            "Binder placement strategy",
            ["hotspots", "default"] if hotspots else ["default", "hotspots"],
            index=0,
            format_func=lambda value: "Use selected hotspots" if value == "hotspots" else "Default placement",
            help="Leave this automatic unless you are debugging RFdiffusion3 behavior.",
            key=f"foundry_infer_ori_strategy_{source_key}",
        )

    foundry_mpnn_sequences = 1
    foundry_mpnn_model_type = "protein_mpnn"
    foundry_mpnn_checkpoint = "/weights/proteinmpnn_v_48_020.pt"
    foundry_prepare_target_msa = True
    if foundry_vanilla_pipeline:
        st.markdown("#### Foundry MPNN And RF3")
        foundry_model_options = _foundry_mpnn_model_options()
        loaded_checkpoint = str(loaded_params.get("mpnn_checkpoint_path") or loaded_params.get("checkpoint_path") or "/weights/proteinmpnn_v_48_020.pt")
        loaded_model_type = str(loaded_params.get("mpnn_model_type") or loaded_params.get("model_type") or _foundry_mpnn_model_from_checkpoint(loaded_checkpoint))
        native_cols = st.columns(3)
        with native_cols[0]:
            foundry_mpnn_sequences = st.number_input(
                "MPNN sequences per backbone",
                min_value=1,
                max_value=1000,
                value=int(loaded_params.get("mpnn_sequences_per_backbone") or loaded_params.get("batch_size") or 1),
                step=1,
                key=f"foundry_mpnn_sequences_{source_key}",
            )
        with native_cols[1]:
            foundry_mpnn_model_type = st.selectbox(
                "Foundry MPNN model",
                list(foundry_model_options),
                format_func=lambda key: foundry_model_options[key]["label"],
                index=list(foundry_model_options).index(loaded_model_type) if loaded_model_type in foundry_model_options else 0,
                key=f"foundry_mpnn_model_type_{source_key}",
            )
            foundry_mpnn_checkpoint = foundry_model_options[foundry_mpnn_model_type]["checkpoint"]
        with native_cols[2]:
            foundry_prepare_target_msa = st.checkbox(
                "Use cached target MSA for RF3",
                value=bool(loaded_params.get("prepare_target_msa", True)),
                key=f"foundry_prepare_target_msa_{source_key}",
                help="Uses /mnt/db/reference_files/boltz_models/msa_repository; missing target MSAs are created through the shared Boltz2 MSA cache helper.",
            )
        st.caption(foundry_model_options[foundry_mpnn_model_type]["description"])

    _tool_payload_expander(
        "RFdiffusion3 / Foundry",
        {
            "docker_image": "mn-foundry:cu128",
            "mode": "vanilla_foundry_pipeline" if foundry_vanilla_pipeline else "rfd3_generation_only",
            "rfd3_input_json": {
                "design_1": {
                    "dialect": 2,
                    "input": "target.pdb",
                    "contig": foundry_contig,
                    "is_non_loopy": bool(foundry_is_non_loopy),
                    "infer_ori_strategy": foundry_infer_ori_strategy,
                    "select_hotspots": _hotspot_atom_map(hotspots) if hotspots else None,
                }
            },
            "rfd3_command_args": {
                "out_dir": "rfd3",
                "inputs": "rfd3_inputs_staged.json",
                "ckpt_path": "/weights/rfd3_latest.ckpt",
                "diffusion_batch_size": 1,
                "n_batches": int(foundry_num_designs),
                "inference_sampler.num_timesteps": int(foundry_timesteps),
                "skip_existing": False,
                "prevalidate_inputs": True,
            },
            "foundry_mpnn": {
                "enabled": foundry_vanilla_pipeline,
                "batch_size": int(foundry_mpnn_sequences),
                "number_of_batches": 1,
                "model_type": foundry_mpnn_model_type,
                "checkpoint_path": foundry_mpnn_checkpoint,
            },
            "rf3": {
                "enabled": foundry_vanilla_pipeline,
                "checkpoint_path": "/weights/rf3_foundry_01_24_latest_remapped.ckpt",
                "target_msa": bool(foundry_prepare_target_msa),
            },
        },
    )
    foundry_run_label = "Run Foundry vanilla pipeline" if foundry_vanilla_pipeline else "Run RFD3 generation"
    if st.button(foundry_run_label, type="primary", disabled=not target_chains):
        try:
            with st.spinner("Queueing RFdiffusion3 / Foundry..."):
                run_dir = run_rfdiffusion3_foundry(
                    target_pdb=target_pdb,
                    target_chains=target_chains,
                    binder_length=binder_length,
                    hotspots=hotspots,
                    campaign_name=campaign_name,
                    num_designs=int(foundry_num_designs),
                    timesteps=int(foundry_timesteps),
                    contig=foundry_contig,
                    is_non_loopy=bool(foundry_is_non_loopy),
                    infer_ori_strategy=foundry_infer_ori_strategy,
                    run_vanilla_pipeline=foundry_vanilla_pipeline,
                    mpnn_sequences_per_backbone=int(foundry_mpnn_sequences),
                    mpnn_model_type=foundry_mpnn_model_type,
                    mpnn_checkpoint_path=foundry_mpnn_checkpoint,
                    prepare_target_msa=bool(foundry_prepare_target_msa),
                    gpu_device=design_gpu_device,
                    queue_cpu_cores=design_cpu_cores,
                )
            st.success("RFdiffusion3 / Foundry job queued. It will continue if you close Streamlit.")
            st.link_button("Open result", result_link("design", run_dir.name))
        except Exception as exc:
            st.error(str(exc))

with boltzgen_tab:
    st.caption("BoltzGen can run its native protein-anything workflow: design, inverse folding, folding, design folding, analysis, and filtering.")
    st.subheader("Run Mode")
    boltzgen_run_mode = st.segmented_control(
        "BoltzGen workflow",
        ["vanilla_pipeline", "generation_only"],
        selection_mode="single",
        default="vanilla_pipeline",
        format_func={
            "vanilla_pipeline": "Vanilla pipeline",
            "generation_only": "Generation only",
        }.get,
        key=f"boltzgen_run_mode_{source_key}",
    )
    boltzgen_vanilla_pipeline = boltzgen_run_mode == "vanilla_pipeline"

    st.markdown("#### 1. BoltzGen Generation")
    boltzgen_num_designs = int(num_candidates)
    gen_cols = st.columns(3)
    with gen_cols[0]:
        st.metric("Design attempts", boltzgen_num_designs)
    with gen_cols[1]:
        boltzgen_budget = st.number_input(
            "Final design budget",
            min_value=1,
            max_value=max(1, int(boltzgen_num_designs)),
            value=min(int(loaded_params.get("budget") or 1), max(1, int(boltzgen_num_designs))),
            step=1,
            disabled=not boltzgen_vanilla_pipeline,
            help="BoltzGen final diversity-optimized design count. This maps to --budget.",
            key=f"boltzgen_budget_{source_key}",
        )
    with gen_cols[2]:
        boltzgen_sampling_steps = st.number_input(
            "Sampling steps",
            min_value=1,
            max_value=500,
            value=int(loaded_params.get("sampling_steps") or 20),
            step=1,
            key=f"boltzgen_sampling_steps_{source_key}",
        )

    if boltzgen_vanilla_pipeline:
        st.markdown("#### 2. Native BoltzGen Steps")
        st.caption("Runs design -> inverse folding -> folding -> design folding -> analysis -> filtering. Final candidates are normalized from final_ranked_designs.")
    else:
        st.markdown("#### 2. Debug Output")
        st.caption("Stops after BoltzGen design and normalizes intermediate_designs. Use this only for quick smoke checks.")

    first_chain = target_chains[0] if target_chains else "A"
    _tool_payload_expander(
        "BoltzGen",
        {
            "docker_image": "mn-boltzgen:latest",
            "yaml_spec": {
                "entities": [
                    {"protein": {"id": "B", "sequence": binder_length}},
                    {
                        "file": {
                            "path": "target.pdb",
                            "include": [{"chain": {"id": first_chain, "res_index": "selected target residue span"}}],
                            "binding_types": [{"chain": {"id": first_chain, "binding": hotspots}}],
                            "structure_groups": "all",
                        }
                    },
                ]
            },
            "command_args": {
                "protocol": "protein-anything",
                "steps": "all" if boltzgen_vanilla_pipeline else "design",
                "num_designs": int(boltzgen_num_designs),
                "budget": int(boltzgen_budget) if boltzgen_vanilla_pipeline else None,
                "diffusion_batch_size": 1,
                "devices": 1,
                "sampling_steps": int(boltzgen_sampling_steps),
                "compile_pairformer": False,
                "compile_structure": False,
            },
            "workflow": {
                "mode": boltzgen_run_mode,
                "native_outputs": "final_ranked_designs" if boltzgen_vanilla_pipeline else "intermediate_designs",
            },
        },
    )
    boltzgen_run_label = "Run BoltzGen vanilla pipeline" if boltzgen_vanilla_pipeline else "Run BoltzGen generation"
    if st.button(boltzgen_run_label, type="primary", disabled=not target_chains):
        try:
            with st.spinner("Queueing BoltzGen..."):
                run_dir = run_boltzgen(
                    target_pdb=target_pdb,
                    target_chains=target_chains,
                    binder_length=binder_length,
                    hotspots=hotspots,
                    campaign_name=campaign_name,
                    num_designs=int(boltzgen_num_designs),
                    budget=int(boltzgen_budget),
                    sampling_steps=int(boltzgen_sampling_steps),
                    run_vanilla_pipeline=boltzgen_vanilla_pipeline,
                    gpu_device=design_gpu_device,
                    queue_cpu_cores=design_cpu_cores,
                )
            st.success("BoltzGen job queued. It will continue if you close Streamlit.")
            st.link_button("Open BoltzGen result", result_link("design", run_dir.name))
        except Exception as exc:
            st.error(str(exc))

with pxdesign_tab:
    st.caption("Runs the native PXDesign pipeline against the selected prepared target. The accepted PXDesign structures are normalized for the Analysis page.")
    st.subheader("Run Mode")
    loaded_px_mode = str(loaded_params.get("run_mode") or loaded_params.get("preset") or "extended")
    if loaded_px_mode == "pipeline":
        loaded_px_mode = str(loaded_params.get("preset") or "extended")
    if loaded_px_mode not in {"generation_only", "preview", "extended"}:
        loaded_px_mode = "extended"
    pxdesign_mode = st.segmented_control(
        "PXDesign workflow",
        ["generation_only", "preview", "extended"],
        selection_mode="single",
        default=loaded_px_mode,
        format_func={
            "generation_only": "Generation only",
            "preview": "Preview",
            "extended": "Extended",
        }.get,
        help="Generation only returns raw PXDesign-d backbones. Preview runs AF2-IG filtering. Extended also runs Protenix filtering and is closest to the paper pipeline.",
        key=f"pxdesign_mode_{source_key}",
    )
    pxdesign_mode = str(pxdesign_mode or "preview")
    pxdesign_preset = "preview" if pxdesign_mode == "generation_only" else pxdesign_mode
    if pxdesign_mode == "extended":
        st.caption("Extended mode is the production path from the PXDesign docs. It benefits from precomputed target MSAs and is much heavier.")
    elif pxdesign_mode == "preview":
        st.caption("Preview mode is the faster AF2-IG path for checking target crop, hotspots, and binder length.")
    else:
        st.caption("Generation-only runs PXDesign-d inference and does not produce ranked validation metrics.")
    st.subheader("PXDesign Generation")
    try:
        px_default_length = int(design_workflow.parse_binder_lengths(binder_length)[0])
    except Exception:
        px_default_length = 55
    px_cols = st.columns(3)
    with px_cols[0]:
        pxdesign_num_designs = st.number_input(
            "Design attempts",
            1,
            1000,
            int(num_candidates),
            key=f"pxdesign_num_designs_{source_key}",
        )
    with px_cols[1]:
        pxdesign_binder_length = st.number_input(
            "Binder length used by PXDesign",
            1,
            1000,
            int(loaded_params.get("pxdesign_binder_length") or px_default_length),
            help="PXDesign currently takes one integer length. If the campaign length is a range, this uses the first value unless changed here.",
            key=f"pxdesign_binder_length_{source_key}",
        )
    with px_cols[2]:
        pxdesign_steps = st.number_input(
            "Diffusion steps",
            1,
            2000,
            int(loaded_params.get("n_steps") or 400),
            key=f"pxdesign_steps_{source_key}",
        )
    px_runtime_cols = st.columns(4)
    with px_runtime_cols[0]:
        pxdesign_n_max_runs = st.number_input(
            "Max PXDesign runs",
            1,
            1000,
            int(loaded_params.get("n_max_runs") or 1),
            help="PXDesign can retry internal runs until it reaches this limit.",
            key=f"pxdesign_n_max_runs_{source_key}",
        )
    with px_runtime_cols[1]:
        pxdesign_dtype = st.selectbox(
            "dtype",
            ["bf16", "fp32"],
            index=0 if str(loaded_params.get("dtype") or "bf16") == "bf16" else 1,
            key=f"pxdesign_dtype_{source_key}",
        )
    with px_runtime_cols[2]:
        pxdesign_use_fast_ln = st.checkbox(
            "Use fast layer norm",
            value=bool(loaded_params.get("use_fast_ln", True)),
            key=f"pxdesign_use_fast_ln_{source_key}",
        )
    with px_runtime_cols[3]:
        px_deepspeed_default = bool(loaded_params["use_deepspeed_evo_attention"]) if "use_deepspeed_evo_attention" in loaded_params else True
        pxdesign_use_deepspeed = st.checkbox(
            "Use DeepSpeed EvoAttention",
            value=px_deepspeed_default,
            help="PXDesign docs recommend this kernel optimization for modern GPUs in extended mode. It is used by the Protenix filter.",
            key=f"pxdesign_use_deepspeed_{source_key}",
        )
    pxdesign_prepare_msa = st.checkbox(
        "Use cached target MSA for Protenix evaluation",
        value=bool(loaded_params.get("prepare_target_msa", True)),
        disabled=pxdesign_mode == "generation_only",
        help=(
            "For PXDesign preview/extended runs, materializes per-chain MSA directories "
            "from /mnt/db/reference_files/boltz_models/msa_repository. Missing MSAs are "
            "created with the same Boltz2 MSA-server cache helper used by Genie3."
        ),
        key=f"pxdesign_prepare_msa_{source_key}",
    )
    first_chain = target_chains[0] if target_chains else "A"
    px_hotspots_by_chain: dict[str, list[int]] = {}
    for token in hotspots.split(",") if hotspots else []:
        try:
            px_hotspots_by_chain.setdefault(token[0], []).append(int(token[1:]))
        except ValueError:
            continue
    px_yaml_chains = {
        chain: (
            {
                **({"hotspots": px_hotspots_by_chain[chain]} if px_hotspots_by_chain.get(chain) else {}),
                **({"msa": f"input/msa/{chain}"} if pxdesign_prepare_msa and pxdesign_mode != "generation_only" else {}),
            }
            or "all"
        )
        for chain in (target_chains or [first_chain])
    }
    _tool_payload_expander(
        "PXDesign",
        {
            "docker_image": "mn-pxdesign:cu128",
            "yaml_spec": {
                "target": {
                    "file": "input/target.pdb",
                    "chains": px_yaml_chains,
                },
                "binder_length": int(pxdesign_binder_length),
            },
            "command_args": {
                "mode": pxdesign_mode,
                "preset": pxdesign_preset if pxdesign_mode != "generation_only" else None,
                "input": "artifacts/raw/pxdesign/input/target_binder.yaml",
                "output": "artifacts/raw/pxdesign/output",
                "N_sample": int(pxdesign_num_designs),
                "N_step": int(pxdesign_steps),
                "N_max_runs": int(pxdesign_n_max_runs) if pxdesign_mode != "generation_only" else None,
                "dtype": pxdesign_dtype,
                "use_fast_ln": bool(pxdesign_use_fast_ln),
                "use_deepspeed_evo_attention": bool(pxdesign_use_deepspeed),
                "load_checkpoint_dir": "/ref/pxdesign/release_data/checkpoint",
            },
            "normalization": (
                "output/**/*.cif -> normalized backbone candidates"
                if pxdesign_mode == "generation_only"
                else "output/design_outputs/*/summary.csv -> normalized complex-refolding candidates"
            ),
        },
    )
    px_run_label = "Run PXDesign generation" if pxdesign_mode == "generation_only" else f"Run PXDesign {pxdesign_preset} pipeline"
    if st.button(px_run_label, type="primary", disabled=not target_chains):
        try:
            with st.spinner("Queueing PXDesign..."):
                run_dir = run_pxdesign(
                    target_pdb=target_pdb,
                    target_chains=target_chains,
                    binder_length=str(int(pxdesign_binder_length)),
                    hotspots=hotspots,
                    campaign_name=campaign_name,
                    num_designs=int(pxdesign_num_designs),
                    n_steps=int(pxdesign_steps),
                    dtype=pxdesign_dtype,
                    preset=pxdesign_preset,
                    run_mode="generation_only" if pxdesign_mode == "generation_only" else "pipeline",
                    n_max_runs=int(pxdesign_n_max_runs),
                    use_fast_ln=bool(pxdesign_use_fast_ln),
                    use_deepspeed_evo_attention=bool(pxdesign_use_deepspeed),
                    prepare_target_msa=bool(pxdesign_prepare_msa),
                    gpu_device=design_gpu_device,
                    queue_cpu_cores=design_cpu_cores,
                )
            st.success("PXDesign job queued. It will continue if you close Streamlit.")
            st.link_button("Open result", result_link("design", run_dir.name))
        except Exception as exc:
            st.error(str(exc))

with genie3_tab:
    st.caption("Runs Genie3's native binder-design workflow against the selected prepared target.")
    genie3_mode = st.segmented_control(
        "Genie3 workflow",
        ["generation_only", "full_vanilla_pipeline"],
        selection_mode="single",
        default=str(loaded_params.get("run_mode") or "full_vanilla_pipeline"),
        format_func={
            "generation_only": "Generation only",
            "full_vanilla_pipeline": "Full vanilla pipeline",
        }.get,
        key=f"genie3_run_mode_{source_key}",
    )
    genie3_full_pipeline = str(genie3_mode or "full_vanilla_pipeline") == "full_vanilla_pipeline"
    st.markdown("#### 1. Genie3 Generation")
    genie3_gen_cols = st.columns(3)
    with genie3_gen_cols[0]:
        genie3_num_designs = st.number_input(
            "Design attempts",
            1,
            1000,
            int(num_candidates),
            key=f"genie3_num_designs_{source_key}",
        )
    with genie3_gen_cols[1]:
        genie3_seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=999999,
            value=int(loaded_params.get("seed") or 7),
            step=1,
            key=f"genie3_seed_{source_key}",
        )
    with genie3_gen_cols[2]:
        genie3_num_devices = st.number_input(
            "GPU devices",
            min_value=1,
            max_value=8,
            value=int(loaded_params.get("num_devices") or 1),
            step=1,
            key=f"genie3_num_devices_{source_key}",
        )
    genie3_cond_options = ["hotspot", "extended", "common", "iter_common", "iter_common_prob"]
    loaded_cond = str(loaded_params.get("cond_strategy") or "extended")
    if loaded_cond not in genie3_cond_options:
        loaded_cond = "extended"
    genie3_cond_strategy = st.selectbox(
        "Conditioning strategy",
        genie3_cond_options,
        index=genie3_cond_options.index(loaded_cond),
        help="Genie3 interface conditioning. 'extended' uses residues around the selected hotspots; iterative modes reuse previous round successes.",
        key=f"genie3_cond_strategy_{source_key}",
    )
    genie3_adv = st.expander("Genie3 advanced generation", expanded=False)
    with genie3_adv:
        genie3_direction_scale = st.number_input(
            "direction_scale",
            min_value=0.0,
            max_value=2.0,
            value=float(loaded_params.get("direction_scale") or 0.0),
            step=0.1,
            help="Repo default for binder design is 0.0.",
            key=f"genie3_direction_scale_{source_key}",
        )
        genie3_compile = st.checkbox(
            "Enable torch.compile",
            value=bool(loaded_params.get("compile_generation", False)),
            help="Genie3 beam-search examples enable this, but the app keeps it off by default for stable container runs.",
            key=f"genie3_compile_{source_key}",
        )
        genie3_beam = st.checkbox(
            "Use beam search",
            value=bool(loaded_params.get("enable_beam_search", False)),
            help="Adds Genie3 inference.search=beam and ColabFold reward. More expensive, but can improve quality.",
            key=f"genie3_beam_{source_key}",
        )
        genie3_beam_width = st.number_input(
            "Beam width",
            min_value=1,
            max_value=16,
            value=int(loaded_params.get("beam_width") or 4),
            step=1,
            disabled=not genie3_beam,
            key=f"genie3_beam_width_{source_key}",
        )
    if genie3_full_pipeline:
        st.markdown("#### 2. Native Evaluation")
        eval_cols = st.columns(4)
        with eval_cols[0]:
            genie3_num_seq = st.number_input(
                "inverse_folding.num_seq",
                min_value=1,
                max_value=64,
                value=int(loaded_params.get("inverse_folding_num_seq") or 1),
                step=1,
                help="Genie3 repo example default for binder design is 1.",
                key=f"genie3_inverse_folding_num_seq_{source_key}",
            )
        with eval_cols[1]:
            loaded_folding_model = str(loaded_params.get("folding_model_name") or "colabfold")
            if loaded_folding_model not in {"colabfold", "boltz2"}:
                loaded_folding_model = "colabfold"
            genie3_folding_model = st.selectbox(
                "folding.model_name",
                ["colabfold", "boltz2"],
                index=["colabfold", "boltz2"].index(loaded_folding_model),
                help="Genie3's repo default is ColabFold. Use Boltz2 when the Genie3 container does not include colabfold_batch.",
                key=f"genie3_folding_model_{source_key}",
            )
        with eval_cols[2]:
            folding_mode_options = ["template", "msa"] if genie3_folding_model == "colabfold" else ["msa"]
            loaded_folding_mode = str(loaded_params.get("folding_mode") or ("msa" if genie3_folding_model == "boltz2" else "template"))
            if loaded_folding_mode not in folding_mode_options:
                loaded_folding_mode = folding_mode_options[0]
            genie3_folding_mode = st.selectbox(
                "folding.mode",
                folding_mode_options,
                index=folding_mode_options.index(loaded_folding_mode),
                help="Template mode is the Genie3/ColabFold repo default. Boltz2 evaluation uses MSA mode.",
                key=f"genie3_folding_mode_{source_key}",
            )
        with eval_cols[3]:
            genie3_num_models = st.number_input(
                "folding.num_models",
                min_value=1,
                max_value=10,
                value=int(loaded_params.get("folding_num_models") or 5),
                step=1,
                key=f"genie3_folding_num_models_{source_key}",
            )
        genie3_num_recycles = st.number_input(
            "folding.num_recycles",
            min_value=1,
            max_value=50,
            value=int(loaded_params.get("folding_num_recycles") or 20),
            step=1,
            key=f"genie3_folding_num_recycles_{source_key}",
        )
    else:
        genie3_num_seq = 1
        genie3_folding_model = "colabfold"
        genie3_folding_mode = "template"
        genie3_num_models = 5
        genie3_num_recycles = 20
    _tool_payload_expander(
        "Genie3",
        {
            "docker_image": "mn-genie3:cu128",
            "command": "genie3 run" if genie3_full_pipeline else "genie3 generate",
            "experiment_yaml": "artifacts/raw/genie3/experiment.yaml",
            "dataset": "artifacts/raw/genie3/dataset/mn_app",
            "generation": {
                "source": "target",
                "n_sample": int(genie3_num_designs),
                "cond_strategy": genie3_cond_strategy,
                "direction_scale": float(genie3_direction_scale),
                "beam_search": bool(genie3_beam),
            },
            "evaluation": {
                "version": "binder",
                "inverse_folding.num_seq": int(genie3_num_seq),
                "folding.model_name": genie3_folding_model,
                "folding.mode": genie3_folding_mode,
                "folding.num_models": int(genie3_num_models),
                "folding.num_recycles": int(genie3_num_recycles),
            } if genie3_full_pipeline else None,
            "target": {
                "chains": target_chains,
                "hotspots": hotspots,
                "binder_length": binder_length,
            },
        },
    )
    genie3_run_label = "Run Genie3 full vanilla pipeline" if genie3_full_pipeline else "Run Genie3 generation"
    if st.button(genie3_run_label, type="primary", disabled=not target_chains):
        try:
            with st.spinner("Queueing Genie3..."):
                run_dir = run_genie3(
                    target_pdb=target_pdb,
                    target_chains=target_chains,
                    binder_length=binder_length,
                    hotspots=hotspots,
                    campaign_name=campaign_name,
                    num_designs=int(genie3_num_designs),
                    seed=int(genie3_seed),
                    num_devices=int(genie3_num_devices),
                    direction_scale=float(genie3_direction_scale),
                    cond_strategy=genie3_cond_strategy,
                    inverse_folding_num_seq=int(genie3_num_seq),
                    folding_model_name=genie3_folding_model,
                    folding_mode=genie3_folding_mode,
                    folding_num_models=int(genie3_num_models),
                    folding_num_recycles=int(genie3_num_recycles),
                    compile_generation=bool(genie3_compile),
                    run_mode=str(genie3_mode or "full_vanilla_pipeline"),
                    enable_beam_search=bool(genie3_beam),
                    beam_width=int(genie3_beam_width),
                    gpu_device=design_gpu_device,
                    queue_cpu_cores=design_cpu_cores,
                )
            st.success("Genie3 job queued. It will continue if you close Streamlit.")
            st.link_button("Open result", result_link("design", run_dir.name))
        except Exception as exc:
            st.error(str(exc))

with esm_tab:
    esm_mode = st.segmented_control(
        "ESM workflow",
        ["Native binder design", "Sequence screening"],
        default="Native binder design",
        key=f"esm_workflow_mode_{source_key}",
    )
    if esm_mode == "Native binder design":
        st.subheader("ESMFold2 Binder Design")
        native_cols = st.columns(4)
        with native_cols[0]:
            esm_target_chain = st.selectbox(
                "Target chain",
                target_chains,
                key=f"esm_native_target_chain_{source_key}",
            )
        with native_cols[1]:
            requested_lengths = design_workflow.parse_binder_lengths(binder_length)
            esm_native_length = st.number_input(
                "Binder length",
                min_value=40,
                max_value=200,
                value=int(requested_lengths[0]),
                step=1,
                key=f"esm_native_binder_length_{source_key}",
            )
        with native_cols[2]:
            esm_native_designs = st.number_input(
                "Design attempts",
                min_value=1,
                max_value=20,
                value=int(loaded_params.get("num_designs") or min(int(num_candidates), 2)),
                step=1,
                key=f"esm_native_num_designs_{source_key}",
            )
        with native_cols[3]:
            esm_native_seed = st.number_input(
                "Seed",
                min_value=0,
                max_value=999999,
                value=int(loaded_params.get("seed") or 0),
                step=1,
                key=f"esm_native_seed_{source_key}",
            )
        native_advanced = st.expander("Advanced ESM binder parameters")
        with native_advanced:
            advanced_cols = st.columns(3)
            with advanced_cols[0]:
                esm_native_steps = st.number_input(
                    "Optimization steps",
                    min_value=1,
                    max_value=500,
                    value=int(loaded_params.get("optimization_steps") or 150),
                    step=5,
                    key=f"esm_native_steps_{source_key}",
                )
            with advanced_cols[1]:
                esm_native_lr = st.number_input(
                    "Learning rate",
                    min_value=0.001,
                    max_value=1.0,
                    value=float(loaded_params.get("learning_rate") or 0.1),
                    step=0.01,
                    format="%.3f",
                    key=f"esm_native_learning_rate_{source_key}",
                )
            with advanced_cols[2]:
                esm_native_model = st.selectbox(
                    "Experimental checkpoint",
                    [esm_binder_workflow.DEFAULT_BINDER_MODEL],
                    key=f"esm_native_model_{source_key}",
                )
            esm_native_compile = st.checkbox(
                "Compile model",
                value=bool(loaded_params.get("compile_model") or False),
                key=f"esm_native_compile_{source_key}",
            )
            esm_native_checkpoint_lm = st.checkbox(
                "Activation-checkpoint ESMC",
                value=bool(loaded_params.get("checkpoint_lm") or False),
                key=f"esm_native_checkpoint_lm_{source_key}",
            )
        st.caption(
            "Lean shared-model profile for a 24 GB GPU. Native ESM design uses one target chain; "
            "selected hotspots are preserved for downstream scoring."
        )
        _tool_payload_expander(
            "ESMFold2 native binder-design payload",
            {
                "image": esm_binder_workflow.ESMFOLD2_BINDER_IMAGE,
                "checkpoint": str(
                    esm_binder_workflow.ESMFOLD2_BINDER_MODEL_ROOT / esm_native_model
                ),
                "target": {
                    "pdb": str(target_pdb),
                    "chain": esm_target_chain,
                    "hotspots_for_downstream_scoring": hotspots,
                },
                "binder": {
                    "length": int(esm_native_length),
                    "num_designs": int(esm_native_designs),
                },
                "optimization": {
                    "steps": int(esm_native_steps),
                    "learning_rate": float(esm_native_lr),
                    "seed": int(esm_native_seed),
                    "compile": bool(esm_native_compile),
                    "checkpoint_lm": bool(esm_native_checkpoint_lm),
                    "memory_profile": "lean_shared_model",
                },
            },
        )
        if st.button("Run ESMFold2 binder design", type="primary", disabled=not target_chains):
            try:
                with st.spinner("Queueing native ESMFold2 binder design..."):
                    run_dir = run_esmfold2_native_binder_design(
                        target_pdb=target_pdb,
                        target_chain=esm_target_chain,
                        hotspots=hotspots,
                        binder_length=int(esm_native_length),
                        num_designs=int(esm_native_designs),
                        campaign_name=campaign_name,
                        optimization_steps=int(esm_native_steps),
                        learning_rate=float(esm_native_lr),
                        seed=int(esm_native_seed),
                        model_name=esm_native_model,
                        compile_model=bool(esm_native_compile),
                        checkpoint_lm=bool(esm_native_checkpoint_lm),
                        gpu_device=design_gpu_device,
                        queue_cpu_cores=design_cpu_cores,
                    )
                st.success("ESMFold2 binder-design job queued. It will continue if you close Streamlit.")
                st.link_button("Open result", result_link("design", run_dir.name))
            except Exception as exc:
                st.error(str(exc))
    else:
        st.subheader("ESMFold2 Binder Screening")
        esm_cols = st.columns(4)
        with esm_cols[0]:
            esm_num_designs = st.number_input(
                "Sequence proposals",
                1,
                100,
                int(loaded_params.get("num_designs") or min(max(int(num_candidates), 1), 8)),
                key=f"esm_num_designs_{source_key}",
            )
        with esm_cols[1]:
            esm_num_steps = st.number_input(
                "Sampling steps",
                1,
                200,
                int(loaded_params.get("num_sampling_steps") or 32),
                key=f"esm_num_sampling_steps_{source_key}",
            )
        with esm_cols[2]:
            esm_num_loops = st.number_input(
                "Recycling loops",
                1,
                10,
                int(loaded_params.get("num_loops") or 3),
                key=f"esm_num_loops_{source_key}",
            )
        with esm_cols[3]:
            esm_seed = st.number_input(
                "Seed",
                0,
                999999,
                int(loaded_params.get("seed") or 11),
                key=f"esm_seed_{source_key}",
            )
        esm_binder_sequences = st.text_area(
            "Optional binder sequences",
            value=str(loaded_params.get("binder_sequences_text") or ""),
            placeholder="Paste FASTA or one sequence per line.",
            height=140,
            key=f"esm_binder_sequences_{source_key}",
        )
        esm_contact_cutoff = st.number_input(
            "Contact cutoff",
            min_value=3.0,
            max_value=20.0,
            value=float(loaded_params.get("contact_cutoff") or 8.0),
            step=0.5,
            key=f"esm_contact_cutoff_{source_key}",
        )
        if st.button("Run ESMFold2 screening", type="primary", disabled=not target_chains):
            try:
                with st.spinner("Queueing ESMFold2 screening..."):
                    run_dir = run_esmfold2_binder_screening(
                        target_pdb=target_pdb,
                        target_chains=target_chains,
                        hotspots=hotspots,
                        binder_length=binder_length,
                        num_designs=int(esm_num_designs),
                        binder_sequences_text=esm_binder_sequences,
                        campaign_name=campaign_name,
                        num_loops=int(esm_num_loops),
                        num_sampling_steps=int(esm_num_steps),
                        seed=int(esm_seed),
                        device="auto",
                        contact_cutoff=float(esm_contact_cutoff),
                        queue_cpu_cores=design_cpu_cores,
                    )
                st.success("ESMFold2 screening job queued. It will continue if you close Streamlit.")
                st.link_button("Open result", result_link("design", run_dir.name))
            except Exception as exc:
                st.error(str(exc))

with protpardelle_tab:
    st.caption(
        "Native Protpardelle-1c binder-generation workflow. The app writes a target motif PDB, "
        "maps hotspots into Protpardelle numbering, runs scaffold generation, then runs ProteinMPNN "
        "and ESMFold self-consistency inside the tool."
    )
    st.subheader("Protpardelle-1c Generation")
    protpardelle_cols = st.columns(4)
    with protpardelle_cols[0]:
        protpardelle_num_designs = st.number_input(
            "Design attempts",
            1,
            1000,
            int(loaded_params.get("num_designs") or max(int(num_candidates), 100)),
            key=f"protpardelle_num_designs_{source_key}",
            help="Protpardelle's BindCraft benchmark uses 100 backbone attempts. Small 10-design runs are useful for smoke tests but can easily produce no acceptable designs.",
        )
    with protpardelle_cols[1]:
        protpardelle_num_mpnn = st.number_input(
            "MPNN sequences per design",
            0,
            100,
            int(loaded_params.get("num_mpnn_seqs") or 2),
            key=f"protpardelle_num_mpnn_{source_key}",
            help="The native Protpardelle BindCraft benchmark uses 2 MPNN sequences per backbone.",
        )
    with protpardelle_cols[2]:
        protpardelle_batch_size = st.number_input(
            "Sampling batch size",
            1,
            128,
            int(loaded_params.get("batch_size") or 1),
            key=f"protpardelle_batch_size_{source_key}",
        )
    with protpardelle_cols[3]:
        protpardelle_seed = st.number_input(
            "Seed",
            0,
            999999,
            int(loaded_params.get("seed") or 7),
            key=f"protpardelle_seed_{source_key}",
        )
    preset_options = {
        "binder_generation": {
            "label": "Binder generation cc83 epoch 2616",
            "model_name": "cc83",
            "model_epoch": "2616",
            "sampling_config": "sampling_sidechain_conditional",
        },
        "binder_generation_cc95": {
            "label": "Binder generation cc95 epoch 3490",
            "model_name": "cc95",
            "model_epoch": "3490",
            "sampling_config": "sampling_sidechain_conditional",
        },
    }
    current_preset_key = "binder_generation_cc95" if str(loaded_params.get("model_name")) == "cc95" else "binder_generation"
    preset_key = st.selectbox(
        "Model preset",
        list(preset_options),
        index=list(preset_options).index(current_preset_key),
        format_func=lambda key: preset_options[key]["label"],
        key=f"protpardelle_model_preset_{source_key}",
    )
    preset = preset_options[preset_key]
    advanced = st.expander("Advanced Protpardelle sampling parameters")
    with advanced:
        adv_cols = st.columns(3)
        with adv_cols[0]:
            protpardelle_step_scale = st.number_input(
                "step_scale",
                0.1,
                5.0,
                float(loaded_params.get("step_scale") or 1.2),
                step=0.1,
                key=f"protpardelle_step_scale_{source_key}",
            )
        with adv_cols[1]:
            protpardelle_schurn = st.number_input(
                "schurn",
                0,
                1000,
                int(loaded_params.get("schurn") or 200),
                key=f"protpardelle_schurn_{source_key}",
            )
        with adv_cols[2]:
            protpardelle_crop_cond_start = st.number_input(
                "crop_cond_start",
                0.0,
                1.0,
                float(loaded_params.get("crop_cond_start") or 0.0),
                step=0.05,
                key=f"protpardelle_crop_cond_start_{source_key}",
            )
    lengths = design_workflow.parse_binder_lengths(binder_length)
    binder_range = [lengths[0], lengths[0]] if len(lengths) == 1 else [lengths[0], lengths[1]]
    protpardelle_preview = {
        "docker_image": "mn-protpardelle-1c:cu128",
        "reference_mount": "/mnt/db/reference_files/protpardelle-1c:/ref/protpardelle-1c:ro",
        "target_chains": target_chains,
        "hotspots_original_numbering": hotspots,
        "generated_inputs": {
            "motif_pdb": "artifacts/raw/protpardelle_1c/input/motifs/mn_app_target.pdb",
            "sampling_yaml": "artifacts/raw/protpardelle_1c/input/protpardelle_sampling.yaml",
        },
        "command_args": {
            "model": [
                preset["model_name"],
                preset["model_epoch"],
                preset["sampling_config"],
            ],
            "num_samples": int(protpardelle_num_designs),
            "num_mpnn_seqs": int(protpardelle_num_mpnn),
            "batch_size": int(protpardelle_batch_size),
            "binder_length_range": binder_range,
            "step_scale": float(protpardelle_step_scale),
            "schurn": int(protpardelle_schurn),
            "crop_cond_start": float(protpardelle_crop_cond_start),
            "seed": int(protpardelle_seed),
        },
    }
    _tool_payload_expander(
        "Protpardelle-1c payload sent to algorithm",
        protpardelle_preview,
    )
    if st.button("Run Protpardelle-1c native pipeline", type="primary", disabled=not target_chains):
        try:
            with st.spinner("Queueing Protpardelle-1c..."):
                run_dir = run_protpardelle_1c(
                    target_pdb=target_pdb,
                    target_chains=target_chains,
                    binder_length=binder_length,
                    hotspots=hotspots,
                    campaign_name=campaign_name,
                    num_designs=int(protpardelle_num_designs),
                    num_mpnn_seqs=int(protpardelle_num_mpnn),
                    model_name=preset["model_name"],
                    model_epoch=preset["model_epoch"],
                    sampling_config=preset["sampling_config"],
                    step_scale=float(protpardelle_step_scale),
                    schurn=int(protpardelle_schurn),
                    crop_cond_start=float(protpardelle_crop_cond_start),
                    batch_size=int(protpardelle_batch_size),
                    seed=int(protpardelle_seed),
                    gpu_device=design_gpu_device,
                    queue_cpu_cores=design_cpu_cores,
                )
            st.success("Protpardelle-1c job queued. It will continue if you close Streamlit.")
            st.link_button("Open result", result_link("design", run_dir.name))
        except Exception as exc:
            st.error(str(exc))

with complexa_tab:
    st.caption("Native Proteina-Complexa binder pipeline. Target chains, hotspots, and binder length come from the selected prepared structure.")
    complexa_cols = st.columns(4)
    with complexa_cols[0]:
        complexa_steps = st.number_input(
            "Generation steps",
            1,
            1000,
            int(loaded_params.get("n_steps") or 400),
            key=f"complexa_steps_{source_key}",
        )
    with complexa_cols[1]:
        complexa_replicas = st.number_input(
            "Best-of-N replicas",
            1,
            100,
            int(loaded_params.get("replicas") or 2),
            key=f"complexa_replicas_{source_key}",
        )
    with complexa_cols[2]:
        complexa_seed = st.number_input(
            "Seed",
            0,
            999999,
            int(loaded_params.get("seed") or 5),
            key=f"complexa_seed_{source_key}",
        )
    with complexa_cols[3]:
        complexa_batch_size = st.number_input(
            "GPU batch size",
            1,
            16,
            int(loaded_params.get("batch_size") or 1),
            help="Lower values are slower but avoid CUDA out-of-memory during best-of-N generation.",
            key=f"complexa_batch_size_{source_key}",
        )
    complexa_lengths = design_workflow.parse_binder_lengths(binder_length)
    complexa_length_range = (
        [complexa_lengths[0], complexa_lengths[0]]
        if len(complexa_lengths) == 1
        else [complexa_lengths[0], complexa_lengths[1]]
    )
    complexa_target_input = design_workflow._target_input_spec(target_pdb, target_chains)
    complexa_hotspot_list = [token for token in hotspots.split(",") if token]
    _tool_payload_expander(
        "Proteina-Complexa",
        {
            "docker_image": "mn-proteina-complexa:latest",
            "contract": "configs/search_binder_local_pipeline.yaml",
            "stages": ["design"],
            "command_args": {
                "run_name": campaign_name or "mn_app_complexa",
                "generation.task_name": "MN_APP_TARGET",
                "generation.target_dict_cfg.MN_APP_TARGET.target_path": "/work/artifacts/raw/proteina_complexa/input/target.pdb",
                "generation.target_dict_cfg.MN_APP_TARGET.target_input": complexa_target_input,
                "generation.target_dict_cfg.MN_APP_TARGET.hotspot_residues": complexa_hotspot_list,
                "generation.target_dict_cfg.MN_APP_TARGET.binder_length": complexa_length_range,
                "generation.dataloader.dataset.nres.nsamples": int(num_candidates),
                "generation.dataloader.batch_size": int(complexa_batch_size),
                "generation.search.max_batch_size": int(complexa_batch_size),
                "generation.search.best_of_n.replicas": int(complexa_replicas),
                "generation.args.nsteps": int(complexa_steps),
                "seed": int(complexa_seed),
                "ckpt_name": "complexa.ckpt",
                "autoencoder_ckpt_path": "/workspace/protein-foundation-models/ckpts/complexa_ae.ckpt",
            },
        },
    )
    if st.button("Run Proteina-Complexa vanilla pipeline", type="primary"):
        try:
            with st.spinner("Queueing Proteina-Complexa..."):
                run_dir = run_proteina_complexa(
                    target_pdb=target_pdb,
                    target_chains=target_chains,
                    binder_length=binder_length,
                    hotspots=hotspots,
                    campaign_name=campaign_name,
                    num_designs=int(num_candidates),
                    n_steps=int(complexa_steps),
                    replicas=int(complexa_replicas),
                    seed=int(complexa_seed),
                    batch_size=int(complexa_batch_size),
                    gpu_device=design_gpu_device,
                    queue_cpu_cores=design_cpu_cores,
                )
            st.success("Proteina-Complexa job queued. It will continue if you close Streamlit.")
            st.link_button("Open result", result_link("design", run_dir.name))
        except Exception as exc:
            st.error(str(exc))
