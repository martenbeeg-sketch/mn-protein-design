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
from mn_protein_design.app.pages.common import result_link
from mn_protein_design.core.candidates import candidate_stage_counts, read_candidates
from mn_protein_design.core.jobs import read_json
from mn_protein_design.core.structures import filter_pdb_text
from mn_protein_design.workflows import design as design_workflow
from mn_protein_design.workflows.campaigns import run_lineage_steps

design_workflow = importlib.reload(design_workflow)
build_rfdiffusion_run_parameters = design_workflow.build_rfdiffusion_run_parameters
default_target_contig = design_workflow.default_target_contig
design_jobs_for_target = design_workflow.design_jobs_for_target
prepared_design_targets = design_workflow.prepared_design_targets
run_boltzgen = design_workflow.run_boltzgen
run_bindcraft = design_workflow.run_bindcraft
run_genie3 = design_workflow.run_genie3
run_proteina_complexa = design_workflow.run_proteina_complexa
run_protpardelle_1c = design_workflow.run_protpardelle_1c
run_pxdesign = design_workflow.run_pxdesign
run_rfdiffusion3_foundry = design_workflow.run_rfdiffusion3_foundry
run_rfdiffusion_classic = design_workflow.run_rfdiffusion_classic
target_label = design_workflow.target_label
normalize_rfdiffusion3_contig = design_workflow._normalize_rfdiffusion3_contig

ResidueId = tuple[str, int]


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
    st.info("Prepare or crop a target before starting a design campaign.")
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
    key=f"design_target_viewer_{target['run_id']}",
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
campaign_cols = st.columns(2)
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

st.subheader("Generator")
rfdiffusion_tab, bindcraft_tab, foundry_tab, boltzgen_tab, pxdesign_tab, genie3_tab, protpardelle_tab, complexa_tab = st.tabs(
    [
        "RFdiffusion classic",
        "BindCraft",
        "RFdiffusion3 / Foundry",
        "BoltzGen",
        "PXDesign",
        "Genie3",
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
            "ligandmpnn": "LigandMPNN (ProteinMPNN weights)",
            "fastrelax": "ProteinMPNN-FastRelax",
        }
        if full_pipeline:
            sequence_design_method = "ligandmpnn"
            full_mpnn_model = st.selectbox(
                "MPNN model",
                ["protein_mpnn", "soluble_mpnn", "ligand_mpnn"],
                format_func={
                    "protein_mpnn": "ProteinMPNN",
                    "soluble_mpnn": "SolubleMPNN",
                    "ligand_mpnn": "LigandMPNN",
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
            extra_run_parameters=extra_run_parameters,
        )
    except ValueError as exc:
        generated_run_parameters = str(loaded_params.get("rfdiffusion_run_parameters") or f"diffuser.T={int(timesteps)}")
        st.warning(str(exc))
    st.markdown("#### 5. Files, Payload, And Run")
    with st.expander("Editable files and final settings sent to RFdiffusion", expanded=True):
        st.caption("These editable copies are written into the run folder under artifacts before Docker starts.")
        editable_run_parameters_key = f"rfdiffusion_editable_run_parameters_{source_key}"
        if st.session_state.pop(f"rfdiffusion_guidance_preset_{source_key}_changed", False):
            st.session_state[editable_run_parameters_key] = generated_run_parameters
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
            "docker_image": "ovo-rfdiffusion:latest",
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
            with st.spinner("Running RFdiffusion generation..."):
                run_dir = run_rfdiffusion_classic(
                    target_pdb=target_pdb,
                    target_chains=target_chains,
                    contig=contig,
                    binder_length=binder_length,
                    hotspots=hotspots,
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
                    save_trajectory=bool(save_trajectory),
                    editable_run_parameters=edited_run_parameters,
                    edited_target_pdb_text=edited_target_text,
                    mpnn_num_sequences=int(mpnn_num_sequences),
                    mpnn_sampling_temp=float(mpnn_sampling_temp),
                    mpnn_omit_aa=mpnn_omit_aa,
                    mpnn_bias_aa=mpnn_bias_aa,
                    mpnn_run_parameters=mpnn_run_parameters,
                    sequence_design_method=sequence_design_method,
                    mpnn_fastrelax_cycles=int(mpnn_fastrelax_cycles),
                    refolding_test=refolding_test,
                    execution_backend=execution_backend,
                )
            st.success("RFdiffusion job finished.")
            st.link_button("Open result", result_link("design", run_dir.name))
            if full_pipeline:
                steps = [
                    {
                        "module": "sequence_design",
                        "tool": "ligandmpnn",
                        "params": {
                            "model_type": str(full_mpnn_model),
                            "design_chains": "",
                            "num_seq_per_target": int(mpnn_num_sequences),
                            "sampling_temp": float(mpnn_sampling_temp),
                            "omit_aas": str(mpnn_omit_aa or "CX"),
                            "seed": None,
                            "require_backbone_hotspot_filter_pass": bool(apply_backbone_hotspot_prefilter),
                        },
                    },
                    {
                        "module": "monomer_refolding",
                        "tool": str(monomer_tool),
                        "params": {"min_plddt": float(min_binder_plddt)},
                    },
                    {
                        "module": "complex_refolding",
                        "tool": str(complex_tool),
                        "params": {
                            "require_monomer_success": True,
                            "template_mode": complex_template_mode,
                            "multimer": complex_multimer,
                            "num_recycles": complex_num_recycles,
                        },
                    },
                    {
                        "module": "analysis",
                        "tool": "ranking",
                        "params": {
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
                ]
                with st.spinner("Running sequence design, refolding, and analysis..."):
                    child_runs = run_lineage_steps(
                        campaign_name.strip() or f"RFdiffusion full pipeline from {run_dir.name}",
                        _source_from_run_dir(run_dir),
                        steps,
                    )
                if child_runs and read_json(child_runs[-1] / "result.json").get("success") is not True:
                    st.error("Full pipeline stopped on a failed step. Open the child job for logs.")
                else:
                    st.success("Full RFdiffusion pipeline finished.")
                _lineage_links(child_runs)
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
            "docker_image": "ovo-bindcraft:latest",
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
            with st.spinner("Running BindCraft..."):
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
                )
            st.success("BindCraft job finished.")
            st.link_button("Open result", result_link("design", run_dir.name))
        except Exception as exc:
            st.error(str(exc))

with foundry_tab:
    st.caption("RFdiffusion3 / Foundry replaces only the backbone generator. Downstream sequence design, refolding, and analysis use the same normalized pipeline.")
    st.subheader("Run Mode")
    foundry_run_mode = st.segmented_control(
        "RFdiffusion3 workflow",
        ["generation_only", "full_vanilla_pipeline"],
        selection_mode="single",
        default="full_vanilla_pipeline",
        format_func={
            "generation_only": "Generation only",
            "full_vanilla_pipeline": "Full vanilla pipeline",
        }.get,
        key=f"foundry_run_mode_{source_key}",
    )
    foundry_full_pipeline = str(foundry_run_mode or "full_vanilla_pipeline") == "full_vanilla_pipeline"

    st.markdown("#### 1. RFdiffusion3 Generation")
    foundry_num_designs = int(num_candidates)
    foundry_cols = st.columns(2)
    with foundry_cols[0]:
        foundry_timesteps = st.number_input(
            "Diffusion timesteps",
            min_value=1,
            max_value=200,
            value=int(loaded_params.get("timesteps") or 50),
            step=1,
            key=f"foundry_timesteps_{source_key}",
        )
    with foundry_cols[1]:
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
        "Edit RFdiffusion3 contig manually",
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
        "RFdiffusion3 contig",
        help="Auto-built from the prepared target plus binder length unless manual editing is enabled.",
        disabled=not foundry_manual_contig,
        key=foundry_contig_key,
    )
    foundry_infer_ori_strategy = "hotspots" if hotspots else "default"
    st.caption(
        "Binder placement uses selected hotspots."
        if hotspots
        else "Binder placement uses the default RFdiffusion3 orientation because no hotspots are selected."
    )
    with st.expander("Advanced RFdiffusion3 placement override", expanded=False):
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
    foundry_monomer_tool = "boltz2_monomer"
    foundry_complex_tool = "af2_initial_guess"
    foundry_complex_template_mode = "target_template"
    foundry_complex_multimer = True
    foundry_complex_num_recycles = 3
    foundry_keep_top_n = 100
    foundry_min_binder_plddt = 70.0
    foundry_max_ipae = 10.0
    foundry_min_ipsae = 0.0
    foundry_max_binder_rmsd = 5.0
    foundry_min_final_hotspot_contact_fraction: float | None = None
    foundry_max_final_hotspot_distance: float | None = None

    if foundry_full_pipeline:
        st.markdown("#### 2. Sequence Design")
        foundry_mpnn_cols = st.columns(2)
        with foundry_mpnn_cols[0]:
            foundry_mpnn_sequences = st.number_input(
                "Sequences per backbone",
                min_value=1,
                max_value=1000,
                value=int(loaded_params.get("batch_size") or 1),
                step=1,
                key=f"foundry_mpnn_sequences_{source_key}",
            )
        with foundry_mpnn_cols[1]:
            foundry_model_options = _foundry_mpnn_model_options()
            loaded_checkpoint = str(loaded_params.get("checkpoint_path") or "/weights/proteinmpnn_v_48_020.pt")
            loaded_model_type = str(loaded_params.get("model_type") or _foundry_mpnn_model_from_checkpoint(loaded_checkpoint))
            foundry_mpnn_model_type = st.selectbox(
                "MPNN model",
                list(foundry_model_options),
                format_func=lambda key: foundry_model_options[key]["label"],
                index=list(foundry_model_options).index(loaded_model_type) if loaded_model_type in foundry_model_options else 0,
                key=f"foundry_mpnn_model_type_{source_key}",
            )
            foundry_mpnn_checkpoint = foundry_model_options[foundry_mpnn_model_type]["checkpoint"]
            st.caption(foundry_model_options[foundry_mpnn_model_type]["description"])

        st.markdown("#### 3. Refolding")
        refold_cols = st.columns(2)
        with refold_cols[0]:
            foundry_monomer_tool = st.selectbox(
                "Monomer refolding",
                ["boltz2_monomer", "esmfold", "af2_monomer"],
                format_func={
                    "boltz2_monomer": "Boltz2 monomer",
                    "esmfold": "ESMFold",
                    "af2_monomer": "AF2 monomer contract",
                }.get,
                key=f"foundry_full_monomer_tool_{source_key}",
            )
        with refold_cols[1]:
            foundry_complex_tool = st.selectbox(
                "Complex refolding",
                ["af2_initial_guess", "boltz2_initial_guess"],
                format_func={
                    "af2_initial_guess": "AF2 initial guess",
                    "boltz2_initial_guess": "Boltz2 initial guess",
                }.get,
                key=f"foundry_full_complex_tool_{source_key}",
            )
        if foundry_complex_tool == "af2_initial_guess":
            foundry_af2_options = {
                "af2_model_1_multimer_tt_3rec": {
                    "label": "AF2 multimer, target template",
                    "template_mode": "target_template",
                    "multimer": True,
                    "num_recycles": 3,
                },
                "af2_model_1_ptm_tt_3rec": {
                    "label": "AF2 monomer model, target template",
                    "template_mode": "target_template",
                    "multimer": False,
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
            foundry_af2_refolding_test = st.selectbox(
                "AF2 complex refolding model",
                list(foundry_af2_options),
                format_func=lambda key: foundry_af2_options[key]["label"],
                index=0,
                key=f"foundry_full_af2_refolding_test_{source_key}",
            )
            foundry_complex_template_mode = str(foundry_af2_options[foundry_af2_refolding_test]["template_mode"])
            foundry_complex_multimer = bool(foundry_af2_options[foundry_af2_refolding_test]["multimer"])
            foundry_complex_num_recycles = int(foundry_af2_options[foundry_af2_refolding_test]["num_recycles"])
        else:
            foundry_boltz_template_mode = st.segmented_control(
                "Boltz2 complex template",
                ["target_template", "no_template"],
                selection_mode="single",
                default="target_template",
                format_func={"target_template": "Target template", "no_template": "No template"}.get,
                key=f"foundry_full_boltz_template_mode_{source_key}",
            )
            foundry_complex_template_mode = str(foundry_boltz_template_mode or "target_template")

        st.markdown("#### 4. Filtering And Analysis")
        analysis_cols = st.columns(5)
        with analysis_cols[0]:
            foundry_keep_top_n = st.number_input("Keep top results", min_value=1, max_value=10000, value=100, step=10, key=f"foundry_keep_top_n_{source_key}")
        with analysis_cols[1]:
            foundry_min_binder_plddt = st.number_input("Min binder pLDDT", min_value=0.0, max_value=100.0, value=70.0, step=1.0, key=f"foundry_min_binder_plddt_{source_key}")
        with analysis_cols[2]:
            foundry_max_ipae = st.number_input("Max iPAE", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key=f"foundry_max_ipae_{source_key}")
        with analysis_cols[3]:
            foundry_min_ipsae = st.number_input("Min ipSAE", min_value=0.0, max_value=1.0, value=0.0, step=0.05, key=f"foundry_min_ipsae_{source_key}")
        with analysis_cols[4]:
            foundry_max_binder_rmsd = st.number_input("Max binder RMSD", min_value=0.0, max_value=100.0, value=5.0, step=0.5, key=f"foundry_max_binder_rmsd_{source_key}")
        foundry_hotspot_filter_enabled = st.checkbox(
            "Filter final complexes by hotspot/site recovery",
            value=False,
            disabled=not bool(hotspots),
            key=f"foundry_hotspot_filter_enabled_{source_key}",
        )
        hotspot_filter_cols = st.columns(2)
        with hotspot_filter_cols[0]:
            foundry_min_final_hotspot_contact_fraction = st.number_input(
                "Final min hotspot contact fraction",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                disabled=not foundry_hotspot_filter_enabled,
                key=f"foundry_min_hotspot_contact_fraction_{source_key}",
            )
        with hotspot_filter_cols[1]:
            foundry_max_final_hotspot_distance = st.number_input(
                "Final max nearest hotspot distance",
                min_value=0.0,
                max_value=50.0,
                value=8.0,
                step=0.5,
                disabled=not foundry_hotspot_filter_enabled,
                key=f"foundry_max_hotspot_distance_{source_key}",
            )
        if not foundry_hotspot_filter_enabled:
            foundry_min_final_hotspot_contact_fraction = None
            foundry_max_final_hotspot_distance = None
    else:
        st.caption("Generation-only mode stops after RFdiffusion3 / Foundry. Downstream modules can be launched later from Campaigns.")

    _tool_payload_expander(
        "RFdiffusion3 / Foundry",
        {
            "docker_image": "ovoex-foundry-cu128:latest",
            "input_json": {
                "design_1": {
                    "dialect": 2,
                    "input": "target.pdb",
                    "contig": foundry_contig,
                    "is_non_loopy": bool(foundry_is_non_loopy),
                    "infer_ori_strategy": foundry_infer_ori_strategy,
                    "select_hotspots": _hotspot_atom_map(hotspots) if hotspots else None,
                }
            },
            "command_args": {
                "out_dir": "rfd3",
                "inputs": "rfd3_inputs_staged.json",
                "ckpt_path": "/weights/rfd3_latest.ckpt",
                "diffusion_batch_size": 1,
                "n_batches": int(foundry_num_designs),
                "inference_sampler.num_timesteps": int(foundry_timesteps),
                "skip_existing": False,
                "prevalidate_inputs": True,
            },
            "full_pipeline": {
                "enabled": foundry_full_pipeline,
                "sequence_design": {
                    "tool": "foundry_mpnn",
                    "number_of_batches": 1,
                    "batch_size": int(foundry_mpnn_sequences),
                    "model_type": foundry_mpnn_model_type,
                    "checkpoint_path": foundry_mpnn_checkpoint,
                },
                "monomer_refolding": {"tool": foundry_monomer_tool, "min_plddt": float(foundry_min_binder_plddt)},
                "complex_refolding": {
                    "tool": foundry_complex_tool,
                    "template_mode": foundry_complex_template_mode,
                    "multimer": foundry_complex_multimer,
                    "num_recycles": foundry_complex_num_recycles,
                },
                "analysis": {
                    "keep_top_n": int(foundry_keep_top_n),
                    "thresholds": {
                        "min_binder_plddt": float(foundry_min_binder_plddt),
                        "min_confidence": 0.0,
                        "min_iptm": 0.0,
                        "min_ipsae": float(foundry_min_ipsae),
                        "max_ipae": float(foundry_max_ipae),
                        "max_ipde": 20.0,
                        "max_binder_rmsd": float(foundry_max_binder_rmsd),
                        **(
                            {
                                "min_hotspot_contact_fraction": float(foundry_min_final_hotspot_contact_fraction),
                                "max_hotspot_distance": float(foundry_max_final_hotspot_distance),
                            }
                            if foundry_min_final_hotspot_contact_fraction is not None
                            and foundry_max_final_hotspot_distance is not None
                            else {}
                        ),
                    },
                },
            },
        },
    )
    foundry_run_label = "Run RFdiffusion3 full pipeline" if foundry_full_pipeline else "Run RFdiffusion3 generation"
    if st.button(foundry_run_label, type="primary", disabled=not target_chains):
        try:
            with st.spinner("Running RFdiffusion3 / Foundry..."):
                run_dir = run_rfdiffusion3_foundry(
                    target_pdb=target_pdb,
                    target_chains=target_chains,
                    binder_length=binder_length,
                    hotspots=hotspots,
                    num_designs=int(foundry_num_designs),
                    timesteps=int(foundry_timesteps),
                    contig=foundry_contig,
                    is_non_loopy=bool(foundry_is_non_loopy),
                    infer_ori_strategy=foundry_infer_ori_strategy,
                )
            st.success("RFdiffusion3 / Foundry job finished.")
            st.link_button("Open result", result_link("design", run_dir.name))
            if foundry_full_pipeline:
                steps = [
                    {
                        "module": "sequence_design",
                        "tool": "foundry_mpnn",
                        "params": {
                            "number_of_batches": 1,
                            "batch_size": int(foundry_mpnn_sequences),
                            "model_type": foundry_mpnn_model_type,
                            "checkpoint_path": foundry_mpnn_checkpoint,
                        },
                    },
                    {
                        "module": "monomer_refolding",
                        "tool": str(foundry_monomer_tool),
                        "params": {"min_plddt": float(foundry_min_binder_plddt)},
                    },
                    {
                        "module": "complex_refolding",
                        "tool": str(foundry_complex_tool),
                        "params": {
                            "require_monomer_success": True,
                            "template_mode": foundry_complex_template_mode,
                            "multimer": foundry_complex_multimer,
                            "num_recycles": foundry_complex_num_recycles,
                        },
                    },
                    {
                        "module": "analysis",
                        "tool": "ranking",
                        "params": {
                            "keep_top_n": int(foundry_keep_top_n),
                            "thresholds": {
                                "min_binder_plddt": float(foundry_min_binder_plddt),
                                "min_confidence": 0.0,
                                "min_iptm": 0.0,
                                "min_ipsae": float(foundry_min_ipsae),
                                "max_ipae": float(foundry_max_ipae),
                                "max_ipde": 20.0,
                                "max_binder_rmsd": float(foundry_max_binder_rmsd),
                                **(
                                    {
                                        "min_hotspot_contact_fraction": float(foundry_min_final_hotspot_contact_fraction),
                                        "max_hotspot_distance": float(foundry_max_final_hotspot_distance),
                                    }
                                    if foundry_min_final_hotspot_contact_fraction is not None
                                    and foundry_max_final_hotspot_distance is not None
                                    else {}
                                ),
                            },
                        },
                    },
                ]
                with st.spinner("Running Foundry MPNN, refolding, and analysis..."):
                    child_runs = run_lineage_steps(
                        campaign_name.strip() or f"RFdiffusion3 full pipeline from {run_dir.name}",
                        _source_from_run_dir(run_dir),
                        steps,
                    )
                if child_runs and read_json(child_runs[-1] / "result.json").get("success") is not True:
                    st.error("Full pipeline stopped on a failed step. Open the child job for logs.")
                else:
                    st.success("Full RFdiffusion3 pipeline finished.")
                _lineage_links(child_runs)
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
            "docker_image": "boltzgen:latest",
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
            with st.spinner("Running BoltzGen..."):
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
                )
            st.success("BoltzGen job finished.")
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
    first_chain = target_chains[0] if target_chains else "A"
    px_hotspots_by_chain: dict[str, list[int]] = {}
    for token in hotspots.split(",") if hotspots else []:
        try:
            px_hotspots_by_chain.setdefault(token[0], []).append(int(token[1:]))
        except ValueError:
            continue
    px_yaml_chains = {
        chain: {"hotspots": px_hotspots_by_chain[chain]} if px_hotspots_by_chain.get(chain) else "all"
        for chain in (target_chains or [first_chain])
    }
    _tool_payload_expander(
        "PXDesign",
        {
            "docker_image": "mnprot-pxdesign-cu128:latest",
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
            with st.spinner("Running PXDesign..."):
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
                )
            st.success("PXDesign job finished.")
            st.link_button("Open result", result_link("design", run_dir.name))
        except Exception as exc:
            st.error(str(exc))

with genie3_tab:
    st.caption("Current adapter uses the bundled Genie3 BinderBench PDL1 contract.")
    genie3_num_designs = st.number_input("Candidates", 1, 100, int(num_candidates), key=f"genie3_num_designs_{source_key}")
    _tool_payload_expander(
        "Genie3",
        {
            "docker_image": "mnprot-genie3-cu128:latest",
            "contract": "smoke_tests/genie3_pdl1_binder_smoke.yaml",
            "command_args": {
                "config": "/work/config/genie3_pdl1_binder_smoke.yaml",
                "num_devices": 1,
                "verbose": True,
            },
            "note": "The current Genie3 adapter does not yet convert the selected custom target into a Genie3 dataset item.",
        },
    )
    if st.button("Run Genie3 contract", type="primary"):
        try:
            with st.spinner("Running Genie3..."):
                run_dir = run_genie3(int(genie3_num_designs))
            st.success("Genie3 job finished.")
            st.link_button("Open result", result_link("design", run_dir.name))
        except Exception as exc:
            st.error(str(exc))

with protpardelle_tab:
    st.caption("Current adapter uses the bundled Protpardelle-1c PDL1 motif contract.")
    settings_cols = st.columns(2)
    with settings_cols[0]:
        protpardelle_num_designs = st.number_input("Candidates", 1, 100, int(num_candidates), key=f"protpardelle_num_designs_{source_key}")
    with settings_cols[1]:
        protpardelle_seed = st.number_input("Seed", 0, 999999, int(loaded_params.get("seed") or 7), key=f"protpardelle_seed_{source_key}")
    _tool_payload_expander(
        "Protpardelle-1c",
        {
            "docker_image": "mnprot-protpardelle-1c-cu128:latest",
            "contract": "smoke_tests/protpardelle_pdl1_smoke.yaml",
            "command_args": {
                "config": "/work/config/protpardelle_pdl1_smoke.yaml",
                "motif_dir": "/opt/protpardelle-1c/examples/motifs/bindcraft",
                "num_samples": int(protpardelle_num_designs),
                "num_mpnn_seqs": 0,
                "batch_size": 1,
                "seed": int(protpardelle_seed),
            },
        },
    )
    if st.button("Run Protpardelle-1c contract", type="primary"):
        try:
            with st.spinner("Running Protpardelle-1c..."):
                run_dir = run_protpardelle_1c(int(protpardelle_num_designs), int(protpardelle_seed))
            st.success("Protpardelle-1c job finished.")
            st.link_button("Open result", result_link("design", run_dir.name))
        except Exception as exc:
            st.error(str(exc))

with complexa_tab:
    st.caption("Current adapter uses the bundled Proteina-Complexa PDL1 search_binder_local_pipeline contract.")
    settings_cols = st.columns(3)
    with settings_cols[0]:
        complexa_run_name = st.text_input("Run name", value=str(loaded_params.get("run_name") or "mn_app_complexa"), key=f"complexa_run_name_{source_key}")
    with settings_cols[1]:
        complexa_steps = st.number_input("Generation steps", 1, 1000, int(loaded_params.get("n_steps") or 20), key=f"complexa_steps_{source_key}")
    with settings_cols[2]:
        complexa_replicas = st.number_input("Best-of-N replicas", 1, 100, int(loaded_params.get("replicas") or 2), key=f"complexa_replicas_{source_key}")
    _tool_payload_expander(
        "Proteina-Complexa",
        {
            "docker_image": "ovoex-proteina-complexa:latest",
            "contract": "configs/search_binder_local_pipeline.yaml with task 02_PDL1",
            "stages": ["generate", "filter", "evaluate", "analyze"],
            "command_args": {
                "run_name": complexa_run_name,
                "generation.task_name": "02_PDL1",
                "generation.dataloader.dataset.nres.nsamples": 1,
                "generation.search.best_of_n.replicas": int(complexa_replicas),
                "generation.args.nsteps": int(complexa_steps),
                "ckpt_name": "complexa.ckpt",
                "autoencoder_ckpt_path": "/workspace/protein-foundation-models/ckpts/complexa_ae.ckpt",
            },
        },
    )
    if st.button("Run Proteina-Complexa contract", type="primary"):
        try:
            with st.spinner("Running Proteina-Complexa..."):
                run_dir = run_proteina_complexa(str(complexa_run_name), int(complexa_steps), int(complexa_replicas))
            st.success("Proteina-Complexa job finished.")
            st.link_button("Open result", result_link("design", run_dir.name))
        except Exception as exc:
            st.error(str(exc))
