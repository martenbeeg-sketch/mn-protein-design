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
from mn_protein_design.core.jobs import read_json
from mn_protein_design.core.structures import filter_pdb_text
from mn_protein_design.workflows import design as design_workflow

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

ResidueId = tuple[str, int]


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
    return default_target_contig(target_pdb, target_chains, binder_length).replace("/0 ", ",/0,")


def _tool_payload_expander(tool_name: str, payload: dict) -> None:
    with st.expander(f"{tool_name} payload sent to algorithm", expanded=False):
        st.json(payload)


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

st.subheader("Common Campaign Settings")
target_chains = _target_chain_ids(target)
st.caption(f"Using all chains in the selected prepared structure: `{','.join(target_chains) or 'unknown'}`")
binder_length = st.text_input(
    "Binder length",
    value=str(loaded_params.get("binder_length") or "55-55"),
    help="Use a fixed length like 55 or a range like 55-80.",
    key=f"binder_length_{source_key}",
)
num_candidates = st.number_input(
    "Candidates",
    min_value=1,
    max_value=1000,
    value=int(loaded_params.get("num_designs") or loaded_params.get("number_of_final_designs") or loaded_params.get("max_trajectories") or 1),
    step=1,
    key=f"num_candidates_{source_key}",
)

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
    computed_contig = (
        str(loaded_params.get("contig"))
        if loaded_params.get("contig")
        else default_target_contig(target_pdb, target_chains, binder_length)
        if target_chains
        else ""
    )
    contig = st.text_input(
        "RFdiffusion contig",
        value=computed_contig,
        help="For binder design this is usually fixed target residues, then /0, then binder length.",
        key=f"rfdiffusion_contig_{source_key}",
    )
    settings_cols = st.columns(3)
    with settings_cols[0]:
        rfdiffusion_num_designs = st.number_input(
            "Backbone candidates",
            min_value=1,
            max_value=1000,
            value=int(loaded_params.get("num_designs") or num_candidates),
            step=1,
            key=f"rfdiffusion_num_designs_{source_key}",
        )
    with settings_cols[1]:
        model_options = ["Complex_base", "Complex_beta"]
        loaded_model = str(loaded_params.get("model_weights") or "Complex_base")
        model_weights = st.selectbox(
            "Model weights",
            model_options,
            index=model_options.index(loaded_model) if loaded_model in model_options else 0,
            key=f"model_weights_{source_key}",
        )
    with settings_cols[2]:
        timesteps = st.number_input(
            "Diffusion timesteps",
            min_value=1,
            max_value=200,
            value=int(loaded_params.get("timesteps") or 50),
            step=1,
            key=f"timesteps_{source_key}",
        )
    extra_run_parameters = st.text_input(
        "Extra RFdiffusion parameters",
        value=str(loaded_params.get("extra_run_parameters") or ""),
        placeholder="for example potentials.guiding_potentials=[]",
        key=f"extra_run_parameters_{source_key}",
    )
    with st.expander("RFdiffusion advanced parameters", expanded=False):
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

    with st.expander("ProteinMPNN and refolding settings for downstream steps", expanded=False):
        sequence_design_options = {
            "ligandmpnn": "LigandMPNN (ProteinMPNN weights)",
            "fastrelax": "ProteinMPNN-FastRelax",
        }
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
        refolding_options = [
            "",
            "af2_model_1_ptm_tt_3rec",
            "af2_model_1_ptm_tbt_3rec",
            "af2_model_1_ptm_ct_3rec",
            "af2_model_1_multimer_tt_3rec",
            "af2_model_1_multimer_tbt_3rec",
            "af2_model_1_multimer_ct_3rec",
        ]
        refolding_labels = {
            "": "No refolding",
            "af2_model_1_ptm_tt_3rec": "AF2 monomer model, target template",
            "af2_model_1_ptm_tbt_3rec": "AF2 monomer model, target + binder templates",
            "af2_model_1_ptm_ct_3rec": "AF2 monomer model, complex template",
            "af2_model_1_multimer_tt_3rec": "AF2 multimer, target template",
            "af2_model_1_multimer_tbt_3rec": "AF2 multimer, target + binder templates",
            "af2_model_1_multimer_ct_3rec": "AF2 multimer, complex template",
        }
        loaded_refolding = str(loaded_params.get("refolding_test") or "af2_model_1_multimer_tt_3rec")
        refolding_test = st.selectbox(
            "Refolding test",
            refolding_options,
            format_func=refolding_labels.get,
            index=refolding_options.index(loaded_refolding) if loaded_refolding in refolding_options else 1,
            help="Stored in the app pipeline contract for the downstream validation stage.",
            key=f"rfdiffusion_refolding_test_{source_key}",
        )
        st.caption(
            "AF2 monomer options still predict the binder-target complex using residue-index offsets; "
            "they are not binder-only monomer validation."
        )

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
    with st.expander("Editable files and final settings sent to RFdiffusion", expanded=True):
        st.caption("These editable copies are written into the run folder under artifacts before Docker starts.")
        edited_run_parameters = st.text_area(
            "rfdiffusion_run_parameters.txt",
            value=str(loaded_params.get("rfdiffusion_run_parameters") or generated_run_parameters),
            height=110,
            key=f"rfdiffusion_editable_run_parameters_{source_key}",
        )
        edited_target_text = st.text_area(
            "target.pdb",
            value=target_view_text,
            height=220,
            key=f"rfdiffusion_editable_target_pdb_{source_key}",
        )
        pipeline_payload = {
            "pipeline": "mn-protein-design-rfdiffusion",
            "execution_backend": execution_backend,
            "design_type": "binder",
            "rfdiffusion_input_pdb": "/work/artifacts/input/target.pdb",
            "rfdiffusion_num_designs": int(rfdiffusion_num_designs),
            "rfdiffusion_contig": contig,
            "hotspot": hotspots,
            "rfdiffusion_run_parameters": edited_run_parameters,
            "backbone_filters": backbone_filters or "none",
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
                "backbone_metrics",
                "sequence_design",
                "refolding_validation" if refolding_test else "skip_refolding_validation",
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
                "rfdiffusion_run_parameters": edited_run_parameters,
            },
        },
    )
    if st.button("Run RFdiffusion campaign", type="primary", disabled=not target_chains):
        try:
            with st.spinner("Running RFdiffusion campaign..."):
                run_dir = run_rfdiffusion_classic(
                    target_pdb=target_pdb,
                    target_chains=target_chains,
                    contig=contig,
                    binder_length=binder_length,
                    hotspots=hotspots,
                    num_designs=int(rfdiffusion_num_designs),
                    timesteps=int(timesteps),
                    model_weights=model_weights,
                    extra_run_parameters=extra_run_parameters,
                    partial_diffusion=bool(partial_diffusion),
                    contigmap_length=contigmap_length,
                    inpaint_seq=inpaint_seq,
                    backbone_filters=backbone_filters,
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
        except Exception as exc:
            st.error(str(exc))

with bindcraft_tab:
    settings_cols = st.columns(3)
    with settings_cols[0]:
        bindcraft_final_designs = st.number_input(
            "Desired final designs",
            min_value=1,
            max_value=100,
            value=int(loaded_params.get("number_of_final_designs") or num_candidates),
            step=1,
            help="BindCraft stops when this many accepted designs are found or when the time limit is reached.",
            key=f"bindcraft_final_designs_{source_key}",
        )
    with settings_cols[1]:
        bindcraft_time_limit = st.number_input(
            "Time limit seconds",
            min_value=60,
            max_value=14 * 24 * 3600,
            value=int(loaded_params.get("time_limit_seconds") or 900),
            step=60,
            key=f"bindcraft_time_limit_{source_key}",
        )
    with settings_cols[2]:
        bindcraft_max_trajectories = st.number_input(
            "Max trajectories",
            min_value=1,
            max_value=1000,
            value=int(loaded_params.get("max_trajectories") or num_candidates),
            step=1,
            help="Useful for quick trajectory-style design runs.",
            key=f"bindcraft_max_trajectories_{source_key}",
        )
    option_cols = st.columns(3)
    with option_cols[0]:
        bindcraft_single_af_model = st.checkbox(
            "Use only AF model 0",
            value=bool(loaded_params.get("single_af_model", True)),
            help="Matches the smoke-test speed setting. Disable for fuller native BindCraft behavior.",
            key=f"bindcraft_single_af_model_{source_key}",
        )
    with option_cols[1]:
        bindcraft_enable_mpnn = st.checkbox(
            "Enable BindCraft MPNN stage",
            value=bool(loaded_params.get("enable_mpnn", False)),
            help="Off gives faster trajectory candidates. On moves closer to native full BindCraft.",
            key=f"bindcraft_enable_mpnn_{source_key}",
        )
    with option_cols[2]:
        filter_options = [
            "no_filters.json",
            "default_filters.json",
            "relaxed_filters.json",
            "peptide_filters.json",
            "peptide_relaxed_filters.json",
        ]
        loaded_filter = str(loaded_params.get("filter_settings") or "no_filters.json")
        bindcraft_filter_settings = st.selectbox(
            "Filter settings",
            filter_options,
            index=filter_options.index(loaded_filter) if loaded_filter in filter_options else 0,
            key=f"bindcraft_filter_settings_{source_key}",
        )
    _tool_payload_expander(
        "BindCraft",
        {
            "docker_image": "ovo-bindcraft:latest",
            "input_json": {
                "design_path": "output",
                "starting_pdb": "target.pdb",
                "binder_name": "design",
                "chains": ",".join(target_chains),
                "target_hotspot_residues": hotspots,
                "lengths": binder_length,
                "number_of_final_designs": int(bindcraft_final_designs),
            },
            "advanced_settings_overrides": {
                "enable_mpnn": bool(bindcraft_enable_mpnn),
                "num_seqs": 1,
                "max_mpnn_sequences": 1,
                "max_trajectories": int(bindcraft_max_trajectories),
                "af_params_dir": "alphafold_models_path",
            },
            "filter_settings": bindcraft_filter_settings,
            "runtime": {
                "time_limit_seconds": int(bindcraft_time_limit),
                "single_af_model": bool(bindcraft_single_af_model),
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
                    number_of_final_designs=int(bindcraft_final_designs),
                    time_limit_seconds=int(bindcraft_time_limit),
                    single_af_model=bool(bindcraft_single_af_model),
                    enable_mpnn=bool(bindcraft_enable_mpnn),
                    max_trajectories=int(bindcraft_max_trajectories),
                    filter_settings=bindcraft_filter_settings,
                )
            st.success("BindCraft job finished.")
            st.link_button("Open result", result_link("design", run_dir.name))
        except Exception as exc:
            st.error(str(exc))

with foundry_tab:
    st.caption("Uses the selected prepared target and hotspots through Foundry staged JSON.")
    settings_cols = st.columns(2)
    with settings_cols[0]:
        foundry_num_designs = st.number_input(
            "Candidates",
            min_value=1,
            max_value=100,
            value=int(loaded_params.get("num_designs") or num_candidates),
            step=1,
            key=f"foundry_num_designs_{source_key}",
        )
    with settings_cols[1]:
        foundry_timesteps = st.number_input(
            "Timesteps",
            min_value=1,
            max_value=200,
            value=int(loaded_params.get("timesteps") or 50),
            step=1,
            key=f"foundry_timesteps_{source_key}",
        )
    foundry_contig = _rfdiffusion3_contig(target_pdb, target_chains, binder_length)
    _tool_payload_expander(
        "RFdiffusion3 / Foundry",
        {
            "docker_image": "ovoex-foundry-cu128:latest",
            "input_json": {
                "design_1": {
                    "dialect": 2,
                    "input": "target.pdb",
                    "contig": foundry_contig,
                    "is_non_loopy": True,
                    "infer_ori_strategy": "hotspots" if hotspots else "default",
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
        },
    )
    if st.button("Run RFdiffusion3 / Foundry campaign", type="primary", disabled=not target_chains):
        try:
            with st.spinner("Running RFdiffusion3 / Foundry..."):
                run_dir = run_rfdiffusion3_foundry(
                    target_pdb=target_pdb,
                    target_chains=target_chains,
                    binder_length=binder_length,
                    hotspots=hotspots,
                    num_designs=int(foundry_num_designs),
                    timesteps=int(foundry_timesteps),
                )
            st.success("RFdiffusion3 / Foundry job finished.")
            st.link_button("Open result", result_link("design", run_dir.name))
        except Exception as exc:
            st.error(str(exc))

with boltzgen_tab:
    st.caption("Uses the selected prepared target and hotspots in a BoltzGen protein-anything design-only spec.")
    settings_cols = st.columns(2)
    with settings_cols[0]:
        boltzgen_num_designs = st.number_input(
            "Candidates",
            min_value=1,
            max_value=100,
            value=int(loaded_params.get("num_designs") or num_candidates),
            step=1,
            key=f"boltzgen_num_designs_{source_key}",
        )
    with settings_cols[1]:
        boltzgen_sampling_steps = st.number_input(
            "Sampling steps",
            min_value=1,
            max_value=500,
            value=int(loaded_params.get("sampling_steps") or 20),
            step=1,
            key=f"boltzgen_sampling_steps_{source_key}",
        )
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
                "steps": "design",
                "num_designs": int(boltzgen_num_designs),
                "diffusion_batch_size": 1,
                "devices": 1,
                "sampling_steps": int(boltzgen_sampling_steps),
                "compile_pairformer": False,
                "compile_structure": False,
            },
        },
    )
    if st.button("Run BoltzGen campaign", type="primary", disabled=not target_chains):
        try:
            with st.spinner("Running BoltzGen..."):
                run_dir = run_boltzgen(
                    target_pdb=target_pdb,
                    target_chains=target_chains,
                    binder_length=binder_length,
                    hotspots=hotspots,
                    num_designs=int(boltzgen_num_designs),
                    sampling_steps=int(boltzgen_sampling_steps),
                )
            st.success("BoltzGen job finished.")
            st.link_button("Open result", result_link("design", run_dir.name))
        except Exception as exc:
            st.error(str(exc))

with pxdesign_tab:
    st.caption("Current adapter uses the bundled PXDesign PDL1 quick-start contract.")
    settings_cols = st.columns(3)
    with settings_cols[0]:
        pxdesign_num_designs = st.number_input("Candidates", 1, 100, int(num_candidates), key=f"pxdesign_num_designs_{source_key}")
    with settings_cols[1]:
        pxdesign_steps = st.number_input("Diffusion steps", 1, 1000, int(loaded_params.get("n_steps") or 400), key=f"pxdesign_steps_{source_key}")
    with settings_cols[2]:
        pxdesign_dtype = st.selectbox("dtype", ["bf16", "fp32"], index=0, key=f"pxdesign_dtype_{source_key}")
    pxdesign_use_msa = st.checkbox("Use MSA", value=bool(loaded_params.get("use_msa", False)), key=f"pxdesign_use_msa_{source_key}")
    _tool_payload_expander(
        "PXDesign",
        {
            "docker_image": "mnprot-pxdesign-cu128:latest",
            "contract": "bundled /opt/PXDesign/examples/PDL1_quick_start.yaml",
            "command_args": {
                "input": "/opt/PXDesign/examples/PDL1_quick_start.yaml",
                "output": "artifacts/raw/pxdesign",
                "N_sample": int(pxdesign_num_designs),
                "N_step": int(pxdesign_steps),
                "dtype": pxdesign_dtype,
                "use_msa": bool(pxdesign_use_msa),
                "load_checkpoint_dir": "/ref/pxdesign/release_data/checkpoint",
            },
        },
    )
    if st.button("Run PXDesign contract", type="primary"):
        try:
            with st.spinner("Running PXDesign..."):
                run_dir = run_pxdesign(int(pxdesign_num_designs), int(pxdesign_steps), pxdesign_dtype, bool(pxdesign_use_msa))
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
