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
from mn_protein_design.app.pages.common import gpu_run_panel, result_link
from mn_protein_design.core.jobs import collect_jobs, read_json
from mn_protein_design.core.structures import filter_pdb_text
from mn_protein_design.runtime import runs_root
from mn_protein_design.workflows.design import prepared_design_targets, target_label
from mn_protein_design.workflows.design_campaigns import (
    CANDIDATE_POOL_COVERAGE,
    DESIGN_CAMPAIGN_GROUP,
    ENGINE_LABELS,
    ENGINE_ORDER,
    create_design_campaign,
    create_design_campaign_collection,
)

ResidueId = tuple[str, int]


def _target_chain_ids(target: dict) -> list[str]:
    chains = target.get("chains") or []
    if chains and isinstance(chains[0], dict):
        return [str(row["chain_id"]) for row in chains]
    return [str(chain) for chain in chains]


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
    st.info("Prepare or crop a target before starting a design campaign.")
    st.stop()

target_tab, workflows_tab, harmonize_tab, refinement_tab, evaluation_tab, run_tab, results_tab = st.tabs(
    ["Target", "Vanilla Workflows", "Harmonize", "Sequence Refinement", "Evaluation", "Review & Run", "Results"]
)

labels = [target_label(row) for row in targets]
with target_tab:
    selected_label = st.selectbox("Prepared target", labels, key="design_campaign_target")
    target = targets[labels.index(selected_label)]
    target_pdb = Path(target["target_pdb"])
    available_chains = _target_chain_ids(target)
    selected_chains = st.multiselect(
        "Target chains",
        available_chains,
        default=available_chains,
        key=f"design_campaign_chains_{target.get('run_id', target_pdb.name)}",
    )
    target_cols = st.columns([2, 1, 1, 1, 1])
    with target_cols[0]:
        campaign_name = st.text_input("Campaign name", placeholder="PDL1 multi-engine 60aa")
    with target_cols[1]:
        binder_length = st.text_input("Binder length", value="60", help="Fixed length or range, for example 55 or 55-80.")
    with target_cols[2]:
        design_attempts = st.number_input("Design attempts", min_value=1, max_value=1000, value=2, step=1)
    with target_cols[3]:
        sequences_per_backbone = st.number_input(
            "Sequences per backbone", min_value=1, max_value=1000, value=1, step=1
        )
    with target_cols[4]:
        random_seed = st.number_input("Random seed", min_value=0, max_value=2147483647, value=0, step=1)
    target_text = target_pdb.read_text(errors="ignore")
    target_view_text = (
        filter_pdb_text(target_text, keep_chains=set(selected_chains))
        if selected_chains
        else target_text
    )
    hotspot_key = f"design_campaign_hotspots_{target_pdb}"
    hotspot_text_key = f"{hotspot_key}_text"
    st.session_state.setdefault(hotspot_key, set())
    st.session_state.setdefault(hotspot_text_key, _hotspot_text(set(st.session_state[hotspot_key])))
    active_hotspots = {
        residue for residue in set(st.session_state[hotspot_key]) if residue[0] in set(selected_chains)
    }
    viewer_value = molstar_custom_component(
        [
            StructureVisualization(
                pdb=target_view_text,
                color="chain-id",
                color_params={"palette": "pastel-1"},
                representation_type="cartoon",
                highlighted_selections=[f"{chain}{residue}" for chain, residue in sorted(active_hotspots)],
                chains=_hotspot_chains(active_hotspots),
            )
        ],
        key=f"design_campaign_target_viewer_{target.get('run_id', target_pdb.name)}_{','.join(selected_chains)}",
        height=560,
        show_controls=True,
        selection_mode=True,
    )
    clicked_hotspots = _viewer_residues(viewer_value, target_view_text)
    hotspot_buttons = st.columns([1, 1, 1, 3])
    if hotspot_buttons[0].button("Add clicked", disabled=not clicked_hotspots):
        st.session_state[hotspot_key] = active_hotspots | clicked_hotspots
        st.session_state[hotspot_text_key] = _hotspot_text(set(st.session_state[hotspot_key]))
        st.rerun()
    if hotspot_buttons[1].button("Remove clicked", disabled=not clicked_hotspots):
        st.session_state[hotspot_key] = active_hotspots - clicked_hotspots
        st.session_state[hotspot_text_key] = _hotspot_text(set(st.session_state[hotspot_key]))
        st.rerun()
    if hotspot_buttons[2].button("Clear", disabled=not active_hotspots):
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
    vanilla_engine_order = [engine for engine in ENGINE_ORDER if engine != "template_redesign"]
    selected_engines = st.multiselect(
        "Vanilla workflows",
        vanilla_engine_order,
        default=["rfdiffusion_classic", "rfdiffusion3_foundry", "boltzgen", "esmfold2_binder_design"],
        format_func=lambda value: ENGINE_LABELS[value],
    )
    continue_after_failure = st.checkbox("Continue campaign when one workflow fails", value=True)

    engine_configs: dict[str, dict] = {}
    if "rfdiffusion_classic" in selected_engines:
        with st.expander("RFdiffusion classic", expanded=True):
            cols = st.columns(4)
            rf_timesteps = cols[0].number_input("Diffusion timesteps", 1, 200, 50, key="dc_rf_timesteps")
            rf_model = cols[1].selectbox("Model weights", ["Complex_base", "Complex_beta"], key="dc_rf_model")
            rf_sequence_method = cols[2].selectbox(
                "Sequence design", ["ligandmpnn", "proteinmpnn"], key="dc_rf_sequence_method"
            )
            rf_backend = cols[3].selectbox("Execution backend", ["docker", "nextflow"], key="dc_rf_backend")
            rf_contig = st.text_input(
                "Contig override",
                value="",
                placeholder="Automatically derived from target chains and binder length",
                key="dc_rf_contig",
            )
            engine_configs["rfdiffusion_classic"] = {
                "timesteps": int(rf_timesteps),
                "model_weights": rf_model,
                "sequence_design_method": rf_sequence_method,
                "execution_backend": rf_backend,
                "contig": rf_contig,
                "analysis_keep_top_n": int(design_attempts),
            }

    if "bindcraft" in selected_engines:
        with st.expander("BindCraft", expanded=True):
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
            }

    if "rfdiffusion3_foundry" in selected_engines:
        with st.expander("RFdiffusion3 / Foundry", expanded=True):
            cols = st.columns(2)
            foundry_timesteps = cols[0].number_input("Diffusion steps", 1, 1000, 50, key="dc_foundry_timesteps")
            foundry_msa = cols[1].checkbox("Prepare target MSA", value=True, key="dc_foundry_msa")
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

    if "boltzgen" in selected_engines:
        with st.expander("BoltzGen", expanded=True):
            cols = st.columns(2)
            boltz_budget = cols[0].number_input(
                "Final budget", 1, 1000, min(2, int(design_attempts)), key="dc_boltz_budget"
            )
            boltz_steps = cols[1].number_input("Sampling steps", 1, 1000, 20, key="dc_boltz_steps")
            engine_configs["boltzgen"] = {
                "budget": min(int(boltz_budget), int(design_attempts)),
                "sampling_steps": int(boltz_steps),
            }

    if "pxdesign" in selected_engines:
        with st.expander("PXDesign", expanded=True):
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

    if "genie3" in selected_engines:
        with st.expander("Genie3", expanded=True):
            cols = st.columns(4)
            genie_condition = cols[0].selectbox(
                "Conditioning",
                ["extended", "hotspot", "common", "iter_common", "iter_common_prob"],
                key="dc_genie_condition",
            )
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

    if "esmfold2_binder_design" in selected_engines:
        with st.expander("ESMFold2 binder design", expanded=True):
            cols = st.columns(3)
            esm_chain = cols[0].selectbox("Target chain", selected_chains or available_chains, key="dc_esm_chain")
            esm_steps = cols[1].number_input("Optimization steps", 1, 10000, 150, key="dc_esm_steps")
            esm_lr = cols[2].number_input(
                "Learning rate", min_value=0.001, max_value=10.0, value=0.1, step=0.01, format="%.3f", key="dc_esm_lr"
            )
            esm_options = st.columns(2)
            esm_compile = esm_options[0].checkbox("Compile model", value=False, key="dc_esm_compile")
            esm_checkpoint = esm_options[1].checkbox("Checkpoint language model", value=False, key="dc_esm_checkpoint")
            engine_configs["esmfold2_binder_design"] = {
                "target_chain": esm_chain,
                "optimization_steps": int(esm_steps),
                "learning_rate": float(esm_lr),
                "compile_model": bool(esm_compile),
                "checkpoint_lm": bool(esm_checkpoint),
            }

    if "protpardelle_1c" in selected_engines:
        with st.expander("Protpardelle-1c", expanded=True):
            cols = st.columns(4)
            prot_model = cols[0].selectbox(
                "Model preset",
                ["cc83", "cc95"],
                format_func={"cc83": "cc83 epoch 2616", "cc95": "cc95 epoch 3490"}.get,
                key="dc_prot_model",
            )
            prot_batch = cols[1].number_input("Sampling batch size", 1, 128, 1, key="dc_prot_batch")
            prot_step_scale = cols[2].number_input("Step scale", 0.1, 5.0, 1.2, 0.1, key="dc_prot_step")
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

    if "proteina_complexa" in selected_engines:
        with st.expander("Proteina-Complexa", expanded=True):
            cols = st.columns(3)
            complexa_steps = cols[0].number_input("Generation steps", 1, 1000, 400, key="dc_complexa_steps")
            complexa_replicas = cols[1].number_input("Best-of-N replicas", 1, 100, 2, key="dc_complexa_replicas")
            complexa_batch = cols[2].number_input("GPU batch size", 1, 16, 1, key="dc_complexa_batch")
            engine_configs["proteina_complexa"] = {
                "n_steps": int(complexa_steps),
                "replicas": int(complexa_replicas),
                "batch_size": int(complexa_batch),
            }

with harmonize_tab:
    common_validation_enabled = st.checkbox(
        "Run common BindCraft-compatible validation",
        value=True,
        help="Runs AF2 target-template prediction, ipSAE, the common confidence gate, then PyRosetta relaxation and interface scoring.",
    )
    common_cols = st.columns(4)
    common_cols[0].text_input(
        "Validation profile",
        value="BindCraft default target-template",
        disabled=True,
    )
    common_recycles = common_cols[1].number_input(
        "AF2 recycles", min_value=1, max_value=50, value=3, disabled=not common_validation_enabled
    )
    common_min_ipsae = common_cols[2].number_input(
        "Minimum ipSAE",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        disabled=not common_validation_enabled,
        help="Zero calculates ipSAE without rejecting on it.",
    )
    common_pyrosetta_nprocs = common_cols[3].number_input(
        "PyRosetta processes",
        min_value=1,
        max_value=64,
        value=4,
        disabled=not common_validation_enabled,
    )
    harmonize_cols = st.columns(3)
    survivors_per_engine = harmonize_cols[0].number_input(
        "Maximum survivors per workflow", min_value=1, max_value=1000, value=2, step=1
    )
    passing_only = harmonize_cols[1].checkbox(
        "Prefer native passing designs",
        value=True,
        disabled=common_validation_enabled,
        help="Used only when common validation is disabled.",
    )
    keep_best_failed = harmonize_cols[2].checkbox("Keep best available when none pass", value=True)
    st.dataframe(
        pd.DataFrame(
            [
                {"stage": 1, "criterion": "Two AF2 target-template models: complex binder pLDDT >= 80"},
                {"stage": 1, "criterion": "Both models: pTM >= 0.55 and ipTM >= 0.50"},
                {"stage": 1, "criterion": "Both models: interface PAE <= 10.85"},
                {"stage": 1, "criterion": "Both binder-only models: pLDDT >= 80 and RMSD <= 3.5 A"},
                {"stage": 1, "criterion": "ipSAE calculated before structural scoring"},
                {"stage": 2, "criterion": "DSSP binder/interface secondary-structure percentages"},
                {"stage": 2, "criterion": "Both models and average: binder loop <= 90%"},
                {"stage": 2, "criterion": "Both models and average: design-reference hotspot RMSD <= 6 A"},
                {"stage": 2, "criterion": "PyRosetta relax and BindCraft-compatible interface filters"},
                {"stage": 3, "criterion": "Rank common passes, then retain per-workflow survivors"},
            ]
        ),
        hide_index=True,
        width="stretch",
    )
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

with refinement_tab:
    refinement_defaults = {
        "interface": "redesign_non_interface",
        "all": "redesign_all",
        "selected_unlocked": "redesign_selected_unlocked",
        "selected_locked": "redesign_selected_locked",
    }
    template_refinement_mode = refinement_defaults.get(template_lock_mode, "redesign_non_interface")
    refinement_mode = st.segmented_control(
        "Sequence refinement",
        ["none", "redesign_non_interface", "redesign_all", "redesign_selected_unlocked", "redesign_selected_locked"],
        default=template_refinement_mode if template_enabled else "none",
        format_func={
            "none": "None",
            "redesign_non_interface": "Keep interface",
            "redesign_all": "Redesign full binder",
            "redesign_selected_unlocked": "Redesign selected",
            "redesign_selected_locked": "Lock selected",
        }.get,
        disabled=template_enabled,
    )
    refinement_engine = st.selectbox(
        "Sequence engine",
        ["ProteinMPNN", "LigandMPNN", "Soluble ProteinMPNN"],
        disabled=refinement_mode == "none",
    )
    refinement_cols = st.columns(3)
    refinement_sequences = refinement_cols[0].number_input(
        "Sequences per structure", 1, 1000, 2, disabled=refinement_mode == "none"
    )
    refinement_temperature = refinement_cols[1].number_input(
        "Sampling temperature",
        0.0001,
        2.0,
        0.1,
        step=0.05,
        format="%.4f",
        disabled=refinement_mode == "none",
    )
    refinement_omit_aas = refinement_cols[2].text_input(
        "Omit amino acids",
        "CX",
        disabled=refinement_mode == "none",
    )
    if refinement_mode == "redesign_non_interface":
        st.caption(
            "Binder residues within 5 Å of the target are fixed for each harmonized complex."
        )
    if template_enabled and template_lock_mode in {"selected_unlocked", "selected_locked"}:
        st.caption("Selected template residues are forwarded to LigandMPNN as fixed residues.")

with evaluation_tab:
    evaluation_mode = st.segmented_control(
        "Validation mode",
        ["none", "metrics_only", "refold_and_metrics"],
        default="none",
        format_func={"none": "None", "metrics_only": "Metrics only", "refold_and_metrics": "Refold + metrics"}.get,
    )
    evaluation_cols = st.columns(4)
    evaluation_ipsae = evaluation_cols[0].checkbox("ipSAE", value=True, disabled=evaluation_mode == "none")
    evaluation_rosetta = evaluation_cols[1].checkbox("Rosetta", value=False, disabled=evaluation_mode == "none")
    evaluation_pymol = evaluation_cols[2].checkbox("PyMOL", value=False, disabled=evaluation_mode == "none")
    evaluation_refolder = evaluation_cols[3].selectbox(
        "Refolder",
        ["AF3", "ESMFold2", "Boltz-2", "RF3"],
        disabled=evaluation_mode != "refold_and_metrics",
    )
    evaluation_settings = st.columns(5)
    evaluation_recycles = evaluation_settings[0].number_input(
        "Recycles / loops",
        1,
        48,
        10,
        key="design_campaign_evaluation_recycles",
        disabled=evaluation_mode != "refold_and_metrics",
    )
    evaluation_steps = evaluation_settings[1].number_input(
        "Sampling / diffusion steps",
        1,
        1000,
        68,
        key="design_campaign_evaluation_steps",
        disabled=evaluation_mode != "refold_and_metrics",
    )
    evaluation_samples = evaluation_settings[2].number_input(
        "Samples",
        1,
        20,
        3,
        key="design_campaign_evaluation_samples",
        disabled=evaluation_mode != "refold_and_metrics",
    )
    evaluation_target_msa = evaluation_settings[3].checkbox(
        "Use target MSA",
        value=evaluation_refolder != "ESMFold2",
        key="design_campaign_evaluation_target_msa",
        disabled=evaluation_mode != "refold_and_metrics",
    )
    evaluation_rosetta_nprocs = evaluation_settings[4].number_input(
        "PyRosetta processes",
        1,
        64,
        4,
        key="design_campaign_evaluation_rosetta_nprocs",
        disabled=evaluation_mode == "none" or not evaluation_rosetta,
    )
    if evaluation_mode == "metrics_only" and evaluation_ipsae:
        st.caption("ipSAE requires a compatible PAE artifact for the existing structure.")

with run_tab:
    execution_rows = []
    if "rfdiffusion_classic" in selected_engines:
        execution_rows.append(
            {
                "workflow": "RFdiffusion classic",
                "generation workload": f"{design_attempts} diffusion backbones",
                "sequence workload": f"{sequences_per_backbone} MPNN sequences per backbone",
                "native final limit": f"top {design_attempts} after refolding/analysis",
            }
        )
    if "bindcraft" in selected_engines:
        execution_rows.append(
            {
                "workflow": "BindCraft",
                "generation workload": f"up to {design_attempts} trajectories",
                "sequence workload": f"{sequences_per_backbone} MPNN sequences per trajectory",
                "native final limit": (
                    f"{design_attempts} completed designs with native filters disabled"
                    if common_validation_enabled
                    else str(engine_configs["bindcraft"]["number_of_final_designs"])
                ),
            }
        )
    if "rfdiffusion3_foundry" in selected_engines:
        execution_rows.append(
            {
                "workflow": "RFdiffusion3 / Foundry",
                "generation workload": f"{design_attempts} diffusion designs",
                "sequence workload": f"{sequences_per_backbone} MPNN sequences per backbone",
                "native final limit": "native pipeline",
            }
        )
    if "boltzgen" in selected_engines:
        execution_rows.append(
            {
                "workflow": "BoltzGen",
                "generation workload": f"{design_attempts} generated designs",
                "sequence workload": "native sequence-design pipeline",
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
                "sequence workload": "native PXDesign pipeline",
                "native final limit": engine_configs["pxdesign"]["preset"],
            }
        )
    if "genie3" in selected_engines:
        execution_rows.append(
            {
                "workflow": "Genie3",
                "generation workload": f"{design_attempts} generated structures",
                "sequence workload": f"{sequences_per_backbone} inverse-folded sequences per structure",
                "native final limit": "native folding evaluation",
            }
        )
    if "esmfold2_binder_design" in selected_engines:
        execution_rows.append(
            {
                "workflow": "ESMFold2 binder design",
                "generation workload": f"{design_attempts} optimization starts",
                "sequence workload": "one optimized sequence per start",
                "native final limit": "all completed starts",
            }
        )
    if "protpardelle_1c" in selected_engines:
        execution_rows.append(
            {
                "workflow": "Protpardelle-1c",
                "generation workload": f"{design_attempts} scaffold samples",
                "sequence workload": f"{sequences_per_backbone} MPNN sequences per scaffold",
                "native final limit": "native ESMFold consistency filters",
            }
        )
    if "proteina_complexa" in selected_engines:
        execution_rows.append(
            {
                "workflow": "Proteina-Complexa",
                "generation workload": f"{design_attempts} requested designs",
                "sequence workload": "native Complexa pipeline",
                "native final limit": f"best of {engine_configs['proteina_complexa']['replicas']} replicas",
            }
        )
    review_rows = [
        {"setting": "Target", "value": target_pdb.name},
        {"setting": "Chains", "value": ", ".join(selected_chains)},
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
        {"setting": "Sequences per backbone", "value": str(sequences_per_backbone)},
        {"setting": "Random seed", "value": str(random_seed)},
        {"setting": "Workflows", "value": ", ".join(ENGINE_LABELS[engine] for engine in selected_engines)},
        {"setting": "Survivors", "value": f"{survivors_per_engine} per workflow"},
        {
            "setting": "Common validation",
            "value": (
                f"AF2 target-template, {common_recycles} recycles; ipSAE >= {common_min_ipsae:g}; "
                f"PyRosetta x{common_pyrosetta_nprocs}"
                if common_validation_enabled
                else "Disabled; native workflow metrics"
            ),
        },
        {
            "setting": "Sequence refinement",
            "value": (
                f"{refinement_mode}; {refinement_engine}; {int(refinement_sequences)} sequences; "
                f"T={float(refinement_temperature):g}; omit {refinement_omit_aas or 'none'}"
                if refinement_mode != "none"
                else "Disabled"
            ),
        },
        {
            "setting": "Evaluation",
            "value": (
                f"{evaluation_mode}; "
                + (
                    f"{evaluation_refolder}, {int(evaluation_recycles)} recycles, "
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
            survivors_per_engine=int(survivors_per_engine),
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
                "num_recycles": int(evaluation_recycles),
                "num_sampling_steps": int(evaluation_steps),
                "num_samples": int(evaluation_samples),
                "use_target_msa": bool(evaluation_target_msa),
                "pyrosetta_nprocs": int(evaluation_rosetta_nprocs),
            },
            gpu_device=gpu_device,
        )
        st.success(f"Campaign queued: {run_dir.name}")
        st.link_button("Open campaign result", result_link(DESIGN_CAMPAIGN_GROUP, run_dir.name))

with results_tab:
    if st.button("Refresh", key="refresh_design_campaigns"):
        st.rerun()
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
