from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import streamlit as st

from mn_protein_design.app.components.molstar_viewer import molstar_custom_component, StructureVisualization
from mn_protein_design.core.residue_selection import ResidueSelection
from mn_protein_design.core.structures import (
    detect_nonstandard_residues,
    download_alphafold_db_pdb,
    download_pdb,
    filter_pdb_text,
    pdb_summary,
)
from mn_protein_design.runtime import app_home
from mn_protein_design.workflows.target_prep import prepare_target


st.title("Target Preparation")
st.caption("Create canonical target artifacts for downstream detection and design tasks. This first pass only trims chains by residue range; hotspot/interface cropping comes later.")

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


def _chain_options() -> list[str]:
    if not state["pdb_text"]:
        return []
    return [chain["chain_id"] for chain in pdb_summary(state["pdb_text"])["chains"]]


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


left, right = st.columns([0.9, 1.25], gap="large")

with left:
    st.subheader("1. Import")
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
        st.subheader("2. Select Chains")
        chain_options = [chain["chain_id"] for chain in summary["chains"]]
        state["selected_chains"] = st.multiselect(
            "Chains to keep",
            options=chain_options,
            default=[c for c in state["selected_chains"] if c in chain_options] or chain_options,
        )
        state["remove_waters"] = st.checkbox("Remove waters", value=state["remove_waters"])
        state["remove_hetero"] = st.checkbox("Remove hetero atoms", value=state["remove_hetero"])
        state["prepare_msa"] = st.checkbox(
            "Prepare shared target MSAs",
            value=state.get("prepare_msa", True),
            help=(
                "Caches one A3M per selected target chain in "
                "/mnt/db/reference_files/boltz_models/msa_repository. The same sequence-hashed "
                "cache is reused by practical/full AF3, ColabFold, Boltz-style, RF3, "
                "Protenix, and ESMFold2 workflows when supported."
            ),
        )
        nonstandard = detect_nonstandard_residues(_selected_text_for_nonstandard_detection())
        if nonstandard:
            st.warning(f"Detected {len(nonstandard)} nonstandard residue(s) in the selected target.")
            with st.expander("Nonstandard residues", expanded=True):
                st.dataframe(nonstandard, hide_index=True, width="stretch")
            known_mappable = [residue for residue in nonstandard if residue.get("known_mapping")]
            if known_mappable:
                mapped_names = sorted({f"{residue['resname']}->{residue['mapped_to']}" for residue in known_mappable})
                state["map_known_modified_residues"] = st.checkbox(
                    f"Repair known non-canonical residues ({', '.join(mapped_names)})",
                    value=state["map_known_modified_residues"],
                    help="Uses conservative mn-ligand style residue mappings before PDBFixer. For CAS, the output residue is CYS with only protein-compatible atoms kept.",
                )
            else:
                state["map_known_modified_residues"] = False
            state["replace_nonstandard_residues"] = st.checkbox(
                "Replace nonstandard residues with PDBFixer",
                value=state["replace_nonstandard_residues"],
                help="Creates target_repaired.pdb after known mappings, before cleaning/trimming. This does not add missing residues or minimize the structure.",
            )
        else:
            state["map_known_modified_residues"] = False
            state["replace_nonstandard_residues"] = False
            st.caption("No nonstandard residues detected in the selected target.")

        st.subheader("3. Trim Chains")
        st.caption("Trim each selected chain to one residue interval. Spatial cropping around interfaces or hotspots will be handled after detection.")
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
            with start_col:
                start = st.number_input(
                    f"{chain_id} start",
                    min_value=residues[0],
                    max_value=residues[-1],
                    value=start_default,
                    step=1,
                    key=f"target_prep_{chain_id}_start",
                )
            with end_col:
                end = st.number_input(
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
            ranges = [f"{s.chain_id}:{s.start}-{s.end}" for s in _residue_selections()]
            with st.spinner("Writing target artifacts..."):
                run_dir = prepare_target(
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
            st.success(f"Prepared target job {run_dir.name}")
            st.markdown(f"[Open result](/results?task_group=target-prep&run_id={run_dir.name})")

with right:
    st.subheader("Structure Preview")
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
