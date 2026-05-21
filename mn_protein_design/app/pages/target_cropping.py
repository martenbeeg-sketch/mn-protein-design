from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from mn_protein_design.app.components.molstar_viewer import (
    ChainVisualization,
    StructureVisualization,
    molstar_custom_component,
)
from mn_protein_design.core.structures import (
    filter_pdb_to_residues,
    pdb_summary,
    residues_on_plane_sides,
    residues_within_spheres,
)
from mn_protein_design.workflows.detection import completed_target_jobs, target_label
from mn_protein_design.workflows.target_crop import run_target_crop


st.title("Target Cropping")
st.caption(
    "Build an editable residue selection from Mol* clicks and sphere expansions. "
    "Confirm the selection to preview the exact crop before creating the job."
)


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------

ResidueId = tuple[str, int]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _pdb_residue_index_map(pdb_text: str) -> dict[str, dict[int, int]]:
    """
    Build chain-specific mapping from sequential residue index to real PDB residue number.

    Example:
    If chain A starts at PDB residue 150:
        mapping["A"][1] = 150
        mapping["A"][2] = 151
        mapping["A"][3] = 152

    This fixes Mol* components that return sequence indices instead of PDB residue IDs.
    """
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

        insertion_code = line[26].strip()
        residue_key = (resseq, insertion_code)

        if chain not in residues_by_chain:
            residues_by_chain[chain] = []
            last_seen_by_chain[chain] = None

        if last_seen_by_chain[chain] != residue_key:
            residues_by_chain[chain].append(resseq)
            last_seen_by_chain[chain] = residue_key

    for chain, pdb_residue_numbers in residues_by_chain.items():
        mapping[chain] = {
            sequence_index: pdb_residue_number
            for sequence_index, pdb_residue_number in enumerate(
                pdb_residue_numbers,
                start=1,
            )
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
    """
    Convert Mol* component return value into real PDB residue IDs:
    {(chain_id, pdb_residue_number), ...}

    Some Mol* wrappers return sequence indices starting at 1 instead of PDB residue
    numbers. For a PDB chain that starts at 150, Mol* may return 1 for the first
    residue. This function maps that back to 150.
    """
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

            # OVO's Mol* component returns auth_seq_id, i.e. the PDB residue
            # number. Only fall back to sequence-index mapping if the returned
            # number is not present as a real residue number in this chain.
            if raw_residue in real_residue_numbers:
                pdb_residue = raw_residue
            else:
                pdb_residue = chain_map.get(raw_residue, raw_residue)

            residues.add((chain, pdb_residue))

    return residues


def _compress_selection_for_highlight(residues: set[ResidueId]) -> list[str]:
    """
    Format residues for Mol* highlighting.

    Example:
    A45
    A45-80
    """
    compressed: list[str] = []
    by_chain: dict[str, list[int]] = {}

    for chain, residue in sorted(residues):
        by_chain.setdefault(chain, []).append(residue)

    for chain, values in sorted(by_chain.items()):
        values = sorted(set(values))
        if not values:
            continue

        start = prev = values[0]

        for residue in values[1:]:
            if residue == prev + 1:
                prev = residue
                continue

            compressed.append(
                f"{chain}{start}" if start == prev else f"{chain}{start}-{prev}"
            )
            start = prev = residue

        compressed.append(
            f"{chain}{start}" if start == prev else f"{chain}{start}-{prev}"
        )

    return compressed


def _compress_selection_for_text(residues: set[ResidueId]) -> str:
    """
    Format final residues as manual crop text for run_target_crop.

    Example:
    A:45
    A:45-80
    """
    chunks: list[str] = []
    by_chain: dict[str, list[int]] = {}

    for chain, residue in sorted(residues):
        by_chain.setdefault(chain, []).append(residue)

    for chain, values in sorted(by_chain.items()):
        values = sorted(set(values))
        if not values:
            continue

        start = prev = values[0]

        for residue in values[1:]:
            if residue == prev + 1:
                prev = residue
                continue

            chunks.append(
                f"{chain}:{start}" if start == prev else f"{chain}:{start}-{prev}"
            )
            start = prev = residue

        chunks.append(
            f"{chain}:{start}" if start == prev else f"{chain}:{start}-{prev}"
        )

    return ", ".join(chunks)


def _chain_visualizations_for_residues(
    active_residues: set[ResidueId],
    preview_residues: set[ResidueId] | None = None,
) -> list[ChainVisualization] | None:
    preview_residues = set(preview_residues or set()) - set(active_residues)
    if not active_residues and not preview_residues:
        return None

    visualizations: list[ChainVisualization] = []

    active_by_chain: dict[str, list[int]] = {}
    for chain, residue in sorted(active_residues):
        active_by_chain.setdefault(chain, []).append(residue)

    for chain, values in sorted(active_by_chain.items()):
        visualizations.append(
            ChainVisualization(
                chain_id=chain,
                color="uniform",
                color_params={"value": "0x2563eb"},
                representation_type="cartoon+ball-and-stick",
                residues=sorted(set(values)),
                label="Active crop residues",
            )
        )

    preview_by_chain: dict[str, list[int]] = {}
    for chain, residue in sorted(preview_residues):
        preview_by_chain.setdefault(chain, []).append(residue)

    for chain, values in sorted(preview_by_chain.items()):
        visualizations.append(
            ChainVisualization(
                chain_id=chain,
                color="uniform",
                color_params={"value": "0x22c55e"},
                representation_type="cartoon+ball-and-stick",
                residues=sorted(set(values)),
                label="Plane side preview",
            )
        )

    return visualizations


def _format_residue_labels(residues: set[ResidueId]) -> str:
    if not residues:
        return "None"

    return ", ".join(f"{chain}{residue}" for chain, residue in sorted(residues))


def _plane_preview_residues(state: dict) -> set[ResidueId]:
    sides = state.get("plane_sides") or {}
    side_key = state.get("plane_side", "side_a")
    return set(sides.get(side_key, set()))


def _set_plane_preview(state: dict) -> None:
    state["plane_preview_residues"] = _plane_preview_residues(state)


def _clear_plane_preview(state: dict) -> None:
    state["plane_residues"] = set()
    state["plane_sides"] = {}
    state["plane_info"] = {}
    state["plane_preview_residues"] = set()


def _count_by_chain(residues: set[ResidueId]) -> dict[str, int]:
    counts: dict[str, int] = {}

    for chain, _residue in residues:
        counts[chain] = counts.get(chain, 0) + 1

    return dict(sorted(counts.items()))


def _format_chain_counts(residues: set[ResidueId]) -> str:
    if not residues:
        return "None"

    return ", ".join(
        f"{chain}: {count}" for chain, count in _count_by_chain(residues).items()
    )


def _push_history(state: dict) -> None:
    """
    Store previous active selection for undo.
    """
    state["history"].append(set(state["active_residues"]))

    if len(state["history"]) > 30:
        state["history"] = state["history"][-30:]


def _reset_editor_state(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "sphere_radius_angstrom": 8.0,
        "plane_tolerance_angstrom": 0.5,
        "plane_side": "side_a",
        "plane_residues": set(),
        "plane_sides": {},
        "plane_info": {},
        "plane_preview_residues": set(),
        "active_residues": set(),
        "confirmed_draft_residues": set(),
        "history": [],
        "viewer_key_nonce": 0,
        "debug_mapping": False,
    }


def _rerun_viewer() -> None:
    """
    Force Mol* component reload after Python-side selection changes.
    """
    st.session_state["target_crop_flexible_editor"]["viewer_key_nonce"] += 1
    st.rerun()


def _sequence_indices_for_residues(residues: set[ResidueId], residue_index_map: dict[str, dict[int, int]]) -> list[int]:
    pdb_to_sequence_index: dict[tuple[str, int], int] = {}
    for chain, chain_map in residue_index_map.items():
        for sequence_index, pdb_residue_number in chain_map.items():
            pdb_to_sequence_index[(chain, pdb_residue_number)] = sequence_index - 1
    return sorted(
        {
            pdb_to_sequence_index[residue]
            for residue in residues
            if residue in pdb_to_sequence_index
        }
    )


# ---------------------------------------------------------------------
# Target loading
# ---------------------------------------------------------------------

targets = completed_target_jobs()
if not targets:
    st.info("Prepare a target first, then return here to crop it.")
    st.stop()

target = st.selectbox("Prepared target", options=targets, format_func=target_label)

target_pdb = Path(target["target_pdb"])
if not target_pdb.exists():
    st.error(f"Prepared target file is missing: {target_pdb}")
    st.stop()

pdb_text = target_pdb.read_text(errors="ignore")
summary = pdb_summary(pdb_text)
residue_index_map = _pdb_residue_index_map(pdb_text)
residue_number_set = _pdb_residue_number_set(pdb_text)


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------

state = st.session_state.setdefault(
    "target_crop_flexible_editor",
    _reset_editor_state(target["run_id"]),
)

if state.get("run_id") != target["run_id"]:
    st.session_state["target_crop_flexible_editor"] = _reset_editor_state(
        target["run_id"]
    )
    state = st.session_state["target_crop_flexible_editor"]

for key, value in {
    "sphere_radius_angstrom": 8.0,
    "plane_tolerance_angstrom": 0.5,
    "plane_side": "side_a",
    "plane_residues": set(),
    "plane_sides": {},
    "plane_info": {},
    "plane_preview_residues": set(),
    "active_residues": set(),
    "confirmed_draft_residues": set(),
    "history": [],
    "viewer_key_nonce": 0,
    "debug_mapping": False,
}.items():
    state.setdefault(key, value)


# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------

left, right = st.columns([0.82, 1.45], gap="large")


# ---------------------------------------------------------------------
# Left controls
# ---------------------------------------------------------------------

with left:
    st.subheader("Selection Tools")

    state["sphere_radius_angstrom"] = st.number_input(
        "Sphere radius around clicked residues (Å)",
        min_value=1.0,
        max_value=40.0,
        value=float(state["sphere_radius_angstrom"]),
        step=0.5,
        help=(
            "Click one or more residues in Mol*, then add a sphere around those "
            "clicked residues. You can repeat this with different residues."
        ),
    )

    st.info(
        "Workflow: click residues in Mol*, add them or add a sphere, optionally "
        "remove clicked residues, then confirm the crop candidate."
    )


# ---------------------------------------------------------------------
# Main Mol* viewer
# ---------------------------------------------------------------------

with right:
    st.subheader("Full Structure Editor")

    metric_cols = st.columns(5)
    metric_cols[0].metric("Chains", len(summary["chains"]))
    metric_cols[1].metric(
        "Residues",
        sum(chain["residue_count"] for chain in summary["chains"]),
    )
    metric_cols[2].metric("Atoms", summary["atom_count"])
    metric_cols[3].metric("Waters", summary["water_count"])
    metric_cols[4].metric("Active selected", len(state["active_residues"]))

    st.caption(
        "The full structure remains visible. Active crop residues are blue; plane previews are green."
    )

    viewer_value = molstar_custom_component(
        structures=[
            StructureVisualization(
                pdb=pdb_text,
                color="chain-id",
                representation_type="cartoon+ball-and-stick",
                chains=_chain_visualizations_for_residues(
                    state["active_residues"],
                    state["plane_preview_residues"],
                ),
            )
        ],
        key=(
            f"crop_flexible_editor_"
            f"{target['run_id']}_"
            f"{state['viewer_key_nonce']}_"
            f"{len(state['active_residues'])}_"
            f"{len(state['plane_preview_residues'])}_"
            f"{state.get('plane_side', 'side_a')}"
        ),
        height=720,
        show_controls=True,
        selection_mode=True,
        sequence_highlight_color="#2563eb" if state["active_residues"] else None,
        sequence_highlight_indices=_sequence_indices_for_residues(state["active_residues"], residue_index_map),
        download_filename=f"{target['target_name']}_crop_editor",
    )


# ---------------------------------------------------------------------
# Current Mol* temporary selection
# ---------------------------------------------------------------------

viewer_selected = _viewer_residues(viewer_value, residue_index_map, residue_number_set)

# Candidate uses both persistent active selection and current Mol* selection.
current_candidate = set(state["active_residues"]) | set(viewer_selected)


# ---------------------------------------------------------------------
# Selection actions
# ---------------------------------------------------------------------

with left:
    st.subheader("Current Mol* Input")

    st.write(f"Currently clicked in Mol*: `{len(viewer_selected)}`")

    if viewer_selected:
        st.caption("Mol* input by chain: " + _format_chain_counts(viewer_selected))
    else:
        st.caption("No current Mol* click selection.")

    st.subheader("Active Selection")

    st.write(f"Active residues: `{len(state['active_residues'])}`")

    if state["active_residues"]:
        st.caption(
            "Active by chain: " + _format_chain_counts(state["active_residues"])
        )
    else:
        st.caption("No active residues yet.")

    col_add, col_remove = st.columns(2)

    if col_add.button(
        "Add clicked residues",
        disabled=not bool(viewer_selected),
        use_container_width=True,
        help="Adds the current Mol* clicked residues to the persistent active selection.",
    ):
        _push_history(state)
        state["active_residues"] = set(state["active_residues"]) | set(viewer_selected)
        _rerun_viewer()

    if col_remove.button(
        "Remove clicked residues",
        disabled=not bool(viewer_selected),
        use_container_width=True,
        help="Removes the current Mol* clicked residues from the persistent active selection.",
    ):
        _push_history(state)
        state["active_residues"] = set(state["active_residues"]) - set(viewer_selected)
        _rerun_viewer()

    st.divider()

    if st.button(
        "Add sphere around clicked residues",
        type="primary",
        disabled=not bool(viewer_selected),
        use_container_width=True,
        help=(
            "Adds all residues within the radius around the current Mol* clicked residues. "
            "To add another sphere, click another residue and press this again."
        ),
    ):
        sphere_residues = residues_within_spheres(
            pdb_text,
            set(viewer_selected),
            float(state["sphere_radius_angstrom"]) * 2.0,
        )

        _push_history(state)

        before_count = len(state["active_residues"])

        state["active_residues"] = (
            set(state["active_residues"])
            | set(viewer_selected)
            | set(sphere_residues)
        )

        after_count = len(state["active_residues"])

        if after_count == before_count:
            st.warning(
                "Sphere did not add new residues. Try a larger radius or another residue."
            )
        else:
            st.success(f"Added {after_count - before_count} residues by sphere.")

        _rerun_viewer()

    st.caption(
        "Sphere expansion uses the CA atom of each currently clicked residue as center, "
        "falling back to the residue centroid if CA is not present. "
        "This allows you to add one sphere, then click another residue and add another sphere."
    )

    st.divider()

    st.subheader("Plane Slice")

    new_plane_tolerance = st.number_input(
        "Plane tolerance (Å)",
        min_value=0.0,
        max_value=10.0,
        value=float(state["plane_tolerance_angstrom"]),
        step=0.1,
        help="Residues very close to the plane are included on both sides.",
    )

    if new_plane_tolerance != state["plane_tolerance_angstrom"]:
        state["plane_tolerance_angstrom"] = float(new_plane_tolerance)
        if state.get("plane_residues"):
            try:
                sides, plane_info = residues_on_plane_sides(
                    pdb_text,
                    set(state["plane_residues"]),
                    float(state["plane_tolerance_angstrom"]),
                )
                state["plane_sides"] = sides
                state["plane_info"] = plane_info
                _set_plane_preview(state)
            except ValueError as exc:
                _clear_plane_preview(state)
                st.warning(str(exc))
        _rerun_viewer()

    if st.button(
        "Define plane from clicked residues",
        disabled=len(viewer_selected) != 3,
        use_container_width=True,
        help=(
            "Click exactly three residues in Mol*. They define a plane. "
            "The chosen side is previewed in green until you add it."
        ),
    ):
        try:
            sides, plane_info = residues_on_plane_sides(
                pdb_text,
                set(viewer_selected),
                float(state["plane_tolerance_angstrom"]),
            )
            state["plane_residues"] = set(viewer_selected)
            state["plane_sides"] = sides
            state["plane_info"] = plane_info
            state["plane_side"] = "side_a"
            _set_plane_preview(state)
            _rerun_viewer()
        except ValueError as exc:
            st.warning(str(exc))

    if state.get("plane_sides"):
        plane_residues = set(state["plane_residues"])
        st.caption("Plane anchors: " + _format_residue_labels(plane_residues))

        side_options = ["side_a", "side_b"]
        selected_side = st.radio(
            "Preview side",
            options=side_options,
            index=side_options.index(state.get("plane_side", "side_a")),
            format_func=lambda value: "Side A" if value == "side_a" else "Side B",
            horizontal=True,
            help="The preview side is shown in green. It is not part of the crop until added.",
        )

        if selected_side != state.get("plane_side"):
            state["plane_side"] = selected_side
            _set_plane_preview(state)
            _rerun_viewer()

        _set_plane_preview(state)
        preview_residues = set(state["plane_preview_residues"])

        side_a_count = len(state["plane_sides"].get("side_a", set()))
        side_b_count = len(state["plane_sides"].get("side_b", set()))
        st.caption(
            f"Side A: {side_a_count} residues. "
            f"Side B: {side_b_count} residues. "
            f"Near plane: {state.get('plane_info', {}).get('on_plane_count', 0)} residues."
        )
        st.caption("Preview by chain: " + _format_chain_counts(preview_residues))

        if st.button(
            "Add previewed plane side",
            type="primary",
            disabled=not bool(preview_residues),
            use_container_width=True,
            help="Converts the green preview residues into the blue active crop selection.",
        ):
            _push_history(state)
            before_count = len(state["active_residues"])
            state["active_residues"] = set(state["active_residues"]) | preview_residues
            added_count = len(state["active_residues"]) - before_count
            _clear_plane_preview(state)
            if added_count:
                st.success(f"Added {added_count} residues from the plane side.")
            _rerun_viewer()

        if st.button(
            "Clear plane preview",
            use_container_width=True,
        ):
            _clear_plane_preview(state)
            _rerun_viewer()
    else:
        st.caption(
            "Click exactly three residues in Mol* to define a plane. "
            "The selected side will preview in green, not blue."
        )

    st.divider()

    col_undo, col_clear = st.columns(2)

    if col_undo.button(
        "Undo",
        disabled=not bool(state["history"]),
        use_container_width=True,
    ):
        state["active_residues"] = state["history"].pop()
        _rerun_viewer()

    if col_clear.button(
        "Clear active",
        disabled=not bool(state["active_residues"]),
        use_container_width=True,
    ):
        _push_history(state)
        state["active_residues"] = set()
        _rerun_viewer()

    if st.button(
        "Reset editor",
        use_container_width=True,
        help="Clears active selection, confirmed draft, undo history, and Mol* temporary state.",
    ):
        state["active_residues"] = set()
        state["confirmed_draft_residues"] = set()
        state["history"] = []
        _clear_plane_preview(state)
        _rerun_viewer()

    st.divider()

    st.subheader("Crop Candidate")

    st.write(f"Candidate residues: `{len(current_candidate)}`")

    if current_candidate:
        st.caption("Candidate by chain: " + _format_chain_counts(current_candidate))
    else:
        st.caption("No crop candidate yet.")

    if st.button(
        "Confirm candidate as draft crop",
        type="primary",
        disabled=not bool(current_candidate),
        use_container_width=True,
        help=(
            "Uses active residues plus the current Mol* clicked residues. "
            "This is the crop that will be previewed below."
        ),
    ):
        state["confirmed_draft_residues"] = set(current_candidate)
        _rerun_viewer()

    st.divider()

    state["debug_mapping"] = st.checkbox(
        "Show Mol* / PDB residue mapping debug",
        value=state.get("debug_mapping", False),
    )

    if state["debug_mapping"]:
        with st.expander("Debug Mol* residue mapping", expanded=True):
            st.write("Raw Mol* value:")
            st.json(viewer_value if viewer_value else {})

            st.write("Mapped Mol* residues used by Python:")
            st.write(sorted(viewer_selected))

            st.write("PDB sequence-index mapping preview:")
            preview = {
                chain: dict(list(chain_map.items())[:20])
                for chain, chain_map in residue_index_map.items()
            }
            st.json(preview)

            st.write("PDB residue-number preview:")
            st.json({chain: sorted(values)[:20] for chain, values in residue_number_set.items()})

            st.write("Sphere center residues:")
            st.write(sorted(viewer_selected))


# ---------------------------------------------------------------------
# Confirmed draft crop
# ---------------------------------------------------------------------

confirmed_draft = set(state["confirmed_draft_residues"])

if confirmed_draft:
    cropped_text, crop_stats = filter_pdb_to_residues(
        pdb_text,
        confirmed_draft,
        remove_waters=True,
        remove_hetero=False,
    )
else:
    cropped_text = ""
    crop_stats = {"residue_count": 0}


# ---------------------------------------------------------------------
# Draft summary and final job button
# ---------------------------------------------------------------------

with left:
    st.subheader("Confirmed Draft Crop")

    st.write(f"Confirmed crop residues: `{len(confirmed_draft)}`")

    if confirmed_draft:
        st.caption("Confirmed by chain: " + _format_chain_counts(confirmed_draft))

        with st.expander("Show residue selection that will be saved"):
            st.code(_compress_selection_for_text(confirmed_draft))

    if confirmed_draft and crop_stats.get("residue_count", 0) > 0:
        preview_summary = pdb_summary(cropped_text)

        crop_cols = st.columns(2)
        crop_cols[0].metric("Crop residues", crop_stats["residue_count"])
        crop_cols[1].metric("Crop atoms", preview_summary["atom_count"])

        crop_cols = st.columns(2)
        crop_cols[0].metric("Crop chains", len(preview_summary["chains"]))
        crop_cols[1].metric("Crop waters", preview_summary["water_count"])

    elif confirmed_draft:
        st.warning(
            "The confirmed draft selection does not match residues in the structure."
        )
    else:
        st.info("No confirmed draft yet.")

    can_create_job = bool(
        confirmed_draft
        and crop_stats.get("residue_count", 0) > 0
    )

    if st.button(
        "Create cropping job",
        type="primary",
        disabled=not can_create_job,
        use_container_width=True,
    ):
        final_selection_text = _compress_selection_for_text(confirmed_draft)

        with st.spinner("Writing cropped target artifacts..."):
            run_dir = run_target_crop(
                target_pdb,
                target_name=target["target_name"],
                manual_range_text=final_selection_text,
                viewer_selections=[],
                sphere_enabled=False,
                sphere_diameter_angstrom=0.0,
                remove_waters=True,
                remove_hetero=False,
            )

        st.success(f"Cropping job finished: {run_dir.name}")
        st.markdown(
            f"[Open result](/results?task_group=target-crop&run_id={run_dir.name})"
        )


# ---------------------------------------------------------------------
# Second viewer: exact crop preview
# ---------------------------------------------------------------------

st.subheader("Confirmed Crop Preview")

if not confirmed_draft:
    st.info(
        "Confirm the current candidate first. The cropped structure will appear here."
    )
elif crop_stats.get("residue_count", 0) == 0:
    st.warning("The confirmed crop does not contain matching residues.")
else:
    molstar_custom_component(
        structures=[
            StructureVisualization(
                pdb=cropped_text,
                color="chain-id",
                representation_type="cartoon+ball-and-stick",
            )
        ],
        key=(
            f"crop_confirmed_preview_"
            f"{target['run_id']}_"
            f"{len(confirmed_draft)}"
        ),
        height=560,
        show_controls=True,
        selection_mode=False,
        download_filename=f"{target['target_name']}_confirmed_crop_preview",
    )
