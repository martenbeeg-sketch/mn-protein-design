from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from mn_protein_design.app.pages.common import gpu_run_panel, show_pipeline_links
from mn_protein_design.workflows.capacity_benchmark import (
    ENGINE_LABELS,
    capacity_parent_rows,
    create_refolding_capacity_benchmark,
)
from mn_protein_design.workflows.design import prepared_design_targets, target_label
from mn_protein_design.workflows.detection import ppi_target_jobs
from mn_protein_design.workflows.refolding import _structure_chains
from mn_protein_design.workflows.target_msa import boltz_msa_paths, validate_a3m_file


DESIGN_GENERATOR_LABELS = {
    "bindcraft": "BindCraft",
    "rfdiffusion_classic": "RFdiffusion classic",
    "rfdiffusion3_foundry": "RFdiffusion3 / Foundry",
    "boltzgen": "BoltzGen",
    "pxdesign": "PXDesign",
    "genie3": "Genie3",
    "esmfold2_binder_design": "ESMFold2 binder design",
    "protpardelle_1c": "Protpardelle-1c",
    "proteina_complexa": "Proteina-Complexa",
}


def _capacity_target_label(row: dict) -> str:
    try:
        return target_label(row)
    except Exception:
        source = row.get("source_label") or row.get("source_category") or "target"
        chains = ",".join(str(chain) for chain in row.get("chains") or [] if str(chain))
        suffix = f", chains {chains}" if chains else ""
        return f"{row.get('target_name') or row.get('job_code') or 'target'} ({source}{suffix})"


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
                    "select": False,
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
                    "protenix_msa": msa_ready,
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
    return pd.DataFrame(rows).sort_values(["source_category", "aa_length", "target", "chain"], ascending=[True, True, True, True]).copy()


st.title("Capacity Benchmark")
st.caption("Measure practical folding/refolding size limits now, with a separate design-capacity test scaffold.")

setup_tab, engines_tab, review_tab, results_tab = st.tabs(
    ["Setup & Matrix", "Engines / Generators", "Review & Run", "Results"]
)

with setup_tab:
    capacity_kind = st.segmented_control(
        "Capacity test",
        ["folding", "design"],
        default="folding",
        format_func={
            "folding": "Folding / refolding engines",
            "design": "Design generators",
        }.get,
    )
    if capacity_kind == "folding":
        st.subheader("Folding Capacity")
        st.caption(
            "Folding capacity uses one real sequence from the target library, including prepared targets and "
            "benchmark targets, then increases the number of identical copies in a multimer. "
            "No new biological sequence is invented for this test."
        )
    else:
        st.subheader("Design Capacity")
        st.caption(
            "Design capacity uses one prepared target and increases the requested binder length across selected "
            "design generators. This setup is separated from the folding engine test."
        )
    benchmark_name = st.text_input("Benchmark name", value="Folding capacity check")
    if capacity_kind == "folding":
        matrix_mode = st.segmented_control(
            "Folding test type",
            ["target_panel", "sequence_copy_multimer", "synthetic_stress"],
            default="sequence_copy_multimer",
            format_func={
                "target_panel": "Target panel: selected targets",
                "sequence_copy_multimer": "Same-sequence multimer copy ladder",
                "synthetic_stress": "Synthetic stress mode",
            }.get,
        )
    else:
        matrix_mode = "fixed_target_binder_ladder"
        st.caption("Design capacity uses a fixed target and increasing requested binder lengths.")
    targets = ppi_target_jobs() if capacity_kind == "folding" else prepared_design_targets()
    target_chain_table = _prepared_target_chain_rows(targets)
    target = None
    target_pdb = None
    selected_chains = []
    if matrix_mode in {"sequence_copy_multimer", "fixed_target_binder_ladder"} and not targets:
        st.warning("Prepare, crop, or import a target first, or switch to synthetic stress mode.")
    elif matrix_mode in {"sequence_copy_multimer", "fixed_target_binder_ladder"}:
        labels = [_capacity_target_label(row) for row in targets]
        selected_label = st.selectbox("Target", labels, key="capacity_target")
        target = targets[labels.index(selected_label)]
        target_pdb = Path(str(target.get("target_pdb") or ""))
        try:
            available_chains = _structure_chains(target_pdb)
        except Exception:
            available_chains = list(target.get("chains") or [])
        selected_chains = st.multiselect(
            "Target chains",
            available_chains,
            default=list(target.get("chains") or available_chains),
            key=f"capacity_target_chains_{target.get('run_id', target_pdb.name)}",
        )
        st.caption(f"Target PDB: `{target_pdb}`")
    if capacity_kind == "folding" and not target_chain_table.empty:
        with st.expander("Available target chains", expanded=matrix_mode == "target_panel"):
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
                            "protenix_msa",
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
                "Imported PDB rows only show MSA-ready when their exact chain sequence already exists in the shared "
                "sequence-hashed A3M cache. Prepared/cropped rows commonly show yes because target preparation "
                "already generated or reused those MSAs."
            )
    preset = st.segmented_control(
        "Run depth",
        ["capacity_only", "practical", "full_workflow"],
        default="capacity_only",
        format_func={
            "capacity_only": "Capacity only",
            "practical": "Practical",
            "full_workflow": "Full workflow",
        }.get,
        help=(
            "Capacity only runs the smallest useful inference/generation settings to find OOM limits. "
            "Practical uses small real settings. Full workflow enables normal heavier validation/post-processing."
        ),
    )
    depth_notes = {
        "capacity_only": "Minimal inference/generation: model load plus one tiny prediction path, no Rosetta/PyMOL/common metrics.",
        "practical": "Full prediction settings with cached/prepared target MSAs where supported; no heavy common/Rosetta/PyMOL post-processing.",
        "full_workflow": "Full workflow: cached/prepared target MSAs, standard settings, common interface metrics, PyRosetta, and PyMOL where supported.",
    }
    st.caption(depth_notes.get(str(preset), ""))
    launch_now = st.checkbox("Launch child jobs immediately", value=True)
    stop_note = st.info(
        "Each engine-size cell is a separate queued job. One OOM should not prevent the other engines or sizes from running."
    )

with setup_tab:
    st.subheader("Capacity Test Matrix")
    default_binders = "50,100,150"
    default_targets = "128,256,384,512,621"
    default_copies = "1,2,3,4" if preset == "capacity_only" else "1,2,3,4,5,6"
    if matrix_mode == "sequence_copy_multimer":
        copy_counts_text = st.text_input("Multimer copy counts", value=default_copies)
        target_lengths_text = ""
        binder_lengths_text = ""
        st.caption(
            "The selected source chain sequence is copied into 2-mer, 3-mer, 4-mer, etc. systems. "
            "Use Practical/Full workflow with cached/prepared target MSAs to estimate realistic refolding limits; "
            "Capacity only is a no-MSA upper-bound stress test."
        )
    elif matrix_mode == "synthetic_stress":
        target_lengths_text = st.text_input("Synthetic target lengths", value=default_targets)
        binder_lengths_text = st.text_input("Synthetic binder lengths", value=default_binders)
        copy_counts_text = ""
    elif matrix_mode == "fixed_target_binder_ladder":
        target_lengths_text = ""
        binder_lengths_text = st.text_input("Binder lengths to design/test against the target", value=default_binders)
        copy_counts_text = ""
        st.caption("This is the design-capacity ladder: one fixed target with increasing requested binder length.")
    else:
        target_lengths_text = ""
        binder_lengths_text = ""
        copy_counts_text = ""

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

    target_lengths = _parse_lengths(target_lengths_text)
    binder_lengths = _parse_lengths(binder_lengths_text)
    copy_counts = _parse_lengths(copy_counts_text)
    target_panel_entries: list[dict[str, object]] = []
    fixed_target_len = None
    source_sequence_len = None
    if matrix_mode in {"sequence_copy_multimer", "fixed_target_binder_ladder"} and target_pdb is not None and target_pdb.exists():
        try:
            from mn_protein_design.core.structures import pdb_summary
            from mn_protein_design.workflows.refolding import _sequences_by_chain

            target_summary = pdb_summary(target_pdb.read_text(errors="ignore"))
            selected = set(selected_chains)
            fixed_target_len = sum(
                int(row.get("residue_count") or len(row.get("residues") or []) or 0)
                for row in target_summary.get("chains") or []
                if not selected or row.get("chain_id") in selected
            )
            if matrix_mode == "sequence_copy_multimer" and selected_chains:
                source_sequence_len = len(str(_sequences_by_chain(target_pdb).get(selected_chains[0]) or "").replace("X", ""))
        except Exception:
            fixed_target_len = None
            source_sequence_len = None
    if matrix_mode == "target_panel":
        st.caption(
            "Select target chains to test as independent systems. "
            "Set copy_count to 1 for monomer capacity, or >1 to make a repeated-chain multimer for that specific target."
        )
        if target_chain_table.empty:
            st.info("No target-chain rows are available yet.")
            matrix_rows = []
        else:
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
                        "protenix_msa",
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
                            for col in [
                                "target",
                                "chain",
                                "aa_length",
                                "copy_count",
                                "total_sequence_length",
                                "source",
                                "target_msa",
                                "target_pdb",
                                "row_key",
                            ]
                            if col in selected_panel.columns
                        ]
                    ],
                    hide_index=True,
                    width="stretch",
                    key="capacity_target_panel_copy_count_editor",
                    column_config={
                        "copy_count": st.column_config.NumberColumn("copy_count", min_value=1, max_value=26, step=1),
                    },
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
    elif matrix_mode == "sequence_copy_multimer":
        matrix_rows = [
            {
                "test": "folding",
                "source_chain": selected_chains[0] if selected_chains else "",
                "copy_count": copies,
                "sequence_length": source_sequence_len,
                "target_length": (source_sequence_len * (copies - 1)) if source_sequence_len else None,
                "binder_length": source_sequence_len,
                "total_length": (source_sequence_len * copies) if source_sequence_len else None,
            }
            for copies in copy_counts
            if copies >= 2
        ]
    elif matrix_mode == "fixed_target_binder_ladder":
        matrix_rows = [
            {
                "test": "design capacity",
                "target": (target.get("target_name") if target else "prepared target"),
                "target_length": fixed_target_len,
                "binder_length": binder,
                "total_length": (fixed_target_len + binder) if fixed_target_len else None,
            }
            for binder in binder_lengths
        ]
    else:
        matrix_rows = [
            {"test": "synthetic stress", "target": "synthetic", "target_length": target, "binder_length": binder, "total_length": target + binder}
            for target in target_lengths
            for binder in binder_lengths
        ]
    matrix_df = pd.DataFrame(matrix_rows)
    if matrix_mode == "sequence_copy_multimer" and not matrix_df.empty:
        matrix_df = matrix_df.rename(
            columns={
                "total_length": "total_sequence_length",
            }
        )
        matrix_df = matrix_df[
            [
                col
                for col in [
                    "test",
                    "source_chain",
                    "sequence_length",
                    "copy_count",
                    "total_sequence_length",
                ]
                if col in matrix_df.columns
            ]
        ]
    st.dataframe(matrix_df, hide_index=True, width="stretch")
    if matrix_rows:
        totals = [row["total_length"] for row in matrix_rows if row.get("total_length") is not None]
        if totals:
            st.caption(f"{len(matrix_rows)} systems | total residues {min(totals)}-{max(totals)}")
        else:
            st.caption(f"{len(matrix_rows)} systems")

with engines_tab:
    if capacity_kind == "folding":
        st.subheader("Folding / Refolding Engines")
        engine_keys = list(ENGINE_LABELS)
        if "capacity_selected_engines" not in st.session_state:
            st.session_state["capacity_selected_engines"] = [
                "alphafast_af3",
                "colabfold",
                "af2_initial_guess",
                "esmfold2",
                "boltz2_initial_guess",
                "rf3",
                "protenix",
                "boltzgen_fold",
            ]
        bulk_cols = st.columns([1, 1, 6])
        if bulk_cols[0].button("Select all", key="capacity_select_all"):
            st.session_state["capacity_selected_engines"] = engine_keys
            st.rerun()
        if bulk_cols[1].button("Deselect all", key="capacity_deselect_all"):
            st.session_state["capacity_selected_engines"] = []
            st.rerun()
        selected_engines = st.multiselect(
            "Refolding engines",
            engine_keys,
            default=st.session_state["capacity_selected_engines"],
            format_func=lambda key: ENGINE_LABELS.get(key, key),
            key="capacity_selected_engines",
        )
        selected_generators = []
        st.caption(
            "BoltzGen Fold is included but coordinate-conditioned. The other engines use the normal refolding contract."
        )

        with st.expander("Per-engine run-depth settings", expanded=False):
            st.write(
                pd.DataFrame(
                    [
                        {
                            "engine": ENGINE_LABELS[key],
                            "capacity only": capacity,
                            "practical": practical,
                            "full workflow": full,
                            "MSA use": msa_use,
                        }
                        for key, capacity, practical, full, msa_use in [
                            ("alphafast_af3", "1 recycle", "10 recycles", "10 recycles", "practical/full use cached target MSAs"),
                            ("colabfold", "1 model, 1 recycle", "3 models, 3 recycles", "3 models, 3 recycles", "practical/full use cached target MSAs"),
                            ("af2_initial_guess", "1 recycle", "3 recycles", "3 recycles", "target template; not shared-MSA driven"),
                            ("esmfold2", "1 loop, 16 steps", "10 loops, 68 steps", "10 loops, 68 steps", "practical/full use cached target MSAs"),
                            ("boltz2_initial_guess", "1 recycle, 20 steps, 1 sample", "10 recycles, 200 steps, 3 samples", "10 recycles, 200 steps, 3 samples", "practical/full use cached target MSAs"),
                            ("rf3", "2 recycles, 10 steps, 1 sample", "10 recycles, 50 steps, 5 samples", "10 recycles, 50 steps, 5 samples", "practical/full use cached target MSAs"),
                            ("protenix", "1 cycle, 10 steps, 1 sample", "3 cycles, 50 steps, 5 samples", "3 cycles, 50 steps, 5 samples", "practical/full use cached target MSAs"),
                            ("boltzgen_fold", "1 recycle, 20 steps, 1 sample", "3 recycles, 200 steps, 5 samples", "3 recycles, 200 steps, 5 samples", "coordinate/template conditioned; target MSA not central"),
                        ]
                    ]
                )
            )
            st.caption(
                "Practical and Full use the same prediction settings. Full additionally enables common interface "
                "metrics, PyRosetta, and PyMOL post-processing."
            )
    else:
        st.subheader("Design Generators")
        generator_keys = list(DESIGN_GENERATOR_LABELS)
        if "capacity_selected_generators" not in st.session_state:
            st.session_state["capacity_selected_generators"] = [
                "bindcraft",
                "rfdiffusion_classic",
                "rfdiffusion3_foundry",
                "boltzgen",
                "pxdesign",
                "genie3",
                "esmfold2_binder_design",
            ]
        bulk_cols = st.columns([1, 1, 6])
        if bulk_cols[0].button("Select all", key="capacity_select_all_generators"):
            st.session_state["capacity_selected_generators"] = generator_keys
            st.rerun()
        if bulk_cols[1].button("Deselect all", key="capacity_deselect_all_generators"):
            st.session_state["capacity_selected_generators"] = []
            st.rerun()
        selected_generators = st.multiselect(
            "Design generators",
            generator_keys,
            default=st.session_state["capacity_selected_generators"],
            format_func=lambda key: DESIGN_GENERATOR_LABELS.get(key, key),
            key="capacity_selected_generators",
        )
        selected_engines = []
        design_cols = st.columns(4)
        design_attempts = design_cols[0].number_input("Design attempts per length", min_value=1, max_value=1000, value=1, step=1)
        sequences_per_structure = design_cols[1].number_input("Sequences per structure", min_value=1, max_value=1000, value=1, step=1)
        survivor_limit = design_cols[2].number_input("Survivors per generator/length", min_value=1, max_value=1000, value=1, step=1)
        run_common_validation = design_cols[3].checkbox("Run common validation after generation", value=False)
        if preset == "capacity_only":
            st.caption("Capacity-only design means one minimal generation attempt per length/generator, no refolding or common validation.")
        elif preset == "practical":
            st.caption("Practical design uses a small real generator workload and can optionally keep a small survivor set.")
        else:
            st.caption("Full workflow design should run the normal generator path plus downstream validation once phase two execution is wired.")
        st.info("Design-capacity execution is the next phase. This tab now captures the generator setup and workload shape.")

with review_tab:
    st.subheader("Review & Run")
    gpu_device = gpu_run_panel(key="capacity_benchmark", default="0")
    selected_tool_count = len(selected_engines) if capacity_kind == "folding" else len(selected_generators)
    cells = len(matrix_rows) * selected_tool_count
    summary = pd.DataFrame(
        [
            {"setting": "Benchmark", "value": benchmark_name},
            {"setting": "Capacity test", "value": capacity_kind},
            {"setting": "Run depth", "value": preset},
            {"setting": "GPU", "value": gpu_device},
            {"setting": "Test type", "value": matrix_mode},
            {"setting": "Systems", "value": len(matrix_rows)},
            {"setting": "Engines/generators", "value": selected_tool_count},
            {"setting": "Child jobs / cells", "value": cells},
            {"setting": "Launch immediately", "value": "yes" if launch_now else "no"},
        ]
    )
    st.dataframe(summary, hide_index=True, width="stretch")
    if capacity_kind == "design":
        st.dataframe(
            pd.DataFrame(
                [
                    {"setting": "Selected generators", "value": ", ".join(DESIGN_GENERATOR_LABELS.get(key, key) for key in selected_generators)},
                    {"setting": "Design attempts per length", "value": design_attempts},
                    {"setting": "Sequences per structure", "value": sequences_per_structure},
                    {"setting": "Survivors per generator/length", "value": survivor_limit},
                    {"setting": "Common validation", "value": "yes" if run_common_validation else "no"},
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.warning("Design-capacity execution is not wired yet. This setup will be used for phase two.")
    run_disabled = not matrix_rows or not selected_engines or capacity_kind != "folding"
    if matrix_mode in {"sequence_copy_multimer", "fixed_target_binder_ladder"} and (target_pdb is None or not target_pdb.exists() or not selected_chains):
        run_disabled = True
        st.warning("Select a readable target and target chain before running.")
    if matrix_mode == "target_panel" and not target_panel_entries:
        run_disabled = True
        st.warning("Select at least one target-chain row in Setup & Matrix before running.")
    if st.button("Create capacity benchmark", type="primary", disabled=run_disabled):
        try:
            run_dir = create_refolding_capacity_benchmark(
                benchmark_name=str(benchmark_name or "Refolding capacity benchmark"),
                target_lengths=target_lengths,
                binder_lengths=binder_lengths,
                copy_counts=copy_counts,
                engines=list(selected_engines),
                gpu_device=str(gpu_device),
                preset=str(preset or "capacity_only"),
                launch=bool(launch_now),
                target_pdb=target_pdb if matrix_mode in {"sequence_copy_multimer", "fixed_target_binder_ladder"} else None,
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
    st.subheader("Results")
    parents = capacity_parent_rows()
    if not parents:
        st.info("No capacity benchmark runs yet.")
    else:
        st.markdown("**Benchmark Runs (Umbrella Jobs)**")
        st.caption(
            "Each row owns one target, run depth, test matrix, scheduler, and its child engine jobs. "
            "Select an umbrella to inspect or extend only that benchmark."
        )
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
        st.dataframe(
            parent_df[[col for col in display_cols if col in parent_df.columns]],
            hide_index=True,
            width="stretch",
            column_config={"result": st.column_config.LinkColumn("result", display_text="Open")},
        )
        st.caption("Use **Open** to view the matrix, plots, capacity profile, and additional run actions.")
        st.stop()
        selected_indices = list((event.selection or {}).get("rows") or [])
        selected_parent = parent_df.iloc[selected_indices[0]] if selected_indices else parent_df.iloc[0]
        selected_run_dir = Path(str(selected_parent["run_dir"]))
        children = capacity_child_rows(selected_run_dir)
        selected_payload = read_json(selected_run_dir / "input.json")
        selected_params = (
            selected_payload.get("params")
            if isinstance(selected_payload.get("params"), dict)
            else {}
        )
        selected_engine_keys = [
            str(engine) for engine in (selected_params.get("engines") or []) if str(engine)
        ]
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
                "Creates a new practical umbrella for the same target. Each engine starts at its own "
                "largest completed system from the selected capacity run, rather than using one shared size."
            )
            practical_seeds: list[dict[str, object]] = []
            if children:
                child_seed_df = pd.DataFrame(children)
                child_seed_df["total_length_numeric"] = pd.to_numeric(
                    child_seed_df["total_length"], errors="coerce"
                )
                for _engine_key, group in child_seed_df.groupby("engine_key", dropna=True):
                    valid = group.dropna(subset=["total_length_numeric"]).copy()
                    completed = valid[valid["status"].astype(str) == "completed"]
                    if completed.empty:
                        continue
                    best = completed.sort_values("total_length_numeric").iloc[-1]
                    failures = valid[
                        valid["status"].astype(str).isin({"failed", "skipped"})
                        & (valid["total_length_numeric"] > best["total_length_numeric"])
                    ].sort_values("total_length_numeric")
                    next_failure = failures.iloc[0] if not failures.empty else {}
                    practical_seeds.append(
                        {
                            "engine": best.get("engine"),
                            "copy_count": best.get("copy_count"),
                            "sequence_length": best.get("sequence_length"),
                            "total_sequence_length": int(best["total_length_numeric"]),
                            "source_run": best.get("run_id"),
                            "next_failed_total_sequence_length": (
                                int(next_failure["total_length_numeric"])
                                if isinstance(next_failure, pd.Series)
                                and pd.notna(next_failure.get("total_length_numeric"))
                                else ""
                            ),
                            "next_failed_status": next_failure.get("status", "") if isinstance(next_failure, pd.Series) else "",
                            "next_failure_kind": next_failure.get("failure_kind", "") if isinstance(next_failure, pd.Series) else "",
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
                practical_launch = st.checkbox(
                    "Launch practical cells now",
                    value=True,
                    key=f"capacity_practical_launch_{selected_run_dir.name}",
                )
                if st.button(
                    "Create practical benchmark umbrella",
                    type="primary",
                    key=f"capacity_create_practical_{selected_run_dir.name}",
                ):
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
            st.caption(
                "Adds cells to this selected matrix and keeps the same parent result. "
                "No new general benchmark job is created."
            )
            missing_engines = [
                engine for engine in ENGINE_LABELS if engine not in selected_engine_keys
            ]
            add_engines = st.multiselect(
                "Add folding engines",
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
                if str(row.get("status") or "")
                in {"completed", "failed", "stopped", "skipped", "cancelled"}
            ]
            recalc_labels = {
                str(row.get("run_id") or ""): (
                    f"{row.get('engine')} | {row.get('total_length')} residues | "
                    f"{row.get('status')} | {row.get('job_code')}"
                )
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
            launch_extension = st.checkbox(
                "Launch added cells now",
                value=True,
                key=f"capacity_launch_extension_{selected_run_dir.name}",
            )
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
                        f"Added {extension['added_systems']} systems, "
                        f"{extension['added_engines']} engines, "
                        f"{extension['added_cells']} new cells, and "
                        f"{extension['recalculated_cells']} recalculated cells."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))
        st.markdown("**Selected Capacity Matrix**")
        if not children:
            st.info("This capacity run has no child jobs yet.")
        else:
            child_df = pd.DataFrame(children)
            child_modes = set(child_df.get("matrix_mode", pd.Series(dtype=str)).dropna().astype(str))
            if child_modes and child_modes <= {"sequence_copy_multimer"}:
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
                child_display = child_display.rename(
                    columns={
                        "total_length": "total_sequence_length",
                    }
                )
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
            folding_only = set(plot_df.get("matrix_mode", pd.Series(dtype=str)).dropna().astype(str)) <= {
                "sequence_copy_multimer",
                "target_panel",
            }
            target_panel_only = set(plot_df.get("matrix_mode", pd.Series(dtype=str)).dropna().astype(str)) <= {"target_panel"}
            if folding_only:
                plot_df["total_sequence_length"] = plot_df["total_length"]
            plot_df["target_label"] = (
                plot_df["target_name"].fillna("").astype(str)
                if "target_name" in plot_df.columns
                else pd.Series([""] * len(plot_df), index=plot_df.index, dtype=str)
            )
            empty_target_label = plot_df["target_label"].str.strip() == ""
            if "candidate_id" in plot_df.columns:
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
                plot_df["x_label"] = plot_df["total_length"]
                x_encoding = alt.X("total_length:O", title="total residues")
            status_order = ["completed", "failed", "queued", "running", "preparing", "skipped", "cancelled", "paused"]
            status_colors = ["#0072CE", "#7CC1F2", "#FF2D2D", "#F4A3A8", "#F59E0B", "#9CA3AF", "#6B7280", "#A78BFA"]
            plot_df["status_rank"] = plot_df["status"].apply(lambda value: status_order.index(value) if value in status_order else len(status_order))
            if not plot_df.empty:
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
                        color=alt.Color(
                            "status:N",
                            title="status",
                            scale=alt.Scale(domain=status_order, range=status_colors),
                        ),
                        tooltip=tooltip,
                    )
                    .properties(width=720, height=720)
                )
                st.altair_chart(chart, width="content")
